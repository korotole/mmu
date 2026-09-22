"""One process-wide pipeline: index loaded once, models kept warm, results cached.

The CLI used to rebuild ``Store`` + ``Retriever`` + backend on every command; the web server
cannot afford that, and neither can a chat session. ``Engine`` owns them for one corpus:

* the index (chunks, dense matrix, BM25) is read from disk once and kept in memory;
* the embedder / reranker / LLM are resolved once and asked to stay resident in Ollama
  (``warm()``), so a query is embed -> rerank -> generate without model reloads;
* retrieval results and complete answers are memoised, so repeated questions (the common case
  on a shared web UI) cost nothing;
* everything that talks to Ollama is serialised with a lock - Ollama runs one model at a time
  on a laptop anyway, and concurrent requests would only evict each other.

``Engines`` is the per-corpus registry the server uses; the CLI uses one ``Engine`` directly.
"""

from __future__ import annotations

import collections
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from . import config, embed
from .llm import Answerer, Backend, pick_backend
from .retrieve import Hit, RetrievalConfig, Retriever
from .store import Store


@dataclass(frozen=True)
class Query:
    """Everything that makes two retrievals identical (the cache key)."""

    question: str
    top_k: int = 10
    lang: str | None = None  # None = auto-detect
    edition: str = "full"
    rerank: bool | None = None

    def config(self) -> RetrievalConfig:
        return RetrievalConfig.for_question(self.question, top_k=self.top_k, lang=self.lang,
                                            edition=self.edition, rerank=self.rerank)


@dataclass
class Answer:
    question: str
    answer: str
    hits: list[Hit]
    backend: str
    lang: str
    timings: dict[str, float] = field(default_factory=dict)
    cached: bool = False

    def sources(self) -> list[dict]:
        return [{"n": i + 1, "title": h.chunk.title, "chapter": h.chunk.chapter, "section": h.chunk.section,
                 "url": h.chunk.page_url, "lang": h.chunk.lang, "book": getattr(h.chunk, "book", None),
                 "rerank": None if h.rerank_score is None else round(h.rerank_score, 3),
                 "score": round(h.score, 4)} for i, h in enumerate(self.hits)]

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer, "backend": self.backend, "lang": self.lang,
                "sources": self.sources(), "timings": {k: round(v, 2) for k, v in self.timings.items()}, "cached": self.cached}


class _LRU:
    """Tiny thread-safe LRU dict (functools.lru_cache cannot key on our dataclasses + corpus)."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._d: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None

    def put(self, key, value) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        return len(self._d)


class Engine:
    """Retrieval + generation for one corpus, safe to share between threads."""

    def __init__(self, corpus: str | None = None, *, backend: str | None = None, model: str | None = None,
                 progress: Callable[[str], None] | None = None, cache_size: int = 256):
        self.corpus = config.Corpus(corpus or "")
        self.backend_name, self.model_name = backend, model
        self.say = progress or (lambda _msg: None)
        self._store: Store | None = None
        self._backend: Backend | None = None
        self._retrievers: dict[RetrievalConfig, Retriever] = {}
        self._lock = threading.RLock()  # guards lazy init + all Ollama traffic
        self.hits_cache: _LRU = _LRU(cache_size)
        self.answer_cache: _LRU = _LRU(cache_size)

    # ------------------------------------------------------------ lazy parts
    @property
    def store(self) -> Store:
        with self._lock:
            if self._store is None:
                t0 = time.time()
                self._store = Store(self.corpus.index_dir).load()
                self.say(f"index {self.corpus.label}: {len(self._store.chunks)} chunks loaded in {time.time() - t0:.2f}s")
            return self._store

    @property
    def backend(self) -> Backend:
        with self._lock:
            if self._backend is None:
                self._backend = pick_backend(self.backend_name, self.model_name)
            return self._backend

    @property
    def label(self) -> str:
        return f"{self.backend.name}:{self.backend.model}"

    def retriever(self, cfg: RetrievalConfig) -> Retriever:
        """One Retriever per distinct config (they are cheap, but the reranker lookup is not)."""
        with self._lock:
            r = self._retrievers.get(cfg)
            if r is None:
                r = self._retrievers[cfg] = Retriever(self.store, cfg, progress=self.say)
            return r

    # ---------------------------------------------------------------- warm-up
    def warm(self, *, llm: bool | None = None) -> dict[str, float]:
        """Load the index and push every model into Ollama's memory so the first query is fast.

        ``llm=None`` warms the LLM only when it is local (Ollama); Claude needs no warm-up.
        Returns seconds spent per stage.
        """
        timings: dict[str, float] = {}
        with self._lock:
            t0 = time.time()
            store = self.store
            timings["index"] = time.time() - t0

            t0 = time.time()
            store.embedder().embed_query("warm-up")
            timings["embedder"] = time.time() - t0

            reranker = embed.get_reranker()
            if reranker is not None:
                t0 = time.time()
                reranker.score("warm-up", ["warm-up"])
                timings["reranker"] = time.time() - t0

            be = self.backend
            if llm or (llm is None and be.name == "ollama"):
                warm = getattr(be, "warm", None)
                if warm is not None:
                    t0 = time.time()
                    warm()
                    timings["llm"] = time.time() - t0
        self.say("warm: " + " · ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
        return timings

    # ---------------------------------------------------------------- search
    def search(self, q: Query) -> list[Hit]:
        """Hybrid retrieval (+ rerank) for one question; memoised per corpus + Query."""
        key = (self.corpus.name, q)
        hits = self.hits_cache.get(key)
        if hits is not None:
            return hits
        cfg = q.config()
        with self._lock:  # embedder + reranker live in Ollama: one at a time
            hits = self.retriever(cfg).search(q.question)
        self.hits_cache.put(key, hits)
        return hits

    # ----------------------------------------------------------------- answer
    def stream(self, q: Query, history: Sequence[dict] = ()) -> tuple[list[Hit], Iterator[str]]:
        """Retrieve first (so callers can show sources immediately), then stream the answer."""
        hits = self.search(q)
        answerer = Answerer(self.retriever(q.config()), self.backend)
        return hits, self._generate(answerer, q.question, hits, list(history))

    def _generate(self, answerer: Answerer, question: str, hits: list[Hit], history: list[dict]) -> Iterator[str]:
        if self.backend.name == "ollama":
            with self._lock:  # a local LLM shares the GPU with the retrieval models
                yield from answerer.stream(question, hits, history)
        else:
            yield from answerer.stream(question, hits, history)

    def answer(self, q: Query, history: Sequence[dict] = (), *, use_cache: bool = True) -> Answer:
        """Complete answer with timings. Only history-free questions are cached (history changes the answer)."""
        key = (self.corpus.name, q, self.label)
        if use_cache and not history:
            cached = self.answer_cache.get(key)
            if cached is not None:
                return Answer(**{**cached.__dict__, "cached": True})
        t0 = time.time()
        hits = self.search(q)
        t1 = time.time()
        answerer = Answerer(self.retriever(q.config()), self.backend)
        text = "".join(self._generate(answerer, q.question, hits, list(history)))
        t2 = time.time()
        ans = Answer(question=q.question, answer=text, hits=hits, backend=self.label, lang=q.config().lang or "",
                     timings={"retrieve": t1 - t0, "generate": t2 - t1, "total": t2 - t0})
        if not history:
            self.answer_cache.put(key, ans)
        return ans

    # ------------------------------------------------------------------ misc
    def info(self) -> dict:
        """What this engine serves - for `doctor`, `/api/health` and the UI header."""
        meta = self.store.meta
        return {"corpus": self.corpus.label, "chunks": meta.get("n_chunks"), "dim": meta.get("dim"),
                "embed": f"{meta.get('embed_backend')}:{meta.get('embed_model')}", "books": self.store.books,
                "reranker": getattr(embed.get_reranker(), "model", None), "llm": self.label,
                "cached_hits": len(self.hits_cache), "cached_answers": len(self.answer_cache)}

    def clear_cache(self) -> None:
        self.hits_cache.clear()
        self.answer_cache.clear()


class Engines:
    """Registry of one Engine per corpus name ("" = the default crawled textbook)."""

    def __init__(self, **engine_kwargs):
        self.kw = engine_kwargs
        self._engines: dict[str, Engine] = {}
        self._lock = threading.Lock()

    def get(self, corpus: str | None = None) -> Engine:
        name = corpus or ""
        with self._lock:
            e = self._engines.get(name)
            if e is None:
                if not config.Corpus(name).index_dir.joinpath("meta.json").exists():
                    raise FileNotFoundError(f"no index for corpus {name or 'default'!r} - run `stavby index`")
                e = self._engines[name] = Engine(name, **self.kw)
            return e

    @staticmethod
    def available() -> list[str]:
        """Corpus names with a built index; the default corpus is reported as ``""``."""
        names = [""] if (config.DATA_DIR / "index" / "meta.json").exists() else []
        root = config.DATA_DIR / "corpora"
        if root.is_dir():
            names += sorted(p.name for p in root.iterdir() if (p / "index" / "meta.json").exists())
        return names
