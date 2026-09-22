"""Embedding + reranking.

Embedding backends (same interface, vectors differ per model):

* ``ollama`` - Ollama ``/api/embed``. Default model is the official Qwen3-Embedding-8B GGUF at Q8_0
               (``hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q8_0``: 4096-d, 100+ languages, last-token
               pooling declared in the GGUF header, top of the open multilingual MTEB tables).
* ``st``     - sentence-transformers with HuggingFace weights, runs on MPS/CUDA/CPU.

Reranker backends:

* ``ollama`` - Qwen3-Reranker served as a causal LM through Ollama ``/api/generate``. The model
               answers "yes"/"no" to "does this document answer the query?"; the relevance score
               is P(yes) computed from the returned token logprobs (this is the official scoring
               recipe from the Qwen3-Reranker model card, just executed via Ollama).
* ``st``     - a sentence-transformers CrossEncoder (e.g. BAAI/bge-reranker-v2-m3) from HF.

Selection is ``auto`` by default: whichever backend already has its model available locally.
The index remembers which embedder built it so queries are embedded identically (``store.py``).
"""

from __future__ import annotations

import abc
import functools
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import httpx
import numpy as np

from . import config

QWEN_EMBED_INSTRUCTION = "Given a question about construction technology and construction management, retrieve textbook passages that answer the question"
QWEN_RERANK_INSTRUCTION = "Given a question about construction technology and construction management, judge whether the textbook passage answers the question"


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ----------------------------------------------------------------- availability


def hf_cached(repo_id: str) -> bool:
    """True when the repo has weight files in the local HF cache (not just config/tokenizer)."""
    try:
        from huggingface_hub import scan_cache_dir

        repos = scan_cache_dir().repos
    except Exception:  # noqa: BLE001 - no huggingface_hub, no cache dir, unreadable cache: all "not cached"
        return False
    return any(repo.repo_id == repo_id
               and any(f.file_name.endswith((".safetensors", ".bin")) for rev in repo.revisions for f in rev.files)
               for repo in repos)


def ollama_has_model(model: str, url: str = config.OLLAMA_URL) -> bool:
    try:
        r = httpx.get(url.rstrip("/") + "/api/tags", timeout=2)
        names = {m["name"] for m in r.json().get("models", [])}
        return model in names or f"{model}:latest" in names
    except (httpx.HTTPError, ValueError, KeyError):  # unreachable Ollama or unexpected payload
        return False


# -------------------------------------------------------------------- embedding


class Embedder(abc.ABC):
    """Turns texts into L2-normalised vectors, applying the model's prompt conventions."""

    backend: str

    def __init__(self, model: str):
        self.model = model

    @property
    def key(self) -> str:
        """``"<backend>:<model>"`` - recorded in the index so queries match the corpus."""
        return f"{self.backend}:{self.model}"

    def embed_texts(self, texts: list[str], batch_size: int = 16, show_progress: bool = True) -> np.ndarray:
        return self._encode(self._wrap(texts, query=False), batch_size, show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode(self._wrap([query], query=True), 1, False)[0]

    def _wrap(self, texts: list[str], *, query: bool) -> list[str]:
        """Model-specific prompt conventions: E5 query:/passage:, Qwen3 instruction on queries only."""
        name = self.model.lower()
        if "e5" in name:
            return [("query: " if query else "passage: ") + t for t in texts]
        if "qwen3-embedding" in name and query:
            return [f"Instruct: {QWEN_EMBED_INSTRUCTION}\nQuery: {t}" for t in texts]
        return texts

    @abc.abstractmethod
    def _encode(self, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray: ...


class OllamaEmbedder(Embedder):
    backend = "ollama"

    def __init__(self, model: str = config.OLLAMA_EMBED_MODEL, url: str = config.OLLAMA_URL,
                 num_ctx: int = config.OLLAMA_EMBED_NUM_CTX, keep_alive: str | int = config.OLLAMA_KEEP_ALIVE):
        super().__init__(model)
        self.url = url.rstrip("/")
        self.num_ctx, self.keep_alive = num_ctx, keep_alive
        self._client = httpx.Client(timeout=900)  # one keep-alive connection per embedder

    def _encode(self, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            payload = {"model": self.model, "input": texts[i:i + batch_size], "truncate": True,
                       "keep_alive": self.keep_alive, "options": {"num_ctx": self.num_ctx}}
            r = self._client.post(f"{self.url}/api/embed", json=payload)
            if r.status_code == 404:
                raise RuntimeError(f"Ollama has no model {self.model!r}; run `ollama pull {self.model}`")
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        v = np.asarray(out, dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


class StEmbedder(Embedder):
    """sentence-transformers / HuggingFace weights; the model is loaded on first use."""

    backend = "st"

    def __init__(self, model: str = config.EMBED_MODEL):
        super().__init__(model)
        self._st = None

    def _encode(self, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(self.model, device=_device())
        return self._st.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                               show_progress_bar=show_progress, convert_to_numpy=True)


def resolve_embed_backend(preferred: str | None = None) -> tuple[str, str]:
    """Return (backend, model_name) for embeddings."""
    pref = preferred or config.EMBED_BACKEND
    if pref == "st":
        return "st", config.EMBED_MODEL
    if pref == "ollama":
        return "ollama", config.OLLAMA_EMBED_MODEL
    if ollama_has_model(config.OLLAMA_EMBED_MODEL):
        return "ollama", config.OLLAMA_EMBED_MODEL
    if hf_cached(config.EMBED_MODEL):
        return "st", config.EMBED_MODEL
    return "ollama", config.OLLAMA_EMBED_MODEL  # will fail loudly with a pull hint if absent


@functools.lru_cache(maxsize=4)
def get_embedder(backend: str | None = None, model: str | None = None) -> Embedder:
    """The embedder for an explicit backend/model, or whichever one is available."""
    if not (backend and model):
        backend, resolved = resolve_embed_backend(backend)
        model = model or resolved
    return OllamaEmbedder(model) if backend == "ollama" else StEmbedder(model)


# -------------------------------------------------------------------- reranking


class Reranker(Protocol):
    model: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class OllamaQwenReranker:
    """Qwen3-Reranker through Ollama: score = P("yes") from the first generated token's logprobs."""

    _SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
               'Note that the answer can only be "yes" or "no".')

    def __init__(self, model: str = config.OLLAMA_RERANK_MODEL, url: str = config.OLLAMA_URL,
                 workers: int = config.RERANK_WORKERS, num_ctx: int = config.OLLAMA_RERANK_NUM_CTX,
                 keep_alive: str | int = config.OLLAMA_KEEP_ALIVE, num_batch: int = config.OLLAMA_RERANK_NUM_BATCH):
        self.model, self.url, self.workers = model, url.rstrip("/"), max(1, workers)
        self.num_ctx, self.keep_alive, self.num_batch = num_ctx, keep_alive, num_batch
        # Persistent connection pool: one TCP connection per worker, reused across queries.
        self._client = httpx.Client(timeout=600, limits=httpx.Limits(max_connections=self.workers + 1,
                                                                     max_keepalive_connections=self.workers + 1))

    @staticmethod
    def score_from_top_logprobs(top_logprobs: list[dict]) -> float:
        """P(yes) = sigmoid(lse(yes variants) - lse(no variants)).

        The model spreads mass over case variants ("yes"/"Yes"/"YES"); aggregating them with
        log-sum-exp keeps the full signal. Missing variants count as logprob -30.
        """
        yes: list[float] = []
        no: list[float] = []
        for t in top_logprobs:
            tok = t["token"].strip().lower()
            if tok == "yes":
                yes.append(t["logprob"])
            elif tok == "no":
                no.append(t["logprob"])

        def lse(xs: list[float]) -> float:
            if not xs:
                return -30.0
            m = max(xs)
            return m + math.log(sum(math.exp(x - m) for x in xs))

        d = lse(yes) - lse(no)
        return 1.0 / (1.0 + math.exp(-d)) if d > -700 else 0.0

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if self.workers == 1 or len(passages) == 1:
            return [self._score_one(query, p) for p in passages]
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda p: self._score_one(query, p), passages))

    def _score_one(self, query: str, doc: str) -> float:
        prompt = (f"<|im_start|>system\n{self._SYSTEM}<|im_end|>\n"
                  f"<|im_start|>user\n<Instruct>: {QWEN_RERANK_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}<|im_end|>\n"
                  f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
        options = {"num_predict": 1, "temperature": 0, "num_ctx": self.num_ctx}
        if self.num_batch:
            options["num_batch"] = self.num_batch
        payload = {"model": self.model, "prompt": prompt, "raw": True, "stream": False,
                   "logprobs": True, "top_logprobs": 20, "keep_alive": self.keep_alive, "options": options}
        r = self._client.post(f"{self.url}/api/generate", json=payload)
        if r.status_code == 404:
            raise RuntimeError(f"Ollama has no model {self.model!r}; run `ollama pull {self.model}`")
        r.raise_for_status()
        lp = r.json().get("logprobs") or []
        if not lp:
            raise RuntimeError("Ollama did not return logprobs; upgrade Ollama (needs /api/generate logprobs support)")
        return self.score_from_top_logprobs(lp[0].get("top_logprobs", []))


class CrossEncoderReranker:
    """HuggingFace CrossEncoder (e.g. BAAI/bge-reranker-v2-m3) via sentence-transformers."""

    def __init__(self, model: str = config.RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        self.model = model
        self._ce = CrossEncoder(model, device=_device(), max_length=1024)

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self._ce.predict([(query, p) for p in passages])] if passages else []


def resolve_rerank_backend(preferred: str | None = None) -> str | None:
    """'ollama' | 'st' | None (no reranker available)."""
    pref = preferred or config.RERANK_BACKEND
    if pref in ("ollama", "st", "none"):
        return None if pref == "none" else pref
    if ollama_has_model(config.OLLAMA_RERANK_MODEL):
        return "ollama"
    if hf_cached(config.RERANK_MODEL):
        return "st"
    return None


@functools.lru_cache(maxsize=2)
def get_reranker(backend: str | None = None) -> Reranker | None:
    b = resolve_rerank_backend(backend)
    if b == "ollama":
        return OllamaQwenReranker()
    if b == "st":
        return CrossEncoderReranker()
    return None
