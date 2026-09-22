"""Hybrid retrieval: dense ∪ lexical (BM25) fused with RRF, then an optional reranker.

``dense top-40 ∪ bm25 top-40 → RRF(k=60) → reranker on the top RERANK_POOL →
RRF(reranker rank, retrieval rank) → top_k → neighbour de-dup``.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import config, embed
from .chunk import Chunk
from .store import Store

RRF_K = 60

_CS_WORDS = {"co", "je", "jak", "jake", "jaké", "jaky", "jaký", "jsou", "se", "na", "pro", "podle", "proc", "proč", "kdo", "kdy",
             "kde", "ktery", "který", "ktere", "které", "stavba", "stavby", "staveb", "staveniste", "staveniště", "prace", "práce",
             "musi", "musí", "lze", "nebo", "mezi", "pri", "při", "do", "od", "za", "ze", "ve", "v", "a", "i", "u", "k", "s", "z", "o"}
_EN_WORDS = {"what", "how", "which", "who", "when", "where", "why", "is", "are", "the", "of", "and", "for", "to", "in", "on", "a", "an", "does", "do"}


def guess_lang(text: str) -> str:
    """cs when Czech diacritics or clearly Czech function words dominate; en otherwise."""
    if any(ch in "ěščřžýáíéůúďťňĚŠČŘŽÝÁÍÉŮÚĎŤŇ" for ch in text):
        return "cs"
    words = [w.strip("?.,!:;()\"'").lower() for w in text.split()]
    cs = sum(w in _CS_WORDS for w in words)
    en = sum(w in _EN_WORDS for w in words)
    return "cs" if cs > en else "en"


@dataclass(frozen=True)  # hashable: Engine keys its Retriever cache on it
class RetrievalConfig:
    top_k: int = 10  # chunks passed to the LLM
    dense_k: int = 40  # candidates from dense search
    bm25_k: int = 40  # candidates from lexical search
    rerank: bool | None = None  # None = auto (on when reranker weights are cached)
    lang: str | None = None  # "cs" | "en" | None (both)
    editions: tuple[str, ...] = ("full",)

    @classmethod
    def for_question(cls, question: str, *, top_k: int = 10, lang: str | None = None,
                     edition: str = "full", rerank: bool | None = None) -> RetrievalConfig:
        """Config for one question: language auto-detected unless given, `all` = both editions."""
        return cls(top_k=top_k, rerank=rerank, lang=lang or guess_lang(question),
                   editions=("full", "demo") if edition == "all" else (edition,))


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None
    bm25_rank: int | None
    rerank_score: float | None = None


def _rrf(*rankings: list[int]) -> dict[int, float]:
    """Reciprocal-rank fusion of orderings of ids; keys keep the first ranking's order."""
    fused: dict[int, float] = {}
    for ids in rankings:
        for rank, i in enumerate(ids):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
    return fused


class _EmbedderKey:
    """Hash wrapper so the query-embedding LRU keys on ``embedder.key`` (backend:model), not identity."""

    __slots__ = ("embedder", "key")

    def __init__(self, embedder: embed.Embedder):
        self.embedder = embedder
        self.key = getattr(embedder, "key", repr(embedder))

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _EmbedderKey) and other.key == self.key


@functools.lru_cache(maxsize=512)
def _query_embedding(k: _EmbedderKey, query: str) -> np.ndarray:
    return k.embedder.embed_query(query)


def embed_query_cached(embedder: embed.Embedder, query: str) -> np.ndarray:
    """``embedder.embed_query`` memoised per (embedder key, query): ``search`` then ``ask`` embeds once."""
    return _query_embedding(_EmbedderKey(embedder), query)


class Retriever:
    """Searches one ``Store`` with one ``RetrievalConfig``.

    ``progress`` receives one short status line per stage (for the CLI).
    """

    def __init__(self, store: Store, cfg: RetrievalConfig | None = None,
                 progress: Callable[[str], None] | None = None):
        self.store = store
        self.cfg = cfg or RetrievalConfig()
        self.say = progress or (lambda _msg: None)
        self.reranker = embed.get_reranker() if self.cfg.rerank in (None, True) else None
        if self.cfg.rerank is True and self.reranker is None:
            raise RuntimeError("rerank requested but no reranker model is available (see `stavby doctor`)")

    def search(self, query: str) -> list[Hit]:
        cfg, store = self.cfg, self.store
        mask = store.mask(cfg.lang, cfg.editions)

        t0 = time.time()
        emb = store.embedder()
        dense = store.dense_search(embed_query_cached(emb, query), cfg.dense_k, mask)
        lexical = store.bm25_search(query, cfg.bm25_k, mask)
        fused = _rrf([i for i, _ in dense], [i for i, _ in lexical])
        self.say(f"embed {emb.key} · dense {len(dense)} ∪ bm25 {len(lexical)} → {len(fused)} candidates · {time.time() - t0:.1f}s")

        dense_rank = {i: r for r, (i, _) in enumerate(dense)}
        bm25_rank = {i: r for r, (i, _) in enumerate(lexical)}
        pool = max(cfg.top_k, config.RERANK_POOL) if self.reranker else cfg.top_k
        hits = [Hit(store.chunks[i], s, dense_rank.get(i), bm25_rank.get(i))
                for i, s in sorted(fused.items(), key=lambda kv: -kv[1])[:pool]]

        if self.reranker and hits:
            hits = self._rerank(query, hits)
        return self._dedupe(hits[: cfg.top_k])

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Re-score the pool with the reranker and re-order it (see RERANK_FUSION)."""
        t0 = time.time()
        for h, s in zip(hits, self.reranker.score(query, [h.chunk.text for h in hits])):
            h.rerank_score = s
        if config.RERANK_FUSION == "rrf":
            # Fuse the reranker order with the retrieval order so a confident reranker "yes" on an
            # off-topic passage cannot leapfrog everything the retrievers agreed on.
            by_rerank = sorted(range(len(hits)), key=lambda i: -(hits[i].rerank_score or 0))
            fused = _rrf(list(range(len(hits))), by_rerank)
            hits = [hits[i] for i in sorted(fused, key=lambda i: -fused[i])]
        else:
            hits.sort(key=lambda h: -(h.rerank_score or 0))
        self.say(f"rerank {self.reranker.model} · {len(hits)} passages · fusion={config.RERANK_FUSION} · {time.time() - t0:.1f}s")
        return hits

    @staticmethod
    def _dedupe(hits: list[Hit]) -> list[Hit]:
        """Adjacent chunks of the same page overlap; keep the higher-ranked one when bodies overlap heavily."""
        kept: list[Hit] = []
        for h in hits:
            neighbour = any(
                k.chunk.page_url == h.chunk.page_url and abs(k.chunk.ordinal - h.chunk.ordinal) == 1
                and (k.chunk.body[-150:] in h.chunk.body or h.chunk.body[-150:] in k.chunk.body)
                for k in kept
            )
            if not neighbour:
                kept.append(h)
        return kept
