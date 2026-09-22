"""On-disk index: chunk metadata (JSONL) + dense matrix (npy) + BM25 (bm25s).

Corpus is small (thousands of chunks), so brute-force cosine over a normalised
float32 matrix is faster than any ANN library and has zero extra dependencies.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import bm25s
import numpy as np

from . import config, embed
from .chunk import Chunk

CHUNKS_FILE = "chunks.jsonl"
DENSE_FILE = "dense.npy"
BM25_DIR = "bm25"
META_FILE = "meta.json"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STEM_LEN = 6  # crude prefix stemming tames Czech inflection for BM25


def tokenize(text: str) -> list[str]:
    """Lowercase, diacritics-folded, prefix-stemmed tokens (kept alongside full form)."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok.isdigit() and len(tok) > 4:
            continue
        folded = "".join(c for c in unicodedata.normalize("NFD", tok) if unicodedata.category(c) != "Mn")
        out.append(folded)
        if len(folded) > _STEM_LEN:
            out.append(folded[:_STEM_LEN])
    return out


class Store:
    def __init__(self, path: Path | None = None):
        self.path = path or config.corpus().index_dir
        self.chunks: list[Chunk] = []
        self.dense: np.ndarray | None = None
        self.bm25: bm25s.BM25 | None = None
        self.meta: dict = {}
        self._langs: np.ndarray | None = None  # per-chunk lang / edition, for vectorised masks
        self._editions: np.ndarray | None = None
        self._masks: dict[tuple[str | None, tuple[str, ...] | None], np.ndarray | None] = {}

    # ----------------------------------------------------------------- build
    def build(self, chunks: list[Chunk], embeddings: np.ndarray, embed_key: str, books: Sequence[str] = ()) -> None:
        """``embed_key`` is ``"<backend>:<model>"`` so queries can be embedded the same way.

        ``books`` are the titles of ingested PDF sources; the key is omitted for the crawled
        textbook so an index built before PDF support stays valid.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / CHUNKS_FILE).open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        dense = embeddings.astype(np.float32)
        dense /= np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9
        np.save(self.path / DENSE_FILE, dense)

        retriever = bm25s.BM25()
        retriever.index([tokenize(c.text) for c in chunks], show_progress=False)
        retriever.save(str(self.path / BM25_DIR))

        backend, _, model = embed_key.partition(":")
        self.meta = {"embed_backend": backend, "embed_model": model, "n_chunks": len(chunks), "dim": int(dense.shape[1])}
        if books:
            self.meta["books"] = sorted(set(books))
        (self.path / META_FILE).write_text(json.dumps(self.meta, indent=2))
        self.chunks, self.dense, self.bm25 = chunks, dense, retriever
        self._index_attrs()

    # ------------------------------------------------------------------ load
    def load(self) -> Store:
        if not (self.path / META_FILE).exists():
            raise FileNotFoundError(f"No index in {self.path} - run `stavby index` first")
        self.meta = json.loads((self.path / META_FILE).read_text())
        with (self.path / CHUNKS_FILE).open(encoding="utf-8") as f:
            self.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        self.dense = np.load(self.path / DENSE_FILE)
        self.bm25 = bm25s.BM25.load(str(self.path / BM25_DIR))
        self._index_attrs()
        return self

    def _index_attrs(self) -> None:
        """Per-chunk lang/edition arrays so ``mask()`` is a numpy compare, not a Python loop."""
        self._langs = np.array([c.lang for c in self.chunks], dtype=object)
        self._editions = np.array([c.edition for c in self.chunks], dtype=object)
        self._masks = {}

    @property
    def books(self) -> list[str]:
        """Titles of the sources in this index (empty for the crawled textbook)."""
        return self.meta.get("books", [])

    def embedder(self) -> embed.Embedder:
        """The embedder that built this index (backend + model recorded in meta.json)."""
        return embed.get_embedder(self.meta.get("embed_backend") or None, self.meta.get("embed_model") or None)

    # ---------------------------------------------------------------- search
    def dense_search(self, qvec: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        q = qvec.astype(np.float32)
        q /= np.linalg.norm(q) + 1e-9
        scores = self.dense @ q
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if np.isfinite(scores[i])]

    def bm25_search(self, query: str, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        n = len(self.chunks)
        docs, scores = self.bm25.retrieve([toks], k=min(n, max(k * 4, 50)), show_progress=False)
        out = []
        for i, s in zip(docs[0].tolist(), scores[0].tolist()):
            if s <= 0 or (mask is not None and not mask[i]):
                continue
            out.append((int(i), float(s)))
            if len(out) >= k:
                break
        return out

    def mask(self, lang: str | None, editions: tuple[str, ...] | None) -> np.ndarray | None:
        """Boolean keep-mask over chunks, or None when nothing is filtered out."""
        if lang is None and not editions:
            return None
        key = (lang, tuple(editions) if editions else None)
        m = self._masks.get(key)
        if m is None:
            if self._langs is None or len(self._langs) != len(self.chunks):
                self._index_attrs()
            m = np.ones(len(self.chunks), dtype=bool)
            if lang:
                m &= self._langs == lang
            if editions:
                m &= np.isin(self._editions, list(editions))
            self._masks[key] = m
        return m
