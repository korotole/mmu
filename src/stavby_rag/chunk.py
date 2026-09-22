"""Split extracted pages into retrieval chunks.

Strategy: paragraph-aware sliding window. Paragraphs are packed until the
chunk reaches ``CHUNK_CHARS``; the last ``CHUNK_OVERLAP_CHARS`` of the
previous chunk are carried over so sentences split across a boundary are
still retrievable. Every chunk is prefixed with a breadcrumb (edition,
chapter, section title) so the embedding carries its context.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from . import config
from .crawl import Page


@dataclass
class Chunk:
    id: str
    page_url: str
    lang: str
    edition: str
    chapter: int | None
    section: str | None
    title: str
    ordinal: int
    text: str  # what gets embedded/indexed (breadcrumb + body)
    body: str  # body only, shown to the user
    images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _split_paragraphs(text: str) -> list[str]:
    """Paragraphs, with over-long ones cut on sentence boundaries."""
    out: list[str] = []
    for p in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not p:
            continue
        if len(p) <= config.CHUNK_CHARS:
            out.append(p)
            continue
        buf = ""
        for s in re.split(r"(?<=[.!?])\s+(?=[A-ZÁ-Ž0-9])", p):
            if len(buf) + len(s) > config.CHUNK_CHARS and buf:
                out.append(buf.strip())
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            out.append(buf)
    return out


class Chunker:
    """Sliding window over one page: packs paragraphs, emits overlapping chunks."""

    def __init__(self, page: Page):
        self.page = page
        self.crumb = self._breadcrumb(page)
        self.chunks: list[Chunk] = []
        self.buf: list[str] = []
        self.size = 0
        self.carry = ""  # tail of the previous chunk, prepended to the next one

    def run(self) -> list[Chunk]:
        for p in _split_paragraphs(self.page.text):
            if self.size + len(p) > config.CHUNK_CHARS and self.buf and self.size > config.CHUNK_OVERLAP_CHARS:
                self._emit()
                self.buf = [self.carry] if self.carry else []
                self.size = len(self.carry)
            self.buf.append(p)
            self.size += len(p) + 2
        self._emit()
        return self.chunks

    @staticmethod
    def _breadcrumb(page: Page) -> str:
        parts = [page.book or ("Učebnice: Příprava a realizace staveb" if page.lang == "cs"
                               else "Textbook: Construction preparation and realisation")]
        if page.chapter is not None:
            parts.append(("Kapitola " if page.lang == "cs" else "Chapter ") + str(page.chapter))
        if page.title:
            parts.append(page.title)
        return " > ".join(parts)

    def _emit(self) -> None:
        """Turn the buffer into a chunk (unless too short or pure overlap) and set the next carry."""
        page, body = self.page, "\n\n".join(self.buf).strip()
        if len(body) < config.MIN_CHUNK_CHARS or body == self.carry:
            return
        ordinal = len(self.chunks)
        self.chunks.append(Chunk(
            id=hashlib.sha1(f"{page.url}#{ordinal}".encode()).hexdigest()[:16],
            page_url=page.url, lang=page.lang, edition=page.edition, chapter=page.chapter,
            section=page.section, title=page.title, ordinal=ordinal,
            text=f"{self.crumb}\n\n{body}", body=body, images=page.images,
        ))
        carry = body[-config.CHUNK_OVERLAP_CHARS:]
        # Start the overlap at a word boundary.
        self.carry = carry[carry.find(" ") + 1:] if " " in carry else carry


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    return [c for page in pages for c in Chunker(page).run()]
