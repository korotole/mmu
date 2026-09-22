"""PDF books as a second source.

A PDF yields the same ``Page`` records the crawler yields - one per PDF page - so chunking,
indexing, retrieval and answering are unchanged.

Text comes from the PDF's text layer when it is trustworthy. Two common failure modes are
handled by OCR of the rendered page (Apple Vision through ``ocrmac``, macOS only):

* scanned PDFs with no text layer at all;
* legacy PDFs whose fonts lack Unicode maps, so accented letters arrive as ``\\x00`` or stray
  symbols (a 1990s Czech textbook typically loses every ř/č/ě/š/ž this way).

OCR output is put back into reading order (one or two columns / a two-page spread, paragraphs
by indentation and vertical gaps) and cached per file+page under ``data/ocr_cache``.

Structure (chapter, section, title) comes from the bookmark outline when the PDF has one,
otherwise from numbered headings found in the text ("3 Výrobní proces …", "3.1 Základní pojmy …"),
otherwise from the file name ("03_Vyrobni_proces.pdf" -> chapter 3, "Vyrobni proces").
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # `fitz` is the classic name of the PyMuPDF module

from . import config
from .crawl import Page
from .retrieve import guess_lang

LANG_SAMPLE_PAGES = 20  # pages of text used to auto-detect the language
OCR_DPI = 200
BROKEN_RATIO = 0.005  # share of undecodable glyphs above which the text layer is not trusted
_BROKEN_CHARS = set("\x00�⌃✏") | {chr(c) for c in range(1, 32) if c not in (9, 10, 13)}
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_HYPHEN_JOIN_RE = re.compile(r"(\w)- (?=[a-záčďéěíňóřšťúůýž])")  # "Není- li" -> "Není-li" after OCR line joins
_SECTION_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\b")
# "3.2 Struktury …" / "3 Výrobní proces …" as a whole short line = a heading.
_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\s+([A-ZÁ-Ž][^\n]{2,90})$")
_FILENAME_RE = re.compile(r"^(\d{1,2})?[_\-\s]*(.+?)$")
VISION_LANGS = {"cs": "cs-CZ", "en": "en-US"}

# A bookmark: (level, heading, first page). Levels are 1-based, pages are 1-based.
Bookmark = tuple[int, str, int]


def clean_pdf_text(raw: str) -> str:
    """De-hyphenate line-end hyphens, join hard-wrapped lines, collapse whitespace."""
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", raw.replace("\xa0", " "))
    paragraphs = []
    for p in re.split(r"\n\s*\n", text):  # a blank line stays a paragraph break
        p = re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", " ", p)).strip()
        if p:
            paragraphs.append(p)
    return "\n\n".join(paragraphs)


def broken_ratio(text: str) -> float:
    """Share of characters that cannot come from a healthy text layer."""
    return sum(1 for ch in text if ch in _BROKEN_CHARS) / len(text) if text else 0.0


# ------------------------------------------------------------------------- OCR


@dataclass(frozen=True)
class Line:
    """One recognised text line; box in page-normalised units, origin top-left."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def reading_order(lines: list[Line]) -> str:
    """Rebuild text from OCR lines: columns left to right, then top to bottom, with paragraph breaks.

    Two columns (or a scanned two-page spread) are detected when lines cluster on both sides
    of the page middle. Inside a column a paragraph starts at an indented first line or after
    a vertical gap clearly larger than the line pitch.
    """
    if not lines:
        return ""
    left = [ln for ln in lines if ln.xc < 0.5]
    right = [ln for ln in lines if ln.xc >= 0.5]
    two_columns = min(len(left), len(right)) >= 3 and all(ln.x1 - ln.x0 < 0.55 for ln in lines)
    columns = [left, right] if two_columns else [lines]

    paragraphs: list[str] = []
    for col in columns:
        col = sorted(col, key=lambda ln: ln.y0)
        if not col:
            continue
        pitch = statistics.median(ln.height for ln in col)
        margin = statistics.median(ln.x0 for ln in col)
        buf: list[str] = []
        prev: Line | None = None
        for ln in col:
            indented = ln.x0 > margin + 0.015
            gap = prev is not None and (ln.y0 - prev.y1) > 0.8 * pitch
            if buf and (indented or gap):
                paragraphs.append(" ".join(buf))
                buf = []
            buf.append(ln.text.strip())
            prev = ln
        if buf:
            paragraphs.append(" ".join(buf))
    return _HYPHEN_JOIN_RE.sub(r"\1-", "\n\n".join(paragraphs))


class VisionOcr:
    """Apple Vision OCR (via ``ocrmac``) with a per-file/page JSON cache."""

    def __init__(self, lang: str | None, cache_dir: Path | None = None, dpi: int = OCR_DPI):
        if sys.platform != "darwin":
            raise RuntimeError("OCR needs macOS (Apple Vision); this PDF has no usable text layer")
        from ocrmac import ocrmac  # noqa: F401  (import error surfaces here, once)

        self.langs = [VISION_LANGS[lang]] if lang in VISION_LANGS else list(VISION_LANGS.values())
        self.cache_dir = cache_dir or (config.DATA_DIR / "ocr_cache")
        self.dpi = dpi

    def page_text(self, doc: fitz.Document, index: int, file_key: str) -> str:
        cache = self.cache_dir / f"{file_key}-{self.dpi}.json"
        cached: dict[str, str] = json.loads(cache.read_text()) if cache.exists() else {}
        key = str(index)
        if key not in cached:
            cached[key] = reading_order(self._recognise(doc[index]))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(cached, ensure_ascii=False))
        return cached[key]

    def _recognise(self, page: fitz.Page) -> list[Line]:
        from ocrmac import ocrmac

        pix = page.get_pixmap(dpi=self.dpi)
        png = pix.tobytes("png")
        from io import BytesIO

        from PIL import Image

        image = Image.open(BytesIO(png))
        results = ocrmac.OCR(image, language_preference=self.langs, recognition_level="accurate").recognize()
        lines = []
        for text, _confidence, (x, y, w, h) in results:  # Vision boxes: origin bottom-left, normalised
            lines.append(Line(text=text, x0=x, y0=1 - y - h, x1=x + w, y1=1 - y))
        return lines


# ------------------------------------------------------------------------ book


def _last_before(entries: list[Bookmark], page: int) -> int | None:
    """1-based index of the last bookmark starting at or before ``page``, else None."""
    found = None
    for i, (_level, _heading, start) in enumerate(entries, 1):
        if start <= page:
            found = i
    return found


def headings_in(text: str) -> list[tuple[str, str]]:
    """(number, heading) for every numbered heading line, e.g. ("3.1", "3.1 Základní pojmy …")."""
    out = []
    for line in text.split("\n"):
        m = _HEADING_RE.match(line.strip())
        if m and not line.strip().endswith((".", ",", ";")):
            out.append((m.group(1), line.strip()))
    return out


class PdfBook:
    """One PDF file, read as ``Page`` records - the non-crawled half of the corpus.

    ``ocr``: "auto" (OCR pages whose text layer is missing or corrupt), "always", "never".
    """

    def __init__(self, path: str | Path, title: str | None = None, lang: str | None = None, ocr: str = "auto"):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"{self.path} not found")
        self.title = title
        self.lang = lang
        self.ocr = ocr
        m = _FILENAME_RE.match(self.path.stem)
        self.file_chapter = int(m.group(1)) if m and m.group(1) else None
        self.file_title = re.sub(r"[_\-]+", " ", m.group(2)).strip() if m else self.path.stem

    def pages(self, log: Callable[[str], None] = print) -> list[Page]:
        with fitz.open(str(self.path)) as doc:
            title = self.title or (doc.metadata or {}).get("title") or self.file_title
            outline: list[Bookmark] = [(level, heading.strip(), page) for level, heading, page in doc.get_toc() if page >= 1]
            texts = self._texts(doc, log)

        chapters = [b for b in outline if b[0] == 1]
        lang = self.lang or guess_lang("\n".join(texts[:LANG_SAMPLE_PAGES]))
        empty = sum(1 for t in texts if len(t) < config.MIN_CHUNK_CHARS)
        if empty:
            log(f"warning: {empty}/{len(texts)} pages of {self.path.name} yielded no text and were skipped")

        pages = []
        heading, section, chapter = None, None, self.file_chapter
        for n, text in enumerate(texts, 1):
            if len(text) < config.MIN_CHUNK_CHARS:
                continue
            if outline:  # bookmarks win
                i = _last_before(outline, n)
                heading = outline[i - 1][1] if i else None
                chapter = _last_before(chapters, n) if chapters else chapter
                m = _SECTION_RE.match(heading) if heading else None
                section = m.group(1) if m else None
            else:  # numbered headings in the text; the last one seen carries forward
                for number, line in headings_in(text):
                    heading = line
                    if "." in number:
                        section, chapter = number, int(number.split(".")[0])
                    else:
                        section, chapter = None, int(number)
            pages.append(Page(
                url=f"file://{self.path}#page={n}",
                path=self.path.name,
                edition_key="pdf",
                lang=lang,
                edition="full",
                chapter=chapter,
                section=section,
                title=heading or f"{title} – p. {n}",
                text=text,
                content_type="application/pdf",
                fetched_at=time.time(),
                book=title,
            ))
        return pages

    def _texts(self, doc: fitz.Document, log: Callable[[str], None]) -> list[str]:
        """Per-page text: the text layer where trustworthy, OCR otherwise (per ``self.ocr``)."""
        layer = [clean_pdf_text(page.get_text()) for page in doc]
        if self.ocr == "never":
            return layer
        needs = [self.ocr == "always" or not t or broken_ratio(t) > BROKEN_RATIO for t in layer]
        if not any(needs):
            return layer
        ocr = VisionOcr(self.lang or guess_lang("\n".join(layer[:LANG_SAMPLE_PAGES])))
        file_key = hashlib.sha1(self.path.read_bytes()).hexdigest()[:16]
        log(f"{self.path.name}: OCR {sum(needs)}/{len(layer)} pages "
            f"({'forced' if self.ocr == 'always' else 'text layer missing or corrupt'}) …")
        t0 = time.time()
        out = [ocr.page_text(doc, i, file_key) if need else text for i, (text, need) in enumerate(zip(layer, needs))]
        log(f"{self.path.name}: OCR done in {time.time() - t0:.0f}s")
        return out
