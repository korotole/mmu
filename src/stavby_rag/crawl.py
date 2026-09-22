"""Scoped crawler + text extractor for the CTU multimedia textbook.

The site is a late-1990s HTML 4 frameset site encoded in windows-1250. Each
chapter lives in ``kapN/`` (full edition) or the edition root (demo edition):

* ``frameN.html``  - frameset shell (no content, links the three frames)
* ``nadpisN.html`` - chapter header
* ``obsahN.html``  - chapter table of contents (links to sections)
* ``oN.html``      - chapter introduction
* ``textNM.html``  - section N.M body text (the actual content)
* ``literN.html``  - references / list of figures

We mirror every in-scope HTML page to ``data/raw`` and emit one JSON record
per *content* page to ``data/pages.jsonl``.
"""

from __future__ import annotations

import collections
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Comment

from . import config

LINK_RE = re.compile(r"""(?:href|src)\s*=\s*["']?([^"'\s>]+)""", re.IGNORECASE)
# Only these are fetched; everything else (images, Office theme files, media) is skipped.
FETCH_EXT = (".html", ".htm", ".txt", "/")
# Pages that are pure navigation chrome and carry no teaching content: framesets, headers, site map,
# chapter tables of contents (obsahN, obsahNsmall), figure/photo lists and thumbnail galleries.
CHROME_RE = re.compile(r"^(frame\d+|nadpis\d+\w*|index|mapa_stranek|default|obsah\d+\w*|obrazky\d*|nahledy\d*|foto\d+\w*)\.html?$", re.IGNORECASE)
# Titles that carry no information beyond the section number.
GENERIC_TITLE_RE = re.compile(r"^\s*(kapitola|chapter|část|part|podkapitola|subchapter|untitled|obsah|contents)?\s*[\d.\s-]*\s*$", re.IGNORECASE)
FIGURE_RE = re.compile(r"^\s*(obr[aá]zek|obr\.|figure|figere|fig\.|foto|photo|tabulka|table)\b", re.IGNORECASE)


@dataclass
class Page:
    url: str
    path: str  # relative path inside data/raw
    edition_key: str  # e.g. "online-priprava"
    lang: str  # cs | en
    edition: str  # full | demo
    chapter: int | None
    section: str | None  # "1.1", "13.2", ...
    title: str
    text: str
    headings: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    content_type: str = "text/html"
    fetched_at: float = 0.0
    book: str = ""  # source title for non-crawled sources (PDF books); "" = the CTU textbook


# --------------------------------------------------------------------------- crawl


class Crawler:
    """Breadth-first crawl restricted to the four textbook editions.

    Mirrors every fetched byte under ``data/raw`` and re-uses that mirror on the next
    run (``refresh=True`` re-downloads), so re-extraction costs no requests.
    """

    def __init__(self, limit: int = 20000, refresh: bool = False, log=print):
        self.limit, self.refresh, self.log = limit, refresh, log
        self.seen: set[str] = set()
        self.stats: collections.Counter = collections.Counter()

    def run(self, seeds: list[str] | None = None) -> list[Page]:
        queue = collections.deque(seeds or [config.SITE_ROOT + k + "/" for k in config.EDITIONS])
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        pages: list[Page] = []
        with httpx.Client(headers={"User-Agent": config.USER_AGENT}, timeout=30, follow_redirects=True) as client:
            while queue and len(self.seen) < self.limit:
                url = queue.popleft()
                if url in self.seen:
                    continue
                self.seen.add(url)
                if not url.lower().endswith(FETCH_EXT):
                    self.stats["skipped_non_text"] += 1
                    continue

                rel = urlparse(url).path[len("/aitom/podklady/"):]
                local = config.RAW_DIR / (rel + "index.html" if rel.endswith("/") else rel)
                fetched = self._read(client, url, local)
                if fetched is None:
                    continue
                body, ctype = fetched

                html = _decode(body, ctype)
                for m in LINK_RE.finditer(html):
                    link = _normalise(url, m.group(1))
                    if link and link not in self.seen:
                        queue.append(link)

                page = page_from_html(url, local, html)
                if page:
                    pages.append(page)
                if len(self.seen) % 100 == 0:
                    self.log(f"  … {len(self.seen)} urls, {len(pages)} content pages")

        self.log(f"crawl done: {dict(self.stats)}; content pages: {len(pages)}")
        return pages

    def _read(self, client: httpx.Client, url: str, local: Path) -> tuple[bytes, str] | None:
        """Bytes + content type from the local mirror, else from the network."""
        if local.exists() and not self.refresh:
            self.stats["cached"] += 1
            return local.read_bytes(), "text/html"
        r = self._get(client, url)
        if r is None:
            self.stats["error"] += 1
            return None
        time.sleep(config.CRAWL_DELAY_S)
        if r.status_code != 200:
            self.stats[f"http_{r.status_code}"] += 1
            return None
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(r.content)
        self.stats["fetched"] += 1
        return r.content, r.headers.get("content-type", "").split(";")[0].strip()

    def _get(self, client: httpx.Client, url: str, attempts: int = 4) -> httpx.Response | None:
        """The origin drops connections under even mild load; back off and retry."""
        delay = 1.0
        for i in range(attempts):
            try:
                return client.get(url)
            except httpx.HTTPError as e:
                if i == attempts - 1:
                    self.log(f"  ! {url}: {e} (gave up after {attempts} attempts)")
                    return None
                time.sleep(delay)
                delay *= 2
        return None


def _normalise(base: str, raw: str) -> str | None:
    """Absolute in-scope URL for a raw href/src, or None if out of scope."""
    raw = raw.strip().replace("\\", "/")
    if not raw or raw.startswith(("mailto:", "javascript:", "#", "data:")):
        return None
    url = urljoin(base, raw).split("#", 1)[0]
    if not config.SCOPE_RE.match(url):
        return None
    return url


def _decode(body: bytes, content_type: str) -> str:
    """Decode with the charset the page declares (windows-1250 unless stated otherwise)."""
    m = re.search(rb"charset=([\w-]+)", body[:2048], re.IGNORECASE) or re.search(rb"charset=([\w-]+)", content_type.encode())
    enc = m.group(1).decode() if m else config.DEFAULT_ENCODING
    try:
        return body.decode(enc, errors="replace")
    except LookupError:
        return body.decode(config.DEFAULT_ENCODING, errors="replace")


# --------------------------------------------------------------------------- extract


def _chapter_section(url: str, title: str, body_title: str) -> tuple[int | None, str | None]:
    name = Path(urlparse(url).path).name
    chapter = None
    m = re.search(r"/kap(\d+)/", url)
    if m:
        chapter = int(m.group(1))
    # text131.html -> 13.1 ; text11.html -> 1.1 ; text13uvod.html -> chapter 13 intro
    m = re.match(r"(?:text|o|obsah|liter)(\d+)", name, re.IGNORECASE)
    section = None
    for candidate in (body_title, title):
        s = re.search(r"\b(\d{1,2}\.\d{1,2}(?:\.\d+)?)\b", candidate or "")
        if s:
            section = s.group(1)
            break
    # The section number printed on the page wins over the folder: the EN edition keeps the CZ
    # folder layout (kap5/) but renumbers its sections (4.1 ...) because it has no chapter 2.
    if section:
        chapter = int(section.split(".")[0])
    if chapter is None and m:
        digits = m.group(1)
        if section:
            chapter = int(section.split(".")[0])
        elif len(digits) <= 2:
            chapter = int(digits) if len(digits) == 1 or int(digits) <= 13 else int(digits[0])
        else:
            chapter = int(digits[:2]) if int(digits[:2]) <= 13 else int(digits[0])
    if chapter is not None and section is None and m and re.match(r"text", name, re.IGNORECASE):
        digits = m.group(1)[len(str(chapter)):]
        if digits:
            section = f"{chapter}.{digits}"
    return chapter, section


def _best_title(body_title: str, title: str, headings: list[str], text: str) -> str:
    """Prefer an informative title: <body title>, then <title>, then first heading, then first text line."""
    for cand in (body_title, title):
        cand = (cand or "").strip()
        if cand and not GENERIC_TITLE_RE.match(cand):
            return cand
    for h in headings:
        if h and not GENERIC_TITLE_RE.match(h):
            return h[:120]
    for line in text.splitlines():
        line = line.strip(" |")
        if len(line) > 3 and not GENERIC_TITLE_RE.match(line):
            return line[:120]
    return (body_title or title or "").strip()


def clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    # Block-level elements become paragraph breaks, <br> becomes newline.
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "table", "blockquote"]):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    for tag in soup.find_all(["td", "th"]):
        tag.insert_after(" | ")
    text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(\s*\|\s*){2,}", " | ", text)
    return text.strip()


def page_from_html(url: str, local: Path, html: str) -> Page | None:
    """One Page record, or None when the document is navigation chrome or too short."""
    name = Path(urlparse(url).path).name or "index.html"
    if CHROME_RE.match(name):
        return None
    soup = BeautifulSoup(html, "lxml")
    if soup.find("frameset"):
        return None
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").replace("\xa0", " ")
    body_title = soup.body.get("title", "") if soup.body else ""
    headings = [h.get_text(" ", strip=True).replace("\xa0", " ") for h in soup.find_all(["h1", "h2", "h3", "h4"])]
    images = []
    for img in soup.find_all("img"):
        src = (img.get("src") or "").replace("\\", "/")
        if src and not re.search(r"sipka|logo|spacer|blank", src, re.IGNORECASE):
            images.append(urljoin(url, src))
    links = []
    for a in soup.find_all("a", href=True):
        link = _normalise(url, a["href"])
        if link:
            links.append(link)
    text = clean_text(soup)
    if len(text) < config.MIN_CHUNK_CHARS:
        return None
    key = config.SCOPE_RE.match(url).group(1)
    meta = config.EDITIONS[key]
    chapter, section = _chapter_section(url, title, body_title)
    best_title = re.sub(r"\s+", " ", _best_title(body_title, title, headings, text) or name).strip()
    if FIGURE_RE.match(best_title):
        section = None  # "Obrázek 3.2" is a figure caption page, not section 3.2
    return Page(
        url=url,
        path=str(local.relative_to(config.RAW_DIR)),
        edition_key=key,
        lang=meta["lang"],
        edition=meta["edition"],
        chapter=chapter,
        section=section,
        title=best_title,
        text=text,
        headings=headings,
        images=sorted(set(images)),
        links=sorted(set(links)),
        fetched_at=time.time(),
    )


# --------------------------------------------------------------------------- persistence


def save_pages(pages: list[Page], path: Path | None = None) -> None:
    path = path or config.corpus().pages_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def load_pages(path: Path | None = None) -> list[Page]:
    path = path or config.corpus().pages_file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `stavby crawl` (or `stavby ingest FILE.pdf`) first")
    with path.open(encoding="utf-8") as f:
        return [Page(**json.loads(line)) for line in f if line.strip()]
