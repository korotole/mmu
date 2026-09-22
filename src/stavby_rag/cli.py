"""`stavby` command line interface."""

from __future__ import annotations

import collections
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import config, embed
from .chunk import chunk_pages
from .crawl import Crawler, load_pages, save_pages
from .engine import Engine, Query
from .llm import OllamaBackend
from .pdf import PdfBook
from .retrieve import Hit, RetrievalConfig, Retriever
from .store import Store

app = typer.Typer(help="Local RAG over the CTU multimedia textbook 'Příprava a realizace staveb'.", no_args_is_help=True)
console = Console()

LangOpt = Annotated[str | None, typer.Option("--lang", "-l", help="Restrict to 'cs' or 'en' (default: auto-detect from question).")]
EditionOpt = Annotated[str, typer.Option("--edition", help="full | demo | all")]
CorpusOpt = Annotated[str | None, typer.Option("--corpus", help="Corpus name (default: $STAVBY_CORPUS, else the crawled data/ corpus).")]


def _print_version(value: bool) -> None:
    if value:
        from . import __version__

        print(f"stavby {__version__}")
        raise typer.Exit()


@app.callback()
def select_corpus(
    corpus: CorpusOpt = None,
    version: Annotated[bool, typer.Option("--version", "-V", callback=_print_version, is_eager=True,
                                          help="Print the version and exit.")] = False,
):
    config.use_corpus(corpus)  # `stavby --corpus X cmd`; every command also takes --corpus


class Ui:
    """Console presentation: dim status lines per stage, streamed answers, tables."""

    def __init__(self, console: Console = console):
        self.console = console

    def meta(self, msg: str) -> None:
        self.console.print(f"[dim]{msg}[/]", highlight=False)

    def answer(self, engine: Engine, q: Query, history: list[dict], *, show_sources: bool) -> str:
        """Retrieve, stream the answer to the console, and return it."""
        cfg = q.config()
        self.meta(f"lang={cfg.lang} · editions={'/'.join(cfg.editions)} · index {engine.store.meta.get('n_chunks')} chunks")
        with self.console.status("retrieving…", spinner="dots"):
            hits, stream = engine.stream(q, history)
        self.meta(f"generate {engine.label} · {len(hits)} passages")

        buf: list[str] = []
        t0 = time.time()
        with self.console.status("thinking…", spinner="dots"):
            first = next(stream, None)
        self.meta(f"first token after {time.time() - t0:.1f}s")
        if first is not None:
            buf.append(first)
            self.console.print(first, end="")
            for tok in stream:
                buf.append(tok)
                self.console.print(tok, end="", highlight=False, markup=False)
        self.console.print()
        if show_sources:
            self.console.print()
            for n, h in enumerate(hits, 1):
                c = h.chunk
                self.console.print(f"[dim][{n}][/] {c.title} [dim](ch. {c.chapter}, {c.lang})[/] {c.page_url}")
        return "".join(buf)

    def hits(self, query: str, hits: list[Hit], show_text: bool) -> None:
        table = Table(title=f"top {len(hits)} for: {query}", show_lines=show_text)
        table.add_column("#"); table.add_column("ch/sec"); table.add_column("title"); table.add_column("rrf", justify="right")
        table.add_column("dense", justify="right"); table.add_column("bm25", justify="right"); table.add_column("rerank P(yes)", justify="right")
        table.add_column("url", overflow="fold")
        for n, h in enumerate(hits, 1):
            c = h.chunk
            table.add_row(str(n), f"{c.chapter}/{c.section or '-'}", c.title[:50], f"{h.score:.3f}",
                          "-" if h.dense_rank is None else str(h.dense_rank + 1),
                          "-" if h.bm25_rank is None else str(h.bm25_rank + 1),
                          "-" if h.rerank_score is None else f"{h.rerank_score:.2f}", c.page_url)
        self.console.print(table)
        if show_text:
            for n, h in enumerate(hits, 1):
                self.console.print(Panel(h.chunk.body, title=f"[{n}] {h.chunk.title}", subtitle=h.chunk.page_url))


ui = Ui()


# ---------------------------------------------------------------------- crawl


@app.command()
def crawl(
    limit: int = typer.Option(20000, help="Max URLs to visit."),
    refresh: bool = typer.Option(False, help="Re-download even if a local copy exists."),
    corpus: CorpusOpt = None,
):
    """Mirror the textbook (all four editions) into data/raw and extract page text."""
    config.use_corpus(corpus)
    t0 = time.time()
    pages = Crawler(limit=limit, refresh=refresh, log=console.print).run()
    save_pages(pages)
    by = collections.Counter((p.lang, p.edition) for p in pages)
    console.print(f"[green]saved {len(pages)} pages[/] to {config.corpus().pages_file} in {time.time() - t0:.0f}s: {dict(by)}")


# --------------------------------------------------------------------- ingest


@app.command()
def ingest(
    files: Annotated[list[Path], typer.Argument(help="PDF book(s) to read.")],
    title: str | None = typer.Option(None, "--title", "-t", help="Book title (default: PDF metadata title, else the file name)."),
    lang: str | None = typer.Option(None, "--lang", "-l", help="cs | en (default: auto-detect from the text)."),
    ocr: str = typer.Option("auto", "--ocr", help="auto | always | never - OCR pages whose text layer is missing or corrupt (macOS, Apple Vision)."),
    corpus: CorpusOpt = None,
):
    """Read PDF book(s) into a corpus's pages.jsonl (then run `stavby index --corpus ...`)."""
    config.use_corpus(corpus)
    pages = []
    for f in files:
        book = PdfBook(f, title=title, lang=lang, ocr=ocr).pages(log=console.print)
        console.print(f"{f.name}: {len(book)} pages · book=[bold]{book[0].book if book else '?'}[/] · lang={book[0].lang if book else '?'}")
        pages += book
    if not pages:
        raise typer.Exit(1)
    save_pages(pages)
    console.print(f"[green]saved {len(pages)} pages[/] to {config.corpus().pages_file} (corpus [bold]{config.corpus().label}[/])")


# ---------------------------------------------------------------------- index


@app.command()
def index(
    batch_size: int = typer.Option(16),
    editions: str = typer.Option("all", help="Which editions to index: full | demo | all"),
    embed_backend: str = typer.Option("auto", "--embed-backend", help="ollama | st | auto"),
    embed_model: str | None = typer.Option(None, "--embed-model", help="override model name for the chosen backend"),
    corpus: CorpusOpt = None,
):
    """Chunk the crawled pages, embed (bge-m3 by default), build BM25, write data/index."""
    config.use_corpus(corpus)
    pages = load_pages()
    if editions != "all":
        pages = [p for p in pages if p.edition == editions]
    chunks = chunk_pages(pages)
    emb = embed.get_embedder(None if embed_backend == "auto" else embed_backend, embed_model)
    console.print(f"{len(pages)} pages -> {len(chunks)} chunks; embedding with [bold]{emb.key}[/] …")
    t0 = time.time()
    vecs = emb.embed_texts([c.text for c in chunks], batch_size=batch_size)
    Store().build(chunks, vecs, emb.key, books=[p.book for p in pages if p.book])
    console.print(f"[green]index built[/] in {time.time() - t0:.0f}s -> {config.corpus().index_dir}")


# --------------------------------------------------------------------- search


@app.command()
def search(
    query: str,
    k: int = typer.Option(10, "-k"),
    lang: LangOpt = None,
    edition: EditionOpt = "full",
    rerank: bool | None = typer.Option(None, "--rerank/--no-rerank", help="Rerank (default: auto, on when a reranker model is available)."),
    show_text: bool = typer.Option(False, "--text", help="Print chunk bodies."),
    corpus: CorpusOpt = None,
):
    """Hybrid retrieval only (no LLM) - inspect what the answerer would see."""
    config.use_corpus(corpus)
    engine = Engine(config.corpus().name, progress=ui.meta)
    hits = engine.search(Query(query, top_k=k, lang=lang, edition=edition, rerank=rerank))
    ui.hits(query, hits, show_text)


# ------------------------------------------------------------------------ ask


@app.command()
def ask(
    question: str,
    backend: str | None = typer.Option(None, "--backend", "-b", help="anthropic | ollama (default: auto)"),
    model: str | None = typer.Option(None, "--model", "-m"),
    k: int = typer.Option(10, "-k"),
    lang: LangOpt = None,
    edition: EditionOpt = "full",
    rerank: bool | None = typer.Option(None, "--rerank/--no-rerank", help="Rerank (default: auto, on when a reranker model is available)."),
    no_sources: bool = typer.Option(False, "--no-sources"),
    as_json: bool = typer.Option(False, "--json", help="Emit {answer, sources} JSON instead of pretty output."),
    corpus: CorpusOpt = None,
):
    """Ask one question; answer is grounded in retrieved passages with [n] citations."""
    config.use_corpus(corpus)
    engine = Engine(config.corpus().name, backend=backend, model=model, progress=None if as_json else ui.meta)
    q = Query(question, top_k=k, lang=lang, edition=edition, rerank=rerank)
    if not as_json:
        ui.answer(engine, q, [], show_sources=not no_sources)
        return
    print(json.dumps(engine.answer(q).to_dict(), ensure_ascii=False, indent=2))


@app.command()
def chat(
    backend: str | None = typer.Option(None, "--backend", "-b"),
    model: str | None = typer.Option(None, "--model", "-m"),
    k: int = typer.Option(10, "-k"),
    lang: LangOpt = None,
    edition: EditionOpt = "full",
    corpus: CorpusOpt = None,
):
    """Interactive multi-turn session (retrieval runs on every turn). Ctrl-D or /quit to exit."""
    config.use_corpus(corpus)
    os.environ.setdefault("STAVBY_OLLAMA_KEEP_ALIVE", "-1")  # keep models resident between turns
    engine = Engine(config.corpus().name, backend=backend, model=model, progress=ui.meta)
    history: list[dict] = []
    console.print(Panel("Ask about the textbook. Commands: /quit, /reset, /lang cs|en", title="stavby chat"))
    with console.status("warming models…", spinner="dots"):
        engine.warm()
    while True:
        try:
            q = console.input("[bold cyan]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/reset":
            history.clear(); console.print("[dim]history cleared[/]"); continue
        if q.startswith("/lang "):
            lang = q.split()[1]; console.print(f"[dim]lang={lang}[/]"); continue
        answer = ui.answer(engine, Query(q, top_k=k, lang=lang, edition=edition), history, show_sources=True)
        # Keep history compact: store the bare question, not the passages, so the context stays small.
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
        history[:] = history[-8:]


# ----------------------------------------------------------------------- warm


@app.command()
def warm(
    backend: str | None = typer.Option(None, "--backend", "-b"),
    model: str | None = typer.Option(None, "--model", "-m"),
    corpus: CorpusOpt = None,
):
    """Load the index and push the models into Ollama's memory (keep_alive from $STAVBY_OLLAMA_KEEP_ALIVE, default 30m)."""
    config.use_corpus(corpus)
    t = Engine(config.corpus().name, backend=backend, model=model, progress=ui.meta).warm()
    console.print("[green]warm[/] " + " · ".join(f"{k} {v:.1f}s" for k, v in t.items()))


# ----------------------------------------------------------------------- serve


def _ensure_ollama() -> None:
    """Start `ollama serve` in the background when the API is not reachable and the binary exists."""
    import httpx

    try:
        if httpx.get(config.OLLAMA_URL.rstrip("/") + "/api/version", timeout=2).status_code == 200:
            return
    except httpx.HTTPError:
        pass
    import shutil
    import subprocess

    if shutil.which("ollama") is None:
        console.print("[red]ollama not installed[/] - run: [bold]./install.sh[/]  or see https://ollama.com/download")
        raise typer.Exit(1)
    console.print("ollama not reachable - starting [bold]ollama serve[/] in the background…")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    for _ in range(30):
        try:
            if httpx.get(config.OLLAMA_URL.rstrip("/") + "/api/version", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    console.print("[red]could not start ollama[/] - run `ollama serve` manually and retry.")
    raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (0.0.0.0 to expose on the LAN)."),
    port: int = typer.Option(8000, "--port", "-p"),
    backend: str | None = typer.Option(None, "--backend", "-b", help="anthropic | ollama (default: auto)"),
    model: str | None = typer.Option(None, "--model", "-m"),
    corpus: CorpusOpt = None,
    warm: bool = typer.Option(True, "--warm/--no-warm", help="Load the index and models in the background at startup."),
    reload: bool = typer.Option(False, "--reload", help="Restart on code changes (development)."),
):
    """HTTP API + web UI (needs the `web` extra): http://127.0.0.1:8000/ and /api/ask.

    Bootstraps a fresh machine as far as it safely can: starts Ollama if it is not reachable
    (`ollama serve`, only when the binary exists), and points at the index if one is missing -
    everything else is an explicit `stavby crawl` / `index` away.
    """
    config.use_corpus(corpus)
    if not (config.corpus().index_dir / "meta.json").exists():
        console.print(f"[red]no index for corpus {config.corpus().label!r}[/] - run: [bold]stavby crawl && stavby index[/]")
        raise typer.Exit(1)
    _ensure_ollama()
    # Env, not arguments: the module-level `stavby_rag.server:app` (used by --reload's child process) reads these.
    os.environ.setdefault("STAVBY_OLLAMA_KEEP_ALIVE", "-1")  # keep models resident between requests
    if corpus:
        os.environ["STAVBY_CORPUS"] = corpus
    if backend:
        os.environ["STAVBY_SERVE_BACKEND"] = backend
    if model:
        os.environ["STAVBY_SERVE_MODEL"] = model
    os.environ["STAVBY_SERVE_WARM"] = "1" if warm else "0"
    try:
        import uvicorn

        from .server import create_app
    except ImportError as e:
        console.print(f"[red]web server dependencies missing[/] ({e}).\n"
                      "Install the web extra: [bold]uv sync --extra web[/]  or  [bold]uv tool install 'stavby-rag[web]'[/]")
        raise typer.Exit(1) from e
    console.print(f"stavby serve · corpus [bold]{config.corpus().label}[/] · http://{host}:{port}/  (Ctrl-C to stop)")
    if reload:
        uvicorn.run("stavby_rag.server:app", host=host, port=port, log_level="info", reload=True)
    else:
        uvicorn.run(create_app(backend=backend, model=model, warm=warm), host=host, port=port, log_level="info")


# ----------------------------------------------------------------------- eval


@dataclass
class EvalScore:
    """hit@1 / hit@k tallies for the expected chapter and, where given, the expected sections."""

    k: int
    total: int = 0
    chapter_1: int = 0
    chapter_k: int = 0
    sec_total: int = 0
    sec_1: int = 0
    sec_3: int = 0

    def record(self, row: dict, hits: list[Hit]) -> tuple[str, str, bool]:
        """Score one question; returns (expected label, retrieved label, expected chapter in top-k)."""
        chapters = [h.chunk.chapter for h in hits]
        sections = [h.chunk.section or "-" for h in hits]
        in_topk = row["chapter"] in chapters
        self.total += 1
        self.chapter_1 += bool(chapters) and chapters[0] == row["chapter"]
        self.chapter_k += in_topk
        expected = str(row["chapter"])
        if row.get("sections"):
            want = set(row["sections"])
            self.sec_total += 1
            self.sec_1 += sections[0] in want
            self.sec_3 += any(s in want for s in sections[:3])
            expected += " §" + "/".join(row["sections"])
        got = " ".join(f"{c}/{s}" for c, s in zip(chapters, sections))
        return expected, got, in_topk

    def summary(self) -> list[str]:
        lines = [(f"chapter: hit@1 = {self.chapter_1}/{self.total} ({self.chapter_1 / self.total:.0%})"
                  f"   hit@{self.k} = {self.chapter_k}/{self.total} ({self.chapter_k / self.total:.0%})")]
        if self.sec_total:
            lines.append(f"section: hit@1 = {self.sec_1}/{self.sec_total} ({self.sec_1 / self.sec_total:.0%})"
                         f"   hit@3 = {self.sec_3}/{self.sec_total} ({self.sec_3 / self.sec_total:.0%})")
        return lines


@app.command()
def eval(
    questions: str = typer.Option("eval/questions.jsonl", help="JSONL with {q, lang, chapter[, section]}"),
    k: int = typer.Option(10, "-k"),
    rerank: bool | None = typer.Option(None, "--rerank/--no-rerank"),
    corpus: CorpusOpt = None,
):
    """Retrieval sanity check: is the expected chapter among the top-k passages? Reports hit@1 / hit@k."""
    config.use_corpus(corpus)
    store = Store().load()
    with open(questions, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    score = EvalScore(k)
    t = Table(title=f"retrieval eval (k={k})"); t.add_column("lang"); t.add_column("exp"); t.add_column("got top-k chapter/section"); t.add_column("q", overflow="fold")
    for row in rows:
        cfg = RetrievalConfig(top_k=k, rerank=rerank, lang=row.get("lang"), editions=("full",))
        expected, got, ok = score.record(row, Retriever(store, cfg).search(row["q"]))
        t.add_row(row.get("lang", "?"), expected, ("[green]" if ok else "[red]") + got + "[/]", row["q"][:70])
    console.print(t)
    for line in score.summary():
        console.print(line)


# ---------------------------------------------------------------------- misc


@app.command()
def stats(corpus: CorpusOpt = None):
    """Corpus and index statistics."""
    config.use_corpus(corpus)
    console.print(f"corpus [bold]{config.corpus().label}[/] · {config.corpus().dir}")
    pages = load_pages()
    by = collections.Counter((p.lang, p.edition) for p in pages)
    t = Table(title="pages"); t.add_column("lang"); t.add_column("edition"); t.add_column("pages", justify="right"); t.add_column("chars", justify="right")
    for (lang, ed), n in sorted(by.items()):
        t.add_row(lang, ed, str(n), f"{sum(len(p.text) for p in pages if p.lang == lang and p.edition == ed):,}")
    console.print(t)
    books = sorted({p.book for p in pages if p.book})
    console.print("books:", ", ".join(books) if books else "- (crawled textbook)")
    chapters = collections.Counter(p.chapter for p in pages if p.lang == "cs" and p.edition == "full")
    console.print("cs/full pages per chapter:", dict(sorted(chapters.items(), key=lambda kv: (kv[0] is None, kv[0]))))
    meta = config.corpus().index_dir / "meta.json"
    if meta.exists():
        console.print("index:", json.loads(meta.read_text()))
    else:
        console.print("[yellow]no index yet[/]")


@app.command()
def doctor(corpus: CorpusOpt = None):
    """Check backends, models and data files."""
    config.use_corpus(corpus)
    console.print(f"corpus [bold]{config.corpus().label}[/] · {config.corpus().dir}")
    rows = [
        ("pages.jsonl", config.corpus().pages_file.exists()),
        ("index", (config.corpus().index_dir / "meta.json").exists()),
        ("ANTHROPIC_API_KEY", bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))),
        (f"ollama @ {config.OLLAMA_URL}", OllamaBackend.reachable()),
    ]
    try:
        import torch

        rows.append(("torch mps", torch.backends.mps.is_available()))
    except Exception:  # noqa: BLE001 - a broken torch install must not break `doctor`
        rows.append(("torch", False))
    for name, ok in rows:
        console.print(f"[{'green' if ok else 'red'}]{'ok ' if ok else 'no '}[/] {name}")

    def row(ok, label):
        console.print(f"[{'green' if ok else 'yellow'}]{'ok ' if ok else 'no '}[/] {label}")

    row(embed.ollama_has_model(config.OLLAMA_EMBED_MODEL), f"ollama embed model: {config.OLLAMA_EMBED_MODEL}")
    row(embed.ollama_has_model(config.OLLAMA_RERANK_MODEL), f"ollama rerank model: {config.OLLAMA_RERANK_MODEL}")
    row(embed.ollama_has_model(config.OLLAMA_MODEL), f"ollama LLM: {config.OLLAMA_MODEL}")
    row(embed.hf_cached(config.EMBED_MODEL), f"HF cache (st embed): {config.EMBED_MODEL}")
    row(embed.hf_cached(config.RERANK_MODEL), f"HF cache (st rerank): {config.RERANK_MODEL}")
    console.print(f"resolved: embed={':'.join(embed.resolve_embed_backend())} · rerank={embed.resolve_rerank_backend() or 'none'} · claude={config.ANTHROPIC_MODEL}")
    meta = config.corpus().index_dir / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        console.print(f"index built with: {m.get('embed_backend')}:{m.get('embed_model')} ({m.get('n_chunks')} chunks, {m.get('dim')}-d)")
        if m.get("books"):
            console.print("books:", ", ".join(m["books"]))


if __name__ == "__main__":
    app()
