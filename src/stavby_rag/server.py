"""HTTP API + single-page web UI over :class:`engine.Engines` (``stavby serve``).

Run with ``stavby serve`` or ``uvicorn stavby_rag.server:app``. Every engine call runs off the
event loop (threadpool or a dedicated thread), because ``Engine`` serialises Ollama traffic with
a lock and would otherwise stall the loop. Streaming answers are sent as Server-Sent Events in a
fixed order: ``meta`` -> ``sources`` -> ``token``* -> ``done`` (or ``error``).

Environment: ``STAVBY_API_TOKEN`` (optional bearer token for ``/api/*``), ``STAVBY_CORS_ORIGINS``
(comma list, default ``*``), ``STAVBY_WARM_CORPORA`` (extra corpora to warm at startup),
``STAVBY_SERVE_BACKEND`` / ``STAVBY_SERVE_MODEL`` / ``STAVBY_SERVE_WARM`` (used by the module-level
``app`` so ``--reload`` can pass CLI options through to the reloaded process).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from importlib import resources
from typing import Literal

# The web server keeps models resident between requests; must be set before ``config`` is imported.
os.environ.setdefault("STAVBY_OLLAMA_KEEP_ALIVE", "-1")

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from . import __version__, config  # noqa: E402
from . import engine as eng  # noqa: E402
from .engine import Answer, Engine, Query  # noqa: E402
from .retrieve import Hit  # noqa: E402

logger = logging.getLogger("stavby.server")
MAX_HISTORY = 8


# ------------------------------------------------------------------ schemas
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    corpus: str | None = None  # None = the server's default corpus; "" = the crawled textbook
    lang: Literal["cs", "en"] | None = None
    k: int = Field(10, ge=1, le=50)
    edition: Literal["full", "demo", "all"] = "full"
    rerank: bool | None = None
    history: list[Turn] = Field(default_factory=list)
    stream: bool = False

    def query(self) -> Query:
        return Query(question=self.question.strip(), top_k=self.k, lang=self.lang, edition=self.edition, rerank=self.rerank)

    def turns(self) -> list[dict]:
        return [t.model_dump() for t in self.history][-MAX_HISTORY:]


# ------------------------------------------------------------------ helpers
def _sources(hits: list[Hit]) -> list[dict]:
    """Same shape as ``Answer.sources()`` so the UI has one renderer."""
    return Answer(question="", answer="", hits=hits, backend="", lang="").sources()


def _corpus_row(name: str) -> dict:
    """Cheap listing from ``meta.json`` - does not load the index."""
    meta_file = config.Corpus(name).index_dir / "meta.json"
    try:
        meta = json.loads(meta_file.read_text())
    except (OSError, ValueError):
        meta = {}
    return {"name": name, "label": config.Corpus(name).label, "chunks": meta.get("n_chunks"), "books": meta.get("books", [])}


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _in_thread(gen: Iterator[str]) -> AsyncIterator[str]:
    """Drive a sync generator in ONE dedicated thread and hand its items to the event loop.

    ``Engine._generate`` holds an ``RLock`` across yields, so every ``next()`` must run in the
    same thread - a threadpool (``iterate_in_threadpool``) may hop threads and break the lock.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    end = object()

    def worker() -> None:
        try:
            for item in gen:
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as e:  # noqa: BLE001 - forwarded to the consumer
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, end)

    threading.Thread(target=worker, name="stavby-generate", daemon=True).start()
    while True:
        item = await queue.get()
        if item is end:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _cache_key(e: Engine, q: Query):
    return (e.corpus.name, q, e.label)


# --------------------------------------------------------------------- app
def create_app(engines: eng.Engines | None = None, *, backend: str | None = None, model: str | None = None,
               warm: bool = True) -> FastAPI:
    engines = engines or eng.Engines(backend=backend, model=model, progress=logger.info)

    def default_corpus() -> str:
        return config.corpus().name

    def get_engine(corpus: str | None) -> Engine:
        try:
            return engines.get(default_corpus() if corpus is None else corpus)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    def warm_all() -> None:
        names = [default_corpus()] + [n.strip() for n in os.environ.get("STAVBY_WARM_CORPORA", "").split(",") if n.strip()]
        for name in dict.fromkeys(names):
            try:
                engines.get(name).warm()
            except Exception as e:  # noqa: BLE001 - warm-up is best effort; requests still work lazily
                logger.warning("warm-up of corpus %r failed: %s", name or "default", e)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if warm:
            threading.Thread(target=warm_all, name="stavby-warm", daemon=True).start()
        yield

    app = FastAPI(title="stavby-rag", version=__version__, lifespan=lifespan)
    origins = [o.strip() for o in os.environ.get("STAVBY_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    # ------------------------------------------------------------- errors
    @app.exception_handler(HTTPException)
    async def _http_error(_r: Request, exc: HTTPException):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(Exception)
    async def _any_error(_r: Request, exc: Exception):
        status = 404 if isinstance(exc, FileNotFoundError) else 400 if isinstance(exc, ValueError) else 500
        logger.exception("request failed")
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=status)

    # --------------------------------------------------------------- auth
    def auth_required() -> bool:
        return bool(os.environ.get("STAVBY_API_TOKEN"))

    async def require_token(request: Request) -> None:
        token = os.environ.get("STAVBY_API_TOKEN")
        if not token:
            return
        header = request.headers.get("authorization", "")
        given = header[7:] if header.lower().startswith("bearer ") else request.query_params.get("token", "")
        if given != token:
            raise HTTPException(401, "missing or invalid token", headers={"WWW-Authenticate": "Bearer"})

    # ------------------------------------------------------------- routes
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index_page():
        return resources.files("stavby_rag").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    async def health():
        info: dict | None = None
        error: str | None = None
        try:
            info = await run_in_threadpool(lambda: get_engine(None).info())
        except Exception as e:  # noqa: BLE001 - health must answer even when the engine cannot start
            error = str(getattr(e, "detail", e))
        body = {"status": "ok" if error is None else "degraded", "version": __version__, "auth_required": auth_required(),
                "corpora": engines.available(), "default_corpus": default_corpus(), "default": info}
        if error:
            body["error"] = error
        return body

    api = APIRouter(prefix="/api", dependencies=[Depends(require_token)])

    @api.get("/corpora")
    async def corpora():
        return [_corpus_row(n) for n in engines.available()]

    @api.get("/search")
    async def search(q: str, corpus: str | None = None, k: int = 10, lang: Literal["cs", "en"] | None = None,
                     edition: Literal["full", "demo", "all"] = "full", rerank: bool | None = None):
        e = get_engine(corpus)
        query = Query(question=q.strip(), top_k=max(1, min(k, 50)), lang=lang, edition=edition, rerank=rerank)
        hits = await run_in_threadpool(e.search, query)
        return {"lang": query.config().lang, "corpus": e.corpus.label,
                "hits": [{**s, "body": h.chunk.body} for s, h in zip(_sources(hits), hits)]}

    @api.post("/ask")
    async def ask(req: AskRequest):
        e = get_engine(req.corpus)
        if req.stream:
            return _stream_response(e, req)
        ans = await run_in_threadpool(e.answer, req.query(), req.turns())
        return ans.to_dict()

    @api.post("/ask/stream")
    async def ask_stream(req: AskRequest):
        return _stream_response(get_engine(req.corpus), req)

    @api.post("/cache/clear")
    async def cache_clear():
        n = 0
        for name in engines.available():
            try:
                engines.get(name).clear_cache()
                n += 1
            except FileNotFoundError:
                pass
        return {"cleared": n}

    def _stream_response(e: Engine, req: AskRequest) -> StreamingResponse:
        q, history = req.query(), req.turns()

        async def events() -> AsyncIterator[str]:
            t0 = time.time()
            try:
                cached = e.answer_cache.get(_cache_key(e, q)) if not history else None
                if cached is not None:
                    yield _sse("meta", {"lang": cached.lang, "backend": cached.backend, "corpus": e.corpus.label, "cached": True})
                    yield _sse("sources", cached.sources())
                    yield _sse("token", {"t": cached.answer})
                    yield _sse("done", {"timings": {k: round(v, 2) for k, v in cached.timings.items()}, "cached": True})
                    return
                hits, tokens = await run_in_threadpool(e.stream, q, history)
                t1 = time.time()
                yield _sse("meta", {"lang": q.config().lang, "backend": e.label, "corpus": e.corpus.label, "cached": False})
                yield _sse("sources", _sources(hits))
                parts: list[str] = []
                async for tok in _in_thread(tokens):
                    parts.append(tok)
                    yield _sse("token", {"t": tok})
                t2 = time.time()
                timings = {"retrieve": round(t1 - t0, 2), "generate": round(t2 - t1, 2), "total": round(t2 - t0, 2)}
                if not history:  # same rule as Engine.answer: only history-free answers are reusable
                    e.answer_cache.put(_cache_key(e, q), Answer(question=q.question, answer="".join(parts), hits=hits,
                                                                backend=e.label, lang=q.config().lang or "", timings=timings))
                yield _sse("done", {"timings": timings, "cached": False})
            except Exception as exc:  # noqa: BLE001 - the HTTP status is already 200; report in-band
                logger.exception("stream failed")
                yield _sse("error", {"error": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    app.include_router(api)
    return app


app = create_app(backend=os.environ.get("STAVBY_SERVE_BACKEND") or None, model=os.environ.get("STAVBY_SERVE_MODEL") or None,
                 warm=os.environ.get("STAVBY_SERVE_WARM", "1") not in ("0", "false", "no"))
