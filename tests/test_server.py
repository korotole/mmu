"""HTTP API tests with a fake Engine (no index, no models). Run: `uv run pytest -q tests/test_server.py`."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from stavby_rag import engine as eng  # noqa: E402
from stavby_rag import server  # noqa: E402
from stavby_rag.chunk import Chunk  # noqa: E402
from stavby_rag.retrieve import Hit  # noqa: E402


def _hit(n: int) -> Hit:
    c = Chunk(id=f"c{n}", page_url=f"https://example.test/p{n}.htm", lang="cs", edition="full", chapter=n, section=f"{n}.1",
              title=f"Kapitola {n}", ordinal=0, text="t", body=f"tělo {n}")
    return Hit(c, score=1.0 / n, dense_rank=n - 1, bm25_rank=None, rerank_score=0.9)


class FakeEngine:
    """Quacks like engine.Engine for the parts the server touches."""

    def __init__(self, name: str = ""):
        self.corpus = type("C", (), {"name": name, "label": name or "default"})()
        self.label = "fake:model"
        self.answer_cache = eng._LRU(8)
        self.hits_cache = eng._LRU(8)
        self.calls: list = []
        self.warmed = False

    def warm(self):
        self.warmed = True
        return {}

    def info(self):
        return {"corpus": self.corpus.label, "chunks": 2, "llm": self.label, "embed": "fake:e"}

    def search(self, q):
        self.calls.append(("search", q))
        return [_hit(1), _hit(2)]

    def stream(self, q, history=()):
        self.calls.append(("stream", q, list(history)))
        hits = self.search(q)

        def gen():
            for tok in ("Odpověď ", "podle ", "[1] a [2]."):
                yield tok

        return hits, gen()

    def answer(self, q, history=(), *, use_cache=True):
        hits, toks = self.stream(q, history)
        return eng.Answer(question=q.question, answer="".join(toks), hits=hits, backend=self.label, lang=q.lang or "cs",
                          timings={"retrieve": 0.1, "generate": 0.2, "total": 0.3})

    def clear_cache(self):
        self.answer_cache.clear()
        self.hits_cache.clear()


class FakeEngines:
    def __init__(self):
        self.engines = {"": FakeEngine(), "kniha": FakeEngine("kniha")}

    def get(self, corpus=None):
        try:
            return self.engines[corpus or ""]
        except KeyError:
            raise FileNotFoundError(f"no index for corpus {corpus!r}") from None

    @staticmethod
    def available():
        return ["", "kniha"]


@pytest.fixture
def engines():
    return FakeEngines()


@pytest.fixture
def client(engines, monkeypatch):
    monkeypatch.delenv("STAVBY_API_TOKEN", raising=False)
    monkeypatch.delenv("STAVBY_CORPUS", raising=False)
    with TestClient(server.create_app(engines, warm=False)) as c:
        yield c


def _events(resp) -> list[tuple[str, dict]]:
    out, ev = [], "message"
    for line in resp.iter_lines():
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            out.append((ev, json.loads(line[5:])))
            ev = "message"
    return out


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["auth_required"] is False
    assert body["corpora"] == ["", "kniha"] and body["default"]["llm"] == "fake:model"
    assert body["version"]


def test_index_and_corpora(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Stavby · učebnice" in r.text and "/api/ask" in r.text
    rows = client.get("/api/corpora").json()
    assert [c["name"] for c in rows] == ["", "kniha"] and rows[0]["label"] == "default"


def test_ask_non_stream(client, engines):
    r = client.post("/api/ask", json={"question": "Co je etapa?", "history": [{"role": "user", "content": "x"}] * 12})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Odpověď podle [1] a [2]."
    assert [s["n"] for s in body["sources"]] == [1, 2] and body["sources"][0]["url"].endswith("p1.htm")
    assert body["timings"]["total"] == 0.3 and body["cached"] is False
    _, q, history = engines.engines[""].calls[0]
    assert q.question == "Co je etapa?" and q.top_k == 10 and len(history) == 8  # history capped


def test_ask_corpus_routing_and_errors(client, engines):
    r = client.post("/api/ask", json={"question": "q", "corpus": "kniha", "lang": "en", "k": 3})
    assert r.status_code == 200 and engines.engines["kniha"].calls[0][1].lang == "en"
    assert engines.engines["kniha"].calls[0][1].top_k == 3
    r = client.post("/api/ask", json={"question": "q", "corpus": "missing"})
    assert r.status_code == 404 and "no index" in r.json()["error"]
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_ask_stream_event_order(client, engines):
    with client.stream("POST", "/api/ask", json={"question": "Co je etapa?", "stream": True}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = _events(r)
    names = [e for e, _ in events]
    assert names[0] == "meta" and names[1] == "sources" and names[-1] == "done"
    assert set(names[2:-1]) == {"token"} and len(names[2:-1]) == 3
    meta, sources, done = events[0][1], events[1][1], events[-1][1]
    assert meta["backend"] == "fake:model" and meta["corpus"] == "default" and meta["lang"] == "cs"
    assert [s["n"] for s in sources] == [1, 2] and sources[1]["title"] == "Kapitola 2"
    assert "".join(t["t"] for e, t in events if e == "token") == "Odpověď podle [1] a [2]."
    assert done["cached"] is False and set(done["timings"]) == {"retrieve", "generate", "total"}

    # Second identical question is served from the engine's answer cache (one token, cached=True).
    with client.stream("POST", "/api/ask/stream", json={"question": "Co je etapa?"}) as r:
        events = _events(r)
    assert [e for e, _ in events] == ["meta", "sources", "token", "done"] and events[-1][1]["cached"] is True
    assert len([c for c in engines.engines[""].calls if c[0] == "stream"]) == 1


def test_stream_reports_engine_errors_in_band(client, engines):
    def boom(q, history=()):
        raise RuntimeError("ollama down")

    engines.engines[""].stream = boom
    with client.stream("POST", "/api/ask/stream", json={"question": "q"}) as r:
        events = _events(r)
    assert events == [("error", {"error": "RuntimeError: ollama down"})]


def test_search_and_cache_clear(client, engines):
    r = client.get("/api/search", params={"q": "etapa", "k": 2, "lang": "cs"})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) == 2 and hits[0]["body"] == "tělo 1" and hits[0]["n"] == 1
    engines.engines[""].answer_cache.put("k", "v")
    assert client.post("/api/cache/clear").json() == {"cleared": 2}
    assert len(engines.engines[""].answer_cache) == 0


def test_auth(engines, monkeypatch):
    monkeypatch.setenv("STAVBY_API_TOKEN", "s3cret")
    with TestClient(server.create_app(engines, warm=False)) as c:
        assert c.get("/api/health").json()["auth_required"] is True  # health stays open
        r = c.post("/api/ask", json={"question": "q"})
        assert r.status_code == 401 and r.json() == {"error": "missing or invalid token"}
        assert c.get("/api/corpora", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.post("/api/ask", json={"question": "q"}, headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/api/corpora", params={"token": "s3cret"}).status_code == 200
        assert c.get("/").status_code == 200  # the UI itself is public; it asks for the token


def test_warm_runs_on_startup(engines, monkeypatch):
    import time

    monkeypatch.delenv("STAVBY_API_TOKEN", raising=False)
    monkeypatch.delenv("STAVBY_CORPUS", raising=False)
    monkeypatch.setenv("STAVBY_WARM_CORPORA", "kniha")
    with TestClient(server.create_app(engines, warm=True)):
        for _ in range(50):
            if engines.engines[""].warmed and engines.engines["kniha"].warmed:
                break
            time.sleep(0.02)
    assert engines.engines[""].warmed and engines.engines["kniha"].warmed
