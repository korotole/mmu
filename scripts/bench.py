"""Latency benchmark for the local pipeline: ``uv run python scripts/bench.py [-n 20] [--no-engine]``.

Times index load, cold + warm query embedding, one rerank of N real passages (with prefill tok/s
from Ollama's ``prompt_eval_*`` fields) and ``Engine.search`` cold / warm / cached for 3 questions.
Honours the STAVBY_* env vars (OLLAMA_HOST, STAVBY_RERANK_WORKERS, STAVBY_OLLAMA_RERANK_NUM_CTX ...).
"""

from __future__ import annotations

import argparse
import statistics
import time

import httpx

from stavby_rag import config, embed
from stavby_rag.engine import Engine, Query
from stavby_rag.store import Store

QUESTIONS = ["Co je stavebně technologická studie?",
             "Jak se sestavuje harmonogram výstavby a co je kritická cesta?",
             "Jaké jsou zásady zařízení staveniště?"]


def timed(fn, *a):
    t0 = time.perf_counter()
    out = fn(*a)
    return out, time.perf_counter() - t0


def ollama_prefill(url: str, model: str, prompt: str, num_ctx: int, num_batch: int = 0) -> tuple[int, float]:
    """(prompt tokens, prompt_eval seconds) for one raw 1-token generate - raw prefill speed."""
    opts = {"num_predict": 1, "num_ctx": num_ctx, "temperature": 0}
    if num_batch:
        opts["num_batch"] = num_batch
    r = httpx.post(f"{url}/api/generate", json={"model": model, "prompt": prompt, "raw": True, "stream": False,
                                                "keep_alive": config.OLLAMA_KEEP_ALIVE, "options": opts}, timeout=600).json()
    return r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 0) / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=config.RERANK_POOL, help="passages to rerank")
    ap.add_argument("--no-engine", action="store_true", help="skip the Engine.search rounds")
    ap.add_argument("--num-batch", type=int, default=0, help="also time raw prefill with options.num_batch=N")
    args = ap.parse_args()
    print(f"ollama={config.OLLAMA_URL} workers={config.RERANK_WORKERS} embed_ctx={config.OLLAMA_EMBED_NUM_CTX} "
          f"rerank_ctx={config.OLLAMA_RERANK_NUM_CTX} keep_alive={config.OLLAMA_KEEP_ALIVE}", flush=True)

    class Rows(list):  # print each row as soon as it is measured (long runs, possible crashes)
        def append(self, row):
            super().append(row)
            print(f"  {row[0]:<32} {row[1]}", flush=True)

    rows = Rows()

    store, t = timed(lambda: Store().load())
    rows.append(("index load", f"{t:.2f}s ({len(store.chunks)} chunks)"))

    emb = store.embedder()
    _, cold = timed(emb.embed_query, QUESTIONS[0])
    warm = [timed(emb.embed_query, q)[1] for q in QUESTIONS[1:] * 2]
    rows.append(("embed query cold / warm", f"{cold:.2f}s / {statistics.median(warm):.3f}s  ({emb.key})"))

    rr = embed.get_reranker()
    if rr is not None:
        passages = [c.text for c in store.chunks if c.lang == "cs"][:args.n]
        _, load = timed(rr.score, QUESTIONS[0], passages[:1])
        _, t = timed(rr.score, QUESTIONS[0], passages)
        workers = getattr(rr, "workers", "-")
        rows.append((f"rerank {len(passages)} passages", f"{t:.1f}s  ({t / len(passages):.2f}s/passage, workers={workers}, first call {load:.1f}s)"))
        if isinstance(rr, embed.OllamaQwenReranker):
            for nb in (0, args.num_batch) if args.num_batch else (0,):
                toks, dur = ollama_prefill(rr.url, rr.model, "Text: " + passages[3] * 2, rr.num_ctx, nb)
                rows.append((f"raw prefill num_batch={nb or 'default'}", f"{toks} tok in {dur:.2f}s = {toks / max(dur, 1e-9):.0f} tok/s"))
    else:
        rows.append(("rerank", "no reranker available"))

    if not args.no_engine:
        eng = Engine()
        eng._store = store
        for label in ("cold", "warm", "cached"):
            if label == "warm":
                eng.clear_cache()
            ts = [timed(eng.search, Query(q))[1] for q in QUESTIONS]
            rows.append((f"Engine.search {label} x{len(QUESTIONS)}", " / ".join(f"{t:.2f}s" for t in ts)))

    ps = httpx.get(f"{config.OLLAMA_URL}/api/ps", timeout=5).json().get("models", [])
    rows.append(("ollama /api/ps", ", ".join(f"{m['name'].split('/')[-1]} ctx={m.get('context_length')} vram={m.get('size_vram', 0) / 2**30:.1f}G" for m in ps) or "nothing loaded"))


if __name__ == "__main__":
    main()
