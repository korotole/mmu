"""Central configuration: paths, crawl scope, model names.

Paths: ``STAVBY_HOME`` (alias ``STAVBY_ROOT``) > the source checkout when run from one > ``~/.stavby``.
``<cwd>/.env`` and ``<home>/.env`` are loaded (setdefault, no override) before the ``os.environ.get``
defaults below are evaluated.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Read simple ``KEY=VALUE`` lines into ``os.environ`` without overriding what is already set.

    Comments and blank lines are ignored, an optional ``export`` prefix and surrounding quotes
    are stripped. Deliberately minimal: no interpolation, no multi-line values.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and value:  # an empty value keeps the variable unset, so defaults still apply
            os.environ.setdefault(key, value)


def _resolve_home() -> Path:
    """Where stavby keeps its data.

    ``STAVBY_HOME`` (or the older alias ``STAVBY_ROOT``) wins. Otherwise a source checkout is
    detected by ``pyproject.toml`` two levels above this file and used as-is, so development keeps
    ``data/`` inside the repo. Installed as a package (site-packages), fall back to ``~/.stavby``.
    """
    env = os.environ.get("STAVBY_HOME") or os.environ.get("STAVBY_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file():
        return checkout
    return Path.home() / ".stavby"


# The cwd's ``.env`` (a project checkout) is read first so it may set STAVBY_HOME; then the home's
# own ``.env`` (written by install.sh). Both only fill in variables that are not already exported.
_load_dotenv(Path.cwd() / ".env")
HOME = _resolve_home()
_load_dotenv(HOME / ".env")
PROJECT_ROOT = HOME  # backwards-compatible name
DATA_DIR = HOME / "data"
RAW_DIR = DATA_DIR / "raw"  # mirrored HTML files (the crawled site, shared by all corpora)


@dataclass(frozen=True)
class Corpus:
    """Where one corpus lives: the default ``data/`` tree, or ``data/corpora/<name>/``.

    A corpus is one independent set of sources: the crawled textbook (default) or a set of
    ingested PDF books. Paths are resolved on access, never at import time.
    """

    name: str = ""

    @property
    def dir(self) -> Path:
        return DATA_DIR / "corpora" / self.name if self.name else DATA_DIR

    @property
    def pages_file(self) -> Path:
        return self.dir / "pages.jsonl"  # one extracted page per line

    @property
    def index_dir(self) -> Path:
        return self.dir / "index"

    @property
    def label(self) -> str:
        return self.name or "default"


_selected: str | None = None  # set by `--corpus`; otherwise STAVBY_CORPUS


def corpus() -> Corpus:
    """The active corpus: ``--corpus``, else ``STAVBY_CORPUS``, else the default ``data/`` tree."""
    return Corpus(_selected or os.environ.get("STAVBY_CORPUS") or "")


def use_corpus(name: str | None) -> None:
    """Select a corpus for the rest of the process; ``None`` keeps the current one."""
    global _selected
    if name:
        _selected = name

SITE_ROOT = "https://technologie.fsv.cvut.cz/aitom/podklady/"
LANDING_PAGE = (
    "https://technologie.fsv.cvut.cz/vyuka/podklady-k-vyuce-education/"
    "multimedialni-ucebnice-priprava-a-realizace-objektu-a-staveb/"
)

# The four editions of the textbook that hang off the landing page.
EDITIONS: dict[str, dict[str, str]] = {
    "online-priprava": {"lang": "cs", "edition": "full"},
    "online-priprava-demo": {"lang": "cs", "edition": "demo"},
    "online-priprava-en": {"lang": "en", "edition": "full"},
    "online-priprava-en-demo": {"lang": "en", "edition": "demo"},
}

SCOPE_RE = re.compile(
    r"^https://technologie\.fsv\.cvut\.cz/aitom/podklady/(online-priprava(?:-en)?(?:-demo)?)/"
)

USER_AGENT = "stavby-rag/0.2 (+educational RAG over the public CTU textbook; https://github.com/korotole/mmu)"
CRAWL_DELAY_S = 0.1
DEFAULT_ENCODING = "cp1250"

# Embeddings. Default: official Qwen3-Embedding-8B GGUF at Q8_0 (near-lossless, 4096-d, last-token
# pooling declared in the GGUF header) served by Ollama. STAVBY_EMBED_BACKEND: ollama | st | auto.
EMBED_BACKEND = os.environ.get("STAVBY_EMBED_BACKEND", "auto")
OLLAMA_EMBED_MODEL = os.environ.get("STAVBY_OLLAMA_EMBED_MODEL", "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q8_0")
EMBED_MODEL = os.environ.get("STAVBY_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")  # HF id for the `st` backend (bf16, 16 GB)

# Reranker. Default: Qwen3-Reranker-8B at Q8_0 via Ollama logprobs; `st` alternative is a HF
# CrossEncoder. STAVBY_RERANK_BACKEND: ollama | st | none | auto (auto = whichever is present).
RERANK_BACKEND = os.environ.get("STAVBY_RERANK_BACKEND", "auto")
OLLAMA_RERANK_MODEL = os.environ.get("STAVBY_OLLAMA_RERANK_MODEL", "dengcao/Qwen3-Reranker-8B:Q8_0")
RERANK_MODEL = os.environ.get("STAVBY_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_POOL = int(os.environ.get("STAVBY_RERANK_POOL", "12"))  # fused candidates sent to the reranker
# Pool 20/12/8 measured identical on the 48-question eval (section hit@1 88%, hit@3 98%); 12 keeps
# a margin for unseen queries at ~21 s rerank (vs 34 s at 20). See scripts/bench.py + docs/DESIGN.md.
# How the reranker order is combined with the retrieval (RRF) order:
#   "rerank" - reranker order only;  "rrf" - reciprocal-rank fusion of reranker rank and retrieval rank.
RERANK_FUSION = os.environ.get("STAVBY_RERANK_FUSION", "rrf")

# Chunking (characters; Czech runs ~3.5-4.5 chars/token). ~1400 chars (~350 tokens) is a retrieval
# precision choice, not a model limit: Qwen3-Embedding accepts 32k tokens, the reranker sees whole chunks.
CHUNK_CHARS = 1400
CHUNK_OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 80

# LLM backends.
ANTHROPIC_MODEL = os.environ.get("STAVBY_ANTHROPIC_MODEL", "claude-opus-5")
OLLAMA_MODEL = os.environ.get("STAVBY_OLLAMA_MODEL", "qwen3:14b")
OLLAMA_THINK = os.environ.get("STAVBY_OLLAMA_THINK", "1") not in ("0", "false", "no")  # Qwen3 reasoning mode
OLLAMA_NUM_CTX = int(os.environ.get("STAVBY_OLLAMA_NUM_CTX", "16384"))
ANTHROPIC_EFFORT = os.environ.get("STAVBY_ANTHROPIC_EFFORT", "high")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# How long Ollama keeps a model resident after a request ("30m", "-1" = forever). Engine/serve set
# STAVBY_OLLAMA_KEEP_ALIVE=-1 so embedder + reranker + LLM are never reloaded between queries.
# Ollama wants a duration string ("30m") or a bare number of seconds (-1 = forever), not "-1" as text.
_keep_alive = os.environ.get("STAVBY_OLLAMA_KEEP_ALIVE", "30m").strip()
OLLAMA_KEEP_ALIVE: str | int = int(_keep_alive) if _keep_alive.lstrip("-").isdigit() else _keep_alive
# Context windows requested per call (smaller = less KV memory, so more models stay co-resident).
# Chunks are <= ~1400 chars + breadcrumb (~700 tokens; `truncate` guards the rest); a rerank prompt is ~700-900 tokens.
OLLAMA_EMBED_NUM_CTX = int(os.environ.get("STAVBY_OLLAMA_EMBED_NUM_CTX", "2048"))
OLLAMA_RERANK_NUM_CTX = int(os.environ.get("STAVBY_OLLAMA_RERANK_NUM_CTX", "2048"))
# Optional llama.cpp batch size for reranker prefill (0 = Ollama default, -b 512). See scripts/bench.py.
OLLAMA_RERANK_NUM_BATCH = int(os.environ.get("STAVBY_OLLAMA_RERANK_NUM_BATCH", "0"))
# Per-source body characters in the answer prompt (0 = full ~1400-char chunk). 1000 verified
# equal answer quality on the eval questions while cutting ~7 s of prefill per query (M4 Pro).
PROMPT_SOURCE_CHARS = int(os.environ.get("STAVBY_PROMPT_SOURCE_CHARS", "1000"))
# Parallel reranker requests. Ollama serves one request per model at a time (OLLAMA_NUM_PARALLEL=1,
# its default), so extra workers only queue; pair STAVBY_RERANK_WORKERS=4 with OLLAMA_NUM_PARALLEL=4.
RERANK_WORKERS = int(os.environ.get("STAVBY_RERANK_WORKERS", "1"))

