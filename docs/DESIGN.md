# stavby-rag - design notes and research log

> Original long-form README (2026-09-10): source-site research, model selection, eval results.
> The short user-facing README lives at the repo root; this file is the "why".


Local, Glean-style question answering over one source: the CTU Faculty of Civil Engineering
multimedia textbook **"Příprava a realizace staveb a objektů"** (department K122), published at
<https://technologie.fsv.cvut.cz/vyuka/podklady-k-vyuce-education/multimedialni-ucebnice-priprava-a-realizace-objektu-a-staveb/>.

You ask a question in Czech or English from the terminal; the tool retrieves the relevant textbook
passages and an LLM writes a grounded answer with numbered citations back to the original pages.

```
$ stavby ask "Jaké jsou stupně rozestavěnosti a technologické etapy?"
ollama:qwen3:8b · 8 passages · lang=cs
Stupně rozestavěnosti ... [1][3] ...

[1] 4.1 Stupně rozestavěnosti (ch. 4, cs) https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava/kap4/text41.html
[2] ...
```

---

## 1. Research findings

### 1.1 The source

* WordPress landing page links to **four editions** of a legacy HTML 4 site under
  `/aitom/podklady/`: `online-priprava` (CZ full), `online-priprava-demo` (CZ demo),
  `online-priprava-en` (EN full), `online-priprava-en-demo` (EN demo).
* Every edition is a **frameset site encoded in windows-1250**. A chapter `N` consists of
  `frameN.html` (frameset shell), `nadpisN.html` (header), `obsahN.html` (table of contents),
  `oN.html` (intro), `textNM.html` (section N.M body) and `literN.html` (references / figure list).
  Full edition nests these in `kapN/`; demo edition keeps them flat. Chapters 9 and 13 deviate
  (`obsah9.htm`, `text13uvod.html`, anchors `text132.html#1321`). Some links use backslashes.
* 13 chapters in CZ; the EN full edition has 11 (chapters 2 and 7 missing). The EN edition keeps
  the CZ folder layout (`kap5/`) but renumbers its sections (`4.1 Preproduction planning`), so the
  chapter number is taken from the printed section number, not the folder. Crawl result:
  **398 content pages** (CZ full 174 pages / 749k chars, EN full 145 / 433k, plus the two demo
  editions). The origin drops connections under mild load, so the crawler retries with backoff.
* The landing page says the full edition needs a password from the lectures, but the server
  serves it without HTTP auth; the crawler simply mirrors what is publicly reachable.
* Content is prose with inline enumerations, some tables and figure images (JPG/GIF). No PDFs
  or videos were found in scope, so the crawler mirrors HTML only. Figures are recorded per page
  (URLs) but not OCR'd/captioned.
  Chapter tables of contents, figure lists, thumbnail and photo galleries are navigation and are
  excluded from the index (they matched keywords but carried nothing an answer could use).

### 1.2 What "Glean-like" means here

Glean = connectors -> index (lexical + semantic) -> permission-aware retrieval -> LLM answer with
citations. For a single public site the connector is a crawler, permissions are moot, and the
rest is a standard **retrieval-augmented generation (RAG)** pipeline. Building it custom is
feasible and preferable: the corpus is a few hundred pages, so brute-force vector search on one
laptop beats any hosted vector DB in cost, latency and simplicity.

### 1.3 Model choices

| Component | Choice | Why |
|---|---|---|
| Embeddings | **Qwen3-Embedding-8B**, official Qwen GGUF at **Q8_0** (8 GB, 4096-d, last-token pooling declared in the GGUF header) served by Ollama; `st` backend with the bf16 HF weights as alternative | Top open multilingual retriever (MTEB multilingual ≈70.6 vs ≈63-64 for bge-m3 / mE5), 100+ languages incl. Czech, cross-lingual. Q8_0 is within noise of bf16; f16 (15 GB) would evict every other model on each query. Query side gets the Qwen instruction prefix, documents do not. Smaller options stay selectable (`bge-m3`, `multilingual-e5-*`, Ollama's Q4 `qwen3-embedding:8b`). |
| Lexical | BM25 (`bm25s`) with lowercase + diacritics folding + 6-char prefix stemming | Czech is highly inflected; a crude prefix stem gives BM25 recall for `rozestavěnosti` vs `rozestavěnost`. Exact terminology, section numbers and law numbers are lexical wins dense models miss. |
| Fusion | Reciprocal Rank Fusion (k=60) | Robust, parameter-free combination of the two rankings. |
| Reranker | **Qwen3-Reranker-8B** at **Q8_0** (8.7 GB GGUF) run through Ollama; score = P("yes") from token logprobs of the official yes/no judging prompt (case variants aggregated by log-sum-exp); final order = **RRF of reranker rank and retrieval rank** | Official Qwen3 reranker recipe executed via Ollama, one runtime for everything. Fusing with the retrieval order beat both pure reranking and no reranking on the eval set (section hit@1 79 % → 91 %). Re-scores the top 12 fused candidates (pool 20/12/8 measured identical on the eval set: section hit@1 88 %, hit@3 98 %). The bf16 original (16 GB) is only usable via the `st`/transformers path and does not fit next to an LLM in 24 GB; Q8_0 is the highest-fidelity build that does. `STAVBY_RERANK_FUSION=rerank` for reranker-only order, `--no-rerank` to skip. |
| Answer LLM | **Claude** (`claude-opus-5` via Anthropic API) when `ANTHROPIC_API_KEY` is set, else **local Ollama `qwen3:14b`** | Claude gives the best Czech answers and citation discipline. Qwen3 14B is the best Czech-capable open model that fits in 24 GB alongside the reranker (Ollama evicts idle models, so peak RAM is reranker + LLM ≈ 18 GB). `qwen3:8b` remains a faster drop-in via `--model`. |

Sources: [BentoML open-source embedding guide 2026](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models),
[Qwen3 vs BGE-M3 multilingual retrieval](https://medium.com/@mrAryanKumar/comparative-analysis-of-qwen-3-and-bge-m3-embedding-models-for-multilingual-information-retrieval-72c0e6895413),
[Multilingual E5 report](https://arxiv.org/pdf/2402.05672),
[Can local AI handle Czech? (ithope.cz)](https://www.ithope.cz/en/blog/umi-lokalni-ai-cesky-co-ukazuji-benchmarky-a-co-z-toho-plyne-pro-firmu/),
[Local Czech translation with Qwen3/Gemma3](https://www.thinkdifferent.blog/blog/local-ai-translation-beats-google-translate-and-it-s-private/),
[OpenEuroLLM-Czech on Ollama](https://ollama.com/jobautomation/OpenEuroLLM-Czech).

---

## 2. Architecture

```
                 ┌──────────────┐   windows-1250 HTML    ┌────────────────┐
 technologie.  ─▶│  Crawler     │──────────────────────▶│ data/raw/      │  mirror (4 editions)
 fsv.cvut.cz     │  scoped BFS  │                        └────────────────┘
                 │  + extractor │───── Page records ────▶ data/pages.jsonl
                 └──────────────┘        (url, lang, edition, chapter, section, title, text, images)
                          │
                          ▼
                 ┌──────────────┐  breadcrumb + 1400-char windows, 200 overlap
                 │  Chunker     │────────────────────────────────────────────▶ Chunk[]
                 └──────────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
   ┌──────────────────┐      ┌──────────────────┐
   │ Embedder         │      │ store.tokenize   │
   │  Qwen3-Embedding │      │  BM25 (bm25s)    │        data/index/
   │  via ollama | st │      └──────────────────┘
   │  dense.npy       │
   └──────────────────┘
             └────────────┬────────────┘
                          ▼
                 ┌──────────────┐  dense top-40 ∪ BM25 top-40 → RRF → Qwen3-Reranker (top 12) → RRF → top-k
                 │  Retriever   │
                 └──────────────┘
                          ▼
                 ┌──────────────┐  system prompt + numbered passages + question
                 │  Answerer    │  AnthropicBackend | OllamaBackend  (streaming)
                 └──────────────┘
                          ▼
                 ┌──────────────┐
                 │  cli.py + Ui │  stavby crawl | ingest | index | search | ask | chat | eval | stats
                 └──────────────┘
```

### Data model

* **Page** (`data/pages.jsonl`): one per content HTML page. Navigation chrome (framesets, headers,
  site map) is dropped. Metadata parsed from the URL and `<body title>`: `lang`, `edition`,
  `chapter`, `section`, `title`, figure `images`.
* **Chunk** (`data/index/chunks.jsonl`): ~1400 characters of body text with a breadcrumb prefix
  (`Učebnice … > Kapitola 4 > 4.1 Stupně rozestavěnosti`) so the embedding knows its context.
  Stable id = sha1(url#ordinal).
* **Index** (`data/index/`): `dense.npy` (L2-normalised float32, N×4096), `bm25/` (bm25s
  sparse index), `meta.json` (records `embed_backend` + `embed_model`, so queries are embedded
  exactly like the corpus regardless of current env).

### Retrieval

1. Filter mask by language (auto-detected from the question via Czech diacritics or Czech function
   words, so `Co jsou stupne rozestavenosti` still counts as Czech; override with `--lang`) and
   edition (`full` by default; the demo edition is a strict subset).
2. Dense cosine top-40 and BM25 top-40, fused with RRF.
3. Qwen3-Reranker scores the top 20 fused candidates; the reranker order and the retrieval order
   are fused again with RRF (a confident "yes" on an off-topic passage cannot leapfrog everything
   both retrievers agreed on). Keep top-k (10), drop near-duplicate neighbouring chunks.

### Generation

The system prompt pins the model to the passages, demands `[n]` citations, and mirrors the
question's language. History in `chat` keeps only bare questions/answers, not the passages, so
context stays small; retrieval runs fresh on every turn. Claude requests use prompt caching on
the system prompt, `effort: high`, and the server-side refusal fallback. The local Qwen3 runs
with its reasoning mode on (`STAVBY_OLLAMA_THINK=0` to trade accuracy for speed); reasoning
tokens are not printed, only the answer.

---

## 3. Usage

```bash
# 0. prerequisites: Python 3.12 via uv, ~3 GB disk for models, optional Ollama
uv sync

# 1. mirror the textbook (idempotent; re-run with --refresh to re-download)
uv run stavby crawl

# 2. models (all via Ollama; ≈26 GB total)
ollama pull hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q8_0   # embeddings, 8 GB (official Qwen GGUF)
ollama pull dengcao/Qwen3-Reranker-8B:Q8_0            # reranker, 8.7 GB (on by default when present)
ollama pull qwen3:14b                                 # local answer LLM, 9.3 GB (skip if you use Claude)
#    On a link that resets long transfers, `ollama pull` restarts big blobs from zero. Robust alternative:
#    fetch the registry manifest, download the model layer with `aria2c -c -x 8` (resumable), verify sha256,
#    and `ollama create <name> -f Modelfile` (FROM ./file.gguf plus the manifest's TEMPLATE/PARAMETER layers).

# 3. build the index (~1.5k chunks; ~25 min with the 8B Q8_0 embedder on M4 Pro, ~1 min with bge-m3)
uv run stavby index
#    smaller/faster alternatives: --embed-model bge-m3 (after `ollama pull bge-m3`) or --embed-model qwen3-embedding:8b (Q4)
#    optional reranker (≈2.2 GB), picked up automatically once cached:
uv run python -c "from stavby_rag.embed import get_reranker; get_reranker()"

# 4a. answer with the local model
uv run stavby ask "Co je stavebně technologická studie a co obsahuje?"

# 4b. answer with Claude
export ANTHROPIC_API_KEY=sk-ant-...
uv run stavby ask "What are the stages of construction readiness?" --backend anthropic

# inspect retrieval only
uv run stavby search "harmonogram výstavby" --text

# multi-turn
uv run stavby chat

# machine-readable
uv run stavby ask "..." --json

uv run stavby eval      # retrieval hit@1 / hit@k on eval/questions.jsonl (expected chapter per question)
uv run stavby stats     # corpus + index numbers
uv run stavby doctor    # what is configured / reachable
uv run pytest -q        # pipeline tests with a stub embedder (no weights needed)
```

### PDF books

Any PDF can be a second source: `ingest` turns each PDF page into the same `Page` record the
crawler produces, so chunking, retrieval and answering are identical. Independent corpora live
side by side under `data/corpora/<name>/`; `--corpus` (or `STAVBY_CORPUS`) selects one, and
without it every command uses the crawled textbook in `data/`.

```bash
uv run stavby ingest book.pdf another.pdf --corpus mybooks   # PDFs  -> data/corpora/mybooks/pages.jsonl
uv run stavby index --corpus mybooks                         # chunk + embed + BM25
uv run stavby ask "Co je kritická cesta?" --corpus mybooks    # search/ask/chat/eval/stats/doctor take --corpus
```

* **Text.** The PDF text layer is used when it is healthy. When it is missing (scanned book) or
  corrupt (legacy fonts without Unicode maps: Czech ř/č/ě/š/ž come out as `\x00` and stray
  symbols, which is exactly what the textbook's own PDF edition does) the page is rendered and
  OCR'd with Apple Vision (`ocrmac`, macOS, ~1.5 s per page, Czech and English). OCR lines are put
  back into reading order: a two-column page or a scanned two-page spread is split at the middle,
  paragraphs are rebuilt from indentation and vertical gaps, line-end hyphenation is repaired.
  Results are cached in `data/ocr_cache/` per file and page. `--ocr always|never` overrides the
  automatic decision.
* **Structure.** Chapter, section and title come from the bookmark outline when the PDF has one;
  otherwise from numbered headings found in the text (`3 Výrobní proces …`, `3.1 Základní pojmy …`,
  carried forward across pages); otherwise from the file name (`03_Vyrobni_proces.pdf` → chapter 3).
* **Book title** (`--title`, else PDF metadata, else the file name) goes into every chunk's
  breadcrumb, is stored in the index (`meta.json` → `books`) and is named in the system prompt.
  Citations point to `file://…/book.pdf#page=N`.

Model weights live in `~/.ollama/models` (≈26 GB for embedding + reranker + LLM) and, for the
`st` backends, `~/.cache/huggingface/hub`. All downloads are resumable; on a throttled network the
pulls are the long pole, not the crawl or the indexing.

Memory on a 24 GB Mac: embedder ≈8.2 GB, reranker ≈8.2 GB (both at 2k ctx with q8_0 KV), qwen3:14b
≈10.5-11.4 GB (16k ctx, q8_0 KV). macOS caps Metal's wired memory at ~2/3 of RAM (~16 GB) by
default. Verified on this machine even with the cap raised (`sudo sysctl
iogpu.wired_limit_mb=18432`) the embedder (9.0 GB) and reranker (7.7 GB) still cannot co-reside:
two Q8_0 8B models + macOS need more than 24 GB, and Ollama's scheduler additionally evicts
whenever the next model's prediction exceeds *free system pages* (sched.go, `system_limited`) -
6.2 GB free vs 8.1 GB predicted. Each query alternates the two models (~4-6 s reload each). On a
32 GB machine the raised cap plus more free RAM should let both stay resident; on 24 GB the
wired-limit change is harmless but not sufficient. No smaller Qwen3-Reranker GGUF exists (Q8_0
only, checked Ollama registry + HF), and the embedder must match the index (Q8_0), so there is no
quantization escape.

Measured (M4 Pro 24 GB, warm, tuned Ollama: flash attention + q8_0 KV): query embed 0.15 s,
rerank 12 passages ≈ 21 s at the default pool (20 ≈ 34 s, 8 ≈ 14 s; 1.0-1.7 s/passage; `num_batch`, `OLLAMA_NUM_PARALLEL` and parallel
workers measured - no effect, one GPU serialises), generation 14-18 s per answer with
`STAVBY_OLLAMA_THINK=1` (≈ 21 % slower than `=0`; thinking is on by default for precision).
`scripts/bench.py` reproduces all of this.

Generation (qwen3:14b, measured with a real 10-passage prompt): prefill 4 091 tok at ~137 tok/s
(≈ 30 s) + generation at ~11.6-12.2 tok/s - both at the M4 Pro memory-bandwidth ceiling; `num_ctx`,
`num_gpu`, `num_thread`, `num_batch` measured, no effect. Levers that work: (1) `STAVBY_PROMPT_SOURCE_CHARS=1000`
trims each source body in the prompt (URL lines are not sent at all - the UI maps `[n]` from the
sources array), ≈ -7 s prefill, answer quality verified unchanged; (2) llama.cpp prefix caching
makes repeated questions prefill in 0.1 s and follow-ups share the history prefix automatically;
(3) `STAVBY_OLLAMA_THINK=0` skips Qwen3's thinking phase (on a real RAG prompt thinking burned
~1 900 tokens ≈ +120 s wall - it re-derives what retrieval already found, but is kept on by
default for multi-step questions). The remaining big win would be speculative decoding
(14B target + Qwen3-0.6B draft via llama.cpp `--model-draft`, est. 2x token rate, same output
distribution), but llama-server cannot share Metal with Ollama's retrieval models (verified:
`Compute error`), so it requires migrating all models off Ollama - not done.

**Corporate network note.** Behind TLS-intercepting proxies Python fails with
`CERTIFICATE_VERIFY_FAILED` on the HuggingFace CDN while `curl` (keychain-aware) works. Export the
keychain CAs to a PEM and point Python at it (`SSL_CERT_FILE`), and disable the xet transfer
backend (`HF_HUB_DISABLE_XET=1`) - see `.env.example`.

Configuration is via environment variables (see `.env.example`): `STAVBY_EMBED_BACKEND`,
`STAVBY_OLLAMA_EMBED_MODEL`, `STAVBY_EMBED_MODEL`, `STAVBY_RERANK_BACKEND`,
`STAVBY_OLLAMA_RERANK_MODEL`, `STAVBY_RERANK_MODEL`, `STAVBY_RERANK_POOL`, `STAVBY_RERANK_FUSION`, `STAVBY_ANTHROPIC_MODEL`,
`STAVBY_ANTHROPIC_EFFORT`, `STAVBY_OLLAMA_MODEL`, `STAVBY_OLLAMA_THINK`, `STAVBY_OLLAMA_NUM_CTX`, `OLLAMA_HOST`,
`STAVBY_CORPUS`.

---

## 4. Layout

```
src/stavby_rag/
  config.py    Corpus (data/ or data/corpora/<name>/), crawl scope, model ids, chunk sizes
  crawl.py     Page, Crawler (scoped BFS + cp1250 decoding), HTML -> Page extraction, pages.jsonl I/O
  pdf.py       PdfBook: a PDF book as Page records (outline -> chapter/section/title)
  chunk.py     Chunk, Chunker (paragraph-aware windows with overlap + breadcrumbs)
  embed.py     Embedder (OllamaEmbedder | StEmbedder), Reranker (OllamaQwenReranker P(yes) via
               logprobs | CrossEncoderReranker)
  store.py     Store: on-disk index (chunks.jsonl, dense.npy, bm25/), tokenizer, masks
  retrieve.py  RetrievalConfig + guess_lang, Retriever: hybrid search + RRF + rerank + de-dup
  llm.py       Answerer (retrieve -> prompt -> stream), Anthropic/Ollama backends, backend auto-pick
  cli.py       typer CLI, Ui (console output), EvalScore
tests/         pytest suite (stub embedder; extraction, chunking, hybrid retrieval, filters)
eval/          questions.jsonl - question -> expected chapter, for `stavby eval`
data/          raw mirror, pages.jsonl, index, corpora/<name>/ (git-ignored)
```

---

## 5. Limitations / next steps

* Figures are referenced but not understood. Next: caption each figure once with a vision model
  (Claude or a local VLM) and index the captions as extra chunks.
* Evaluation is retrieval-only (`stavby eval`, 48 CZ/EN questions with expected chapter, 43 of
  them with expected section(s)). Corpus = full CZ + EN editions with navigation pages removed:

  | configuration | chapter hit@1 | section hit@1 | section hit@3 |
  |---|---|---|---|
  | Qwen3-Embedding-8B Q8_0 + BM25 + RRF, no reranker | 90 % | 79 % | 98 % |
  | + Qwen3-Reranker-8B Q8_0, reranker order only | 88 % | 81 % | 98 % |
  | + Qwen3-Reranker-8B Q8_0, RRF(reranker, retrieval), pool 30 | **94 %** | **91 %** | 98 % |
  | + same with rerank pool 20 **(default)** | **94 %** | **91 %** | 98 % |

  Earlier baselines on the 23-question set before cleanup: bge-m3 83 % chapter hit@1;
  Qwen3-Embedding-8B 87 %. Removing table-of-contents / figure-list pages from the index moved
  section hit@1 from 56 % to 72 % on that set by itself. Remaining rank-1 misses are all cases
  where a definitions section (3.1 "Základní pojmy") legitimately also answers the question.
  Next: gold answers + an LLM judge for answer quality.
* Demo editions are crawled but not searched by default (`--edition all` to include).
* Query rewriting (e.g. translating a Czech question to English to search the EN edition) is not
  done; bge-m3's cross-lingual embedding partly covers this.
