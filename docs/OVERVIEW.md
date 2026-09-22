# Stavby RAG – how it works

*Plain-language overview for people who will use or run the textbook assistant. Last updated 2026‑09‑17.*

## What it is

A question-answering assistant for one book: the CTU (ČVUT) Faculty of Civil Engineering multimedia textbook *Příprava a realizace staveb a objektů* (department K122). You ask in Czech or English, it answers in the same language, and every claim carries a numbered citation that links back to the exact page of the textbook. It runs entirely on one laptop; nothing about the book leaves the machine unless you opt in to Claude for the final answer.

Two ways to talk to it:

- **Terminal** – `stavby ask "…"` for one question, `stavby chat` for a conversation.
- **Browser** – `stavby serve` starts a small web app on http://localhost:8000 with a chat window, streaming answers and clickable sources. The same server exposes a JSON/SSE API so other tools can call it.

## The idea in one paragraph

Large language models are good at writing and bad at remembering specifics. So we do not ask the model to *know* the textbook. We keep the textbook in a searchable index, find the handful of passages that are relevant to the question, and hand exactly those passages to the model with strict instructions: answer only from them, cite them, say when they do not cover the question. This pattern is called retrieval-augmented generation (RAG). The quality of the answer therefore depends mostly on the quality of the search, which is where most of the engineering went.

## Building the index (done once)

1. **Crawl.** The textbook is a 1990s-style frameset site in Windows‑1250 encoding with four editions (Czech/English, full/demo). A scoped crawler mirrors the pages, decodes them, throws away navigation chrome, and keeps ~400 content pages with their chapter, section, title and figure list.
2. **PDF books (optional).** Any PDF can be added as a separate corpus. Scanned or broken PDFs are OCR'd with the macOS Vision framework, reading order is rebuilt, headings become chapter/section.
3. **Chunk.** Pages are cut into ~1400-character windows with 200 characters of overlap. Each chunk gets a breadcrumb prefix (*Učebnice › Kapitola 4 › 4.1 Stupně rozestavěnosti*) so the embedding model knows where the text sits. Result: ~1.4k chunks.
4. **Embed.** Every chunk is turned into a 4096-dimensional vector by Qwen3‑Embedding‑8B (the official 8‑bit GGUF build, served by Ollama). Vectors are stored as one small matrix; brute-force cosine search over 1.4k rows takes under a millisecond, so no vector database is needed.
5. **Lexical index.** The same chunks go into a BM25 index with lowercasing, diacritics folding and a crude 6-letter prefix stem, because Czech inflection defeats exact matching (*rozestavěnosti* vs *rozestavěnost*).

## Answering a question (every time)

1. **Language and scope.** Czech is detected from diacritics or function words; the search is restricted to that language and to the full edition.
2. **Two searches, fused.** Top‑40 by vector similarity and top‑40 by BM25 are merged with reciprocal rank fusion. Dense search catches paraphrases; BM25 catches exact terminology, section numbers and law numbers.
3. **Rerank.** The top 20 candidates are judged one by one by Qwen3‑Reranker‑8B: “does this passage answer the question, yes or no?”. The probability of *yes* is read straight from the model's token log-probabilities. The reranker order is fused with the retrieval order again, so one confident but off-topic *yes* cannot leapfrog everything both searches agreed on. This step lifted section-level accuracy from 79 % to 91 % on our 48-question evaluation set.
4. **Generate.** The best 10 passages, numbered, go to the answer model together with a fixed system prompt. Default is Claude (claude-opus-5) when an API key is present; otherwise the local qwen3:14b through Ollama. The answer streams back token by token with `[n]` citations, and the UI turns those into links.

## Where the time goes, and what we did about it

On a 24 GB Apple M4 Pro the models are big: embedder 8 GB, reranker 8.7 GB, local LLM 9.3 GB. The reranker is the slow step – it has to read ~650 tokens for each of 20 passages, and prompt processing runs at roughly 450 tokens per second on an idle machine (about 22 seconds per question), half that when other heavy processes share the GPU. Loading a model from disk costs another 4–5 seconds, and Ollama evicts models when memory is short.

What the engine does to hide this:

- **Load once, keep warm.** The web server (and `stavby chat`) keep the index in memory and ask Ollama to keep the embedder and reranker resident (`keep_alive=-1`). Contexts were shrunk to 2k tokens so both fit together on a machine that is not otherwise loaded.
- **Warm-up at start.** `stavby serve` pushes every model into memory before the first user arrives.
- **Caches.** Query embeddings, retrieval results and full answers are memoised; a repeated question on the shared web UI returns instantly, and the Ollama prompt cache makes even a partial repeat cheap.
- **Serialise the GPU.** All Ollama traffic goes through one lock. Parallel requests on one laptop would only make the models evict each other.
- **Recommended Ollama settings** (`install.sh --tune-ollama`): flash attention and an 8‑bit KV cache roughly halve the memory the contexts need. Measured and rejected: parallel Ollama slots (no faster on Metal), bigger prompt batches (noise).

Using Claude for the final answer is the fastest configuration: only the two retrieval models need to live in memory, and they can stay there permanently.

## Installing

```
git clone https://github.com/korotole/mmu stavby-rag && cd stavby-rag
./install.sh --with-index            # uv + Ollama + models + `stavby` command + crawl + index
./install.sh --claude                # also store an Anthropic key for answers
stavby doctor                        # what is configured and reachable
stavby ask "Co je stavebně technologická studie?"
stavby serve                         # http://localhost:8000
```

The package is a normal Python wheel (`uv build`, `uv tool install "stavby-rag[web]"`). Data lives in `~/.stavby/data` (or in the repo's `data/` when run from a checkout); configuration is environment variables or a `.env` file there.

## Numbers worth knowing

| | |
|---|---|
| Content pages crawled | 398 (4 editions) |
| Chunks in the default index | 1 401 × 4096‑d |
| Retrieval eval (48 questions) | chapter hit@1 94 %, section hit@1 91 %, section hit@3 98 % |
| Reranker cost | ~20 passages × ~650 tokens per question |
| Models on disk | ~26 GB (embedder + reranker + local LLM) |

## Limits and next steps

Figures are referenced but not understood (captioning them with a vision model is the obvious next step). Evaluation covers retrieval only; answer quality has no automatic judge yet. Demo editions are indexed but excluded from search by default. There is no query rewriting between Czech and English.
