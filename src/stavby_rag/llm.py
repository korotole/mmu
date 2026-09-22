"""Answer generation: the grounded-answer pipeline and its two LLM backends.

Backends expose ``stream(system, messages) -> Iterator[str]`` so the CLI can print
tokens as they arrive. Backend selection: explicit ``--backend``, else
``ANTHROPIC_API_KEY`` present -> anthropic, else a reachable Ollama -> ollama.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from typing import Protocol

import httpx

from . import config
from .retrieve import Hit, Retriever

CTU_TEXTBOOK = """the CTU (ČVUT) Faculty of Civil Engineering multimedia textbook
"Příprava a realizace staveb a objektů" (Construction preparation and realisation), department K122."""

PROMPT_RULES = """Answer ONLY from the numbered source passages provided in the user turn. If the passages do not contain the
answer, say so plainly and suggest which chapter might cover it - never invent regulations, numbers or names.
Cite every factual statement with the passage number in square brackets, e.g. [2] or [1][4].
Reply in the language of the question (Czech questions -> Czech answers, English -> English). Keep the
technical terminology of the textbook. Be concise and structured; use short paragraphs or bullets."""


def system_prompt(books: Sequence[str] = ()) -> str:
    """Generation prompt; ``books`` are the index's source titles (empty = the CTU textbook)."""
    source = "the following source(s): " + "; ".join(books) if books else CTU_TEXTBOOK
    return f"You are a study assistant for {source}\n\n{PROMPT_RULES}"


class Backend(Protocol):
    name: str
    model: str

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]: ...


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str = config.ANTHROPIC_MODEL):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic()

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        import anthropic

        kwargs = {
            "model": self.model,
            "max_tokens": 8000,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "output_config": {"effort": config.ANTHROPIC_EFFORT},
        }
        try:
            # Server-side refusal fallback: if the primary model declines, the API re-runs on a fallback model.
            with self.client.beta.messages.stream(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            ) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except anthropic.BadRequestError as e:
            if "fallback" not in str(e).lower():
                raise
            with self.client.messages.stream(**kwargs) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        if final.stop_reason == "refusal":
            details = getattr(final, "stop_details", None)
            yield f"\n\n[model refused: {getattr(details, 'explanation', '') or 'no explanation'}]"


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = config.OLLAMA_MODEL, url: str = config.OLLAMA_URL,
                 num_ctx: int = config.OLLAMA_NUM_CTX, keep_alive: str | int = config.OLLAMA_KEEP_ALIVE):
        self.model = model
        self.url = url.rstrip("/")
        self.num_ctx, self.keep_alive = num_ctx, keep_alive
        self._client = httpx.Client(timeout=600)  # reused across calls (keep-alive connection)

    @staticmethod
    def reachable(url: str = config.OLLAMA_URL) -> bool:
        try:
            return httpx.get(url.rstrip("/") + "/api/version", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def warm(self) -> None:
        """Load the model into Ollama now (1-token generate) with the same ``num_ctx`` as real calls,
        so the KV cache is allocated once and later requests do not trigger a reload."""
        payload = {"model": self.model, "prompt": "hi", "stream": False, "keep_alive": self.keep_alive,
                   "options": {"num_predict": 1, "num_ctx": self.num_ctx}}
        r = self._client.post(f"{self.url}/api/generate", json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama {r.status_code}: {r.text[:300]}")

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": True,
            "think": config.OLLAMA_THINK,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.2, "num_ctx": self.num_ctx},
        }
        with self._client.stream("POST", f"{self.url}/api/chat", json=payload) as r:
            if r.status_code != 200:
                r.read()
                raise RuntimeError(f"Ollama {r.status_code}: {r.text[:300]}")
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if "error" in obj:
                    raise RuntimeError(obj["error"])
                tok = obj.get("message", {}).get("content", "")
                if tok:
                    yield tok
                # message.thinking carries Qwen3's reasoning tokens; deliberately not surfaced.
                if obj.get("done"):
                    break


def pick_backend(name: str | None = None, model: str | None = None) -> Backend:
    if name is None:
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            name = "anthropic"
        elif OllamaBackend.reachable():
            name = "ollama"
        else:
            raise RuntimeError(
                "No LLM backend: set ANTHROPIC_API_KEY for Claude, or start Ollama (`ollama serve`) "
                f"and pull `{config.OLLAMA_MODEL}`."
            )
    if name == "anthropic":
        return AnthropicBackend(model or config.ANTHROPIC_MODEL)
    if name == "ollama":
        return OllamaBackend(model or config.OLLAMA_MODEL)
    raise ValueError(f"unknown backend {name!r}")


class Answerer:
    """Retrieval + grounded generation: retrieve passages, then stream a cited answer."""

    def __init__(self, retriever: Retriever, backend: Backend):
        self.retriever = retriever
        self.backend = backend

    @property
    def label(self) -> str:
        return f"{self.backend.name}:{self.backend.model}"

    def retrieve(self, question: str) -> list[Hit]:
        return self.retriever.search(question)

    def stream(self, question: str, hits: list[Hit], history: Sequence[dict] = ()) -> Iterator[str]:
        messages = [*history, {"role": "user", "content": self.user_turn(question, hits)}]
        return self.backend.stream(system_prompt(self.retriever.store.books), messages)

    def answer(self, question: str) -> tuple[str, list[Hit]]:
        """Retrieve and generate in one go, without streaming to the console."""
        hits = self.retrieve(question)
        return "".join(self.stream(question, hits)), hits

    @staticmethod
    def user_turn(question: str, hits: list[Hit]) -> str:
        """Numbered passages the answer must cite, then the question.

        Source bodies are cut to ``STAVBY_PROMPT_SOURCE_CHARS`` (default 1000 of the ~1400-char
        chunk): prefill is ~30 s for a 10-passage prompt on an M4 Pro, and the tail of a chunk is
        usually covered by the next chunk's 200-char overlap. The UI gets URLs from the sources
        array, so the prompt itself needs only number + title + location.
        """
        limit = config.PROMPT_SOURCE_CHARS
        parts = []
        for n, h in enumerate(hits, 1):
            c = h.chunk
            loc = f"chapter {c.chapter}" if c.chapter is not None else ""
            if c.section:
                loc += f", section {c.section}"
            body = c.body if limit <= 0 else c.body[:limit]
            parts.append(f"[{n}] ({c.lang}, {loc}) {c.title}\n{body}")
        return "Source passages:\n\n" + "\n\n---\n\n".join(parts) + f"\n\n===\n\nQuestion: {question}"
