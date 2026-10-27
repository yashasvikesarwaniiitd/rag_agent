"""LLM adapters: Claude Haiku 5.5, a scripted fake for tests, and an extractive fake that
lets you try the whole app at $0 with no API key."""
from __future__ import annotations

import re
from collections.abc import Callable

from ..ports import LLMResult

DEFAULT_MODEL = "claude-haiku-5-5"


class AnthropicLLM:
    def __init__(self, client=None, model: str = DEFAULT_MODEL):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model

    def complete(self, *, system: str, user: str, max_tokens: int = 1024) -> LLMResult:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            # The system prompt is identical on every question, so cache it.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            # Haiku 5.5 has no server-side fallback. Without this check an empty text would
            # be treated as an answer and fail the citation gate with a misleading reason.
            detail = getattr(getattr(resp, "stop_details", None), "category", None)
            raise RuntimeError(f"the model declined this request (category: {detail or 'unknown'})")
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = getattr(resp, "usage", None)
        return LLMResult(text=text, stop_reason=resp.stop_reason,
                         input_tokens=getattr(usage, "input_tokens", 0) or 0,
                         output_tokens=getattr(usage, "output_tokens", 0) or 0,
                         model=getattr(resp, "model", self.model))


class ScriptedLLM:
    """Returns canned replies in order (or computes one from the prompt) and records every
    call, so tests can assert both on what the pipeline did and on what it sent."""

    def __init__(self, replies: list[str] | Callable[[str, str], str]):
        self._replies = replies
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 1024) -> LLMResult:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        if callable(self._replies):
            text = self._replies(system, user)
        else:
            text = self._replies[len(self.calls) - 1]
        return LLMResult(text=text, model="scripted")


_SOURCE_BLOCK = re.compile(r"<source id=\"(\d+)\"[^>]*>\s*(.*?)\s*</source>", re.S)


class ExtractiveLLM:
    """No model at all: quotes the first sentences of the top-ranked source and cites it.
    It shows whether RETRIEVAL found the right page, which is the first thing to check."""

    def __init__(self, max_chars: int = 400):
        self.max_chars = max_chars

    def complete(self, *, system: str, user: str, max_tokens: int = 1024) -> LLMResult:
        blocks = _SOURCE_BLOCK.findall(user)
        if not blocks:
            return LLMResult(text="NOT_FOUND", model="extractive")
        n, text = blocks[0]
        excerpt = " ".join(text.split())[: self.max_chars].rstrip()
        return LLMResult(text=f"Most relevant passage: \"{excerpt}...\" [{n}]",
                         model="extractive")
