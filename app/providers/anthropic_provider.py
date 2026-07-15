"""Anthropic Claude backend for ModelProvider."""

from __future__ import annotations
import os
from typing import Optional

from app.providers.base import ModelProvider, extract_json, with_retry


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self._client = None
        self._init_attempted = False

    def _client_or_none(self):
        if self._init_attempted:
            return self._client
        self._init_attempted = True
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except Exception:
            self._client = None
        return self._client

    def available(self) -> bool:
        return self._client_or_none() is not None

    def _call(self, system: str, user: str, max_tokens: int, temperature: float) -> Optional[str]:
        client = self._client_or_none()
        if client is None:
            self.last_error = "client could not be constructed (missing/invalid ANTHROPIC_API_KEY, or the anthropic package isn't installed)"
            return None

        def _do():
            resp = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

        def _capture(exc: Exception):
            self.last_error = f"{type(exc).__name__}: {exc}"

        return with_retry(_do, on_error=_capture)

    def complete_json(self, system: str, user: str, max_tokens: int = 400,
                       temperature: float = 0.0) -> Optional[dict]:
        text = self._call(system, user, max_tokens, temperature)
        return extract_json(text) if text else None

    def complete_text(self, system: str, user: str, max_tokens: int = 500,
                       temperature: float = 0.0) -> Optional[str]:
        return self._call(system, user, max_tokens, temperature)