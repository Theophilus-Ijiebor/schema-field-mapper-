"""
ModelProvider abstraction.

Every place in this codebase that needs "ask a model a bounded JSON question"
(field-match reasoning, table-match reasoning, eval judges for relevance /
faithfulness) goes through this interface instead of calling a vendor SDK
directly. That's what makes the deployment target swappable: today it's
Anthropic Claude and/or Azure AI Foundry; the graph and eval code never know
which one actually answered.

Contract: `complete_json` takes a system prompt and a user prompt and returns
either a parsed dict (on success) or None (on any failure -- missing
credentials, network error, malformed response). Callers must always have a
deterministic fallback for None; providers never raise for
"I couldn't get a good answer", only for genuine programmer error.
"""

from __future__ import annotations
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def with_retry(fn: Callable[[], T], attempts: int = 3, base_delay: float = 0.5,
                on_error: Optional[Callable[[Exception], None]] = None) -> Optional[T]:
    """
    Small retry-with-backoff helper shared by every provider implementation.
    Transient failures (rate limits, brief network blips) shouldn't
    immediately demote a run to the offline fallback -- but a genuinely
    unavailable/misconfigured provider still fails fast after `attempts`
    tries rather than hanging. Any exception on the final attempt is
    swallowed and None is returned, matching every provider's "None means
    no answer, never raise for a bad answer" contract used everywhere else
    in this pipeline (graph nodes, eval judges) -- but `on_error`, if given,
    is called with the final exception first, so a caller that specifically
    wants to know *why* (e.g. app/providers/selftest.py) can surface it
    without changing that contract for everyone else.
    """
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    if last_exc is not None and on_error is not None:
        on_error(last_exc)
    return None


class ModelProvider(ABC):
    name: str = "base"
    last_error: Optional[str] = None  # set by subclasses after a failed call; read by selftest.py

    @abstractmethod
    def available(self) -> bool:
        """True if this provider has what it needs (credentials, network) to run."""
        ...

    @abstractmethod
    def complete_json(self, system: str, user: str, max_tokens: int = 400,
                       temperature: float = 0.0) -> Optional[dict]:
        ...

    def complete_text(self, system: str, user: str, max_tokens: int = 500,
                       temperature: float = 0.0) -> Optional[str]:
        """
        Raw free-text completion (no forced JSON parsing). Used by the
        DeepEval adapter (app/evals/deepeval_adapter.py), whose GEval metrics
        send their own judge prompts and parse the response themselves.
        Default implementation returns None (offline-style no-op); real
        providers override this.
        """
        return None


def extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None