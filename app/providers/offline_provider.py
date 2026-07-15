"""
Offline deterministic provider.

`available()` is always True: this is the provider of last resort, and the
graph must always be able to complete a run even with zero external
credentials. It doesn't actually call a model -- `complete_json` always
returns None, signalling "no LLM answer available." Callers (graph nodes,
eval judges) are required to have a non-LLM fallback path for exactly this
case; this provider's job is just to make that path the *explicit, typed*
outcome of "no provider configured" rather than a crash.
"""

from __future__ import annotations
from typing import Optional

from app.providers.base import ModelProvider


class OfflineProvider(ModelProvider):
    name = "offline"

    def available(self) -> bool:
        return True

    def complete_json(self, system: str, user: str, max_tokens: int = 400,
                       temperature: float = 0.0) -> Optional[dict]:
        return None
