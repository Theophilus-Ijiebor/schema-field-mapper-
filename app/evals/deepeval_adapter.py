"""
DeepEval model adapter.

DeepEval (https://github.com/confident-ai/deepeval) is a maintained,
widely-used LLM evaluation library -- its GEval metric implements the
standard "LLM-as-judge with an explicit rubric and chain-of-thought scoring"
pattern properly (structured evaluation-step generation, JSON-schema-checked
output, consistent 0-1 scoring) instead of the ad-hoc single-shot
prompt-and-parse this codebase's `app/evals/metrics.py` used before this
integration. This module is what lets the *same* multi-provider
ModelProvider abstraction (Anthropic / Azure AI Foundry) back a DeepEval
metric, instead of hard-wiring GEval to one specific vendor's SDK the way
DeepEval's own built-in model wrappers do.

`ProviderBackedDeepEvalModel` implements DeepEval's `DeepEvalBaseLLM`
interface as a thin relay onto `ModelProvider.complete_text` -- DeepEval
builds its own judge prompts internally (evaluation-step generation, then
scoring against those steps); this adapter's only job is "send this prompt
to whichever provider is configured, return the raw text," and DeepEval
does the rest (including parsing the JSON it asked for out of that text).
"""

from __future__ import annotations
from typing import Optional

from deepeval.models.base_model import DeepEvalBaseLLM

from app.providers.base import ModelProvider

DEEPEVAL_SYSTEM_PROMPT = (
    "You are a careful evaluator. Follow the user's instructions exactly and "
    "respond in the exact format requested (usually JSON). Do not add any "
    "text outside the requested format."
)


class ProviderBackedDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, provider: ModelProvider):
        self._provider = provider
        super().__init__(model=provider.name)

    def load_model(self):
        return self

    def generate(self, prompt: str, **kwargs) -> str:
        text = self._provider.complete_text(DEEPEVAL_SYSTEM_PROMPT, prompt, max_tokens=1024, temperature=0.0)
        if text is None:
            raise RuntimeError(
                f"Provider '{self._provider.name}' returned no response for a DeepEval judge prompt "
                f"(no credentials configured, or the call failed after retries)."
            )
        return text

    async def a_generate(self, prompt: str, **kwargs) -> str:
        # DeepEval's async path is only exercised when GEval(async_mode=True);
        # this codebase always constructs metrics with async_mode=False (see
        # deepeval_metrics.py) so this just relays to the sync path.
        return self.generate(prompt, **kwargs)

    def get_model_name(self) -> str:
        return f"schema-field-mapper::{self._provider.name}"


def build_deepeval_model(provider: ModelProvider) -> Optional[ProviderBackedDeepEvalModel]:
    """Returns None if the provider can't actually answer (offline fallback),
    so callers know to use the heuristic metrics instead."""
    if not provider.available() or provider.name == "offline":
        return None
    return ProviderBackedDeepEvalModel(provider)
