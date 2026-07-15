"""
Provider selection.

Resolution order (first available wins):
  1. An explicit `preferred` name (e.g. from TenantConfig.model_provider --
     a tenant can pin itself to "azure_ai_foundry" for data-residency
     reasons, say).
  2. Anthropic, if ANTHROPIC_API_KEY is set.
  3. Azure AI Foundry, if its endpoint/key/deployment are set.
  4. Offline deterministic fallback -- always available, so this function
     never returns a provider that can't run.
"""

from __future__ import annotations
from typing import Optional

from app.providers.base import ModelProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.azure_foundry_provider import AzureAIFoundryProvider
from app.providers.offline_provider import OfflineProvider

_REGISTRY = {
    "anthropic": AnthropicProvider,
    "azure_ai_foundry": AzureAIFoundryProvider,
    "offline": OfflineProvider,
}


def get_provider(preferred: Optional[str] = None) -> ModelProvider:
    if preferred and preferred in _REGISTRY:
        candidate = _REGISTRY[preferred]()
        if candidate.available():
            return candidate
        # Fall through deliberately -- a tenant pinned to an unavailable
        # provider still gets a working (if lower-quality) run rather than
        # a hard failure. The chosen provider's `.name` is always recorded
        # in the resulting MappingDocument so this degradation is visible.

    for name in ("anthropic", "azure_ai_foundry"):
        candidate = _REGISTRY[name]()
        if candidate.available():
            return candidate

    return OfflineProvider()
