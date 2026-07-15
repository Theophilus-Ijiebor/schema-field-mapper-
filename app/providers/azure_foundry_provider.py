"""
Azure AI Foundry backend for ModelProvider.

Talks to a Foundry chat-completions deployment via the `openai` SDK's
`AzureOpenAI` client (the standard OpenAI-compatible integration path for
chat-completions-shaped models on Foundry).

Required configuration (env vars, read lazily so import never fails without
them):

    AZURE_AI_FOUNDRY_ENDPOINT       e.g. https://<your-resource>.openai.azure.com
                                     (the classic resource root -- no /openai/v1
                                     suffix; this client builds the request path
                                     itself from api_version + deployment name)
    AZURE_AI_FOUNDRY_API_KEY        Foundry / Azure OpenAI resource key
    AZURE_AI_FOUNDRY_DEPLOYMENT     the deployed model's *deployment name*,
                                     not the base model name
    AZURE_AI_FOUNDRY_API_VERSION    defaults to "2024-10-21"

If any of these are missing, `available()` returns False and the caller
falls through to the next provider in the chain (see app/providers/factory.py).
"""

from __future__ import annotations
import os
from typing import Optional

from app.providers.base import ModelProvider, extract_json, with_retry


class AzureAIFoundryProvider(ModelProvider):
    name = "azure_ai_foundry"

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None,
                 deployment: Optional[str] = None, api_version: Optional[str] = None):
        self.endpoint = endpoint or os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT")
        self.api_key = api_key or os.environ.get("AZURE_AI_FOUNDRY_API_KEY")
        self.deployment = deployment or os.environ.get("AZURE_AI_FOUNDRY_DEPLOYMENT")
        self.api_version = api_version or os.environ.get("AZURE_AI_FOUNDRY_API_VERSION", "2024-10-21")
        self._client = None
        self._init_attempted = False

    def _client_or_none(self):
        if self._init_attempted:
            return self._client
        self._init_attempted = True
        if not (self.endpoint and self.api_key and self.deployment):
            return None
        try:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        except Exception:
            self._client = None
        return self._client

    def available(self) -> bool:
        return self._client_or_none() is not None

    def _call(self, system: str, user: str, max_tokens: int, temperature: float,
              json_mode: bool) -> Optional[str]:
        client = self._client_or_none()
        if client is None:
            self.last_error = ("client could not be constructed -- check AZURE_AI_FOUNDRY_ENDPOINT, "
                                "_API_KEY, and _DEPLOYMENT are all set")
            return None

        def _request(use_json_mode: bool, token_param: str) -> str:
            kwargs = {token_param: max_tokens}
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(
                model=self.deployment,  # Azure addresses models by deployment name
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
            return resp.choices[0].message.content or ""

        def _capture(exc: Exception):
            self.last_error = f"{type(exc).__name__}: {exc}"

        # Newer model families behind Azure AI Foundry (o1, gpt-5.x, and
        # other reasoning-capable models) reject the classic `max_tokens`
        # param and require `max_completion_tokens` instead; older
        # deployments (gpt-4, gpt-35-turbo) still require the classic name.
        # Rather than hardcode per-model behavior, detect it from the API's
        # own error text and retry once with the other name -- keeps this
        # provider working across both deployment generations with zero
        # config from the caller.
        token_param = "max_tokens"
        result = with_retry(lambda: _request(json_mode, token_param), on_error=_capture)

        if result is None and self.last_error and "max_completion_tokens" in self.last_error:
            token_param = "max_completion_tokens"
            result = with_retry(lambda: _request(json_mode, token_param), on_error=_capture)

        if result is not None or not json_mode:
            return result

        # Some non-OpenAI "Models-as-a-Service" deployments behind Foundry
        # don't support response_format={"type": "json_object"} at all --
        # fall back to a plain-text request once; our system prompts already
        # instruct JSON-only output, and extract_json()'s regex fallback
        # handles a model that wraps its JSON in prose or markdown fencing.
        return with_retry(lambda: _request(False, token_param), attempts=1, on_error=_capture)

    def complete_json(self, system: str, user: str, max_tokens: int = 400,
                       temperature: float = 0.0) -> Optional[dict]:
        text = self._call(system, user, max_tokens, temperature, json_mode=True)
        return extract_json(text) if text else None

    def complete_text(self, system: str, user: str, max_tokens: int = 500,
                       temperature: float = 0.0) -> Optional[str]:
        return self._call(system, user, max_tokens, temperature, json_mode=False)