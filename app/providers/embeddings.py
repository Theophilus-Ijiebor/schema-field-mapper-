"""
Embedding provider abstraction, used by app/retrieval/retriever.py.

Same philosophy as ModelProvider (base.py): one interface, multiple backends,
graceful degradation to the TF-IDF path when nothing is configured.

Compliance note re: the assignment's "don't show both schemas to one LLM
call" constraint -- embedding a field's text is not an LLM *reasoning* call,
it's a representation lookup (the same role TF-IDF plays today). `embed()`
is always called once for all SOURCE field texts and once for all
DESTINATION field texts, in two separate requests.
"""

from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Optional

from app.providers.base import with_retry


class EmbeddingProvider(ABC):
    name: str = "base"
    last_error: Optional[str] = None  # set after a failed call; read by selftest.py

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Returns one embedding vector per input text, or None on any failure."""
        ...


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Two modes, auto-detected:
      * Azure AI Foundry embedding deployment -- AZURE_AI_FOUNDRY_ENDPOINT,
        AZURE_AI_FOUNDRY_API_KEY, AZURE_AI_FOUNDRY_EMBEDDING_DEPLOYMENT.
      * Direct OpenAI -- OPENAI_API_KEY (+ optional OPENAI_EMBEDDING_MODEL,
        default "text-embedding-3-small").
    Azure preferred if both configured.
    """
    name = "openai_embeddings"

    def __init__(self):
        self._client = None
        self._mode: Optional[str] = None
        self._model: Optional[str] = None
        self._init_attempted = False

    def _client_or_none(self):
        if self._init_attempted:
            return self._client
        self._init_attempted = True

        azure_endpoint = os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT")
        azure_key = os.environ.get("AZURE_AI_FOUNDRY_API_KEY")
        azure_embed_deployment = os.environ.get("AZURE_AI_FOUNDRY_EMBEDDING_DEPLOYMENT")

        try:
            if azure_endpoint and azure_key and azure_embed_deployment:
                from openai import AzureOpenAI
                self._client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_key,
                    api_version=os.environ.get("AZURE_AI_FOUNDRY_API_VERSION", "2024-10-21"),
                )
                self._mode = "azure_ai_foundry"
                self._model = azure_embed_deployment
                return self._client

            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI
                self._client = OpenAI(api_key=openai_key)
                self._mode = "openai"
                self._model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                return self._client
        except Exception:
            self._client = None

        return None

    def available(self) -> bool:
        return self._client_or_none() is not None

    def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        client = self._client_or_none()
        if client is None:
            self.last_error = "client could not be constructed -- check credentials are set"
            return None
        if not texts:
            return None

        def _do():
            resp = client.embeddings.create(model=self._model, input=texts)
            return [item.embedding for item in resp.data]

        def _capture(exc: Exception):
            self.last_error = f"{type(exc).__name__}: {exc}"

        return with_retry(_do, on_error=_capture)


def get_embedding_provider(preferred: Optional[str] = None) -> Optional[EmbeddingProvider]:
    """Returns an available EmbeddingProvider, or None if nothing is configured
    (callers must fall back to TF-IDF in that case)."""
    if preferred == "tfidf":
        return None
    candidate = OpenAIEmbeddingProvider()
    return candidate if candidate.available() else None