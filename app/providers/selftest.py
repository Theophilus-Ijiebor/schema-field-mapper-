"""
Provider self-test.

    python3 -m app.providers.selftest

Reports, for every configured backend, whether it's actually reachable right
now: credentials present is necessary but not sufficient (a key can be
present and still be revoked, wrong region, wrong deployment name, etc.), so
this makes one real minimal call to each configured provider rather than
just checking environment variables.
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()  # picks up a .env file in the cwd, if present -- see .env.example

from app.providers.anthropic_provider import AnthropicProvider
from app.providers.azure_foundry_provider import AzureAIFoundryProvider
from app.providers.embeddings import OpenAIEmbeddingProvider

PROBE_SYSTEM = 'Respond with ONLY this exact JSON: {"ok": true}'
PROBE_USER = "ping"


def _probe_chat(provider) -> dict:
    if not provider.available():
        return {"provider": provider.name, "configured": False, "live": False, "detail": "no credentials found"}
    result = provider.complete_json(PROBE_SYSTEM, PROBE_USER, max_tokens=20)
    live = bool(result and result.get("ok") is True)
    if live:
        detail = "responded correctly"
    elif provider.last_error:
        detail = f"call failed: {provider.last_error}"
    elif result is not None:
        detail = f"call succeeded but returned unexpected content: {result!r}"
    else:
        detail = "credentials present but call failed (no error was captured -- unexpected)"
    return {"provider": provider.name, "configured": True, "live": live, "detail": detail}


def _probe_embedding(provider) -> dict:
    if not provider.available():
        return {"provider": provider.name, "configured": False, "live": False, "detail": "no credentials found"}
    vecs = provider.embed(["ping"])
    live = bool(vecs and len(vecs) == 1 and len(vecs[0]) > 0)
    if live:
        detail = f"returned a {len(vecs[0])}-dim vector"
    elif provider.last_error:
        detail = f"call failed: {provider.last_error}"
    else:
        detail = "credentials present but call failed (no error was captured -- unexpected)"
    return {"provider": provider.name, "configured": True, "live": live, "detail": detail}


def run_selftest() -> list[dict]:
    results = [
        _probe_chat(AnthropicProvider()),
        _probe_chat(AzureAIFoundryProvider()),
        _probe_embedding(OpenAIEmbeddingProvider()),
    ]
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(run_selftest(), indent=2))