"""
Observability: structured event logging + LangSmith tracing.

Two independent layers:

  1. `log_event` -- always-on structured JSON-lines logging to stderr. This
     is the audit trail every run produces regardless of what's configured;
     it's what you'd ship to your log aggregator (Datadog, CloudWatch, etc.)
     in production.

  2. LangSmith tracing -- when LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) is
     set, `tenant_trace()` wraps a graph run in a LangSmith trace with a
     tenant-scoped project name (`<LANGSMITH_PROJECT>-<tenant_id>`), so each
     tenant's runs are visually and access-isolated in the LangSmith UI. When
     no key is set, `tenant_trace()` is a no-op context manager -- the graph
     runs identically either way, tracing is purely additive.

Neither layer is required for the graph to function; both degrade to no-ops
without credentials, matching the same philosophy as the model providers.
"""

from __future__ import annotations
import contextlib
import json
import os
import sys
import time
import uuid
from typing import Iterator, Optional


def log_event(event: str, **fields) -> None:
    record = {"ts": round(time.time(), 3), "event": event, **fields}
    print(json.dumps(record, default=str), file=sys.stderr)


def langsmith_enabled() -> bool:
    return bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))


@contextlib.contextmanager
def tenant_trace(tenant_id: Optional[str], run_id: Optional[str] = None) -> Iterator[None]:
    """
    Scope a graph invocation to a per-tenant LangSmith project. No-ops
    cleanly if LangSmith isn't configured.

    Project name is `<LANGSMITH_PROJECT>-<tenant_id>`, where LANGSMITH_PROJECT
    defaults to "schema-field-mapper" if unset. LangSmith creates a project
    automatically on its first received trace -- you don't need to
    pre-create it in the UI, and if you do, it needs to be named to match
    (or just leave LANGSMITH_PROJECT unset and let the default apply).
    """
    if not langsmith_enabled():
        yield
        return

    base_project = os.environ.get("LANGSMITH_PROJECT", "schema-field-mapper")
    project = f"{base_project}-{tenant_id or 'default'}"
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    try:
        from langchain_core.tracers.context import tracing_v2_enabled
    except Exception as exc:
        # Only LangSmith *setup* failures (e.g. the package isn't installed)
        # are swallowed here -- this must never guard the actual `yield`,
        # otherwise an exception raised by the wrapped pipeline code gets
        # thrown into this same except block, which would then try to
        # `yield` a second time and crash the generator protocol itself
        # (RuntimeError: generator didn't stop after throw()), masking the
        # real error.
        log_event("langsmith.trace_error", error=f"could not import LangSmith tracing: {exc}")
        yield
        return

    with tracing_v2_enabled(project_name=project):
        log_event("langsmith.trace_start", tenant_id=tenant_id, run_id=run_id, project=project)
        try:
            yield
        except Exception:
            log_event("langsmith.trace_end", tenant_id=tenant_id, run_id=run_id, project=project, ok=False)
            raise
        else:
            log_event("langsmith.trace_end", tenant_id=tenant_id, run_id=run_id, project=project, ok=True)


def log_feedback(run_key: str, metric: str, score: float, comment: Optional[str] = None) -> None:
    """
    Records an eval score (accuracy / relevance / faithfulness / ...) to the
    local structured log, keyed by this pipeline's own run identifier.

    This used to also attempt a LangSmith feedback push on every call, but
    it was calling client.create_feedback(run_id=None, ...) -- `run_key`
    here is this pipeline's own locally-generated id (see new_run_id()), not
    a LangSmith trace/run UUID, so there was never a valid id to send and
    every call logged a guaranteed "One of run_id, trace_id, or project_id
    must be provided" error. Rather than keep making a network call that can
    only fail, this is intentionally local-only now. Eval scores remain
    fully visible in the markdown report (app/evals/run_eval.py) and in this
    structured log regardless of whether LangSmith is configured -- neither
    depends on the LangSmith push actually succeeding.
    """
    log_event("eval.metric", run_key=run_key, metric=metric, score=score, comment=comment)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]