"""
Production HTTP API.

    uvicorn app.api.main:app --reload

Multi-tenant: every request authenticates via an `X-API-Key` header, which
resolves to a (tenant_id, role) pair (app/tenancy/auth.py). Every mapping run
is scoped to that tenant (its own confidence threshold, synonym overrides,
preferred model provider) and stored under a tenant-namespaced thread_id so
one tenant can never read or resume another tenant's run.

Endpoints:
    GET  /healthz
    POST /v1/mappings/run                          (role: operator+)
    GET  /v1/mappings/{run_id}                      (role: viewer+)
    POST /v1/mappings/{run_id}/resume                (role: reviewer+)
    GET  /v1/evals/{run_id}                          (role: viewer+)
    POST /v1/admin/tenants                           (role: admin, no tenant scoping -- bootstrap only)
    POST /v1/admin/tenants/{tenant_id}/config         (role: admin)
    POST /v1/admin/tenants/{tenant_id}/api-keys       (role: admin)
"""

from __future__ import annotations
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # picks up a .env file in the cwd, if present -- see .env.example

from fastapi import FastAPI, Header, HTTPException
from langgraph.types import Command

from app.api.schemas import (
    ResumeRequest, RunResponse, TenantCreateRequest,
    TenantConfigUpdateRequest, ApiKeyCreateRequest,
)
from app.tenancy.auth import (
    resolve_auth, require_role, rate_limiter,
    AuthError, PermissionDeniedError, RateLimitExceededError,
)
from app.tenancy.models import Role
from app.tenancy import registry
from app.graph.build_graph import get_compiled_graph
from app.observability.tracing import new_run_id, tenant_trace, log_event
from app.core.schemas import SOURCE_FIELDS
from app.evals.metrics import score_against_golden, score_quality
from app.evals.run_eval import _source_field_lookup, _dest_field_lookup, GOLDEN_PATH
from app.providers.factory import get_provider
import json

app = FastAPI(
    title="Schema Field Mapper API",
    version="1.0",
    description="Multi-tenant, HITL-gated, LangGraph-orchestrated schema field mapping service.",
)


def _authenticate(x_api_key: Optional[str], required_role: Role):
    try:
        ctx = resolve_auth(x_api_key)
        require_role(ctx, required_role)
        rate_limiter.check(ctx.tenant_id)
        return ctx
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except PermissionDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RateLimitExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))


def _thread_id(tenant_id: str, run_id: str) -> str:
    return f"{tenant_id}:{run_id}"


def _state_to_response(tenant_id: str, run_id: str, result: dict, provider_name: Optional[str]) -> RunResponse:
    if "__interrupt__" in result:
        items = result["__interrupt__"][0].value["items"]
        return RunResponse(run_id=run_id, tenant_id=tenant_id, status="pending_review",
                            provider=provider_name, pending_review_items=items)
    return RunResponse(run_id=run_id, tenant_id=tenant_id, status="completed",
                        provider=provider_name, mapping=result.get("mapping_document"),
                        validation=result.get("validation_report"))


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/v1/mappings/run", response_model=RunResponse)
def run_mapping(x_api_key: Optional[str] = Header(None)):
    ctx = _authenticate(x_api_key, Role.OPERATOR)
    run_id = new_run_id()
    thread_id = _thread_id(ctx.tenant_id, run_id)
    graph = get_compiled_graph()

    with tenant_trace(ctx.tenant_id, run_id):
        result = graph.invoke(
            {"tenant_id": ctx.tenant_id, "run_id": run_id},
            config={"configurable": {"thread_id": thread_id}},
        )
    log_event("api.run_mapping", tenant_id=ctx.tenant_id, run_id=run_id, role=ctx.role.value)
    return _state_to_response(ctx.tenant_id, run_id, result, result.get("resolved_provider_name"))


@app.get("/v1/mappings/{run_id}", response_model=RunResponse)
def get_mapping(run_id: str, x_api_key: Optional[str] = Header(None)):
    ctx = _authenticate(x_api_key, Role.VIEWER)
    thread_id = _thread_id(ctx.tenant_id, run_id)
    graph = get_compiled_graph()
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})

    if not snap.values:
        raise HTTPException(status_code=404, detail="Run not found for this tenant.")

    if snap.next:  # still paused on a node (human_review)
        interrupts = snap.tasks[0].interrupts if snap.tasks else ()
        items = interrupts[0].value["items"] if interrupts else []
        return RunResponse(run_id=run_id, tenant_id=ctx.tenant_id, status="pending_review",
                            provider=snap.values.get("resolved_provider_name"), pending_review_items=items)

    return RunResponse(run_id=run_id, tenant_id=ctx.tenant_id, status="completed",
                        provider=snap.values.get("resolved_provider_name"),
                        mapping=snap.values.get("mapping_document"),
                        validation=snap.values.get("validation_report"))


@app.post("/v1/mappings/{run_id}/resume", response_model=RunResponse)
def resume_mapping(run_id: str, body: ResumeRequest, x_api_key: Optional[str] = Header(None)):
    ctx = _authenticate(x_api_key, Role.REVIEWER)
    thread_id = _thread_id(ctx.tenant_id, run_id)
    graph = get_compiled_graph()

    decisions = [{**d.model_dump(), "reviewer": ctx.key_id} for d in body.decisions]
    with tenant_trace(ctx.tenant_id, run_id):
        result = graph.invoke(Command(resume=decisions), config={"configurable": {"thread_id": thread_id}})
    log_event("api.resume_mapping", tenant_id=ctx.tenant_id, run_id=run_id, decisions=len(decisions))
    return _state_to_response(ctx.tenant_id, run_id, result, result.get("resolved_provider_name"))


@app.get("/v1/evals/{run_id}")
def eval_mapping(run_id: str, x_api_key: Optional[str] = Header(None)):
    ctx = _authenticate(x_api_key, Role.VIEWER)
    thread_id = _thread_id(ctx.tenant_id, run_id)
    graph = get_compiled_graph()
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not snap.values or snap.next:
        raise HTTPException(status_code=409, detail="Run is not complete yet, or does not exist for this tenant.")

    candidate = snap.values["mapping_document"]
    with open(GOLDEN_PATH) as fh:
        golden = json.load(fh)

    regression = score_against_golden(candidate, golden)
    quality = score_quality(get_provider(), candidate, _source_field_lookup, _dest_field_lookup)
    return {
        "run_id": run_id,
        "accuracy": regression.overall_accuracy,
        "precision": regression.overall_precision,
        "recall": regression.overall_recall,
        "f1": regression.overall_f1,
        "type_transform_agreement": regression.type_transform_agreement,
        "relevance": quality.mean_relevance,
        "faithfulness": quality.mean_faithfulness,
        "mismatches": regression.mismatches,
    }


# --- Admin / tenant management -------------------------------------------

@app.post("/v1/admin/tenants")
def create_tenant(body: TenantCreateRequest, x_api_key: Optional[str] = Header(None)):
    # Bootstrap note: creating the *first* tenant has no tenant to scope an
    # admin key to yet. In this deliverable that bootstrap step is done via
    # the CLI (`python3 -m app.tenancy.registry`), which seeds demo tenants
    # directly against the DB. This endpoint is for an already-provisioned
    # platform-admin key (out of scope to fully implement here) to onboard
    # *additional* tenants; documented in README's "what I'd add next".
    raise HTTPException(status_code=501, detail=(
        "Platform-admin-scoped tenant creation is not wired up in this deliverable -- "
        "seed tenants via `python3 -m app.tenancy.registry` (see README)."
    ))


@app.post("/v1/admin/tenants/{tenant_id}/config")
def update_tenant_config(tenant_id: str, body: TenantConfigUpdateRequest, x_api_key: Optional[str] = Header(None)):
    ctx = _authenticate(x_api_key, Role.ADMIN)
    if ctx.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Admins may only manage their own tenant's config.")
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    cfg = registry.set_tenant_config(tenant_id, **updates)
    return cfg.model_dump()


@app.post("/v1/admin/tenants/{tenant_id}/api-keys")
def create_api_key(tenant_id: str, body: ApiKeyCreateRequest, x_api_key: Optional[str] = Header(None)):
    ctx = _authenticate(x_api_key, Role.ADMIN)
    if ctx.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Admins may only issue keys for their own tenant.")
    try:
        role = Role(body.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown role '{body.role}'.")
    raw_key = registry.create_api_key(tenant_id, role, body.label)
    return {"api_key": raw_key, "note": "This key is shown once. Store it securely."}
