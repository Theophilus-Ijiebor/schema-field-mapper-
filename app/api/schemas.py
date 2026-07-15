"""HTTP request/response models for the FastAPI surface."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class ReviewDecisionIn(BaseModel):
    source_table: str
    source_field: str
    approved: bool
    override_destination_field: Optional[str] = None
    comment: Optional[str] = None


class ResumeRequest(BaseModel):
    decisions: list[ReviewDecisionIn]


class RunResponse(BaseModel):
    run_id: str
    tenant_id: str
    status: str  # "pending_review" | "completed"
    provider: Optional[str] = None
    pending_review_items: Optional[list[dict]] = None
    mapping: Optional[dict] = None
    validation: Optional[dict] = None


class TenantCreateRequest(BaseModel):
    tenant_id: str
    name: str
    tier: str = "standard"


class TenantConfigUpdateRequest(BaseModel):
    confidence_review_threshold: Optional[float] = None
    extra_synonyms: Optional[dict[str, list[str]]] = None
    model_provider: Optional[str] = None
    rate_limit_per_minute: Optional[int] = None


class ApiKeyCreateRequest(BaseModel):
    role: str
    label: str
