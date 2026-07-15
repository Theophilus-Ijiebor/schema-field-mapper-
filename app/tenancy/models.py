"""Tenant / RBAC domain models."""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    ADMIN = "admin"        # manage tenants, API keys, tenant config
    OPERATOR = "operator"  # trigger mapping runs
    REVIEWER = "reviewer"  # approve/override HITL review items
    VIEWER = "viewer"      # read-only access to results/evals


# Roles that satisfy each permission. ADMIN implicitly satisfies everything.
ROLE_HIERARCHY = {
    Role.VIEWER: {Role.VIEWER, Role.OPERATOR, Role.REVIEWER, Role.ADMIN},
    Role.OPERATOR: {Role.OPERATOR, Role.ADMIN},
    Role.REVIEWER: {Role.REVIEWER, Role.ADMIN},
    Role.ADMIN: {Role.ADMIN},
}


class Tenant(BaseModel):
    tenant_id: str
    name: str
    tier: str = "standard"  # "standard" | "enterprise"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    active: bool = True


class TenantConfig(BaseModel):
    tenant_id: str
    confidence_review_threshold: float = 0.85
    extra_synonyms: dict[str, list[str]] = Field(default_factory=dict)
    model_provider: Optional[str] = None  # "anthropic" | "azure_ai_foundry" | "offline" | None (auto)
    embedding_provider: Optional[str] = None  # "openai_embeddings" | "tfidf" | None (auto: use embeddings if configured, else tfidf)
    rate_limit_per_minute: int = 60


class ApiKeyRecord(BaseModel):
    key_id: str
    tenant_id: str
    role: Role
    label: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    revoked: bool = False
