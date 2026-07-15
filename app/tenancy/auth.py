"""
Authentication + RBAC + rate limiting.

`resolve_auth` turns a raw `X-API-Key` header value into an AuthContext
(tenant_id + role); `require_role` enforces the permission a given endpoint
needs; `RateLimiter` is a per-tenant token bucket. All three are used by
app/api/main.py as FastAPI dependencies, and by app/cli/main.py for the
equivalent terminal-driven checks.

The rate limiter is in-process/in-memory, which is the right call for a
single-process deliverable; a real multi-instance deployment would back this
with Redis (INCR + TTL) without changing the call sites below.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional

from app.tenancy.models import Role, ROLE_HIERARCHY
from app.tenancy.registry import resolve_api_key, get_tenant, get_tenant_config


class AuthError(Exception):
    pass


class PermissionDeniedError(Exception):
    pass


class RateLimitExceededError(Exception):
    pass


@dataclass
class AuthContext:
    tenant_id: str
    role: Role
    key_id: str


def resolve_auth(raw_api_key: Optional[str]) -> AuthContext:
    if not raw_api_key:
        raise AuthError("Missing API key.")
    record = resolve_api_key(raw_api_key)
    if record is None:
        raise AuthError("Invalid or revoked API key.")
    tenant = get_tenant(record.tenant_id)
    if tenant is None or not tenant.active:
        raise AuthError("Tenant is not active.")
    return AuthContext(tenant_id=record.tenant_id, role=record.role, key_id=record.key_id)


def require_role(ctx: AuthContext, allowed: Role) -> None:
    if ctx.role not in ROLE_HIERARCHY.get(allowed, set()):
        raise PermissionDeniedError(
            f"Role '{ctx.role.value}' is not permitted to perform an action requiring '{allowed.value}'."
        )


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    """Simple per-tenant token-bucket rate limiter, refilled continuously."""

    def __init__(self):
        self._buckets: dict[str, _Bucket] = {}

    def check(self, tenant_id: str, limit_per_minute: Optional[int] = None) -> None:
        limit = limit_per_minute
        if limit is None:
            cfg = get_tenant_config(tenant_id)
            limit = cfg.rate_limit_per_minute if cfg else 60

        now = time.monotonic()
        bucket = self._buckets.get(tenant_id)
        if bucket is None:
            bucket = _Bucket(tokens=float(limit), last_refill=now)
            self._buckets[tenant_id] = bucket

        elapsed = now - bucket.last_refill
        refill_rate = limit / 60.0  # tokens per second
        bucket.tokens = min(float(limit), bucket.tokens + elapsed * refill_rate)
        bucket.last_refill = now

        if bucket.tokens < 1.0:
            raise RateLimitExceededError(
                f"Rate limit exceeded for tenant '{tenant_id}' ({limit}/min)."
            )
        bucket.tokens -= 1.0


rate_limiter = RateLimiter()
