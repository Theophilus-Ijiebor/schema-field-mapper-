"""
SQLite-backed tenant registry.

This is the enterprise-tenancy seam: every mapping run is scoped to a
tenant, every tenant has its own confidence threshold / domain-synonym
overrides / preferred model provider, and every tenant's API keys carry a
role that gates what that key is allowed to do (see app/tenancy/auth.py).

SQLite is used here because this is a single-process deliverable; the
schema below (tenants / tenant_configs / api_keys) maps directly onto a
Postgres table set with no redesign if this needed to run multi-process --
that migration is a connection-string change, not a schema change.
"""

from __future__ import annotations
import hashlib
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

from app.tenancy.models import Tenant, TenantConfig, ApiKeyRecord, Role

#  Defaults to a local-disk tmp path rather than a location under the project
#  directory: this repo may live on a synced/network mount (e.g. a Cowork
#  workspace folder) where SQLite's file-locking + fsync requirements aren't
#  reliably supported. In a real deployment this env var would point at a
#  proper persistent volume (or you'd swap the sqlite3 calls in this module
#  for a Postgres connection -- the schema below maps 1:1 onto Postgres
#  tables with no redesign).
DB_PATH = os.environ.get("TENANCY_DB_PATH", "/tmp/schema_field_mapper/tenants.db")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _cursor() -> Iterator[sqlite3.Cursor]:
    conn = _connect()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'standard',
                created_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tenant_configs (
                tenant_id TEXT PRIMARY KEY REFERENCES tenants(tenant_id),
                confidence_review_threshold REAL NOT NULL DEFAULT 0.85,
                extra_synonyms TEXT NOT NULL DEFAULT '{}',
                model_provider TEXT,
                embedding_provider TEXT,
                rate_limit_per_minute INTEGER NOT NULL DEFAULT 60
            )
        """)
        # Additive migration for DBs created before embedding_provider existed.
        existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(tenant_configs)").fetchall()}
        if "embedding_provider" not in existing_cols:
            cur.execute("ALTER TABLE tenant_configs ADD COLUMN embedding_provider TEXT")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                key_hash TEXT NOT NULL UNIQUE,
                tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                role TEXT NOT NULL,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0
            )
        """)


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Tenants
# ---------------------------------------------------------------------------

def create_tenant(tenant_id: str, name: str, tier: str = "standard") -> Tenant:
    init_db()
    tenant = Tenant(tenant_id=tenant_id, name=name, tier=tier)
    with _cursor() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO tenants (tenant_id, name, tier, created_at, active) VALUES (?, ?, ?, ?, 1)",
            (tenant.tenant_id, tenant.name, tenant.tier, tenant.created_at),
        )
        cur.execute(
            "INSERT OR IGNORE INTO tenant_configs (tenant_id) VALUES (?)",
            (tenant.tenant_id,),
        )
    return tenant


def get_tenant(tenant_id: str) -> Optional[Tenant]:
    init_db()
    with _cursor() as cur:
        row = cur.execute("SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone()
    if not row:
        return None
    return Tenant(tenant_id=row["tenant_id"], name=row["name"], tier=row["tier"],
                  created_at=row["created_at"], active=bool(row["active"]))


def list_tenants() -> list[Tenant]:
    init_db()
    with _cursor() as cur:
        rows = cur.execute("SELECT * FROM tenants").fetchall()
    return [Tenant(tenant_id=r["tenant_id"], name=r["name"], tier=r["tier"],
                    created_at=r["created_at"], active=bool(r["active"])) for r in rows]


def set_tenant_config(tenant_id: str, **kwargs) -> TenantConfig:
    init_db()
    current = get_tenant_config(tenant_id) or TenantConfig(tenant_id=tenant_id)
    updated = current.model_copy(update=kwargs)
    with _cursor() as cur:
        cur.execute("""
            INSERT INTO tenant_configs (tenant_id, confidence_review_threshold, extra_synonyms, model_provider, embedding_provider, rate_limit_per_minute)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(tenant_id) DO UPDATE SET
                confidence_review_threshold = excluded.confidence_review_threshold,
                extra_synonyms = excluded.extra_synonyms,
                model_provider = excluded.model_provider,
                embedding_provider = excluded.embedding_provider,
                rate_limit_per_minute = excluded.rate_limit_per_minute
        """, (tenant_id, updated.confidence_review_threshold, json.dumps(updated.extra_synonyms),
              updated.model_provider, updated.embedding_provider, updated.rate_limit_per_minute))
    return updated


def get_tenant_config(tenant_id: str) -> Optional[TenantConfig]:
    init_db()
    with _cursor() as cur:
        row = cur.execute("SELECT * FROM tenant_configs WHERE tenant_id = ?", (tenant_id,)).fetchone()
    if not row:
        return None
    return TenantConfig(
        tenant_id=row["tenant_id"],
        confidence_review_threshold=row["confidence_review_threshold"],
        extra_synonyms=json.loads(row["extra_synonyms"] or "{}"),
        model_provider=row["model_provider"],
        embedding_provider=row["embedding_provider"] if "embedding_provider" in row.keys() else None,
        rate_limit_per_minute=row["rate_limit_per_minute"],
    )


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

def create_api_key(tenant_id: str, role: Role, label: str) -> str:
    """Returns the RAW key. Only the hash is persisted -- the raw value is
    shown exactly once, matching how every real secrets-issuing system works."""
    init_db()
    raw_key = f"sfm_{tenant_id}_{secrets.token_urlsafe(24)}"
    key_id = secrets.token_hex(8)
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO api_keys (key_id, key_hash, tenant_id, role, label, created_at, revoked) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'), 0)",
            (key_id, _hash_key(raw_key), tenant_id, role.value, label),
        )
    return raw_key


def revoke_api_key(key_id: str) -> None:
    init_db()
    with _cursor() as cur:
        cur.execute("UPDATE api_keys SET revoked = 1 WHERE key_id = ?", (key_id,))


def resolve_api_key(raw_key: str) -> Optional[ApiKeyRecord]:
    init_db()
    key_hash = _hash_key(raw_key)
    with _cursor() as cur:
        row = cur.execute(
            "SELECT * FROM api_keys WHERE key_hash = ? AND revoked = 0", (key_hash,)
        ).fetchone()
    if not row:
        return None
    return ApiKeyRecord(
        key_id=row["key_id"], tenant_id=row["tenant_id"], role=Role(row["role"]),
        label=row["label"], created_at=row["created_at"], revoked=bool(row["revoked"]),
    )


def list_api_keys(tenant_id: str) -> list[ApiKeyRecord]:
    init_db()
    with _cursor() as cur:
        rows = cur.execute("SELECT * FROM api_keys WHERE tenant_id = ?", (tenant_id,)).fetchall()
    return [ApiKeyRecord(key_id=r["key_id"], tenant_id=r["tenant_id"], role=Role(r["role"]),
                          label=r["label"], created_at=r["created_at"], revoked=bool(r["revoked"]))
            for r in rows]


# ---------------------------------------------------------------------------
# Demo seed data
# ---------------------------------------------------------------------------

def seed_demo_data(reset: bool = False) -> dict:
    """
    Creates two example tenants with different configs/providers and one API
    key per role for each, so the API/CLI/README examples all have
    something real to point at. Returns the raw API keys (shown once).

    The two tenants are deliberately pinned to *different* model providers
    (rather than left on "auto") so that a live run with both
    ANTHROPIC_API_KEY and the AZURE_AI_FOUNDRY_* vars set actually exercises
    both backends through identical application code -- the strongest
    available proof that the ModelProvider abstraction is real and not just
    a wrapper around one vendor's SDK. Each pin still degrades gracefully
    (see app/providers/factory.py's fall-through behavior) if that
    particular provider's credentials aren't present, so this is safe to run
    with only one key, or none at all.
    """
    init_db()
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        init_db()

    created = {}

    acme = create_tenant("acme-hr", "Acme Corp HR Migration", tier="enterprise")
    set_tenant_config("acme-hr", confidence_review_threshold=0.85,
                       extra_synonyms={"comp": ["compensation"]}, model_provider="anthropic",
                       embedding_provider=None, rate_limit_per_minute=120)
    created["acme-hr"] = {
        role.value: create_api_key("acme-hr", role, f"acme-{role.value}-key")
        for role in Role
    }

    globex = create_tenant("globex-hr", "Globex People Platform Pilot", tier="standard")
    set_tenant_config("globex-hr", confidence_review_threshold=0.92,  # stricter tenant -> more HITL
                       extra_synonyms={}, model_provider="azure_ai_foundry",
                       embedding_provider=None, rate_limit_per_minute=30)
    created["globex-hr"] = {
        role.value: create_api_key("globex-hr", role, f"globex-{role.value}-key")
        for role in Role
    }

    return created


if __name__ == "__main__":
    keys = seed_demo_data(reset=True)
    print(json.dumps(keys, indent=2))
