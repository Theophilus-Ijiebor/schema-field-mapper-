---
name: schema-field-mapper
description: Use this skill whenever the user needs to map fields between two database schemas (e.g. a legacy relational/MySQL schema and a modern document/MongoDB schema), generate a field-level migration mapping document, or evaluate/operate the Schema Field Mapper service (multi-tenant LangGraph pipeline with HITL review, Anthropic/Azure AI Foundry model providers, and a relevance/faithfulness/accuracy eval harness). Trigger on requests like "map these two schemas," "generate a migration mapping for X," "run the schema mapper for tenant Y," "show me which fields need human review," or "score this mapping against the golden set." Do NOT use this for generic schema design, unrelated ETL work, or database administration tasks that don't involve producing a field-to-field mapping document.
---

# Schema Field Mapper

A production-grade pipeline that maps every field in a source schema to its
semantically equivalent field in a destination schema, and explains why.
Built as a LangGraph state machine (not a single-prompt LLM call) so no
individual reasoning step ever sees both full schemas at once -- retrieval
narrows candidates first, then a bounded LLM (or deterministic offline
fallback) call picks the best match for one field at a time.

## When to use this skill

- The user has two schemas (or one schema + a description of a target
  system) and wants a field-by-field mapping with confidence scores and
  human-readable reasoning.
- The user wants to operate the service itself: run a mapping for a tenant,
  resolve a pending human-review queue, check eval metrics, or manage
  tenants/API keys.
- The user asks about the architecture (LangGraph graph shape, HITL
  interrupts, provider abstraction, tenancy/RBAC model) for this specific
  deliverable.

## How the pipeline is organized

```
app/
  core/        schema definitions, normalization, Pydantic domain models, JSON-schema validation
  retrieval/   TF-IDF candidate retrieval (table-level + field-level), tenant-aware synonym injection
  rules/       deterministic type-transform + denormalization/join rules
  providers/   pluggable ModelProvider: Anthropic, Azure AI Foundry, offline deterministic fallback
  graph/       LangGraph StateGraph: load_config -> match_tables -> match_fields -> [human_review] -> apply_rules -> assemble -> validate
  tenancy/     SQLite-backed tenant registry, RBAC (admin/operator/reviewer/viewer), rate limiting
  observability/ structured event logging + LangSmith tracing (tenant-scoped projects)
  evals/       golden-set regression scoring (accuracy/precision/recall/F1/type-transform agreement)
               + LLM-judge relevance & faithfulness metrics with heuristic fallback
  api/         FastAPI multi-tenant HTTP surface (run / resume / get / eval / tenant admin)
  cli/         terminal entry point with an interactive HITL resolution loop
```

## Running it

**CLI (simplest, drives the full run + HITL loop in one process):**
```bash
cd schema_field_mapper
pip install -r requirements.txt
python3 -m app.tenancy.registry            # seed demo tenants + print API keys (first run only)
python3 -m app.cli.main --tenant acme-hr   # interactive: prompts for each low-confidence field
```

**HTTP API:**
```bash
uvicorn app.api.main:app --reload
curl -X POST localhost:8000/v1/mappings/run -H "X-API-Key: <operator key>"
# -> {"status": "pending_review", "run_id": "...", "pending_review_items": [...]}
curl -X POST localhost:8000/v1/mappings/<run_id>/resume -H "X-API-Key: <reviewer key>" \
     -d '{"decisions": [{"source_table": "...", "source_field": "...", "approved": true}]}'
```

**Eval harness:**
```bash
python3 -m app.evals.run_eval                       # fresh run, auto-approves reviews, scores vs golden
python3 -m app.evals.run_eval --input output/x.json  # score an existing mapping document
```

## Interpreting results

- `confidence < tenant's confidence_review_threshold` (default 0.85) means
  the field was routed to human review, not auto-accepted -- check
  `pending_review_items` / the CLI prompt rather than assuming the run
  completed.
- `unmapped_source_fields` on a table means the pipeline found no
  destination field it was confident matches -- this is a deliberate,
  audited outcome (e.g. a source `dob` field with no destination
  counterpart), not a bug to silently paper over.
- `unmapped_destination_fields` with a `reason` mentioning a join usually
  means that destination field is populated by a *different* source table
  during migration (denormalization), not a gap in the mapping.
- Eval `relevance` and `faithfulness` scores below ~0.6 on a field are worth
  a manual look -- they flag either a weak semantic match or a reasoning
  sentence that doesn't actually ground itself in the real field names/types.

## Extending to a new schema pair

Replace the contents of `app/core/schemas.py` (`SOURCE_FIELDS`, `DEST_FIELDS`,
table/collection lists) with the new schema pair -- every other module
(retrieval, rules, graph, API) is schema-shape-agnostic and needs no changes.
Re-run `python3 -m app.evals.build_golden` once you've manually verified a
run's output, to refresh the regression baseline for the new schema pair.
