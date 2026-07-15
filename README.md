# Schema Field Mapper -- Production Architecture

A multi-tenant, LangGraph-orchestrated pipeline that maps every field in a
legacy MySQL HR schema to its semantically equivalent field in a modern
MongoDB people-platform schema, with human-in-the-loop review, pluggable
model deployment (Anthropic / Azure AI Foundry), full observability, and an
eval harness scoring accuracy, relevance, and faithfulness.

This started as a take-home MVP (a flat script satisfying the assignment's
literal deliverables) and was rebuilt into the system described below to
demonstrate how it would actually be run as a service. The original MVP
files (`schemas.py`, `pipeline.py`, `main.py`, etc.) are still present at the
repo root for reference; everything described from here on lives under
`app/`.

## TL;DR -- run it

```bash
cd schema_field_mapper
pip install -r requirements.txt
python3 -m app.tenancy.registry            # seeds 2 demo tenants, prints API keys (save them)
python3 -m app.evals.build_golden          # builds the regression fixture (already committed, re-run only if logic changes)

python3 -m app.providers.selftest          # which chat/embedding backends are actually reachable right now
python3 -m app.cli.main --tenant acme-hr   # interactive run with real HITL prompts
# or:
uvicorn app.api.main:app --reload          # HTTP surface
python3 -m app.evals.run_eval              # eval report (accuracy/precision/recall/F1/relevance/faithfulness)
```

Have live Anthropic and/or Azure AI Foundry credentials? See
**`LIVE_RUN_GUIDE.md`** for the exact steps to run both demo tenants against
real models instead of the offline fallback, including how to tell from the
output that it actually used them.

## Why this isn't a single prompt

The assignment's constraint -- *you cannot pass both schemas to an LLM in a
single prompt and receive a finished mapping* -- is really a constraint on
architecture, not phrasing. A single mega-prompt produces an
un-auditable guess: there's no way to tell whether `f_name -> fullName.firstName`
reflects understanding or just prompt proximity. So the pipeline is a
retrieval-then-reason graph instead:

1. **Retrieve** -- every field on both sides is embedded independently into a
   shared TF-IDF vector space (`app/retrieval`), giving each source table a
   ranked shortlist of destination collections, and each source field a
   top-3 shortlist of destination fields *within* the matched collection.
2. **Reason** -- one bounded LLM call per source table (confirm the
   collection) and one per source field (pick from its top-3 shortlist,
   explain why). **Each call sees exactly one item plus a handful of
   candidates -- never the other side's full schema.**
3. **Transform** -- `type_transform` and value-mapping `notes` come from a
   deterministic rules layer (`app/rules`), not LLM invention.
4. **Review** -- anything below the tenant's confidence threshold is routed
   to a human via a genuine LangGraph `interrupt()`, not silently accepted.
5. **Assemble & validate** -- Pydantic models + JSON Schema + a coverage
   check (every source field mapped or explicitly unmapped; no destination
   field claimed twice) before anything is returned.

## Directory structure

```
schema_field_mapper/
├── SKILL.md                     Claude Skill manifest for this pipeline
├── .env.example                 every config var, documented
├── requirements.txt
├── README.md                    this file
│
├── (schemas.py, pipeline.py, main.py, ...)   <- original flat MVP, kept for reference
│
└── app/
    ├── core/
    │   ├── schemas.py           the two schemas, as data (swap this to point at a new schema pair)
    │   ├── normalize.py         tokenization, abbreviation/synonym expansion, content-word gate
    │   ├── models.py            Pydantic: FieldMapping, TableMapping, MappingDocument, ReviewItem/Decision
    │   └── validation.py        JSON Schema + field-coverage validation
    │
    ├── retrieval/
    │   └── retriever.py         TF-IDF candidate retrieval, table + field level, tenant-synonym aware
    │
    ├── rules/
    │   └── type_transforms.py   deterministic type/value-transform rules, denormalization/join rules
    │
    ├── providers/                pluggable LLM + embedding backends -- see "Model deployment" below
    │   ├── base.py                ModelProvider interface + with_retry backoff helper
    │   ├── anthropic_provider.py
    │   ├── azure_foundry_provider.py
    │   ├── offline_provider.py
    │   ├── embeddings.py          EmbeddingProvider: OpenAI / Azure AI Foundry embeddings
    │   ├── factory.py
    │   └── selftest.py            live reachability check for every configured backend
    │
    ├── graph/                    LangGraph orchestration -- see "The graph" below
    │   ├── state.py
    │   ├── reasoning.py          bounded per-table / per-field prompts
    │   ├── nodes.py
    │   └── build_graph.py
    │
    ├── tenancy/                  enterprise multi-tenancy -- see "Tenancy & RBAC" below
    │   ├── models.py
    │   ├── registry.py           SQLite tenant/config/API-key store
    │   └── auth.py                auth resolution, RBAC, rate limiting
    │
    ├── observability/
    │   └── tracing.py            structured JSON-lines logging + LangSmith tracing
    │
    ├── evals/                    see "Eval harness" below
    │   ├── golden/golden_mapping.json
    │   ├── build_golden.py
    │   ├── metrics.py             regression scoring + heuristic quality-metric fallback
    │   ├── deepeval_adapter.py    DeepEvalBaseLLM wrapper around our ModelProvider
    │   ├── deepeval_metrics.py    GEval relevance/faithfulness metrics
    │   └── run_eval.py
    │
    ├── api/
    │   ├── main.py               FastAPI multi-tenant HTTP surface
    │   └── schemas.py
    │
    └── cli/
        └── main.py               terminal entry point with interactive HITL loop
```

## The graph

```
START -> load_config -> match_tables -> match_fields --+
                                                          |
                          pending_reviews non-empty? -----+
                               yes -> human_review --+
                               no  ----------------- apply_rules -> assemble -> validate -> END
```

- **`load_config`** resolves the tenant's confidence threshold, domain-synonym
  overrides, and preferred model provider from `app/tenancy`.
- **`match_tables`** / **`match_fields`** run retrieval + bounded reasoning
  as described above. Every field decision below the tenant's
  `confidence_review_threshold` (default 0.85) is *not* auto-accepted -- it's
  collected into `pending_reviews`.
- **`human_review`** is a genuine suspend point: it calls LangGraph's
  `interrupt()`, which pauses the run (state durably checkpointed via
  `SqliteSaver`) and returns the pending items to whoever is driving the run
  -- the CLI prompts for them interactively; the API returns
  `{"status": "pending_review", ...}` from `POST /v1/mappings/run` and waits
  for `POST /v1/mappings/{run_id}/resume`. This is real suspend-and-resume,
  not a logging statement -- a process can restart between pause and resume
  and the run picks up exactly where it left off, because the checkpoint is
  on disk, not in memory.
- **`apply_rules`** fills in `unmapped_destination_fields` (including the
  `department.name` / `location.city` style fields that only exist in the
  destination via a join, annotated with why).
- **`assemble`** builds a `MappingDocument` (Pydantic) and serializes it to
  exactly the JSON shape the assignment specifies.
- **`validate`** runs JSON Schema validation plus a coverage check and
  attaches a `validation_report`.

Demonstrated live in this session: a fresh run against the two schemas
routes 4 fields (`mgr_emp_id`, `created_ts`, `updated_ts`, `dept_stat`) to
human review under the default 0.85 threshold; resuming with a mix of
approve / reject / override decisions produces a fully validated document
(see `app/evals/golden/golden_mapping.json`, built exactly this way).

## Model deployment (Anthropic / Azure AI Foundry / offline)

`app/providers` defines one interface, `ModelProvider.complete_json(system,
user) -> dict | None`, with three implementations:

| Provider | Backing | Config |
|---|---|---|
| `anthropic_provider.py` | Claude via the `anthropic` SDK | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| `azure_foundry_provider.py` | A model deployed in Azure AI Foundry, called via the OpenAI-compatible chat-completions API (`openai.AzureOpenAI`) | `AZURE_AI_FOUNDRY_ENDPOINT`, `AZURE_AI_FOUNDRY_API_KEY`, `AZURE_AI_FOUNDRY_DEPLOYMENT`, `AZURE_AI_FOUNDRY_API_VERSION` |
| `offline_provider.py` | Deterministic fallback (`complete_json` always returns `None`) | none -- always available |

`app/providers/factory.py` resolves in order: an explicit tenant preference
(`TenantConfig.model_provider`) -> Anthropic if configured -> Azure AI
Foundry if configured -> offline. Every graph node and eval judge goes
through this factory, never a vendor SDK directly, so **swapping deployment
target is a config change, not a code change**. `TenantConfig.model_provider`
lets one tenant pin itself to Azure AI Foundry (e.g. for data-residency
requirements) while another uses Anthropic or runs fully offline -- see the
`globex-hr` demo tenant, which is pinned to `offline` on purpose.

**To actually deploy against Azure AI Foundry**: create a Foundry project,
deploy a chat-capable model under "Deployments," and set the four
`AZURE_AI_FOUNDRY_*` env vars (or a tenant's config) to the values shown in
the Foundry portal. No other change is needed -- `azure_foundry_provider.py`
is complete, runnable code; it just has no credentials to call against in
this environment, so the artifacts in this deliverable were generated via
the offline fallback (see "What's real vs. stubbed" below).

## Retrieval backend: embeddings or TF-IDF

`app/retrieval/retriever.py` supports two interchangeable backends behind
one interface (`top_candidates` / `best_tables`), auto-selected by
`app/providers/embeddings.py`:

- **Dense embeddings** (OpenAI `text-embedding-3-small`, or an Azure AI
  Foundry embedding deployment -- Foundry is preferred if both are
  configured, so a Foundry-pinned tenant stays fully inside Foundry for both
  chat and embeddings) understand semantic relationships directly -- "hire
  date" and "start date" land close together in vector space without any
  hand-written synonym. Source and destination field texts are still
  embedded in two *separate* API calls, never combined into one request,
  for the same reason no reasoning prompt ever sees both schemas.
- **TF-IDF** (local, no network) is the fallback whenever no embedding
  provider is configured, and is what actually produced this deliverable's
  artifacts -- see "What's real vs. stubbed."

Selection is per-tenant (`TenantConfig.embedding_provider`), resolved once
in `load_config_node` and threaded through `match_tables_node` /
`match_fields_node`. Run `python3 -m app.providers.selftest` to see, right
now, which chat and embedding backends are actually reachable (not just
configured -- it makes one real minimal call to each).

## Reliability: retries and graceful degradation

Every provider call (`AnthropicProvider`, `AzureAIFoundryProvider`,
`OpenAIEmbeddingProvider`) goes through `app/providers/base.py`'s
`with_retry` -- up to 3 attempts with exponential backoff before giving up,
so a transient rate limit or network blip doesn't immediately demote a run
to a lower-quality fallback path. If a provider is still unreachable after
retries, every layer that depends on it (retrieval, reasoning, eval judges)
has an explicit, tested fallback rather than raising -- offline TF-IDF for
retrieval, the deterministic reasoning fallback in `app/graph/reasoning.py`,
and the heuristic quality metrics in `app/evals/metrics.py`. Nothing in this
pipeline hard-crashes because an external API had a bad moment.

## Eval quality metrics: DeepEval GEval, not a hand-rolled judge prompt

The relevance and faithfulness metrics are implemented on top of
[DeepEval](https://github.com/confident-ai/deepeval)'s `GEval` -- a
maintained library implementing the standard rubric-driven LLM-as-judge
pattern properly: it first asks the judge model to derive explicit
evaluation steps from a criteria description, then scores the test case
against those steps and returns both a 0-1 score and a written rationale,
rather than a single ad-hoc "give me a score" prompt.

`app/evals/deepeval_adapter.py`'s `ProviderBackedDeepEvalModel` implements
DeepEval's `DeepEvalBaseLLM` interface as a thin relay onto this codebase's
own `ModelProvider` (`complete_text`) -- so DeepEval's judge runs on
whichever backend a tenant is actually configured for (Anthropic or Azure AI
Foundry), instead of being hard-wired to one vendor's SDK the way DeepEval's
built-in model wrappers are. `app/evals/deepeval_metrics.py` builds one
`GEval` metric each for relevance and faithfulness, with criteria text
matching this project's own definitions, and scores one `LLMTestCase` per
field mapping (source field metadata as `input`, the pipeline's chosen
destination field + reasoning as `actual_output`, both fields' real metadata
as `context` for faithfulness to be checked against).

`run_eval.py` uses DeepEval automatically whenever a live chat provider is
configured (`build_deepeval_model` returns `None` for the offline provider,
which is the signal to fall back to `app/evals/metrics.py`'s heuristic
judges instead) -- the eval report's header states which engine actually ran.

## Tenancy & RBAC

`app/tenancy/registry.py` is a SQLite-backed store (`tenants`,
`tenant_configs`, `api_keys` tables -- deliberately Postgres-shaped, so
swapping the connection is a config change, not a schema redesign).

| Role | Can do |
|---|---|
| `viewer` | Read run status, results, eval reports |
| `operator` | Everything `viewer` can, plus start new mapping runs |
| `reviewer` | Everything `viewer` can, plus resolve HITL review items |
| `admin` | Everything above, plus manage that tenant's config and API keys |

Every API key hashes to a `(tenant_id, role)` pair (`app/tenancy/auth.py`);
`RunResponse`-producing endpoints enforce the minimum role via
`require_role`, and a per-tenant token-bucket `RateLimiter` throttles request
volume using each tenant's configured `rate_limit_per_minute`. Two demo
tenants are seeded by `python3 -m app.tenancy.registry`:

- **`acme-hr`** -- enterprise tier, default 0.85 review threshold, pinned to
  the **Anthropic** provider, 120 req/min.
- **`globex-hr`** -- standard tier, stricter 0.92 review threshold (more
  fields get routed to human review), pinned to the **Azure AI Foundry**
  provider, 30 req/min.

The two tenants are pinned to *different* providers on purpose: running the
same schema pair through both, via identical application code, is the
actual demonstration that the `ModelProvider` abstraction works, not just
that one SDK call succeeds. Both pins degrade gracefully to the offline
fallback if that tenant's provider isn't configured (see
`app/providers/factory.py`) -- see **`LIVE_RUN_GUIDE.md`** for running both
live with real credentials.

Every mapping run's LangGraph thread is namespaced `{tenant_id}:{run_id}`, so
one tenant's API key can never read or resume another tenant's run (verified
in this session: a `globex-hr` key against an `acme-hr` run_id returns 404,
not another tenant's data).

## Observability

`app/observability/tracing.py` provides two independent layers:

- **Structured logging** -- every node emits a JSON-lines event to stderr
  (`load_config`, `match_tables`, `match_fields`, `human_review.paused`,
  `human_review.resumed`, `validate`, ...). Always on, ship straight to any
  log aggregator.
- **LangSmith tracing** -- when `LANGSMITH_API_KEY` is set, `tenant_trace()`
  wraps a graph invocation in a per-tenant LangSmith project
  (`schema-field-mapper-<tenant_id>`), so each tenant's runs are visually and
  access-isolated in the LangSmith UI. `log_feedback()` also pushes eval
  scores (accuracy, relevance, faithfulness, ...) as LangSmith feedback,
  keyed per run, so eval history sits next to trace history. Both no-op
  cleanly without a key -- the graph behaves identically either way.

## Eval harness

`app/evals/run_eval.py` produces two families of metrics:

**Regression metrics** (candidate vs. `evals/golden/golden_mapping.json`,
a fixture built once via `build_golden.py` from a manually-audited run):
- `accuracy` -- fraction of golden's mapped fields the candidate maps to the
  same destination field.
- `precision` / `recall` / `F1` -- treats "did the candidate map this field
  at all" as a classification against golden's mapped/unmapped split.
- `type_transform_agreement` -- fraction of correctly-mapped fields whose
  `type_transform` string also matches golden exactly.

**Quality metrics** (candidate alone, no golden needed -- the metrics named
explicitly in the assignment):
- `relevance` -- LLM-as-judge (via the same `ModelProvider` abstraction,
  one field at a time) scoring how relevant the chosen destination field is
  to the source field, 0-1. Falls back to a business-token-overlap heuristic
  offline.
- `faithfulness` -- LLM-as-judge scoring whether the `reasoning` sentence's
  claims are actually grounded in the real field metadata (catches generic
  template reasoning that doesn't reference the actual fields). Falls back
  to checking whether the reasoning text literally names both fields.

Sample report (this run, offline fallback, `acme-hr`'s output scored against
golden -- full report at `output/eval_report.md`):

```
Overall accuracy: 1.0   Precision: 1.0  Recall: 1.0  F1: 1.0
Type-transform agreement: 1.0
Mean relevance: 0.703    Mean faithfulness: 1.0
```

Accuracy/precision/recall are 1.0 here because this run and the golden
fixture were both produced by the same (offline) pipeline -- that's the
intended regression-test behavior: change the retrieval/reasoning/rules
logic later, and any resulting mismatch shows up here immediately, itemized
under "Mismatches." Relevance and faithfulness are *not* circular against
golden -- they're independent judgments of the candidate alone, which is why
relevance in particular is more textured (0.2 for the `_id` structural-rule
mappings, since the judge is scoring lexical/semantic relevance and doesn't
know those are intentionally rule-routed rather than retrieval-matched).

## API reference

| Method & path | Role | Purpose |
|---|---|---|
| `GET /healthz` | none | liveness |
| `POST /v1/mappings/run` | operator+ | start a new run; returns `pending_review` or `completed` |
| `GET /v1/mappings/{run_id}` | viewer+ | poll a run's status/result |
| `POST /v1/mappings/{run_id}/resume` | reviewer+ | submit HITL decisions, continue the run |
| `GET /v1/evals/{run_id}` | viewer+ | score a completed run against the golden set |
| `POST /v1/admin/tenants/{tenant_id}/config` | admin | update that tenant's thresholds/synonyms/provider |
| `POST /v1/admin/tenants/{tenant_id}/api-keys` | admin | issue a new API key for that tenant |

All exercised end-to-end in this session via `fastapi.testclient.TestClient`:
health check, an operator starting a run, hitting the review gate, a
reviewer resuming it, a viewer polling the result and pulling the eval
report, and negative cases (wrong role -> 403, invalid key -> 401).

## What's real vs. what's stubbed

Everything in this repo is real, runnable code -- there is no mocked
business logic. Two things are worth being explicit about, because "above
and beyond" only means something if it's honestly scoped:

- **Azure AI Foundry**: the provider implementation is complete and calls a
  real OpenAI-compatible endpoint via the standard `openai.AzureOpenAI`
  client -- but this environment has no Foundry deployment or credentials,
  so it has not been exercised against a live endpoint. The graceful
  `available() -> False` fallback means the rest of the system doesn't
  notice either way.
- **LangSmith**: tracing/feedback code is complete and follows the standard
  `tracing_v2_enabled` / `Client.create_feedback` integration points, but was
  not exercised against a real LangSmith project for the same reason (no API
  key in this environment).
- **Embeddings + DeepEval**: both integrations were verified mechanically
  with a controlled fake `ModelProvider` standing in for a real API (returns
  canned but correctly-shaped responses) -- this confirmed the actual
  DeepEval `GEval` machinery runs end-to-end through `ProviderBackedDeepEvalModel`
  (evaluation-step generation, scoring, `LLMTestCase` construction) rather
  than just type-checking. It has not been run against a live judge model,
  so the *scores* in `output/eval_report.md` are still from the heuristic
  path, not DeepEval's.
- Ahead of a real credentialed run, this codebase got one more hardening
  pass worth naming: every entrypoint now calls `load_dotenv()` on import
  (previously `.env` was documented but never actually read, which would
  have silently kept every run on the offline fallback with no error), and
  `AzureAIFoundryProvider` now retries once without
  `response_format={"type": "json_object"}` if a call fails with it set --
  some Foundry "Models-as-a-Service" deployments (non-OpenAI models) don't
  support OpenAI's JSON-mode parameter, and this avoids silently discarding
  every response from those deployments.
- The **artifacts actually generated** in this deliverable
  (`output/*/mapping_*.json`, `output/eval_report.md`,
  `evals/golden/golden_mapping.json`) were produced by the deterministic
  offline provider, exercising the exact same graph/tenancy/HITL/eval code
  paths a live-LLM run would -- just with the reasoning step's fallback
  branch instead of a network call. Every mismatch/edge case discussed here
  (the `dob` unmapped field, the `_id` structural override, the
  denormalized `department.name`/`location.city` join fields) was verified
  by direct inspection, not assumed.

## What I'd add next for a real production rollout

- **Postgres instead of SQLite** for tenancy + checkpointing (both are
  already schema-compatible; it's a connection-string change).
- **Redis-backed rate limiting** instead of the in-process token bucket, so
  it holds correctly across multiple API instances.
- **Per-field graph fan-out** using LangGraph's `Send()` API to parallelize
  field-level LLM calls within a table instead of a sequential Python loop
  inside one node -- worth it once tables get large enough that latency
  matters.
- **OpenTelemetry** alongside LangSmith for infra-level metrics (latency,
  error rate) rather than only LLM-run tracing.
- **CI-gated evals** -- fail a PR if `run_eval.py`'s accuracy/relevance/
  faithfulness drop below a floor against `golden_mapping.json`.
- **Postgres row-level security or a proper multi-tenant DB isolation
  strategy** instead of relying purely on application-layer `tenant_id`
  filtering, for defense in depth.
