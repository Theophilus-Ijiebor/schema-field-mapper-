# Schema Field Mapper — Code Walkthrough

Everything here refers to the `app/` package. Ignore any loose `.py` files sitting at the
project root (`main.py`, `normalize.py`, `pipeline.py`, `retrieval.py`, `schemas.py`,
`type_transforms.py`, `validate.py`, `llm_reasoner.py`) — those are leftovers from before
the code got restructured into `app/`, and nothing runs them anymore.

## The data flow, in one paragraph

A run starts in `app/cli/main.py`, which builds a LangGraph state machine
(`app/graph/build_graph.py`) and invokes it. The graph loads tenant config, then for each
source table it asks `app/retrieval/retriever.py` for a shortlist of candidate destination
collections and fields (no LLM involved yet), then hands each individual field plus its
shortlist to an LLM through `app/providers/` for a final decision. Low-confidence decisions
get paused for a human to approve via LangGraph's interrupt mechanism. Once everything is
resolved, the result is validated against the required JSON shape and written to disk. All
of this is optionally traced to LangSmith and scoped per-tenant through
`app/tenancy/`. Separately, `app/evals/` can replay a run and score it for accuracy against
a hand-checked "golden" mapping, plus quality metrics from an LLM judge.

## `app/core/` — the domain layer

**`schemas.py`** — Both database schemas from the assignment, transcribed as plain Python
dataclasses instead of JSON strings, so the rest of the code can work with typed objects
instead of parsing text everywhere. `SourceField` holds a MySQL column (table, name,
sql_type, comment, constraints); `DestField` holds a Mongo field (collection, path, bson_type,
comment). `SOURCE_FIELDS` and `DEST_FIELDS` are the full lists — 34 and 39 entries. This file
is the only place either schema exists in full; nothing downstream ever holds both schemas
in memory as one blob that could accidentally get pasted into a single prompt.

**`normalize.py`** — Turns one field's metadata into a short bag-of-words string for the
retrieval stage. `ABBREVIATIONS` expands things like `dt` to `date`, `cd` to `code`, `nm` to
`name`. `SYNONYMS` bridges vocabulary gaps that pure lexical matching would miss — `hire`
lines up with `start`/`begin`, `term` lines up with `end`/`termination`/`separation`. The
`content_tokens()` gate filters out near-meaningless generic words (`id`, `code`, `name` on
their own) so two unrelated fields that both happen to be "a code field" don't score falsely
high on term overlap alone. `normalize_field_text()` is the actual entry point — takes a
field's name, type, comment, and constraints and returns one normalized string.

**`models.py`** — Pydantic models for everything the pipeline produces: `FieldMapping`
(source_field, destination_field, type_transform, confidence, reasoning, notes — exactly the
assignment's required shape), `TableMapping` (wraps a list of `FieldMapping` plus
unmapped_source_fields / unmapped_destination_fields), `MappingDocument` (the top-level
object — mapping_version, source, destination, generated_at, tables[]), with a
`to_spec_json()` method that serializes to exactly the JSON shape the assignment asks for.
Also `ReviewItem` and `ReviewDecision` for the human-in-the-loop step.

**`validation.py`** — Checks a finished `MappingDocument` against structural rules before
it's written out: every required key present, confidence in [0,1], no duplicate destination
fields claimed by two different source fields, dot-notation paths well-formed. This is what
produces the `"event": "validate", "ok": true, "problems": 0` line you see in every run's
log output.

## `app/retrieval/retriever.py` — candidate shortlisting, no LLM

This is the mechanism that actually makes the "never show both schemas to one LLM call"
constraint possible. `SchemaRetriever` builds a vector space over all source field texts and
all destination field texts, using one of two interchangeable backends:

- **Embeddings** (when `app/providers/embeddings.py` has a configured provider) — real
  OpenAI or Azure AI Foundry embedding models. Source texts and destination texts are
  embedded in two separate API calls, never combined into one request.
- **TF-IDF** (scikit-learn, fully offline) — the fallback when no embedding provider is
  configured. `.fit()` is called on both sides together purely to build shared vocabulary
  statistics (that's not a semantic comparison, just term-frequency bookkeeping), but
  `.transform()` — the actual vectorization — is still called separately per side.

`best_tables()` averages each table's/collection's field vectors into a centroid and
compares centroids with cosine similarity, to find which destination collection a source
table probably belongs to. `top_candidates()` does the same at the individual-field level,
restricted to whichever collection was already chosen — returning the top 2-3 destination
field candidates for one source field. Both of these are the only two entry points the
graph nodes use, and both return a short candidate list — never the full opposing schema.

## `app/rules/type_transforms.py` — deterministic type logic

Rather than have the LLM invent a type-transform description from scratch every time,
`infer_type_transform()` runs a small rules table first: `TINYINT(1)` to `Boolean`,
`DECIMAL` to `Number`, `DATETIME` to `ISODate`, and so on. `CODE_VALUE_MAPS` holds the actual
value-level transforms for coded fields (`rec_stat`: A/I/T to active/inactive/terminated;
`dept_stat`: A/I to true/false) so the `notes` field is populated consistently instead of
depending on the model to remember to spell out the value mapping. `DENORMALIZED_DEST_FIELDS`
and `JOIN_HINTS` identify destination fields that only exist via a foreign-key join
(`department.name`, `location.city`, etc.) so those get reported as unmapped-with-explanation
rather than force-matched.

## `app/providers/` — swappable LLM and embedding backends

**`base.py`** — The `ModelProvider` abstract base class every backend implements:
`complete_json()` (structured JSON response), `complete_text()` (raw text, used by the
DeepEval judge), `available()` (do we have what we need to run). `with_retry()` is a shared
retry-with-backoff helper (3 attempts, exponential delay) used by every provider; it accepts
an optional `on_error` callback so a caller like `selftest.py` can capture the real exception
message instead of just getting `None` back. `extract_json()` parses a model's JSON reply,
falling back to a regex extraction if the model wrapped its answer in markdown fencing or
a little prose.

**`anthropic_provider.py`** — Talks to Claude via the `anthropic` SDK. Lazily constructs the
client only when first needed (so importing this module never fails just because a key isn't
set yet), reads `ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL` from the environment.

**`azure_foundry_provider.py`** — Talks to an Azure AI Foundry chat deployment via the
`openai` SDK's `AzureOpenAI` client. Automatically falls back from `max_tokens` to
`max_completion_tokens` if the deployed model rejects the older parameter name (newer model
families require the new name; older ones still need the old one — detected from the API's
own error text and retried once rather than hardcoded), and falls back from JSON mode to
plain-text-with-regex-extraction if a non-OpenAI model deployed behind Foundry doesn't
support `response_format={"type": "json_object"}` at all.

**`offline_provider.py`** — Deterministic no-LLM fallback. If no real provider is configured
or every real provider fails, this returns the retrieval stage's own top-ranked candidate
with a templated reasoning string, so the pipeline always finishes with a complete,
schema-valid result rather than crashing.

**`factory.py`** — `get_provider()` resolution order: explicit tenant preference to Anthropic
to Azure AI Foundry to offline. Whichever is tried first that's actually `available()` wins;
everything upstream just calls `get_provider()` and never needs to know which backend
answered.

**`embeddings.py`** — Same philosophy as `base.py` but for embedding calls specifically
(`EmbeddingProvider`, `OpenAIEmbeddingProvider`). Auto-detects Azure AI Foundry embedding
deployment vs. direct OpenAI based on which env vars are set, preferring Azure if both are
present so a Foundry-pinned tenant stays inside Foundry for both chat and embeddings.

**`selftest.py`** — `python3 -m app.providers.selftest`. Makes one real minimal call to each
configured provider and reports whether it actually works right now, not just whether
credentials are present.

## `app/graph/` — orchestration

**`state.py`** — The `GraphState` TypedDict LangGraph threads through every node: tenant_id,
run_id, resolved provider, retrieval backend, in-progress mapping document, pending review
items, etc.

**`nodes.py`** — The actual pipeline steps as graph nodes: `load_config_node` (resolve
tenant settings, provider, embedding backend), `match_tables_node` (retrieval + one narrow
LLM call per table), `match_fields_node` (retrieval + one narrow LLM call per field),
`validate_node` (runs `app/core/validation.py`). Every LLM call in this file follows the same
shape: one source-side thing, a short pre-filtered candidate list, structured JSON out.

**`build_graph.py`** — Wires the nodes into a `StateGraph`, adds the human-review interrupt
point (using LangGraph's `interrupt()`, checkpointed to SQLite so a paused run survives a
Ctrl-C and can be resumed later with `--resume <run_id>`), compiles it.

## `app/tenancy/` — multi-tenant config and access control

**`models.py`** — `TenantConfig` (confidence threshold, provider preference, embedding
provider preference, synonym overrides) and role definitions
(admin/operator/reviewer/viewer, via `ROLE_HIERARCHY`).

**`registry.py`** — SQLite-backed store for tenants, their config, and hashed API keys.
`seed_demo_data()` sets up two demo tenants: `acme-hr` (threshold 0.85, pinned to Anthropic)
and `globex-hr` (threshold 0.92, pinned to Azure AI Foundry).

## `app/observability/tracing.py` — logging and tracing

`log_event()` is always-on structured JSON-lines logging to stderr. `tenant_trace()` is a
context manager that, only when `LANGSMITH_API_KEY` is set, wraps a graph run in a LangSmith
trace under a tenant-scoped project name; it's a clean no-op otherwise. `log_feedback()`
records eval scores to the local structured log.

## `app/evals/` — scoring the pipeline against itself

**`build_golden.py`** — Runs the pipeline once and saves the result as the reference used
for regression scoring.

**`metrics.py`** — `score_against_golden()` computes accuracy/precision/recall/F1 and
type-transform agreement. `score_quality()` is the offline/heuristic fallback for relevance
and faithfulness.

**`deepeval_adapter.py` / `deepeval_metrics.py`** — When a live provider is configured,
quality scoring switches to DeepEval's `GEval` metric, a rubric-driven LLM-as-judge scoring
relevance and faithfulness. `ProviderBackedDeepEvalModel` adapts DeepEval to call through
this project's own `ModelProvider`.

**`run_eval.py`** — `python3 -m app.evals.run_eval --report-out eval_report.md`. Runs the
graph fresh (or scores an existing output), computes both regression and quality metrics,
renders the markdown report.

## `app/cli/main.py` and `app/api/main.py` — entrypoints

**`cli/main.py`** — `python3 -m app.cli.main --tenant <id>`. Builds the graph, invokes it,
prompts for human review decisions in the terminal when the graph pauses, resumes the graph
with the decision. Supports `--resume <run_id>`.

**`api/main.py`** — FastAPI surface exposing the same pipeline over HTTP, with API-key auth
checked against the tenancy registry and RBAC-gated endpoints.

## The constraint, concretely

`app/core/schemas.py` is the only file holding both full schemas as data.
`app/retrieval/retriever.py` compares them, but only through vector similarity, and even
that happens on each side independently. By the time anything reaches `app/providers/` — the
only place an LLM is actually invoked — the prompt built in `app/graph/nodes.py` contains
exactly one source-side item and a 2-3 item destination-side shortlist. That's the whole
enforcement mechanism: it's a property of what data physically gets passed into the
prompt-building code, not a rule the LLM is trusted to follow on its own.
