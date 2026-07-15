# Schema Field Mapper — Step-by-Step Execution Trace

This follows one real command from start to finish:

    python3 -m app.cli.main --tenant acme-hr

## `app/cli/main.py` — the entry point

**Step 1.** `load_dotenv()` runs first, before any project code, populating `os.environ`
from `.env`.

**Step 2.** `main()` parses CLI args (`--tenant`, `--resume`, `--auto-approve`,
`--output-dir`), calls `run(...)`.

**Step 3.** `run()` generates a run id (`new_run_id()`) unless `--resume` was given.

**Step 4.** `run()` enters `with tenant_trace(tenant_id, run_id):`.

## `app/observability/tracing.py` — tracing wrapper

**Step 5.** `tenant_trace()` checks `langsmith_enabled()`. If not set, it's a no-op and
execution falls through to Step 6. If set, it enters `tracing_v2_enabled(...)`, logs
`langsmith.trace_start`, yields control back.

## `app/graph/build_graph.py` — assembling the pipeline

**Step 6.** `graph = get_compiled_graph()` builds the `StateGraph`: four nodes
(`load_config`, `match_tables`, `match_fields`, `validate`), the human-review interrupt
point, a SQLite checkpointer, compiled.

**Step 7.** `run()` calls `graph.invoke({"tenant_id": ..., "run_id": ...}, config=...)` —
LangGraph starts executing nodes, threading a shared `GraphState` through each one.

## `app/graph/nodes.py` — `load_config_node`

**Step 8.** Looks up tenant config via `app/tenancy/registry.py` — threshold, pinned
provider, pinned embedding provider, synonym overrides.

**Step 9.** Calls `get_provider()` (`app/providers/factory.py`) — tenant preference to
Anthropic to Azure AI Foundry to offline, first `available()` one wins.

**Step 10.** Calls `get_embedding_provider()` (`app/providers/embeddings.py`) — Azure or
OpenAI embeddings, or `None` (meaning TF-IDF fallback).

**Step 11.** Logs `load_config` with the resolved provider, threshold, retrieval backend.

## `app/graph/nodes.py` — `match_tables_node`, and `app/retrieval/retriever.py`

**Step 12.** Constructs a `SchemaRetriever`, passing tenant synonym overrides and the
embedding provider from Step 10.

**Step 13.** For every source and destination field, `normalize_field_text()`
(`app/core/normalize.py`) builds a normalized text string.

**Step 14.** If an embedding provider is available: source texts and destination texts are
embedded in two separate API calls. Otherwise TF-IDF fits on both together (vocabulary only)
and transforms each side separately. Sets `self.backend`.

**Step 15.** `_build_centroids()` averages each table's/collection's field vectors into one
centroid vector (reshaped to guarantee 2D).

**Step 16.** For each source table: `retriever.best_tables(source_table)` computes cosine
similarity between centroids, returns top 2 candidate collections. Pure math, no LLM yet.

**Step 17.** Builds a prompt with only the current table's fields plus the top-2 candidates,
calls `provider.complete_json()` — into `anthropic_provider.py` or
`azure_foundry_provider.py`, through `with_retry()`, parsed by `extract_json()`.

**Step 18.** Logs `match_tables` with destination_collection, confidence, match_source
(`llm` or `fallback`). Repeats per table.

## `app/graph/nodes.py` — `match_fields_node`, and `app/rules/type_transforms.py`

**Step 19.** For every source field: `retriever.top_candidates()` returns top 2-3
destination field candidates via cosine similarity.

**Step 20.** `infer_type_transform()` runs first — rules table for common type pairs and
coded-field value transforms.

**Step 21.** Builds a prompt with only this one field plus its candidates, calls
`provider.complete_json()`.

**Step 22.** If confidence >= threshold: auto-approved into the `MappingDocument`. If below:
added to a pending-review list with reasoning and alternatives.

**Step 23.** Logs `match_fields` with total pending review count.

## Human-in-the-loop interrupt

**Step 24.** If items are pending, the graph calls LangGraph's `interrupt()`, suspending
execution, checkpointed to SQLite, returns control to `run()`.

**Step 25.** `run()` prints each `REVIEW NEEDED` block, prompts for Y/n/alternate/skip.

**Step 26.** Once all answered, `run()` calls `graph.invoke(Command(resume=decisions),
config=...)`, resuming from the exact pause point.

## `app/graph/nodes.py` — `validate_node`, and `app/core/validation.py`

**Step 27.** Runs the assembled `MappingDocument` through validation — required keys,
confidence range, no duplicate destination claims, well-formed dot paths.

## Back to `app/cli/main.py` — writing output

**Step 28.** `MappingDocument.to_spec_json()` serializes the result, written to
`output/<tenant>/mapping_<run_id>.json`.

**Step 29.** The `tenant_trace` block exits — logs `langsmith.trace_end` if active.

**Step 30.** Prints the summary line, logs `cli.run_complete`, process exits.

## If asked "what happens when a live API call fails"

Every provider call goes through `with_retry()`: 3 attempts, exponential backoff, exception
captured not raised, `None` returned on total failure. The nodes treat `None` the same as
low confidence — fall back to retrieval's own top candidate with templated reasoning, log
`match_source: "fallback"`, and the pipeline still finishes with a complete, valid output.

## If asked "how do you know the constraint is actually satisfied"

Step 14 is the only place source and destination text get vectorized, always two separate
calls. Step 17/21 is the only place an LLM is invoked, and by then the prompt contains one
source-side item and a 2-3 item destination shortlist — never the other side's full field
list.
