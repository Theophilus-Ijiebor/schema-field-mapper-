# Schema Field Mapper — Design Write-up

## Problem

Map every field in `legacy_hrm` (MySQL) to its semantic equivalent in `people_platform`
(MongoDB), under one hard constraint: no single LLM call may see both schemas in full and
hand back a finished mapping. That constraint rules out the obvious approach — dump both
schemas into one big prompt — so the pipeline is built as two separate stages instead of
one.

## Architecture: retrieval, then reasoning

**Stage 1 — Candidate retrieval (no LLM).** Every source field and every destination field
is turned into a short normalized text string (name, type, comment, constraints — with
common abbreviations expanded and a synonym table applied, e.g. "hire" ~ "start", "term" ~
"end/termination"). These are embedded into a shared vector space — real embeddings
(OpenAI / Azure AI Foundry) when configured, TF-IDF as a fully offline fallback — with
source-side and destination-side texts embedded in two **separate calls**. Cosine
similarity gives a shortlist: which destination collection a source table most likely
belongs to, and within that collection, the top-k candidate destination fields for each
source field.

**Stage 2 — Reasoning (one LLM call per decision, narrow context).** The LLM is only ever
shown one source field plus its top-k shortlist of destination candidates — never the full
source schema, never the full destination schema, and never both sides in bulk. Two prompt
shapes:

- *Table-level*: "Given this one source table's fields and comments, which of these
  candidate destination collections fits best?" (candidates already narrowed by retrieval).
- *Field-level*: "Given this one source field (name, type, comment, constraints) and these
  2-3 candidate destination fields (path, type, comment), which is the best match, with what
  confidence, and why?"

Both prompts require a structured JSON response (`destination_field`, `confidence`,
`reasoning`) so output is directly usable without a third parsing pass. This two-stage
split is what makes the constraint achievable: retrieval does the cross-schema comparison
mathematically (via vector similarity), and the LLM only ever reasons about a narrow,
single-source-field decision.

## Design decisions

**Confidence threshold + human-in-the-loop.** Matches below a configurable threshold
(default 0.85) are routed to a human review step rather than auto-accepted. Ambiguous
cases — e.g. `dept_cd` vs. `costCenterCode` vs. `code` — are exactly the kind of decision
an LLM can get plausibly wrong with a reasonable-sounding justification, so low-confidence
matches are surfaced with the model's reasoning and alternative candidates rather than
silently accepted.

**Type transforms are inferred, not guessed per-field by the LLM.** A small rules layer
maps common SQL→BSON type pairs (`TINYINT(1)` → `Boolean`, `DECIMAL` → `Number`, `DATETIME`
→ `ISODate`) and flags fields backed by a fixed code table (e.g. `rec_stat`: `A`/`I`/`T`) as
`CHAR code → String enum` with the concrete value mapping recorded in `notes`. This keeps
type-transform output consistent instead of relying on the LLM to reinvent the same
transform description differently each run.

**Denormalized / joined fields are flagged, not silently dropped.** Several destination
fields (`department.name`, `location.city`, etc.) don't exist directly in the source table —
they're only reachable by following a foreign key (`dept_id`, `office_loc_id`) into another
table. Rather than force a false 1:1 field mapping, these are listed under
`unmapped_destination_fields` with an explanation of which join produces them, and the
foreign-key field itself is mapped with a note describing the required lookup.

**Fields with no destination equivalent are left unmapped, not forced.** `dob` (date of
birth) has no counterpart anywhere in the destination schema — likely intentionally, for
PII/compliance reasons. Rather than force it onto a loosely related field, it's reported
under `unmapped_source_fields`.

**Deterministic fallback.** If no LLM provider is configured (or a call fails), the pipeline
falls back to the retrieval stage's own top-ranked candidate plus a templated reasoning
string, so the pipeline always produces a complete, schema-valid output — degraded
confidence and reasoning quality, but never a hard failure.

## Output

A single JSON document (`mapping_version`, `source`, `destination`, `generated_at`, and one
entry per source table under `tables[]`), where each table lists its `field_mappings[]`
(with the required `source_field` / `destination_field` / `type_transform` / `confidence` /
`reasoning` / `notes` per the spec), plus `unmapped_source_fields` and
`unmapped_destination_fields` for full accountability of every field on both sides.

## Result

33 of 34 source fields mapped across all three tables (`emp_master` → `employees`,
`dept_info` → `departments`, `locations` → `locations`), with `dob` correctly left
unmapped and all seven denormalized destination fields correctly attributed to their join
source rather than force-mapped.
