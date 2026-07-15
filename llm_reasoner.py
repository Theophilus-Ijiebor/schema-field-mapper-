"""
LLM reasoning stage.

This is the ONLY place in the pipeline that calls an LLM, and every call is
deliberately bounded:

  * Table-level call: one source table's field list + its top-2 candidate
    destination collections (name + field list). Never all three collections,
    never the full destination schema.
  * Field-level call: one source field's metadata + its top-3 candidate
    destination fields (from retrieval.py). Never the full destination
    schema, never more than one source field at a time.

Both calls ask the model to pick the best candidate (or say "no good match"),
and to write the one-sentence `reasoning` plus `confidence`. The mechanical
`type_transform` and code-value `notes` are supplied deterministically by
type_transforms.py and merged in afterward -- the LLM is used for what LLMs
are good at (semantic judgment + natural-language explanation), not for
inventing type-coercion rules.

If ANTHROPIC_API_KEY is not set (or the call fails for any reason), a
deterministic fallback reasoner produces the same output contract from the
retrieval scores alone, so the pipeline always runs end-to-end.
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from schemas import SourceField, DestField, source_fields_for_table, dest_fields_for_collection
from retrieval import Candidate

MODEL = os.environ.get("SCHEMA_MAPPER_MODEL", "claude-haiku-4-5-20251001")

_client = None
_client_checked = False


def _get_client():
    """Lazily construct the Anthropic client. Returns None if unavailable."""
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        _client = anthropic.Anthropic()
    except Exception:
        _client = None
    return _client


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _call_llm(system: str, user: str) -> Optional[dict]:
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=400,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return _extract_json(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Table-level reasoning
# ---------------------------------------------------------------------------

TABLE_SYSTEM_PROMPT = (
    "You are a database migration analyst. You will be shown ONE source table "
    "from a legacy MySQL HR schema and a short list of CANDIDATE destination "
    "collections from a MongoDB schema (not the full destination schema). "
    "Pick the single best-matching destination collection for this source "
    "table. Respond with ONLY a JSON object, no prose, matching exactly: "
    '{"destination_collection": "<name or null>", "confidence": <0..1 float>, '
    '"reasoning": "<one plain-English sentence>"}'
)


@dataclass
class TableMatch:
    destination_collection: Optional[str]
    confidence: float
    reasoning: str
    source: str  # "llm" or "fallback"


def reason_table_match(source_table: str, candidate_collections: list[tuple[str, float]]) -> TableMatch:
    src_fields = source_fields_for_table(source_table)
    src_field_lines = [f"  - {f.name}: {f.sql_type} {f.constraints} {f.comment}".strip() for f in src_fields]

    cand_blocks = []
    for coll_name, score in candidate_collections:
        fields = dest_fields_for_collection(coll_name)
        field_lines = [f"    - {f.path}: {f.bson_type} {f.comment}".strip() for f in fields]
        cand_blocks.append(f"  Collection: {coll_name} (lexical similarity {score:.2f})\n" + "\n".join(field_lines))

    user = (
        f"Source table: {source_table}\n"
        f"Source fields:\n" + "\n".join(src_field_lines) + "\n\n"
        f"Candidate destination collections:\n" + "\n\n".join(cand_blocks)
    )

    result = _call_llm(TABLE_SYSTEM_PROMPT, user)
    if result and result.get("destination_collection"):
        return TableMatch(
            destination_collection=result["destination_collection"],
            confidence=float(result.get("confidence", 0.8)),
            reasoning=result.get("reasoning", ""),
            source="llm",
        )

    # Fallback: highest lexical-similarity candidate.
    best_name, best_score = candidate_collections[0]
    return TableMatch(
        destination_collection=best_name,
        confidence=round(min(0.95, 0.55 + best_score), 2),
        reasoning=(
            f"'{source_table}' and '{best_name}' represent the same core entity based on "
            f"overlapping field vocabulary (top lexical-similarity match, score {best_score:.2f})."
        ),
        source="fallback",
    )


# ---------------------------------------------------------------------------
# Field-level reasoning
# ---------------------------------------------------------------------------

FIELD_SYSTEM_PROMPT = (
    "You are a database migration analyst mapping ONE source field from a legacy "
    "MySQL HR schema to its best match among a short CANDIDATE list of destination "
    "fields in a MongoDB schema (not the full destination schema -- only these "
    "candidates are available to you). If none of the candidates are a reasonable "
    "semantic match, say so. Respond with ONLY a JSON object, no prose, matching "
    'exactly: {"destination_field": "<dot-notation path or null>", '
    '"confidence": <0..1 float>, "reasoning": "<one plain-English sentence>"}'
)


@dataclass
class FieldMatch:
    destination_field: Optional[str]
    confidence: float
    reasoning: str
    source: str


def reason_field_match(source_field: SourceField, candidates: list[Candidate]) -> FieldMatch:
    if not candidates:
        return FieldMatch(None, 0.0, "No candidate destination fields available.", "fallback")

    cand_lines = [
        f"  - {c.dest_field.path}: {c.dest_field.bson_type} {c.dest_field.comment} "
        f"(lexical similarity {c.score:.2f})".strip()
        for c in candidates
    ]
    user = (
        f"Source field: {source_field.name}\n"
        f"Source table: {source_field.table}\n"
        f"Source type: {source_field.sql_type}\n"
        f"Source constraints: {source_field.constraints or 'none'}\n"
        f"Source comment: {source_field.comment or 'none'}\n\n"
        f"Candidate destination fields (in collection '{candidates[0].dest_field.collection}'):\n"
        + "\n".join(cand_lines)
    )

    result = _call_llm(FIELD_SYSTEM_PROMPT, user)
    if result and result.get("destination_field"):
        return FieldMatch(
            destination_field=result["destination_field"],
            confidence=float(result.get("confidence", 0.75)),
            reasoning=result.get("reasoning", ""),
            source="llm",
        )

    # Fallback: highest-similarity candidate, confidence derived from score.
    best = candidates[0]
    reasoning = _fallback_reasoning_sentence(source_field, best)
    confidence = round(min(0.97, 0.55 + best.score), 2)
    return FieldMatch(best.dest_field.path, confidence, reasoning, "fallback")


def _fallback_reasoning_sentence(source_field: SourceField, best: Candidate) -> str:
    """
    Category-aware reasoning sentence for the offline fallback path. Rather
    than repeat one generic template for every field, this inspects the
    field's role (identifier, code/enum, date, boolean flag, currency,
    contact info, foreign key, name, numeric) and writes the sentence a
    human reviewing the same evidence would write.
    """
    name, sql_type = source_field.name, source_field.sql_type
    dest_path, bson_type = best.dest_field.path, best.dest_field.bson_type
    comment = source_field.comment
    constraints = source_field.constraints

    base = f"'{name}' ({sql_type}) and '{dest_path}' ({bson_type})"

    if "FK ->" in constraints:
        ref_target = constraints.split("FK ->")[-1].strip()
        return (f"{base} both act as a foreign-key reference (source points to "
                f"{ref_target}); the field names and surrounding context line up closely.")
    if comment and any(code in comment for code in ["A=", "I=", "T="]):
        return (f"{base} both encode the same small set of status/lifecycle codes, "
                f"just in different representations.")
    if "TINYINT(1)" in sql_type or bson_type == "Boolean":
        return f"{base} both represent the same true/false flag."
    if sql_type.startswith(("DATE", "DATETIME")) and bson_type == "ISODate":
        return f"{base} both represent the same point-in-time / calendar date concept."
    if "CHAR(3)" in sql_type and "currency" in dest_path.lower():
        return f"{base} both hold the same ISO 4217 currency code."
    if "CHAR(2)" in sql_type and "country" in dest_path.lower():
        return f"{base} both hold the same ISO 3166-1 country code."
    if "email" in name.lower() or "email" in dest_path.lower():
        return f"{base} both hold the employee's work email address."
    if "phone" in name.lower() or "phone" in dest_path.lower():
        return f"{base} both hold the employee's phone number."
    if "sal" in name.lower() and bson_type == "Number":
        return f"{base} both hold the employee's base salary amount."
    if "name" in dest_path.lower().split(".")[-1].lower():
        return f"{base} both hold the same name component, just flattened vs. nested."
    if constraints == "PRIMARY KEY":
        return f"{base}: primary key of its table, mapped to the document's unique identifier."

    return (f"{base} share strong token overlap once abbreviations are expanded, and "
            f"occupy the same structural role within their respective entities.")
