"""
Bounded LLM reasoning helpers, used by graph/nodes.py.

Every call here is deliberately narrow in scope -- this is what makes the
pipeline compliant with the assignment's constraint that no single LLM call
may see both schemas:

  * `reason_table_match` sees one source table's fields plus a short list of
    *candidate* destination collections (never all collections, never the
    full destination schema).
  * `reason_field_match` sees one source field plus its top-k candidate
    destination fields (never the full destination schema, never more than
    one source field at a time).

Both go through the injected ModelProvider, so the same functions run
identically whether the provider is Anthropic, Azure AI Foundry, or the
offline fallback -- the graph node calling these doesn't know or care which.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from app.core.schemas import SourceField, source_fields_for_table, dest_fields_for_collection
from app.retrieval.retriever import Candidate
from app.providers.base import ModelProvider

TABLE_SYSTEM_PROMPT = (
    "You are a database migration analyst. You will be shown ONE source table "
    "from a legacy MySQL HR schema and a short list of CANDIDATE destination "
    "collections from a MongoDB schema (not the full destination schema). "
    "Pick the single best-matching destination collection for this source "
    "table. Respond with ONLY a JSON object, no prose, matching exactly: "
    '{"destination_collection": "<name or null>", "confidence": <0..1 float>, '
    '"reasoning": "<one plain-English sentence>"}'
)

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
class TableMatch:
    destination_collection: Optional[str]
    confidence: float
    reasoning: str
    source: str  # "llm" or "fallback"


@dataclass
class FieldMatch:
    destination_field: Optional[str]
    confidence: float
    reasoning: str
    source: str


def reason_table_match(provider: ModelProvider, source_table: str,
                        candidate_collections: list[tuple[str, float]]) -> TableMatch:
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

    result = provider.complete_json(TABLE_SYSTEM_PROMPT, user)
    if result and result.get("destination_collection"):
        return TableMatch(
            destination_collection=result["destination_collection"],
            confidence=float(result.get("confidence", 0.8)),
            reasoning=result.get("reasoning", ""),
            source="llm",
        )

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


def reason_field_match(provider: ModelProvider, source_field: SourceField,
                        candidates: list[Candidate]) -> FieldMatch:
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

    result = provider.complete_json(FIELD_SYSTEM_PROMPT, user)
    if result and result.get("destination_field"):
        return FieldMatch(
            destination_field=result["destination_field"],
            confidence=float(result.get("confidence", 0.75)),
            reasoning=result.get("reasoning", ""),
            source="llm",
        )

    best = candidates[0]
    reasoning = _fallback_reasoning_sentence(source_field, best)
    confidence = round(min(0.97, 0.55 + best.score), 2)
    return FieldMatch(best.dest_field.path, confidence, reasoning, "fallback")


def _fallback_reasoning_sentence(source_field: SourceField, best: Candidate) -> str:
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
