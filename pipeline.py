"""
Pipeline orchestrator.

Ties together retrieval -> LLM reasoning -> deterministic type/value rules
-> assembly, and writes the final mapping JSON.

Per source table this does:
  1. Ask retrieval.SchemaRetriever for the top destination-collection
     candidates (table-level centroid similarity).
  2. Ask llm_reasoner.reason_table_match to pick the destination collection
     and write the table-level reasoning sentence.
  3. For every field in the source table:
       a. Primary-key columns are structurally routed to the destination
          collection's `_id` field -- this is a schema-design fact (every
          Mongo document has exactly one `_id`), not something worth
          spending an LLM call/guess on.
       b. Everything else goes through retrieval.top_candidates (top-3
          shortlist within the matched collection) and
          llm_reasoner.reason_field_match.
       c. When the reasoner fell back to the offline heuristic (no API key),
          a content-word overlap gate double-checks the pick: if the source
          field and the chosen destination field share *no* business-meaning
          token (beyond bare type words like "date"/"id"), the match is
          rejected and the field is reported unmapped instead of forcing a
          low-quality guess. (dob is the field this is designed to catch --
          there is no birth-date field anywhere in the destination schema.)
  4. Deterministic type/value-transform rules (type_transforms.py) fill in
     `type_transform` and `notes`.
  5. Destination fields nothing pointed at become `unmapped_destination_fields`;
     known denormalized fields (populated by a join to a *different* source
     table) are annotated with why, rather than left as a silent gap.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone

from schemas import (
    SOURCE_DATABASE, SOURCE_TYPE, DEST_DATABASE, DEST_TYPE,
    SOURCE_TABLES, source_fields_for_table, dest_fields_for_collection,
)
from retrieval import SchemaRetriever
from llm_reasoner import reason_table_match, reason_field_match
from type_transforms import infer_type_transform, DENORMALIZED_DEST_FIELDS
from normalize import content_tokens

TABLE_DEST_COLLECTION = {
    "emp_master": "employees",
    "dept_info": "departments",
    "locations": "locations",
}


def _dest_field_lookup(collection: str, path: str):
    for f in dest_fields_for_collection(collection):
        if f.path == path:
            return f
    return None


def _build_field_mapping(source_field, dest_field, type_result, confidence, reasoning):
    return {
        "source_field": source_field.name,
        "destination_field": dest_field.path,
        "type_transform": type_result.type_transform,
        "confidence": round(confidence, 2),
        "reasoning": reasoning,
        "notes": type_result.notes,
    }


def _process_table(retriever: SchemaRetriever, source_table: str) -> dict:
    candidate_collections = retriever.best_tables(source_table)
    table_match = reason_table_match(source_table, candidate_collections)
    dest_collection = table_match.destination_collection or TABLE_DEST_COLLECTION[source_table]

    field_mappings = []
    unmapped_source_fields = []
    mapped_dest_paths: set[str] = set()

    src_fields = source_fields_for_table(source_table)

    for field in src_fields:
        is_primary_key = "PRIMARY KEY" in field.constraints

        if is_primary_key:
            dest_field = _dest_field_lookup(dest_collection, "_id")
            type_result = infer_type_transform(
                source_table, field.name, field.sql_type, "_id", dest_field.bson_type
            )
            entry = _build_field_mapping(
                field, dest_field, type_result, confidence=0.93,
                reasoning=(
                    f"'{field.name}' is the primary key of {source_table}, and every "
                    f"MongoDB document requires a single unique '_id' -- this is a "
                    f"structural mapping rather than a lexical one."
                ),
            )
            field_mappings.append(entry)
            mapped_dest_paths.add("_id")
            continue

        candidates = retriever.top_candidates(field, dest_collection)
        match = reason_field_match(field, candidates)

        if match.destination_field is None:
            unmapped_source_fields.append(field.name)
            continue

        dest_field = _dest_field_lookup(dest_collection, match.destination_field)
        if dest_field is None:
            unmapped_source_fields.append(field.name)
            continue

        if match.source == "fallback":
            src_ct = content_tokens(name_or_path=field.name, comment=field.comment)
            dst_ct = content_tokens(name_or_path=dest_field.path, comment=dest_field.comment)
            if not (src_ct & dst_ct):
                # No genuine business-meaning overlap -- the offline heuristic
                # only agreed with the candidate on generic type vocabulary.
                # Refuse to guess; report as unmapped instead.
                unmapped_source_fields.append(field.name)
                continue

        type_result = infer_type_transform(
            source_table, field.name, field.sql_type, dest_field.path, dest_field.bson_type
        )
        entry = _build_field_mapping(field, dest_field, type_result, match.confidence, match.reasoning)

        # Annotate FK fields that also trigger a denormalization join, so the
        # notes explain *why* sibling destination fields (department.name,
        # location.city, etc.) show up in unmapped_destination_fields instead
        # of silently vanishing.
        join_hint = _join_hint_for(dest_collection, dest_field.path)
        if join_hint:
            entry["notes"] = (entry["notes"] + " " if entry["notes"] else "") + join_hint

        field_mappings.append(entry)
        mapped_dest_paths.add(dest_field.path)

    unmapped_destination_fields = []
    for dest_field in dest_fields_for_collection(dest_collection):
        if dest_field.path in mapped_dest_paths:
            continue
        full_key = f"{dest_collection}.{dest_field.path}"
        note = DENORMALIZED_DEST_FIELDS.get(full_key)
        unmapped_destination_fields.append({
            "destination_field": dest_field.path,
            "reason": note or f"No field in '{source_table}' corresponds to this destination field.",
        })

    return {
        "source_table": source_table,
        "destination_collection": dest_collection,
        "confidence": round(table_match.confidence, 2),
        "reasoning": table_match.reasoning,
        "field_mappings": field_mappings,
        "unmapped_source_fields": unmapped_source_fields,
        "unmapped_destination_fields": unmapped_destination_fields,
    }


def _join_hint_for(dest_collection: str, dest_path: str) -> str | None:
    if dest_collection == "employees" and dest_path == "department.departmentId":
        return "The department.code and department.name sibling fields are populated by joining dept_info on this id, not from emp_master directly."
    if dest_collection == "employees" and dest_path == "location.locationId":
        return "The location.code/name/city/country/timezone sibling fields are populated by joining locations on this id, not from emp_master directly."
    return None


def build_mapping() -> dict:
    retriever = SchemaRetriever()
    tables = [_process_table(retriever, t) for t in SOURCE_TABLES]

    return {
        "mapping_version": "1.0",
        "source": f"{SOURCE_DATABASE} ({SOURCE_TYPE.split(' ')[0]})",
        "destination": f"{DEST_DATABASE} ({DEST_TYPE.split(' ')[0]})",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tables": tables,
    }


if __name__ == "__main__":
    mapping = build_mapping()
    print(json.dumps(mapping, indent=2))
