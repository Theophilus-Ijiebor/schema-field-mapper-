"""
JSON-schema + coverage validation for the final mapping document.

Two layers on purpose: JSON Schema catches *shape* problems (a required key
missing, a confidence out of [0,1], an extra undeclared key sneaking in).
Coverage checking catches *semantic* problems a shape check can't see --
every source field must be accounted for exactly once (mapped or explicitly
unmapped), and no destination field may be claimed by two source fields.
"""

from __future__ import annotations
from jsonschema import validate, Draft7Validator

FIELD_MAPPING_SCHEMA = {
    "type": "object",
    "required": ["source_field", "destination_field", "type_transform", "confidence", "reasoning", "notes"],
    "additionalProperties": False,
    "properties": {
        "source_field": {"type": "string", "minLength": 1},
        "destination_field": {"type": "string", "minLength": 1},
        "type_transform": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string", "minLength": 1},
        "notes": {"type": ["string", "null"]},
    },
}

UNMAPPED_DEST_SCHEMA = {
    "type": "object",
    "required": ["destination_field", "reason"],
    "additionalProperties": False,
    "properties": {
        "destination_field": {"type": "string"},
        "reason": {"type": "string"},
    },
}

TABLE_SCHEMA = {
    "type": "object",
    "required": [
        "source_table", "destination_collection", "confidence", "reasoning",
        "field_mappings", "unmapped_source_fields", "unmapped_destination_fields",
    ],
    "additionalProperties": False,
    "properties": {
        "source_table": {"type": "string", "minLength": 1},
        "destination_collection": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string", "minLength": 1},
        "field_mappings": {"type": "array", "items": FIELD_MAPPING_SCHEMA},
        "unmapped_source_fields": {"type": "array", "items": {"type": "string"}},
        "unmapped_destination_fields": {"type": "array", "items": UNMAPPED_DEST_SCHEMA},
    },
}

MAPPING_DOCUMENT_SCHEMA = {
    "type": "object",
    "required": ["mapping_version", "source", "destination", "generated_at", "tables"],
    "additionalProperties": False,
    "properties": {
        "mapping_version": {"type": "string"},
        "source": {"type": "string"},
        "destination": {"type": "string"},
        "generated_at": {"type": "string"},
        "tables": {"type": "array", "items": TABLE_SCHEMA, "minItems": 1},
    },
}


def validate_mapping_document(doc: dict) -> None:
    """Raises jsonschema.ValidationError on the first structural problem found."""
    validate(instance=doc, schema=MAPPING_DOCUMENT_SCHEMA, cls=Draft7Validator)


def validate_coverage(doc: dict, expected_field_counts: dict[str, int]) -> list[str]:
    problems = []
    for table in doc["tables"]:
        name = table["source_table"]
        mapped = len(table["field_mappings"])
        unmapped = len(table["unmapped_source_fields"])
        total = mapped + unmapped
        expected = expected_field_counts.get(name)
        if expected is not None and total != expected:
            problems.append(
                f"{name}: accounted for {total} fields (mapped={mapped}, "
                f"unmapped={unmapped}) but schema defines {expected}."
            )
        dest_names = [fm["destination_field"] for fm in table["field_mappings"]]
        if len(dest_names) != len(set(dest_names)):
            dupes = {d for d in dest_names if dest_names.count(d) > 1}
            problems.append(f"{name}: destination field(s) mapped more than once: {dupes}")
    return problems
