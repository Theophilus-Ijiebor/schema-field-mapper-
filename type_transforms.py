"""
Deterministic type-transform and value-transform rules.

Once a (source_field -> destination_field) pair has been chosen, the
type_transform string and any value-mapping notes are largely mechanical --
they follow from the pair of declared types plus a handful of well-known
HR-domain code conventions (A/I/T status codes, ISO 4217, ISO 3166-1,
TINYINT(1) booleans, etc.). Treating this as a deterministic rules layer
rather than something the LLM freely invents keeps the output consistent and
auditable, and it is also what a real schema-mapping tool would do: an LLM is
good at *picking the right field*, a rules table is better at *stating the
mechanical type coercion*.

The LLM (or the offline fallback reasoner) still writes the human-readable
`reasoning` sentence and confirms/adjusts confidence -- this module only
supplies the structural facts it can reason from.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# Known single-character / short-code enumerations seen in the source schema.
CODE_VALUE_MAPS = {
    ("emp_master", "rec_stat"): {
        "target_kind": "string_enum",
        "map": {"A": "active", "I": "inactive", "T": "terminated"},
        "note_template": "Transform: A -> active, I -> inactive, T -> terminated",
    },
    ("dept_info", "dept_stat"): {
        "target_kind": "boolean",
        "map": {"A": True, "I": False},
        "note_template": "Transform: A -> true, I -> false",
    },
}


@dataclass
class TypeTransformResult:
    type_transform: str
    notes: Optional[str]


def _base_sql_type(sql_type: str) -> str:
    return sql_type.split("(")[0].upper()


def infer_type_transform(source_table: str, source_name: str, sql_type: str,
                          dest_path: str, bson_type: str) -> TypeTransformResult:
    base = _base_sql_type(sql_type)
    key = (source_table, source_name)

    # 1. Known coded enumerations take priority -- they need an explicit value map.
    if key in CODE_VALUE_MAPS:
        spec = CODE_VALUE_MAPS[key]
        if spec["target_kind"] == "boolean":
            return TypeTransformResult(
                type_transform=f"{sql_type} code -> Boolean",
                notes=spec["note_template"],
            )
        return TypeTransformResult(
            type_transform=f"{sql_type} code -> String enum",
            notes=spec["note_template"],
        )

    # 2. Primary / foreign key integer columns -> ObjectId.
    if base == "INT" and bson_type == "ObjectId":
        if dest_path == "_id":
            return TypeTransformResult(
                type_transform=f"{sql_type} -> ObjectId",
                notes="Requires an ID-generation/lookup strategy; retain the original "
                      "integer ID as a legacy field for traceability during migration.",
            )
        return TypeTransformResult(
            type_transform=f"{sql_type} -> ObjectId (reference)",
            notes="Foreign key resolved to the corresponding document's ObjectId via a "
                  "lookup table built during migration.",
        )

    # 3. Boolean-pattern integers.
    if base == "TINYINT" and bson_type == "Boolean":
        return TypeTransformResult(type_transform=f"{sql_type} -> Boolean", notes=None)

    # 4. Dates / timestamps.
    if base in {"DATE", "DATETIME", "TIMESTAMP"} and bson_type == "ISODate":
        return TypeTransformResult(type_transform=f"{sql_type} -> ISODate", notes=None)

    # 5. Numeric.
    if base == "DECIMAL" and bson_type == "Number":
        return TypeTransformResult(
            type_transform=f"{sql_type} -> Number",
            notes="Watch for float precision loss; consider Decimal128 if exact "
                  "precision must be preserved.",
        )

    # 6. ISO code columns kept as strings but worth flagging in notes.
    if base == "CHAR" and "3" in sql_type and bson_type == "String":
        return TypeTransformResult(
            type_transform=f"{sql_type} -> String",
            notes="ISO 4217 currency code preserved as-is.",
        )
    if base == "CHAR" and "2" in sql_type and bson_type == "String":
        return TypeTransformResult(
            type_transform=f"{sql_type} -> String",
            notes="ISO 3166-1 alpha-2 country code preserved as-is.",
        )

    # 7. Plain string passthrough.
    if base in {"VARCHAR", "CHAR", "TEXT"} and bson_type == "String":
        return TypeTransformResult(type_transform=f"{sql_type} -> String", notes=None)

    # Fallback: literal type pair, no known value transform.
    return TypeTransformResult(type_transform=f"{sql_type} -> {bson_type}", notes=None)


# ---------------------------------------------------------------------------
# Denormalization / join rules
# ---------------------------------------------------------------------------
# The employees collection embeds a denormalized snapshot of department and
# location data (department.code/name, location.code/name/city/country/
# timezone). None of those destination fields has a *direct* counterpart in
# emp_master -- they are only reachable by joining dept_info via dept_id and
# locations via office_loc_id. This is a structural fact about the two
# schemas, not something for the LLM to guess field-by-field, so it is
# encoded here and used by the pipeline to annotate those destination fields
# instead of leaving them silently unmapped.

DENORMALIZED_DEST_FIELDS = {
    "employees.department.code": "Populated by joining emp_master.dept_id -> dept_info.dept_cd during migration (not a direct emp_master field).",
    "employees.department.name": "Populated by joining emp_master.dept_id -> dept_info.dept_nm during migration (not a direct emp_master field).",
    "employees.location.code": "Populated by joining emp_master.office_loc_id -> locations.loc_cd during migration (not a direct emp_master field).",
    "employees.location.name": "Populated by joining emp_master.office_loc_id -> locations.loc_nm during migration (not a direct emp_master field).",
    "employees.location.city": "Populated by joining emp_master.office_loc_id -> locations.city during migration (not a direct emp_master field).",
    "employees.location.country": "Populated by joining emp_master.office_loc_id -> locations.country_cd during migration (not a direct emp_master field).",
    "employees.location.timezone": "Populated by joining emp_master.office_loc_id -> locations.tz_cd during migration (not a direct emp_master field).",
}
