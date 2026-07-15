"""
Deterministic type-transform and value-transform rules.

Once a (source_field -> destination_field) pair has been chosen, the
type_transform string and any value-mapping notes follow mechanically from
the pair of declared types plus a handful of well-known HR-domain code
conventions. This is a rules layer, not an LLM guess -- see the MVP README
for the rationale (an LLM is good at *picking the right field*, a rules
table is better at *stating the mechanical type coercion*, consistently,
every time).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

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

    if base == "TINYINT" and bson_type == "Boolean":
        return TypeTransformResult(type_transform=f"{sql_type} -> Boolean", notes=None)

    if base in {"DATE", "DATETIME", "TIMESTAMP"} and bson_type == "ISODate":
        return TypeTransformResult(type_transform=f"{sql_type} -> ISODate", notes=None)

    if base == "DECIMAL" and bson_type == "Number":
        return TypeTransformResult(
            type_transform=f"{sql_type} -> Number",
            notes="Watch for float precision loss; consider Decimal128 if exact "
                  "precision must be preserved.",
        )

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

    if base in {"VARCHAR", "CHAR", "TEXT"} and bson_type == "String":
        return TypeTransformResult(type_transform=f"{sql_type} -> String", notes=None)

    return TypeTransformResult(type_transform=f"{sql_type} -> {bson_type}", notes=None)


# Destination fields in `employees` that are populated by joining a *different*
# source table (dept_info via dept_id, locations via office_loc_id) rather
# than by any direct emp_master column. Encoded here because it's a
# structural fact about the schema pair, not a per-field semantic judgment.
DENORMALIZED_DEST_FIELDS = {
    "employees.department.code": "Populated by joining emp_master.dept_id -> dept_info.dept_cd during migration (not a direct emp_master field).",
    "employees.department.name": "Populated by joining emp_master.dept_id -> dept_info.dept_nm during migration (not a direct emp_master field).",
    "employees.location.code": "Populated by joining emp_master.office_loc_id -> locations.loc_cd during migration (not a direct emp_master field).",
    "employees.location.name": "Populated by joining emp_master.office_loc_id -> locations.loc_nm during migration (not a direct emp_master field).",
    "employees.location.city": "Populated by joining emp_master.office_loc_id -> locations.city during migration (not a direct emp_master field).",
    "employees.location.country": "Populated by joining emp_master.office_loc_id -> locations.country_cd during migration (not a direct emp_master field).",
    "employees.location.timezone": "Populated by joining emp_master.office_loc_id -> locations.tz_cd during migration (not a direct emp_master field).",
}

JOIN_HINTS = {
    ("employees", "department.departmentId"):
        "The department.code and department.name sibling fields are populated by joining "
        "dept_info on this id, not from emp_master directly.",
    ("employees", "location.locationId"):
        "The location.code/name/city/country/timezone sibling fields are populated by "
        "joining locations on this id, not from emp_master directly.",
}
