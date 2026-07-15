"""
Source-of-truth schema definitions for the Schema Field Mapper pipeline.

Both schemas are encoded as plain Python data structures, transcribed directly
from the assignment PDF. Each field record carries every piece of metadata a
human engineer would use to reason about a mapping: the raw type, constraints,
FK references, and the inline comment. That metadata is what gets turned into
text for embedding and what gets handed to the LLM reasoning stage later.

Nothing in this module talks to an LLM. It is pure data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SourceField:
    table: str
    name: str
    sql_type: str
    constraints: str = ""     # e.g. "PRIMARY KEY", "UNIQUE NOT NULL", "FK -> dept_info.dept_id"
    comment: str = ""         # inline comment from the DDL, e.g. "0 or 1"

    @property
    def full_path(self) -> str:
        return f"{self.table}.{self.name}"

    def as_text(self) -> str:
        """Canonical text representation used for embedding / LLM context."""
        parts = [f"table={self.table}", f"field={self.name}", f"type={self.sql_type}"]
        if self.constraints:
            parts.append(f"constraints={self.constraints}")
        if self.comment:
            parts.append(f"comment={self.comment}")
        return " | ".join(parts)


@dataclass
class DestField:
    collection: str
    path: str                 # dot-notation path, e.g. "fullName.firstName"
    bson_type: str
    comment: str = ""

    @property
    def full_path(self) -> str:
        return f"{self.collection}.{self.path}"

    def as_text(self) -> str:
        parts = [f"collection={self.collection}", f"field={self.path}", f"type={self.bson_type}"]
        if self.comment:
            parts.append(f"comment={self.comment}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Dataset A -- legacy_hrm (MySQL, relational)
# ---------------------------------------------------------------------------

SOURCE_DATABASE = "legacy_hrm"
SOURCE_TYPE = "MySQL (Relational)"

SOURCE_FIELDS: list[SourceField] = [
    # emp_master
    SourceField("emp_master", "emp_id", "INT", "PRIMARY KEY"),
    SourceField("emp_master", "emp_cd", "VARCHAR(20)", "UNIQUE NOT NULL", "human-readable employee code"),
    SourceField("emp_master", "f_name", "VARCHAR(50)", "NOT NULL"),
    SourceField("emp_master", "l_name", "VARCHAR(50)", "NOT NULL"),
    SourceField("emp_master", "dob", "DATE"),
    SourceField("emp_master", "hire_dt", "DATETIME"),
    SourceField("emp_master", "term_dt", "DATETIME", "", "null if still active"),
    SourceField("emp_master", "dept_id", "INT", "FK -> dept_info.dept_id"),
    SourceField("emp_master", "mgr_emp_id", "INT", "FK -> emp_master.emp_id"),
    SourceField("emp_master", "job_lvl_cd", "VARCHAR(10)", "", "e.g. L1, L2, IC3, M1"),
    SourceField("emp_master", "base_sal", "DECIMAL(12,2)"),
    SourceField("emp_master", "sal_currency", "CHAR(3)", "", "ISO 4217, e.g. USD"),
    SourceField("emp_master", "work_email", "VARCHAR(120)", "UNIQUE"),
    SourceField("emp_master", "work_phone", "VARCHAR(20)"),
    SourceField("emp_master", "office_loc_id", "INT", "FK -> locations.loc_id"),
    SourceField("emp_master", "is_remote", "TINYINT(1)", "", "0 or 1"),
    SourceField("emp_master", "rec_stat", "CHAR(1)", "", "A=Active, I=Inactive, T=Terminated"),
    SourceField("emp_master", "created_ts", "DATETIME", "", "record creation timestamp"),
    SourceField("emp_master", "updated_ts", "DATETIME", "", "last update timestamp"),

    # dept_info
    SourceField("dept_info", "dept_id", "INT", "PRIMARY KEY"),
    SourceField("dept_info", "dept_cd", "VARCHAR(20)", "UNIQUE"),
    SourceField("dept_info", "dept_nm", "VARCHAR(100)"),
    SourceField("dept_info", "parent_dept_id", "INT", "FK -> dept_info.dept_id", "self-referencing"),
    SourceField("dept_info", "dept_head_id", "INT", "FK -> emp_master.emp_id"),
    SourceField("dept_info", "cost_ctr_cd", "VARCHAR(20)", "", "finance cost center code"),
    SourceField("dept_info", "dept_stat", "CHAR(1)", "", "A=Active, I=Inactive"),

    # locations
    SourceField("locations", "loc_id", "INT", "PRIMARY KEY"),
    SourceField("locations", "loc_cd", "VARCHAR(20)", "UNIQUE"),
    SourceField("locations", "loc_nm", "VARCHAR(100)"),
    SourceField("locations", "city", "VARCHAR(80)"),
    SourceField("locations", "state_prov", "VARCHAR(80)"),
    SourceField("locations", "country_cd", "CHAR(2)", "", "ISO 3166-1 alpha-2"),
    SourceField("locations", "postal_cd", "VARCHAR(20)"),
    SourceField("locations", "tz_cd", "VARCHAR(50)", "", "IANA timezone"),
]

SOURCE_TABLES = ["emp_master", "dept_info", "locations"]


# ---------------------------------------------------------------------------
# Dataset B -- people_platform (MongoDB, document)
# ---------------------------------------------------------------------------

DEST_DATABASE = "people_platform"
DEST_TYPE = "MongoDB (Document)"

DEST_FIELDS: list[DestField] = [
    # employees
    DestField("employees", "_id", "ObjectId"),
    DestField("employees", "employeeCode", "String", "unique human-readable ID"),
    DestField("employees", "fullName.firstName", "String"),
    DestField("employees", "fullName.lastName", "String"),
    DestField("employees", "employment.startDate", "ISODate"),
    DestField("employees", "employment.endDate", "ISODate", "null if currently employed"),
    DestField("employees", "employment.status", "String", "active / inactive / terminated"),
    DestField("employees", "employment.jobLevel", "String", "e.g. L1, IC3, M1"),
    DestField("employees", "employment.isRemote", "Boolean"),
    DestField("employees", "employment.managerId", "ObjectId", "ref -> employees._id"),
    DestField("employees", "compensation.baseSalary", "Number"),
    DestField("employees", "compensation.currency", "String", "ISO 4217"),
    DestField("employees", "contact.email", "String"),
    DestField("employees", "contact.phone", "String"),
    DestField("employees", "department.departmentId", "ObjectId", "ref -> departments._id"),
    DestField("employees", "department.code", "String"),
    DestField("employees", "department.name", "String"),
    DestField("employees", "location.locationId", "ObjectId", "ref -> locations._id"),
    DestField("employees", "location.code", "String"),
    DestField("employees", "location.name", "String"),
    DestField("employees", "location.city", "String"),
    DestField("employees", "location.country", "String", "ISO 3166-1 alpha-2"),
    DestField("employees", "location.timezone", "String", "IANA timezone"),
    DestField("employees", "meta.createdAt", "ISODate"),
    DestField("employees", "meta.updatedAt", "ISODate"),

    # departments
    DestField("departments", "_id", "ObjectId"),
    DestField("departments", "code", "String"),
    DestField("departments", "name", "String"),
    DestField("departments", "parentDepartmentId", "ObjectId", "self-ref"),
    DestField("departments", "headEmployeeId", "ObjectId", "ref -> employees._id"),
    DestField("departments", "costCenterCode", "String"),
    DestField("departments", "isActive", "Boolean"),

    # locations
    DestField("locations", "_id", "ObjectId"),
    DestField("locations", "code", "String"),
    DestField("locations", "name", "String"),
    DestField("locations", "city", "String"),
    DestField("locations", "stateOrProvince", "String"),
    DestField("locations", "country", "String", "ISO 3166-1 alpha-2"),
    DestField("locations", "postalCode", "String"),
    DestField("locations", "timezone", "String"),
]

DEST_COLLECTIONS = ["employees", "departments", "locations"]


def source_fields_for_table(table: str) -> list[SourceField]:
    return [f for f in SOURCE_FIELDS if f.table == table]


def dest_fields_for_collection(collection: str) -> list[DestField]:
    return [f for f in DEST_FIELDS if f.collection == collection]
