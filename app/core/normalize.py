"""
Text normalization for schema field names.

Database field names are compressed and abbreviated (f_name, dept_nm, emp_cd).
Destination field names are expanded, human-readable camelCase (firstName,
departmentId). Before any similarity comparison happens we expand both sides
into a common vocabulary of full words so lexical / vector similarity has
something meaningful to latch onto.

This module does no I/O and calls no LLM -- it is a deterministic text
transform used to build the embedding input for the retrieval stage. A
tenant can override/extend SYNONYMS via TenantConfig.extra_synonyms (see
app/tenancy/models.py) without touching this file.
"""

from __future__ import annotations
import re

ABBREVIATIONS = {
    "emp": "employee",
    "cd": "code",
    "nm": "name",
    "dept": "department",
    "mgr": "manager",
    "lvl": "level",
    "sal": "salary",
    "dt": "date",
    "ts": "timestamp",
    "id": "id",
    "loc": "location",
    "stat": "status",
    "ctr": "center",
    "prov": "province",
    "postal": "postal",
    "tz": "timezone",
    "fk": "foreignkey",
    "pk": "primarykey",
    "f": "first",
    "l": "last",
    "obj": "object",
    "isremote": "is remote",
}

SYNONYMS = {
    "hire": ["start", "begin"],
    "term": ["end", "termination", "separation"],
    "head": ["manager", "lead"],
    "mgr": ["head", "lead"],
}

GENERIC_TOKENS = {
    "date", "string", "number", "object", "id", "objectid", "boolean",
    "varchar", "char", "int", "decimal", "timestamp", "isodate", "tinyint",
    "key", "primary", "unique", "not", "foreignkey", "primarykey", "ref",
}


def _split_tokens(name: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    raw_tokens = re.split(r"[^A-Za-z0-9]+", spaced)
    return [t.lower() for t in raw_tokens if t]


def expand_tokens(name: str, extra_synonyms: dict[str, list[str]] | None = None) -> list[str]:
    """Split an identifier into tokens, expand abbreviations, add synonyms."""
    synonyms = {**SYNONYMS, **(extra_synonyms or {})}
    tokens = _split_tokens(name)
    expanded: list[str] = []
    for tok in tokens:
        replacement = ABBREVIATIONS.get(tok, tok)
        words = replacement.split()
        expanded.extend(words)
        for w in words:
            expanded.extend(synonyms.get(w, []))
        expanded.extend(synonyms.get(tok, []))
    return expanded


def normalize_field_text(*, name_or_path: str, type_str: str, comment: str = "",
                          constraints: str = "", extra: str = "",
                          extra_synonyms: dict[str, list[str]] | None = None) -> str:
    """Build the normalized bag-of-words string used as embedding input for a field."""
    segments = name_or_path.split(".")
    words: list[str] = []
    for seg in segments:
        words.extend(expand_tokens(seg, extra_synonyms))

    type_words = re.findall(r"[A-Za-z]+", type_str)
    words.extend(w.lower() for w in type_words)

    if constraints:
        words.extend(re.findall(r"[A-Za-z]+", constraints.lower()))
    if comment:
        words.extend(re.findall(r"[A-Za-z]{2,}", comment.lower()))
    if extra:
        words.extend(re.findall(r"[A-Za-z]+", extra.lower()))

    return " ".join(words)


def content_tokens(*, name_or_path: str, comment: str = "",
                    extra_synonyms: dict[str, list[str]] | None = None) -> set[str]:
    """
    Business-meaning tokens for a field (name/path + comment, minus pure
    data-type/structural vocabulary). Used as the offline fallback's
    accept/reject gate -- see app/graph/nodes.py.
    """
    segments = name_or_path.split(".")
    words: list[str] = []
    for seg in segments:
        words.extend(expand_tokens(seg, extra_synonyms))
    if comment:
        words.extend(re.findall(r"[A-Za-z]{2,}", comment.lower()))
    return {w for w in words if w not in GENERIC_TOKENS}
