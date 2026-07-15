"""
Text normalization for schema field names.

Database field names are compressed and abbreviated (f_name, dept_nm, emp_cd).
Destination field names are expanded, human-readable camelCase (firstName,
departmentId). Before any similarity comparison happens we expand both sides
into a common vocabulary of full words so lexical / vector similarity has
something meaningful to latch onto.

This module does no I/O and calls no LLM -- it is a deterministic text
transform used to build the embedding input for the retrieval stage.
"""

from __future__ import annotations
import re

# Common HR / DB abbreviations -> expanded word(s).
# Keys are matched against underscore/camelCase-split tokens (lowercased).
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

# Domain synonyms: words that mean the same thing in HR/employment vocabulary
# but don't share a common substring, so lexical overlap alone would miss
# them (e.g. "hire date" and "start date"). Applied as *additional* tokens
# alongside the original word, never a replacement, so nothing is lost.
SYNONYMS = {
    "hire": ["start", "begin"],
    "term": ["end", "termination", "separation"],
    "head": ["manager", "lead"],
    "mgr": ["head", "lead"],
}

# Tokens that only encode data-type / structural vocabulary rather than the
# *business meaning* of a field. Two fields sharing only these tokens (e.g.
# both being "a date") is not evidence they represent the same concept --
# see content_tokens() below, used as an accept/reject gate downstream.
GENERIC_TOKENS = {
    "date", "string", "number", "object", "id", "objectid", "boolean",
    "varchar", "char", "int", "decimal", "timestamp", "isodate", "tinyint",
    "key", "primary", "unique", "not", "foreignkey", "primarykey", "ref",
}


def _split_tokens(name: str) -> list[str]:
    """Split snake_case and camelCase identifiers into lowercase word tokens."""
    # Insert boundaries at camelCase transitions, then split on non-alphanumerics.
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    raw_tokens = re.split(r"[^A-Za-z0-9]+", spaced)
    return [t.lower() for t in raw_tokens if t]


def expand_tokens(name: str) -> list[str]:
    """Split an identifier into tokens, expand abbreviations, add synonyms."""
    tokens = _split_tokens(name)
    expanded: list[str] = []
    for tok in tokens:
        replacement = ABBREVIATIONS.get(tok, tok)
        words = replacement.split()
        expanded.extend(words)
        for w in words:
            expanded.extend(SYNONYMS.get(w, []))
        expanded.extend(SYNONYMS.get(tok, []))
    return expanded


def normalize_field_text(*, name_or_path: str, type_str: str, comment: str = "",
                          constraints: str = "", extra: str = "") -> str:
    """
    Build the normalized bag-of-words string used as embedding input for a
    single field. Dotted paths (destination side) are split per segment so
    'fullName.firstName' expands to 'full name first name'.
    """
    segments = name_or_path.split(".")
    words: list[str] = []
    for seg in segments:
        words.extend(expand_tokens(seg))

    type_words = re.findall(r"[A-Za-z]+", type_str)
    words.extend(w.lower() for w in type_words)

    if constraints:
        words.extend(re.findall(r"[A-Za-z]+", constraints.lower()))
    if comment:
        # Comments are natural language already; light tokenization only,
        # skip pure codes/examples like "A=Active" fragments' punctuation.
        words.extend(re.findall(r"[A-Za-z]{2,}", comment.lower()))
    if extra:
        words.extend(re.findall(r"[A-Za-z]+", extra.lower()))

    return " ".join(words)


def content_tokens(*, name_or_path: str, comment: str = "") -> set[str]:
    """
    Tokens that carry *business meaning* for a field: the name/path plus its
    comment, with pure data-type/structural words filtered out. Used as an
    accept/reject gate for the offline fallback reasoner -- two fields that
    only agree on "this is a date" are not proven to be the same field; two
    fields that agree on "start"/"hire" or "end"/"term" are.
    """
    segments = name_or_path.split(".")
    words: list[str] = []
    for seg in segments:
        words.extend(expand_tokens(seg))
    if comment:
        words.extend(re.findall(r"[A-Za-z]{2,}", comment.lower()))
    return {w for w in words if w not in GENERIC_TOKENS}
