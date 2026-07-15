"""
Entry point: run the full pipeline and write the mapping JSON to disk.

Usage:
    python3 main.py [output_path]

    ANTHROPIC_API_KEY=sk-...  python3 main.py     # uses real Claude calls
    python3 main.py                                # offline deterministic fallback
"""

from __future__ import annotations
import json
import os
import sys

from pipeline import build_mapping
from validate import validate_mapping_document, validate_coverage
from schemas import SOURCE_FIELDS

EXPECTED_FIELD_COUNTS = {}
for f in SOURCE_FIELDS:
    EXPECTED_FIELD_COUNTS[f.table] = EXPECTED_FIELD_COUNTS.get(f.table, 0) + 1


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "output/mapping_output.json"

    using_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print(f"[schema-field-mapper] LLM reasoning mode: "
          f"{'Anthropic API (ANTHROPIC_API_KEY set)' if using_llm else 'offline deterministic fallback (no API key found)'}",
          file=sys.stderr)

    mapping = build_mapping()

    print("[schema-field-mapper] validating output against JSON schema...", file=sys.stderr)
    validate_mapping_document(mapping)

    problems = validate_coverage(mapping, EXPECTED_FIELD_COUNTS)
    if problems:
        print("[schema-field-mapper] COVERAGE PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)
    print("[schema-field-mapper] validation passed: every source field is mapped or explicitly unmapped, "
          "no destination field is mapped twice.", file=sys.stderr)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(mapping, fh, indent=2)

    total_mapped = sum(len(t["field_mappings"]) for t in mapping["tables"])
    total_unmapped_src = sum(len(t["unmapped_source_fields"]) for t in mapping["tables"])
    print(f"[schema-field-mapper] wrote {out_path} "
          f"({total_mapped} fields mapped, {total_unmapped_src} source fields unmapped)", file=sys.stderr)


if __name__ == "__main__":
    main()
