"""
Builds evals/golden/golden_mapping.json -- the reference/ground-truth mapping
used by run_eval.py to score regressions.

This runs the real graph end-to-end (auto-approving every HITL review item
with the pipeline's own top proposal, since a manual field-by-field audit --
documented in the MVP README -- already confirmed every one of those
proposals is correct for this schema pair) and persists the result as the
fixture every future pipeline change gets scored against. If you change the
retrieval/reasoning/rules logic and a metric in `run_eval.py` regresses,
that's the signal something broke -- re-run this script deliberately (not
automatically) once you've verified the new output is actually still
correct.
"""

from __future__ import annotations
import json
import os

from dotenv import load_dotenv
load_dotenv()  # picks up a .env file in the cwd, if present -- see .env.example

from langgraph.types import Command
from app.graph.build_graph import get_compiled_graph

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "golden_mapping.json")


def build_and_save() -> dict:
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": "golden-build"}}
    result = graph.invoke({"tenant_id": None, "run_id": "golden-build"}, config=config)

    while "__interrupt__" in result:
        items = result["__interrupt__"][0].value["items"]
        decisions = [
            {
                "source_table": it["source_table"],
                "source_field": it["source_field"],
                "approved": True,
                "reviewer": "golden-fixture-builder",
                "comment": "Auto-approved pipeline's own top proposal; manually audited in README.",
            }
            for it in items
        ]
        result = graph.invoke(Command(resume=decisions), config=config)

    doc = result["mapping_document"]
    assert result["validation_report"]["ok"], result["validation_report"]

    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    with open(GOLDEN_PATH, "w") as fh:
        json.dump(doc, fh, indent=2)
    return doc


if __name__ == "__main__":
    doc = build_and_save()
    total = sum(len(t["field_mappings"]) for t in doc["tables"])
    print(f"wrote {GOLDEN_PATH} ({total} field mappings across {len(doc['tables'])} tables)")
