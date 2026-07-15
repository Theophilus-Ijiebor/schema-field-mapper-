"""
Eval CLI.

    python3 -m app.evals.run_eval                      # runs the graph fresh, scores it
    python3 -m app.evals.run_eval --input path/to.json  # scores an existing mapping doc

Produces a markdown report covering:
  - accuracy / precision / recall / F1 against evals/golden/golden_mapping.json
  - type_transform agreement
  - relevance and faithfulness (LLM-judge if a provider is configured, else
    a documented heuristic fallback)

Also pushes each metric to LangSmith as feedback (no-op without credentials)
via app.observability.tracing.log_feedback, so eval history is queryable
alongside trace history when LangSmith is configured.
"""

from __future__ import annotations
import argparse
import json
import os

from dotenv import load_dotenv
load_dotenv()  # picks up a .env file in the cwd, if present -- see .env.example

from langgraph.types import Command
from app.graph.build_graph import get_compiled_graph
from app.core.schemas import SOURCE_FIELDS, DEST_FIELDS
from app.evals.metrics import score_against_golden, score_quality
from app.evals.deepeval_adapter import build_deepeval_model
from app.evals.deepeval_metrics import score_quality_deepeval
from app.providers.factory import get_provider
from app.observability.tracing import log_feedback, new_run_id

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "golden_mapping.json")


def _source_field_lookup(table: str, name: str) -> dict | None:
    for f in SOURCE_FIELDS:
        if f.table == table and f.name == name:
            return {"name": f.name, "sql_type": f.sql_type, "comment": f.comment, "constraints": f.constraints}
    return None


def _dest_field_lookup(collection: str, path: str) -> dict | None:
    for f in DEST_FIELDS:
        if f.collection == collection and f.path == path:
            return {"path": f.path, "bson_type": f.bson_type, "comment": f.comment}
    return None


def run_fresh(tenant_id: str | None, auto_approve: bool) -> dict:
    graph = get_compiled_graph()
    run_id = f"eval-{new_run_id()}"
    config = {"configurable": {"thread_id": run_id}}
    result = graph.invoke({"tenant_id": tenant_id, "run_id": run_id}, config=config)

    while "__interrupt__" in result:
        if not auto_approve:
            raise RuntimeError(
                f"Run paused for human review ({len(result['__interrupt__'][0].value['items'])} item(s)) "
                f"and --no-auto-approve was set. Use the CLI/API to resolve it, then re-score with --input."
            )
        items = result["__interrupt__"][0].value["items"]
        decisions = [
            {"source_table": it["source_table"], "source_field": it["source_field"],
             "approved": True, "reviewer": "eval-harness-auto-approve"}
            for it in items
        ]
        result = graph.invoke(Command(resume=decisions), config=config)

    return result["mapping_document"]


def render_report(regression, quality, provider_name: str, quality_engine: str = "heuristic") -> str:
    lines = [
        "# Schema Field Mapper -- Eval Report",
        "",
        f"Model provider used for LLM-judge metrics: **{provider_name}**",
        f"Quality-metric engine: **{quality_engine}**",
        "",
        "## Regression metrics (vs. golden_mapping.json)",
        "",
        f"- Overall accuracy: **{regression.overall_accuracy}**",
        f"- Precision: **{regression.overall_precision}**  Recall: **{regression.overall_recall}**  F1: **{regression.overall_f1}**",
        f"- Type-transform agreement: **{regression.type_transform_agreement}**",
        "",
        "| Table | Correct | Total | Accuracy |",
        "|---|---|---|---|",
    ]
    for table, stats in regression.per_table.items():
        lines.append(f"| {table} | {stats['correct']} | {stats['total']} | {stats['accuracy']} |")

    if regression.mismatches:
        lines += ["", "### Mismatches", ""]
        for m in regression.mismatches:
            lines.append(f"- `{m['table']}.{m['source_field']}`: golden=`{m['golden']}` candidate=`{m['candidate']}`")
    else:
        lines += ["", "No mismatches against golden."]

    lines += [
        "",
        "## Quality metrics",
        "",
        f"- Mean relevance: **{quality.mean_relevance}**",
        f"- Mean faithfulness: **{quality.mean_faithfulness}**",
        "",
        "| Table | Source field | Destination field | Relevance | Faithfulness |",
        "|---|---|---|---|---|",
    ]
    for p in quality.per_field:
        lines.append(f"| {p['table']} | {p['source_field']} | {p['destination_field']} | {p['relevance']} | {p['faithfulness']} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to an existing mapping_output.json to score instead of running fresh.")
    parser.add_argument("--tenant", default=None, help="Tenant id to run under (fresh-run mode only).")
    parser.add_argument("--no-auto-approve", action="store_true",
                         help="Fail instead of auto-approving HITL review items (fresh-run mode only).")
    parser.add_argument("--report-out", default=None, help="Write the markdown report to this path.")
    args = parser.parse_args()

    if args.input:
        with open(args.input) as fh:
            candidate = json.load(fh)
    else:
        candidate = run_fresh(args.tenant, auto_approve=not args.no_auto_approve)

    with open(GOLDEN_PATH) as fh:
        golden = json.load(fh)

    regression = score_against_golden(candidate, golden)

    provider = get_provider()
    deepeval_model = build_deepeval_model(provider)
    if deepeval_model is not None:
        quality_engine = "deepeval (GEval)"
        try:
            dq = score_quality_deepeval(deepeval_model, candidate, _source_field_lookup, _dest_field_lookup)
            quality = type("Q", (), {
                "mean_relevance": dq.mean_relevance,
                "mean_faithfulness": dq.mean_faithfulness,
                "per_field": dq.per_field,
            })()
        except Exception as exc:
            print(f"DeepEval scoring failed ({exc}); falling back to heuristic quality metrics.")
            quality_engine = "heuristic (DeepEval call failed)"
            quality = score_quality(provider, candidate, _source_field_lookup, _dest_field_lookup)
    else:
        quality_engine = "heuristic (no live provider configured)"
        quality = score_quality(provider, candidate, _source_field_lookup, _dest_field_lookup)

    for metric, value in [
        ("accuracy", regression.overall_accuracy),
        ("precision", regression.overall_precision),
        ("recall", regression.overall_recall),
        ("f1", regression.overall_f1),
        ("type_transform_agreement", regression.type_transform_agreement),
        ("relevance", quality.mean_relevance),
        ("faithfulness", quality.mean_faithfulness),
    ]:
        log_feedback(candidate.get("run_id") or "eval-run", metric, value)

    report = render_report(regression, quality, provider.name, quality_engine)
    print(report)
    if args.report_out:
        with open(args.report_out, "w") as fh:
            fh.write(report)


if __name__ == "__main__":
    main()
