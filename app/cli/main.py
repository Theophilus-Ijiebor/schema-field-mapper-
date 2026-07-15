"""
Terminal entry point.

    python3 -m app.cli.main --tenant acme-hr
    python3 -m app.cli.main --tenant acme-hr --auto-approve
    python3 -m app.cli.main --tenant acme-hr --resume <run_id>   # continue a previously paused run

Drives the same LangGraph graph the API uses, including the real
interrupt/resume HITL cycle -- when the graph pauses, this prompts for each
pending item interactively in the terminal instead of over HTTP.
"""

from __future__ import annotations
import argparse
import json
import os

from dotenv import load_dotenv
load_dotenv()  # picks up a .env file in the cwd, if present -- see .env.example

from langgraph.types import Command

from app.graph.build_graph import get_compiled_graph
from app.observability.tracing import new_run_id, tenant_trace, log_event


def _thread_id(tenant_id: str | None, run_id: str) -> str:
    return f"{tenant_id or 'default'}:{run_id}"


def prompt_for_decision(item: dict) -> dict:
    print("\n" + "-" * 72)
    print(f"REVIEW NEEDED  [{item['source_table']}.{item['source_field']}]")
    print(f"  proposed destination: {item['proposed_destination_field']}")
    print(f"  proposed confidence : {item['proposed_confidence']}")
    print(f"  reasoning           : {item['proposed_reasoning']}")
    print(f"  why flagged         : {item['reason_for_review']}")
    print(f"  other candidates    : {', '.join(item['candidates'])}")
    choice = input("  approve? [Y/n/other destination field/'skip' to leave unmapped]: ").strip()

    if choice.lower() in ("", "y", "yes"):
        return {"source_table": item["source_table"], "source_field": item["source_field"],
                "approved": True, "reviewer": os.environ.get("USER", "cli-user")}
    if choice.lower() in ("n", "no", "skip"):
        return {"source_table": item["source_table"], "source_field": item["source_field"],
                "approved": False, "reviewer": os.environ.get("USER", "cli-user"),
                "comment": "rejected interactively via CLI"}
    return {"source_table": item["source_table"], "source_field": item["source_field"],
            "approved": True, "override_destination_field": choice,
            "reviewer": os.environ.get("USER", "cli-user"), "comment": "overridden interactively via CLI"}


def run(tenant_id: str | None, run_id: str | None, auto_approve: bool, output_dir: str) -> dict:
    graph = get_compiled_graph()
    is_resume = run_id is not None
    run_id = run_id or new_run_id()
    thread_id = _thread_id(tenant_id, run_id)
    config = {"configurable": {"thread_id": thread_id}}

    with tenant_trace(tenant_id, run_id):
        if is_resume:
            snap = graph.get_state(config)
            if not snap.next:
                print(f"Run {run_id} already completed; nothing to resume.")
                result = {"mapping_document": snap.values.get("mapping_document"),
                           "validation_report": snap.values.get("validation_report")}
                return _finish(tenant_id, run_id, result, output_dir)
            items = snap.tasks[0].interrupts[0].value["items"]
            decisions = _collect_decisions(items, auto_approve)
            result = graph.invoke(Command(resume=decisions), config=config)
        else:
            print(f"Starting run {run_id} for tenant={tenant_id or '(default)'}...")
            result = graph.invoke({"tenant_id": tenant_id, "run_id": run_id}, config=config)

        while "__interrupt__" in result:
            items = result["__interrupt__"][0].value["items"]
            print(f"\n{len(items)} field(s) need human review (run_id={run_id}, thread persisted -- "
                  f"safe to Ctrl-C and resume later with --resume {run_id}).")
            decisions = _collect_decisions(items, auto_approve)
            result = graph.invoke(Command(resume=decisions), config=config)

    return _finish(tenant_id, run_id, result, output_dir)


def _collect_decisions(items: list[dict], auto_approve: bool) -> list[dict]:
    if auto_approve:
        return [{"source_table": it["source_table"], "source_field": it["source_field"],
                  "approved": True, "reviewer": "cli-auto-approve"} for it in items]
    return [prompt_for_decision(it) for it in items]


def _finish(tenant_id: str | None, run_id: str, result: dict, output_dir: str) -> dict:
    doc = result["mapping_document"]
    validation = result.get("validation_report", {})

    tenant_dir = os.path.join(output_dir, tenant_id or "default")
    os.makedirs(tenant_dir, exist_ok=True)
    out_path = os.path.join(tenant_dir, f"mapping_{run_id}.json")
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=2)

    total_mapped = sum(len(t["field_mappings"]) for t in doc["tables"])
    total_unmapped = sum(len(t["unmapped_source_fields"]) for t in doc["tables"])
    print(f"\nWrote {out_path}")
    print(f"  {total_mapped} field(s) mapped, {total_unmapped} unmapped, validation ok={validation.get('ok')}")
    log_event("cli.run_complete", tenant_id=tenant_id, run_id=run_id, mapped=total_mapped, unmapped=total_unmapped)
    return doc


def main():
    parser = argparse.ArgumentParser(description="Schema Field Mapper CLI")
    parser.add_argument("--tenant", default=None, help="Tenant id (see app/tenancy/registry.py to seed demo tenants).")
    parser.add_argument("--resume", dest="run_id", default=None, help="Resume a previously paused run_id.")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve all HITL review items instead of prompting.")
    parser.add_argument("--output-dir", default="output", help="Directory to write the mapping JSON under (tenant-scoped subfolder).")
    args = parser.parse_args()

    run(args.tenant, args.run_id, args.auto_approve, args.output_dir)


if __name__ == "__main__":
    main()
