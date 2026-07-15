"""
Graph wiring.

    START -> load_config -> match_tables -> match_fields --+
                                                             |
                        (pending_reviews non-empty?)---------+
                             yes -> human_review -----+
                             no  ------------------- apply_rules -> assemble -> validate -> END

`human_review` is where the graph genuinely suspends via `interrupt()` (see
graph/nodes.py). The checkpointer is what makes resuming possible across
process boundaries -- e.g. the FastAPI process that started the run can
exit, and a later request with the same thread_id (run_id) can resume it,
because the paused state is durably persisted, not held in memory.

Checkpoint storage defaults to a local-disk tmp path for the same reason as
the tenancy DB (see tenancy/registry.py's DB_PATH comment) -- override via
GRAPH_CHECKPOINT_DB for a real deployment.
"""

from __future__ import annotations
import os

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph.state import GraphState
from app.graph.nodes import (
    load_config_node, match_tables_node, match_fields_node,
    route_after_match_fields, human_review_node, apply_rules_node,
    assemble_node, validate_node,
)

CHECKPOINT_DB = os.environ.get("GRAPH_CHECKPOINT_DB", "/tmp/schema_field_mapper/graph_checkpoints.db")


def build_graph() -> StateGraph:
    g = StateGraph(GraphState)
    g.add_node("load_config", load_config_node)
    g.add_node("match_tables", match_tables_node)
    g.add_node("match_fields", match_fields_node)
    g.add_node("human_review", human_review_node)
    g.add_node("apply_rules", apply_rules_node)
    g.add_node("assemble", assemble_node)
    g.add_node("validate", validate_node)

    g.add_edge(START, "load_config")
    g.add_edge("load_config", "match_tables")
    g.add_edge("match_tables", "match_fields")
    g.add_conditional_edges("match_fields", route_after_match_fields, {
        "human_review": "human_review",
        "apply_rules": "apply_rules",
    })
    g.add_edge("human_review", "apply_rules")
    g.add_edge("apply_rules", "assemble")
    g.add_edge("assemble", "validate")
    g.add_edge("validate", END)
    return g


_compiled = None
_saver_cm = None


def get_compiled_graph():
    """
    Lazily compile + checkpoint the graph once per process. Returns the
    compiled, checkpointed graph ready for `.invoke(...)` /
    `.invoke(Command(resume=...), ...)`.
    """
    global _compiled, _saver_cm
    if _compiled is not None:
        return _compiled

    os.makedirs(os.path.dirname(CHECKPOINT_DB), exist_ok=True)
    import sqlite3
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    saver = SqliteSaver(conn)
    _compiled = build_graph().compile(checkpointer=saver)
    return _compiled
