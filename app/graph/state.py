"""
Graph state.

A single mutable state object flows through every node. Field mapping is
computed per-table, per-field inside plain Python loops within a node
(rather than as separate graph steps per field) -- with only 34 fields total
across 3 tables, a graph node per field would add checkpointing overhead
without adding clarity. Where a *decision point* actually matters --
low-confidence matches needing a human -- that's where the graph branches
(see build_graph.py's conditional edge into `human_review`).
"""

from __future__ import annotations
from typing import Optional, TypedDict


class TableSkeleton(TypedDict):
    source_table: str
    destination_collection: str
    table_confidence: float
    table_reasoning: str
    field_mappings: list[dict]          # finalized FieldMapping-shaped dicts
    unmapped_source_fields: list[str]
    pending_in_table: list[dict]        # ReviewItem-shaped dicts still awaiting a decision


class GraphState(TypedDict, total=False):
    tenant_id: str
    run_id: str

    # Resolved per-tenant configuration for this run.
    confidence_review_threshold: float
    extra_synonyms: dict
    provider_name_preference: Optional[str]
    resolved_provider_name: str
    embedding_provider_preference: Optional[str]
    resolved_retrieval_backend: str

    # Working data.
    tables: list[TableSkeleton]
    pending_reviews: list[dict]         # flattened ReviewItem dicts across all tables, awaiting HITL
    review_decisions: list[dict]        # ReviewDecision dicts, populated on resume

    # Output.
    mapping_document: dict              # MappingDocument.to_spec_json() result
    validation_report: dict
    node_trace: list[str]               # lightweight execution trace for observability/debugging
