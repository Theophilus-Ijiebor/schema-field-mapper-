"""
Typed domain models shared across the graph, API, and eval layers.

Using Pydantic here (rather than passing raw dicts around, as the v1 MVP
did) buys three things a production system needs: validation at every
boundary (LangGraph state, HTTP responses, eval inputs all go through the
same model), a single source of truth for the output contract the
assignment specifies, and free JSON Schema generation for the API's OpenAPI
docs.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MatchSource(str, Enum):
    LLM = "llm"
    FALLBACK = "fallback"
    STRUCTURAL_RULE = "structural_rule"
    HUMAN_OVERRIDE = "human_override"


class FieldMapping(BaseModel):
    source_field: str
    destination_field: str
    type_transform: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    notes: Optional[str] = None

    # Extra provenance kept internally for observability/eval, not part of
    # the assignment's required output keys -- excluded on serialization
    # unless explicitly requested (see MappingDocument.to_spec_json).
    match_source: MatchSource = MatchSource.FALLBACK
    retrieval_score: Optional[float] = None

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 2)


class UnmappedDestinationField(BaseModel):
    destination_field: str
    reason: str


class TableMapping(BaseModel):
    source_table: str
    destination_collection: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    field_mappings: list[FieldMapping] = Field(default_factory=list)
    unmapped_source_fields: list[str] = Field(default_factory=list)
    unmapped_destination_fields: list[UnmappedDestinationField] = Field(default_factory=list)


class MappingDocument(BaseModel):
    mapping_version: str = "1.0"
    source: str
    destination: str
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    tables: list[TableMapping]

    # Production metadata -- not part of the assignment's literal schema, so
    # it's kept out of `to_spec_json()` but available for the API / audit log.
    tenant_id: Optional[str] = None
    run_id: Optional[str] = None
    model_provider: Optional[str] = None

    def to_spec_json(self) -> dict:
        """
        Exactly the shape the assignment's Expected Output Format specifies:
        mapping_version / source / destination / generated_at / tables[...],
        with each field_mapping limited to source_field / destination_field /
        type_transform / confidence / reasoning / notes.
        """
        return {
            "mapping_version": self.mapping_version,
            "source": self.source,
            "destination": self.destination,
            "generated_at": self.generated_at,
            "tables": [
                {
                    "source_table": t.source_table,
                    "destination_collection": t.destination_collection,
                    "confidence": t.confidence,
                    "reasoning": t.reasoning,
                    "field_mappings": [
                        {
                            "source_field": fm.source_field,
                            "destination_field": fm.destination_field,
                            "type_transform": fm.type_transform,
                            "confidence": fm.confidence,
                            "reasoning": fm.reasoning,
                            "notes": fm.notes,
                        }
                        for fm in t.field_mappings
                    ],
                    "unmapped_source_fields": t.unmapped_source_fields,
                    "unmapped_destination_fields": [
                        u.model_dump() for u in t.unmapped_destination_fields
                    ],
                }
                for t in self.tables
            ],
        }


class ReviewItem(BaseModel):
    """One field-mapping decision routed to a human reviewer via the HITL interrupt."""
    source_table: str
    source_field: str
    proposed_destination_field: Optional[str]
    proposed_confidence: float
    proposed_reasoning: str
    candidates: list[str] = Field(default_factory=list)
    reason_for_review: str


class ReviewDecision(BaseModel):
    source_table: str
    source_field: str
    approved: bool
    override_destination_field: Optional[str] = None
    reviewer: str = "unknown"
    comment: Optional[str] = None
