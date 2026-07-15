"""
DeepEval-backed relevance & faithfulness metrics.

Uses DeepEval's `GEval` -- a rubric-driven LLM-as-judge metric that first
asks the judge model to generate explicit evaluation steps from the
criteria, then scores against those steps and returns a 0-1 score plus a
written rationale. This replaces `app/evals/metrics.py`'s hand-rolled
single-shot judge prompts *when a live model provider is configured*; the
heuristic functions in `metrics.py` remain the fallback when none is (so
`run_eval.py` never has a hard dependency on live credentials).

Each field mapping becomes one `LLMTestCase`:
  - `input`     -- the source field being mapped (name/type/comment)
  - `actual_output` -- what the pipeline produced: the chosen destination
                        field plus its reasoning sentence (this is the
                        artifact being judged)
  - `context`   -- the ground-truth field metadata for both the source field
                    and the destination field it was mapped to; faithfulness
                    is measured against this, i.e. "does the reasoning only
                    claim things supported by these facts."

Only ONE field's metadata is ever in a given test case's context -- same
one-item-at-a-time discipline as the rest of this pipeline's LLM calls.
"""

from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Optional

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.evals.deepeval_adapter import ProviderBackedDeepEvalModel

RELEVANCE_CRITERIA = (
    "Determine how RELEVANT the destination field named in 'actual_output' is "
    "as a semantic match for the source field described in 'input'. A highly "
    "relevant mapping connects fields that represent the same real-world "
    "concept (e.g. an employee's first name maps to a first-name field), even "
    "if the field names/formats differ. An irrelevant mapping connects fields "
    "that represent different concepts that merely share a data type."
)

FAITHFULNESS_CRITERIA = (
    "Determine whether the reasoning given in 'actual_output' is FAITHFUL to "
    "the field metadata provided in 'context' -- i.e. every claim it makes "
    "about the source or destination field (name, type, meaning, format) is "
    "actually supported by that metadata, with no fabricated or unsupported "
    "claims. Generic reasoning that doesn't reference the actual fields "
    "should score low even if it sounds plausible."
)


def build_metrics(deepeval_model: ProviderBackedDeepEvalModel) -> tuple[GEval, GEval]:
    relevance = GEval(
        name="FieldMappingRelevance",
        criteria=RELEVANCE_CRITERIA,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=deepeval_model,
        threshold=0.5,
        async_mode=False,
    )
    faithfulness = GEval(
        name="FieldMappingFaithfulness",
        criteria=FAITHFULNESS_CRITERIA,
        evaluation_params=[
            LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT,
        ],
        model=deepeval_model,
        threshold=0.5,
        async_mode=False,
    )
    return relevance, faithfulness


def _build_test_case(fm: dict, src: dict, dst: dict) -> LLMTestCase:
    input_text = (
        f"Source field '{src['name']}' (type {src['sql_type']}, comment: "
        f"{src.get('comment') or 'none'}) in a legacy MySQL HR schema."
    )
    actual_output = (
        f"Mapped to destination field '{fm['destination_field']}' (type {dst['bson_type']}). "
        f"Reasoning given: {fm['reasoning']}"
    )
    context = [
        f"Source field metadata: name={src['name']}, type={src['sql_type']}, "
        f"constraints={src.get('constraints') or 'none'}, comment={src.get('comment') or 'none'}",
        f"Destination field metadata: path={dst['path']}, type={dst['bson_type']}, "
        f"comment={dst.get('comment') or 'none'}",
    ]
    return LLMTestCase(input=input_text, actual_output=actual_output, context=context)


@dataclass
class DeepEvalQualityReport:
    mean_relevance: float
    mean_faithfulness: float
    per_field: list = dc_field(default_factory=list)
    judge_model: str = ""


def score_quality_deepeval(deepeval_model: ProviderBackedDeepEvalModel, candidate: dict,
                            source_field_lookup, dest_field_lookup) -> DeepEvalQualityReport:
    relevance_metric, faithfulness_metric = build_metrics(deepeval_model)
    per_field = []

    for table in candidate["tables"]:
        for fm in table["field_mappings"]:
            src = source_field_lookup(table["source_table"], fm["source_field"])
            dst = dest_field_lookup(table["destination_collection"], fm["destination_field"])
            if src is None or dst is None:
                continue

            test_case = _build_test_case(fm, src, dst)
            try:
                relevance_metric.measure(test_case)
                rel_score, rel_reason = relevance_metric.score, relevance_metric.reason
            except Exception as exc:
                rel_score, rel_reason = None, f"DeepEval relevance judge failed: {exc}"

            try:
                faithfulness_metric.measure(test_case)
                faith_score, faith_reason = faithfulness_metric.score, faithfulness_metric.reason
            except Exception as exc:
                faith_score, faith_reason = None, f"DeepEval faithfulness judge failed: {exc}"

            per_field.append({
                "table": table["source_table"],
                "source_field": fm["source_field"],
                "destination_field": fm["destination_field"],
                "relevance": round(rel_score, 3) if rel_score is not None else None,
                "relevance_explanation": rel_reason,
                "faithfulness": round(faith_score, 3) if faith_score is not None else None,
                "faithfulness_explanation": faith_reason,
            })

    scored_rel = [p["relevance"] for p in per_field if p["relevance"] is not None]
    scored_faith = [p["faithfulness"] for p in per_field if p["faithfulness"] is not None]
    return DeepEvalQualityReport(
        mean_relevance=round(sum(scored_rel) / len(scored_rel), 3) if scored_rel else 0.0,
        mean_faithfulness=round(sum(scored_faith) / len(scored_faith), 3) if scored_faith else 0.0,
        per_field=per_field,
        judge_model=deepeval_model.get_model_name(),
    )
