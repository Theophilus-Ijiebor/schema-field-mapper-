"""
Evaluation metrics.

Two families:

  Regression metrics (candidate vs. golden_mapping.json) -- exact-match
  comparisons that catch the pipeline getting *worse* after a code change:
  accuracy, precision/recall on the mapped-vs-unmapped decision, and
  type-transform agreement.

  Quality metrics (candidate alone, no golden needed) -- the RAG-style
  metrics the assignment asked for by name: relevance and faithfulness.
  Each has an LLM-as-judge implementation (via the same ModelProvider
  abstraction used everywhere else, bounded to one field at a time) and a
  deterministic heuristic fallback so `run_eval.py` always produces a full
  report even with zero API credentials.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from app.providers.base import ModelProvider
from app.core.normalize import content_tokens


# ---------------------------------------------------------------------------
# Regression metrics (vs. golden)
# ---------------------------------------------------------------------------

@dataclass
class RegressionReport:
    per_table: dict = field(default_factory=dict)
    overall_accuracy: float = 0.0
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    type_transform_agreement: float = 0.0
    mismatches: list = field(default_factory=list)


def _index_by_table(doc: dict) -> dict:
    return {t["source_table"]: t for t in doc["tables"]}


def score_against_golden(candidate: dict, golden: dict) -> RegressionReport:
    cand_by_table = _index_by_table(candidate)
    gold_by_table = _index_by_table(golden)

    report = RegressionReport()
    total_correct = total_golden_mapped = 0
    tp = fp = fn = 0
    tt_matches = tt_total = 0

    for table_name, gold_table in gold_by_table.items():
        cand_table = cand_by_table.get(table_name, {"field_mappings": [], "unmapped_source_fields": []})
        gold_map = {fm["source_field"]: fm for fm in gold_table["field_mappings"]}
        cand_map = {fm["source_field"]: fm for fm in cand_table["field_mappings"]}
        gold_unmapped = set(gold_table["unmapped_source_fields"])
        cand_unmapped = set(cand_table.get("unmapped_source_fields", []))

        table_correct = 0
        table_mismatches = []
        for src_field, gold_fm in gold_map.items():
            total_golden_mapped += 1
            cand_fm = cand_map.get(src_field)
            if cand_fm and cand_fm["destination_field"] == gold_fm["destination_field"]:
                table_correct += 1
                total_correct += 1
                tp += 1
                tt_total += 1
                if cand_fm["type_transform"] == gold_fm["type_transform"]:
                    tt_matches += 1
            else:
                fn += 1
                table_mismatches.append({
                    "table": table_name, "source_field": src_field,
                    "golden": gold_fm["destination_field"],
                    "candidate": cand_fm["destination_field"] if cand_fm else None,
                })

        # False positives: candidate mapped a field golden says is unmapped.
        for src_field in cand_unmapped & set(gold_map.keys()):
            fp += 1  # candidate *failed* to map something golden mapped -> already counted as fn above via gold_map loop
        for src_field, cand_fm in cand_map.items():
            if src_field in gold_unmapped:
                fp += 1
                table_mismatches.append({
                    "table": table_name, "source_field": src_field,
                    "golden": None, "candidate": cand_fm["destination_field"],
                })

        table_total = len(gold_map)
        report.per_table[table_name] = {
            "accuracy": round(table_correct / table_total, 3) if table_total else 1.0,
            "correct": table_correct,
            "total": table_total,
        }
        report.mismatches.extend(table_mismatches)

    report.overall_accuracy = round(total_correct / total_golden_mapped, 3) if total_golden_mapped else 1.0
    report.overall_precision = round(tp / (tp + fp), 3) if (tp + fp) else 1.0
    report.overall_recall = round(tp / (tp + fn), 3) if (tp + fn) else 1.0
    report.overall_f1 = (
        round(2 * report.overall_precision * report.overall_recall /
              (report.overall_precision + report.overall_recall), 3)
        if (report.overall_precision + report.overall_recall) else 0.0
    )
    report.type_transform_agreement = round(tt_matches / tt_total, 3) if tt_total else 1.0
    return report


# ---------------------------------------------------------------------------
# Quality metrics (relevance, faithfulness) -- LLM-judge with heuristic fallback
# ---------------------------------------------------------------------------

RELEVANCE_JUDGE_PROMPT = (
    "You are grading a database-schema field mapping. Given a source field and "
    "the destination field it was mapped to, judge how RELEVANT the destination "
    "field is as a semantic match for the source field, on a scale of 0.0 (unrelated) "
    "to 1.0 (clearly the correct match). Respond with ONLY JSON: "
    '{"relevance": <0..1 float>, "explanation": "<one sentence>"}'
)

FAITHFULNESS_JUDGE_PROMPT = (
    "You are auditing a one-sentence explanation written to justify a database "
    "field mapping. Judge whether the explanation's claims are FAITHFUL to the "
    "given field metadata -- i.e. it does not fabricate facts not supported by the "
    "source/destination field name, type, or comment. Score 0.0 (fabricated/unsupported) "
    "to 1.0 (fully grounded). Respond with ONLY JSON: "
    '{"faithfulness": <0..1 float>, "explanation": "<one sentence>"}'
)


def relevance_score(provider: ModelProvider, source_field: dict, dest_field: dict) -> tuple[float, str]:
    user = (
        f"Source field: {source_field['name']} ({source_field['sql_type']}) "
        f"comment={source_field.get('comment') or 'none'}\n"
        f"Destination field it was mapped to: {dest_field['path']} ({dest_field['bson_type']}) "
        f"comment={dest_field.get('comment') or 'none'}"
    )
    result = provider.complete_json(RELEVANCE_JUDGE_PROMPT, user)
    if result and "relevance" in result:
        return float(result["relevance"]), result.get("explanation", "")

    # Heuristic fallback: business-token overlap as a relevance proxy.
    src_ct = content_tokens(name_or_path=source_field["name"], comment=source_field.get("comment", ""))
    dst_ct = content_tokens(name_or_path=dest_field["path"], comment=dest_field.get("comment", ""))
    overlap = src_ct & dst_ct
    score = min(1.0, 0.4 + 0.2 * len(overlap)) if overlap else 0.2
    return round(score, 2), f"heuristic: {len(overlap)} shared business token(s) ({sorted(overlap)})"


def faithfulness_score(provider: ModelProvider, field_mapping: dict, source_field: dict,
                        dest_field: dict) -> tuple[float, str]:
    user = (
        f"Source field metadata: {source_field['name']} ({source_field['sql_type']}) "
        f"comment={source_field.get('comment') or 'none'}\n"
        f"Destination field metadata: {dest_field['path']} ({dest_field['bson_type']}) "
        f"comment={dest_field.get('comment') or 'none'}\n"
        f"Explanation to audit: \"{field_mapping['reasoning']}\""
    )
    result = provider.complete_json(FAITHFULNESS_JUDGE_PROMPT, user)
    if result and "faithfulness" in result:
        return float(result["faithfulness"]), result.get("explanation", "")

    # Heuristic fallback: does the reasoning actually reference the real
    # field identifiers, or could it have been written about any field pair?
    reasoning_lower = field_mapping["reasoning"].lower()
    mentions_source = source_field["name"].lower() in reasoning_lower
    mentions_dest = dest_field["path"].split(".")[-1].lower() in reasoning_lower
    if mentions_source and mentions_dest:
        return 1.0, "heuristic: reasoning explicitly names both the source and destination field"
    if mentions_source or mentions_dest:
        return 0.6, "heuristic: reasoning names only one of the two fields"
    return 0.3, "heuristic: reasoning does not name either field explicitly (generic template risk)"


@dataclass
class QualityReport:
    mean_relevance: float
    mean_faithfulness: float
    per_field: list


def score_quality(provider: ModelProvider, candidate: dict, source_field_lookup, dest_field_lookup) -> QualityReport:
    """
    source_field_lookup(table, name) -> dict, dest_field_lookup(collection, path) -> dict
    are injected so this module doesn't import app.core.schemas directly,
    keeping it usable against any candidate doc (including ones built from a
    different schema pair in the future).
    """
    per_field = []
    for table in candidate["tables"]:
        for fm in table["field_mappings"]:
            src = source_field_lookup(table["source_table"], fm["source_field"])
            dst = dest_field_lookup(table["destination_collection"], fm["destination_field"])
            if src is None or dst is None:
                continue
            rel, rel_expl = relevance_score(provider, src, dst)
            faith, faith_expl = faithfulness_score(provider, fm, src, dst)
            per_field.append({
                "table": table["source_table"],
                "source_field": fm["source_field"],
                "destination_field": fm["destination_field"],
                "relevance": round(rel, 2),
                "relevance_explanation": rel_expl,
                "faithfulness": round(faith, 2),
                "faithfulness_explanation": faith_expl,
            })

    mean_rel = round(sum(p["relevance"] for p in per_field) / len(per_field), 3) if per_field else 0.0
    mean_faith = round(sum(p["faithfulness"] for p in per_field) / len(per_field), 3) if per_field else 0.0
    return QualityReport(mean_relevance=mean_rel, mean_faithfulness=mean_faith, per_field=per_field)
