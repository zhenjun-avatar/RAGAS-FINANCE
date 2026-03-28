"""RAG answering service using LlamaIndex retrieval and DeepSeek generation."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import replace
from typing import Any, AsyncGenerator, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import config
from .langfuse_tracing import TraceContext, tracer
from .llamaindex_retrieval import retrieval_service
from .llm import get_llm
from .node_repository import enqueue_evaluation_job, ensure_schema
from .rag_stage_log import log_rag, rag_request_scope


def _norm_accn(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _node_metadata(node: dict[str, Any]) -> dict[str, Any]:
    metadata = node.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _node_accessions(node: dict[str, Any]) -> tuple[str, ...]:
    meta = _node_metadata(node)
    raw_values = [
        node.get("finance_accns"),
        meta.get("finance_accns"),
        node.get("sec_accession"),
        meta.get("sec_accession"),
    ]
    out: list[str] = []
    for raw in raw_values:
        values = raw if isinstance(raw, list) else [raw]
        for item in values:
            accn = _norm_accn(item)
            if accn and accn not in out:
                out.append(accn)
    return tuple(out)


def _node_filing_key(node: dict[str, Any]) -> str:
    accns = _node_accessions(node)
    if accns:
        return accns[0]
    document_id = node.get("document_id")
    return f"document:{document_id}" if document_id is not None else "unknown"


def _node_signal_score(node: dict[str, Any]) -> float:
    for key in ("rerank_score", "score", "dense_score", "sparse_score", "fusion_score"):
        value = node.get(key)
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return 0.0


def _ranked_filing_distribution(nodes: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for node in nodes:
        filing = _node_filing_key(node)
        totals[filing] += max(0.25, _node_signal_score(node))
        counts[filing] += 1
    ranked = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
    return [
        {
            "filing": filing,
            "weight": round(weight, 4),
            "count": counts.get(filing, 0),
        }
        for filing, weight in ranked[:limit]
    ]


def _sql_filing_distribution(rows: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        filing = _norm_accn(row.get("accn")) or f"document:{row.get('document_id')}"
        totals[filing] += 1.0
        counts[filing] += 1
    ranked = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
    return [
        {
            "filing": filing,
            "weight": round(weight, 4),
            "count": counts.get(filing, 0),
        }
        for filing, weight in ranked[:limit]
    ]


def _merge_ranked_nodes(*node_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for nodes in node_lists:
        for node in nodes:
            node_id = str(node.get("node_id") or "").strip()
            if not node_id:
                continue
            current = merged.get(node_id)
            if current is None or float(node.get("score") or 0.0) > float(current.get("score") or 0.0):
                merged[node_id] = dict(node)
    return sorted(merged.values(), key=lambda item: float(item.get("score") or 0.0), reverse=True)


_NARRATIVE_SQL_METRIC_HINTS: dict[str, tuple[str, ...]] = {
    "liquidity": (
        "Cash",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "AssetsCurrent",
        "LiabilitiesCurrent",
        "WorkingCapital",
    ),
    "capital_resources": (
        "Cash",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "LongTermDebt",
        "ShortTermBorrowings",
        "AdditionalPaidInCapitalCommonStock",
    ),
    "going_concern": (
        "AccumulatedDeficit",
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "NetIncomeLoss",
        "AssetsCurrent",
        "LiabilitiesCurrent",
    ),
    "working_capital": (
        "WorkingCapital",
        "AssetsCurrent",
        "LiabilitiesCurrent",
        "Cash",
    ),
}


def _row_metric_local_name(row: dict[str, Any]) -> str:
    value = str(row.get("metric_key") or "").strip()
    if "." in value:
        value = value.split(".")[-1].strip()
    return value


def _row_text_blob(row: dict[str, Any]) -> str:
    parts = (
        row.get("metric_key"),
        row.get("metric_label"),
        row.get("taxonomy"),
        row.get("form"),
        row.get("frame"),
        row.get("value_text"),
    )
    return " ".join(str(part).strip().lower() for part in parts if str(part or "").strip())


def _preferred_sql_metric_keys(evidence_plan: Any) -> set[str]:
    keys: set[str] = set()
    for target in getattr(evidence_plan, "narrative_targets", ()) or ():
        for key in _NARRATIVE_SQL_METRIC_HINTS.get(str(target).strip(), ()):
            keys.add(key)
    for key in getattr(getattr(evidence_plan, "sql_plan", None), "metric_exact_keys", ()) or ():
        local = str(key).strip()
        if local:
            keys.add(local.split(".")[-1].strip())
    return keys


def _preferred_sql_terms(evidence_plan: Any) -> tuple[str, ...]:
    out: list[str] = []
    for value in getattr(evidence_plan, "term_targets", ()) or ():
        text = str(value).strip().lower()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _rank_sql_rows_for_plan(
    rows: list[dict[str, Any]],
    *,
    evidence_plan: Any,
    preferred_accns: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {
            "sql_primary_filing_preferred": list(preferred_accns or []),
            "sql_preferred_metric_keys": [],
            "sql_rows_kept": 0,
            "sql_rows_filtered_out": 0,
        }
    preferred_accn_set = {
        str(value).strip()
        for value in (preferred_accns or [])
        if str(value).strip()
    }
    preferred_metrics = _preferred_sql_metric_keys(evidence_plan)
    preferred_terms = _preferred_sql_terms(evidence_plan)

    def _score(row: dict[str, Any]) -> float:
        score = 0.0
        accn = _norm_accn(row.get("accn"))
        if accn and accn in preferred_accn_set:
            score += 4.0
        metric_local = _row_metric_local_name(row)
        if metric_local and metric_local in preferred_metrics:
            score += 5.0
        blob = _row_text_blob(row)
        if preferred_terms and any(term in blob for term in preferred_terms):
            score += 2.0
        if row.get("value_numeric") is not None:
            score += 0.5
        return score

    scored = [(row, _score(row), idx) for idx, row in enumerate(rows)]
    kept = [(row, score, idx) for row, score, idx in scored if score > 0]
    ordered = sorted(
        kept if kept else scored,
        key=lambda item: (
            float(item[1]),
            str(item[0].get("period_end") or ""),
            str(item[0].get("filed_date") or ""),
            int(item[0].get("id") or 0),
            -item[2],
        ),
        reverse=True,
    )
    ordered_rows = [row for row, _, _ in ordered]
    return ordered_rows, {
        "sql_primary_filing_preferred": sorted(preferred_accn_set),
        "sql_preferred_metric_keys": sorted(preferred_metrics),
        "sql_rows_kept": len(kept) if kept else len(rows),
        "sql_rows_filtered_out": max(0, len(rows) - len(kept)) if kept else 0,
    }


async def _collect_sql_rows_for_accn(
    *,
    accn: str,
    document_ids: list[int],
    sql_plan: Any,
    query_observations_by_filters: Any,
    query_observations_by_metric_hints: Any,
    query_observations_for_documents: Any,
    exact_limit: int,
    hint_limit: int,
    recent_limit: int,
) -> list[dict[str, Any]]:
    forms = list(sql_plan.form_filters) if sql_plan.form_filters else None
    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    exact_rows = await query_observations_by_filters(
        document_ids,
        metric_keys=list(sql_plan.metric_exact_keys) or None,
        forms=forms,
        period_end_dates=list(sql_plan.period_end_dates) or None,
        period_years=list(sql_plan.period_years) or None,
        accns=[accn],
        limit=max(1, exact_limit),
    )
    for row in exact_rows:
        row_id = int(row["id"])
        if row_id not in seen_ids:
            rows.append(row)
            seen_ids.add(row_id)

    hints = list(sql_plan.metric_sql_hints)
    if hints and len(rows) < exact_limit:
        hint_rows = await query_observations_by_metric_hints(
            document_ids,
            hints,
            accns=[accn],
            limit=max(1, hint_limit),
        )
        for row in hint_rows:
            row_id = int(row["id"])
            if row_id not in seen_ids:
                rows.append(row)
                seen_ids.add(row_id)

    if len(rows) < exact_limit and recent_limit > 0:
        recent_rows = await query_observations_for_documents(
            document_ids,
            limit=max(1, recent_limit),
            forms=forms,
            accns=[accn],
        )
        for row in recent_rows:
            row_id = int(row["id"])
            if row_id not in seen_ids:
                rows.append(row)
                seen_ids.add(row_id)
    return rows


async def _execute_finance_sql_plan(
    *,
    document_ids: list[int],
    evidence_plan: Any,
    accns: list[str] | None = None,
    preferred_accns: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from tools.finance.financial_facts_repository import (
        query_observations_by_filters,
        query_observations_by_metric_hints,
        query_observations_for_documents,
    )
    from tools.finance.finance_query_plan import (
        FINANCE_SQL_EXACT_LIMIT,
        FINANCE_SQL_EXACT_MIN_ROWS,
        FINANCE_SQL_HINT_QUERY_LIMIT,
        FINANCE_SQL_MERGED_CAP,
        FINANCE_SQL_RECENT_LIMIT,
    )

    sql_plan = evidence_plan.sql_plan
    budget = getattr(evidence_plan, "retrieval_budget", None)
    merged_cap = min(
        FINANCE_SQL_MERGED_CAP,
        max(16, int(getattr(budget, "sql_row_budget", FINANCE_SQL_MERGED_CAP))),
    )
    question_mode = str(getattr(evidence_plan, "question_mode", "") or "")
    rag_led = question_mode in {"narrative_only", "mixed_narrative_first"}
    if rag_led:
        # Narrative-led asks should keep SQL as evidence support, not dominate context budget.
        merged_cap = min(merged_cap, 40)
    exact_limit = min(FINANCE_SQL_EXACT_LIMIT, merged_cap)
    hint_limit = min(FINANCE_SQL_HINT_QUERY_LIMIT, merged_cap)
    recent_limit = min(FINANCE_SQL_RECENT_LIMIT, merged_cap)
    if rag_led:
        recent_limit = min(recent_limit, 8)
    hints = list(sql_plan.metric_sql_hints)
    forms = list(sql_plan.form_filters) if sql_plan.form_filters else None
    clean_accns = list(dict.fromkeys(str(v).strip() for v in (accns or []) if str(v).strip())) or None
    clean_preferred_accns = [
        str(v).strip()
        for v in (preferred_accns or [])
        if str(v).strip()
    ] or None

    if rag_led and clean_accns and clean_preferred_accns:
        primary_accns = [accn for accn in clean_preferred_accns if accn in set(clean_accns)]
        secondary_accns = [accn for accn in clean_accns if accn not in set(primary_accns)]
        primary_budget = min(
            merged_cap,
            max(len(primary_accns), int(round(merged_cap * 0.75))),
        )
        secondary_budget = max(0, merged_cap - primary_budget)
        per_primary_limit = max(1, primary_budget // max(1, len(primary_accns)))
        per_secondary_limit = (
            max(1, secondary_budget // max(1, len(secondary_accns)))
            if secondary_accns and secondary_budget > 0
            else 0
        )

        bucketed_rows: list[dict[str, Any]] = []
        bucket_meta: list[dict[str, Any]] = []
        for accn in primary_accns:
            accn_rows = await _collect_sql_rows_for_accn(
                accn=accn,
                document_ids=document_ids,
                sql_plan=sql_plan,
                query_observations_by_filters=query_observations_by_filters,
                query_observations_by_metric_hints=query_observations_by_metric_hints,
                query_observations_for_documents=query_observations_for_documents,
                exact_limit=per_primary_limit,
                hint_limit=max(1, per_primary_limit // 2),
                recent_limit=min(recent_limit, max(0, per_primary_limit // 4)),
            )
            bucketed_rows.extend(accn_rows[:per_primary_limit])
            bucket_meta.append({"accn": accn, "kind": "primary", "rows": min(len(accn_rows), per_primary_limit)})
        for accn in secondary_accns:
            if per_secondary_limit <= 0:
                break
            accn_rows = await _collect_sql_rows_for_accn(
                accn=accn,
                document_ids=document_ids,
                sql_plan=sql_plan,
                query_observations_by_filters=query_observations_by_filters,
                query_observations_by_metric_hints=query_observations_by_metric_hints,
                query_observations_for_documents=query_observations_for_documents,
                exact_limit=per_secondary_limit,
                hint_limit=max(1, per_secondary_limit // 2),
                recent_limit=min(recent_limit, max(0, per_secondary_limit // 4)),
            )
            bucketed_rows.extend(accn_rows[:per_secondary_limit])
            bucket_meta.append({"accn": accn, "kind": "secondary", "rows": min(len(accn_rows), per_secondary_limit)})

        merged, ranking_meta = _rank_sql_rows_for_plan(
            bucketed_rows,
            evidence_plan=evidence_plan,
            preferred_accns=clean_preferred_accns,
        )
        merged = merged[:merged_cap]
        filing_counts: dict[str, int] = defaultdict(int)
        for row in merged:
            accn = _norm_accn(row.get("accn"))
            if accn:
                filing_counts[accn] += 1
        return merged, {
            "sql_strategy": "rag_led_bucketed_supplement",
            "sql_exact_row_count": len(merged),
            "sql_metric_hints": list(sql_plan.metric_sql_hints),
            "sql_hint_row_count": 0,
            "sql_hint_supplement_count": 0,
            "sql_recent_supplement_count": 0,
            "sql_accn_filters": clean_accns,
            "sql_merged_cap": merged_cap,
            "sql_rag_led_budget_cap": rag_led,
            "sql_recent_limit": recent_limit,
            "sql_bucket_mode": True,
            "sql_bucket_rows": bucket_meta,
            "sql_filing_row_counts": dict(filing_counts),
            **ranking_meta,
        }

    exact_rows: list[dict[str, Any]] = []
    if (
        sql_plan.metric_exact_keys
        or forms
        or sql_plan.period_end_dates
        or sql_plan.period_years
        or clean_accns
    ):
        exact_rows = await query_observations_by_filters(
            document_ids,
            metric_keys=list(sql_plan.metric_exact_keys) or None,
            forms=forms,
            period_end_dates=list(sql_plan.period_end_dates) or None,
            period_years=list(sql_plan.period_years) or None,
            accns=clean_accns,
            limit=exact_limit,
        )

    merged = list(exact_rows)
    merged_ids = {int(r["id"]) for r in merged}
    hint_added_count = 0
    recent_added_count = 0

    hint_rows: list[dict[str, Any]] = []
    if hints and len(merged) < FINANCE_SQL_EXACT_MIN_ROWS:
        hint_rows = await query_observations_by_metric_hints(
            document_ids,
            hints,
            accns=clean_accns,
            limit=hint_limit,
        )
        for row in hint_rows:
            row_id = int(row["id"])
            if row_id not in merged_ids and len(merged) < merged_cap:
                merged.append(row)
                merged_ids.add(row_id)
                hint_added_count += 1

    recent_rows: list[dict[str, Any]] = []
    if len(merged) < FINANCE_SQL_EXACT_MIN_ROWS:
        recent_rows = await query_observations_for_documents(
            document_ids,
            limit=recent_limit,
            forms=forms,
            accns=clean_accns,
        )
        for row in recent_rows:
            row_id = int(row["id"])
            if row_id not in merged_ids and len(merged) < merged_cap:
                merged.append(row)
                merged_ids.add(row_id)
                recent_added_count += 1

    ranking_meta: dict[str, Any] = {}
    if rag_led:
        merged, ranking_meta = _rank_sql_rows_for_plan(
            merged,
            evidence_plan=evidence_plan,
            preferred_accns=clean_preferred_accns or clean_accns,
        )
        merged = merged[:merged_cap]

    return merged, {
        "sql_strategy": "structured_first_with_fallback",
        "sql_exact_row_count": len(exact_rows),
        "sql_metric_hints": hints,
        "sql_hint_row_count": len(hint_rows),
        "sql_hint_supplement_count": hint_added_count,
        "sql_recent_supplement_count": recent_added_count,
        "sql_accn_filters": clean_accns,
        "sql_merged_cap": merged_cap,
        "sql_rag_led_budget_cap": rag_led,
        "sql_recent_limit": recent_limit,
        **ranking_meta,
    }


def _derive_controller_narrative_targets(nodes: list[dict[str, Any]]) -> tuple[str, ...]:
    out: list[str] = []
    for node in nodes:
        for key in ("finance_statement", "source_section", "content_type"):
            value = str(node.get(key) or "").strip()
            if not value:
                continue
            if any(token in value.lower() for token in ("management", "liquidity", "risk", "going_concern")):
                if value not in out:
                    out.append(value)
    return tuple(out[:8])


def _primary_modality_for_question_mode(question_mode: str) -> str:
    if question_mode in {"narrative_only", "mixed_narrative_first"}:
        return "rag"
    return "sql"


def _dist_top_filings(items: list[dict[str, Any]], *, limit: int) -> list[str]:
    out: list[str] = []
    for item in items:
        filing = _norm_accn(item.get("filing"))
        if not filing or filing.startswith("document:") or filing in out:
            continue
        out.append(filing)
        if len(out) >= max(1, limit):
            break
    return out


def _coordinated_target_accns(
    evidence_plan: Any,
    *,
    sql_dist: list[dict[str, Any]],
    rag_dist: list[dict[str, Any]],
) -> tuple[str, list[str], list[str], list[str]]:
    max_accns = max(1, int(getattr(evidence_plan.retrieval_budget, "max_second_pass_accns", 3)))
    question_mode = str(getattr(evidence_plan, "question_mode", "") or "")
    primary_modality = _primary_modality_for_question_mode(question_mode)
    primary_dist = rag_dist if primary_modality == "rag" else sql_dist
    secondary_dist = sql_dist if primary_modality == "rag" else rag_dist
    primary_top = _dist_top_filings(primary_dist, limit=max_accns)
    secondary_top = _dist_top_filings(secondary_dist, limit=max_accns)
    merged: list[str] = []
    for filing in [*primary_top, *secondary_top]:
        if filing not in merged:
            merged.append(filing)
        if len(merged) >= max_accns:
            break
    return primary_modality, primary_top, secondary_top, merged


def _dist_weight_map(*dists: list[dict[str, Any]]) -> dict[str, float]:
    weights: dict[str, float] = defaultdict(float)
    for dist in dists:
        for item in dist:
            filing = _norm_accn(item.get("filing"))
            if not filing or filing.startswith("document:"):
                continue
            weights[filing] += float(item.get("weight") or 0.0)
    return weights


def _evidence_coverage(
    evidence_plan: Any,
    sql_rows: list[dict[str, Any]],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    reranked = (retrieval.get("debug") or {}).get("reranked") or []
    ranked_nodes = reranked or retrieval.get("nodes") or []
    has_narrative = any(int(node.get("level") or 0) == 0 for node in ranked_nodes)
    has_structured = bool(sql_rows)
    rag_filing_dist = _ranked_filing_distribution(ranked_nodes)
    sql_filing_dist = _sql_filing_distribution(sql_rows)
    dominant_filing = rag_filing_dist[0]["filing"] if rag_filing_dist else None
    sql_dominant_filing = sql_filing_dist[0]["filing"] if sql_filing_dist else None
    return {
        "has_narrative": has_narrative,
        "has_structured": has_structured,
        "dominant_filing": dominant_filing,
        "sql_dominant_filing": sql_dominant_filing,
        "filing_alignment": bool(
            dominant_filing and sql_dominant_filing and dominant_filing == sql_dominant_filing
        ),
        "narrative_target_count": len(getattr(evidence_plan, "narrative_targets", ()) or ()),
        "sql_row_count": len(sql_rows),
    }


def _should_run_second_pass(evidence_plan: Any, coverage: dict[str, Any], top_accns: list[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not top_accns:
        return False, reasons
    if evidence_plan.evidence_requirements.need_narrative and not coverage["has_narrative"]:
        reasons.append("missing_narrative")
    if evidence_plan.evidence_requirements.need_numeric_fact and not coverage["has_structured"]:
        reasons.append("missing_structured")
    sql_dom = str(coverage.get("sql_dominant_filing") or "").strip()
    rag_dom = str(coverage.get("dominant_filing") or "").strip()
    if sql_dom and rag_dom and sql_dom != rag_dom:
        reasons.append("filing_divergence")
    return bool(reasons), reasons


def _build_second_pass_query(base_query: str, evidence_plan: Any, controller_meta: dict[str, Any]) -> str:
    query = (base_query or "").strip()
    additions: list[str] = []
    target_accns = [str(v).strip() for v in (controller_meta.get("target_accns") or []) if str(v).strip()]
    if target_accns:
        additions.append("Likely filings: " + ", ".join(target_accns[:3]))
    primary_modality = str(controller_meta.get("primary_modality") or "").strip()
    if primary_modality:
        additions.append(f"Coordination mode: {primary_modality}-led")
    narrative_targets = [
        str(v).strip()
        for v in (getattr(evidence_plan, "narrative_targets", ()) or ())
        if str(v).strip()
    ]
    if narrative_targets:
        additions.append("Narrative targets: " + ", ".join(narrative_targets[:4]))
    if not additions:
        return query
    return f"{query}\n\n" + "\n".join(additions)


def _reconcile_evidence_plan(
    evidence_plan: Any,
    *,
    sql_rows: list[dict[str, Any]],
    retrieval: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    dbg = retrieval.get("debug") or {}
    ranked_hits = dbg.get("reranked") or dbg.get("pre_rerank") or []
    sql_dist = _sql_filing_distribution(sql_rows)
    rag_dist = _ranked_filing_distribution(ranked_hits)
    primary_modality, primary_top, secondary_top, target_accns = _coordinated_target_accns(
        evidence_plan,
        sql_dist=sql_dist,
        rag_dist=rag_dist,
    )
    weight_map = _dist_weight_map(sql_dist, rag_dist)
    sql_top_all = set(_dist_top_filings(sql_dist, limit=8))
    rag_top_all = set(_dist_top_filings(rag_dist, limit=8))

    from tools.finance.finance_query_plan import FilingHypothesis

    filing_hypotheses = tuple(
        FilingHypothesis(
            accession=filing,
            weight=round(weight_map.get(filing, 0.0), 4),
            reasons=tuple(
                dict.fromkeys(
                    reason
                    for reason in (
                        primary_modality if filing in primary_top else None,
                        "sql" if filing in sql_top_all else None,
                        "rag" if filing in rag_top_all else None,
                    )
                    if reason
                )
            ),
        )
        for filing in target_accns
    )
    controller_targets = _derive_controller_narrative_targets(ranked_hits)
    narrative_targets = tuple(
        dict.fromkeys([*(getattr(evidence_plan, "narrative_targets", ()) or ()), *controller_targets])
    )
    term_targets = tuple(
        dict.fromkeys(
            [
                *(getattr(evidence_plan, "term_targets", ()) or ()),
                *controller_targets,
                *target_accns[:2],
            ]
        )
    )
    refined_plan = replace(
        evidence_plan,
        filing_hypotheses=filing_hypotheses,
        narrative_targets=narrative_targets[:8],
        term_targets=term_targets[:16],
    )
    coverage = _evidence_coverage(refined_plan, sql_rows, retrieval)
    needs_second_pass, second_pass_reasons = _should_run_second_pass(
        refined_plan,
        coverage,
        target_accns,
    )
    return refined_plan, {
        "question_mode": getattr(refined_plan, "question_mode", None),
        "primary_modality": primary_modality,
        "sql_filing_distribution": sql_dist,
        "rag_filing_distribution": rag_dist,
        "primary_top_filings": primary_top,
        "secondary_top_filings": secondary_top,
        "target_accns": target_accns,
        "merged_filing_hypotheses": [item.to_debug_dict() for item in filing_hypotheses],
        "coverage": coverage,
        "second_pass_reasons": second_pass_reasons,
        "run_second_pass": needs_second_pass,
    }


async def _finance_sql_bundle(
    question: str,
    document_ids: list[int],
) -> tuple[Any, str, list[dict[str, Any]], dict[str, Any], Any | None]:
    """Rule-first (+ optional LLM) routing; fetch SEC observations when need_sql."""
    from tools.finance.financial_facts_repository import document_ids_with_sec_observations
    from tools.finance.finance_query_plan import build_finance_evidence_plan
    from tools.finance.finance_intent import resolve_finance_intent
    from tools.finance.finance_query_plan_llm import build_finance_evidence_plan_llm
    from tools.finance.question_router import FinanceRoute, format_sql_observations_for_prompt

    if not config.finance_sql_routing_enabled:
        return FinanceRoute(False, True, "default", "routing_disabled"), "", [], {}, None

    with_sec = await document_ids_with_sec_observations(document_ids)
    if not with_sec:
        return FinanceRoute(False, True, "default", "no_sec_rows"), "", [], {}, None

    with tracer.span("finance-intent", input_payload={"question": (question or "")[:240]}):
        intent = await resolve_finance_intent(question)
    route = intent.route
    intent_debug = {
        "question_kind": intent.question_kind,
        "sql_prompt_max_rows": intent.sql_prompt_max_rows,
        "effective_sql_prompt_max_rows": intent.effective_sql_prompt_max_rows(),
        "intent_detail": intent.intent_detail,
    }

    if not route.need_sql:
        evidence_plan = build_finance_evidence_plan(question, question_kind=intent.question_kind)
        return route, "", [], {"finance_intent": intent_debug, "evidence_plan": evidence_plan.to_debug_dict()}, evidence_plan

    plan_source = "heuristic"
    evidence_plan = build_finance_evidence_plan(question, question_kind=intent.question_kind)
    if config.finance_llm_sql_planner_enabled:
        try:
            with tracer.span("finance-plan-llm", input_payload={"question": (question or "")[:240]}):
                plan_llm = await build_finance_evidence_plan_llm(
                    question,
                    question_kind=intent.question_kind,
                )
            if plan_llm is not None:
                evidence_plan = plan_llm
                plan_source = "llm"
        except Exception:
            plan_source = "heuristic"
    merged, sql_exec_meta = await _execute_finance_sql_plan(
        document_ids=sorted(with_sec),
        evidence_plan=evidence_plan,
    )
    row_cap = intent.effective_sql_prompt_max_rows()
    row_cap = min(row_cap, int(getattr(evidence_plan.retrieval_budget, "sql_row_budget", row_cap)))
    prompt_rows = min(len(merged), row_cap)
    meta = {
        **sql_exec_meta,
        "finance_query_plan": evidence_plan.sql_plan.to_debug_dict(),
        "evidence_plan": evidence_plan.to_debug_dict(),
        "retrieval_metadata_filters": evidence_plan.to_retrieval_filters(),
        "retrieval_query": evidence_plan.retrieval_query,
        "finance_intent": intent_debug,
        "sql_plan_source": plan_source,
    }
    return (
        route,
        format_sql_observations_for_prompt(merged, max_rows=prompt_rows),
        merged,
        meta,
        evidence_plan,
    )


def _merge_sql_and_rag_context(sql_block: str, rag_block: str) -> str:
    sql_block = (sql_block or "").strip()
    rag_block = (rag_block or "").strip()
    if sql_block and rag_block:
        return (
            "[Structured facts from SEC filings (database; prefer for numbers and dates)]\n"
            f"{sql_block}\n\n"
            "[Retrieved passages]\n"
            f"{rag_block}"
        )
    if sql_block:
        return (
            "[Structured facts from SEC filings (database; prefer for numbers and dates)]\n" + sql_block
        )
    return rag_block


def _build_context(nodes: list[dict[str, Any]], limit: int = 10) -> str:
    parts = []
    for idx, node in enumerate(sorted(nodes, key=lambda item: item.get("score", 0.0), reverse=True)[:limit], start=1):
        parts.append(
            "\n".join(
                [
                    f"[Source {idx}]",
                    f"document_id: {node.get('document_id')}",
                    f"node_id: {node.get('node_id')}",
                    f"node_type: {node.get('node_type')}",
                    f"level: {node.get('level')}",
                    node.get("text", ""),
                ]
            )
        )
    return "\n\n".join(parts)


def _relevance_level(score: float) -> str:
    if score >= 0.55:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def _build_citations(nodes: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    ranked = sorted(nodes, key=lambda item: item.get("score", 0.0), reverse=True)[:limit]
    citations = []
    for item in ranked:
        sc = float(item.get("score") or 0.0)
        doc_id = item.get("document_id")
        if doc_id is None:
            continue
        citations.append(
            {
                "document_id": int(doc_id),
                "section_number": int(item.get("order_index") or 0) + 1,
                "quote": (item.get("text") or "")[:280],
                "relevance_score": sc,
                "relevance_level": _relevance_level(sc),
                "node_id": item.get("node_id"),
            }
        )
    return citations


def _build_pipeline_trace(
    retrieval: dict[str, Any],
    trace_ctx: TraceContext,
    *,
    include_full_debug: bool,
    sql_context_chars: int | None = None,
    rag_context_chars: int | None = None,
) -> dict[str, Any]:
    dbg = retrieval.get("debug") or {}
    summary_dense_hits = dbg.get("summary_dense_hits") or []
    summary_sparse_hits = dbg.get("summary_sparse_hits") or []
    leaf_dense_hits = dbg.get("leaf_dense_hits") or []
    leaf_sparse_hits = dbg.get("leaf_sparse_hits") or []

    def _ids(items: list[dict[str, Any]]) -> set[str]:
        out: set[str] = set()
        for item in items:
            node_id = item.get("node_id")
            if node_id:
                out.add(str(node_id))
        return out

    def _overlap(dense_items: list[dict[str, Any]], sparse_items: list[dict[str, Any]]) -> dict[str, Any]:
        dense_ids = _ids(dense_items)
        sparse_ids = _ids(sparse_items)
        intersection = dense_ids & sparse_ids
        union = dense_ids | sparse_ids
        union_count = len(union)
        jaccard = (len(intersection) / union_count) if union_count else 0.0
        return {
            "intersection_count": len(intersection),
            "union_count": union_count,
            "jaccard": round(jaccard, 4),
            "dense_only_ids": sorted(dense_ids - sparse_ids),
            "sparse_only_ids": sorted(sparse_ids - dense_ids),
        }

    def _id_list(items: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for item in items:
            node_id = item.get("node_id")
            if node_id:
                out.append(str(node_id))
        return out

    def _slim_ranked_hits(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in items:
            node_id = item.get("node_id")
            if not node_id:
                continue
            out.append(
                {
                    "node_id": str(node_id),
                    "document_id": item.get("document_id"),
                    "level": item.get("level"),
                    "title": item.get("title"),
                    "domain": item.get("domain"),
                    "content_type": item.get("content_type"),
                    "source_section": item.get("source_section"),
                    "finance_statement": item.get("finance_statement"),
                    "finance_period": item.get("finance_period"),
                    "dense_score": item.get("dense_score"),
                    "sparse_score": item.get("sparse_score"),
                    "fusion_score": item.get("fusion_score"),
                    "rerank_score": item.get("rerank_score"),
                }
            )
        return out

    leaf_fused_ids = _id_list(dbg.get("leaf_candidates") or [])
    pre_rerank_ids = _id_list(dbg.get("pre_rerank") or [])
    final_ranked_ids = _id_list(dbg.get("reranked") or [])
    added_from_summary_ids = sorted(list(set(pre_rerank_ids) - set(leaf_fused_ids)))
    dropped_by_rerank_ids = sorted(list(set(pre_rerank_ids) - set(final_ranked_ids)))
    pre_rerank_hits = _slim_ranked_hits(dbg.get("pre_rerank") or [])
    final_ranked_hits = _slim_ranked_hits(dbg.get("reranked") or [])

    def _slim_pass(pass_dbg: dict[str, Any]) -> dict[str, Any]:
        return {
            "retrieval_counts": pass_dbg.get("retrieval_counts"),
            "metadata_filters": pass_dbg.get("metadata_filters"),
            "sparse_query_profile": pass_dbg.get("sparse_query_profile"),
            "sparse_query_slots": pass_dbg.get("sparse_query_slots"),
            "fusion_quotas": pass_dbg.get("fusion_quotas"),
            "fused_filing_distribution": pass_dbg.get("fused_filing_distribution"),
        }

    out: dict[str, Any] = {
        "retrieval_counts": dbg.get("retrieval_counts"),
        "finance_route": dbg.get("finance_route"),
        "evidence_plan": dbg.get("evidence_plan"),
        "evidence_controller": dbg.get("evidence_controller"),
        "sql_rag_narrowing": dbg.get("sql_rag_narrowing"),
        "runtime_config": {
            "opensearch_sparse_analyzer_env": os.getenv("OPENSEARCH_SPARSE_ANALYZER"),
            "opensearch_sparse_search_analyzer_env": os.getenv("OPENSEARCH_SPARSE_SEARCH_ANALYZER"),
            "opensearch_sparse_analyzer_effective": config.opensearch_sparse_analyzer,
            "opensearch_sparse_search_analyzer_effective": config.opensearch_sparse_search_analyzer,
            "opensearch_sparse_index": config.opensearch_sparse_index,
            "opensearch_sparse_index_finance": config.opensearch_sparse_index_finance,
            "opensearch_sparse_search_scope": config.opensearch_sparse_search_scope,
            "sparse_backend": config.sparse_backend,
        },
        "retrieval_compare_meta": {
            "sparse_top_k": config.sparse_top_k,
            "dense_top_k": config.dense_top_k,
            "summary_levels": [1, 2],
            "leaf_levels": [0],
            "text_preview_max_chars": config.pipeline_trace_sparse_text_chars,
            "sparse_query_profile": dbg.get("sparse_query_profile"),
            "sparse_query_slots": dbg.get("sparse_query_slots"),
            "counts_meaning": (
                "summary_sparse = sparse 在 level∈{1,2} 上的命中条数（≤sparse_top_k，实际可能更少）；"
                "leaf_sparse = sparse 在 level=0 上的命中条数（同上）。"
                "两路各自独立查询，与 dense 的 summary_dense/leaf_dense 含义对称。"
            ),
        },
        "retrieval_dense_hits": {
            "summary": summary_dense_hits,
            "leaf": leaf_dense_hits,
        },
        "retrieval_sparse_hits": {
            "summary": summary_sparse_hits,
            "leaf": leaf_sparse_hits,
        },
        "retrieval_overlap": {
            "summary": _overlap(summary_dense_hits, summary_sparse_hits),
            "leaf": _overlap(leaf_dense_hits, leaf_sparse_hits),
        },
        "retrieval_rerank_compare": {
            "leaf_fused_ids": leaf_fused_ids,
            "pre_rerank_ids": pre_rerank_ids,
            "final_ranked_ids": final_ranked_ids,
            "added_from_summary_ids": added_from_summary_ids,
            "dropped_by_rerank_ids": dropped_by_rerank_ids,
        },
        "retrieval_rerank_hits": {
            "input": pre_rerank_hits,
            "output": final_ranked_hits,
        },
        "retrieval_filing_distribution": dbg.get("fused_filing_distribution"),
        "retrieval_passes": {
            key: _slim_pass(value)
            for key, value in (dbg.get("retrieval_passes") or {}).items()
            if isinstance(value, dict)
        },
        "second_pass": dbg.get("second_pass"),
        "rerank": dbg.get("rerank"),
        "langfuse": {
            **tracer.diagnostics(),
            "trace_id_this_request": trace_ctx.trace_id or None,
            "trace_span_active": trace_ctx.enabled,
        },
        "langgraph_planner": config.enable_langgraph_planner,
        "prompt_context_breakdown": {
            "sql_context_chars": int(sql_context_chars or 0),
            "rag_context_chars": int(rag_context_chars or 0),
            "combined_context_chars": int((sql_context_chars or 0) + (rag_context_chars or 0)),
        },
        "notes": [
            "rerank.mode=remote_success 且 remote_http_called=true 表示已调用 Bocha HTTP；"
            "skipped_not_configured 表示未配 URL/Key，仅用融合序截断。",
            "summary_sparse/leaf_sparse=0：OpenSearch 时多为稀疏索引无文档（切换 SPARSE_BACKEND 后需对该文档重新 ingest）；"
            "Postgres 时多为全文分词、level 过滤或 search_vector 无命中。",
            "Langfuse：数据发往 diagnostics.export_base_url；请在同一部署的 Web UI → Traces 用 trace_id 搜索。"
            "若 OTEL_SDK_DISABLED=true 或 flush 报错，UI 中可能仍无记录。",
            "retrieval_dense_hits / retrieval_sparse_hits：分别是 dense 与 sparse 的原始命中（summary=level 1–2，leaf=level 0）；"
            "含 text_preview，长度见 retrieval_compare_meta，与 RRF 融合后的顺序可能不同。",
            "evidence_plan / evidence_controller / retrieval_passes：用于观察 tri-modal controller 的首轮计划、"
            "filling 分布、二轮收敛与 quota 决策。",
            "second_pass.reasons 包含 filing_divergence 时，表示 SQL 与 RAG 的主导 filing 不一致，"
            "系统会对 top accessions 做一轮定向 SQL + dense/sparse 收敛，而不是锁单文档。",
            "sql_rag_narrowing：hybrid 且 FINANCE_SQL_NARROW_RAG_ENABLED 时，用 SQL 行的 accn / taxonomy.metric_key 与节点 text（及扁平 metadata）"
            "做子串匹配。FINANCE_SQL_NARROW_RAG_STRICT=true 时池子长度压到 top_k：先尽数命中（按分），不足再用高分未命中回填；"
            "false 时整池命中在前、未命中在后，由后续 [:top_k] 截断。",
        ],
    }
    if include_full_debug:
        out["retrieval_debug_top_keys"] = list(dbg.keys())
    return out


async def answer_question(
    *,
    question: str,
    document_ids: list[int],
    detail_level: str,
    top_k: int,
    trace_name: str = "rag-answer",
    include_pipeline_trace: bool = False,
    include_full_retrieval_debug: bool = False,
    report_locale: str | None = None,
) -> dict[str, Any]:
    await ensure_schema()
    request_started_at = time.perf_counter()
    trace_ctx = tracer.start_request(
        trace_name,
        input_payload={
            "question": question,
            "document_ids": document_ids,
            "detail_level": detail_level,
            "top_k": top_k,
        },
    )
    log_request_id = trace_ctx.trace_id or f"local-{uuid.uuid4().hex[:12]}"
    with rag_request_scope(log_request_id):
        log_rag(
            "ask_start",
            document_ids=document_ids,
            document_count=len(document_ids),
            top_k=top_k,
            detail_level=detail_level,
            question_len=len(question or ""),
            include_pipeline_trace=include_pipeline_trace,
            include_full_retrieval_debug=include_full_retrieval_debug,
            langfuse_trace_id=trace_ctx.trace_id or None,
        )
        try:
            return await _answer_question_body(
                question=question,
                document_ids=document_ids,
                detail_level=detail_level,
                top_k=top_k,
                trace_ctx=trace_ctx,
                request_started_at=request_started_at,
                include_pipeline_trace=include_pipeline_trace,
                include_full_retrieval_debug=include_full_retrieval_debug,
                report_locale=report_locale,
            )
        except Exception as exc:
            log_rag("ask_error", level="error", error=str(exc)[:500])
            tracer.end_request(trace_ctx, error=str(exc), metadata={"document_ids": document_ids})
            raise


async def _answer_question_body(
    *,
    question: str,
    document_ids: list[int],
    detail_level: str,
    top_k: int,
    trace_ctx: TraceContext,
    request_started_at: float,
    include_pipeline_trace: bool,
    include_full_retrieval_debug: bool,
    report_locale: str | None = None,
) -> dict[str, Any]:
    try:
        from tools.finance.product_surface import (
            build_evidence_ui_bundle,
            build_external_evaluation_snapshot,
            select_vertical_scenario,
        )
        from tools.finance.report_locale import LIMITATIONS_NO_EVIDENCE, resolve_report_locale

        route, sql_context_text, sql_rows, sql_bundle_meta, evidence_plan = await _finance_sql_bundle(
            question, document_ids
        )
        retrieve_query = (sql_bundle_meta.get("retrieval_query") or "").strip() or question
        retrieval_metadata_filters = sql_bundle_meta.get("retrieval_metadata_filters") or None

        if config.enable_langgraph_planner and route.need_rag:
            with tracer.span("langgraph-plan-and-retrieve", input_payload={"question": question}) as retrieval_span:
                from .rag_graph import graph_app

                graph_result = await graph_app.ainvoke(
                    {
                        "question": question,
                        "retrieval_query": retrieve_query,
                        "retrieval_metadata_filters": retrieval_metadata_filters,
                        "document_ids": document_ids,
                        "detail_level": detail_level,
                        "top_k": top_k,
                    }
                )
                retrieval = {
                    "nodes": graph_result.get("retrieved_nodes", []),
                    "debug": {"planned_queries": graph_result.get("planned_queries", [])},
                }
                if retrieval_span and hasattr(retrieval_span, "update"):
                    retrieval_span.update(output=retrieval["debug"])
        elif config.enable_langgraph_planner:
            retrieval = {"nodes": [], "debug": {"finance_skipped_graph": True}}
        elif route.need_rag:
            with tracer.span(
                "llamaindex-retrieve",
                input_payload={
                    "question": question,
                    "retrieval_query": retrieve_query,
                    "retrieval_metadata_filters": retrieval_metadata_filters,
                },
            ) as retrieval_span:
                retrieval = await retrieval_service.retrieve(
                    query=retrieve_query,
                    document_ids=document_ids,
                    metadata_filters=retrieval_metadata_filters,
                    evidence_plan=evidence_plan,
                )
                if retrieval_span and hasattr(retrieval_span, "update"):
                    retrieval_span.update(
                        output={
                            "candidate_count": len(retrieval["nodes"]),
                            "events": retrieval.get("debug", {}).get("events", []),
                            "reranked": retrieval.get("debug", {}).get("reranked", []),
                        }
                    )
        else:
            retrieval = {"nodes": [], "debug": {}}

        dbg = retrieval.get("debug") or {}
        first_pass_debug = dict(dbg)
        controller_meta: dict[str, Any] = {}
        if route.need_rag and evidence_plan is not None:
            refined_plan, controller_meta = _reconcile_evidence_plan(
                evidence_plan,
                sql_rows=sql_rows,
                retrieval=retrieval,
            )
            evidence_plan = refined_plan
            sql_bundle_meta["evidence_plan"] = evidence_plan.to_debug_dict()
            dbg["evidence_plan"] = evidence_plan.to_debug_dict()
            dbg["evidence_controller"] = controller_meta
            dbg["retrieval_passes"] = {"first": first_pass_debug}

            if controller_meta.get("run_second_pass") and not config.enable_langgraph_planner:
                second_filters = dict(evidence_plan.to_retrieval_filters())
                if controller_meta.get("target_accns"):
                    second_filters["finance_accns"] = list(controller_meta["target_accns"])
                second_query = _build_second_pass_query(retrieve_query, evidence_plan, controller_meta)
                second_sql_meta: dict[str, Any] | None = None
                if route.need_sql and controller_meta.get("target_accns"):
                    from tools.finance.question_router import format_sql_observations_for_prompt

                    second_sql_rows, second_sql_meta = await _execute_finance_sql_plan(
                        document_ids=document_ids,
                        evidence_plan=evidence_plan,
                        accns=list(controller_meta["target_accns"]),
                        preferred_accns=list(controller_meta.get("primary_top_filings") or []),
                    )
                    merged_sql_rows = {int(row["id"]): row for row in sql_rows}
                    for row in second_sql_rows:
                        merged_sql_rows[int(row["id"])] = row
                    sql_rows = list(merged_sql_rows.values())
                    row_cap = min(
                        len(sql_rows),
                        int((sql_bundle_meta.get("finance_intent") or {}).get("effective_sql_prompt_max_rows") or len(sql_rows)),
                        int(getattr(evidence_plan.retrieval_budget, "sql_row_budget", len(sql_rows))),
                    )
                    sql_context_text = format_sql_observations_for_prompt(sql_rows, max_rows=row_cap)
                second_retrieval = await retrieval_service.retrieve(
                    query=second_query,
                    document_ids=document_ids,
                    metadata_filters=second_filters,
                    evidence_plan=evidence_plan,
                )
                retrieval["nodes"] = _merge_ranked_nodes(retrieval.get("nodes") or [], second_retrieval.get("nodes") or [])
                dbg["retrieval_passes"]["second"] = second_retrieval.get("debug") or {}
                dbg["second_pass"] = {
                    "applied": True,
                    "reasons": list(controller_meta.get("second_pass_reasons") or []),
                    "query": second_query,
                    "metadata_filters": second_filters,
                    "sql_meta": second_sql_meta,
                    "node_count_after_merge": len(retrieval["nodes"]),
                }
            else:
                dbg["second_pass"] = {
                    "applied": False,
                    "reasons": list(controller_meta.get("second_pass_reasons") or []),
                }

        narrow_stats: dict[str, Any] = {"applied": False}
        if (
            config.finance_sql_narrow_rag_enabled
            and route.need_sql
            and route.need_rag
            and sql_rows
            and retrieval.get("nodes")
        ):
            from tools.finance.sql_evidence_narrowing import prioritize_nodes_by_sql_evidence

            retrieval["nodes"], narrow_stats = prioritize_nodes_by_sql_evidence(
                retrieval["nodes"],
                sql_rows,
                strict=config.finance_sql_narrow_rag_strict,
                top_k=max(1, top_k),
            )

        dbg = retrieval.get("debug") or dbg
        dbg["sql_rag_narrowing"] = narrow_stats
        dbg["finance_route"] = {
            "need_sql": route.need_sql,
            "need_rag": route.need_rag,
            "source": route.source,
            "detail": route.detail,
            "sql_row_count": len(sql_rows),
            "skipped_retrieval": not route.need_rag,
            **sql_bundle_meta,
        }
        dbg["retrieval_counts_final"] = {
            "node_count_after_controller": len(retrieval.get("nodes") or []),
            "node_count_returned": min(len(retrieval.get("nodes") or []), max(1, top_k)),
        }
        retrieval["debug"] = dbg

        log_rag(
            "ask_retrieve",
            node_count=len(retrieval["nodes"]),
            retrieval_counts=dbg.get("retrieval_counts"),
            rerank_mode=(dbg.get("rerank") or {}).get("mode"),
            rerank_candidates_in=(dbg.get("rerank") or {}).get("candidates_in"),
            rerank_candidates_out=(dbg.get("rerank") or {}).get("candidates_out"),
        )

        resolved_locale = resolve_report_locale(question, report_locale)
        nodes = retrieval["nodes"][: max(1, top_k)]
        rag_context = _build_context(nodes, limit=max(1, top_k))
        context_text = _merge_sql_and_rag_context(sql_context_text, rag_context)
        sql_context_chars = len(sql_context_text or "")
        rag_context_chars = len(rag_context or "")
        citations = _build_citations(nodes)
        llm = get_llm(model_name=config.default_model, temperature=0.1)
        if resolved_locale == "en":
            system_prompt = (
                "You are a rigorous document QA assistant. You must answer primarily from the retrieved context, "
                "never invent facts unsupported by context, and clearly state uncertainty when evidence is insufficient. "
                "Output language must be English only."
            )
            if sql_context_text.strip():
                system_prompt += (
                    " If the context includes 'Structured facts from SEC filings', prioritize those numeric values, "
                    "filing dates, and table fields over free-text inference."
                )
            user_prompt = (
                f"Question: {question}\n\n"
                f"Detail level: {detail_level}\n\n"
                f"Available context:\n{context_text}\n\n"
                "Output requirements:\n"
                "1. Start with a direct answer.\n"
                "2. Add 3-5 key points only when needed.\n"
                "3. Do not fabricate sources; use only the provided context.\n"
                "4. Write the full answer in English."
            )
        else:
            system_prompt = (
                "你是一个严谨的文档问答助手。必须优先依据提供的检索上下文作答，"
                "不能编造未被上下文支持的事实。若证据不足，明确说明。"
                " 输出语言必须为中文。"
            )
            if sql_context_text.strip():
                system_prompt += (
                    " 若上下文中包含「Structured facts from SEC filings」结构化事实，其中的数值、"
                    "申报日期与表格字段优先于纯文本段落推断。"
                )
            user_prompt = (
                f"问题：{question}\n\n"
                f"回答详细度：{detail_level}\n\n"
                f"可用上下文：\n{context_text}\n\n"
                "输出要求：\n"
                "1. 先给出直接回答。\n"
                "2. 如有必要给出 3-5 条要点。\n"
                "3. 不要伪造来源；只基于已给上下文。\n"
                "4. 全文使用中文回答。"
            )
        with tracer.generation(
            "deepseek-answer",
            input_payload={"question": question, "context": context_text[:3000]},
            model=config.default_model,
        ) as generation_span:
            t_llm = time.perf_counter()
            response = await llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
            answer = response.content if hasattr(response, "content") else str(response)
            usage = getattr(response, "response_metadata", None) or {}
            tok = usage.get("token_usage") if isinstance(usage, dict) else None
            log_rag(
                "generation",
                model=config.default_model,
                latency_ms=llm_ms,
                answer_chars=len(answer or ""),
                prompt_chars=len(user_prompt),
                context_chars=len(context_text),
                sql_context_chars=sql_context_chars,
                rag_context_chars=rag_context_chars,
                token_usage=tok if isinstance(tok, dict) else None,
            )
            if generation_span and hasattr(generation_span, "update"):
                generation_span.update(output={"answer": answer, "citation_count": len(citations)})

        if trace_ctx.trace_id:
            await enqueue_evaluation_job(
                trace_id=trace_ctx.trace_id,
                document_ids=document_ids,
                query=question,
                answer=answer,
                context_json=nodes,
                metadata={"detail_level": detail_level},
            )

        tracer.end_request(
            trace_ctx,
            output_payload={
                "answer_preview": answer[:500],
                "citation_count": len(citations),
            },
            metadata={"document_ids": document_ids},
        )

        pipeline_trace = None
        if include_pipeline_trace:
            pipeline_trace = _build_pipeline_trace(
                retrieval,
                trace_ctx,
                include_full_debug=include_full_retrieval_debug,
                sql_context_chars=sql_context_chars,
                rag_context_chars=rag_context_chars,
            )

        lf = tracer.diagnostics()
        log_rag(
            "ask_end",
            trace_id=trace_ctx.trace_id or None,
            citation_count=len(citations),
            sources_used=len({item["document_id"] for item in citations if item.get("document_id") is not None}),
            langfuse_export_base_url=lf.get("export_base_url"),
            langfuse_last_flush_error=lf.get("last_flush_error"),
            pipeline_trace_attached=bool(include_pipeline_trace),
        )

        has_evidence = bool(citations) or bool(sql_rows)
        latency_ms = round((time.perf_counter() - request_started_at) * 1000, 2)
        limitations = None if has_evidence else LIMITATIONS_NO_EVIDENCE.get(resolved_locale, LIMITATIONS_NO_EVIDENCE["zh"])
        vertical_scenario = select_vertical_scenario(question, evidence_plan=evidence_plan, locale=resolved_locale)
        external_evaluation = build_external_evaluation_snapshot(
            citations=citations,
            sql_rows=sql_rows,
            question_mode=getattr(evidence_plan, "question_mode", None) if evidence_plan is not None else None,
            pipeline_trace=pipeline_trace,
            trace_id=trace_ctx.trace_id,
            latency_ms=latency_ms,
            token_usage=tok if isinstance(tok, dict) else None,
            locale=resolved_locale,
        )
        evidence_ui = build_evidence_ui_bundle(
            question=question,
            answer=answer,
            confidence=min(
                0.95,
                0.35 + len(citations) * 0.08 + (0.06 if sql_rows else 0.0),
            ),
            citations=citations,
            sql_rows=sql_rows,
            limitations=limitations,
            trace_id=trace_ctx.trace_id,
            pipeline_trace=pipeline_trace,
            vertical_scenario=vertical_scenario,
            locale=resolved_locale,
        )
        confidence = min(
            0.95,
            0.35 + len(citations) * 0.08 + (0.06 if sql_rows else 0.0),
        )
        result = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources_used": len({item["document_id"] for item in citations if item.get("document_id") is not None}),
            "citation_count": len(citations),
            "limitations": limitations,
            "trace_id": trace_ctx.trace_id,
            "latency_ms": latency_ms,
            "retrieval_debug": retrieval.get("debug", {}),
            "pipeline_trace": pipeline_trace,
            "vertical_scenario": vertical_scenario,
            "external_evaluation": external_evaluation,
            "evidence_ui": evidence_ui,
            "report_locale": resolved_locale,
        }
        return result
    except Exception:
        raise


async def stream_answer_events(
    *,
    question: str,
    document_ids: list[int],
    detail_level: str,
    top_k: int,
    report_locale: str | None = None,
) -> AsyncGenerator[str, None]:
    yield f"data: {json.dumps({'type': 'stage', 'stage': 'retrieve', 'message': '开始检索'}, ensure_ascii=False)}\n\n"
    result = await answer_question(
        question=question,
        document_ids=document_ids,
        detail_level=detail_level,
        top_k=top_k,
        trace_name="rag-answer-stream",
        report_locale=report_locale,
    )
    payload = {k: v for k, v in result.items() if k != "retrieval_debug"}
    yield f"data: {json.dumps({'type': 'result', 'payload': payload}, ensure_ascii=False)}\n\n"
    yield "data: {\"type\": \"done\"}\n\n"
