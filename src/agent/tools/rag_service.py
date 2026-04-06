"""RAG answering service using LlamaIndex retrieval and DeepSeek generation."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import replace
from typing import Any, AsyncGenerator, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import config
from .langfuse_tracing import TraceContext, tracer
from .llamaindex_retrieval import retrieval_service
from .llm import get_llm
from .node_repository import enqueue_evaluation_job, ensure_schema
from .rag_stage_log import log_rag, rag_request_scope
from .report_store import save_langfuse_observability_report


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


def _first_accession_per_document(nodes: list[dict[str, Any]]) -> dict[int, str]:
    """Map document_id -> first SEC accession seen on any node (fills gaps when per-node metadata is missing)."""
    doc_to_accn: dict[int, str] = {}
    for node in nodes:
        doc_id = node.get("document_id")
        if doc_id is None:
            continue
        did = int(doc_id)
        if did in doc_to_accn:
            continue
        accns = _node_accessions(node)
        if accns:
            doc_to_accn[did] = accns[0]
    return doc_to_accn


def _ranked_filing_distribution(nodes: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    doc_accn = _first_accession_per_document(nodes)
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for node in nodes:
        accns = _node_accessions(node)
        if accns:
            filing = accns[0]
        else:
            did = node.get("document_id")
            if did is not None:
                di = int(did)
                filing = doc_accn.get(di, f"document:{di}")
            else:
                filing = "unknown"
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
    question: str = "",
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
    from tools.finance.finance_query_plan import _build_narrative_retrieval_query

    req = getattr(evidence_plan, "evidence_requirements", None)
    need_narrative = bool(getattr(req, "need_narrative", False)) if req is not None else False
    narrative_retrieval_query = _build_narrative_retrieval_query(
        question,
        narrative_targets=narrative_targets[:8],
        need_narrative=need_narrative,
    )
    refined_plan = replace(
        evidence_plan,
        filing_hypotheses=filing_hypotheses,
        narrative_targets=narrative_targets[:8],
        term_targets=term_targets[:16],
        narrative_retrieval_query=narrative_retrieval_query
        or str(getattr(evidence_plan, "narrative_retrieval_query", "") or ""),
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
    from tools.finance.finance_filing_resolver import attach_filing_hypotheses
    from tools.finance.finance_query_plan import build_finance_evidence_plan
    from tools.finance.finance_intent import resolve_finance_intent
    from tools.finance.finance_query_plan_llm import build_finance_evidence_plan_llm
    from tools.finance.question_router import FinanceRoute, format_sql_observations_for_prompt

    if not config.finance_sql_routing_enabled:
        plan_source = "heuristic"
        evidence_plan = build_finance_evidence_plan(question, question_kind=None)
        if config.finance_llm_sql_planner_enabled:
            try:
                plan_llm = await build_finance_evidence_plan_llm(question, question_kind=None)
                if plan_llm is not None:
                    evidence_plan = plan_llm
                    plan_source = "llm"
            except Exception:
                plan_source = "heuristic"
        filing_scope_debug: dict[str, Any] | None = None
        try:
            evidence_plan, filing_scope_debug = await attach_filing_hypotheses(
                evidence_plan,
                document_ids=sorted(set(document_ids)),
            )
        except Exception as exc:
            filing_scope_debug = {"error": str(exc)}
        # No SQL rows will exist when routing is off; do not require structured facts for second-pass.
        _req = evidence_plan.evidence_requirements
        evidence_plan = replace(
            evidence_plan,
            evidence_requirements=replace(_req, need_numeric_fact=False),
        )
        meta = {
            "finance_query_plan": evidence_plan.sql_plan.to_debug_dict(),
            "evidence_plan": evidence_plan.to_debug_dict(),
            "retrieval_metadata_filters": evidence_plan.to_retrieval_filters(),
            "retrieval_query": evidence_plan.retrieval_query,
            "narrative_retrieval_query": getattr(evidence_plan, "narrative_retrieval_query", "") or "",
            "sql_plan_source": plan_source,
            "filing_scope_resolver": filing_scope_debug,
            "finance_sql_routing": "disabled",
        }
        return FinanceRoute(False, True, "default", "routing_disabled"), "", [], meta, evidence_plan

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
    filing_scope_debug: dict[str, Any] | None = None
    try:
        # Resolve filings from rag_documents for the *request* corpus. ``with_sec`` is only
        # documents that have sec_financial_observations rows — often a subset or different ids
        # than EDGAR HTML ingest document_ids, which would yield candidate_count=0 incorrectly.
        evidence_plan, filing_scope_debug = await attach_filing_hypotheses(
            evidence_plan,
            document_ids=sorted(set(document_ids)),
        )
    except Exception as exc:
        filing_scope_debug = {"error": str(exc)}
    if not route.need_sql:
        return (
            route,
            "",
            [],
            {
                "finance_intent": intent_debug,
                "finance_query_plan": evidence_plan.sql_plan.to_debug_dict(),
                "evidence_plan": evidence_plan.to_debug_dict(),
                "retrieval_metadata_filters": evidence_plan.to_retrieval_filters(),
                "retrieval_query": evidence_plan.retrieval_query,
                "narrative_retrieval_query": getattr(evidence_plan, "narrative_retrieval_query", "") or "",
                "sql_plan_source": plan_source,
                "filing_scope_resolver": filing_scope_debug,
            },
            evidence_plan,
        )
    if _should_defer_sql_for_rag_first(evidence_plan, route):
        meta = {
            "finance_query_plan": evidence_plan.sql_plan.to_debug_dict(),
            "evidence_plan": evidence_plan.to_debug_dict(),
            "retrieval_metadata_filters": evidence_plan.to_retrieval_filters(),
            "retrieval_query": evidence_plan.retrieval_query,
            "narrative_retrieval_query": getattr(evidence_plan, "narrative_retrieval_query", "") or "",
            "finance_intent": intent_debug,
            "sql_plan_source": plan_source,
            "filing_scope_resolver": filing_scope_debug,
            "sql_execution_deferred": True,
            "mixed_narrative_evidence_order": "rag_before_sql",
        }
        return (
            route,
            "",
            [],
            meta,
            evidence_plan,
        )
    merged, sql_exec_meta = await _execute_finance_sql_plan(
        document_ids=sorted(with_sec),
        evidence_plan=evidence_plan,
    )
    row_cap = intent.effective_sql_prompt_max_rows()
    row_cap = min(row_cap, int(getattr(evidence_plan.retrieval_budget, "sql_row_budget", row_cap)))
    if bool(getattr(getattr(evidence_plan, "evidence_requirements", None), "need_narrative", False)):
        row_cap = min(row_cap, 12)
    prompt_rows = min(len(merged), row_cap)
    meta = {
        **sql_exec_meta,
        "finance_query_plan": evidence_plan.sql_plan.to_debug_dict(),
        "evidence_plan": evidence_plan.to_debug_dict(),
        "retrieval_metadata_filters": evidence_plan.to_retrieval_filters(),
        "retrieval_query": evidence_plan.retrieval_query,
        "narrative_retrieval_query": getattr(evidence_plan, "narrative_retrieval_query", "") or "",
        "finance_intent": intent_debug,
        "sql_plan_source": plan_source,
        "filing_scope_resolver": filing_scope_debug,
    }
    return (
        route,
        format_sql_observations_for_prompt(merged, max_rows=prompt_rows),
        merged,
        meta,
        evidence_plan,
    )


def _rag_metadata_filters(
    evidence_plan: Any | None,
    filters: dict[str, Any] | None,
    *,
    sql_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Narrative passages usually lack finance_metric_* metadata; drop those filters when MD&A text is required.

    When SQL rows are available, extract their top accession numbers and inject
    as ``finance_accns`` so the retrieval layer can run a filing-scoped leaf
    search even in the first pass (before ``_reconcile_evidence_plan``).
    """
    if not isinstance(filters, dict) or not filters:
        return filters if filters is None or isinstance(filters, dict) else None
    if evidence_plan is None:
        return dict(filters)
    req = getattr(evidence_plan, "evidence_requirements", None)
    need_narrative = req is not None and bool(getattr(req, "need_narrative", False))
    drop = frozenset({"finance_metric_exact_keys", "finance_metric_keys"}) if need_narrative else frozenset()
    out = {k: v for k, v in filters.items() if k not in drop}
    if "finance_accns" not in out and need_narrative:
        accns = _top_accessions_from_sql_rows(sql_rows, limit=6)
        if not accns and evidence_plan is not None:
            accns = _top_accessions_from_filing_hypotheses(evidence_plan, limit=6, min_weight=0.40)
        if accns:
            out["finance_accns"] = accns
    return out if out else None


_HARD_FILTER_KEYS_BASE = frozenset(
    {
        "finance_accns",
        "finance_form_base",
        "finance_period",
        "finance_metric_exact_keys",
    }
)
_SOFT_FILTER_KEYS = frozenset(
    {
        "section_role",
        "leaf_role",
        "topic_tags",
        "section_path",
        "finance_statement",
        "finance_metric_keys",
        "content_type",
    }
)
_NARRATIVE_TARGET_SOFT_HINTS: dict[str, dict[str, list[str]]] = {
    "liquidity": {
        "section_role": ["liquidity"],
        "leaf_role": ["liquidity_driver"],
        "topic_tags": ["liquidity"],
        "rerank_terms": ["liquidity", "capital resources", "working capital"],
    },
    "margin_cost_structure": {
        "section_role": ["results_of_operations"],
        "leaf_role": ["results_driver"],
        "topic_tags": ["gross_margin", "cost_of_revenue", "operating_expense"],
        "rerank_terms": ["gross margin", "cost of revenue", "operating expenses"],
    },
    "going_concern": {
        "section_role": ["liquidity"],
        "leaf_role": ["liquidity_driver"],
        "topic_tags": ["liquidity"],
        "rerank_terms": ["going concern", "substantial doubt"],
    },
    "management_discussion": {
        "section_role": ["mda", "results_of_operations"],
        "leaf_role": ["results_driver"],
        "topic_tags": ["revenue", "operating_income"],
        "rerank_terms": ["management discussion and analysis", "results of operations"],
    },
    "risk_factors": {
        "section_role": ["risk_factors"],
        "leaf_role": ["risk_factor_item"],
        "topic_tags": [],
        "rerank_terms": ["risk factors"],
    },
}


def _retrieval_soft_hints(evidence_plan: Any | None) -> dict[str, list[str]]:
    if evidence_plan is None:
        return {}
    raw_targets = getattr(evidence_plan, "narrative_targets", ()) or ()
    out: dict[str, list[str]] = {
        "section_role": [],
        "leaf_role": [],
        "topic_tags": [],
        "rerank_terms": [],
    }
    for target in raw_targets:
        hints = _NARRATIVE_TARGET_SOFT_HINTS.get(str(target or "").strip())
        if not hints:
            continue
        for key, values in hints.items():
            bucket = out.setdefault(key, [])
            for value in values:
                text = str(value or "").strip()
                if text and text not in bucket:
                    bucket.append(text)
    return {k: v for k, v in out.items() if v}


def _normalize_hard_metadata_filters(
    filters: dict[str, Any] | None,
    *,
    need_narrative: bool,
) -> dict[str, list[str]] | None:
    if not isinstance(filters, dict) or not filters:
        return None
    allowed_keys = set(_HARD_FILTER_KEYS_BASE)
    if need_narrative:
        allowed_keys.discard("finance_metric_exact_keys")
    out: dict[str, list[str]] = {}
    for key, raw_values in filters.items():
        if key not in allowed_keys:
            continue
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        clean = [str(v).strip() for v in values if str(v).strip()]
        if clean:
            out[key] = list(dict.fromkeys(clean))
    return out if out else None


def _build_retrieval_filter_policy(
    evidence_plan: Any | None,
    filters: dict[str, Any] | None,
    *,
    sql_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    req = getattr(evidence_plan, "evidence_requirements", None)
    need_narrative = req is not None and bool(getattr(req, "need_narrative", False))
    rewritten = _rag_metadata_filters(evidence_plan, filters, sql_rows=sql_rows)
    hard_filters = _normalize_hard_metadata_filters(rewritten, need_narrative=need_narrative)
    soft_hints = _retrieval_soft_hints(evidence_plan)
    raw_keys = sorted((filters or {}).keys()) if isinstance(filters, dict) else []
    rewritten_keys = sorted((rewritten or {}).keys()) if isinstance(rewritten, dict) else []
    hard_keys = sorted((hard_filters or {}).keys()) if isinstance(hard_filters, dict) else []
    dropped_after_rewrite = [key for key in raw_keys if key not in rewritten_keys]
    soft_candidate_keys = [key for key in rewritten_keys if key in _SOFT_FILTER_KEYS]
    accn_source: str | None = None
    if hard_filters and hard_filters.get("finance_accns"):
        if sql_rows and _top_accessions_from_sql_rows(sql_rows, limit=6):
            accn_source = "sql_rows"
        elif evidence_plan is not None and _top_accessions_from_filing_hypotheses(evidence_plan, limit=6):
            accn_source = "filing_hypotheses"
        else:
            accn_source = "input_filters"
    return {
        "hard_filters": hard_filters,
        "soft_hints": soft_hints,
        "debug": {
            "need_narrative": need_narrative,
            "raw_filters": dict(filters or {}) if isinstance(filters, dict) else None,
            "rewritten_filters": dict(rewritten or {}) if isinstance(rewritten, dict) else None,
            "hard_filters": dict(hard_filters or {}) if isinstance(hard_filters, dict) else None,
            "soft_hints": dict(soft_hints or {}),
            "raw_filter_keys": raw_keys,
            "rewritten_filter_keys": rewritten_keys,
            "hard_filter_keys": hard_keys,
            "dropped_filter_keys": dropped_after_rewrite,
            "soft_candidate_keys": soft_candidate_keys,
            "finance_accns_source": accn_source,
        },
    }


def _top_accessions_from_sql_rows(rows: list[dict[str, Any]] | None, *, limit: int = 6) -> list[str]:
    """Extract the most frequent accession numbers from SQL observation rows."""
    if not rows:
        return []
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        accn = _norm_accn(row.get("accn"))
        if accn:
            counts[accn] += 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return [accn for accn, _ in ranked[:limit]]


def _top_accessions_from_filing_hypotheses(
    evidence_plan: Any,
    *,
    limit: int = 6,
    min_weight: float = 0.0,
) -> list[str]:
    hyps = getattr(evidence_plan, "filing_hypotheses", ()) or ()
    ranked = sorted(
        hyps,
        key=lambda h: (
            -float(getattr(h, "weight", 0) or 0.0),
            str(getattr(h, "accession", "") or ""),
        ),
    )
    out: list[str] = []
    for h in ranked:
        weight = float(getattr(h, "weight", 0) or 0.0)
        if weight < float(min_weight):
            continue
        a = _norm_accn(getattr(h, "accession", None))
        if a and a not in out:
            out.append(a)
        if len(out) >= limit:
            break
    return out


def _top_accessions_from_rag_hits(retrieval: dict[str, Any], *, limit: int = 8) -> list[str]:
    dbg = retrieval.get("debug") or {}
    ranked = dbg.get("reranked") or dbg.get("pre_rerank") or retrieval.get("nodes") or []
    weights: dict[str, float] = defaultdict(float)
    for idx, node in enumerate(ranked):
        for accn in _node_accessions(node):
            weights[accn] += max(0.01, _node_signal_score(node) / (1.0 + idx * 0.05))
    ordered = sorted(weights.items(), key=lambda pair: (-pair[1], pair[0]))
    return [accn for accn, _ in ordered[:limit]]


def _should_defer_sql_for_rag_first(evidence_plan: Any | None, route: Any) -> bool:
    if evidence_plan is None:
        return False
    if str(getattr(evidence_plan, "question_mode", "") or "") != "mixed_narrative_first":
        return False
    if not getattr(route, "need_sql", False) or not getattr(route, "need_rag", False):
        return False
    return str(getattr(config, "mixed_narrative_evidence_order", "sql_before_rag")) == "rag_before_sql"


def _sql_prompt_row_cap(evidence_plan: Any, sql_bundle_meta: dict[str, Any]) -> int:
    intent = sql_bundle_meta.get("finance_intent") or {}
    row_cap = int(intent.get("effective_sql_prompt_max_rows") or 20)
    row_cap = min(row_cap, int(getattr(evidence_plan.retrieval_budget, "sql_row_budget", row_cap)))
    if bool(getattr(getattr(evidence_plan, "evidence_requirements", None), "need_narrative", False)):
        row_cap = min(row_cap, 12)
    return max(1, row_cap)


def _mixed_evidence_merge_instruction(evidence_plan: Any | None) -> str | None:
    if evidence_plan is None:
        return None
    req = getattr(evidence_plan, "evidence_requirements", None)
    if req is None or not (
        bool(getattr(req, "need_narrative", False)) and bool(getattr(req, "need_numeric_fact", False))
    ):
        return None
    targets = getattr(evidence_plan, "narrative_targets", ()) or ()
    if "risk_factors" in targets:
        return (
            "Mixed task: use narrative passages for Item 1A / Risk Factors wording and qualitative claims; "
            "use structured XBRL rows for metrics, units, periods, and accession. Prefer pairing by filing "
            "(accession) when both appear under the same header below."
        )
    return (
        "Mixed task: pair narrative passages with structured facts by filing (accession). "
        "Use passages for qualitative discussion; use structured rows for numbers, dates, and forms."
    )


def _merge_sql_and_rag_context(
    sql_block: str,
    rag_block: str,
    *,
    sql_rows: list[dict[str, Any]] | None = None,
    rag_nodes: list[dict[str, Any]] | None = None,
    merge_instruction: str | None = None,
) -> str:
    sql_block = (sql_block or "").strip()
    rag_block = (rag_block or "").strip()
    note = (merge_instruction or "").strip()
    if sql_rows and rag_nodes and sql_block and rag_block:
        aligned = _build_accession_aligned_context(sql_rows, rag_nodes)
        if aligned:
            body = aligned if not note else f"{note}\n\n{aligned}"
            return body
    if sql_block and rag_block:
        body = (
            "[Structured facts from SEC filings (database; prefer for numbers and dates)]\n"
            f"{sql_block}\n\n"
            "[Retrieved passages]\n"
            f"{rag_block}"
        )
        return body if not note else f"{note}\n\n{body}"
    if sql_block:
        body = "[Structured facts from SEC filings (database; prefer for numbers and dates)]\n" + sql_block
        return body if not note else f"{note}\n\n{body}"
    return rag_block if not note else f"{note}\n\n{rag_block}"


def _build_accession_aligned_context(
    sql_rows: list[dict[str, Any]],
    rag_nodes: list[dict[str, Any]],
) -> str | None:
    """Group SQL facts and narrative passages by accession for easier LLM alignment."""
    from tools.finance.question_router import format_sql_observations_for_prompt

    accn_sql: dict[str, list[dict[str, Any]]] = defaultdict(list)
    other_sql: list[dict[str, Any]] = []
    for row in sql_rows:
        accn = _norm_accn(row.get("accn"))
        if accn:
            accn_sql[accn].append(row)
        else:
            other_sql.append(row)

    accn_rag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    other_rag: list[dict[str, Any]] = []
    for node in rag_nodes:
        node_accns = _node_accessions(node)
        if node_accns:
            accn_rag[node_accns[0]].append(node)
        else:
            other_rag.append(node)

    all_accns = list(dict.fromkeys([*accn_sql.keys(), *accn_rag.keys()]))
    paired_accns = [a for a in all_accns if a in accn_sql and a in accn_rag]
    if not paired_accns:
        return None

    sections: list[str] = []
    for accn in all_accns:
        sql_group = accn_sql.get(accn, [])
        rag_group = accn_rag.get(accn, [])
        if not sql_group and not rag_group:
            continue
        form = ""
        period = ""
        if sql_group:
            form = str(sql_group[0].get("form") or "").strip()
            period = str(sql_group[0].get("period_end") or "").strip()
        header = f"== Filing {accn}"
        if form:
            header += f" ({form}"
            if period:
                header += f", period ending {period}"
            header += ")"
        header += " =="
        parts = [header]
        if rag_group:
            parts.append("[Narrative passages]")
            for idx, node in enumerate(rag_group, 1):
                text = (node.get("text") or "").strip()
                if text:
                    parts.append(f"  Passage {idx}: {text}")
        if sql_group:
            parts.append("[Structured XBRL facts]")
            parts.append(format_sql_observations_for_prompt(sql_group, max_rows=len(sql_group)))
        sections.append("\n".join(parts))

    if other_sql:
        sections.append(
            "[Other structured facts]\n"
            + format_sql_observations_for_prompt(other_sql, max_rows=len(other_sql))
        )
    if other_rag:
        parts = ["[Other retrieved passages]"]
        for idx, node in enumerate(other_rag, 1):
            text = (node.get("text") or "").strip()
            if text:
                parts.append(f"  Passage {idx}: {text}")
        sections.append("\n".join(parts))

    return "\n\n".join(sections)


def _rag_node_relevance_key(item: dict[str, Any]) -> float:
    # rerank_score is the primary signal (query-aligned); fall back to evidence score.
    r = item.get("rerank_score")
    if r is not None:
        return float(r)
    return float(item.get("score") or 0.0)


# Maps narrative_target values → substrings that identify nodes by title.
# Used to guarantee that nodes relevant to the query intent always enter the context window.
_NARRATIVE_TITLE_HINTS: dict[str, tuple[str, ...]] = {
    "liquidity": ("liquidity", "capital resource"),
    "results_of_operations": ("results of operations",),
    "management_discussion": ("management's discussion", "management discussion"),
    "risk_factors": ("risk factor",),
    "segment": ("segment",),
    "revenue": ("revenue", "net sales"),
    "operating": ("operating income", "operating expense"),
}


def _mandatory_ids_for_targets(
    nodes: list[dict[str, Any]],
    narrative_targets: Sequence[str],
) -> set[str]:
    """Return node_ids whose title matches any narrative_target keyword."""
    keywords = [
        kw.lower()
        for t in narrative_targets
        for kw in _NARRATIVE_TITLE_HINTS.get(t, (t.replace("_", " "),))
    ]
    if not keywords:
        return set()
    result: set[str] = set()
    for node in nodes:
        title = (node.get("title") or "").lower()
        if any(kw in title for kw in keywords):
            nid = str(node.get("node_id") or "")
            if nid:
                result.add(nid)
    return result


def _take_by_budget(
    nodes: list[dict[str, Any]],
    k: int,
    char_budget: int = 0,
    mandatory_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select context nodes by character budget with mandatory slot guarantee.

    When char_budget > 0:
      1. Mandatory nodes (matching narrative_targets titles) are inserted first.
      2. Remaining budget is filled greedily by descending rerank_score.
    When char_budget == 0: falls back to fixed top-k by score.
    Always returns at least 1 node.
    """
    ranked = sorted(nodes, key=_rag_node_relevance_key, reverse=True)
    if char_budget <= 0:
        return ranked[: max(1, k)]

    mandatory_ids = mandatory_ids or set()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    chars = 0

    # Pass 1: mandatory slots (preserving rerank order among them)
    for node in ranked:
        nid = str(node.get("node_id") or "")
        if nid not in mandatory_ids or nid in seen:
            continue
        node_chars = len(node.get("text") or "")
        if chars + node_chars <= char_budget or not selected:
            selected.append(node)
            seen.add(nid)
            chars += node_chars

    # Pass 2: fill remaining budget by score
    for node in ranked:
        nid = str(node.get("node_id") or "")
        if nid in seen:
            continue
        node_chars = len(node.get("text") or "")
        if chars + node_chars > char_budget:
            break
        selected.append(node)
        seen.add(nid)
        chars += node_chars

    return selected or ranked[:1]


def _build_context(nodes: list[dict[str, Any]], limit: int = 10) -> str:
    parts = []
    for idx, node in enumerate(
        sorted(nodes, key=_rag_node_relevance_key, reverse=True)[:limit],
        start=1,
    ):
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
    ranked = sorted(nodes, key=_rag_node_relevance_key, reverse=True)[:limit]
    doc_accn = _first_accession_per_document(ranked)
    citations = []
    for item in ranked:
        sc = _rag_node_relevance_key(item)
        doc_id = item.get("document_id")
        if doc_id is None:
            continue
        int_doc = int(doc_id)
        accns = _node_accessions(item)
        accn = accns[0] if accns else doc_accn.get(int_doc)
        citations.append(
            {
                "document_id": int_doc,
                "section_number": int(item.get("order_index") or 0) + 1,
                "quote": (item.get("text") or "")[:280],
                "relevance_score": sc,
                "relevance_level": _relevance_level(sc),
                "node_id": item.get("node_id"),
                **({"accn": accn} if accn else {}),
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
        "scoped_leaf_fused_debug": dbg.get("scoped_leaf_fused_debug"),
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
        "rerank_stage_pre_rerank": dbg.get("rerank_stage_pre_rerank"),
        "rerank_stage_rerank_out": dbg.get("rerank_stage_rerank_out"),
        "rerank_stage_final_ranked": dbg.get("rerank_stage_final_ranked"),
        "retrieval_rerank_hits": {
            "input": pre_rerank_hits,
            "output": final_ranked_hits,
        },
        "retrieval_filing_distribution": dbg.get("fused_filing_distribution"),
        "metadata_filter_policy": dbg.get("metadata_filter_policy") or (dbg.get("finance_route") or {}).get("retrieval_filter_policy"),
        "retrieval_soft_hints": dbg.get("retrieval_soft_hints"),
        "section_policy": dbg.get("section_policy"),
        "answerability": dbg.get("answerability"),
        "coverage_selection": dbg.get("coverage_selection"),
        "post_rerank_selector": dbg.get("post_rerank_selector"),
        "evidence_gate": dbg.get("evidence_gate"),
        "narrative_pool": dbg.get("narrative_pool"),
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
            "section_policy / answerability / evidence_gate / coverage_selection：叙事检索的 question-aware 策略与可回答性统计（来自 retrieval.debug）。",
            "post_rerank_selector：叙事路径在 rerank 之后对 top 结果的近重复剔除与可引用性硬拦（dropped_hard_blocked_ids / dropped_duplicate）。",
            "mixed_narrative_first + sql_before_rag：MIXED_NARRATIVE_SQL_ALIGN_TO_RAG_FILINGS=true 时首轮 RAG 后"
            "用 RAG 顶部 accession 作为 preferred_accns 再拉 SQL（与首轮 SQL accns 取并集），"
            "trace 见 retrieval.debug.sql_aligned_to_rag_filings；rag_before_sql 路径已在 sql_after_rag 对齐，不重复执行。",
        ],
    }
    if include_full_debug:
        out["retrieval_debug_top_keys"] = list(dbg.keys())
    return out


def _build_trace_input_payload(
    *,
    question: str,
    document_ids: list[int],
    detail_level: str,
    top_k: int,
) -> dict[str, Any]:
    return {
        "question_preview": (question or "")[:240],
        "question_len": len(question or ""),
        "document_count": len(document_ids),
        "detail_level": detail_level,
        "top_k": int(top_k),
    }


def _build_trace_metadata_whitelist(
    *,
    route: Any,
    evidence_plan: Any | None,
    retrieval_debug: dict[str, Any],
    sql_rows: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    sources_used: int,
    latency_ms: float,
    token_usage: dict[str, Any] | None,
    has_evidence: bool,
    limitations: str | None,
    report_locale: str,
    confidence: float,
    document_count: int,
    include_pipeline_trace: bool,
) -> dict[str, Any]:
    second_pass = retrieval_debug.get("second_pass") or {}
    evidence_gate = retrieval_debug.get("evidence_gate") or {}
    coverage = retrieval_debug.get("coverage_selection") or {}
    answerability = retrieval_debug.get("answerability") or {}
    prompt_tokens = int((token_usage or {}).get("prompt_tokens") or 0)
    completion_tokens = int((token_usage or {}).get("completion_tokens") or 0)
    total_tokens = int((token_usage or {}).get("total_tokens") or (prompt_tokens + completion_tokens))
    question_mode = str(getattr(evidence_plan, "question_mode", "") or "")
    return {
        "document_count": int(document_count),
        "model": config.default_model,
        "report_locale": report_locale,
        "route_source": str(getattr(route, "source", "") or ""),
        "need_sql": bool(getattr(route, "need_sql", False)),
        "need_rag": bool(getattr(route, "need_rag", False)),
        "question_mode": question_mode or None,
        "sql_row_count": len(sql_rows),
        "citation_count": len(citations),
        "sources_used": int(sources_used),
        "has_evidence": bool(has_evidence),
        "limitations_present": bool(limitations),
        "second_pass_applied": bool(second_pass.get("applied")),
        "second_pass_reasons": list(second_pass.get("reasons") or []),
        "evidence_gate_passed": bool(evidence_gate.get("passed", True)),
        "evidence_gate_missing_reason": str(evidence_gate.get("missing_reason") or "") or None,
        "answerability_passed": int(answerability.get("passed") or 0),
        "coverage_slots": list(coverage.get("slots") or []),
        "latency_ms": round(float(latency_ms), 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "confidence": round(float(confidence), 4),
        "pipeline_trace_attached": bool(include_pipeline_trace),
    }


def _langfuse_usage_details(token_usage: dict[str, Any] | None) -> dict[str, int] | None:
    """Map LangChain ``response_metadata['token_usage']`` to Langfuse ``usage_details`` (ints only)."""
    if not isinstance(token_usage, dict) or not token_usage:
        return None

    def _as_int(val: Any) -> int | None:
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    prompt = _as_int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens"))
    completion = _as_int(token_usage.get("completion_tokens") or token_usage.get("output_tokens"))
    total = _as_int(token_usage.get("total_tokens"))
    out: dict[str, int] = {}
    if prompt is not None:
        out["prompt_tokens"] = max(0, prompt)
    if completion is not None:
        out["completion_tokens"] = max(0, completion)
    if total is not None:
        out["total_tokens"] = max(0, total)
    elif "prompt_tokens" in out and "completion_tokens" in out:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out if out else None


def _limitations_message(
    locale: str,
    missing_reason: str | None,
    *,
    narrative_targets: tuple[str, ...] | list[str] = (),
) -> str | None:
    if not missing_reason:
        return None
    loc = "en" if str(locale).lower() == "en" else "zh"
    target_label = ", ".join(str(t).replace("_", " ").title() for t in narrative_targets[:4]) if narrative_targets else "Narrative"
    messages: dict[str, dict[str, str]] = {
        "missing_substantive_narrative": {
            "en": f"Missing substantive {target_label} evidence; only introductory, disclaimer, or non-explanatory text was retrieved.",
            "zh": f"缺少可用于回答的实质性 {target_label} 叙述证据；当前仅检索到引言、免责声明或非解释性文本。",
        }
    }
    return messages.get(missing_reason, {}).get(loc)


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
        input_payload=_build_trace_input_payload(
            question=question,
            document_ids=document_ids,
            detail_level=detail_level,
            top_k=top_k,
        ),
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
                trace_name=trace_name,
                trace_ctx=trace_ctx,
                request_started_at=request_started_at,
                include_pipeline_trace=include_pipeline_trace,
                include_full_retrieval_debug=include_full_retrieval_debug,
                report_locale=report_locale,
            )
        except Exception as exc:
            log_rag("ask_error", level="error", error=str(exc)[:500])
            tracer.end_request(
                trace_ctx,
                error=str(exc),
                metadata={
                    "document_count": len(document_ids),
                    "detail_level": detail_level,
                    "top_k": int(top_k),
                },
            )
            raise


async def _answer_question_body(
    *,
    question: str,
    document_ids: list[int],
    detail_level: str,
    top_k: int,
    trace_name: str,
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

        document_ids_requested = list(document_ids)
        excluded_ids = config.rag_ask_excluded_document_id_set
        if excluded_ids:
            document_ids = [int(d) for d in document_ids if int(d) not in excluded_ids]

        route, sql_context_text, sql_rows, sql_bundle_meta, evidence_plan = await _finance_sql_bundle(
            question, document_ids
        )
        if excluded_ids:
            sql_bundle_meta = {
                **sql_bundle_meta,
                "rag_ask_excluded_document_ids": sorted(excluded_ids),
                "rag_ask_dropped_document_ids": [d for d in document_ids_requested if int(d) in excluded_ids],
                "document_ids_requested_count": len(document_ids_requested),
                "document_ids_effective_count": len(document_ids),
            }
        structured_retrieval_query = (sql_bundle_meta.get("retrieval_query") or "").strip() or question
        retrieve_query = structured_retrieval_query
        if evidence_plan is not None:
            req0 = getattr(evidence_plan, "evidence_requirements", None)
            if req0 is not None and bool(getattr(req0, "need_narrative", False)):
                nq = str(getattr(evidence_plan, "narrative_retrieval_query", "") or "").strip()
                if nq:
                    retrieve_query = nq
        filter_policy = _build_retrieval_filter_policy(
            evidence_plan,
            sql_bundle_meta.get("retrieval_metadata_filters"),
            sql_rows=sql_rows,
        )
        retrieval_metadata_filters = filter_policy.get("hard_filters")
        retrieval_soft_hints = filter_policy.get("soft_hints") or {}
        sql_bundle_meta["retrieval_filter_policy"] = filter_policy.get("debug")
        if retrieval_metadata_filters is not None:
            sql_bundle_meta["retrieval_metadata_filters"] = retrieval_metadata_filters
        else:
            sql_bundle_meta.pop("retrieval_metadata_filters", None)

        if config.enable_langgraph_planner and route.need_rag:
            with tracer.span("langgraph-plan-and-retrieve", input_payload={"question": question}) as retrieval_span:
                from .rag_graph import graph_app

                graph_result = await graph_app.ainvoke(
                    {
                        "question": question,
                        "retrieval_query": retrieve_query,
                        "retrieval_metadata_filters": retrieval_metadata_filters,
                        "retrieval_soft_hints": retrieval_soft_hints,
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
                    "retrieval_soft_hints": retrieval_soft_hints,
                },
            ) as retrieval_span:
                retrieval = await retrieval_service.retrieve(
                    query=retrieve_query,
                    document_ids=document_ids,
                    metadata_filters=retrieval_metadata_filters,
                    retrieval_soft_hints=retrieval_soft_hints,
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

        if (
            sql_bundle_meta.get("sql_execution_deferred")
            and route.need_sql
            and evidence_plan is not None
        ):
            from tools.finance.question_router import format_sql_observations_for_prompt

            rag_accns = _top_accessions_from_rag_hits(retrieval, limit=8)
            plan_accns = _top_accessions_from_filing_hypotheses(evidence_plan, limit=6)
            accns_union = list(dict.fromkeys([*rag_accns, *plan_accns]))[:10]
            merged_sql, sql_after_meta = await _execute_finance_sql_plan(
                document_ids=document_ids,
                evidence_plan=evidence_plan,
                accns=accns_union or None,
                preferred_accns=rag_accns or None,
            )
            row_cap = _sql_prompt_row_cap(evidence_plan, sql_bundle_meta)
            prompt_rows = min(len(merged_sql), row_cap)
            sql_rows = merged_sql
            sql_context_text = format_sql_observations_for_prompt(merged_sql, max_rows=prompt_rows)
            sql_bundle_meta["sql_execution_deferred"] = False
            sql_bundle_meta["sql_after_rag"] = {
                "preferred_accns_from_rag": rag_accns,
                "accns_from_plan": plan_accns,
                "accns_union": accns_union,
                **sql_after_meta,
            }
            retrieval.setdefault("debug", {})["sql_after_rag"] = sql_bundle_meta["sql_after_rag"]

        elif (
            bool(getattr(config, "mixed_narrative_sql_align_to_rag_filings", True))
            and route.need_sql
            and route.need_rag
            and evidence_plan is not None
            and str(getattr(evidence_plan, "question_mode", "") or "") == "mixed_narrative_first"
            and not sql_bundle_meta.get("sql_after_rag")
            and sql_rows
        ):
            from tools.finance.question_router import format_sql_observations_for_prompt

            rag_accns = _top_accessions_from_rag_hits(retrieval, limit=10)
            if rag_accns:
                prior_accns = _top_accessions_from_sql_rows(sql_rows, limit=8)
                accns_union = list(dict.fromkeys([*rag_accns, *prior_accns]))[:12]
                merged_sql, sql_align_meta = await _execute_finance_sql_plan(
                    document_ids=document_ids,
                    evidence_plan=evidence_plan,
                    accns=accns_union or None,
                    preferred_accns=rag_accns,
                )
                row_cap = _sql_prompt_row_cap(evidence_plan, sql_bundle_meta)
                prompt_rows = min(len(merged_sql), row_cap)
                sql_rows = merged_sql
                sql_context_text = format_sql_observations_for_prompt(merged_sql, max_rows=prompt_rows)
                bundle_align = {
                    "rag_accns": rag_accns,
                    "prior_sql_accns": prior_accns,
                    "accns_union": accns_union,
                    **sql_align_meta,
                }
                sql_bundle_meta["sql_aligned_to_rag_filings"] = bundle_align
                retrieval.setdefault("debug", {})["sql_aligned_to_rag_filings"] = bundle_align

        dbg = retrieval.get("debug") or {}
        first_pass_debug = dict(dbg)
        controller_meta: dict[str, Any] = {}
        if route.need_rag and evidence_plan is not None:
            refined_plan, controller_meta = _reconcile_evidence_plan(
                evidence_plan,
                sql_rows=sql_rows,
                retrieval=retrieval,
                question=question,
            )
            evidence_plan = refined_plan
            sql_bundle_meta["evidence_plan"] = evidence_plan.to_debug_dict()
            dbg["evidence_plan"] = evidence_plan.to_debug_dict()
            dbg["evidence_controller"] = controller_meta
            dbg["retrieval_passes"] = {"first": first_pass_debug}

            if controller_meta.get("run_second_pass") and not config.enable_langgraph_planner:
                second_filter_policy = _build_retrieval_filter_policy(
                    evidence_plan,
                    dict(evidence_plan.to_retrieval_filters()),
                    sql_rows=sql_rows,
                )
                second_filters = dict(second_filter_policy.get("hard_filters") or {})
                second_soft_hints = dict(second_filter_policy.get("soft_hints") or {})
                if controller_meta.get("target_accns"):
                    second_filters["finance_accns"] = list(controller_meta["target_accns"])
                    second_filter_policy["debug"]["hard_filters"] = dict(second_filters)
                    second_filter_policy["debug"]["finance_accns_source"] = "controller_target_accns"
                narrative_base_second = str(getattr(evidence_plan, "narrative_retrieval_query", "") or "").strip()
                second_query = _build_second_pass_query(
                    narrative_base_second or retrieve_query,
                    evidence_plan,
                    controller_meta,
                )
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
                    if bool(getattr(getattr(evidence_plan, "evidence_requirements", None), "need_narrative", False)):
                        row_cap = min(row_cap, 12)
                    sql_context_text = format_sql_observations_for_prompt(sql_rows, max_rows=row_cap)
                second_retrieval = await retrieval_service.retrieve(
                    query=second_query,
                    document_ids=document_ids,
                    metadata_filters=second_filters,
                    retrieval_soft_hints=second_soft_hints,
                    evidence_plan=evidence_plan,
                )
                retrieval["nodes"] = _merge_ranked_nodes(retrieval.get("nodes") or [], second_retrieval.get("nodes") or [])
                dbg["retrieval_passes"]["second"] = second_retrieval.get("debug") or {}
                dbg["second_pass"] = {
                    "applied": True,
                    "reasons": list(controller_meta.get("second_pass_reasons") or []),
                    "query": second_query,
                    "metadata_filters": second_filters,
                    "metadata_filter_policy": second_filter_policy.get("debug"),
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
        all_nodes = retrieval.get("nodes") or []
        narrative_targets = list(getattr(evidence_plan, "narrative_targets", ()) or ())
        mandatory_ids = _mandatory_ids_for_targets(all_nodes, narrative_targets)
        nodes = _take_by_budget(
            all_nodes,
            k=max(1, top_k),
            char_budget=config.context_char_budget,
            mandatory_ids=mandatory_ids,
        )
        rag_context = _build_context(nodes, limit=len(nodes))
        merge_instruction = _mixed_evidence_merge_instruction(evidence_plan)
        context_text = _merge_sql_and_rag_context(
            sql_context_text,
            rag_context,
            sql_rows=sql_rows if route.need_sql else None,
            rag_nodes=nodes if route.need_rag else None,
            merge_instruction=merge_instruction,
        )
        sql_context_chars = len(sql_context_text or "")
        rag_context_chars = len(rag_context or "")
        citations = _build_citations(nodes)
        llm = get_llm(model_name=config.default_model, temperature=0.1)
        if resolved_locale == "en":
            system_prompt = (
                "You are a rigorous document QA assistant. "
                "Answer using only facts that are explicitly stated in the provided context. "
                "Keep your answer concise and grounded; do not add commentary, inferences, or details absent from the context. "
                "If evidence is insufficient, say so. "
                "Quote key figures and conclusions using the original wording from the context as closely as possible; "
                "paraphrase only when the sentence structure requires it. "
                "Do not infer or invent filing names, section titles, or identifiers not explicitly present in the context. "
                "Do not append citation notes such as '(Cited source: Source N)' — state facts directly. "
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
                "2. Add key points only if they are explicitly supported by the context; skip any point you cannot trace to a specific passage.\n"
                "3. Use the original wording for key figures and conclusions; do not rephrase numbers or dates.\n"
                "4. Do not fabricate sources; use only the provided context.\n"
                "5. Write the full answer in English."
            )
        else:
            system_prompt = (
                "你是一个严谨的文档问答助手。"
                "只使用上下文中明确陈述的事实作答，保持简洁，不添加推断或上下文中不存在的细节。"
                "关键数字与结论尽量使用上下文原文措辞，仅在句式确实需要时才转述。"
                "不要推断或捏造上下文中未出现的 filing 名称、章节标题或编号。"
                "不要在答案末尾附加「来源：Source N」之类的引用注释，直接陈述事实即可。"
                "若证据不足，明确说明。输出语言必须为中文。"
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
                "2. 仅在能追溯到上下文具体段落时才补充要点，无法追溯的要点直接省略。\n"
                "3. 关键数字与结论使用原文措辞，不要改写数字或日期。\n"
                "4. 不要伪造来源；只基于已给上下文。\n"
                "5. 全文使用中文回答。"
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
                lf_usage = _langfuse_usage_details(tok if isinstance(tok, dict) else None)
                gen_update: dict[str, Any] = {
                    "output": {
                        "answer_preview": (answer or "")[:500],
                        "citation_count": len(citations),
                    },
                }
                if lf_usage:
                    gen_update["usage_details"] = lf_usage
                generation_span.update(**gen_update)

        if trace_ctx.trace_id:
            await enqueue_evaluation_job(
                trace_id=trace_ctx.trace_id,
                document_ids=document_ids,
                query=question,
                answer=answer,
                context_json=nodes,
                metadata={"detail_level": detail_level},
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
        missing_reason = str((retrieval.get("debug") or {}).get("evidence_gate", {}).get("missing_reason") or "") or None
        limitations = _limitations_message(
            resolved_locale,
            missing_reason,
            narrative_targets=getattr(evidence_plan, "narrative_targets", ()) or (),
        )
        if not limitations and not has_evidence:
            limitations = LIMITATIONS_NO_EVIDENCE.get(resolved_locale, LIMITATIONS_NO_EVIDENCE["zh"])
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
        sources_used = len({item["document_id"] for item in citations if item.get("document_id") is not None})
        trace_metadata = _build_trace_metadata_whitelist(
            route=route,
            evidence_plan=evidence_plan,
            retrieval_debug=retrieval.get("debug", {}),
            sql_rows=sql_rows,
            citations=citations,
            sources_used=sources_used,
            latency_ms=latency_ms,
            token_usage=tok if isinstance(tok, dict) else None,
            has_evidence=has_evidence,
            limitations=limitations,
            report_locale=resolved_locale,
            confidence=confidence,
            document_count=len(document_ids),
            include_pipeline_trace=include_pipeline_trace,
        )
        tracer.end_request(
            trace_ctx,
            output_payload={
                "answer_preview": answer[:500],
                "citation_count": len(citations),
            },
            metadata=trace_metadata,
        )
        try:
            save_langfuse_observability_report(
                trace_id=trace_ctx.trace_id or "",
                trace_name=trace_name,
                request_payload=_build_trace_input_payload(
                    question=question,
                    document_ids=document_ids,
                    detail_level=detail_level,
                    top_k=top_k,
                ),
                output_payload={
                    "answer_preview": answer[:500],
                    "citation_count": len(citations),
                },
                trace_metadata=trace_metadata,
                generation_payload={
                    "model": config.default_model,
                    "latency_ms": llm_ms,
                    "token_usage": tok if isinstance(tok, dict) else {},
                    "answer_chars": len(answer or ""),
                    "prompt_chars": len(user_prompt or ""),
                    "context_chars": len(context_text or ""),
                },
                diagnostics=tracer.diagnostics(),
            )
        except Exception as exc:
            log_rag("langfuse_report_write_error", level="warning", error=str(exc)[:300])
        result = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources_used": sources_used,
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
