"""LlamaIndex-based retrieval layer with sparse+dense hybrid and context completion."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
import uuid
from collections import Counter, defaultdict
from typing import Any, Optional

from llama_index.core.callbacks import CBEventType, CallbackManager
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from loguru import logger

from core.config import config
from .bocha_reranker import reranker
from .llamaindex_callbacks import RecordingCallbackHandler
from .narrative_multi_rerank import narrative_rerank_subqueries, run_multi_query_rerank
from .narrative_section_policy import resolve_section_policy, NarrativeSectionPolicy
from .node_repository import fetch_children, fetch_leaf_descendants, fetch_neighbors, fetch_nodes, fetch_siblings
from .rag_stage_log import log_rag
from .retrieval_fields import (
    RETRIEVAL_INDEX_KEYWORD_FIELDS,
    RETRIEVAL_INDEX_TEXT_FIELDS,
    infer_finance_leaf_role,
    infer_finance_section_role,
    infer_finance_topic_tags,
)
from .retrieval_backends.sparse_query_profiles import build_sparse_query_plan
from .retrieval_backends.factory import get_dense_backend, get_sparse_backend
from .vectorizer import generate_embedding


_NARROWING_META_KEYS: tuple[str, ...] = (
    "sec_accession",
    "finance_accns",
    "finance_metric_exact_keys",
    "finance_metric_keys",
    "finance_form_base",
    "finance_forms",
    "section_leaf",
    "section_path",
    "section_role",
    "leaf_role",
    "topic_tags",
)


def _evidence_score(rec: dict[str, Any]) -> float:
    """Use rerank / vector / BM25 scores for UI; avoid RRF fusion (~0.01–0.05) as primary."""
    r = rec.get("rerank_score")
    if r is not None:
        return float(r)
    dense = rec.get("dense_score")
    sparse = rec.get("sparse_score")
    best = 0.0
    if dense is not None:
        best = max(best, float(dense))
    if sparse is not None:
        best = max(best, float(sparse))
    if best > 0:
        return best
    fus = rec.get("fusion_score")
    if fus is not None:
        return float(fus)
    return 0.0


def _slim_sparse_stage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pipeline 调试：列出 sparse 检索返回的节点（含截断正文）。"""
    max_text = max(200, int(config.pipeline_trace_sparse_text_chars))
    out: list[dict[str, Any]] = []
    for r in rows:
        title = r.get("title") or ""
        if len(title) > 160:
            title = title[:160] + "…"
        body = r.get("text") or ""
        truncated = len(body) > max_text
        if truncated:
            body_preview = body[:max_text] + "…"
        else:
            body_preview = body
        out.append(
            {
                "node_id": r.get("node_id"),
                "document_id": r.get("document_id"),
                "level": r.get("level"),
                "sparse_score": r.get("sparse_score"),
                "title": title,
                "domain": r.get("domain"),
                "content_type": r.get("content_type"),
                "source_section": r.get("source_section"),
                "section_role": r.get("section_role"),
                "leaf_role": r.get("leaf_role"),
                "topic_tags": r.get("topic_tags"),
                "finance_statement": r.get("finance_statement"),
                "finance_period": r.get("finance_period"),
                "text_preview": body_preview,
                "text_truncated": truncated,
            }
        )
    return out


def _slim_dense_stage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pipeline 调试：列出 dense 检索返回的节点（含截断正文）。"""
    max_text = max(200, int(config.pipeline_trace_sparse_text_chars))
    out: list[dict[str, Any]] = []
    for r in rows:
        title = r.get("title") or ""
        if len(title) > 160:
            title = title[:160] + "…"
        body = r.get("text") or r.get("text_preview") or ""
        truncated = len(body) > max_text
        if truncated:
            body_preview = body[:max_text] + "…"
        else:
            body_preview = body
        out.append(
            {
                "node_id": r.get("node_id"),
                "document_id": r.get("document_id"),
                "level": r.get("level"),
                "dense_score": r.get("dense_score"),
                "title": title,
                "domain": r.get("domain"),
                "content_type": r.get("content_type"),
                "source_section": r.get("source_section"),
                "section_role": r.get("section_role"),
                "leaf_role": r.get("leaf_role"),
                "topic_tags": r.get("topic_tags"),
                "finance_statement": r.get("finance_statement"),
                "finance_period": r.get("finance_period"),
                "text_preview": body_preview,
                "text_truncated": truncated,
            }
        )
    return out


def _dump_pre_rerank_debug(
    *,
    query: str,
    pre_rerank: list[dict[str, Any]],
    metadata_filters: dict[str, list[str]],
    need_narrative: bool,
) -> None:
    try:
        debug_dir = Path(__file__).resolve().parents[1] / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        out_path = debug_dir / f"pre_rerank_pool_{stamp}_{suffix}.json"
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "need_narrative": bool(need_narrative),
            "query": query,
            "metadata_filters": dict(metadata_filters or {}),
            "count": len(pre_rerank),
            "items": pre_rerank,
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("pre_rerank_debug_dump saved: {}", out_path.as_posix())
    except Exception as exc:  # pragma: no cover - best effort debug dump
        logger.warning("pre_rerank_debug_dump failed: {}", exc)


def _row_metadata_as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _plan_debug_dict(evidence_plan: Any) -> dict[str, Any]:
    if evidence_plan is None:
        return {}
    if hasattr(evidence_plan, "to_debug_dict"):
        return dict(evidence_plan.to_debug_dict())
    if isinstance(evidence_plan, dict):
        return dict(evidence_plan)
    return {}


def _row_accessions(row: dict[str, Any]) -> tuple[str, ...]:
    metadata = _row_metadata_as_dict(row.get("metadata"))
    raw_values = [
        row.get("finance_accns"),
        metadata.get("finance_accns"),
        row.get("sec_accession"),
        metadata.get("sec_accession"),
    ]
    out: list[str] = []
    for raw in raw_values:
        values = raw if isinstance(raw, list) else [raw]
        for item in values:
            s = str(item or "").strip()
            if s and s not in out:
                out.append(s)
    return tuple(out)


def _row_filing_key(row: dict[str, Any]) -> str:
    accns = _row_accessions(row)
    if accns:
        return accns[0]
    return f"document:{row.get('document_id')}"


def _candidate_bucket(row: dict[str, Any]) -> str:
    source = str(row.get("candidate_source") or "").strip()
    if source:
        return source
    try:
        level = int(row.get("level") or 0)
    except (TypeError, ValueError):
        level = 0
    return "summary" if level > 0 else "leaf"


def _apply_filing_aware_limit(
    items: list[dict[str, Any]],
    *,
    limit: int,
    per_filing_cap: int = 0,
    bucket_caps: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0 or not items:
        return []
    hard_cap = max(1, int(per_filing_cap or 0)) if per_filing_cap else 0
    soft_cap = max(hard_cap, min(limit, hard_cap * 2)) if hard_cap else 0
    bucket_caps = {k: max(0, int(v)) for k, v in (bucket_caps or {}).items() if int(v) > 0}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def _consume(pool: list[dict[str, Any]], *, filing_cap: int, use_bucket_caps: bool) -> None:
        filing_counts: dict[str, int] = defaultdict(int)
        bucket_counts: dict[str, int] = defaultdict(int)
        for chosen in selected:
            filing_counts[_row_filing_key(chosen)] += 1
            bucket_counts[_candidate_bucket(chosen)] += 1
        for item in pool:
            node_id = str(item.get("node_id") or "")
            if not node_id or node_id in selected_ids or len(selected) >= limit:
                continue
            filing_key = _row_filing_key(item)
            bucket = _candidate_bucket(item)
            if filing_cap and filing_counts[filing_key] >= filing_cap:
                continue
            if use_bucket_caps and bucket_caps and bucket_counts[bucket] >= bucket_caps.get(bucket, limit):
                continue
            selected.append(item)
            selected_ids.add(node_id)
            filing_counts[filing_key] += 1
            bucket_counts[bucket] += 1

    _consume(items, filing_cap=hard_cap, use_bucket_caps=True)
    if len(selected) < limit:
        _consume(items, filing_cap=soft_cap, use_bucket_caps=False)
    if len(selected) < limit:
        _consume(items, filing_cap=0, use_bucket_caps=False)
    return selected[:limit]


def _filing_distribution(items: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[_row_filing_key(item)] += 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return [{"filing": filing, "count": count} for filing, count in ranked[:limit]]


def _is_xbrl_like_text(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return (
        text.startswith("us-gaap.")
        or text.startswith("dei.")
        or (" | unit=" in text and " | val=" in text)
        or ("accn=" in text and "fy=" in text and "fp=" in text)
    )


def _is_narrative_candidate(item: dict[str, Any]) -> bool:
    try:
        level = int(item.get("level") or 0)
    except (TypeError, ValueError):
        level = 0
    if level != 0:
        return False
    text = str(item.get("text") or "").strip()
    if not text:
        return False
    if _is_xbrl_like_text(text):
        return False
    return True


def _ensure_narrative_quota(
    *,
    ranked: list[dict[str, Any]],
    fallback_pool: list[dict[str, Any]],
    limit: int,
    min_narrative: int,
) -> tuple[list[dict[str, Any]], int]:
    if not ranked or limit <= 0 or min_narrative <= 0:
        return ranked[: max(1, limit)], 0
    out = list(ranked[:limit])
    current_narrative = [item for item in out if _is_narrative_candidate(item)]
    if len(current_narrative) >= min_narrative:
        return out, 0
    needed = min_narrative - len(current_narrative)
    existing_ids = {str(item.get("node_id") or "") for item in out}
    narrative_pool = [
        item
        for item in fallback_pool
        if _is_narrative_candidate(item) and str(item.get("node_id") or "") not in existing_ids
    ]
    insertions = 0
    while needed > 0 and narrative_pool and out:
        replacement_idx = None
        for idx in range(len(out) - 1, -1, -1):
            if not _is_narrative_candidate(out[idx]):
                replacement_idx = idx
                break
        if replacement_idx is None:
            break
        out[replacement_idx] = narrative_pool.pop(0)
        needed -= 1
        insertions += 1
    return out[:limit], insertions


def _is_heading_like_text(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    heading_markers = (
        "table of contents",
        "item 2.",
        "management's discussion and analysis",
        "management’s discussion and analysis",
        "references to the company",
        "forward-looking statements",
    )
    if any(marker in text for marker in heading_markers):
        return True
    return len(text) <= 420


def _item_narrative_surface(item: dict[str, Any]) -> str:
    metadata = _row_metadata_as_dict(item.get("metadata"))
    parts = [
        item.get("title"),
        metadata.get("source_section"),
        metadata.get("section_leaf"),
        metadata.get("section_path_text"),
        item.get("text"),
    ]
    return " ".join(str(part).strip() for part in parts if part).strip().lower()


def _item_narrative_body(item: dict[str, Any]) -> str:
    return str(item.get("text") or "").strip().lower()


def _item_section_role(item: dict[str, Any]) -> str:
    metadata = _row_metadata_as_dict(item.get("metadata"))
    value = str(item.get("section_role") or metadata.get("section_role") or "").strip().lower()
    if value:
        return value
    return str(
        infer_finance_section_role(
            title=str(item.get("title") or ""),
            text=str(item.get("text") or ""),
            metadata=metadata,
        )
        or "unknown"
    ).strip().lower()


def _item_topic_tags(item: dict[str, Any]) -> tuple[str, ...]:
    metadata = _row_metadata_as_dict(item.get("metadata"))
    raw = item.get("topic_tags")
    values = raw if isinstance(raw, list) else metadata.get("topic_tags")
    if isinstance(values, list) and values:
        out = [str(v).strip().lower() for v in values if str(v).strip()]
        if out:
            return tuple(dict.fromkeys(out))
    inferred = infer_finance_topic_tags(
        title=str(item.get("title") or ""),
        text=str(item.get("text") or ""),
        metadata=metadata,
    )
    return tuple(inferred)


def _item_leaf_role(item: dict[str, Any]) -> str:
    metadata = _row_metadata_as_dict(item.get("metadata"))
    value = str(item.get("leaf_role") or metadata.get("leaf_role") or "").strip().lower()
    if value:
        return value
    return str(
        infer_finance_leaf_role(
            title=str(item.get("title") or ""),
            text=str(item.get("text") or ""),
            metadata=metadata,
        )
        or "other"
    ).strip().lower()


def _is_boilerplate_narrative_text(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    markers = (
        "table of contents",
        "references to the company",
        "forward-looking statements",
        "cautionary note regarding forward-looking",
        "item 2. management",
        "exhibit",
        "appendix",
        "blank check company",
        "formed for the purpose of effecting a merger",
    )
    return any(marker in text for marker in markers)


_CITATIONABILITY_CROSS_REF_PATTERNS: tuple[str, ...] = (
    "incorporated by reference",
    "as discussed in part i, item 1a",
    "as discussed in part ii, item 1a",
    "under the heading “risk factors”",
    'under the heading "risk factors"',
    "see item 1a",
    "there have been no material changes to the company’s risk factors",
    "there have been no material changes to the company's risk factors",
)


def _nonempty_lines(value: str) -> list[str]:
    return [line.strip() for line in str(value or "").splitlines() if line.strip()]


def _sentence_count(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    hits = re.findall(r"[.!?](?:\s|$)", text)
    if hits:
        return len(hits)
    return sum(1 for line in _nonempty_lines(text) if len(line) >= 60)


def _normalized_selector_text(value: str) -> str:
    text = str(value or "").lower()
    if not text:
        return ""
    text = re.sub(r"[a-z][a-z0-9 .,&'/-]*\|\s*q[1-4][^|\n]*form\s+10-[kq]\s*\|\s*\d+", " ", text)
    text = text.replace("table |", " table ")
    text = re.sub(
        r"\b(first|second|third|fourth|three|six|nine|twelve)\s+(quarter|months?)\b",
        " ",
        text,
    )
    text = re.sub(r"\bq[1-4]\b", " ", text)
    text = re.sub(r"\b20\d{2}\b", " <year> ", text)
    text = re.sub(r"\b\d[\d,.\-$()%]*\b", " <num> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1200]


def _citationability_features(item: dict[str, Any]) -> dict[str, Any]:
    text = str(item.get("text") or "").strip()
    body = _item_narrative_body(item)
    lines = _nonempty_lines(text)
    table_lines = [line for line in lines if line.lower().startswith("table |")]
    sentence_count = _sentence_count(text)
    narrative_chars = sum(len(line) for line in lines if not line.lower().startswith("table |"))
    total_chars = max(len(text), 1)
    narrative_ratio = narrative_chars / total_chars
    table_ratio = (sum(len(line) for line in table_lines) / total_chars) if table_lines else 0.0
    heading_like = _is_heading_like_text(text)
    cross_ref_only = (
        any(pattern in body for pattern in _CITATIONABILITY_CROSS_REF_PATTERNS)
        and _explanation_signal_count(text) == 0
        and sentence_count <= 2
    )
    flags: list[str] = []
    score = 0.0
    hard_blocked = False

    if "table of contents" in body:
        flags.append("toc")
        hard_blocked = True
    if len(text) < 120 and sentence_count < 2:
        flags.append("too_short")
        hard_blocked = True
    if heading_like and sentence_count == 0 and len(lines) <= 4:
        flags.append("heading_only")
        hard_blocked = True
    if not hard_blocked:
        if len(text) >= 220:
            score += 0.8
        if sentence_count >= 2:
            score += 0.8
        if narrative_ratio >= 0.6:
            score += 0.8
        if sentence_count == 0:
            score -= 1.0
            flags.append("no_sentence")
        if table_ratio >= 0.75:
            score -= 0.8
            flags.append("table_heavy")
        if cross_ref_only:
            score -= 1.0
            flags.append("cross_ref_only")
        elif any(pattern in body for pattern in _CITATIONABILITY_CROSS_REF_PATTERNS):
            score -= 0.4
            flags.append("cross_ref")
        if heading_like and len(text) < 500:
            score -= 0.4
            flags.append("heading_like")

    return {
        "citationability_score": round(score, 4),
        "citationability_hard_blocked": hard_blocked,
        "citationability_flags": flags,
        "citationability_sentence_count": sentence_count,
        "citationability_table_ratio": round(table_ratio, 4),
        "citationability_norm_text": _normalized_selector_text(text),
    }


def _near_duplicate_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_text = str(left.get("citationability_norm_text") or "")
    right_text = str(right.get("citationability_norm_text") or "")
    if not left_text or not right_text:
        return 0.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def _near_duplicate_threshold(item: dict[str, Any]) -> float:
    table_ratio = float(item.get("citationability_table_ratio") or 0.0)
    base = max(0.7, float(config.narrative_dedupe_similarity_threshold))
    if table_ratio >= 0.6:
        return max(base, 0.9)
    return base


def _is_near_duplicate_of_selected(
    item: dict[str, Any],
    selected: list[dict[str, Any]],
) -> tuple[bool, str | None, float]:
    if not selected:
        return False, None, 0.0
    parent_id = str(item.get("parent_id") or "").strip()
    for chosen in selected:
        chosen_parent = str(chosen.get("parent_id") or "").strip()
        if parent_id and chosen_parent and parent_id == chosen_parent:
            return True, str(chosen.get("node_id") or "").strip() or None, 1.0
        sim = _near_duplicate_similarity(item, chosen)
        if sim >= max(_near_duplicate_threshold(item), _near_duplicate_threshold(chosen)):
            return True, str(chosen.get("node_id") or "").strip() or None, sim
    return False, None, 0.0


def _apply_narrative_post_rerank_selector(
    *,
    items: list[dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not items or limit <= 0 or not bool(config.narrative_post_rerank_selector_enabled):
        return items[: max(1, limit)], {"applied": False}
    enriched = [{**item, **_citationability_features(item)} for item in items]
    selected: list[dict[str, Any]] = []
    duplicate_backfill: list[dict[str, Any]] = []
    dropped_hard: list[str] = []
    dropped_duplicate: list[dict[str, Any]] = []

    for item in enriched:
        node_id = str(item.get("node_id") or "").strip()
        if not node_id:
            continue
        if (
            bool(config.narrative_citationability_filter_enabled)
            and bool(item.get("citationability_hard_blocked"))
        ):
            dropped_hard.append(node_id)
            continue
        if bool(config.narrative_dedupe_enabled):
            is_dup, dup_of, sim = _is_near_duplicate_of_selected(item, selected)
            if is_dup:
                duplicate_backfill.append(item)
                dropped_duplicate.append(
                    {
                        "node_id": node_id,
                        "duplicate_of": dup_of,
                        "similarity": round(sim, 4),
                    }
                )
                continue
        selected.append(item)
        if len(selected) >= max(1, limit):
            continue

    if len(selected) < max(1, limit):
        for item in duplicate_backfill:
            node_id = str(item.get("node_id") or "").strip()
            if not node_id or any(str(s.get("node_id") or "").strip() == node_id for s in selected):
                continue
            selected.append(item)
            if len(selected) >= max(1, limit):
                break

    return selected[: max(1, limit)], {
        "applied": True,
        "input": len(items),
        "selected": len(selected[: max(1, limit)]),
        "dropped_hard_blocked_ids": dropped_hard,
        "dropped_duplicate": dropped_duplicate,
    }


def _is_substantive_narrative_candidate(
    item: dict[str, Any],
    policy: NarrativeSectionPolicy | None = None,
) -> bool:
    """Any non-boilerplate narrative leaf with sufficient content."""
    if not _is_narrative_candidate(item):
        return False
    body = _item_narrative_body(item)
    if not body or _is_boilerplate_narrative_text(body):
        return False
    return True


def _preferred_narrative_tags(plan_debug: dict[str, Any]) -> tuple[str, ...]:
    tags: list[str] = []
    mapping = {
        "revenues": "revenue",
        "revenue": "revenue",
        "costofrevenue": "cost_of_revenue",
        "cost of revenue": "cost_of_revenue",
        "grossprofit": "gross_profit",
        "gross profit": "gross_profit",
        "grossmargin": "gross_margin",
        "gross margin": "gross_margin",
        "operatingincomeloss": "operating_income",
        "operating income": "operating_income",
        "operating loss": "operating_income",
        "operating results": "operating_income",
    }
    for value in plan_debug.get("term_targets") or ():
        key = str(value or "").strip().lower()
        tag = mapping.get(key)
        if tag and tag not in tags:
            tags.append(tag)
    for value in (((plan_debug.get("sql_plan") or {}).get("metric_exact_keys")) or ()):
        key = str(value or "").split(".")[-1].strip().lower()
        tag = mapping.get(key)
        if tag and tag not in tags:
            tags.append(tag)
    return tuple(tags)


def _preferred_accns(plan_debug: dict[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    for item in plan_debug.get("filing_hypotheses") or []:
        accn = str((item or {}).get("accession") or "").strip()
        if accn and accn not in out:
            out.append(accn)
    return tuple(out)


def _explanation_signal_count(text: str) -> int:
    lowered = str(text or "").lower()
    signals = (
        "due to",
        "primarily",
        "mainly",
        "driven by",
        "resulted from",
        "because",
        "increase in",
        "decrease in",
        "offset by",
    )
    return sum(1 for sig in signals if sig in lowered)


def _is_driver_priority_candidate(
    item: dict[str, Any],
    policy: NarrativeSectionPolicy | None = None,
) -> bool:
    """Non-boilerplate narrative leaf with explanation signals — reranker will score quality."""
    if not _is_narrative_candidate(item):
        return False
    if str(item.get("candidate_source") or "").strip() in {"summary", "summary_child"}:
        return False
    body = _item_narrative_body(item)
    if not body or _is_boilerplate_narrative_text(body):
        return False
    return _explanation_signal_count(body) > 0


def _answerability_features(
    item: dict[str, Any],
    *,
    plan_debug: dict[str, Any],
    policy: NarrativeSectionPolicy | None = None,
) -> dict[str, Any]:
    """Filter narrative candidates: hard-block garbage, pass everything else to rerank_score ordering.

    Principle: rerank_score is the sole relevance signal. This function only decides
    whether a node is structurally citable (not boilerplate, not heading-only, not too short).
    Sorting and selection are done purely by rerank_score downstream.
    """
    text = str(item.get("text") or "")
    body = _item_narrative_body(item)
    is_boilerplate = _is_boilerplate_narrative_text(body)
    is_short = len(text.strip()) < 120
    citationability = _citationability_features(item)
    rerank_score = float(item.get("rerank_score") or 0.0)
    disable_answerability_gate = bool(config.narrative_disable_answerability_gate)

    hard_blocked = (
        is_boilerplate
        or is_short
        or bool(citationability["citationability_hard_blocked"])
    )

    passed = (not hard_blocked and rerank_score >= 0.10) or (
        disable_answerability_gate and not hard_blocked
    )

    return {
        "hard_blocked": hard_blocked,
        "answerability_pass": passed,
        "answerability_score": round(rerank_score, 4),  # kept for trace schema compat
        "results_ops_fallback_pass": False,              # kept for report schema compat
        "citationability_hard_blocked": citationability["citationability_hard_blocked"],
        "citationability_flags": citationability["citationability_flags"],
        "citationability_sentence_count": citationability["citationability_sentence_count"],
        "citationability_table_ratio": citationability["citationability_table_ratio"],
        "citationability_norm_text": citationability["citationability_norm_text"],
    }


def _apply_answerability_scores(
    *,
    items: list[dict[str, Any]],
    plan_debug: dict[str, Any],
    policy: NarrativeSectionPolicy | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Annotate items with hard_block / answerability_pass, then sort by rerank_score."""
    annotated: list[dict[str, Any]] = []
    passed = 0
    for item in items:
        features = _answerability_features(item, plan_debug=plan_debug, policy=policy)
        enriched = {**item, **features}
        if features["answerability_pass"]:
            passed += 1
        annotated.append(enriched)
    annotated.sort(
        key=lambda item: (
            float(item.get("rerank_score") or 0.0),
            float(item.get("dense_score") or 0.0),
            float(item.get("sparse_score") or 0.0),
        ),
        reverse=True,
    )
    hard_blocked = sum(1 for item in annotated if bool(item.get("hard_blocked")))
    n = len(annotated)
    return annotated, {
        "scored": n,
        "passed": passed,
        "hard_blocked": hard_blocked,
        "pass_rate": round(passed / n, 4) if n else 0.0,
    }


def _select_narrative_coverage(
    *,
    items: list[dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select top-limit narrative nodes by rerank_score (items already sorted by caller).

    No slot logic: rerank_score is the single selection criterion. Hard-blocked nodes
    have already been filtered out via answerability_pass before this is called.
    """
    if not items or limit <= 0:
        return [], {"selected": 0, "slots": [], "missing_reason": "missing_substantive_narrative"}
    selected = items[: max(1, limit)]
    return selected, {
        "selected": len(selected),
        "slots": ["rerank_ranked"] * len(selected),
        "missing_reason": None if selected else "missing_substantive_narrative",
    }


def _overlay_rerank_scores(
    *,
    pool: list[dict[str, Any]],
    reranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not pool:
        return []
    rerank_by_id = {
        str(item.get("node_id") or "").strip(): item
        for item in reranked
        if str(item.get("node_id") or "").strip()
    }
    out: list[dict[str, Any]] = []
    for item in pool:
        node_id = str(item.get("node_id") or "").strip()
        if not node_id:
            continue
        reranked_item = rerank_by_id.get(node_id)
        merged = dict(item)
        if reranked_item is not None:
            for key in ("rerank_score", "dense_score", "sparse_score", "fusion_score"):
                if reranked_item.get(key) is not None:
                    merged[key] = reranked_item.get(key)
        out.append(merged)
    return out


def _annotate_section_chunk_index(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        parent = str(item.get("parent_id") or "").strip()
        if parent:
            groups.setdefault(parent, []).append(item)
    for children in groups.values():
        children.sort(key=lambda x: int(x.get("order_index") or 0))
        for idx, child in enumerate(children):
            child["section_chunk_index"] = idx
            child["section_chunk_count"] = len(children)
    return items


_EVIDENCE_STRIP_TERMS = {
    "management's discussion and analysis",
    "management discussion",
    "md&a",
    "item 2",
    "item 2.",
    "forward-looking statements",
    "references to the company",
    "table of contents",
    "10-k",
    "10-q",
}

_EVIDENCE_BOOST_TERMS = (
    "primarily due to",
    "driven by",
    "resulted from",
    "increase in revenue",
    "decrease in revenue",
    "cost of revenue change",
    "gross margin change",
    "operating expenses change",
    "administrative expenses",
    "offset by",
    "compared to the prior period",
)

# Section-aligned rerank tails (no SQL metric → MD&A margin bias).
_NARRATIVE_TARGET_RERANK_HINTS: dict[str, str] = {
    "risk_factors": (
        "Item 1A risk factors; customer concentration; major customer dependence; revenue concentration"
    ),
    "management_discussion": "MD&A results of operations; revenue and margin discussion",
    "margin_cost_structure": "gross margin; cost of revenue; operating expense drivers",
    "liquidity": "liquidity; capital resources; working capital",
    "going_concern": "going concern; substantial doubt",
}


def _narrative_target_rerank_boosts(plan_debug: dict[str, Any]) -> str:
    raw = plan_debug.get("narrative_targets") or ()
    if not isinstance(raw, (list, tuple)):
        return ""
    parts: list[str] = []
    seen: set[str] = set()
    for item in raw:
        key = str(item or "").strip()
        hint = _NARRATIVE_TARGET_RERANK_HINTS.get(key)
        if hint and hint not in seen:
            seen.add(hint)
            parts.append(hint)
    return "; ".join(parts[:4])


def _build_evidence_rerank_query(original_query: str, plan_debug: dict[str, Any]) -> str:
    parts: list[str] = []
    for sentence in original_query.replace("?", ".").split("."):
        sentence = sentence.strip()
        if not sentence:
            continue
        if any(term in sentence.lower() for term in _EVIDENCE_STRIP_TERMS):
            continue
        parts.append(sentence)
    base = ". ".join(parts).strip() if parts else original_query
    raw_targets = plan_debug.get("narrative_targets") or ()
    nt_set = {str(x).strip() for x in raw_targets if str(x).strip()}
    target_boosts = _narrative_target_rerank_boosts(plan_debug)

    # Risk-factor (and similar) asks: do not append SQL-derived "explain gross margin" tails.
    if "risk_factors" in nt_set:
        if target_boosts:
            return f"{base} Specifically: {target_boosts}"
        return base

    boost_fragments: list[str] = []
    tags = _preferred_narrative_tags(plan_debug)
    soft_hints = dict(plan_debug.get("retrieval_soft_hints") or {})
    tag_label_map = {
        "revenue": "revenue changes",
        "cost_of_revenue": "cost of revenue",
        "gross_profit": "gross profit",
        "gross_margin": "gross margin",
        "operating_income": "operating income or loss",
        "net_income": "net income",
        "operating_expense": "operating expenses",
    }
    for tag in tags:
        label = tag_label_map.get(tag)
        if label:
            boost_fragments.append(f"explain {label}")
    for term in soft_hints.get("rerank_terms") or ():
        text = str(term or "").strip()
        if text:
            boost_fragments.append(text)
    sql_boost = ", ".join(boost_fragments[:4])
    if target_boosts and not sql_boost:
        return f"{base} Specifically: {target_boosts}"
    if sql_boost and target_boosts:
        return f"{base} Specifically: {target_boosts}; {sql_boost}"
    if sql_boost:
        return f"{base} Specifically: {sql_boost}"
    if target_boosts:
        return f"{base} Specifically: {target_boosts}"
    return base


def _is_heading_like_node(item: dict[str, Any]) -> bool:
    if not _is_narrative_candidate(item):
        return False
    return _is_heading_like_text(str(item.get("text") or ""))


def _select_following_siblings(
    *,
    heading_nodes: list[dict[str, Any]],
    sibling_rows: list[dict[str, Any]],
    per_heading_limit: int,
) -> list[dict[str, Any]]:
    if not heading_nodes or not sibling_rows:
        return []
    heading_by_parent: dict[str, int] = {}
    for item in heading_nodes:
        parent = str(item.get("parent_id") or "").strip()
        if not parent:
            continue
        try:
            idx = int(item.get("order_index") or 0)
        except (TypeError, ValueError):
            idx = 0
        heading_by_parent[parent] = min(idx, heading_by_parent.get(parent, idx))
    if not heading_by_parent:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sibling_rows:
        parent = str(row.get("parent_id") or "").strip()
        if parent in heading_by_parent:
            grouped[parent].append(row)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parent, rows in grouped.items():
        start_idx = heading_by_parent[parent]
        rows_sorted = sorted(
            rows,
            key=lambda item: int(item.get("order_index") or 0),
        )
        kept = 0
        for row in rows_sorted:
            try:
                row_idx = int(row.get("order_index") or 0)
            except (TypeError, ValueError):
                row_idx = 0
            if row_idx < start_idx:
                continue
            nid = str(row.get("node_id") or "").strip()
            if not nid or nid in seen:
                continue
            out.append({**row, "candidate_source": "section_sibling"})
            seen.add(nid)
            kept += 1
            if kept >= max(1, per_heading_limit):
                break
    return out


def _merge_ranked_section_candidates(
    *,
    base: list[dict[str, Any]],
    section_rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in [*base, *section_rows]:
        nid = str(item.get("node_id") or "").strip()
        if not nid:
            continue
        if nid not in merged:
            merged[nid] = dict(item)
            continue
        current = merged[nid]
        if float(item.get("sparse_score") or item.get("dense_score") or item.get("fusion_score") or 0.0) > float(
            current.get("sparse_score") or current.get("dense_score") or current.get("fusion_score") or 0.0
        ):
            merged[nid] = dict(item)
    ordered = sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("rerank_score") or 0.0),
            float(item.get("sparse_score") or 0.0),
            float(item.get("dense_score") or 0.0),
            float(item.get("fusion_score") or 0.0),
        ),
        reverse=True,
    )
    return ordered[: max(1, limit)]


def _narrative_only_pool(
    *,
    ranked: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    limit: int,
    policy: NarrativeSectionPolicy | None = None,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*ranked, *fallback]:
        nid = str(item.get("node_id") or "").strip()
        if not nid or nid in seen:
            continue
        if not _is_substantive_narrative_candidate(item, policy=policy):
            continue
        if str(item.get("candidate_source") or "").strip() in {"summary", "summary_child"}:
            continue
        seen.add(nid)
        merged.append(item)
        if len(merged) >= max(1, limit):
            break
    return merged


def _build_narrative_answerability_pool(
    *,
    ranked: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    limit: int,
    policy: NarrativeSectionPolicy | None = None,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*ranked, *fallback]:
        nid = str(item.get("node_id") or "").strip()
        if not nid or nid in seen or not _is_driver_priority_candidate(item, policy=policy):
            continue
        seen.add(nid)
        merged.append(item)
        if len(merged) >= max(1, limit):
            return merged
    for item in _narrative_only_pool(ranked=ranked, fallback=fallback, limit=max(1, limit), policy=policy):
        nid = str(item.get("node_id") or "").strip()
        if not nid or nid in seen:
            continue
        seen.add(nid)
        merged.append(item)
        if len(merged) >= max(1, limit):
            break
    return merged


def _rank_section_children(
    *,
    summary_fused: list[dict[str, Any]],
    summary_children: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not summary_fused or not summary_children:
        return []
    parent_scores = {
        str(item.get("node_id") or "").strip(): max(0.2, _evidence_score(item))
        for item in summary_fused
        if str(item.get("node_id") or "").strip()
    }
    ranked: list[dict[str, Any]] = []
    for row in summary_children:
        parent_id = str(row.get("parent_id") or "").strip()
        if not parent_id or parent_id not in parent_scores:
            continue
        ranked.append(
            {
                **row,
                "fusion_score": parent_scores[parent_id],
                "candidate_source": "section_child",
            }
        )
    ranked.sort(
        key=lambda item: (
            float(item.get("fusion_score") or 0.0),
            -int(item.get("order_index") or 0),
        ),
        reverse=True,
    )
    return ranked


def _apply_parent_section_score(
    *,
    rows: list[dict[str, Any]],
    summary_fused: list[dict[str, Any]],
    source_name: str,
) -> list[dict[str, Any]]:
    if not rows or not summary_fused:
        return rows
    parent_scores = {
        str(item.get("node_id") or "").strip(): _evidence_score(item)
        for item in summary_fused
        if str(item.get("node_id") or "").strip()
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        parent_id = str(row.get("parent_id") or "").strip()
        section_score = float(parent_scores.get(parent_id) or 0.0)
        out.append(
            {
                **row,
                "section_score": section_score,
                "candidate_source": source_name,
            }
        )
    return out


def _to_text_node(node: dict[str, Any]) -> TextNode:
    metadata = _row_metadata_as_dict(node.get("metadata"))
    metadata.update(
        {
            "node_id": node["node_id"],
            "document_id": node["document_id"],
            "parent_id": node.get("parent_id"),
            "node_type": node.get("node_type"),
            "level": node.get("level"),
            "order_index": node.get("order_index"),
            "title": node.get("title"),
        }
    )
    for field_name in RETRIEVAL_INDEX_KEYWORD_FIELDS + RETRIEVAL_INDEX_TEXT_FIELDS:
        value = node.get(field_name)
        if value not in (None, "", [], {}):
            metadata[field_name] = value
    if node.get("rerank_score") is not None:
        metadata["rerank_score"] = float(node["rerank_score"])
    return TextNode(
        id_=node["node_id"],
        text=node.get("text", ""),
        metadata=metadata,
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    limit: int,
    score_keys: list[str],
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    merged: dict[str, dict[str, Any]] = {}
    for ranked in ranked_lists:
        for index, item in enumerate(ranked):
            node_id = item["node_id"]
            merged.setdefault(node_id, dict(item))
            scores[node_id] += 1.0 / (60 + index + 1)
            for score_key in score_keys:
                if score_key in item:
                    merged[node_id][score_key] = item[score_key]
    ordered = sorted(
        merged.values(),
        key=lambda item: (scores[item["node_id"]], item.get("level", 0)),
        reverse=True,
    )
    for item in ordered:
        item["fusion_score"] = scores[item["node_id"]]
    return ordered[:limit]


def _log_fusion(stage: str, ranked_lists: list[list[dict[str, Any]]], limit: int, merged: list[dict[str, Any]]) -> None:
    log_rag(
        stage,
        inputs=[len(x) for x in ranked_lists],
        limit=limit,
        merged=len(merged),
        head_node_ids=[x.get("node_id") for x in merged[:10]],
    )


def _narrative_leaf_metadata_filters(
    base: dict[str, Any],
    *,
    strip_finance_accns: bool,
) -> dict[str, Any]:
    """Leaf-layer filters; drop finance_accns to search all filings within document_ids."""
    out = dict(base)
    if strip_finance_accns:
        out.pop("finance_accns", None)
    return out


_SUMMARY_HARD_FILTER_KEYS = frozenset(
    {
        "finance_accns",
        "finance_form_base",
        "finance_metric_exact_keys",
    }
)
_LEAF_HARD_FILTER_KEYS = frozenset(
    {
        "finance_accns",
        "finance_form_base",
    }
)


def _stage_hard_metadata_filters(
    base: dict[str, Any] | None,
    *,
    stage: str,
    need_narrative: bool,
    strip_finance_accns: bool = False,
) -> dict[str, list[str]]:
    source = dict(base or {})
    allowed = _SUMMARY_HARD_FILTER_KEYS if stage == "summary" else _LEAF_HARD_FILTER_KEYS
    if need_narrative:
        allowed = frozenset(k for k in allowed if k != "finance_metric_exact_keys")
    out: dict[str, list[str]] = {}
    for key, raw in source.items():
        if key not in allowed:
            continue
        values = raw if isinstance(raw, list) else [raw]
        clean = [str(v).strip() for v in values if str(v).strip()]
        if clean:
            out[key] = list(dict.fromkeys(clean))
    if strip_finance_accns:
        out.pop("finance_accns", None)
    return out


def _leaf_scope_fallback_needed(
    items: list[dict[str, Any]],
    *,
    min_hits: int,
) -> bool:
    return len(items) < max(1, min_hits)


def _rerank_stage_export_rows(
    items: list[dict[str, Any]],
    *,
    text_cap: int,
) -> list[dict[str, Any]]:
    """Compact rows for standalone rerank-stage JSON (pipeline / report export)."""
    cap = max(120, int(text_cap))
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        preview = text[:cap] + ("..." if len(text) > cap else "")
        rows.append(
            {
                "node_id": item.get("node_id"),
                "document_id": item.get("document_id"),
                "level": item.get("level"),
                "title": item.get("title"),
                "source_section": item.get("source_section"),
                "leaf_role": item.get("leaf_role"),
                "section_role": item.get("section_role"),
                "finance_statement": item.get("finance_statement"),
                "finance_period": item.get("finance_period"),
                "candidate_source": item.get("candidate_source"),
                "dense_score": item.get("dense_score"),
                "sparse_score": item.get("sparse_score"),
                "fusion_score": item.get("fusion_score"),
                "rerank_score": item.get("rerank_score"),
                "text_preview": preview,
            }
        )
    return rows


def _scoped_leaf_fused_debug(
    items: list[dict[str, Any]],
    *,
    min_hits: int,
    filing_scoped_accns: list[str],
) -> dict[str, Any]:
    roles_raw = [str(item.get("leaf_role") or "").strip() or "(empty)" for item in items]
    leaf_role_counts = dict(Counter(roles_raw))
    n = len(items)
    mh = max(1, int(min_hits))
    need_fb = _leaf_scope_fallback_needed(items, min_hits=min_hits)
    reason = "len_lt_min_hits" if n < mh else "ok"
    # Legacy keys for log_rag / pipeline traces (no longer role-based).
    substantive_text = sum(
        1 for item in items if len(str(item.get("text") or "").strip()) >= 200
    )
    return {
        "filing_scoped_accns": list(filing_scoped_accns),
        "len": n,
        "min_hits": int(min_hits),
        "ignore_leaf_role": bool(config.narrative_ignore_leaf_role),
        "substantive_leaf_role_count": substantive_text,
        "leaf_role_counts": leaf_role_counts,
        "fallback_needed_by_rule": need_fb,
        "reason": reason,
        "node_ids_head": [str(item.get("node_id") or "") for item in items[:12]],
    }


class NodeHybridRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        document_ids: list[int],
        metadata_filters: Optional[dict[str, list[str]]] = None,
        retrieval_soft_hints: Optional[dict[str, list[str]]] = None,
        evidence_plan: Any = None,
        callback_handler: RecordingCallbackHandler,
    ) -> None:
        self.document_ids = document_ids
        self.metadata_filters = metadata_filters or {}
        self.retrieval_soft_hints = retrieval_soft_hints or {}
        self.evidence_plan = evidence_plan
        self.callback_handler = callback_handler
        self.dense_backend = get_dense_backend()
        self.sparse_backend = get_sparse_backend()
        callback_manager = CallbackManager([callback_handler])
        super().__init__(callback_manager=callback_manager, verbose=config.debug)
        self.last_debug: dict[str, Any] = {}

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        raise NotImplementedError("Use aretrieve() for async retrieval.")

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str.strip()
        query_embedding = query_bundle.embedding
        plan_debug = _plan_debug_dict(self.evidence_plan)
        if self.retrieval_soft_hints:
            plan_debug["retrieval_soft_hints"] = dict(self.retrieval_soft_hints)
        evidence_req = dict(plan_debug.get("evidence_requirements") or {})
        need_narrative = bool(evidence_req.get("need_narrative"))
        retrieval_budget = dict(plan_debug.get("retrieval_budget") or {})
        summary_limit = max(
            2,
            int(retrieval_budget.get("summary_candidates") or max(6, config.retrieve_top_k // 2)),
        )
        leaf_limit = max(
            4,
            int(retrieval_budget.get("leaf_candidates") or config.retrieve_candidate_k),
        )
        per_filing_cap = max(0, int(retrieval_budget.get("per_filing_cap") or 0))
        narrative_targets = tuple(plan_debug.get("narrative_targets") or ())
        section_policy = resolve_section_policy(narrative_targets) if need_narrative else None
        term_targets = tuple(plan_debug.get("term_targets") or ())
        sparse_query_plan = build_sparse_query_plan(
            query,
            narrative_targets=narrative_targets,
            term_targets=term_targets,
        )
        summary_filters = _stage_hard_metadata_filters(
            self.metadata_filters,
            stage="summary",
            need_narrative=need_narrative,
        )

        with self.callback_manager.event(CBEventType.QUERY) as query_event:
            query_event.on_start(payload={"query": query, "document_ids": self.document_ids})
            if query_embedding is None:
                query_embedding = await generate_embedding(query)
            query_event.on_end(payload={"has_embedding": bool(query_embedding)})

        with self.callback_manager.event(CBEventType.RETRIEVE) as retrieve_event:
            retrieve_event.on_start(payload={"mode": "hierarchical-hybrid"})
            if not query_embedding:
                log_rag("dense_skipped", reason="no_query_embedding", document_ids=len(self.document_ids))
            # Section tree: level = path_depth (1=shallowest, higher=more specific).
            # Search all levels up to section_tree_search_depth so the retriever
            # finds the most specific matching section regardless of depth.
            # Non-narrative queries keep the legacy [1, 2] range for compatibility
            # with documents that haven't been re-ingested yet.
            summary_levels = (
                list(range(1, config.section_tree_search_depth + 1))
                if need_narrative
                else [1, 2]
            )
            summary_dense = (
                self.dense_backend.search(
                    query_embedding,
                    document_ids=self.document_ids,
                    levels=summary_levels,
                    limit=config.dense_top_k,
                    metadata_filters=summary_filters,
                    log_stage="dense_summary",
                )
                if query_embedding
                else []
            )
            summary_sparse = await self.sparse_backend.search(
                self.document_ids,
                query,
                levels=summary_levels,
                limit=config.sparse_top_k,
                metadata_filters=summary_filters,
                query_plan=sparse_query_plan,
                log_stage="sparse_summary",
            )
            summary_fused = [
                {**item, "candidate_source": "summary"}
                for item in reciprocal_rank_fusion(
                [summary_dense, summary_sparse],
                limit=max(summary_limit, config.retrieve_top_k // 2),
                score_keys=["dense_score", "sparse_score"],
            )
            ]
            summary_fused = _apply_filing_aware_limit(
                summary_fused,
                limit=summary_limit,
                per_filing_cap=per_filing_cap,
            )
            _log_fusion(
                "fusion_summary",
                [summary_dense, summary_sparse],
                summary_limit,
                summary_fused,
            )

            filing_scoped_leaf_dense: list[dict[str, Any]] = []
            filing_scoped_leaf_sparse: list[dict[str, Any]] = []
            filing_scoped_accns = [
                str(v) for v in (summary_filters.get("finance_accns") or []) if str(v).strip()
            ]
            scoped_leaf_fused_debug: dict[str, Any] | None = None
            if need_narrative:
                leaf_dense = []
                leaf_sparse = []
                leaf_fused = []
                leaf_base_filters = _stage_hard_metadata_filters(
                    self.metadata_filters,
                    stage="leaf",
                    need_narrative=need_narrative,
                )
                fallback_to_all_filings = bool(config.narrative_leaf_all_filings)
                use_filing_scoped = bool(filing_scoped_accns) and bool(query_embedding)
                if use_filing_scoped:
                    scoped_filters = {**leaf_base_filters, "finance_accns": filing_scoped_accns}
                    filing_scoped_leaf_dense = self.dense_backend.search(
                        query_embedding,
                        document_ids=self.document_ids,
                        levels=[0],
                        limit=config.dense_top_k,
                        metadata_filters=scoped_filters,
                        log_stage="dense_leaf_filing_scoped",
                    )
                    filing_scoped_leaf_sparse = await self.sparse_backend.search(
                        self.document_ids,
                        query,
                        levels=[0],
                        limit=config.sparse_top_k,
                        metadata_filters=scoped_filters,
                        query_plan=sparse_query_plan,
                        log_stage="sparse_leaf_filing_scoped",
                    )
                    leaf_fused = [
                        {**item, "candidate_source": "filing_scoped_leaf"}
                        for item in reciprocal_rank_fusion(
                            [filing_scoped_leaf_dense, filing_scoped_leaf_sparse],
                            limit=max(leaf_limit, config.retrieve_candidate_k),
                            score_keys=["dense_score", "sparse_score"],
                        )
                    ]
                    leaf_fused = _apply_filing_aware_limit(
                        leaf_fused,
                        limit=leaf_limit,
                        per_filing_cap=max(per_filing_cap, 8),
                    )
                    _log_fusion(
                        "fusion_leaf_filing_scoped",
                        [filing_scoped_leaf_dense, filing_scoped_leaf_sparse],
                        leaf_limit,
                        leaf_fused,
                    )
                    min_hits_leaf_scope = min(max(2, leaf_limit // 2), 4)
                    scoped_leaf_fused_debug = _scoped_leaf_fused_debug(
                        leaf_fused,
                        min_hits=min_hits_leaf_scope,
                        filing_scoped_accns=filing_scoped_accns,
                    )
                    log_rag(
                        "scoped_leaf_fused",
                        len=scoped_leaf_fused_debug["len"],
                        min_hits=scoped_leaf_fused_debug["min_hits"],
                        leaf_role_counts=scoped_leaf_fused_debug["leaf_role_counts"],
                        substantive_leaf_role_count=scoped_leaf_fused_debug[
                            "substantive_leaf_role_count"
                        ],
                        fallback_needed_by_rule=scoped_leaf_fused_debug[
                            "fallback_needed_by_rule"
                        ],
                        reason=scoped_leaf_fused_debug["reason"],
                        ignore_leaf_role=scoped_leaf_fused_debug["ignore_leaf_role"],
                    )
                    min_hits_for_fallback = min_hits_leaf_scope
                else:
                    min_hits_for_fallback = min(max(2, leaf_limit // 2), 4)
                fallback_needed = bool(query_embedding) and (
                    not use_filing_scoped
                    or (
                        fallback_to_all_filings
                        and _leaf_scope_fallback_needed(
                            leaf_fused,
                            min_hits=min_hits_for_fallback,
                        )
                    )
                )
                if fallback_needed:
                    strip_accns = True
                    leaf_mf = _narrative_leaf_metadata_filters(
                        leaf_base_filters,
                        strip_finance_accns=strip_accns,
                    )
                    leaf_dense = self.dense_backend.search(
                        query_embedding,
                        document_ids=self.document_ids,
                        levels=[0],
                        limit=config.dense_top_k,
                        metadata_filters=leaf_mf,
                        log_stage="dense_leaf_narrative",
                    )
                    leaf_sparse = await self.sparse_backend.search(
                        self.document_ids,
                        query,
                        levels=[0],
                        limit=config.sparse_top_k,
                        metadata_filters=leaf_mf,
                        query_plan=sparse_query_plan,
                        log_stage="sparse_leaf_narrative",
                    )
                    leaf_fused = [
                        {**item, "candidate_source": "leaf"}
                        for item in reciprocal_rank_fusion(
                            [leaf_dense, leaf_sparse],
                            limit=max(leaf_limit, config.retrieve_candidate_k),
                            score_keys=["dense_score", "sparse_score"],
                        )
                    ]
                    leaf_fused = _apply_filing_aware_limit(
                        leaf_fused,
                        limit=leaf_limit,
                        per_filing_cap=per_filing_cap,
                    )
                    _log_fusion(
                        "fusion_leaf_narrative_unscoped",
                        [leaf_dense, leaf_sparse],
                        leaf_limit,
                        leaf_fused,
                    )
            else:
                leaf_dense = (
                    self.dense_backend.search(
                        query_embedding,
                        document_ids=self.document_ids,
                        levels=[0],
                        limit=config.dense_top_k,
                        metadata_filters=self.metadata_filters,
                        log_stage="dense_leaf",
                    )
                    if query_embedding
                    else []
                )
                leaf_sparse = await self.sparse_backend.search(
                    self.document_ids,
                    query,
                    levels=[0],
                    limit=config.sparse_top_k,
                    metadata_filters=self.metadata_filters,
                    query_plan=sparse_query_plan,
                    log_stage="sparse_leaf",
                )
                leaf_fused = [
                    {**item, "candidate_source": "leaf"}
                    for item in reciprocal_rank_fusion(
                        [leaf_dense, leaf_sparse],
                        limit=max(leaf_limit, config.retrieve_candidate_k),
                        score_keys=["dense_score", "sparse_score"],
                    )
                ]
                leaf_fused = _apply_filing_aware_limit(
                    leaf_fused,
                    limit=leaf_limit,
                    per_filing_cap=per_filing_cap,
                )
                _log_fusion("fusion_leaf", [leaf_dense, leaf_sparse], leaf_limit, leaf_fused)
            retrieve_event.on_end(
                payload={
                    "summary_dense": len(summary_dense),
                    "summary_sparse": len(summary_sparse),
                    "leaf_dense": len(leaf_dense) + len(filing_scoped_leaf_dense),
                    "leaf_sparse": len(leaf_sparse) + len(filing_scoped_leaf_sparse),
                }
            )

        selected_section_ids = [
            str(item.get("node_id") or "").strip()
            for item in summary_fused
            if str(item.get("node_id") or "").strip()
        ]
        if need_narrative:
            section_children = await fetch_leaf_descendants(selected_section_ids)
            section_children = _apply_parent_section_score(
                rows=section_children,
                summary_fused=summary_fused,
                source_name="section_child_enum",
            )
            section_children = _annotate_section_chunk_index(section_children)
            section_leaf_pool = [
                {**item, "candidate_source": "section_leaf", "fusion_score": 0.2}
                for item in section_children
                if int(item.get("level") or 0) == 0
            ]
            if leaf_fused:
                pre_rerank = reciprocal_rank_fusion(
                    [section_leaf_pool, leaf_fused],
                    limit=max(config.retrieve_candidate_k, leaf_limit + len(section_leaf_pool)),
                    score_keys=["dense_score", "sparse_score", "fusion_score"],
                )
                _log_fusion(
                    "fusion_narrative_section_plus_filing_scoped",
                    [section_leaf_pool, leaf_fused],
                    max(config.retrieve_candidate_k, leaf_limit + len(section_leaf_pool)),
                    pre_rerank,
                )
            else:
                pre_rerank = section_leaf_pool
            log_rag(
                "section_leaf_enumerate",
                summary_nodes=len(summary_fused),
                scoped_parent_ids=len(selected_section_ids),
                total_children=len(section_children),
                leaf_candidates=len(pre_rerank),
                filing_scoped_leaf_count=len(leaf_fused),
            )
        else:
            summary_children = await fetch_children(
                selected_section_ids,
                limit_per_parent=max(config.hierarchical_group_size, leaf_limit, 8),
            )
            child_candidates = [
                {
                    **item,
                    "fusion_score": 0.2,
                    "candidate_source": "summary_child",
                }
                for item in summary_children
            ]
            pre_rerank = reciprocal_rank_fusion(
                [leaf_fused, child_candidates],
                limit=max(config.retrieve_candidate_k, leaf_limit + summary_limit),
                score_keys=["dense_score", "sparse_score", "fusion_score"],
            )
            pre_rerank = _apply_filing_aware_limit(
                pre_rerank,
                limit=max(config.retrieve_candidate_k, leaf_limit),
                per_filing_cap=per_filing_cap,
                bucket_caps={
                    "leaf": leaf_limit,
                    "summary_child": summary_limit,
                },
            )
            _log_fusion(
                "fusion_pre_rerank",
                [leaf_fused, child_candidates],
                max(config.retrieve_candidate_k, leaf_limit),
                pre_rerank,
            )
            log_rag(
                "summary_children",
                summary_nodes=len(summary_fused),
                child_candidates=len(child_candidates),
                group_size=config.hierarchical_group_size,
            )
        _dump_pre_rerank_debug(
            query=query,
            pre_rerank=pre_rerank,
            metadata_filters=self.metadata_filters,
            need_narrative=need_narrative,
        )

        rerank_stats: dict[str, Any] = {}
        rerank_query = (
            _build_evidence_rerank_query(query, plan_debug)
            if need_narrative
            else query
        )
        rerank_keep = max(config.bocha_top_n, leaf_limit) if need_narrative else config.bocha_top_n
        top_n_cap = min(rerank_keep, max(1, len(pre_rerank)))
        if need_narrative and bool(config.narrative_multi_rerank_enabled):
            subqs = narrative_rerank_subqueries(
                rerank_query,
                plan_debug,
                enabled=True,
                max_queries=int(config.narrative_multi_rerank_max_queries),
            )
            raw_reranked = await run_multi_query_rerank(
                reranker,
                queries=subqs,
                candidates=pre_rerank,
                rerank_keep=top_n_cap,
                out_stats=rerank_stats,
            )
        else:
            reranked = await reranker.rerank(
                query=rerank_query,
                candidates=pre_rerank,
                top_n=top_n_cap,
                out_stats=rerank_stats,
            )
            raw_reranked = reranked if reranked else pre_rerank[:rerank_keep]
        post_rerank_selector: dict[str, Any] = {"applied": False}
        final_ranked = raw_reranked
        if need_narrative:
            final_ranked, post_rerank_selector = _apply_narrative_post_rerank_selector(
                items=raw_reranked,
                limit=min(rerank_keep, max(1, len(raw_reranked))),
            )
        final_ranked = _apply_filing_aware_limit(
            final_ranked,
            limit=min(rerank_keep, max(1, len(final_ranked))),
            per_filing_cap=max(per_filing_cap, 8) if need_narrative else per_filing_cap,
            bucket_caps=(
                {"section_leaf": leaf_limit, "filing_scoped_leaf": leaf_limit}
                if need_narrative
                else {
                    "leaf": leaf_limit,
                    "summary_child": summary_limit,
                }
            ),
        )
        if need_narrative:
            narrative_inserted = 0
        elif bool(evidence_req.get("need_narrative")):
            narrative_floor = min(3, max(2, min(config.retrieve_top_k, len(final_ranked))))
            final_ranked, narrative_inserted = _ensure_narrative_quota(
                ranked=final_ranked,
                fallback_pool=pre_rerank,
                limit=min(config.bocha_top_n, max(1, len(final_ranked))),
                min_narrative=narrative_floor,
            )
            if narrative_inserted > 0:
                log_rag(
                    "narrative_quota_enforced",
                    inserted=narrative_inserted,
                    target=narrative_floor,
                    final_ranked=len(final_ranked),
                )
        else:
            narrative_inserted = 0

        section_sibling_candidates: list[dict[str, Any]] = []
        heading_like_nodes = [item for item in final_ranked if _is_heading_like_node(item)]
        if need_narrative:
            heading_like_nodes = []
        elif bool(evidence_req.get("need_narrative")) and heading_like_nodes:
            parent_ids = [
                str(item.get("parent_id") or "").strip()
                for item in heading_like_nodes
                if str(item.get("parent_id") or "").strip()
            ]
            if parent_ids:
                sibling_rows = await fetch_children(
                    parent_ids,
                    limit_per_parent=max(config.hierarchical_group_size, 8),
                )
                section_sibling_candidates = _select_following_siblings(
                    heading_nodes=heading_like_nodes,
                    sibling_rows=sibling_rows,
                    per_heading_limit=6,
                )
                if section_sibling_candidates:
                    pre_rerank = _merge_ranked_section_candidates(
                        base=pre_rerank,
                        section_rows=section_sibling_candidates,
                        limit=max(config.retrieve_candidate_k, leaf_limit + summary_limit),
                    )
                    reranked_section = await reranker.rerank(
                        query=query,
                        candidates=pre_rerank,
                        top_n=min(config.bocha_top_n, len(pre_rerank)),
                        out_stats=None,
                    )
                    final_ranked = reranked_section if reranked_section else pre_rerank[: config.bocha_top_n]
                    final_ranked = _apply_filing_aware_limit(
                        final_ranked,
                        limit=min(config.bocha_top_n, max(1, len(final_ranked))),
                        per_filing_cap=per_filing_cap,
                        bucket_caps={
                            "leaf": leaf_limit,
                            "summary_child": summary_limit,
                            "section_sibling": max(2, min(6, summary_limit)),
                        },
                    )
                    log_rag(
                        "section_expand",
                        heading_hits=len(heading_like_nodes),
                        sibling_candidates=len(section_sibling_candidates),
                        final_ranked=len(final_ranked),
                    )

        narrative_pool_size = 0
        answerability_stats: dict[str, Any] = {}
        coverage_stats: dict[str, Any] = {}
        evidence_gate: dict[str, Any] = {"passed": True, "missing_reason": None}
        if need_narrative:
            all_section_leaves = _overlay_rerank_scores(
                pool=pre_rerank,
                reranked=raw_reranked,
            )
            narrative_final = _build_narrative_answerability_pool(
                ranked=all_section_leaves,
                fallback=[],
                limit=max(1, len(all_section_leaves)),
                policy=section_policy,
            )
            narrative_scored, answerability_stats = _apply_answerability_scores(
                items=narrative_final,
                plan_debug=plan_debug,
                policy=section_policy,
            )
            disable_answerability_gate = bool(config.narrative_disable_answerability_gate)
            coverage_input = (
                narrative_scored
                if disable_answerability_gate
                else [item for item in narrative_scored if bool(item.get("answerability_pass"))]
            )
            coverage_selected, coverage_stats = _select_narrative_coverage(
                items=coverage_input,
                limit=min(config.retrieve_top_k, 8),
            )
            narrative_pool_size = len(narrative_scored)
            if coverage_selected:
                final_ranked = coverage_selected
            else:
                if disable_answerability_gate:
                    # Keep rerank-driven candidates when gate bypass is enabled.
                    final_ranked = final_ranked[: min(config.retrieve_top_k, len(final_ranked))]
                else:
                    final_ranked = []
                    evidence_gate = {
                        "passed": False,
                        "missing_reason": str(coverage_stats.get("missing_reason") or "missing_substantive_narrative"),
                    }
            log_rag(
                "narrative_pool",
                applied=True,
                pool_size=narrative_pool_size,
                final_ranked=len(final_ranked),
            )
            log_rag(
                "answerability_gate",
                scored=answerability_stats.get("scored"),
                passed=answerability_stats.get("passed"),
                selected=coverage_stats.get("selected"),
                slots=coverage_stats.get("slots"),
                missing_reason=coverage_stats.get("missing_reason"),
            )

        # Section-bounded context expansion.
        # Only expand the top context_sibling_seed_count seeds so low-quality candidates
        # don't drag in large sections of irrelevant siblings.
        # Siblings inherit the seed's rerank_score * decay so they remain eligible for
        # the final top-k context cut in rag_service._take_top_k_by_score.
        sibling_seed_count = max(1, config.context_sibling_seed_count)
        sibling_limit = max(1, config.context_sibling_limit)
        sibling_decay = float(config.context_sibling_score_decay)
        seeds = final_ranked[:sibling_seed_count] if sibling_seed_count > 0 else []
        sibling_records: list[dict[str, Any]] = []
        for seed in seeds:
            inherited_score = float(seed.get("rerank_score") or 0.0) * sibling_decay
            for sib in await fetch_siblings(seed["node_id"], limit=sibling_limit):
                sib = dict(sib)
                # Only set inherited score if sibling has no rerank score of its own.
                if sib.get("rerank_score") is None and inherited_score > 0:
                    sib["rerank_score"] = inherited_score
                sibling_records.append(sib)

        expanded_nodes = await fetch_nodes([item["node_id"] for item in final_ranked])
        merged_by_id = {item["node_id"]: item for item in expanded_nodes}
        for item in sibling_records:
            merged_by_id.setdefault(item["node_id"], item)

        ordered_context: list[dict[str, Any]] = sorted(
            merged_by_id.values(),
            key=lambda item: (item["document_id"], item["level"], item["order_index"]),
        )
        log_rag(
            "context_expand",
            final_ranked=len(final_ranked),
            sibling_seed_count=len(seeds),
            sibling_limit=sibling_limit,
            sibling_rows=len(sibling_records),
            context_nodes_expanded=len(ordered_context),
            retrieve_top_k=config.retrieve_top_k,
        )
        by_score = {item["node_id"]: item for item in final_ranked}
        results = []
        for item in ordered_context:
            fr = by_score.get(item["node_id"], {}) or {}
            row = dict(item)
            if fr.get("rerank_score") is not None:
                row["rerank_score"] = fr["rerank_score"]
            elif item.get("rerank_score") is not None:
                # Inherited score from sibling expansion — keep it for rag_service ranking.
                row["rerank_score"] = item["rerank_score"]
            results.append(
                NodeWithScore(
                    node=_to_text_node(row),
                    score=_evidence_score(row),
                )
            )
        leaf_dense_total = len(leaf_dense) + len(filing_scoped_leaf_dense)
        leaf_sparse_total = len(leaf_sparse) + len(filing_scoped_leaf_sparse)
        _rerank_export_cap = max(200, int(config.pipeline_trace_sparse_text_chars or 800))
        self.last_debug = {
            "retrieval_counts": {
                "summary_dense": len(summary_dense),
                "summary_sparse": len(summary_sparse),
                "leaf_dense": leaf_dense_total,
                "leaf_sparse": leaf_sparse_total,
                "leaf_dense_filing_scoped": len(filing_scoped_leaf_dense),
                "leaf_sparse_filing_scoped": len(filing_scoped_leaf_sparse),
                "pre_rerank_pool": len(pre_rerank),
                "final_ranked": len(final_ranked),
                "context_nodes_expanded": len(ordered_context),
            },
            "sparse_query_profile": sparse_query_plan.profile,
            "sparse_query_slots": sparse_query_plan.slots,
            "metadata_filters": dict(self.metadata_filters),
            "retrieval_soft_hints": dict(self.retrieval_soft_hints),
            "metadata_filter_policy": {
                "summary_hard_filters": dict(summary_filters),
                "leaf_hard_filters": dict(
                    _stage_hard_metadata_filters(
                        self.metadata_filters,
                        stage="leaf",
                        need_narrative=need_narrative,
                    )
                ),
                "leaf_filing_scoped_accns": list(filing_scoped_accns),
                "leaf_scope_strategy": {
                    "need_narrative": need_narrative,
                    "fallback_to_all_filings_enabled": bool(config.narrative_leaf_all_filings),
                    "used_filing_scoped_leaf": bool(filing_scoped_leaf_dense or filing_scoped_leaf_sparse),
                    "used_unscoped_leaf": bool(leaf_dense or leaf_sparse),
                },
            },
            "scoped_leaf_fused_debug": scoped_leaf_fused_debug,
            "evidence_plan": plan_debug,
            "fusion_quotas": {
                "summary_limit": summary_limit,
                "leaf_limit": leaf_limit,
                "per_filing_cap": per_filing_cap,
            },
            "fused_filing_distribution": {
                "summary": _filing_distribution(summary_fused),
                "leaf": _filing_distribution(leaf_fused),
                "pre_rerank": _filing_distribution(pre_rerank),
                "final_ranked": _filing_distribution(final_ranked),
            },
            "narrative_quota": {
                "applied": bool(evidence_req.get("need_narrative")),
                "inserted": int(narrative_inserted),
            },
            "section_expansion": {
                "heading_hits": len(heading_like_nodes),
                "sibling_candidates": len(section_sibling_candidates),
            },
            "post_rerank_selector": dict(post_rerank_selector),
            "narrative_pool": {
                "applied": bool(evidence_req.get("need_narrative")),
                "pool_size": narrative_pool_size,
            },
            "section_policy": {
                "active": section_policy is not None,
                # Role-based allowlists removed; keys kept for trace schema compatibility.
                "allowed_section_roles": [],
                "blocked_leaf_roles": [],
                "open_mode": True,
                "penalize_forward_looking": section_policy.penalize_forward_looking if section_policy else True,
                "penalize_early_position": section_policy.penalize_early_position if section_policy else True,
                "pass_score_threshold": section_policy.pass_score_threshold if section_policy else 2.5,
                "disable_section_hard_filter": bool(config.narrative_disable_section_hard_filter),
                "disable_answerability_gate": bool(config.narrative_disable_answerability_gate),
                "ignore_section_role": bool(config.narrative_ignore_section_role),
                "ignore_leaf_role": bool(config.narrative_ignore_leaf_role),
            },
            "answerability": dict(answerability_stats),
            "coverage_selection": dict(coverage_stats),
            "evidence_gate": dict(evidence_gate),
            "summary_dense_hits": _slim_dense_stage(summary_dense),
            "summary_sparse_hits": _slim_sparse_stage(summary_sparse),
            "leaf_dense_hits": _slim_dense_stage([*leaf_dense, *filing_scoped_leaf_dense]),
            "leaf_sparse_hits": _slim_sparse_stage([*leaf_sparse, *filing_scoped_leaf_sparse]),
            "rerank": dict(rerank_stats),
            "summary_candidates": summary_fused,
            "leaf_candidates": leaf_fused,
            "pre_rerank": pre_rerank,
            "reranked": raw_reranked,
            "rerank_stage_pre_rerank": _rerank_stage_export_rows(
                pre_rerank, text_cap=_rerank_export_cap
            ),
            "rerank_stage_rerank_out": _rerank_stage_export_rows(
                raw_reranked, text_cap=_rerank_export_cap
            ),
            "rerank_stage_final_ranked": _rerank_stage_export_rows(
                final_ranked, text_cap=_rerank_export_cap
            ),
            "events": self.callback_handler.flush_events(),
        }
        log_rag(
            "retrieve_done",
            query_len=len(query),
            dense_hits=leaf_dense_total + len(summary_dense),
            sparse_hits=leaf_sparse_total + len(summary_sparse),
            nodes_returned=len(results),
            pre_rerank_pool=len(pre_rerank),
        )
        return results


class LlamaIndexRetrievalService:
    async def retrieve(
        self,
        *,
        query: str,
        document_ids: list[int],
        metadata_filters: Optional[dict[str, list[str]]] = None,
        retrieval_soft_hints: Optional[dict[str, list[str]]] = None,
        evidence_plan: Any = None,
    ) -> dict[str, Any]:
        callback_handler = RecordingCallbackHandler()
        retriever = NodeHybridRetriever(
            document_ids=document_ids,
            metadata_filters=metadata_filters,
            retrieval_soft_hints=retrieval_soft_hints,
            evidence_plan=evidence_plan,
            callback_handler=callback_handler,
        )
        nodes = await retriever.aretrieve(QueryBundle(query_str=query))
        unique_nodes: list[dict[str, Any]] = []
        seen = set()
        for node_with_score in nodes:
            node_id = node_with_score.node.node_id
            if node_id in seen:
                continue
            seen.add(node_id)
            metadata = dict(node_with_score.node.metadata or {})
            narrowing_meta = {
                key: metadata.get(key)
                for key in _NARROWING_META_KEYS
                if metadata.get(key) not in (None, "", [], {})
            }
            rs = metadata.get("rerank_score")
            unique_nodes.append(
                {
                    "node_id": node_id,
                    "document_id": metadata.get("document_id"),
                    "node_type": metadata.get("node_type"),
                    "level": metadata.get("level"),
                    "order_index": metadata.get("order_index"),
                    "title": metadata.get("title"),
                    "text": node_with_score.node.text,
                    "score": float(node_with_score.score or 0.0),
                    **({"rerank_score": float(rs)} if rs is not None else {}),
                    # Keep minimal structured fields so SQL->RAG narrowing can do exact metadata matching.
                    "metadata": narrowing_meta,
                }
            )
        return {
            "query": query,
            "nodes": unique_nodes,
            "debug": retriever.last_debug,
        }


retrieval_service = LlamaIndexRetrievalService()
