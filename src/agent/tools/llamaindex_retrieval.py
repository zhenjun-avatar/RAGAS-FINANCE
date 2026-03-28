"""LlamaIndex-based retrieval layer with sparse+dense hybrid and context completion."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Optional

from llama_index.core.callbacks import CBEventType, CallbackManager
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from loguru import logger

from core.config import config
from .bocha_reranker import reranker
from .llamaindex_callbacks import RecordingCallbackHandler
from .node_repository import fetch_children, fetch_neighbors, fetch_nodes
from .rag_stage_log import log_rag
from .retrieval_fields import RETRIEVAL_INDEX_KEYWORD_FIELDS
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
                "finance_statement": r.get("finance_statement"),
                "finance_period": r.get("finance_period"),
                "text_preview": body_preview,
                "text_truncated": truncated,
            }
        )
    return out


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
    for field_name in RETRIEVAL_INDEX_KEYWORD_FIELDS + ["search_hints"]:
        value = node.get(field_name)
        if value not in (None, "", [], {}):
            metadata[field_name] = value
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


class NodeHybridRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        document_ids: list[int],
        metadata_filters: Optional[dict[str, list[str]]] = None,
        evidence_plan: Any = None,
        callback_handler: RecordingCallbackHandler,
    ) -> None:
        self.document_ids = document_ids
        self.metadata_filters = metadata_filters or {}
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
        term_targets = tuple(plan_debug.get("term_targets") or ())
        sparse_query_plan = build_sparse_query_plan(
            query,
            narrative_targets=narrative_targets,
            term_targets=term_targets,
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
            summary_dense = (
                self.dense_backend.search(
                    query_embedding,
                    document_ids=self.document_ids,
                    levels=[1, 2],
                    limit=config.dense_top_k,
                    metadata_filters=self.metadata_filters,
                    log_stage="dense_summary",
                )
                if query_embedding
                else []
            )
            summary_sparse = await self.sparse_backend.search(
                self.document_ids,
                query,
                levels=[1, 2],
                limit=config.sparse_top_k,
                metadata_filters=self.metadata_filters,
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
                    "leaf_dense": len(leaf_dense),
                    "leaf_sparse": len(leaf_sparse),
                }
            )

        summary_children = await fetch_children(
            [item["node_id"] for item in summary_fused],
            limit_per_parent=config.hierarchical_group_size,
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

        rerank_stats: dict[str, Any] = {}
        reranked = await reranker.rerank(
            query=query,
            candidates=pre_rerank,
            top_n=min(config.bocha_top_n, len(pre_rerank)),
            out_stats=rerank_stats,
        )
        final_ranked = reranked if reranked else pre_rerank[: config.bocha_top_n]
        final_ranked = _apply_filing_aware_limit(
            final_ranked,
            limit=min(config.bocha_top_n, max(1, len(final_ranked))),
            per_filing_cap=per_filing_cap,
            bucket_caps={
                "leaf": leaf_limit,
                "summary_child": summary_limit,
            },
        )

        neighbor_records: list[dict[str, Any]] = []
        for item in final_ranked[: config.retrieve_top_k]:
            neighbor_records.extend(
                await fetch_neighbors(item["node_id"], radius=config.context_neighbor_radius)
            )
        expanded_nodes = await fetch_nodes([item["node_id"] for item in final_ranked])
        merged_by_id = {item["node_id"]: item for item in expanded_nodes}
        for item in neighbor_records:
            merged_by_id.setdefault(item["node_id"], item)

        ordered_context: list[dict[str, Any]] = sorted(
            merged_by_id.values(),
            key=lambda item: (item["document_id"], item["level"], item["order_index"]),
        )
        log_rag(
            "context_expand",
            final_ranked=len(final_ranked),
            neighbor_radius=config.context_neighbor_radius,
            neighbor_rows=len(neighbor_records),
            context_nodes_expanded=len(ordered_context),
            retrieve_top_k=config.retrieve_top_k,
        )
        by_score = {item["node_id"]: item for item in final_ranked}
        results = [
            NodeWithScore(
                node=_to_text_node(item),
                score=_evidence_score(by_score.get(item["node_id"], {})),
            )
            for item in ordered_context
        ]
        self.last_debug = {
            "retrieval_counts": {
                "summary_dense": len(summary_dense),
                "summary_sparse": len(summary_sparse),
                "leaf_dense": len(leaf_dense),
                "leaf_sparse": len(leaf_sparse),
                "pre_rerank_pool": len(pre_rerank),
                "final_ranked": len(final_ranked),
                "context_nodes_expanded": len(ordered_context),
            },
            "sparse_query_profile": sparse_query_plan.profile,
            "sparse_query_slots": sparse_query_plan.slots,
            "metadata_filters": dict(self.metadata_filters),
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
            "summary_dense_hits": _slim_dense_stage(summary_dense),
            "summary_sparse_hits": _slim_sparse_stage(summary_sparse),
            "leaf_dense_hits": _slim_dense_stage(leaf_dense),
            "leaf_sparse_hits": _slim_sparse_stage(leaf_sparse),
            "rerank": dict(rerank_stats),
            "summary_candidates": summary_fused,
            "leaf_candidates": leaf_fused,
            "pre_rerank": pre_rerank,
            "reranked": final_ranked,
            "events": self.callback_handler.flush_events(),
        }
        log_rag(
            "retrieve_done",
            query_len=len(query),
            dense_hits=len(leaf_dense) + len(summary_dense),
            sparse_hits=len(leaf_sparse) + len(summary_sparse),
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
        evidence_plan: Any = None,
    ) -> dict[str, Any]:
        callback_handler = RecordingCallbackHandler()
        retriever = NodeHybridRetriever(
            document_ids=document_ids,
            metadata_filters=metadata_filters,
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
