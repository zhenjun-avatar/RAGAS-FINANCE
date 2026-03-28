"""OpenSearch-backed sparse retrieval backend."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from loguru import logger
from opensearchpy import OpenSearch, helpers

from core.config import config
from ..retrieval_fields import RETRIEVAL_INDEX_KEYWORD_FIELDS, RETRIEVAL_INDEX_TEXT_FIELDS

from .sparse_query_profiles import SparseQueryPlan, build_sparse_query_plan
from .types import NodeHit, NodeIndexRecord
from ..rag_stage_log import log_rag

_client: Optional[OpenSearch] = None


def _metadata_terms_filter(field: str, values: list[str]) -> dict[str, Any]:
    """Compat filter for old indices where fields were mapped as text+keyword."""
    return {
        "bool": {
            "should": [
                {"terms": {field: values}},
                {"terms": {f"{field}.keyword": values}},
            ],
            "minimum_should_match": 1,
        }
    }


def _configured_index_names() -> list[str]:
    names = [
        config.opensearch_sparse_index_finance,
        config.opensearch_sparse_index,
    ]
    out: list[str] = []
    for name in names:
        normalized = (name or "").strip()
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def _search_index_names() -> list[str]:
    scope = (config.opensearch_sparse_search_scope or "finance").strip().lower()
    default = (config.opensearch_sparse_index or "").strip()

    if scope == "exam":
        logger.warning(
            "[OpenSearchSparse] scope=exam removed; treating as finance + default indices"
        )

    if scope == "finance":
        fin = (config.opensearch_sparse_index_finance or "").strip()
        if fin:
            return [fin]
        logger.warning(
            "[OpenSearchSparse] scope=finance requires OPENSEARCH_SPARSE_INDEX_FINANCE (non-empty); "
            "sparse search uses no index until set"
        )
        return []

    # scope=all：财务索引 + 默认索引（无独立第二业务域时与 finance 类似，多一个兜底索引）
    fin = (config.opensearch_sparse_index_finance or "").strip()
    names: list[str] = []
    if fin:
        names.append(fin)
    if default and default not in names:
        names.append(default)
    return names if names else _configured_index_names()


def _replace_cleanup_index_names() -> list[str]:
    """重入库前按 document_id 删除；scope=finance 时只清财务索引，其余为全部已配置索引。"""
    scope = (config.opensearch_sparse_search_scope or "finance").strip().lower()
    if scope == "finance":
        fin = (config.opensearch_sparse_index_finance or "").strip()
        return [fin] if fin else []
    return _configured_index_names()


def _index_for_domain(domain: Any) -> str:
    normalized = str(domain or "").strip().lower()
    if normalized == "finance" and (config.opensearch_sparse_index_finance or "").strip():
        return str(config.opensearch_sparse_index_finance).strip()
    return config.opensearch_sparse_index


def _field_mapping() -> dict[str, Any]:
    mapping: dict[str, Any] = {"type": "text"}
    if config.opensearch_sparse_analyzer:
        mapping["analyzer"] = config.opensearch_sparse_analyzer
    if config.opensearch_sparse_search_analyzer:
        mapping["search_analyzer"] = config.opensearch_sparse_search_analyzer
    return mapping


def _index_body() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "node_id": {"type": "keyword"},
        "document_id": {"type": "long"},
        "ingest_run_id": {"type": "keyword"},
        "parent_id": {"type": "keyword"},
        "node_type": {"type": "keyword"},
        "level": {"type": "integer"},
        "order_index": {"type": "integer"},
        "title": _field_mapping(),
        "text": _field_mapping(),
        "metadata": {"type": "object", "enabled": False},
    }
    for field_name in RETRIEVAL_INDEX_KEYWORD_FIELDS:
        properties[field_name] = {"type": "keyword"}
    for field_name in RETRIEVAL_INDEX_TEXT_FIELDS:
        properties[field_name] = _field_mapping()
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {"properties": properties},
    }


def _to_source(node: NodeIndexRecord) -> dict[str, Any]:
    raw_doc_id = node["document_id"]
    try:
        doc_id = int(raw_doc_id)  # align with Qdrant payload + terms queries
    except (TypeError, ValueError):
        doc_id = raw_doc_id
    source = {
        "node_id": node["node_id"],
        "document_id": doc_id,
        "ingest_run_id": node.get("ingest_run_id"),
        "parent_id": node.get("parent_id"),
        "node_type": node["node_type"],
        "level": node["level"],
        "order_index": node["order_index"],
        "title": node.get("title"),
        "text": node["text"],
        "metadata": node.get("metadata") or {},
    }
    for field_name in RETRIEVAL_INDEX_KEYWORD_FIELDS + RETRIEVAL_INDEX_TEXT_FIELDS:
        value = node.get(field_name)
        if value not in (None, "", [], {}):
            source[field_name] = value
    return source


def get_client() -> OpenSearch:
    global _client
    if _client is None:
        http_auth = None
        if config.opensearch_user:
            http_auth = (config.opensearch_user, config.opensearch_password or "")
        _client = OpenSearch(
            hosts=[
                {
                    "host": config.opensearch_host,
                    "port": config.opensearch_port,
                    "scheme": "https" if config.opensearch_use_ssl else "http",
                }
            ],
            http_auth=http_auth,
            use_ssl=config.opensearch_use_ssl,
            verify_certs=config.opensearch_verify_certs,
            timeout=config.opensearch_timeout_seconds,
        )
    return _client


def ensure_index(index_name: str) -> None:
    client = get_client()
    if client.indices.exists(index=index_name):
        return
    client.indices.create(index=index_name, body=_index_body())
    logger.info("[OpenSearchSparse] Created index {}", index_name)


def _replace_document_nodes_sync(document_id: int, nodes: list[NodeIndexRecord]) -> None:
    client = get_client()
    for index_name in _replace_cleanup_index_names():
        if not client.indices.exists(index=index_name):
            continue
        client.delete_by_query(
            index=index_name,
            body={"query": {"term": {"document_id": document_id}}},
            conflicts="proceed",
            refresh=True,
        )
    if not nodes:
        return
    grouped: dict[str, list[NodeIndexRecord]] = {}
    for node in nodes:
        grouped.setdefault(_index_for_domain(node.get("domain")), []).append(node)
    for index_name, grouped_nodes in grouped.items():
        ensure_index(index_name)
        actions = [
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": node["node_id"],
                "_source": _to_source(node),
            }
            for node in grouped_nodes
        ]
        helpers.bulk(client, actions, refresh="wait_for")


class OpenSearchSparseBackend:
    async def search(
        self,
        document_ids: list[int],
        query: str,
        *,
        limit: int,
        levels: Optional[list[int]] = None,
        metadata_filters: Optional[dict[str, list[str]]] = None,
        query_plan: Optional[SparseQueryPlan] = None,
        log_stage: Optional[str] = None,
    ) -> list[NodeHit]:
        if not document_ids:
            if log_stage:
                log_rag(log_stage, returned=0, reason="no_document_ids", limit=limit, levels=levels)
            return []
        normalized = (query or "").strip()
        if not normalized:
            if log_stage:
                log_rag(log_stage, returned=0, reason="empty_query", limit=limit, levels=levels)
            return []

        indices = [name for name in _search_index_names() if get_client().indices.exists(index=name)]
        if not indices:
            if log_stage:
                log_rag(log_stage, returned=0, reason="no_sparse_indices", limit=limit, levels=levels)
            return []
        filters: list[dict[str, Any]] = [{"terms": {"document_id": document_ids}}]
        if levels:
            filters.append({"terms": {"level": levels}})
        for key, values in (metadata_filters or {}).items():
            clean = [str(v) for v in values if str(v).strip()]
            if clean:
                filters.append(_metadata_terms_filter(key, clean))
        query_plan = query_plan or build_sparse_query_plan(normalized)
        body = {
            "size": limit,
            "query": query_plan.to_bool_query(filters),
            "_source": [
                "node_id",
                "document_id",
                "ingest_run_id",
                "parent_id",
                "node_type",
                "level",
                "order_index",
                "title",
                "text",
                "metadata",
                *RETRIEVAL_INDEX_KEYWORD_FIELDS,
                *RETRIEVAL_INDEX_TEXT_FIELDS,
            ],
            "sort": [
                "_score",
                {"level": {"order": "desc"}},
                {"order_index": {"order": "asc"}},
            ],
        }
        t0 = time.perf_counter()
        response = await asyncio.to_thread(
            get_client().search,
            index=",".join(indices),
            body=body,
            ignore_unavailable=True,
        )
        hits = response.get("hits", {}).get("hits", [])
        out = [
            {
                **(item.get("_source") or {}),
                "sparse_score": float(item.get("_score") or 0.0),
            }
            for item in hits
        ]
        if log_stage:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            log_rag(
                log_stage,
                level="warning" if not out else "info",
                returned=len(out),
                limit=limit,
                levels=levels,
                document_ids=len(document_ids),
                query_len=len(normalized),
                metadata_filter_keys=sorted((metadata_filters or {}).keys()) or None,
                query_profile=query_plan.profile,
                structured_slots=query_plan.slots or None,
                queried_indices=indices,
                latency_ms=elapsed_ms,
            )
        return out

    async def replace_document_nodes(self, document_id: int, nodes: list[NodeIndexRecord]) -> None:
        await asyncio.to_thread(_replace_document_nodes_sync, document_id, nodes)
