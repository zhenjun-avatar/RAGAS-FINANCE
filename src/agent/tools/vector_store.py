"""Qdrant vector store wrapper with node ID alignment."""

from __future__ import annotations

import time
from typing import Any, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    SearchParams,
    VectorParams,
)

from core.config import config

from .rag_stage_log import log_rag
from .retrieval_fields import RETRIEVAL_INDEX_KEYWORD_FIELDS, RETRIEVAL_INDEX_TEXT_FIELDS

_client: Optional[QdrantClient] = None


def _point_payload_as_dict(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, dict) else {}
    return {}


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=f"http://{config.qdrant_host}:{config.qdrant_port}",
            api_key=config.qdrant_api_key,
            timeout=30.0,
            prefer_grpc=False,
        )
    return _client


def ensure_collection(vector_size: Optional[int] = None) -> None:
    client = get_client()
    size = int(vector_size or config.embedding_dimension)
    collections = client.get_collections().collections
    exists = any(item.name == config.qdrant_collection for item in collections)
    if not exists:
        client.create_collection(
            collection_name=config.qdrant_collection,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
        logger.info("[VectorStore] Created collection {}", config.qdrant_collection)


def _collection_exists(client: QdrantClient) -> bool:
    names = [c.name for c in client.get_collections().collections]
    return config.qdrant_collection in names


def delete_document_nodes(document_id: int) -> None:
    client = get_client()
    if not _collection_exists(client):
        return
    client.delete(
        collection_name=config.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
    )


def _as_int_document_id(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("document_id must not be bool")
    if isinstance(value, int):
        return value
    return int(value)


def upsert_nodes(nodes: list[dict[str, Any]]) -> None:
    if not nodes:
        return
    ensure_collection(len(nodes[0]["vector"]))
    client = get_client()
    points = [
        PointStruct(
            id=node["node_id"],
            vector=node["vector"],
            payload={
                "node_id": node["node_id"],
                "document_id": _as_int_document_id(node["document_id"]),
                "parent_id": node.get("parent_id"),
                "node_type": node["node_type"],
                "level": node["level"],
                "order_index": node["order_index"],
                "title": node.get("title"),
                "text_preview": node["text"][:1000],
                "metadata": node.get("metadata") or {},
                **{
                    field_name: node[field_name]
                    for field_name in RETRIEVAL_INDEX_KEYWORD_FIELDS + RETRIEVAL_INDEX_TEXT_FIELDS
                    if node.get(field_name) not in (None, "", [], {})
                },
            },
        )
        for node in nodes
    ]
    client.upsert(collection_name=config.qdrant_collection, points=points, wait=True)


def dense_search(
    query_vector: list[float],
    *,
    document_ids: list[int],
    limit: int,
    levels: Optional[list[int]] = None,
    parent_ids: Optional[list[str]] = None,
    metadata_filters: Optional[dict[str, list[str]]] = None,
    log_stage: Optional[str] = None,
) -> list[dict[str, Any]]:
    if not query_vector or not document_ids:
        if log_stage:
            log_rag(log_stage, returned=0, reason="no_vector_or_documents", limit=limit, levels=levels)
        return []
    ensure_collection(len(query_vector))
    doc_ids: list[int] = []
    for raw in document_ids:
        try:
            doc_ids.append(_as_int_document_id(raw))
        except (TypeError, ValueError):
            continue
    if not doc_ids:
        if log_stage:
            log_rag(log_stage, returned=0, reason="no_valid_document_ids", limit=limit, levels=levels)
        return []
    must_conditions = [
        FieldCondition(key="document_id", match=MatchAny(any=doc_ids)),
    ]
    if levels:
        must_conditions.append(FieldCondition(key="level", match=MatchAny(any=levels)))
    clean_parent_ids = [str(v).strip() for v in (parent_ids or []) if str(v).strip()]
    if clean_parent_ids:
        must_conditions.append(FieldCondition(key="parent_id", match=MatchAny(any=clean_parent_ids)))
    for key, values in (metadata_filters or {}).items():
        clean = [str(v) for v in values if str(v).strip()]
        if clean:
            must_conditions.append(FieldCondition(key=key, match=MatchAny(any=clean)))

    t0 = time.perf_counter()
    # qdrant-client >= 1.12: use query_points (search() was removed).
    response = get_client().query_points(
        collection_name=config.qdrant_collection,
        query=query_vector,
        query_filter=Filter(must=must_conditions),
        limit=limit,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=128, exact=False),
    )
    results = [
        {
            "node_id": str(item.id),
            "dense_score": item.score,
            **_point_payload_as_dict(item.payload),
        }
        for item in response.points
    ]
    if log_stage:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        top_scores = [round(float(r["dense_score"]), 6) for r in results[:3]]
        log_rag(
            log_stage,
            returned=len(results),
            limit=limit,
            levels=levels,
            document_ids=len(document_ids),
            parent_ids=len(clean_parent_ids) if clean_parent_ids else None,
            metadata_filter_keys=sorted((metadata_filters or {}).keys()) or None,
            top_dense_scores=top_scores or None,
            latency_ms=elapsed_ms,
        )
    return results
