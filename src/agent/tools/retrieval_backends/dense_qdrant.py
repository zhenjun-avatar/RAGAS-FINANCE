"""Qdrant-backed dense retrieval implementation."""

from __future__ import annotations

from typing import Optional

from .types import NodeHit, NodeIndexRecord
from ..vector_store import delete_document_nodes, dense_search, upsert_nodes


class QdrantDenseBackend:
    def search(
        self,
        query_vector: list[float],
        *,
        document_ids: list[int],
        limit: int,
        levels: Optional[list[int]] = None,
        metadata_filters: Optional[dict[str, list[str]]] = None,
        log_stage: Optional[str] = None,
    ) -> list[NodeHit]:
        return dense_search(
            query_vector,
            document_ids=document_ids,
            limit=limit,
            levels=levels,
            metadata_filters=metadata_filters,
            log_stage=log_stage,
        )

    def replace_document_nodes(self, document_id: int, nodes: list[NodeIndexRecord]) -> None:
        delete_document_nodes(document_id)
        if nodes:
            upsert_nodes(nodes)
