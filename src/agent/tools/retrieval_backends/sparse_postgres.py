"""Backward-compatible PostgreSQL sparse retrieval backend."""

from __future__ import annotations

from typing import Optional

from .sparse_query_profiles import SparseQueryPlan
from .types import NodeHit, NodeIndexRecord
from ..node_repository import sparse_search


class PostgresSparseBackend:
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
        return await sparse_search(
            document_ids,
            query,
            limit=limit,
            levels=levels,
            metadata_filters=metadata_filters,
            log_stage=log_stage,
        )

    async def replace_document_nodes(self, document_id: int, nodes: list[NodeIndexRecord]) -> None:
        # PostgreSQL stores canonical nodes directly via node_repository.
        return None
