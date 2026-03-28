"""Shared retrieval backend protocols and payload aliases."""

from __future__ import annotations

from typing import Any, Optional, Protocol

from .sparse_query_profiles import SparseQueryPlan


NodeHit = dict[str, Any]
NodeIndexRecord = dict[str, Any]
MetadataFilters = dict[str, list[str]]


class DenseBackend(Protocol):
    def search(
        self,
        query_vector: list[float],
        *,
        document_ids: list[int],
        limit: int,
        levels: Optional[list[int]] = None,
        metadata_filters: Optional[MetadataFilters] = None,
        log_stage: Optional[str] = None,
    ) -> list[NodeHit]: ...

    def replace_document_nodes(self, document_id: int, nodes: list[NodeIndexRecord]) -> None: ...


class SparseBackend(Protocol):
    async def search(
        self,
        document_ids: list[int],
        query: str,
        *,
        limit: int,
        levels: Optional[list[int]] = None,
        metadata_filters: Optional[MetadataFilters] = None,
        query_plan: Optional[SparseQueryPlan] = None,
        log_stage: Optional[str] = None,
    ) -> list[NodeHit]: ...

    async def replace_document_nodes(self, document_id: int, nodes: list[NodeIndexRecord]) -> None: ...
