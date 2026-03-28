"""Factories for retrieval backend selection."""

from __future__ import annotations

from functools import lru_cache

from core.config import config

from .dense_qdrant import QdrantDenseBackend
from .sparse_opensearch import OpenSearchSparseBackend
from .sparse_postgres import PostgresSparseBackend
from .types import DenseBackend, SparseBackend


@lru_cache(maxsize=1)
def get_dense_backend() -> DenseBackend:
    backend = (config.dense_backend or "qdrant").strip().lower()
    if backend == "qdrant":
        return QdrantDenseBackend()
    raise ValueError(f"Unsupported dense backend: {config.dense_backend!r}")


@lru_cache(maxsize=1)
def get_sparse_backend() -> SparseBackend:
    backend = (config.sparse_backend or "postgres").strip().lower()
    if backend == "postgres":
        return PostgresSparseBackend()
    if backend == "opensearch":
        return OpenSearchSparseBackend()
    raise ValueError(f"Unsupported sparse backend: {config.sparse_backend!r}")
