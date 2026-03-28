#!/usr/bin/env python3
"""Delete ingested document data from PG + vector backends."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from core.config import config
from tools.node_repository import ensure_schema, get_pool


async def _delete_pg(document_id: int) -> int:
    """Return deleted row count in rag_documents."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        cmd = await conn.execute("DELETE FROM rag_documents WHERE id = $1", document_id)
    # asyncpg returns "DELETE <n>"
    try:
        return int(cmd.split()[-1])
    except Exception:
        return 0


async def _delete_one(document_id: int, *, skip_pg: bool) -> dict:
    deleted_pg = None
    if not skip_pg:
        deleted_pg = await _delete_pg(document_id)

    # Dense backend
    dense_backend_name = (config.dense_backend or "qdrant").strip().lower()
    if dense_backend_name == "qdrant":
        from tools.retrieval_backends.dense_qdrant import QdrantDenseBackend

        QdrantDenseBackend().replace_document_nodes(document_id, [])
    else:
        raise ValueError(f"Unsupported dense backend: {config.dense_backend!r}")

    # Sparse backend
    sparse_backend_name = (config.sparse_backend or "postgres").strip().lower()
    if sparse_backend_name == "postgres":
        from tools.retrieval_backends.sparse_postgres import PostgresSparseBackend

        await PostgresSparseBackend().replace_document_nodes(document_id, [])
    elif sparse_backend_name == "opensearch":
        try:
            from tools.retrieval_backends.sparse_opensearch import OpenSearchSparseBackend
        except Exception as exc:
            raise RuntimeError(
                "Sparse backend is opensearch, but opensearch dependency is missing. "
                "Please install opensearch-py in current Python environment."
            ) from exc
        await OpenSearchSparseBackend().replace_document_nodes(document_id, [])
    else:
        raise ValueError(f"Unsupported sparse backend: {config.sparse_backend!r}")

    return {
        "document_id": document_id,
        "deleted_pg_rows": deleted_pg,
        "dense_deleted": True,
        "sparse_deleted": True,
    }


async def main_async(document_ids: list[int], *, skip_pg: bool) -> dict:
    await ensure_schema()
    out = []
    for did in document_ids:
        out.append(await _delete_one(did, skip_pg=skip_pg))
    return {"success": True, "results": out}


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete ingested document data by document_id")
    parser.add_argument(
        "--document-id",
        type=int,
        nargs="+",
        required=True,
        help="One or more document IDs to delete",
    )
    parser.add_argument(
        "--skip-pg",
        action="store_true",
        help="Skip deleting rag_documents row (only clear vector/sparse backends)",
    )
    args = parser.parse_args()

    result = asyncio.run(main_async(args.document_id, skip_pg=args.skip_pg))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
