"""Node-centric PostgreSQL repository for RAG metadata and sparse retrieval."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import asyncpg
from loguru import logger

from core.config import config

from .document_display import build_document_catalog_row, metadata_as_dict
from .rag_stage_log import log_rag


@dataclass
class NodeRecord:
    node_id: str
    document_id: int
    ingest_run_id: str
    node_type: str
    level: int
    order_index: int
    text: str
    parent_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    title: Optional[str] = None


_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=config.effective_database_url,
            min_size=2,
            max_size=10,
        )
    return _pool


async def ensure_schema() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_documents (
                id BIGINT PRIMARY KEY,
                title TEXT,
                source_uri TEXT,
                file_type TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                latest_ingest_run_id UUID,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS rag_ingest_runs (
                id UUID PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                pipeline_version TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                finished_at TIMESTAMPTZ
            );

            CREATE TABLE IF NOT EXISTS rag_nodes (
                id UUID PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                ingest_run_id UUID NOT NULL REFERENCES rag_ingest_runs(id) ON DELETE CASCADE,
                parent_id UUID REFERENCES rag_nodes(id) ON DELETE CASCADE,
                node_type TEXT NOT NULL,
                level INTEGER NOT NULL DEFAULT 0,
                order_index INTEGER NOT NULL DEFAULT 0,
                title TEXT,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                has_vector BOOLEAN NOT NULL DEFAULT FALSE,
                search_vector tsvector GENERATED ALWAYS AS (
                    to_tsvector('simple', COALESCE(title, '') || ' ' || COALESCE(text, ''))
                ) STORED,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS rag_evaluation_jobs (
                id UUID PRIMARY KEY,
                trace_id TEXT NOT NULL,
                document_ids BIGINT[] NOT NULL DEFAULT '{}',
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                context_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                status TEXT NOT NULL DEFAULT 'pending',
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                started_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ,
                error TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_rag_nodes_document_id ON rag_nodes(document_id);
            CREATE INDEX IF NOT EXISTS idx_rag_nodes_parent_id ON rag_nodes(parent_id);
            CREATE INDEX IF NOT EXISTS idx_rag_nodes_level ON rag_nodes(level);
            CREATE INDEX IF NOT EXISTS idx_rag_nodes_ingest_run_id ON rag_nodes(ingest_run_id);
            CREATE INDEX IF NOT EXISTS idx_rag_nodes_search_vector ON rag_nodes USING GIN(search_vector);
            CREATE INDEX IF NOT EXISTS idx_rag_evaluation_jobs_status ON rag_evaluation_jobs(status, created_at);
            """
        )
    from tools.finance.financial_facts_repository import ensure_financial_facts_schema

    await ensure_financial_facts_schema()


async def upsert_document(
    document_id: int,
    *,
    title: Optional[str],
    source_uri: str,
    file_type: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO rag_documents (id, title, source_uri, file_type, metadata)
        VALUES ($1, $2, $3, $4, COALESCE($5::jsonb, '{}'::jsonb))
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            source_uri = EXCLUDED.source_uri,
            file_type = EXCLUDED.file_type,
            metadata = rag_documents.metadata || EXCLUDED.metadata,
            updated_at = NOW()
        """,
        document_id,
        title,
        source_uri,
        file_type,
        json.dumps(metadata or {}),
    )


async def start_ingest_run(document_id: int, metadata: Optional[dict[str, Any]] = None) -> str:
    pool = await get_pool()
    run_id = str(uuid.uuid4())
    await pool.execute(
        """
        INSERT INTO rag_ingest_runs (id, document_id, pipeline_version, status, metadata)
        VALUES ($1::uuid, $2, $3, 'running', COALESCE($4::jsonb, '{}'::jsonb))
        """,
        run_id,
        document_id,
        config.rag_pipeline_version,
        json.dumps(metadata or {}),
    )
    await pool.execute(
        """
        UPDATE rag_documents
        SET latest_ingest_run_id = $2::uuid, updated_at = NOW()
        WHERE id = $1
        """,
        document_id,
        run_id,
    )
    return run_id


async def finish_ingest_run(run_id: str, *, error: Optional[str] = None) -> None:
    pool = await get_pool()
    status = "failed" if error else "completed"
    await pool.execute(
        """
        UPDATE rag_ingest_runs
        SET status = $2, error = $3, finished_at = NOW()
        WHERE id = $1::uuid
        """,
        run_id,
        status,
        error,
    )


async def replace_document_nodes(
    document_id: int,
    ingest_run_id: str,
    nodes: Iterable[NodeRecord],
) -> None:
    pool = await get_pool()
    rows = [
        (
            item.node_id,
            item.document_id,
            item.ingest_run_id,
            item.parent_id,
            item.node_type,
            item.level,
            item.order_index,
            item.title,
            item.text,
            json.dumps(item.metadata or {}),
        )
        for item in nodes
    ]

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM rag_nodes WHERE document_id = $1", document_id)
            if rows:
                await conn.executemany(
                    """
                    INSERT INTO rag_nodes (
                        id, document_id, ingest_run_id, parent_id, node_type, level,
                        order_index, title, text, metadata
                    )
                    VALUES ($1::uuid, $2, $3::uuid, $4::uuid, $5, $6, $7, $8, $9, $10::jsonb)
                    """,
                    rows,
                )
    logger.info(
        "[NodeRepository] Replaced {} nodes for document {}",
        len(rows),
        document_id,
    )


async def mark_nodes_vectorized(node_ids: Iterable[str]) -> None:
    values = list(node_ids)
    if not values:
        return
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE rag_nodes
        SET has_vector = TRUE
        WHERE id = ANY($1::uuid[])
        """,
        values,
    )


async def fetch_nodes(node_ids: Iterable[str]) -> list[dict[str, Any]]:
    values = list(dict.fromkeys(node_ids))
    if not values:
        return []
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT
            id::text AS node_id,
            document_id,
            ingest_run_id::text AS ingest_run_id,
            parent_id::text AS parent_id,
            node_type,
            level,
            order_index,
            title,
            text,
            metadata
        FROM rag_nodes
        WHERE id = ANY($1::uuid[])
        """,
        values,
    )
    mapped = {row["node_id"]: dict(row) for row in rows}
    return [mapped[node_id] for node_id in values if node_id in mapped]


async def fetch_node(node_id: str) -> Optional[dict[str, Any]]:
    rows = await fetch_nodes([node_id])
    return rows[0] if rows else None


async def list_document_nodes(document_id: int) -> list[dict[str, Any]]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT
            id::text AS node_id,
            document_id,
            ingest_run_id::text AS ingest_run_id,
            parent_id::text AS parent_id,
            node_type,
            level,
            order_index,
            title,
            text,
            metadata
        FROM rag_nodes
        WHERE document_id = $1
        ORDER BY level DESC, order_index ASC
        """,
        document_id,
    )
    return [dict(row) for row in rows]


async def list_available_document_ids(*, limit: int = 500) -> list[int]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT DISTINCT document_id
        FROM rag_nodes
        ORDER BY document_id ASC
        LIMIT $1
        """,
        max(1, min(int(limit), 5000)),
    )
    return [int(row["document_id"]) for row in rows]


async def list_document_catalog(*, limit: int = 500) -> list[dict[str, Any]]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT
            n.document_id AS document_id,
            d.title,
            d.source_uri,
            d.file_type,
            d.metadata,
            COALESCE(n.node_count, 0) AS node_count
        FROM (
            SELECT document_id, COUNT(*)::bigint AS node_count
            FROM rag_nodes
            GROUP BY document_id
        ) n
        LEFT JOIN rag_documents d ON d.id = n.document_id
        ORDER BY n.document_id ASC
        LIMIT $1
        """,
        max(1, min(int(limit), 5000)),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        metadata = metadata_as_dict(row.get("metadata"))
        source_uri = row.get("source_uri")
        su = str(source_uri) if source_uri else None
        doc_id = int(row["document_id"])
        catalog = build_document_catalog_row(
            document_id=doc_id,
            title=row["title"],
            file_type=row["file_type"],
            metadata=metadata,
            source_uri=su,
        )
        out.append(
            {
                "document_id": doc_id,
                **catalog,
                "file_type": row["file_type"],
                "node_count": int(row["node_count"] or 0),
                "raw_title": row["title"],
                "source_uri": source_uri,
            }
        )
    return out


def _catalog_period_years_from_metadata(metadata: dict[str, Any]) -> list[int]:
    """Derive filing-catalog years from unified ``finance_period`` or legacy keys."""
    seen: set[int] = set()
    out: list[int] = []
    for key in ("finance_period", "finance_period_years", "period_years"):
        raw = metadata.get(key)
        if raw is None:
            continue
        values = raw if isinstance(raw, list) else [raw]
        for item in values:
            s = str(item).strip()
            if len(s) < 4 or not s[:4].isdigit():
                continue
            y = int(s[:4])
            if 1990 <= y <= 2100 and y not in seen:
                seen.add(y)
                out.append(y)
    return out


async def list_sec_filing_catalog(
    *,
    document_ids: Iterable[int] | None = None,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    pool = await get_pool()
    doc_ids = [int(item) for item in document_ids or []]
    if doc_ids:
        rows = await pool.fetch(
            """
            SELECT
                d.id AS document_id,
                d.title,
                d.source_uri,
                d.file_type,
                d.metadata
            FROM rag_documents d
            WHERE d.id = ANY($1::bigint[])
              AND COALESCE(NULLIF(d.metadata->>'sec_accession', ''), NULLIF(d.metadata->>'accn', '')) IS NOT NULL
            ORDER BY d.id ASC
            LIMIT $2
            """,
            doc_ids,
            max(1, min(int(limit), 5000)),
        )
    else:
        rows = await pool.fetch(
            """
            SELECT
                d.id AS document_id,
                d.title,
                d.source_uri,
                d.file_type,
                d.metadata
            FROM rag_documents d
            WHERE COALESCE(NULLIF(d.metadata->>'sec_accession', ''), NULLIF(d.metadata->>'accn', '')) IS NOT NULL
            ORDER BY d.id ASC
            LIMIT $1
            """,
            max(1, min(int(limit), 5000)),
        )
    out: list[dict[str, Any]] = []
    for row in rows:
        metadata = metadata_as_dict(row.get("metadata"))
        out.append(
            {
                "document_id": int(row["document_id"]),
                "title": row.get("title"),
                "source_uri": row.get("source_uri"),
                "file_type": row.get("file_type"),
                "accession": metadata.get("sec_accession") or metadata.get("accn"),
                "form": metadata.get("form"),
                "filed": metadata.get("sec_filing_date") or metadata.get("filed"),
                "period_end_dates": metadata.get("finance_period_end_dates") or metadata.get("period_end_dates"),
                "period_years": _catalog_period_years_from_metadata(metadata)
                or metadata.get("finance_period_years")
                or metadata.get("period_years"),
                "cik": metadata.get("cik"),
                "entity_name": metadata.get("entity_name"),
                "metadata": metadata,
            }
        )
    return out


async def fetch_children(parent_ids: Iterable[str], limit_per_parent: int = 8) -> list[dict[str, Any]]:
    values = list(dict.fromkeys(parent_ids))
    if not values:
        return []
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT * FROM (
            SELECT
                id::text AS node_id,
                document_id,
                ingest_run_id::text AS ingest_run_id,
                parent_id::text AS parent_id,
                node_type,
                level,
                order_index,
                title,
                text,
                metadata,
                ROW_NUMBER() OVER (PARTITION BY parent_id ORDER BY order_index ASC) AS row_num
            FROM rag_nodes
            WHERE parent_id = ANY($1::uuid[])
        ) ranked
        WHERE row_num <= $2
        ORDER BY level ASC, order_index ASC
        """,
        values,
        limit_per_parent,
    )
    return [dict(row) for row in rows]


async def fetch_siblings(node_id: str, *, limit: int = 0) -> list[dict[str, Any]]:
    """Return level=0 siblings of a leaf node (same parent_id).

    Used for section-bounded context expansion.  If the node has no parent,
    the node itself is returned as a single-item list.

    Args:
        limit: Maximum number of siblings to return (0 = no limit).
    """
    node = await fetch_node(node_id)
    if not node:
        return []
    parent_id = node.get("parent_id")
    if not parent_id:
        return [node]
    pool = await get_pool()
    cap = max(1, int(limit)) if limit > 0 else None
    if cap is not None:
        rows = await pool.fetch(
            """
            SELECT
                id::text AS node_id,
                document_id,
                ingest_run_id::text AS ingest_run_id,
                parent_id::text AS parent_id,
                node_type,
                level,
                order_index,
                title,
                text,
                metadata
            FROM rag_nodes
            WHERE parent_id = $1::uuid AND level = 0
            ORDER BY order_index ASC
            LIMIT $2
            """,
            parent_id,
            cap,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT
                id::text AS node_id,
                document_id,
                ingest_run_id::text AS ingest_run_id,
                parent_id::text AS parent_id,
                node_type,
                level,
                order_index,
                title,
                text,
                metadata
            FROM rag_nodes
            WHERE parent_id = $1::uuid AND level = 0
            ORDER BY order_index ASC
            """,
            parent_id,
        )
    return [dict(r) for r in rows]


async def fetch_leaf_descendants(section_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Return all level=0 leaf descendants of the given section nodes.

    Uses a recursive CTE to traverse the section tree of arbitrary depth, then
    filters to level=0.  Duplicate node_ids are eliminated by the DISTINCT join.
    """
    values = list(dict.fromkeys(str(v) for v in section_ids if str(v).strip()))
    if not values:
        return []
    pool = await get_pool()
    rows = await pool.fetch(
        """
        WITH RECURSIVE tree AS (
            SELECT id AS node_id, parent_id
            FROM rag_nodes
            WHERE id = ANY($1::uuid[])

            UNION ALL

            SELECT n.id, n.parent_id
            FROM rag_nodes n
            INNER JOIN tree t ON n.parent_id = t.node_id
        )
        SELECT DISTINCT ON (n.id)
            n.id::text AS node_id,
            n.document_id,
            n.ingest_run_id::text AS ingest_run_id,
            n.parent_id::text AS parent_id,
            n.node_type,
            n.level,
            n.order_index,
            n.title,
            n.text,
            n.metadata
        FROM rag_nodes n
        INNER JOIN tree t ON n.id = t.node_id
        WHERE n.level = 0
        ORDER BY n.id, n.document_id, n.order_index ASC
        """,
        values,
    )
    return sorted([dict(r) for r in rows], key=lambda r: (r["document_id"], r["order_index"]))


async def fetch_neighbors(node_id: str, radius: int = 1) -> list[dict[str, Any]]:
    node = await fetch_node(node_id)
    if not node:
        return []
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT
            id::text AS node_id,
            document_id,
            ingest_run_id::text AS ingest_run_id,
            parent_id::text AS parent_id,
            node_type,
            level,
            order_index,
            title,
            text,
            metadata
        FROM rag_nodes
        WHERE document_id = $1
          AND level = $2
          AND order_index BETWEEN $3 AND $4
        ORDER BY order_index ASC
        """,
        node["document_id"],
        node["level"],
        max(0, node["order_index"] - radius),
        node["order_index"] + radius,
    )
    return [dict(row) for row in rows]


async def sparse_search(
    document_ids: list[int],
    query: str,
    *,
    limit: int,
    levels: Optional[list[int]] = None,
    parent_ids: Optional[list[str]] = None,
    metadata_filters: Optional[dict[str, list[str]]] = None,
    log_stage: Optional[str] = None,
) -> list[dict[str, Any]]:
    if not document_ids:
        if log_stage:
            log_rag(log_stage, returned=0, reason="no_document_ids", limit=limit, levels=levels)
        return []
    pool = await get_pool()
    normalized = (query or "").strip()
    if not normalized:
        if log_stage:
            log_rag(log_stage, returned=0, reason="empty_query", limit=limit, levels=levels)
        return []
    t0 = time.perf_counter()
    filter_clauses: list[str] = []
    filter_args: list[Any] = []
    parent_filter_sql = ""
    clean_parent_ids = [str(v).strip() for v in (parent_ids or []) if str(v).strip()]
    base_idx = 4 if levels else 3
    if clean_parent_ids:
        filter_args.append(clean_parent_ids)
        parent_ids_idx = base_idx + len(filter_args) - 1
        parent_filter_sql = f"\n              AND parent_id = ANY(${parent_ids_idx}::uuid[])"
    for key, values in (metadata_filters or {}).items():
        clean = [str(v) for v in values if str(v).strip()]
        if not clean:
            continue
        filter_args.append(key)
        key_idx = base_idx + len(filter_args) - 1
        filter_args.append(clean)
        values_idx = base_idx + len(filter_args) - 1
        filter_clauses.append(
            f"COALESCE(metadata->'_retrieval_fields'->>${key_idx}, '[]')::jsonb ?| ${values_idx}::text[]"
        )
    filter_sql = ""
    if filter_clauses:
        filter_sql = "\n              AND " + "\n              AND ".join(filter_clauses)

    if levels:
        rows = await pool.fetch(
            f"""
            WITH query AS (
                SELECT websearch_to_tsquery('simple', $2) AS q
            )
            SELECT
                id::text AS node_id,
                document_id,
                ingest_run_id::text AS ingest_run_id,
                parent_id::text AS parent_id,
                node_type,
                level,
                order_index,
                title,
                text,
                metadata,
                ts_rank_cd(search_vector, query.q) AS sparse_score
            FROM rag_nodes, query
            WHERE document_id = ANY($1::bigint[])
              AND level = ANY($3::int[])
              AND search_vector @@ query.q
              {parent_filter_sql}
              {filter_sql}
            ORDER BY sparse_score DESC, level DESC, order_index ASC
            LIMIT $4
            """,
            document_ids,
            normalized,
            levels,
            limit,
            *filter_args,
        )
    else:
        rows = await pool.fetch(
            f"""
            WITH query AS (
                SELECT websearch_to_tsquery('simple', $2) AS q
            )
            SELECT
                id::text AS node_id,
                document_id,
                ingest_run_id::text AS ingest_run_id,
                parent_id::text AS parent_id,
                node_type,
                level,
                order_index,
                title,
                text,
                metadata,
                ts_rank_cd(search_vector, query.q) AS sparse_score
            FROM rag_nodes, query
            WHERE document_id = ANY($1::bigint[])
              AND search_vector @@ query.q
              {parent_filter_sql}
              {filter_sql}
            ORDER BY sparse_score DESC, level DESC, order_index ASC
            LIMIT $3
            """,
            document_ids,
            normalized,
            limit,
            *filter_args,
        )

    out = [dict(row) for row in rows]
    if log_stage:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        lvl = "warning" if not out else "info"
        log_rag(
            log_stage,
            level=lvl,
            returned=len(out),
            limit=limit,
            levels=levels,
            document_ids=len(document_ids),
            parent_ids=len(clean_parent_ids) if clean_parent_ids else None,
            query_len=len(normalized),
            metadata_filter_keys=sorted((metadata_filters or {}).keys()) or None,
            latency_ms=elapsed_ms,
        )
    return out


async def enqueue_evaluation_job(
    *,
    trace_id: str,
    document_ids: list[int],
    query: str,
    answer: str,
    context_json: list[dict[str, Any]],
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    pool = await get_pool()
    job_id = str(uuid.uuid4())
    await pool.execute(
        """
        INSERT INTO rag_evaluation_jobs (
            id, trace_id, document_ids, query, answer, context_json, metadata
        )
        VALUES ($1::uuid, $2, $3::bigint[], $4, $5, $6::jsonb, COALESCE($7::jsonb, '{}'::jsonb))
        """,
        job_id,
        trace_id,
        document_ids,
        query,
        answer,
        json.dumps(context_json, ensure_ascii=False),
        json.dumps(metadata or {}, ensure_ascii=False),
    )
    return job_id


async def list_pending_evaluation_jobs(limit: int) -> list[dict[str, Any]]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id::text AS id, trace_id, document_ids, query, answer, context_json, metadata
        FROM rag_evaluation_jobs
        WHERE status = 'pending'
        ORDER BY created_at ASC
        LIMIT $1
        """,
        limit,
    )
    return [dict(row) for row in rows]


async def complete_evaluation_job(
    job_id: str,
    *,
    error: Optional[str] = None,
    skipped_reason: Optional[str] = None,
) -> None:
    pool = await get_pool()
    if skipped_reason:
        status = "skipped"
        err = skipped_reason
    elif error:
        status = "failed"
        err = error
    else:
        status = "completed"
        err = None
    await pool.execute(
        """
        UPDATE rag_evaluation_jobs
        SET status = $2, error = $3, finished_at = NOW()
        WHERE id = $1::uuid
        """,
        job_id,
        status,
        err,
    )
