"""PostgreSQL storage for SEC company-facts observations (structured DD layer)."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any, Optional

from ..node_repository import get_pool


async def ensure_financial_facts_schema() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sec_financial_observations (
                id BIGSERIAL PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                ingest_run_id UUID REFERENCES rag_ingest_runs(id) ON DELETE SET NULL,
                cik BIGINT,
                entity_name TEXT,
                taxonomy TEXT NOT NULL,
                metric_key TEXT NOT NULL,
                metric_label TEXT,
                metric_description TEXT,
                unit TEXT NOT NULL,
                value_numeric DOUBLE PRECISION,
                value_text TEXT,
                period_end DATE,
                filed_date DATE,
                form TEXT,
                fy INTEGER,
                fp TEXT,
                accn TEXT,
                frame TEXT,
                raw_entry JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_sec_fin_obs_document_id
                ON sec_financial_observations(document_id);
            CREATE INDEX IF NOT EXISTS idx_sec_fin_obs_cik_filed
                ON sec_financial_observations(cik, filed_date DESC NULLS LAST);
            CREATE INDEX IF NOT EXISTS idx_sec_fin_obs_metric
                ON sec_financial_observations(taxonomy, metric_key);
            """
        )


def _parse_date(val: Any) -> Optional[date]:
    if val is None:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    s = str(val).strip()[:10]
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None
    return None


async def replace_sec_observations(
    document_id: int,
    ingest_run_id: str,
    rows: list[dict[str, Any]],
) -> int:
    """Replace all observations for a document. Returns inserted count."""
    pool = await get_pool()
    if not rows:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM sec_financial_observations WHERE document_id = $1",
                document_id,
            )
        return 0

    db_rows: list[tuple[Any, ...]] = []
    for r in rows:
        db_rows.append(
            (
                document_id,
                ingest_run_id,
                r.get("cik"),
                r.get("entity_name"),
                r["taxonomy"],
                r["metric_key"],
                r.get("metric_label"),
                r.get("metric_description"),
                r["unit"],
                r.get("value_numeric"),
                r.get("value_text"),
                _parse_date(r.get("period_end")),
                _parse_date(r.get("filed_date")),
                str(r["form"]) if r.get("form") is not None else None,
                int(r["fy"]) if r.get("fy") is not None else None,
                str(r["fp"]) if r.get("fp") is not None else None,
                str(r["accn"]) if r.get("accn") is not None else None,
                str(r["frame"]) if r.get("frame") is not None else None,
                json.dumps(r.get("raw_entry") or {}),
            )
        )

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM sec_financial_observations WHERE document_id = $1",
                document_id,
            )
            await conn.executemany(
                """
                INSERT INTO sec_financial_observations (
                    document_id, ingest_run_id, cik, entity_name, taxonomy, metric_key,
                    metric_label, metric_description, unit, value_numeric, value_text,
                    period_end, filed_date, form, fy, fp, accn, frame, raw_entry
                )
                VALUES (
                    $1, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19::jsonb
                )
                """,
                db_rows,
            )
    return len(db_rows)


async def query_observations(
    *,
    cik: int | None = None,
    document_id: int | None = None,
    taxonomy: str | None = None,
    metric_key: str | None = None,
    form: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Filter observations (SQL-first due diligence)."""
    pool = await get_pool()
    parts: list[str] = []
    args: list[Any] = []
    if cik is not None:
        args.append(cik)
        parts.append(f"cik = ${len(args)}")
    if document_id is not None:
        args.append(document_id)
        parts.append(f"document_id = ${len(args)}")
    if taxonomy:
        args.append(taxonomy)
        parts.append(f"taxonomy = ${len(args)}")
    if metric_key:
        args.append(metric_key)
        parts.append(f"metric_key = ${len(args)}")
    if form:
        args.append(form)
        parts.append(f"form = ${len(args)}")
    args.append(limit)
    where_sql = " AND ".join(parts) if parts else "TRUE"
    lim = len(args)
    sql = f"""
        SELECT
            id, document_id, ingest_run_id::text AS ingest_run_id, cik, entity_name,
            taxonomy, metric_key, metric_label, unit, value_numeric, value_text,
            period_end, filed_date, form, fy, fp, accn, frame
        FROM sec_financial_observations
        WHERE {where_sql}
        ORDER BY filed_date DESC NULLS LAST, id DESC
        LIMIT ${lim}
    """
    rows = await pool.fetch(sql, *args)
    return [dict(r) for r in rows]


async def count_observations(document_id: int) -> int:
    pool = await get_pool()
    v = await pool.fetchval(
        "SELECT COUNT(*)::bigint FROM sec_financial_observations WHERE document_id = $1",
        document_id,
    )
    return int(v or 0)


async def document_ids_with_sec_observations(document_ids: list[int]) -> set[int]:
    """Which of the given documents have ingested SEC fact rows."""
    if not document_ids:
        return set()
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT DISTINCT document_id
        FROM sec_financial_observations
        WHERE document_id = ANY($1::bigint[])
        """,
        document_ids,
    )
    return {int(r["document_id"]) for r in rows}


async def query_observations_for_documents(
    document_ids: list[int],
    *,
    limit: int = 150,
    forms: list[str] | None = None,
    accns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Latest observations across one or more documents (SQL-first bundle).

    When ``forms`` is set (e.g. {"10-Q"}), only those SEC form values are returned—
    used for the 'recent tail' so explicit 10-Q/10-K questions do not drown in other forms.
    Hint queries stay unfiltered so cross-form metrics (e.g. YoY) still surface.
    """
    if not document_ids:
        return []
    pool = await get_pool()
    clean_accns = [str(v).strip() for v in (accns or []) if str(v).strip()]
    if forms and clean_accns:
        rows = await pool.fetch(
            """
            SELECT
                id, document_id, ingest_run_id::text AS ingest_run_id, cik, entity_name,
                taxonomy, metric_key, metric_label, unit, value_numeric, value_text,
                period_end, filed_date, form, fy, fp, accn, frame
            FROM sec_financial_observations
            WHERE document_id = ANY($1::bigint[])
              AND form = ANY($2::text[])
              AND accn = ANY($3::text[])
            ORDER BY filed_date DESC NULLS LAST, id DESC
            LIMIT $4
            """,
            document_ids,
            forms,
            clean_accns,
            limit,
        )
    elif forms:
        rows = await pool.fetch(
            """
            SELECT
                id, document_id, ingest_run_id::text AS ingest_run_id, cik, entity_name,
                taxonomy, metric_key, metric_label, unit, value_numeric, value_text,
                period_end, filed_date, form, fy, fp, accn, frame
            FROM sec_financial_observations
            WHERE document_id = ANY($1::bigint[])
              AND form = ANY($2::text[])
            ORDER BY filed_date DESC NULLS LAST, id DESC
            LIMIT $3
            """,
            document_ids,
            forms,
            limit,
        )
    elif clean_accns:
        rows = await pool.fetch(
            """
            SELECT
                id, document_id, ingest_run_id::text AS ingest_run_id, cik, entity_name,
                taxonomy, metric_key, metric_label, unit, value_numeric, value_text,
                period_end, filed_date, form, fy, fp, accn, frame
            FROM sec_financial_observations
            WHERE document_id = ANY($1::bigint[])
              AND accn = ANY($2::text[])
            ORDER BY filed_date DESC NULLS LAST, id DESC
            LIMIT $3
            """,
            document_ids,
            clean_accns,
            limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT
                id, document_id, ingest_run_id::text AS ingest_run_id, cik, entity_name,
                taxonomy, metric_key, metric_label, unit, value_numeric, value_text,
                period_end, filed_date, form, fy, fp, accn, frame
            FROM sec_financial_observations
            WHERE document_id = ANY($1::bigint[])
            ORDER BY filed_date DESC NULLS LAST, id DESC
            LIMIT $2
            """,
            document_ids,
            limit,
        )
    return [dict(r) for r in rows]


_OBS_SELECT = """
        SELECT
            id, document_id, ingest_run_id::text AS ingest_run_id, cik, entity_name,
            taxonomy, metric_key, metric_label, unit, value_numeric, value_text,
            period_end, filed_date, form, fy, fp, accn, frame
        FROM sec_financial_observations
"""


async def query_observations_by_metric_hints(
    document_ids: list[int],
    hints: list[str],
    *,
    limit: int = 120,
    accns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Rows whose metric_key or metric_label matches question-derived hints (ILIKE)."""
    if not document_ids or not hints:
        return []
    clean: list[str] = []
    for h in hints:
        s = re.sub(r"[^A-Za-z0-9_]", "", str(h))[:96]
        if len(s) >= 6:
            clean.append(s)
    clean = list(dict.fromkeys(clean))[:12]
    if not clean:
        return []
    pool = await get_pool()
    parts: list[str] = []
    args: list[Any] = [document_ids]
    clean_accns = [str(v).strip() for v in (accns or []) if str(v).strip()]
    where_parts = ["document_id = ANY($1::bigint[])"]
    if clean_accns:
        args.append(clean_accns)
        where_parts.append(f"accn = ANY(${len(args)}::text[])")
    for h in clean:
        pat = f"%{h}%"
        args.append(pat)
        idx = len(args)
        parts.append(
            f"(metric_key ILIKE ${idx} OR COALESCE(metric_label, '') ILIKE ${idx})"
        )
    args.append(limit)
    lim_idx = len(args)
    or_sql = " OR ".join(parts)
    sql = f"""
        {_OBS_SELECT.strip()}
        WHERE {" AND ".join(where_parts)}
          AND ({or_sql})
        ORDER BY filed_date DESC NULLS LAST, id DESC
        LIMIT ${lim_idx}
    """
    rows = await pool.fetch(sql, *args)
    return [dict(r) for r in rows]


async def query_observations_by_filters(
    document_ids: list[int],
    *,
    metric_keys: list[str] | None = None,
    forms: list[str] | None = None,
    period_end_dates: list[str] | None = None,
    period_years: list[int] | None = None,
    accns: list[str] | None = None,
    limit: int = 80,
) -> list[dict[str, Any]]:
    """Structured SQL query for finance questions.

    This is the preferred path for companyfacts Q&A: use exact-ish filters first,
    then let callers optionally fall back to broader hint/recent queries.
    """
    if not document_ids:
        return []

    args: list[Any] = [document_ids]
    parts = ["document_id = ANY($1::bigint[])"]

    clean_metric_keys = [str(v).strip() for v in (metric_keys or []) if str(v).strip()]
    if clean_metric_keys:
        args.append(clean_metric_keys)
        parts.append(f"metric_key = ANY(${len(args)}::text[])")

    clean_forms = [str(v).strip() for v in (forms or []) if str(v).strip()]
    if clean_forms:
        args.append(clean_forms)
        parts.append(f"form = ANY(${len(args)}::text[])")

    clean_period_end_dates = [str(v).strip() for v in (period_end_dates or []) if str(v).strip()]
    if clean_period_end_dates:
        args.append(clean_period_end_dates)
        parts.append(f"period_end = ANY(${len(args)}::date[])")

    clean_period_years = sorted({int(v) for v in (period_years or []) if int(v) >= 1900})
    if clean_period_years:
        args.append(clean_period_years)
        parts.append(f"EXTRACT(YEAR FROM period_end)::int = ANY(${len(args)}::int[])")

    clean_accns = [str(v).strip() for v in (accns or []) if str(v).strip()]
    if clean_accns:
        args.append(clean_accns)
        parts.append(f"accn = ANY(${len(args)}::text[])")

    args.append(limit)
    sql = f"""
        {_OBS_SELECT.strip()}
        WHERE {" AND ".join(parts)}
        ORDER BY
            period_end DESC NULLS LAST,
            filed_date DESC NULLS LAST,
            id DESC
        LIMIT ${len(args)}
    """
    pool = await get_pool()
    rows = await pool.fetch(sql, *args)
    return [dict(r) for r in rows]
