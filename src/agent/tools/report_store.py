"""Persistent storage for ask results under src/agent/report."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AGENT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = AGENT_ROOT / "report"
DETAIL_DIR = REPORT_DIR / "detail"
INDEX_FILE = REPORT_DIR / "index.jsonl"
SCHEMA_VERSION = "report.v1"

_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record_path(trace_id: str) -> Path:
    return REPORT_DIR / f"ask_result_{trace_id}.json"


def _detail_path(trace_id: str) -> Path:
    return DETAIL_DIR / f"ask_detail_{trace_id}.json"


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _short_text(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max(1, int(limit)):
        return text
    return text[: max(1, int(limit)) - 3].rstrip() + "..."


def _build_detail_snapshot(
    *,
    trace_id: str,
    created_at: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> dict[str, Any]:
    pipeline_trace = response_payload.get("pipeline_trace") if isinstance(response_payload.get("pipeline_trace"), dict) else {}
    rerank_hits = (
        ((pipeline_trace.get("retrieval_rerank_hits") or {}).get("output") or [])
        if isinstance(pipeline_trace.get("retrieval_rerank_hits"), dict)
        else []
    )
    evidence_ui = response_payload.get("evidence_ui") if isinstance(response_payload.get("evidence_ui"), dict) else {}
    narrative_cards = (
        ((evidence_ui.get("evidence") or {}).get("narrative_cards") or [])
        if isinstance(evidence_ui.get("evidence"), dict)
        else []
    )
    rerank_compare = (
        (pipeline_trace.get("retrieval_rerank_compare") or {})
        if isinstance(pipeline_trace.get("retrieval_rerank_compare"), dict)
        else {}
    )
    finance_route = (pipeline_trace.get("finance_route") or {}) if isinstance(pipeline_trace.get("finance_route"), dict) else {}
    external = response_payload.get("external_evaluation") if isinstance(response_payload.get("external_evaluation"), dict) else {}
    sql_rows_raw = (
        (pipeline_trace.get("finance_sql_rows") or []) if isinstance(pipeline_trace.get("finance_sql_rows"), list) else []
    )
    sql_rows = []
    for row in sql_rows_raw[:80]:
        if not isinstance(row, dict):
            continue
        sql_rows.append(
            {
                "id": row.get("id"),
                "document_id": row.get("document_id"),
                "accn": row.get("accn"),
                "taxonomy": row.get("taxonomy"),
                "metric_key": row.get("metric_key"),
                "value": row.get("value"),
                "unit": row.get("unit"),
                "form": row.get("form"),
                "filed": row.get("filed"),
                "label": row.get("label"),
            }
        )
    rag_items = []
    for item in rerank_hits[:20]:
        if not isinstance(item, dict):
            continue
        rag_items.append(
            {
                "node_id": item.get("node_id"),
                "document_id": item.get("document_id"),
                "level": item.get("level"),
                "title": item.get("title"),
                "source_section": item.get("source_section"),
                "finance_statement": item.get("finance_statement"),
                "finance_period": item.get("finance_period"),
                "rerank_score": item.get("rerank_score"),
                "text_excerpt": _short_text(item.get("text"), limit=1200),
            }
        )
    if not rag_items and isinstance(narrative_cards, list):
        for card in narrative_cards[:20]:
            if not isinstance(card, dict):
                continue
            rag_items.append(
                {
                    "node_id": card.get("node_id"),
                    "document_id": card.get("document_id"),
                    "level": None,
                    "title": card.get("title"),
                    "source_section": None,
                    "finance_statement": None,
                    "finance_period": None,
                    "rerank_score": card.get("relevance_score"),
                    "text_excerpt": _short_text(card.get("body"), limit=1200),
                }
            )
    prompt_breakdown = (
        (pipeline_trace.get("prompt_context_breakdown") or {})
        if isinstance(pipeline_trace.get("prompt_context_breakdown"), dict)
        else {}
    )
    token_usage = (
        (((response_payload.get("pipeline_trace") or {}).get("generation") or {}).get("token_usage") or {})
        if isinstance((response_payload.get("pipeline_trace") or {}).get("generation"), dict)
        else {}
    )
    return {
        "meta": {
            "trace_id": trace_id,
            "created_at": created_at,
            "schema_version": "report.detail.v1",
        },
        "request": {
            "question": request_payload.get("question"),
            "document_ids": request_payload.get("document_ids") or [],
            "report_locale": request_payload.get("report_locale"),
            "detail_level": request_payload.get("detail_level"),
            "top_k": request_payload.get("top_k"),
        },
        "rag": {
            "final_ranked_ids": rerank_compare.get("final_ranked_ids") or [],
            "evidence_count": len(rag_items),
            "evidence": rag_items,
        },
        "sql": {
            "sql_row_count": _to_int(finance_route.get("sql_row_count"), 0),
            "sql_rows_kept": _to_int(finance_route.get("sql_rows_kept"), 0),
            "filings_observed": external.get("filings_observed") or [],
            "facts_count": len(sql_rows),
            "facts": sql_rows,
        },
        "prompt": {
            "sql_context_chars": _to_int(prompt_breakdown.get("sql_context_chars"), 0),
            "rag_context_chars": _to_int(prompt_breakdown.get("rag_context_chars"), 0),
            "combined_context_chars": _to_int(prompt_breakdown.get("combined_context_chars"), 0),
            "token_usage": token_usage if isinstance(token_usage, dict) else {},
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_index_line(line_payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with INDEX_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line_payload, ensure_ascii=False) + "\n")


def _build_summary(record: dict[str, Any], file_path: Path, detail_file_path: Path | None = None) -> dict[str, Any]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    req = record.get("request") if isinstance(record.get("request"), dict) else {}
    res = record.get("response") if isinstance(record.get("response"), dict) else {}
    trace_id = str(meta.get("trace_id") or "")
    answer = str(res.get("answer") or "")
    return {
        "trace_id": trace_id,
        "created_at": meta.get("created_at"),
        "source": meta.get("source"),
        "schema_version": meta.get("schema_version"),
        "question": req.get("question"),
        "report_locale": res.get("report_locale"),
        "answer_preview": answer[:220],
        "path": str(file_path.relative_to(AGENT_ROOT)),
        "detail_path": str(detail_file_path.relative_to(AGENT_ROOT)) if detail_file_path is not None else None,
    }


def save_ask_report(
    *,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    source: str = "frontend",
) -> dict[str, Any]:
    trace_id = str(response_payload.get("trace_id") or "").strip() or uuid.uuid4().hex
    created_at = _now_iso()
    prompt_breakdown = (
        ((response_payload.get("pipeline_trace") or {}).get("prompt_context_breakdown") or {})
        if isinstance(response_payload.get("pipeline_trace"), dict)
        else {}
    )
    record: dict[str, Any] = {
        "meta": {
            "trace_id": trace_id,
            "created_at": created_at,
            "source": source,
            "schema_version": SCHEMA_VERSION,
        },
        "request": {
            "question": request_payload.get("question"),
            "document_ids": request_payload.get("document_ids") or [],
            "report_locale": request_payload.get("report_locale"),
            "detail_level": request_payload.get("detail_level"),
            "top_k": request_payload.get("top_k"),
        },
        "response": response_payload,
        "diagnostics": {
            "latency_ms": response_payload.get("latency_ms"),
            "confidence": response_payload.get("confidence"),
            "sql_context_chars": prompt_breakdown.get("sql_context_chars"),
            "rag_context_chars": prompt_breakdown.get("rag_context_chars"),
            "combined_context_chars": prompt_breakdown.get("combined_context_chars"),
        },
    }
    path = _record_path(trace_id)
    detail_path = _detail_path(trace_id)
    detail_record = _build_detail_snapshot(
        trace_id=trace_id,
        created_at=created_at,
        request_payload=request_payload,
        response_payload=response_payload,
    )
    with _LOCK:
        _atomic_write_json(path, record)
        _atomic_write_json(detail_path, detail_record)
        _append_index_line(_build_summary(record, path, detail_path))
    return {
        "trace_id": trace_id,
        "path": str(path.relative_to(AGENT_ROOT)),
        "detail_path": str(detail_path.relative_to(AGENT_ROOT)),
    }


def list_reports(*, limit: int = 50) -> list[dict[str, Any]]:
    if not INDEX_FILE.exists():
        return []
    raw = INDEX_FILE.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in reversed(raw):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        trace_id = str(item.get("trace_id") or "")
        if not trace_id or trace_id in seen:
            continue
        seen.add(trace_id)
        out.append(item)
        if len(out) >= max(1, int(limit)):
            break
    return out


def get_report(trace_id: str) -> dict[str, Any] | None:
    safe = str(trace_id or "").strip()
    if not safe:
        return None
    path = _record_path(safe)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
