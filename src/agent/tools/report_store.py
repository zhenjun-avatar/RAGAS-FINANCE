"""Persistent storage for ask results under src/agent/report."""

from __future__ import annotations

import copy
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import config

AGENT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = AGENT_ROOT / "report"
DETAIL_DIR = REPORT_DIR / "detail"
LANGFUSE_DIR = REPORT_DIR / "langfuse"
INDEX_FILE = REPORT_DIR / "index.jsonl"
SCHEMA_VERSION = "report.v1"

_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record_path(trace_id: str) -> Path:
    return REPORT_DIR / f"ask_result_{trace_id}.json"


def _detail_path(trace_id: str) -> Path:
    return DETAIL_DIR / f"ask_detail_{trace_id}.json"


def _langfuse_path(trace_id: str) -> Path:
    return LANGFUSE_DIR / f"langfuse_trace_{trace_id}.json"


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


def _apply_full_text_cap(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _collect_detail_evidence_node_ids(response_payload: dict[str, Any]) -> list[str]:
    pipeline_trace = (
        response_payload.get("pipeline_trace") if isinstance(response_payload.get("pipeline_trace"), dict) else {}
    )
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
    out: list[str] = []
    for item in rerank_hits[:20]:
        if not isinstance(item, dict):
            continue
        nid = str(item.get("node_id") or "").strip()
        if nid:
            out.append(nid)
    if not out:
        for card in narrative_cards[:20]:
            if not isinstance(card, dict):
                continue
            nid = str(card.get("node_id") or "").strip()
            if nid:
                out.append(nid)
    return list(dict.fromkeys(out))


async def load_evidence_full_text_for_detail(response_payload: dict[str, Any]) -> dict[str, str]:
    """Fetch node.text from DB for detail JSON full evidence (see REPORT_DETAIL_FULL_EVIDENCE)."""
    from tools.node_repository import fetch_nodes

    ids = _collect_detail_evidence_node_ids(response_payload)
    if not ids:
        return {}
    rows = await fetch_nodes(ids)
    return {str(r.get("node_id") or ""): str(r.get("text") or "") for r in rows if r.get("node_id")}


def _build_detail_snapshot(
    *,
    trace_id: str,
    created_at: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    full_evidence: bool = False,
    full_evidence_max_chars: int = 0,
    evidence_full_text: dict[str, str] | None = None,
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
    ev_text = evidence_full_text or {}

    def _attach_full(entry: dict[str, Any], *, nid: str, fallback_inline: str | None) -> None:
        if not full_evidence:
            return
        full = ev_text.get(nid) if nid else None
        if full is None or not str(full).strip():
            full = (fallback_inline or "").strip() or None
        if not full:
            return
        entry["text_full"] = _apply_full_text_cap(full, full_evidence_max_chars)

    for item in rerank_hits[:20]:
        if not isinstance(item, dict):
            continue
        nid = str(item.get("node_id") or "").strip()
        raw_text = item.get("text")
        entry: dict[str, Any] = {
            "node_id": item.get("node_id"),
            "document_id": item.get("document_id"),
            "level": item.get("level"),
            "title": item.get("title"),
            "source_section": item.get("source_section"),
            "finance_statement": item.get("finance_statement"),
            "finance_period": item.get("finance_period"),
            "rerank_score": item.get("rerank_score"),
            "text_excerpt": _short_text(raw_text, limit=1200),
        }
        _attach_full(entry, nid=nid, fallback_inline=str(raw_text) if raw_text is not None else None)
        rag_items.append(entry)
    if not rag_items and isinstance(narrative_cards, list):
        for card in narrative_cards[:20]:
            if not isinstance(card, dict):
                continue
            nid = str(card.get("node_id") or "").strip()
            body = card.get("body")
            entry = {
                "node_id": card.get("node_id"),
                "document_id": card.get("document_id"),
                "level": None,
                "title": card.get("title"),
                "source_section": None,
                "finance_statement": None,
                "finance_period": None,
                "rerank_score": card.get("relevance_score"),
                "text_excerpt": _short_text(body, limit=1200),
            }
            _attach_full(entry, nid=nid, fallback_inline=str(body) if body is not None else None)
            rag_items.append(entry)
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
    narrative_gates: dict[str, Any] = {}
    sp = pipeline_trace.get("section_policy") if isinstance(pipeline_trace.get("section_policy"), dict) else {}
    _raw_answerability = pipeline_trace.get("answerability")
    answerability = _raw_answerability if isinstance(_raw_answerability, dict) else {}
    eg = pipeline_trace.get("evidence_gate") if isinstance(pipeline_trace.get("evidence_gate"), dict) else {}
    cov = pipeline_trace.get("coverage_selection") if isinstance(pipeline_trace.get("coverage_selection"), dict) else {}
    prs = (
        pipeline_trace.get("post_rerank_selector")
        if isinstance(pipeline_trace.get("post_rerank_selector"), dict)
        else {}
    )
    if sp or answerability or eg or cov or prs:
        narrative_gates = {
            "section_policy": sp,
            "answerability": answerability,
            "evidence_gate": eg,
            "coverage_selection": cov,
            "post_rerank_selector": prs,
        }

    return {
        "meta": {
            "trace_id": trace_id,
            "created_at": created_at,
            "schema_version": "report.detail.v1",
            "full_evidence": bool(full_evidence),
            "full_evidence_max_chars": int(full_evidence_max_chars),
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
        "narrative_gates": narrative_gates,
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


def _slim_finance_route_for_disk(fr: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "need_sql",
        "need_rag",
        "source",
        "detail",
        "sql_row_count",
        "skipped_retrieval",
        "finance_intent",
        "retrieval_metadata_filters",
        "filing_scope_resolver",
        "sql_rows_kept",
        "preferred_accns",
    )
    out = {k: fr[k] for k in keys if k in fr}
    ep = fr.get("evidence_plan")
    if isinstance(ep, dict):
        out["evidence_plan"] = {
            kk: ep.get(kk)
            for kk in (
                "question_mode",
                "evidence_requirements",
                "narrative_targets",
                "retrieval_budget",
                "term_targets",
            )
        }
    return out


def _slim_pipeline_trace_for_disk(pt: dict[str, Any]) -> dict[str, Any]:
    """Drop large hit lists / notes from pipeline_trace for ask_result_*.json on disk."""
    keep_keys = (
        "retrieval_counts",
        "scoped_leaf_fused_debug",
        "finance_route",
        "evidence_controller",
        "sql_rag_narrowing",
        "retrieval_compare_meta",
        "retrieval_filing_distribution",
        "metadata_filter_policy",
        "retrieval_soft_hints",
        "section_policy",
        "answerability",
        "coverage_selection",
        "post_rerank_selector",
        "evidence_gate",
        "narrative_pool",
        "retrieval_passes",
        "second_pass",
        "rerank",
        "prompt_context_breakdown",
        "retrieval_rerank_compare",
        "rerank_stage_pre_rerank",
        "rerank_stage_rerank_out",
        "rerank_stage_final_ranked",
    )
    out: dict[str, Any] = {}
    for key in keep_keys:
        if key not in pt:
            continue
        val = pt[key]
        if key == "finance_route" and isinstance(val, dict):
            out[key] = _slim_finance_route_for_disk(val)
        elif key == "retrieval_compare_meta" and isinstance(val, dict):
            out[key] = {k: v for k, v in val.items() if k != "counts_meaning"}
        else:
            out[key] = val
    return out


def _slim_evidence_ui_for_disk(ui: dict[str, Any]) -> dict[str, Any]:
    conclusion = ui.get("conclusion") if isinstance(ui.get("conclusion"), dict) else {}
    evidence = ui.get("evidence") if isinstance(ui.get("evidence"), dict) else {}
    risk = ui.get("risk_panel") if isinstance(ui.get("risk_panel"), dict) else {}
    filings = evidence.get("filings") or []
    if isinstance(filings, list):
        filings = filings[:24]
    else:
        filings = []
    slim_conclusion = {
        k: conclusion.get(k)
        for k in ("trace_id", "confidence", "answer_preview", "question_preview", "answer_chars")
        if k in conclusion
    }
    return {
        "conclusion": slim_conclusion,
        "evidence": {
            "summary": evidence.get("summary"),
            "narrative_cards": evidence.get("narrative_cards") or [],
            "structured_fact_cards": evidence.get("structured_fact_cards") or [],
            "filings": filings,
        },
        "risk_panel": {
            "limitations": risk.get("limitations"),
            "flags": risk.get("flags"),
        },
    }


def _slim_external_evaluation_for_disk(ext: dict[str, Any]) -> dict[str, Any]:
    obs = ext.get("observable_metrics") or []
    slim_obs: list[dict[str, Any]] = []
    if isinstance(obs, list):
        for item in obs:
            if isinstance(item, dict) and item.get("metric_id") is not None:
                slim_obs.append({"metric_id": item["metric_id"], "value": item.get("value")})
    filings = ext.get("filings_observed") or []
    if not isinstance(filings, list):
        filings = []
    return {
        "metric_catalog_version": ext.get("metric_catalog_version"),
        "observable_metrics": slim_obs,
        "filings_observed": filings[:24],
    }


def _slim_vertical_scenario_for_disk(vs: dict[str, Any]) -> dict[str, Any]:
    return {
        k: vs[k]
        for k in ("scenario_id", "name", "report_locale", "question_mode")
        if k in vs
    }


def _slim_ask_response_for_disk(resp: dict[str, Any]) -> None:
    resp.pop("retrieval_debug", None)
    vs = resp.get("vertical_scenario")
    if isinstance(vs, dict):
        resp["vertical_scenario"] = _slim_vertical_scenario_for_disk(vs)
    pt = resp.get("pipeline_trace")
    if isinstance(pt, dict):
        resp["pipeline_trace"] = _slim_pipeline_trace_for_disk(pt)
    ui = resp.get("evidence_ui")
    if isinstance(ui, dict):
        resp["evidence_ui"] = _slim_evidence_ui_for_disk(ui)
    ext = resp.get("external_evaluation")
    if isinstance(ext, dict):
        resp["external_evaluation"] = _slim_external_evaluation_for_disk(ext)


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
    evidence_full_text: dict[str, str] | None = None,
) -> dict[str, Any]:
    trace_id = str(response_payload.get("trace_id") or "").strip() or uuid.uuid4().hex
    created_at = _now_iso()
    prompt_breakdown = (
        ((response_payload.get("pipeline_trace") or {}).get("prompt_context_breakdown") or {})
        if isinstance(response_payload.get("pipeline_trace"), dict)
        else {}
    )
    response_for_disk: dict[str, Any] = response_payload
    if config.report_ask_slim:
        response_for_disk = copy.deepcopy(response_payload)
        _slim_ask_response_for_disk(response_for_disk)
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
        "response": response_for_disk,
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
    full_ev = bool(config.report_detail_full_evidence)
    detail_record = _build_detail_snapshot(
        trace_id=trace_id,
        created_at=created_at,
        request_payload=request_payload,
        response_payload=response_payload,
        full_evidence=full_ev,
        full_evidence_max_chars=int(config.report_detail_full_evidence_max_chars or 0),
        evidence_full_text=evidence_full_text if full_ev else None,
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


def _rewrite_index_excluding(trace_ids: set[str]) -> None:
    """Drop index lines whose trace_id is in trace_ids (caller must hold _LOCK)."""
    ids = {str(x).strip() for x in trace_ids if str(x).strip()}
    if not ids:
        return
    if not INDEX_FILE.exists():
        return
    try:
        raw_lines = INDEX_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    kept: list[str] = []
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        tid = str(obj.get("trace_id") or "").strip()
        if tid in ids:
            continue
        kept.append(line)
    tmp = INDEX_FILE.with_suffix(".jsonl.rebuild_tmp")
    tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    os.replace(tmp, INDEX_FILE)


def delete_reports(*, trace_ids: list[str]) -> dict[str, Any]:
    """Remove ask_result / detail / langfuse files and purge matching index lines."""
    want: list[str] = []
    seen: set[str] = set()
    for t in trace_ids:
        s = str(t or "").strip()
        if s and s not in seen:
            seen.add(s)
            want.append(s)
    deleted: list[str] = []
    missing: list[str] = []
    with _LOCK:
        for tid in want:
            existed = False
            for path_fn in (_record_path, _detail_path, _langfuse_path):
                p = path_fn(tid)
                if p.exists():
                    existed = True
                    try:
                        p.unlink()
                    except OSError:
                        pass
            if existed:
                deleted.append(tid)
            else:
                missing.append(tid)
        _rewrite_index_excluding(set(want))
    return {"deleted": deleted, "missing": missing, "requested": want}


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


def save_langfuse_observability_report(
    *,
    trace_id: str,
    trace_name: str,
    request_payload: dict[str, Any],
    output_payload: dict[str, Any],
    trace_metadata: dict[str, Any],
    generation_payload: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    safe_trace_id = str(trace_id or "").strip() or uuid.uuid4().hex
    created_at = _now_iso()
    record: dict[str, Any] = {
        "meta": {
            "trace_id": safe_trace_id,
            "trace_name": str(trace_name or "rag-answer"),
            "created_at": created_at,
            "schema_version": "report.langfuse.v1",
        },
        "langfuse": {
            "diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
            "request_input": request_payload if isinstance(request_payload, dict) else {},
            "trace_output": output_payload if isinstance(output_payload, dict) else {},
            "trace_metadata": trace_metadata if isinstance(trace_metadata, dict) else {},
            "generation": generation_payload if isinstance(generation_payload, dict) else {},
        },
    }
    path = _langfuse_path(safe_trace_id)
    with _LOCK:
        _atomic_write_json(path, record)
    return {
        "trace_id": safe_trace_id,
        "path": str(path.relative_to(AGENT_ROOT)),
    }
