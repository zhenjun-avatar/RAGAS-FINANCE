"""Resolve primary fiscal period (end, fy, fp) per SEC accession from companyfacts JSON.

Used at ingest when a local ``CIK##########.json`` snapshot exists under ``tools/data/``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

DEI_END = "DocumentPeriodEndDate"
DEI_FY = "DocumentFiscalYearFocus"
DEI_FP = "DocumentFiscalPeriodFocus"


def iter_companyfacts_rows(facts_doc: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    facts = facts_doc.get("facts") or {}
    if not isinstance(facts, dict):
        return
    for namespace, concepts in facts.items():
        if not isinstance(concepts, dict):
            continue
        for concept, payload in concepts.items():
            if not isinstance(payload, dict):
                continue
            for series in (payload.get("units") or {}).values():
                if not isinstance(series, list):
                    continue
                for row in series:
                    if isinstance(row, dict) and row.get("accn"):
                        yield str(namespace), str(concept), row


def _norm_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return text or None


def _norm_fy(value: Any) -> int | None:
    if value is None:
        return None
    try:
        y = int(value)
    except (TypeError, ValueError):
        return None
    if 1990 <= y <= 2100:
        return y
    return None


def _norm_fp(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _dei_value_date(row: dict[str, Any]) -> str | None:
    e = _norm_date(row.get("end"))
    if e:
        return e
    return _norm_date(row.get("val"))


def _dei_value_fy(row: dict[str, Any]) -> int | None:
    v = row.get("val")
    y = _norm_fy(v)
    if y is not None:
        return y
    return _norm_fy(row.get("fy"))


def _dei_value_fp(row: dict[str, Any]) -> str | None:
    v = row.get("val")
    if v is not None and str(v).strip():
        fp = _norm_fp(v)
        if fp:
            return fp
    return _norm_fp(row.get("fp"))


def _counter_mode(counter: Counter[Any]) -> Any | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _gather_dei_by_accn(facts_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ns, concept, row in iter_companyfacts_rows(facts_doc):
        if ns != "dei":
            continue
        accn = str(row["accn"]).strip()
        if concept == DEI_END:
            slot = out.setdefault(accn, {})
            d = _dei_value_date(row)
            if d:
                slot["dei_end"] = d
        elif concept == DEI_FY:
            slot = out.setdefault(accn, {})
            y = _dei_value_fy(row)
            if y is not None:
                slot["dei_fy"] = y
        elif concept == DEI_FP:
            slot = out.setdefault(accn, {})
            fp = _dei_value_fp(row)
            if fp:
                slot["dei_fp"] = fp
    return out


def _aggregate_modes_by_accn(facts_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_accn: dict[str, list[dict[str, Any]]] = {}
    for _ns, _concept, row in iter_companyfacts_rows(facts_doc):
        accn = str(row["accn"]).strip()
        by_accn.setdefault(accn, []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for accn, rows in by_accn.items():
        fy_c: Counter[int] = Counter()
        fp_c: Counter[str] = Counter()
        for r in rows:
            y = _norm_fy(r.get("fy"))
            if y is not None:
                fy_c[y] += 1
            fp = _norm_fp(r.get("fp"))
            if fp:
                fp_c[fp] += 1
        fy_mode = _counter_mode(fy_c)
        fp_mode = _counter_mode(fp_c)
        end_c: Counter[str] = Counter()
        for r in rows:
            if fy_mode is not None and _norm_fy(r.get("fy")) != fy_mode:
                continue
            if fp_mode is not None and _norm_fp(r.get("fp")) != fp_mode:
                continue
            e = _norm_date(r.get("end"))
            if e:
                end_c[e] += 1
        if not end_c:
            for r in rows:
                e = _norm_date(r.get("end"))
                if e:
                    end_c[e] += 1
        end_mode = _counter_mode(end_c)
        result[accn] = {
            "fy_mode": fy_mode,
            "fp_mode": fp_mode,
            "end_mode": end_mode,
            "fy_mode_count": int(fy_c[fy_mode]) if fy_mode is not None else 0,
            "fp_mode_count": int(fp_c[fp_mode]) if fp_mode is not None else 0,
            "end_mode_count": int(end_c[end_mode]) if end_mode else 0,
        }
    return result


def _resolve_from_dei_agg(accn: str, d: dict[str, Any], a: dict[str, Any]) -> dict[str, Any]:
    end = d.get("dei_end")
    fy = d.get("dei_fy")
    fp = d.get("dei_fp")
    src_end = f"dei:{DEI_END}" if end else None
    src_fy = f"dei:{DEI_FY}" if fy is not None else None
    src_fp = f"dei:{DEI_FP}" if fp else None
    if fy is None:
        fy = a.get("fy_mode")
        src_fy = "aggregate:fy_mode"
    if fp is None:
        fp = a.get("fp_mode")
        src_fp = "aggregate:fp_mode"
    if end is None:
        end = a.get("end_mode")
        src_end = "aggregate:end_mode_on_modal_fy_fp_rows" if a.get("fp_mode") else "aggregate:end_mode_on_modal_fy_rows"
        if end is None:
            src_end = "aggregate:end_mode_all_rows"
    return {
        "end": end,
        "fy": fy,
        "fp": fp,
        "sources": {"end": src_end, "fy": src_fy, "fp": src_fp},
        "aggregate_debug": {
            "fy_mode": a.get("fy_mode"),
            "fp_mode": a.get("fp_mode"),
            "end_mode": a.get("end_mode"),
            "fy_mode_count": a.get("fy_mode_count"),
            "fp_mode_count": a.get("fp_mode_count"),
            "end_mode_count": a.get("end_mode_count"),
        },
    }


def resolve_accession_period(facts_doc: dict[str, Any], accession: str) -> dict[str, Any] | None:
    """Return ``{end, fy, fp, sources, aggregate_debug}`` for one accession, or None if absent."""
    accn = str(accession).strip()
    if not accn:
        return None
    dei = _gather_dei_by_accn(facts_doc)
    agg = _aggregate_modes_by_accn(facts_doc)
    if accn not in dei and accn not in agg:
        return None
    return _resolve_from_dei_agg(accn, dei.get(accn, {}), agg.get(accn, {}))


def resolve_all_accessions(facts_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dei = _gather_dei_by_accn(facts_doc)
    agg = _aggregate_modes_by_accn(facts_doc)
    out: dict[str, dict[str, Any]] = {}
    for accn in sorted(set(dei.keys()) | set(agg.keys())):
        out[accn] = _resolve_from_dei_agg(accn, dei.get(accn, {}), agg.get(accn, {}))
    return out


def default_companyfacts_json_path(cik: int) -> Path:
    return Path(__file__).resolve().parent.parent / "data" / f"CIK{int(cik):010d}.json"


def merge_companyfacts_period_into_metadata(
    metadata: dict[str, Any],
    *,
    cik: int,
    accession: str,
    companyfacts_path: Path | None = None,
) -> dict[str, Any]:
    """Attach primary ``finance_period_end_dates`` (single end), ``document_fiscal_*``, and ``finance_period`` year list."""
    path = companyfacts_path or default_companyfacts_json_path(cik)
    if not path.is_file():
        return metadata
    try:
        raw = path.read_text(encoding="utf-8")
        facts_doc = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return metadata
    if not isinstance(facts_doc.get("facts"), dict):
        return metadata
    resolved = resolve_accession_period(facts_doc, accession)
    if not resolved:
        return metadata
    out = dict(metadata)
    end = resolved.get("end")
    fy = resolved.get("fy")
    fp = resolved.get("fp")
    if end:
        out["finance_period_end_dates"] = [str(end)]
    if fy is not None:
        y = int(fy)
        out["document_fiscal_year"] = y
        out["finance_period"] = [str(y)]
    if fp:
        out["document_fiscal_period"] = str(fp)
    return out
