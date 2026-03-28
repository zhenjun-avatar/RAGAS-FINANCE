"""SEC EDGAR companyfacts JSON → flat rows for SQL + RAG text chunks."""

from __future__ import annotations

import re
from typing import Any

# XBRL local names in questions, e.g. EntityPublicFloat, AccountsPayableCurrent
_XBRL_CAMEL_TOKEN = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+\b")

_HINT_NOISE = frozenset(
    {
        "Restated",
        "Consolidated",
        "Financial",
        "Statements",
        "Reporting",
        "Accounting",
        "Standard",
        "Disclosure",
        "Document",
        "Securities",
        "Exchange",
        "Commission",
    }
)


def is_sec_company_facts_payload(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return "cik" in data and "facts" in data and isinstance(data.get("facts"), dict)


def _as_number(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _as_text(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return None
    s = str(val).strip()
    return s or None


def list_accessions_from_company_facts(data: dict[str, Any]) -> list[str]:
    """Distinct accession numbers in appearance order (flattened row walk)."""
    rows = flatten_sec_company_facts(data)
    seen: dict[str, None] = {}
    for row in rows:
        accn = str(row.get("accn") or "").strip()
        if accn and accn not in seen:
            seen[accn] = None
    return list(seen.keys())


def flatten_sec_company_facts(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten `companyfacts`-style JSON into one row per (metric × unit × observation)."""
    cik = data.get("cik")
    entity_name = data.get("entityName")
    facts = data.get("facts") or {}
    rows: list[dict[str, Any]] = []

    if not isinstance(facts, dict):
        return rows

    for taxonomy, metrics in facts.items():
        if not isinstance(metrics, dict):
            continue
        for metric_key, metric_obj in metrics.items():
            if not isinstance(metric_obj, dict):
                continue
            label = metric_obj.get("label")
            description = metric_obj.get("description")
            units = metric_obj.get("units")
            if not isinstance(units, dict):
                continue
            for unit_name, entries in units.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    raw_val = entry.get("val")
                    num = _as_number(raw_val)
                    txt = None if num is not None else _as_text(raw_val)
                    rows.append(
                        {
                            "cik": int(cik) if cik is not None else None,
                            "entity_name": entity_name,
                            "taxonomy": str(taxonomy),
                            "metric_key": str(metric_key),
                            "metric_label": label if isinstance(label, str) else None,
                            "metric_description": description if isinstance(description, str) else None,
                            "unit": str(unit_name),
                            "value_numeric": num,
                            "value_text": txt,
                            "period_end": entry.get("end"),
                            "filed_date": entry.get("filed"),
                            "form": entry.get("form"),
                            "fy": entry.get("fy"),
                            "fp": entry.get("fp"),
                            "accn": entry.get("accn"),
                            "frame": entry.get("frame"),
                            "raw_entry": entry,
                        }
                    )
    return rows


def batch_lines_for_nodes(rows: list[dict[str, Any]], *, lines_per_chunk: int = 64) -> list[str]:
    """Turn flat rows into newline-separated text blocks for BM25/vector nodes."""
    lines: list[str] = []
    for r in rows:
        val = r.get("value_numeric")
        if val is None and r.get("value_text"):
            val_str = r["value_text"]
        else:
            val_str = "" if val is None else str(val)
        label = (r.get("metric_label") or "").replace("\n", " ")[:200]
        lines.append(
            f"{r.get('taxonomy')}.{r.get('metric_key')} | {label} | unit={r.get('unit')} | "
            f"val={val_str} | end={r.get('period_end')} | filed={r.get('filed_date')} | "
            f"form={r.get('form')} | fy={r.get('fy')} fp={r.get('fp')} | accn={r.get('accn')}"
        )
    chunks: list[str] = []
    for i in range(0, len(lines), lines_per_chunk):
        chunks.append("\n".join(lines[i : i + lines_per_chunk]))
    return chunks


def batch_row_groups_for_nodes(
    rows: list[dict[str, Any]],
    *,
    lines_per_chunk: int = 64,
) -> list[list[dict[str, Any]]]:
    """Split SEC observation rows into stable groups matching `batch_lines_for_nodes`."""
    return [rows[i : i + lines_per_chunk] for i in range(0, len(rows), lines_per_chunk)]


def build_chunk_filter_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Metadata fields used for pre-retrieval filtering on finance chunks."""
    metric_keys: list[str] = []
    forms: list[str] = []
    accns: list[str] = []
    period_years: list[str] = []
    period_end_dates: list[str] = []
    seen_metric_keys: set[str] = set()
    seen_forms: set[str] = set()
    seen_accns: set[str] = set()
    seen_years: set[str] = set()
    seen_dates: set[str] = set()
    for row in rows:
        metric_key = str(row.get("metric_key") or "").strip()
        if metric_key and metric_key not in seen_metric_keys:
            seen_metric_keys.add(metric_key)
            metric_keys.append(metric_key)
        form = str(row.get("form") or "").strip()
        if form and form not in seen_forms:
            seen_forms.add(form)
            forms.append(form)
        accn = str(row.get("accn") or "").strip()
        if accn and accn not in seen_accns:
            seen_accns.add(accn)
            accns.append(accn)
        period_end = str(row.get("period_end") or "").strip()
        if period_end:
            if period_end not in seen_dates:
                seen_dates.add(period_end)
                period_end_dates.append(period_end)
            year = period_end[:4]
            if len(year) == 4 and year.isdigit() and year not in seen_years:
                seen_years.add(year)
                period_years.append(year)
    out: dict[str, Any] = {}
    if metric_keys:
        out["finance_metric_exact_keys"] = metric_keys[:24]
    if forms:
        out["finance_forms"] = forms[:8]
    if accns:
        out["finance_accns"] = accns[:24]
    if period_years:
        out["finance_period_years"] = period_years[:12]
    if period_end_dates:
        out["finance_period_end_dates"] = period_end_dates[:24]
    return out


def extract_metric_hints_from_question(question: str, *, max_hints: int = 12) -> list[str]:
    """Pull likely XBRL metric local names from the question for targeted SQL rows."""
    q = question or ""
    seen: dict[str, None] = {}
    out: list[str] = []
    for raw in _XBRL_CAMEL_TOKEN.findall(q):
        t = re.sub(r"[^A-Za-z0-9_]", "", raw)
        if len(t) < 8 or t in _HINT_NOISE:
            continue
        if t not in seen:
            seen[t] = None
            out.append(t)
        if len(out) >= max_hints:
            break
    return out
