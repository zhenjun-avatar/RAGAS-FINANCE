"""Resolve likely SEC filing accessions from query metadata constraints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Iterable

from tools.node_repository import list_sec_filing_catalog

from .finance_query_plan import EvidencePlan, FilingHypothesis, _normalize_form_base


@dataclass(frozen=True)
class FilingCatalogEntry:
    document_id: int
    accession: str
    form_base: str | None
    filed_date: str | None
    period_end_dates: tuple[str, ...]
    period_years: tuple[int, ...]
    candidate_years: tuple[int, ...]
    cik: str | None
    entity_name: str | None


def _norm_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _norm_date(value: Any) -> str | None:
    text = _norm_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


def _date_to_year(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value).year
    except ValueError:
        return None


def _norm_date_list(raw: Any, *, limit: int = 8) -> tuple[str, ...]:
    values = raw if isinstance(raw, list) else [raw]
    out: list[str] = []
    for item in values:
        normalized = _norm_date(item)
        if normalized and normalized not in out:
            out.append(normalized)
        if len(out) >= limit:
            break
    return tuple(out)


def _norm_year_list(raw: Any, *, limit: int = 8) -> tuple[int, ...]:
    values = raw if isinstance(raw, list) else [raw]
    out: list[int] = []
    for item in values:
        try:
            year = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if 1990 <= year <= 2100 and year not in out:
            out.append(year)
        if len(out) >= limit:
            break
    return tuple(out)


def _accession_to_year(accession: str | None) -> int | None:
    text = _norm_text(accession)
    if not text:
        return None
    parts = text.split("-")
    if len(parts) != 3 or len(parts[1]) != 2 or not parts[1].isdigit():
        return None
    yy = int(parts[1])
    return 2000 + yy if yy <= 30 else 1900 + yy


def _candidate_years(*, form_base: str | None, filed_date: str | None, period_end_dates: tuple[str, ...]) -> tuple[int, ...]:
    out: list[int] = []
    for item in period_end_dates:
        year = _date_to_year(item)
        if year and year not in out:
            out.append(year)
    filed_year = _date_to_year(filed_date)
    if filed_year is not None:
        if filed_year not in out:
            out.append(filed_year)
        filed_month = int(filed_date[5:7]) if len(filed_date) >= 7 else 12
        if filed_month <= 4 and form_base in {"10-K", "10-Q"} and filed_year - 1 not in out:
            out.append(filed_year - 1)
    return tuple(out[:4])


def _normalize_catalog_entry(row: dict[str, Any]) -> FilingCatalogEntry | None:
    accession = _norm_text(row.get("accession"))
    if not accession:
        return None
    form_base = None
    if row.get("form"):
        normalized = _normalize_form_base(str(row.get("form")))
        form_base = normalized or None
    filed_date = _norm_date(row.get("filed"))
    period_end_dates = _norm_date_list(row.get("period_end_dates"))
    period_years = _norm_year_list(row.get("period_years"))
    merged_years = list(period_years)
    accession_year = _accession_to_year(accession)
    if accession_year is not None and accession_year not in merged_years:
        merged_years.append(accession_year)
    for year in _candidate_years(
        form_base=form_base,
        filed_date=filed_date,
        period_end_dates=period_end_dates,
    ):
        if year not in merged_years:
            merged_years.append(year)
    return FilingCatalogEntry(
        document_id=int(row["document_id"]),
        accession=accession,
        form_base=form_base,
        filed_date=filed_date,
        period_end_dates=period_end_dates,
        period_years=period_years,
        candidate_years=tuple(merged_years[:6]),
        cik=_norm_text(row.get("cik")),
        entity_name=_norm_text(row.get("entity_name")),
    )


def _target_hypothesis_count(plan: Any, *, limit: int) -> int:
    compare_mode = str(getattr(plan, "compare_mode", "") or "generic")
    if compare_mode in {"yoy", "qoq"}:
        return max(2, min(limit, 4))
    if bool(getattr(plan, "prefer_recent", False)):
        return max(1, min(limit, 2))
    return max(2, min(limit, 3))


def _recency_bonus(entries: list[FilingCatalogEntry]) -> dict[str, float]:
    ranked = sorted(
        entries,
        key=lambda item: (
            item.filed_date or "",
            item.period_end_dates[-1] if item.period_end_dates else "",
            item.accession,
        ),
        reverse=True,
    )
    total = max(1, len(ranked) - 1)
    out: dict[str, float] = {}
    for idx, item in enumerate(ranked):
        out[item.accession] = 0.18 * (1.0 - (idx / total))
    return out


def _score_entry(entry: FilingCatalogEntry, plan: Any, recency_bonus: float) -> tuple[float, tuple[str, ...]]:
    reasons: list[str] = []
    score = 0.02
    forms = {_normalize_form_base(str(item)) for item in (getattr(plan, "form_filters", None) or ()) if str(item).strip()}
    forms.discard("")
    years = {int(item) for item in getattr(plan, "period_years", ()) if str(item).strip()}
    dates = {str(item).strip() for item in getattr(plan, "period_end_dates", ()) if str(item).strip()}
    compare_mode = str(getattr(plan, "compare_mode", "") or "generic")
    prefer_recent = bool(getattr(plan, "prefer_recent", False))

    if forms:
        if entry.form_base in forms:
            score += 0.34
            reasons.append("form_match")
        elif entry.form_base:
            score -= 0.26
            reasons.append("form_mismatch")
        else:
            score -= 0.08
            reasons.append("form_missing")

    if dates:
        if any(item in dates for item in entry.period_end_dates):
            score += 0.34
            reasons.append("date_match")
        elif entry.period_end_dates:
            score -= 0.28
            reasons.append("date_mismatch")
        else:
            score -= 0.08
            reasons.append("date_missing")

    if years:
        if any(item in years for item in entry.candidate_years):
            score += 0.22
            reasons.append("year_match")
        elif entry.candidate_years:
            score -= 0.18
            reasons.append("year_mismatch")
        else:
            score -= 0.06
            reasons.append("year_missing")

    if compare_mode == "qoq":
        if entry.form_base == "10-Q":
            score += 0.12
            reasons.append("compare_qoq")
        elif entry.form_base == "10-K":
            score -= 0.06
    elif compare_mode == "yoy":
        if entry.form_base in {"10-Q", "10-K"}:
            score += 0.08
            reasons.append("compare_yoy")
    elif compare_mode == "latest":
        score += 0.04
        reasons.append("compare_latest")

    if prefer_recent:
        score += recency_bonus
        reasons.append("prefer_recent")
    else:
        score += min(recency_bonus, 0.08)

    if not forms and not years and not dates and compare_mode == "generic" and not prefer_recent:
        score = min(score, 0.18)
        reasons.append("low_specificity")

    return max(0.0, min(score, 0.99)), tuple(dict.fromkeys(reasons))


def _rank_filing_hypotheses(
    entries: list[FilingCatalogEntry],
    *,
    plan: Any,
    limit: int,
) -> tuple[tuple[FilingHypothesis, ...], dict[str, Any]]:
    if not entries:
        return (), {"candidate_count": 0, "selected_count": 0}
    recency_map = _recency_bonus(entries)
    scored: list[tuple[FilingCatalogEntry, FilingHypothesis]] = []
    for entry in entries:
        weight, reasons = _score_entry(entry, plan, recency_map.get(entry.accession, 0.0))
        scored.append(
            (
                entry,
                FilingHypothesis(
                    accession=entry.accession,
                    weight=round(weight, 4),
                    reasons=reasons,
                ),
            )
        )
    scored.sort(
        key=lambda item: (
            float(item[1].weight),
            item[0].filed_date or "",
            item[0].accession,
        ),
        reverse=True,
    )
    seen: set[str] = set()
    chosen: list[FilingHypothesis] = []
    chosen_debug: list[dict[str, Any]] = []
    for entry, hypothesis in scored:
        if hypothesis.accession in seen:
            continue
        seen.add(hypothesis.accession)
        chosen.append(hypothesis)
        chosen_debug.append(
            {
                "accession": entry.accession,
                "document_id": entry.document_id,
                "form_base": entry.form_base,
                "filed_date": entry.filed_date,
                "period_end_dates": list(entry.period_end_dates),
                "candidate_years": list(entry.candidate_years),
                "weight": hypothesis.weight,
                "reasons": list(hypothesis.reasons),
            }
        )
        if len(chosen) >= limit:
            break
    return tuple(chosen), {
        "candidate_count": len(entries),
        "selected_count": len(chosen),
        "selected": chosen_debug,
    }


async def attach_filing_hypotheses(
    evidence_plan: EvidencePlan,
    *,
    document_ids: Iterable[int],
    limit: int = 6,
) -> tuple[EvidencePlan, dict[str, Any]]:
    rows = await list_sec_filing_catalog(document_ids=document_ids, limit=2000)
    entries = [item for item in (_normalize_catalog_entry(dict(row)) for row in rows) if item is not None]
    target_count = _target_hypothesis_count(evidence_plan.sql_plan, limit=limit)
    hypotheses, rank_debug = _rank_filing_hypotheses(entries, plan=evidence_plan.sql_plan, limit=target_count)
    updated = replace(evidence_plan, filing_hypotheses=hypotheses)
    return updated, {
        "constraints": {
            "finance_form_base": [
                _normalize_form_base(str(item))
                for item in (evidence_plan.sql_plan.form_filters or ())
                if str(item).strip()
            ],
            "finance_period": [int(item) for item in evidence_plan.sql_plan.period_years],
            "finance_period_end_dates": [str(item) for item in evidence_plan.sql_plan.period_end_dates],
            "compare_mode": evidence_plan.sql_plan.compare_mode,
            "prefer_recent": evidence_plan.sql_plan.prefer_recent,
        },
        "target_count": target_count,
        **rank_debug,
    }
