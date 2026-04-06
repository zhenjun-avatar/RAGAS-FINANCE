"""Explicit retrieval fields shared by dense/sparse indexes.

Finance-focused. To extend with another domain, add fields to
``RETRIEVAL_INDEX_KEYWORD_FIELDS``, branch in ``_extract_domain`` / ``build_retrieval_fields``,
and wire sparse query + index routing (see ``sparse_query_profiles`` / ``sparse_opensearch``).
"""

from __future__ import annotations

import re
from typing import Any

FIELD_METADATA_KEY = "_retrieval_fields"
RETRIEVAL_SCHEMA_VERSION = "retrieval_fields_v3"
RETRIEVAL_INDEX_KEYWORD_FIELDS = [
    "schema_version",
    "domain",
    "content_type",
    "language",
    "source_section",
    "section_leaf",
    "section_path",
    "topic_tags",
    "finance_statement",
    "finance_period",
    "finance_period_confidence",
    "finance_metric_keys",
    "finance_metric_exact_keys",
    "finance_forms",
    "finance_form_base",
    "finance_accns",
    "finance_period_end_dates",
]
RETRIEVAL_INDEX_TEXT_FIELDS = ["search_hints", "section_path_text"]

_YEAR_PERIOD_RE = re.compile(
    r"(20\d{2})\s*(?:年)?\s*(?:(Q[1-4])|(第?[一二三四1-4]季度)|(半年度)|(年度))?",
    re.IGNORECASE,
)
_TIME_CONTEXT_CUE_RE = re.compile(
    r"(as of|for the (?:fiscal )?year ended|year ended|quarter ended|fiscal year|截至|报告期|年度|季度|期末|end(ed)?\b)",
    re.IGNORECASE,
)
_STANDARD_NOISE_RE = re.compile(r"\b(?:asu|asc|ifrs|topic|note)\b", re.IGNORECASE)

_FINANCE_STATEMENT_LABELS: list[tuple[str, tuple[str, ...]]] = [
    ("balance_sheet", ("资产负债表", "合并资产负债表")),
    ("income_statement", ("利润表", "损益表", "合并利润表")),
    ("cash_flow_statement", ("现金流量表", "合并现金流量表")),
    ("equity_statement", ("所有者权益变动表", "股东权益变动表")),
    ("financial_notes", ("财务报表附注", "附注")),
    ("management_discussion", ("管理层讨论", "经营情况讨论", "md&a")),
]

_FINANCE_METRIC_LABELS: list[tuple[str, tuple[str, ...]]] = [
    ("revenue", ("营业收入", "收入", "营收", "revenue", "revenues")),
    ("net_profit", ("净利润", "归母净利润", "net income", "net loss")),
    ("gross_margin", ("毛利率", "gross margin")),
    ("cash_flow", ("现金流", "经营活动产生的现金流量净额", "cash flow")),
    ("accounts_receivable", ("应收账款",)),
    ("inventory", ("存货",)),
    ("assets", ("总资产", "资产总计")),
    ("liabilities", ("总负债", "负债合计")),
]

_FINANCE_SECTION_HINTS: tuple[str, ...] = (
    "management discussion",
    "md&a",
    "liquidity",
    "results of operations",
    "risk factors",
    "business overview",
    "财务报表附注",
    "管理层讨论",
)

FINANCE_STATEMENT_LABELS = tuple(_FINANCE_STATEMENT_LABELS)
FINANCE_METRIC_LABELS = tuple(_FINANCE_METRIC_LABELS)
_SECTION_ROLE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("results_of_operations", ("results of operations", "operating results", "operating income", "operating loss")),
    ("liquidity", ("liquidity and capital resources", "liquidity", "capital resources")),
    ("critical_accounting", ("critical accounting", "critical accounting policies")),
    ("market_risk", ("quantitative and qualitative disclosures about market risk", "market risk")),
    ("risk_factors", ("risk factors",)),
    ("restatement", ("restatement", "amended and restated", "temporary equity", "equity classification")),
    ("exhibits", ("exhibit", "certification", "sarbanes-oxley", "section 302", "section 906")),
    ("company_overview", ("blank check company", "business combination", "cayman islands exempted company")),
    ("mda", ("management's discussion and analysis", "management’s discussion and analysis", "md&a")),
)
_TOPIC_TAG_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("revenue", ("revenue", "revenues", "sales")),
    ("cost_of_revenue", ("cost of revenue", "costs of revenue", "cost of sales")),
    ("gross_profit", ("gross profit",)),
    ("gross_margin", ("gross margin",)),
    ("operating_expense", ("operating expenses", "general and administrative", "g&a", "administrative expenses")),
    ("operating_income", ("operating income", "operating loss", "operating results")),
    ("net_income", ("net income", "net loss")),
    ("liquidity", ("liquidity", "cash flows", "working capital", "capital resources")),
)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(str(n).lower() in lowered for n in needles if n)


def _text_parts(*parts: Any) -> str:
    return " ".join(str(part).strip() for part in parts if part).strip()


def _normalize_section_path(metadata: dict[str, Any], title: str) -> list[str]:
    raw = metadata.get("section_path")
    if isinstance(raw, list):
        out = [str(item).strip() for item in raw if str(item).strip()]
        if out:
            return out[:12]
    raw = metadata.get("heading_path")
    if isinstance(raw, list):
        out = [str(item).strip() for item in raw if str(item).strip()]
        if out:
            return out[:12]
    section_title = str(metadata.get("section_title") or "").strip()
    if section_title:
        return [section_title]
    clean_title = str(title or "").strip()
    return [clean_title] if clean_title else []


def infer_finance_section_role(
    *,
    title: str | None,
    text: str,
    metadata: dict[str, Any] | None,
) -> str | None:
    meta = dict(metadata or {})
    raw = str(meta.get("section_role") or "").strip().lower()
    if raw:
        return raw
    section_path = _normalize_section_path(meta, title or "")
    surface = _text_parts(" ".join(section_path), title or "", text[:1200]).lower()
    for role, needles in _SECTION_ROLE_RULES:
        if _contains_any(surface, needles):
            return role
    return "unknown" if section_path else None


def infer_finance_topic_tags(
    *,
    title: str | None,
    text: str,
    metadata: dict[str, Any] | None,
) -> list[str]:
    meta = dict(metadata or {})
    raw = meta.get("topic_tags")
    if isinstance(raw, list):
        out = [str(item).strip().lower() for item in raw if str(item).strip()]
        if out:
            return list(dict.fromkeys(out))[:12]
    surface = _text_parts(
        " ".join(_normalize_section_path(meta, title or "")),
        title or "",
        text[:2000],
    ).lower()
    out: list[str] = []
    for tag, needles in _TOPIC_TAG_RULES:
        if _contains_any(surface, needles):
            out.append(tag)
    return out[:12]


def infer_finance_leaf_role(
    *,
    title: str | None,
    text: str,
    metadata: dict[str, Any] | None,
) -> str | None:
    meta = dict(metadata or {})
    raw = str(meta.get("leaf_role") or "").strip().lower()
    if raw:
        return raw
    section_role = infer_finance_section_role(title=title, text=text, metadata=meta) or "unknown"
    surface = _text_parts(
        " ".join(_normalize_section_path(meta, title or "")),
        title or "",
        text[:2000],
    ).lower()
    if not surface:
        return None
    explanation_markers = (
        "due to",
        "primarily",
        "mainly",
        "driven by",
        "resulted from",
        "because",
        "increase in",
        "decrease in",
        "offset by",
    )
    has_explanation = any(marker in surface for marker in explanation_markers)
    if any(
        marker in surface
        for marker in (
            "table of contents",
            "item 2. management",
            "references to the company",
            "the following discussion and analysis should be read in conjunction with",
            "blank check company",
            "formed for the purpose of effecting a merger",
        )
    ):
        return "mda_intro" if section_role in {"mda", "results_of_operations"} else "boilerplate"
    if section_role == "restatement":
        return "restatement_note"
    if section_role == "results_of_operations":
        if has_explanation:
            return "results_driver"
        if any(
            marker in surface
            for marker in (
                "forward-looking statements",
                "cautionary note regarding forward-looking",
                "not historical facts",
                "could cause actual results to differ materially",
            )
        ):
            return "forward_looking"
        return "mda_intro"
    if section_role == "liquidity":
        if has_explanation:
            return "liquidity_driver"
        if any(
            marker in surface
            for marker in (
                "forward-looking statements",
                "cautionary note regarding forward-looking",
                "not historical facts",
                "could cause actual results to differ materially",
            )
        ):
            return "forward_looking"
        return "mda_intro"
    if any(
        marker in surface
        for marker in (
            "forward-looking statements",
            "cautionary note regarding forward-looking",
            "not historical facts",
            "could cause actual results to differ materially",
        )
    ):
        return "forward_looking"
    if section_role in {"market_risk", "exhibits", "company_overview"}:
        return "boilerplate"
    if section_role == "risk_factors":
        return "risk_factor_item" if len(surface) >= 200 else "boilerplate"
    return "other"


def _extract_domain(source_text: str, metadata: dict[str, Any]) -> str:
    hinted = str(metadata.get("domain") or "").strip().lower()
    if hinted:
        return hinted
    if _contains_any(
        source_text,
        (
            "财务报表",
            "资产负债表",
            "利润表",
            "现金流量表",
            "年报",
            "季报",
            "招股说明书",
            "营业收入",
            "净利润",
        ),
    ):
        return "finance"
    return "generic"


def _extract_finance_statement(source_text: str) -> str | None:
    for value, labels in _FINANCE_STATEMENT_LABELS:
        if _contains_any(source_text, labels):
            return value
    return None


def _extract_finance_metric_keys(source_text: str) -> list[str]:
    out: list[str] = []
    for value, labels in _FINANCE_METRIC_LABELS:
        if _contains_any(source_text, labels):
            out.append(value)
    return out[:8]


def _extract_finance_period(source_text: str) -> str | None:
    period, _ = _extract_finance_period_with_confidence(source_text)
    return period


def _extract_finance_period_with_confidence(source_text: str) -> tuple[str | None, str | None]:
    for match in _YEAR_PERIOD_RE.finditer(source_text or ""):
        year, q_code, quarter_zh, half_year, yearly = match.groups()
        start, end = match.span()
        left = max(0, start - 28)
        right = min(len(source_text), end + 28)
        context = source_text[left:right]
        lowered = context.lower()

        # Avoid false periods from accounting-standard identifiers, e.g. "ASU 2014-15".
        if _STANDARD_NOISE_RE.search(lowered) and re.search(rf"{year}\s*-\s*\d{{1,2}}", lowered):
            continue

        # When no explicit quarter/year marker exists, require temporal cues nearby.
        has_explicit_suffix = bool(q_code or quarter_zh or half_year or yearly)
        has_time_cue = bool(_TIME_CONTEXT_CUE_RE.search(lowered))
        if not has_explicit_suffix and not has_time_cue:
            continue

        if q_code:
            period = f"{year}{q_code.upper()}"
            confidence = "high"
            return period, confidence
        elif quarter_zh:
            quarter_text = quarter_zh.replace("第", "").replace("季度", "")
            quarter_map = {"一": "Q1", "二": "Q2", "三": "Q3", "四": "Q4", "1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"}
            period = f"{year}{quarter_map.get(quarter_text, '')}".rstrip()
            confidence = "high"
            return period, confidence
        elif half_year:
            period = f"{year}H1"
            confidence = "high"
            return period, confidence
        elif yearly:
            period = year
            confidence = "high"
            return period, confidence
        else:
            period = year
            confidence = "high" if has_time_cue else "low"
            return period, confidence
    return None, None


def _normalize_finance_form(form: str | None) -> str | None:
    text = str(form or "").strip().upper().replace(" ", "")
    if not text:
        return None
    text = text.replace(".", "-")
    text = text.replace("10K", "10-K").replace("10Q", "10-Q").replace("8K", "8-K")
    if text.endswith("/A"):
        text = text[:-2]
    return text


def _extract_content_type(
    domain: str,
    node_type: str,
    title: str,
    text: str,
    metadata: dict[str, Any],
) -> str:
    hinted = str(metadata.get("content_type") or "").strip().lower()
    if hinted:
        return hinted
    if node_type == "document_summary":
        return "document_summary"
    if node_type == "summary":
        return "section_summary"
    source_text = _text_parts(title, text[:600])
    if domain == "finance":
        if _contains_any(source_text, ("附注",)):
            return "financial_note"
        if _contains_any(source_text, ("资产负债表", "利润表", "现金流量表", "权益变动表")):
            return "financial_statement"
        return "finance_chunk"
    return node_type or "chunk"


def _extract_source_section(title: str, metadata: dict[str, Any]) -> str | None:
    section_title = metadata.get("section_title")
    if section_title:
        return str(section_title)
    heading_path = metadata.get("heading_path")
    if isinstance(heading_path, list) and heading_path:
        return str(heading_path[-1])
    if title:
        return title
    return None


def _build_search_hints(fields: dict[str, Any], metadata: dict[str, Any], title: str) -> str:
    parts: list[str] = []
    source_file_name = metadata.get("source_file_name")
    if source_file_name:
        parts.append(str(source_file_name))
    source_section = fields.get("source_section")
    if source_section:
        parts.append(str(source_section))
    section_leaf = fields.get("section_leaf")
    if section_leaf:
        parts.append(str(section_leaf))
    section_path_text = fields.get("section_path_text")
    if section_path_text:
        parts.append(str(section_path_text))
    section_role = fields.get("section_role")
    if section_role:
        parts.append(str(section_role))
    leaf_role = fields.get("leaf_role")
    if leaf_role:
        parts.append(str(leaf_role))
    for key in (
        "domain",
        "content_type",
        "finance_statement",
    ):
        value = fields.get(key)
        if value:
            parts.append(str(value))
    # finance_period is a weak extracted signal; only promote high-confidence values.
    if fields.get("finance_period_confidence") == "high" and fields.get("finance_period"):
        parts.append(str(fields["finance_period"]))
    for key in ("finance_metric_keys",):
        value = fields.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
    for key in ("finance_forms", "finance_form_base"):
        value = fields.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
    if fields.get("domain") == "finance":
        for tag in fields.get("topic_tags") or []:
            parts.append(str(tag))
        source_section = str(fields.get("source_section") or "")
        title_text = str(title or "")
        if _contains_any(
            f"{source_section} {title_text}".lower(),
            ("management", "discussion", "md&a", "liquidity"),
        ):
            parts.extend(_FINANCE_SECTION_HINTS)
    if title:
        parts.append(title)
    return " ".join(dict.fromkeys(part.strip() for part in parts if part and str(part).strip()))


def build_retrieval_fields(
    *,
    node_type: str,
    level: int,
    title: str | None,
    text: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    meta = dict(metadata or {})
    title_text = title or ""
    source_text = _text_parts(
        meta.get("source_file_name"),
        meta.get("section_title"),
        title_text,
        text[:2000],
    )
    section_path = _normalize_section_path(meta, title_text)
    domain = _extract_domain(source_text, meta)
    fields: dict[str, Any] = {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "domain": domain,
        "content_type": _extract_content_type(domain, node_type, title_text, text, meta),
        "language": str(meta.get("language") or "zh"),
        "source_section": _extract_source_section(title_text, meta),
        "section_leaf": section_path[-1] if section_path else None,
        "section_path": section_path or None,
        "section_path_text": " > ".join(section_path) if section_path else None,
        "node_type": node_type,
        "level": level,
    }
    if domain == "finance":
        fields["topic_tags"] = infer_finance_topic_tags(
            title=title_text,
            text=text,
            metadata=meta,
        )
        fields["finance_statement"] = _extract_finance_statement(source_text)
        period, confidence = _extract_finance_period_with_confidence(source_text)
        fields["finance_metric_keys"] = _extract_finance_metric_keys(source_text)
        for key in (
            "finance_metric_exact_keys",
            "finance_forms",
            "finance_accns",
            "finance_period_end_dates",
        ):
            value = meta.get(key)
            if isinstance(value, list) and value:
                fields[key] = [str(item) for item in value if item][:24]
        # Single field for fiscal period / year filter alignment (planner uses terms on this key).
        meta_fp = meta.get("finance_period")
        legacy_years = meta.get("finance_period_years")
        if isinstance(meta_fp, list) and meta_fp:
            fields["finance_period"] = [str(item).strip() for item in meta_fp if str(item).strip()][:24]
        elif isinstance(legacy_years, list) and legacy_years:
            fields["finance_period"] = [str(item).strip() for item in legacy_years if str(item).strip()][:24]
        elif isinstance(meta_fp, str) and meta_fp.strip():
            fields["finance_period"] = meta_fp.strip()
        elif period:
            fields["finance_period"] = period
            if confidence:
                fields["finance_period_confidence"] = confidence
        forms = fields.get("finance_forms")
        if isinstance(forms, list) and forms:
            normalized = [
                item
                for item in (_normalize_finance_form(form) for form in forms)
                if item
            ]
            if normalized:
                fields["finance_form_base"] = list(dict.fromkeys(normalized))[:12]
    fields["search_hints"] = _build_search_hints(fields, meta, title_text)
    return {key: value for key, value in fields.items() if value not in (None, "", [], {})}

