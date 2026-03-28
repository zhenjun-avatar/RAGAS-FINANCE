"""Derive a structured finance query plan from a natural-language question.

This module is the single place where we map a question to:
1. exact-ish SQL filters (metrics / forms / dates / years / comparison mode)
2. a retrieval query string for dense/sparse retrieval

Extensible: add rows to `_PHRASE_TO_TAGS` or `_FORM_SPECS`; callers stay unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .sec_company_facts import extract_metric_hints_from_question

# Keep in sync with merge cap used when formatting SQL for the LLM.
FINANCE_SQL_EXACT_LIMIT = 80
FINANCE_SQL_EXACT_MIN_ROWS = 12
FINANCE_SQL_RECENT_LIMIT = 150
FINANCE_SQL_HINT_QUERY_LIMIT = 120
FINANCE_SQL_MERGED_CAP = 220
FINANCE_SQL_PROMPT_MAX_ROWS = 120

# Longer phrases first so "cost of revenue" wins over "revenue".
_PHRASE_TO_TAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cost of revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
    ("gross profit", ("GrossProfit",)),
    ("operating income", ("OperatingIncomeLoss",)),
    ("accounts payable", ("AccountsPayableCurrent", "AccountsPayable")),
    ("cash and cash equivalents", ("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash")),
    ("net income", ("NetIncomeLoss", "NetIncomeLossAttributableToParent", "NetIncome")),
    ("净利润", ("NetIncomeLoss", "NetIncomeLossAttributableToParent", "NetIncome")),
    ("净亏损", ("NetIncomeLoss",)),
    ("net loss", ("NetIncomeLoss",)),
    ("revenue", ("Revenues", "Revenue", "SalesRevenueNet", "OperatingRevenues")),
    ("sales", ("SalesRevenueNet", "Revenues")),
    ("eps", ("EarningsPerShareBasic", "EarningsPerShareDiluted")),
    ("ebitda", ("EarningsBeforeInterestTaxesDepreciationAmortization",)),
    ("interest income", ("InvestmentIncomeInterest", "InterestIncomeExpenseNet")),
    ("entity public float", ("EntityPublicFloat",)),
    ("public float", ("EntityPublicFloat",)),
)

# Canonical XBRL metric local-name dictionary used by LLM planner validation.
FINANCE_METRIC_KEY_DICTIONARY: tuple[str, ...] = tuple(
    sorted({tag for _, tags in _PHRASE_TO_TAGS for tag in tags})
)

# (canonical_form, regex) — longer /A variants first.
_FORM_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("10-Q/A", re.compile(r"10\s*[-.]?\s*Q\s*/\s*A\b", re.I)),
    ("10-K/A", re.compile(r"10\s*[-.]?\s*K\s*/\s*A\b", re.I)),
    ("8-K/A", re.compile(r"8\s*[-.]?\s*K\s*/\s*A\b", re.I)),
    ("10-Q", re.compile(r"10\s*[-.]?\s*Q\b", re.I)),
    ("10-K", re.compile(r"10\s*[-.]?\s*K\b", re.I)),
    ("8-K", re.compile(r"8\s*[-.]?\s*K\b", re.I)),
)

_NARRATIVE_TARGET_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "management_discussion",
        (
            "md&a",
            "management discussion",
            "management's discussion",
            "management discussion and analysis",
            "results of operations",
            "讨论与分析",
            "管理层讨论",
        ),
    ),
    (
        "liquidity",
        (
            "liquidity",
            "capital resources",
            "working capital",
            "cash runway",
            "流动性",
            "资本资源",
            "营运资金",
        ),
    ),
    (
        "going_concern",
        (
            "going concern",
            "substantial doubt",
            "持续经营",
        ),
    ),
    (
        "risk_factors",
        (
            "risk factors",
            "风险因素",
        ),
    ),
)


def _normalize_form_base(value: str) -> str:
    text = str(value or "").strip().upper().replace(" ", "")
    text = text.replace(".", "-")
    text = text.replace("10K", "10-K").replace("10Q", "10-Q").replace("8K", "8-K")
    if text.endswith("/A"):
        text = text[:-2]
    return text


@dataclass(frozen=True)
class FinanceQueryPlan:
    """Derived once per ask; drives SQL row selection and retrieval query text."""

    metric_sql_hints: tuple[str, ...]
    metric_exact_keys: tuple[str, ...]
    form_filters: tuple[str, ...] | None
    period_end_dates: tuple[str, ...]
    period_years: tuple[int, ...]
    compare_mode: str
    prefer_recent: bool
    retrieval_query: str

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "metric_sql_hints": list(self.metric_sql_hints),
            "metric_exact_keys": list(self.metric_exact_keys),
            "form_filters": list(self.form_filters) if self.form_filters else None,
            "period_end_dates": list(self.period_end_dates),
            "period_years": list(self.period_years),
            "compare_mode": self.compare_mode,
            "prefer_recent": self.prefer_recent,
            "retrieval_query": self.retrieval_query,
        }

    def to_retrieval_filters(self) -> dict[str, list[str]]:
        filters: dict[str, list[str]] = {}
        if self.metric_exact_keys:
            filters["finance_metric_exact_keys"] = [str(v) for v in self.metric_exact_keys if str(v).strip()]
        if self.form_filters:
            base_forms = [
                _normalize_form_base(v)
                for v in self.form_filters
                if str(v).strip()
            ]
            filters["finance_form_base"] = list(dict.fromkeys(form for form in base_forms if form))
        if self.period_end_dates:
            filters["finance_period_end_dates"] = [str(v) for v in self.period_end_dates if str(v).strip()]
        if self.period_years:
            filters["finance_period_years"] = [str(v) for v in self.period_years]
        return {k: v for k, v in filters.items() if v}


@dataclass(frozen=True)
class FilingHypothesis:
    accession: str
    weight: float
    reasons: tuple[str, ...] = ()

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "accession": self.accession,
            "weight": round(float(self.weight), 4),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RetrievalBudget:
    sql_row_budget: int
    summary_candidates: int
    leaf_candidates: int
    per_filing_cap: int
    max_second_pass_accns: int

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "sql_row_budget": self.sql_row_budget,
            "summary_candidates": self.summary_candidates,
            "leaf_candidates": self.leaf_candidates,
            "per_filing_cap": self.per_filing_cap,
            "max_second_pass_accns": self.max_second_pass_accns,
        }


@dataclass(frozen=True)
class EvidenceRequirements:
    need_narrative: bool
    need_numeric_fact: bool
    need_cross_filing: bool

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "need_narrative": self.need_narrative,
            "need_numeric_fact": self.need_numeric_fact,
            "need_cross_filing": self.need_cross_filing,
        }


@dataclass(frozen=True)
class EvidencePlan:
    question_mode: str
    sql_plan: FinanceQueryPlan
    filing_hypotheses: tuple[FilingHypothesis, ...]
    narrative_targets: tuple[str, ...]
    term_targets: tuple[str, ...]
    retrieval_budget: RetrievalBudget
    evidence_requirements: EvidenceRequirements

    @property
    def retrieval_query(self) -> str:
        return self.sql_plan.retrieval_query

    def to_retrieval_filters(self) -> dict[str, list[str]]:
        return self.sql_plan.to_retrieval_filters()

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "question_mode": self.question_mode,
            "sql_plan": self.sql_plan.to_debug_dict(),
            "filing_hypotheses": [item.to_debug_dict() for item in self.filing_hypotheses],
            "narrative_targets": list(self.narrative_targets),
            "term_targets": list(self.term_targets),
            "retrieval_budget": self.retrieval_budget.to_debug_dict(),
            "evidence_requirements": self.evidence_requirements.to_debug_dict(),
        }


def _alias_metric_hints(question: str) -> list[str]:
    q = (question or "").lower()
    out: list[str] = []
    for phrase, tags in sorted(_PHRASE_TO_TAGS, key=lambda x: -len(x[0])):
        if phrase in q:
            out.extend(tags)
    return list(dict.fromkeys(out))


def filter_metric_keys_with_dictionary(keys: list[str] | tuple[str, ...], *, limit: int = 16) -> tuple[str, ...]:
    """Keep only dictionary-approved XBRL local names (stable order)."""
    if not keys:
        return ()
    allow = set(FINANCE_METRIC_KEY_DICTIONARY)
    out: list[str] = []
    seen: set[str] = set()
    for raw in keys:
        s = str(raw).strip()
        if "." in s:
            s = s.split(".")[-1].strip()
        if not s or s in seen or s not in allow:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max(1, int(limit)):
            break
    return tuple(out)


def _extract_form_filters(question: str) -> tuple[str, ...] | None:
    q = question or ""
    seen: dict[str, None] = {}
    for canonical, rx in _FORM_SPECS:
        if rx.search(q) and canonical not in seen:
            seen[canonical] = None
    return tuple(seen.keys()) if seen else None


def _extract_period_end_dates(question: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for match in re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", question or ""):
        if match not in seen:
            seen.add(match)
            out.append(match)
    return tuple(out[:8])


def _extract_period_years(question: str) -> tuple[int, ...]:
    q = question or ""
    out: list[int] = []
    seen: set[int] = set()
    for raw in re.findall(r"\b(20\d{2})\b", q):
        year = int(raw)
        if year not in seen:
            seen.add(year)
            out.append(year)
    return tuple(out[:6])


def _detect_compare_mode(question: str) -> str:
    q = (question or "").lower()
    if any(token in q for token in ("year over year", "yoy", "同比", "去年同期")):
        return "yoy"
    if any(token in q for token in ("quarter over quarter", "qoq", "环比", "上一季度")):
        return "qoq"
    if re.search(r"\b(latest|most recent|newest)\b", q) or "最新" in q or "最近" in q:
        return "latest"
    if _extract_period_end_dates(question):
        return "point_in_time"
    return "generic"


def _prefer_recent(question: str, compare_mode: str) -> bool:
    q = (question or "").lower()
    if compare_mode in {"latest", "point_in_time"}:
        return True
    return bool(
        re.search(r"\b(latest|most recent|newest|filed)\b", q)
        or "最新" in q
        or "最近" in q
        or "申报" in q
    )


def _build_retrieval_query(question: str, metric_sql_hints: tuple[str, ...]) -> str:
    base = (question or "").strip()
    if not base:
        return ""
    if not metric_sql_hints:
        return base
    tail = " ".join(metric_sql_hints[:12])
    return f"{base}\n\nXBRL metrics: {tail}"


def _detect_narrative_targets(question: str) -> tuple[str, ...]:
    lowered = (question or "").lower()
    matched: list[str] = []
    for value, aliases in _NARRATIVE_TARGET_RULES:
        if any(alias.lower() in lowered for alias in aliases):
            matched.append(value)
    return tuple(matched)


def _term_targets_for_narrative_targets(targets: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for value, aliases in _NARRATIVE_TARGET_RULES:
        if value not in targets:
            continue
        out.extend(list(aliases[:3]))
    return out


def _derive_question_mode(
    question: str,
    plan: FinanceQueryPlan,
    *,
    question_kind: str | None,
    narrative_targets: tuple[str, ...],
) -> str:
    lowered = (question or "").lower()
    if len(plan.period_years) >= 2 or any(
        token in lowered for token in ("compare", "versus", "vs", "变化", "对比", "trend", "evolution")
    ):
        return "cross_filing_compare"
    if question_kind == "numeric":
        return "facts_only"
    if question_kind == "narrative":
        return "narrative_only"
    if narrative_targets and not plan.metric_exact_keys:
        return "mixed_narrative_first"
    if plan.metric_exact_keys or plan.period_end_dates:
        return "mixed_facts_first"
    return "mixed_narrative_first" if narrative_targets else "facts_only"


def _derive_evidence_requirements(question_mode: str) -> EvidenceRequirements:
    if question_mode == "facts_only":
        return EvidenceRequirements(
            need_narrative=False,
            need_numeric_fact=True,
            need_cross_filing=False,
        )
    if question_mode == "narrative_only":
        return EvidenceRequirements(
            need_narrative=True,
            need_numeric_fact=False,
            need_cross_filing=False,
        )
    if question_mode == "cross_filing_compare":
        return EvidenceRequirements(
            need_narrative=True,
            need_numeric_fact=True,
            need_cross_filing=True,
        )
    return EvidenceRequirements(
        need_narrative=True,
        need_numeric_fact=True,
        need_cross_filing=False,
    )


def _derive_retrieval_budget(question_mode: str) -> RetrievalBudget:
    if question_mode == "facts_only":
        return RetrievalBudget(
            sql_row_budget=80,
            summary_candidates=6,
            leaf_candidates=10,
            per_filing_cap=3,
            max_second_pass_accns=2,
        )
    if question_mode == "narrative_only":
        return RetrievalBudget(
            sql_row_budget=32,
            summary_candidates=10,
            leaf_candidates=14,
            per_filing_cap=5,
            max_second_pass_accns=3,
        )
    if question_mode == "cross_filing_compare":
        return RetrievalBudget(
            sql_row_budget=96,
            summary_candidates=10,
            leaf_candidates=14,
            per_filing_cap=4,
            max_second_pass_accns=4,
        )
    return RetrievalBudget(
        sql_row_budget=56,
        summary_candidates=8,
        leaf_candidates=12,
        per_filing_cap=4,
        max_second_pass_accns=3,
    )


def build_finance_evidence_plan(
    question: str,
    *,
    sql_plan: FinanceQueryPlan | None = None,
    question_kind: str | None = None,
) -> EvidencePlan:
    plan = sql_plan or build_finance_query_plan(question)
    narrative_targets = _detect_narrative_targets(question)
    question_mode = _derive_question_mode(
        question,
        plan,
        question_kind=question_kind,
        narrative_targets=narrative_targets,
    )
    requirements = _derive_evidence_requirements(question_mode)
    budget = _derive_retrieval_budget(question_mode)
    terms: list[str] = []
    terms.extend(_term_targets_for_narrative_targets(narrative_targets))
    terms.extend(plan.metric_sql_hints[:6])
    terms.extend(plan.form_filters or ())
    terms.extend(str(year) for year in plan.period_years[:4])
    terms.extend(plan.period_end_dates[:4])
    deduped_terms = tuple(list(dict.fromkeys(str(item).strip() for item in terms if str(item).strip()))[:16])
    return EvidencePlan(
        question_mode=question_mode,
        sql_plan=plan,
        filing_hypotheses=(),
        narrative_targets=narrative_targets,
        term_targets=deduped_terms,
        retrieval_budget=budget,
        evidence_requirements=requirements,
    )


def build_finance_query_plan(question: str) -> FinanceQueryPlan:
    """Merge metric aliases and parse exact SQL filters from the question."""
    pascal = extract_metric_hints_from_question(question, max_hints=12)
    alias = _alias_metric_hints(question)
    merged = list(dict.fromkeys([*pascal, *alias]))[:16]
    forms = _extract_form_filters(question)
    period_end_dates = _extract_period_end_dates(question)
    period_years = _extract_period_years(question)
    compare_mode = _detect_compare_mode(question)
    rq = _build_retrieval_query(question, tuple(merged))
    return FinanceQueryPlan(
        metric_sql_hints=tuple(merged),
        metric_exact_keys=tuple(merged),
        form_filters=forms,
        period_end_dates=period_end_dates,
        period_years=period_years,
        compare_mode=compare_mode,
        prefer_recent=_prefer_recent(question, compare_mode),
        retrieval_query=rq,
    )
