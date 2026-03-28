"""LLM-built FinanceQueryPlan; falls back to heuristic ``build_finance_query_plan`` on failure.

Extend sanitization / schema here; callers in ``rag_service`` only switch on config flag.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from core.config import config
from tools.llm import get_llm

from .finance_query_plan import (
    EvidencePlan,
    FinanceQueryPlan,
    _build_retrieval_query,
    build_finance_evidence_plan,
    filter_metric_keys_with_dictionary,
)
from .question_router import _parse_route_json

_METRIC_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
_COMPARE_MODES = frozenset({"generic", "latest", "yoy", "qoq", "point_in_time"})

_PLANNER_SYSTEM = """You map a user question to SEC XBRL company-facts query fields. Reply ONLY with valid JSON, no markdown:
{
  "metric_sql_hints": ["PascalCaseLocalName", ...],
  "metric_exact_keys": ["same as hints or tighter subset", ...],
  "form_filters": ["10-K", "10-Q"] or null,
  "period_end_dates": ["YYYY-MM-DD", ...],
  "period_years": [2023, 2024],
  "compare_mode": "generic" | "latest" | "yoy" | "qoq" | "point_in_time",
  "prefer_recent": true/false
}

metric_* values are the LOCAL tag name only (e.g. Cash, Revenues), not us-gaap.Cash.
Use empty arrays when unknown. prefer_recent=true when user wants latest filing or most recent period.
"""


def _clean_metric_list(raw: Any, *, limit: int = 16) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    for item in raw:
        s = str(item).strip()
        if "." in s:
            s = s.split(".")[-1].strip()
        if _METRIC_TOKEN.match(s) and s not in out:
            out.append(s)
        if len(out) >= limit:
            break
    return filter_metric_keys_with_dictionary(out, limit=limit)


def _clean_forms(raw: Any) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    norm: list[str] = []
    for item in raw:
        s = str(item).strip().replace(" ", "").upper().replace("10K", "10-K").replace("10Q", "10-Q")
        if re.match(r"^10-[KQ](/A)?$|^8-K(/A)?$", s) and s not in norm:
            norm.append(s)
        if len(norm) >= 8:
            break
    return tuple(norm) if norm else None


def _clean_dates(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    for item in raw:
        m = re.match(r"^(20\d{2}-\d{2}-\d{2})$", str(item).strip())
        if m and m.group(1) not in out:
            out.append(m.group(1))
        if len(out) >= 8:
            break
    return tuple(out)


def _clean_years(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[int] = []
    for item in raw:
        try:
            y = int(item)
        except (TypeError, ValueError):
            continue
        if 1990 <= y <= 2100 and y not in out:
            out.append(y)
        if len(out) >= 6:
            break
    return tuple(out)


def plan_dict_to_finance_plan(question: str, obj: dict[str, Any]) -> FinanceQueryPlan:
    hints = _clean_metric_list(obj.get("metric_sql_hints"))
    exact = _clean_metric_list(obj.get("metric_exact_keys"))
    if not exact and hints:
        exact = hints
    if not hints and exact:
        hints = exact
    forms = _clean_forms(obj.get("form_filters"))
    dates = _clean_dates(obj.get("period_end_dates"))
    years = _clean_years(obj.get("period_years"))
    cm = obj.get("compare_mode")
    compare_mode = cm if isinstance(cm, str) and cm in _COMPARE_MODES else "generic"
    pr = obj.get("prefer_recent")
    prefer_recent = bool(pr) if isinstance(pr, bool) else True
    rq = _build_retrieval_query(question, hints)
    return FinanceQueryPlan(
        metric_sql_hints=hints,
        metric_exact_keys=exact,
        form_filters=forms,
        period_end_dates=dates,
        period_years=years,
        compare_mode=compare_mode,
        prefer_recent=prefer_recent,
        retrieval_query=rq,
    )


async def build_finance_query_plan_llm(question: str) -> FinanceQueryPlan | None:
    """Returns None if the model output is unusable (caller uses heuristic plan)."""
    q = (question or "").strip()[:2000]
    if not q:
        return None
    try:
        llm = get_llm(model_name=config.default_model, temperature=0.0)
        msg = await llm.ainvoke(
            [
                SystemMessage(content=_PLANNER_SYSTEM),
                HumanMessage(content=f"Question:\n{q}"),
            ]
        )
        text = msg.content if hasattr(msg, "content") else str(msg)
        obj = _parse_route_json(text)
        if not obj:
            return None
        plan = plan_dict_to_finance_plan(q, obj)
        if not plan.metric_sql_hints and not plan.metric_exact_keys and not plan.form_filters and not plan.period_end_dates and not plan.period_years:
            return None
        return plan
    except Exception as exc:
        logger.warning("[FinanceQueryPlanLLM] failed: {}", exc)
        return None


async def build_finance_evidence_plan_llm(
    question: str,
    *,
    question_kind: str | None = None,
) -> EvidencePlan | None:
    plan = await build_finance_query_plan_llm(question)
    if plan is None:
        return None
    return build_finance_evidence_plan(
        question,
        sql_plan=plan,
        question_kind=question_kind,
    )
