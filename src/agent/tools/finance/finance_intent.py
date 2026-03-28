"""LLM-driven finance intent: route + optional SQL row budget + hybrid refinement.

Extensible entrypoint: ``resolve_finance_intent``. Add fields to ``FinanceIntent`` and
extend ``_intent_from_llm_dict`` / prompts without changing SQL execution in ``rag_service``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from core.config import config
from tools.llm import get_llm

from .finance_query_plan import FINANCE_SQL_PROMPT_MAX_ROWS
from .question_router import FinanceRoute, default_hybrid_route, route_finance_by_rules, _parse_route_json

QuestionKind = Literal["narrative", "numeric", "mixed"]


@dataclass(frozen=True)
class FinanceIntent:
    """Resolved per request; drives SQL inclusion and prompt row cap."""

    route: FinanceRoute
    question_kind: QuestionKind
    sql_prompt_max_rows: int | None
    intent_detail: str

    def effective_sql_prompt_max_rows(self) -> int:
        cap = self.sql_prompt_max_rows
        if cap is None:
            return FINANCE_SQL_PROMPT_MAX_ROWS
        return max(0, min(int(cap), FINANCE_SQL_PROMPT_MAX_ROWS))


_INTENT_SYSTEM_CLASSIFY = """You classify SEC / financial Q&A intent. Reply with ONLY valid JSON, no markdown:
{
  "need_sql": true/false,
  "need_rag": true/false,
  "question_kind": "narrative" | "numeric" | "mixed",
  "sql_prompt_max_rows": null or integer 0-120
}

Rules:
- need_sql=true: user needs structured XBRL/facts (amounts, dates, metrics, forms, filings).
- need_rag=true: user needs prose (MD&A, risks, explanations, management wording, narrative).
- question_kind=narrative: primary answer is filing text (e.g. MD&A, liquidity discussion); numbers optional.
- question_kind=numeric: primary answer is numbers/metrics from facts.
- question_kind=mixed: both matter equally.
- sql_prompt_max_rows: cap how many fact rows go to the LLM (null = default). Use 0 or need_sql=false when the question is purely narrative (e.g. "what does MD&A say") and facts would only waste context.
- If both need_sql and need_rag false, set both true (hybrid) except when the question is empty.
"""

_INTENT_SYSTEM_HYBRID = """Keyword rules suggest BOTH structured facts and narrative retrieval may apply.
Decide what the user actually needs. Reply with ONLY valid JSON, no markdown:
{
  "need_sql": true/false,
  "need_rag": true/false,
  "question_kind": "narrative" | "numeric" | "mixed",
  "sql_prompt_max_rows": null or integer 0-120
}

If the user only wants management discussion / filing narrative (not metric values), set need_sql=false or sql_prompt_max_rows=0.
If they need specific numbers (revenue, cash, EPS) plus explanation, keep need_sql=true and a reasonable row cap (e.g. 24-80).
sql_prompt_max_rows=null means use server default cap.
"""


def _kind_from_rule(route: FinanceRoute) -> QuestionKind:
    if route.need_sql and not route.need_rag:
        return "numeric"
    if route.need_rag and not route.need_sql:
        return "narrative"
    return "mixed"


def _intent_from_llm_dict(obj: dict[str, Any], *, source: str, detail: str) -> FinanceIntent:
    ns = obj.get("need_sql")
    nr = obj.get("need_rag")
    if not isinstance(ns, bool) or not isinstance(nr, bool):
        raise ValueError("invalid need_sql/need_rag")
    if not ns and not nr:
        route = FinanceRoute(True, True, "llm", f"{detail}_coerced_hybrid")
    else:
        route = FinanceRoute(ns, nr, "llm", detail)

    raw_kind = obj.get("question_kind")
    if raw_kind in ("narrative", "numeric", "mixed"):
        qk: QuestionKind = raw_kind
    else:
        qk = "mixed"

    cap = obj.get("sql_prompt_max_rows")
    sql_cap: int | None
    if cap is None:
        sql_cap = None
    else:
        try:
            sql_cap = int(cap)
        except (TypeError, ValueError):
            sql_cap = None
        else:
            sql_cap = max(0, min(sql_cap, FINANCE_SQL_PROMPT_MAX_ROWS))

    if sql_cap == 0:
        route = FinanceRoute(False, route.need_rag, route.source, f"{route.detail}|sql_cap_0")
        sql_cap = None

    return FinanceIntent(
        route=route,
        question_kind=qk,
        sql_prompt_max_rows=sql_cap,
        intent_detail=detail,
    )


async def _llm_intent(question: str, *, hybrid_refine: bool) -> FinanceIntent:
    q = (question or "").strip()[:2000]
    system = _INTENT_SYSTEM_HYBRID if hybrid_refine else _INTENT_SYSTEM_CLASSIFY
    llm = get_llm(model_name=config.default_model, temperature=0.0)
    msg = await llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=f"Question:\n{q}")]
    )
    text = msg.content if hasattr(msg, "content") else str(msg)
    obj = _parse_route_json(text)
    if not obj:
        raise ValueError("intent_unparseable")
    mode = "hybrid_refine" if hybrid_refine else "llm_intent"
    return _intent_from_llm_dict(obj, source="llm", detail=mode)


async def resolve_finance_intent(question: str) -> FinanceIntent:
    """Rule-first; LLM when ambiguous; optional LLM hybrid refinement when both signals fire."""
    q = (question or "").strip()
    if not q:
        r = FinanceRoute(False, True, "default", "empty_question")
        return FinanceIntent(r, "narrative", None, "empty_question")

    if config.finance_llm_route_only:
        try:
            return await _llm_intent(q, hybrid_refine=False)
        except Exception as exc:
            logger.warning("[FinanceIntent] llm_route_only failed: {}", exc)
            return FinanceIntent(default_hybrid_route(), "mixed", None, "llm_route_only_failed")

    ruled = route_finance_by_rules(q)

    if ruled is None:
        if not config.finance_sql_routing_llm_fallback:
            r = default_hybrid_route()
            return FinanceIntent(r, "mixed", None, "ambiguous_no_llm")
        try:
            return await _llm_intent(q, hybrid_refine=False)
        except Exception as exc:
            logger.warning("[FinanceIntent] LLM classify failed: {}", exc)
            return FinanceIntent(default_hybrid_route(), "mixed", None, "llm_classify_failed")

    if ruled.need_sql and ruled.need_rag and config.finance_llm_hybrid_refine:
        try:
            return await _llm_intent(q, hybrid_refine=True)
        except Exception as exc:
            logger.warning("[FinanceIntent] hybrid refine failed, using rules: {}", exc)
            return FinanceIntent(
                ruled,
                _kind_from_rule(ruled),
                None,
                "hybrid_refine_failed",
            )

    return FinanceIntent(ruled, _kind_from_rule(ruled), None, ruled.detail)


async def llm_finance_route_only(question: str) -> FinanceRoute:
    """Backward-compatible entry used by ``question_router.route_finance_with_llm``."""
    return (await _llm_intent((question or "").strip(), hybrid_refine=False)).route