"""Rule-first intent for finance Q&A; optional LLM when keywords are ambiguous."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

Source = Literal["rule", "llm", "default"]


@dataclass(frozen=True)
class FinanceRoute:
    need_sql: bool
    need_rag: bool
    source: Source
    detail: str


# Chinese + common English tokens for SEC / numeric questions
_SQL_HINTS = (
    "多少",
    "数值",
    "金额",
    "营收",
    "收入",
    "利润",
    "净利润",
    "资产",
    "负债",
    "现金",
    "同比",
    "环比",
    "财年",
    "哪一年",
    "哪期",
    "申报",
    "filed",
    "10-k",
    "10-q",
    "8-k",
    "form",
    "cik",
    "usd",
    "美元",
    "元",
    "revenue",
    "ebitda",
    "eps",
    "metric",
    "xbrl",
    "us-gaap",
)

_RAG_HINTS = (
    "为什么",
    "为何",
    "如何",
    "怎么",
    "风险",
    "描述",
    "解释",
    "分析",
    "原因",
    "影响",
    "管理层",
    "讨论",
    "诉讼",
    "披露",
    "可能",
    "是否",
    "意味着什么",
    "怎么看",
    "summary",
    "narrative",
    "md&a",
    "mda",
)


def _count_hints(text: str, hints: tuple[str, ...]) -> int:
    lower = text.lower()
    n = 0
    for h in hints:
        if h.isascii():
            if re.search(rf"(?<![a-z0-9]){re.escape(h)}(?![a-z0-9])", lower):
                n += 1
        elif h in text:
            n += 1
    return n


def route_finance_by_rules(question: str) -> FinanceRoute | None:
    """
    Returns None if ambiguous (use LLM or default).
    Otherwise returns explicit need_sql / need_rag.
    """
    q = (question or "").strip()
    if not q:
        return FinanceRoute(False, True, "default", "empty_question")

    s = _count_hints(q, _SQL_HINTS)
    r = _count_hints(q, _RAG_HINTS)

    if s > 0 and r > 0:
        return FinanceRoute(True, True, "rule", f"both_signals sql={s} rag={r}")
    if s > 0 and r == 0:
        return FinanceRoute(True, False, "rule", f"sql_only sql={s}")
    if r > 0 and s == 0:
        return FinanceRoute(False, True, "rule", f"rag_only rag={r}")
    return None


def _parse_route_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw)
        if m:
            raw = m.group(1)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


async def route_finance_with_llm(question: str) -> FinanceRoute:
    """Delegates to agentic intent JSON (need_sql, need_rag, question_kind, sql cap)."""
    from tools.finance.finance_intent import llm_finance_route_only

    return await llm_finance_route_only(question)


def default_hybrid_route() -> FinanceRoute:
    """When rules are ambiguous and LLM is off or failed: query SQL + RAG."""
    return FinanceRoute(True, True, "default", "ambiguous_fallback_hybrid")


def format_sql_observations_for_prompt(rows: list[dict[str, Any]], *, max_rows: int = 80) -> str:
    """Compact lines for LLM (auditable facts)."""
    if not rows:
        return "(no matching SEC observation rows)"
    lines: list[str] = []
    for i, row in enumerate(rows[:max_rows], start=1):
        val = row.get("value_numeric")
        if val is None and row.get("value_text"):
            val_s = str(row["value_text"])
        else:
            val_s = "" if val is None else str(val)
        lines.append(
            f"{i}. doc={row.get('document_id')} | {row.get('taxonomy')}.{row.get('metric_key')} | "
            f"{row.get('metric_label') or ''} | {val_s} {row.get('unit') or ''} | "
            f"end={row.get('period_end')} filed={row.get('filed_date')} form={row.get('form')} "
            f"fy={row.get('fy')} fp={row.get('fp')} accn={row.get('accn')}"
        )
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows omitted)")
    return "\n".join(lines)
