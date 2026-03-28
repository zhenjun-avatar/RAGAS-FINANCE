"""Report / product UI locale: infer from question, resolve explicit zh|en|auto, and string tables."""

from __future__ import annotations

from typing import Literal

Locale = Literal["zh", "en"]


def infer_locale_from_question(question: str) -> Locale:
    """Heuristic: CJK-heavy questions → zh; otherwise en."""
    text = (question or "").strip()
    if not text:
        return "zh"
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    n = len(text)
    if cjk >= 4 or (n >= 10 and cjk / n >= 0.18):
        return "zh"
    return "en"


def normalize_report_locale(value: str | None) -> str | None:
    """Returns 'zh', 'en', 'auto', or None if invalid."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("zh", "zh-cn", "cn"):
        return "zh"
    if v in ("en", "en-us", "en-gb"):
        return "en"
    if v in ("auto", ""):
        return "auto"
    return None


def resolve_report_locale(question: str, explicit: str | None) -> Locale:
    """Explicit zh/en wins; 'auto' or invalid uses inference from question."""
    norm = normalize_report_locale(explicit)
    if norm == "zh":
        return "zh"
    if norm == "en":
        return "en"
    return infer_locale_from_question(question)


# --- External evaluation snapshot (observable_metrics labels by metric_id) ---

SNAPSHOT_METRIC_LABELS: dict[Locale, dict[str, str]] = {
    "zh": {
        "evidence_bundle_completeness": "证据包完整度",
        "cross_filing_balance": "跨 filing 平衡度",
        "trace_availability_rate": "Trace 可用率",
        "latency_ms": "本次请求延迟",
        "tokens_per_query": "本次请求 Token",
    },
    "en": {
        "evidence_bundle_completeness": "Evidence bundle completeness",
        "cross_filing_balance": "Cross-filing balance",
        "trace_availability_rate": "Trace availability",
        "latency_ms": "Request latency",
        "tokens_per_query": "Tokens (this request)",
    },
}


def snapshot_metric_label(metric_id: str, locale: Locale) -> str:
    table = SNAPSHOT_METRIC_LABELS.get(locale) or SNAPSHOT_METRIC_LABELS["zh"]
    return table.get(metric_id) or metric_id


# --- Narrative evidence card title prefix ---

def narrative_evidence_title(idx: int, locale: Locale) -> str:
    if locale == "en":
        return f"Evidence {idx}"
    return f"证据 {idx}"


def default_quote_card_title(locale: Locale) -> str:
    return "Excerpt" if locale == "en" else "摘录"


# --- Risk flags (product_surface _risk_flags) ---

RISK_FLAG_TEXT: dict[Locale, dict[str, str]] = {
    "zh": {
        "no_citations": "缺少可引用段落证据",
        "no_sql": "缺少结构化数值支撑",
        "limitations": "当前回答已声明证据不足",
        "filing_divergence": "SQL 与 RAG 的主导 filing 曾出现分歧",
    },
    "en": {
        "no_citations": "No paragraph-level citations available",
        "no_sql": "No structured numeric facts available",
        "limitations": "Answer notes insufficient evidence",
        "filing_divergence": "SQL vs. RAG primary filing diverged",
    },
}


def risk_flag_message(key: str, locale: Locale) -> str:
    table = RISK_FLAG_TEXT.get(locale) or RISK_FLAG_TEXT["zh"]
    return table.get(key) or RISK_FLAG_TEXT["zh"].get(key, key)


# --- Limitations string when no evidence ---

LIMITATIONS_NO_EVIDENCE: dict[Locale, str] = {
    "zh": "未检索到足够证据，回答可能不完整。",
    "en": "Insufficient evidence was retrieved; the answer may be incomplete.",
}


# --- Evidence UI narrative (panels + principles) ---

EVIDENCE_UI_NARRATIVE: dict[Locale, dict[str, object]] = {
    "zh": {
        "panels": (
            "结论区：先给直接回答与适用场景。",
            "证据区：并列展示段落引用、结构化事实、涉及 filing。",
            "风险提示区：清楚说明证据不足、跨 filing 分歧或缺口。",
        ),
        "copy_principles": (
            "先结论，后证据，不要求用户理解底层检索栈。",
            "每个关键结论至少挂一条证据或明确标注缺口。",
            "证据不足时宁可提示风险，不把推断包装成事实。",
        ),
    },
    "en": {
        "panels": (
            "Conclusion: lead with a direct answer and scope.",
            "Evidence: show paragraph quotes, structured facts, and filings side by side.",
            "Risks: clearly flag gaps, low evidence, or cross-filing disagreements.",
        ),
        "copy_principles": (
            "Conclusion first, evidence second—hide retrieval mechanics from users.",
            "Every key claim should cite evidence or mark a gap explicitly.",
            "Prefer caution over implying certainty when evidence is thin.",
        ),
    },
}


def evidence_ui_panels(locale: Locale) -> tuple[str, ...]:
    raw = EVIDENCE_UI_NARRATIVE.get(locale) or EVIDENCE_UI_NARRATIVE["zh"]
    return tuple(raw["panels"])  # type: ignore[arg-type]


def evidence_ui_copy_principles(locale: Locale) -> tuple[str, ...]:
    raw = EVIDENCE_UI_NARRATIVE.get(locale) or EVIDENCE_UI_NARRATIVE["zh"]
    return tuple(raw["copy_principles"])  # type: ignore[arg-type]


# --- CLI / HTML / ASCII report strings ---

REPORT_UI: dict[Locale, dict[str, str]] = {
    "zh": {
        "html_lang": "zh-CN",
        "doc_title": "财务问答结果报告",
        "section_question": "问题",
        "section_conclusion": "结论",
        "section_narrative": "关键文字证据",
        "section_facts": "关键财务事实",
        "section_filings": "涉及披露",
        "empty_none": "暂无",
        "default_scenario": "上市公司披露问答",
        "quote_fallback": "摘录",
        "ascii_banner": "财务问答结果报告",
        "overview": "报告概览",
        "final_answer": "最终结论",
        "quality_snapshot": "结果质量快照",
        "evidence_overview": "证据概览",
        "key_narrative": "关键文字证据",
        "key_facts": "关键财务事实",
        "attention": "需关注事项",
        "notes": "说明",
        "col_item": "项目",
        "col_result": "结果",
        "col_metric": "指标",
        "col_evidence_item": "证据项",
        "col_count": "数量",
        "col_seq": "序号",
        "col_filing_id": "披露编号",
        "col_card": "证据卡片",
        "col_excerpt": "摘录",
        "col_value": "数值",
        "col_disclosure": "披露",
        "report_type": "报告类型",
        "confidence_label": "结论可信度",
        "citation_count": "证据引用数",
        "filing_count": "涉及披露数",
        "evidence_outline_filings": "涉及披露",
        "structured_count": "结构化事实数",
        "gen_latency": "生成耗时",
        "evidence_narrative": "文字证据",
        "evidence_structured": "财务事实",
        "risk_hints": "风险提示",
        "confidence_high": "高",
        "confidence_mid": "中",
        "confidence_low": "低",
    },
    "en": {
        "html_lang": "en",
        "doc_title": "Finance Q&A Report",
        "section_question": "Question",
        "section_conclusion": "Conclusion",
        "section_narrative": "Key textual evidence",
        "section_facts": "Key financial facts",
        "section_filings": "Filings referenced",
        "empty_none": "None",
        "default_scenario": "Listed company disclosure Q&A",
        "quote_fallback": "Excerpt",
        "ascii_banner": "FINANCE QA RESULT REPORT",
        "overview": "Overview",
        "final_answer": "Answer",
        "quality_snapshot": "Quality snapshot",
        "evidence_overview": "Evidence overview",
        "key_narrative": "Key textual evidence",
        "key_facts": "Key financial facts",
        "attention": "Attention",
        "notes": "Notes",
        "col_item": "Item",
        "col_result": "Value",
        "col_metric": "Metric",
        "col_evidence_item": "Evidence",
        "col_count": "Count",
        "col_seq": "#",
        "col_filing_id": "Filing",
        "col_card": "Card",
        "col_excerpt": "Excerpt",
        "col_value": "Value",
        "col_disclosure": "Filing / context",
        "report_type": "Report type",
        "confidence_label": "Confidence",
        "citation_count": "Citations",
        "filing_count": "Filings",
        "evidence_outline_filings": "Filings",
        "structured_count": "Structured facts",
        "gen_latency": "Latency",
        "evidence_narrative": "Text evidence",
        "evidence_structured": "Financial facts",
        "risk_hints": "Risk flags",
        "confidence_high": "High",
        "confidence_mid": "Medium",
        "confidence_low": "Low",
    },
}


def report_ui_strings(locale: Locale) -> dict[str, str]:
    return dict(REPORT_UI.get(locale) or REPORT_UI["zh"])
