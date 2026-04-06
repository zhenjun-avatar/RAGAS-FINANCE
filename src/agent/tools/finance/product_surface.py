"""Product-facing finance QA surface.

Keeps product configuration separate from retrieval orchestration:
1. fixed vertical scenarios
2. external evaluation metric catalog
3. minimal evidence UI payload builders
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.finance.report_locale import (
    evidence_ui_copy_principles,
    evidence_ui_panels,
    narrative_evidence_title,
    risk_flag_message,
    snapshot_metric_label,
)


@dataclass(frozen=True)
class VerticalScenarioDefinition:
    scenario_id: str
    name: str
    name_en: str
    audience: tuple[str, ...]
    value_proposition: str
    typical_questions: tuple[str, ...]
    evidence_expectations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "name_en": self.name_en,
            "audience": list(self.audience),
            "value_proposition": self.value_proposition,
            "typical_questions": list(self.typical_questions),
            "evidence_expectations": list(self.evidence_expectations),
        }


@dataclass(frozen=True)
class EvaluationMetricDefinition:
    metric_id: str
    label: str
    category: str
    description: str
    target_direction: str
    unit: str
    aggregation_hint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "label": self.label,
            "category": self.category,
            "description": self.description,
            "target_direction": self.target_direction,
            "unit": self.unit,
            "aggregation_hint": self.aggregation_hint,
        }


FINANCE_VERTICAL_SCENARIOS: tuple[VerticalScenarioDefinition, ...] = (
    VerticalScenarioDefinition(
        scenario_id="sec_disclosure_qa",
        name="上市公司披露问答",
        name_en="Listed company disclosure Q&A",
        audience=("投研", "IR", "财务分析", "法务支持"),
        value_proposition="结论可引用、数字可对账、跨 filing 可追溯。",
        typical_questions=(
            "公司本期流动性风险怎么表述？",
            "管理层如何解释收入变化？",
            "多个 filing 对同一风险披露是否一致？",
        ),
        evidence_expectations=("段落引用", "结构化事实", "跨 filing 分布"),
    ),
    VerticalScenarioDefinition(
        scenario_id="compliance_draft_assist",
        name="合规问询草稿辅助",
        name_en="Compliance inquiry & draft assist",
        audience=("法务", "合规", "董办"),
        value_proposition="输出可审计的答复草稿，降低问询整理成本。",
        typical_questions=(
            "请汇总最近两期对持续经营风险的披露依据。",
            "生成带证据的问询回复草稿提纲。",
            "列出支持该结论的 filing 段落与数值。",
        ),
        evidence_expectations=("结论草稿", "证据卡片", "风险提示"),
    ),
)


FINANCE_EXTERNAL_METRICS: tuple[EvaluationMetricDefinition, ...] = (
    EvaluationMetricDefinition(
        metric_id="fact_consistency_rate",
        label="事实一致性",
        category="correctness",
        description="答案中的关键结论是否被证据支持，且不与结构化数值冲突。",
        target_direction="higher_better",
        unit="ratio",
        aggregation_hint="按 golden set 统计命中率",
    ),
    EvaluationMetricDefinition(
        metric_id="citation_coverage_rate",
        label="引用覆盖率",
        category="correctness",
        description="关键回答点中有明确引用或结构化事实支撑的比例。",
        target_direction="higher_better",
        unit="ratio",
        aggregation_hint="按问题或 claim 聚合",
    ),
    EvaluationMetricDefinition(
        metric_id="abstain_safety_rate",
        label="谨慎拒答率",
        category="correctness",
        description="证据不足时是否明确提示不确定，而不是编造结论。",
        target_direction="higher_better",
        unit="ratio",
        aggregation_hint="按低证据问题集合评测",
    ),
    EvaluationMetricDefinition(
        metric_id="evidence_bundle_completeness",
        label="证据包完整度",
        category="accountability",
        description="是否同时提供结论、可引用段落、结构化事实或缺口提示。",
        target_direction="higher_better",
        unit="score_0_1",
        aggregation_hint="可做单问快照，也可按批次求均值",
    ),
    EvaluationMetricDefinition(
        metric_id="cross_filing_balance",
        label="跨 filing 平衡度",
        category="accountability",
        description="跨期问题下，证据是否来自多个相关 filing，而非锁死单 accession。",
        target_direction="higher_better",
        unit="score_0_1",
        aggregation_hint="仅对 compare / multi-filing 问题统计",
    ),
    EvaluationMetricDefinition(
        metric_id="trace_availability_rate",
        label="Trace 可用率",
        category="accountability",
        description="是否能回放本次答案的路由、检索、融合与证据分布。",
        target_direction="higher_better",
        unit="ratio",
        aggregation_hint="按请求级别统计",
    ),
    EvaluationMetricDefinition(
        metric_id="p95_latency_ms",
        label="P95 延迟",
        category="efficiency",
        description="用户从提问到收到完整答案的尾延迟。",
        target_direction="lower_better",
        unit="ms",
        aggregation_hint="按时间窗口聚合 p95",
    ),
    EvaluationMetricDefinition(
        metric_id="avg_tokens_per_query",
        label="单问平均 Token",
        category="efficiency",
        description="单个问题端到端消耗的平均 token 成本。",
        target_direction="lower_better",
        unit="tokens",
        aggregation_hint="按问题批量求均值",
    ),
)


FINANCE_EVIDENCE_UI_NARRATIVE: dict[str, Any] = {
    "layout": "single_screen_three_panels",
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
}


def _question_text(question: str) -> str:
    return str(question or "").strip().lower()


def _stringify(value: Any) -> str:
    return str(value or "").strip()


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_label(row: dict[str, Any]) -> str:
    label = _stringify(row.get("metric_label"))
    if label:
        return label
    key = _stringify(row.get("metric_key"))
    if "." in key:
        key = key.split(".")[-1].strip()
    return key or "metric"


def _format_fact_value(row: dict[str, Any]) -> str:
    numeric = _to_float(row.get("value_numeric"))
    if numeric is not None:
        if abs(numeric) >= 1_000_000_000:
            return f"{numeric:,.0f}"
        if abs(numeric) >= 1_000:
            return f"{numeric:,.2f}"
        return f"{numeric:.4f}".rstrip("0").rstrip(".")
    text = _stringify(row.get("value_text"))
    return text[:160] if text else "n/a"


def _fact_subtitle(row: dict[str, Any]) -> str:
    parts = [
        _stringify(row.get("form")),
        _stringify(row.get("period_end")),
        _stringify(row.get("accn")),
    ]
    return " | ".join(part for part in parts if part)


def _preview_text(value: str, *, limit: int) -> str:
    text = _stringify(value)
    if len(text) <= max(1, int(limit)):
        return text
    return text[: max(1, int(limit)) - 3].rstrip() + "..."


def _is_boilerplate_narrative_quote(value: str) -> bool:
    text = _stringify(value).lower()
    if not text:
        return False
    signals = (
        "table of contents",
        "references to the",
        "forward-looking statements",
        "item 2. management",
        "blank check company",
        "formed for the purpose of effecting a merger",
    )
    if any(sig in text for sig in signals):
        return True
    # Structured XBRL-style lines are not narrative evidence.
    return text.startswith("us-gaap.") or text.startswith("dei.")


def _append_unique_filings(filings: list[str], values: list[str], *, cap: int) -> None:
    for raw in values:
        if len(filings) >= cap:
            return
        v = _stringify(raw)
        if v and v not in filings:
            filings.append(v)


def _citation_accessions(citations: list[dict[str, Any]] | None) -> list[str]:
    out: list[str] = []
    if not citations:
        return out
    for item in citations:
        accn = _stringify(item.get("accn") or item.get("sec_accession"))
        if accn and accn not in out:
            out.append(accn)
    return out


def _extract_filings(
    sql_rows: list[dict[str, Any]],
    pipeline_trace: dict[str, Any] | None,
    *,
    citations: list[dict[str, Any]] | None = None,
    cap: int = 6,
) -> list[str]:
    """Prefer accessions from narrative citations (matches model context), then SQL, then controller fallbacks."""
    filings: list[str] = []
    _append_unique_filings(filings, _citation_accessions(citations), cap=cap)
    for row in sql_rows:
        if len(filings) >= cap:
            break
        accn = _stringify(row.get("accn"))
        if accn and accn not in filings:
            filings.append(accn)
    trace = pipeline_trace or {}
    controller = trace.get("evidence_controller") or {}
    _append_unique_filings(
        filings,
        [_stringify(a) for a in (controller.get("target_accns") or [])],
        cap=cap,
    )
    return filings


def _risk_flags(
    *,
    citations: list[dict[str, Any]],
    sql_rows: list[dict[str, Any]],
    limitations: str | None,
    pipeline_trace: dict[str, Any] | None,
    locale: str = "zh",
) -> list[str]:
    loc = "en" if str(locale).lower() == "en" else "zh"
    flags: list[str] = []
    if not citations:
        flags.append(risk_flag_message("no_citations", loc))
    if not sql_rows:
        flags.append(risk_flag_message("no_sql", loc))
    if limitations:
        flags.append(risk_flag_message("limitations", loc))
    reasons = ((pipeline_trace or {}).get("second_pass") or {}).get("reasons") or []
    if "filing_divergence" in reasons:
        flags.append(risk_flag_message("filing_divergence", loc))
    return flags


def select_vertical_scenario(question: str, evidence_plan: Any | None = None, *, locale: str = "zh") -> dict[str, Any]:
    question_text = _question_text(question)
    question_mode = _stringify(getattr(evidence_plan, "question_mode", "")).lower()
    compliance_tokens = ("问询", "回复", "合规", "draft", "comment letter", "监管", "草稿")
    if any(token in question_text for token in compliance_tokens):
        selected = FINANCE_VERTICAL_SCENARIOS[1]
    else:
        selected = FINANCE_VERTICAL_SCENARIOS[0]
    base = selected.to_dict()
    loc = "en" if str(locale).lower() == "en" else "zh"
    if loc == "en":
        base["name"] = base.get("name_en") or base["name"]
    return {
        **base,
        "report_locale": loc,
        "question_mode": question_mode or None,
    }


def build_external_evaluation_snapshot(
    *,
    citations: list[dict[str, Any]],
    sql_rows: list[dict[str, Any]],
    question_mode: str | None,
    pipeline_trace: dict[str, Any] | None,
    trace_id: str | None,
    latency_ms: float | None,
    token_usage: dict[str, Any] | None,
    locale: str = "zh",
) -> dict[str, Any]:
    filings = _extract_filings(sql_rows, pipeline_trace, citations=citations)
    needs_cross_filing = _stringify(question_mode) == "cross_filing_compare"
    evidence_parts = [
        1.0 if citations else 0.0,
        1.0 if sql_rows else 0.0,
        1.0 if (trace_id or pipeline_trace) else 0.0,
    ]
    completeness = round(sum(evidence_parts) / len(evidence_parts), 4)
    cross_filing = 1.0 if not needs_cross_filing else min(1.0, len(filings) / 2.0)
    prompt_tokens = int((token_usage or {}).get("prompt_tokens") or 0)
    completion_tokens = int((token_usage or {}).get("completion_tokens") or 0)
    total_tokens = int((token_usage or {}).get("total_tokens") or (prompt_tokens + completion_tokens))
    loc = "en" if str(locale).lower() == "en" else "zh"
    return {
        "metric_catalog_version": "finance-external-v1",
        "observable_metrics": [
            {
                "metric_id": "evidence_bundle_completeness",
                "label": snapshot_metric_label("evidence_bundle_completeness", loc),
                "value": completeness,
                "unit": "score_0_1",
            },
            {
                "metric_id": "cross_filing_balance",
                "label": snapshot_metric_label("cross_filing_balance", loc),
                "value": round(cross_filing, 4),
                "unit": "score_0_1",
            },
            {
                "metric_id": "trace_availability_rate",
                "label": snapshot_metric_label("trace_availability_rate", loc),
                "value": 1.0 if (trace_id or pipeline_trace) else 0.0,
                "unit": "ratio",
            },
            {
                "metric_id": "latency_ms",
                "label": snapshot_metric_label("latency_ms", loc),
                "value": round(float(latency_ms or 0.0), 2),
                "unit": "ms",
            },
            {
                "metric_id": "tokens_per_query",
                "label": snapshot_metric_label("tokens_per_query", loc),
                "value": total_tokens,
                "unit": "tokens",
            },
        ],
        "offline_metrics_pending": [
            "fact_consistency_rate",
            "citation_coverage_rate",
            "abstain_safety_rate",
            "p95_latency_ms",
            "avg_tokens_per_query",
        ],
        "filings_observed": filings,
    }


def build_evidence_ui_bundle(
    *,
    question: str,
    answer: str,
    confidence: float,
    citations: list[dict[str, Any]],
    sql_rows: list[dict[str, Any]],
    limitations: str | None,
    trace_id: str | None,
    pipeline_trace: dict[str, Any] | None,
    vertical_scenario: dict[str, Any],
    locale: str = "zh",
) -> dict[str, Any]:
    loc = "en" if str(locale).lower() == "en" else "zh"
    filings = _extract_filings(sql_rows, pipeline_trace, citations=citations)
    filtered_citations = [item for item in citations if not _is_boilerplate_narrative_quote(item.get("quote") or "")]
    narrative_cards = []
    for idx, item in enumerate(filtered_citations[:3], start=1):
        accn = _stringify(item.get("accn") or item.get("sec_accession"))
        card: dict[str, Any] = {
            "card_id": f"quote-{idx}",
            "kind": "narrative_quote",
            "title": narrative_evidence_title(idx, loc),
            "body": _stringify(item.get("quote"))[:280],
            "document_id": item.get("document_id"),
            "node_id": item.get("node_id"),
            "relevance_score": round(float(item.get("relevance_score") or 0.0), 4),
            "relevance_level": item.get("relevance_level"),
        }
        if accn:
            card["accn"] = accn
        narrative_cards.append(card)
    structured_cards = [
        {
            "card_id": f"fact-{idx}",
            "kind": "structured_fact",
            "title": _metric_label(row),
            "body": _format_fact_value(row),
            "subtitle": _fact_subtitle(row),
            "metric_key": row.get("metric_key"),
            "taxonomy": row.get("taxonomy"),
        }
        for idx, row in enumerate(sql_rows[:4], start=1)
    ]
    flags = _risk_flags(
        citations=citations,
        sql_rows=sql_rows,
        limitations=limitations,
        pipeline_trace=pipeline_trace,
        locale=loc,
    )
    return {
        "layout": FINANCE_EVIDENCE_UI_NARRATIVE["layout"],
        "conclusion": {
            "question_preview": _preview_text(question, limit=140),
            "answer_preview": _preview_text(answer, limit=320),
            "answer_chars": len(_stringify(answer)),
            "confidence": round(float(confidence or 0.0), 4),
            "scenario_name": vertical_scenario.get("name"),
            "trace_id": trace_id,
        },
        "evidence": {
            "summary": {
                "narrative_card_count": len(narrative_cards),
                "structured_fact_count": len(structured_cards),
                "filing_count": len(filings),
            },
            "narrative_cards": narrative_cards,
            "structured_fact_cards": structured_cards,
            "filings": filings,
        },
        "risk_panel": {
            "limitations": limitations,
            "flags": flags,
        },
        "panels": list(evidence_ui_panels(loc)),
        "copy_principles": list(evidence_ui_copy_principles(loc)),
    }


def get_finance_product_spec() -> dict[str, Any]:
    return {
        "version": "finance-product-surface-v1",
        "vertical_scenarios": [item.to_dict() for item in FINANCE_VERTICAL_SCENARIOS],
        "external_evaluation_metrics": [item.to_dict() for item in FINANCE_EXTERNAL_METRICS],
        "evidence_ui_narrative": {
            "layout": FINANCE_EVIDENCE_UI_NARRATIVE["layout"],
            "panels": list(FINANCE_EVIDENCE_UI_NARRATIVE["panels"]),
            "copy_principles": list(FINANCE_EVIDENCE_UI_NARRATIVE["copy_principles"]),
        },
    }
