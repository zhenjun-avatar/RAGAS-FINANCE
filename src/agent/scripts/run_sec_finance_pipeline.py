#!/usr/bin/env python3
"""
SEC companyfacts（财务）入库 + 可选问答 / 查 SQL 观测。

在 src/agent 目录执行（需 .env、Postgres、Qdrant；OpenSearch 若 SPARSE_BACKEND=opensearch）::

    cd src/agent
    ..\\venv\\Scripts\\python.exe scripts\\run_sec_finance_pipeline.py ingest-direct
    ..\\venv\\Scripts\\python.exe scripts\\run_sec_finance_pipeline.py ingest-direct --document-id 9002
    ..\\venv\\Scripts\\python.exe scripts\\run_sec_finance_pipeline.py sql-sample --document-id 9002
    ..\\venv\\Scripts\\python.exe scripts\\run_sec_finance_pipeline.py ask-direct --question "10-K 相关事实有哪些？"

    # 从 companyfacts JSON 列出 accession，并按 accession 拉取 EDGAR 主 HTML 入库（每份一个 document_id）
    ..\\venv\\Scripts\\python.exe scripts\\run_sec_finance_pipeline.py list-accessions
    ..\\venv\\Scripts\\python.exe scripts\\run_sec_finance_pipeline.py ingest-edgar --document-id-start 9100 --max-filings 5
    # 本地已有 tools\\data\\EDGAR_*.htm：默认与 tools\\data\\CIK0001823776.json 按 accession 对齐 form/filed/公司名
    ..\\venv\\Scripts\\python.exe scripts\\run_sec_finance_pipeline.py ingest-edgar-local --document-id-start 9200
    ..\\venv\\Scripts\\python.exe scripts\\run_sec_finance_pipeline.py ingest-edgar-local --document-id-start 9200 --companyfacts-json tools\\data\\CIK0001823776.json

    # 问答时传入多个 document_id：facts + 各期 EDGAR
    ..\\venv\\Scripts\\python.exe scripts\\run_sec_finance_pipeline.py ask-multi --document-ids 9002,9100,9101 --question "..."
    # 或使用分组（见 tools/data/document_groups.json 示例格式）
    ..\\venv\\Scripts\\python.exe scripts\\run_sec_finance_pipeline.py ask-multi --group A --question "..."

HTTP 子命令需先启动 API::

    ..\\venv\\Scripts\\python.exe -m uvicorn api.server:app --host 0.0.0.0 --port 8000

    ..\\venv\\Scripts\\python.exe scripts\\run_sec_finance_pipeline.py ingest-http
    ..\\venv\\Scripts\\python.exe scripts\\run_sec_finance_pipeline.py ask-http --pipeline-trace
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import re
import sys
from pathlib import Path

import httpx

AGENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = AGENT_ROOT / "tools" / "data" / "CIK0001823776.json"
DEFAULT_DOC_ID = 9002
DEFAULT_EDGAR_DOC_START = 9100
DEFAULT_DATA_DIR = AGENT_ROOT / "tools" / "data"
DEFAULT_LOG_DIR = AGENT_ROOT / "logs"
_EDGAR_LOCAL_STEM = re.compile(r"^EDGAR_(\d+)_(.+)$", re.IGNORECASE)


def _configure_stdout_utf8() -> None:
    """Keep Windows console/pipeline output consistently UTF-8."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _safe_print_json(payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    try:
        print(text)
    except UnicodeEncodeError:
        # Last-resort fallback for unusual streams that reject reconfigure.
        print(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


def _preview(value: object, *, limit: int = 48) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max(1, int(limit)):
        return text
    return text[: max(1, int(limit)) - 3].rstrip() + "..."


def _payload_report_locale(payload: dict[str, object]) -> str:
    """Match backend: explicit zh/en/auto or infer from question."""
    from tools.finance.report_locale import resolve_report_locale

    q = str(payload.get("question") or "")
    raw = payload.get("report_locale")
    explicit = raw.strip() if isinstance(raw, str) else None
    return resolve_report_locale(q, explicit)


def _confidence_label(value: object, u: dict[str, str]) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "-"
    if score >= 0.85:
        return u["confidence_high"]
    if score >= 0.65:
        return u["confidence_mid"]
    return u["confidence_low"]


def _friendly_metric_value(value: object, unit: object) -> str:
    unit_text = str(unit or "").strip()
    if unit_text == "score_0_1":
        return f"{float(value):.2f} / 1.00" if isinstance(value, (int, float)) else f"{value} / 1.00"
    if unit_text == "ratio":
        return f"{float(value) * 100:.0f}%" if isinstance(value, (int, float)) else str(value)
    if unit_text == "ms":
        return f"{value} ms"
    if unit_text == "tokens":
        return f"{value} tokens"
    return str(value)


def _ascii_table(headers: list[str], rows: list[list[object]]) -> str:
    normalized = [[_preview(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in normalized:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_row = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    lines = [border, header_row, border]
    for row in normalized:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    lines.append(border)
    return "\n".join(lines)


def _report_sections(payload: dict[str, object]) -> dict[str, object]:
    from tools.finance.report_locale import report_ui_strings

    loc = _payload_report_locale(payload)
    U = report_ui_strings(loc)
    vertical = payload.get("vertical_scenario") if isinstance(payload.get("vertical_scenario"), dict) else {}
    external = payload.get("external_evaluation") if isinstance(payload.get("external_evaluation"), dict) else {}
    evidence_ui = payload.get("evidence_ui") if isinstance(payload.get("evidence_ui"), dict) else {}
    conclusion = evidence_ui.get("conclusion") if isinstance(evidence_ui.get("conclusion"), dict) else {}
    evidence = evidence_ui.get("evidence") if isinstance(evidence_ui.get("evidence"), dict) else {}
    evidence_summary = evidence.get("summary") if isinstance(evidence.get("summary"), dict) else {}
    risk_panel = evidence_ui.get("risk_panel") if isinstance(evidence_ui.get("risk_panel"), dict) else {}
    scenario_name = str(vertical.get("name") or U["default_scenario"])
    narrative_cards = [item for item in (evidence.get("narrative_cards") or []) if isinstance(item, dict)]
    structured_fact_cards = [item for item in (evidence.get("structured_fact_cards") or []) if isinstance(item, dict)]
    filings_observed = [str(item).strip() for item in (external.get("filings_observed") or []) if str(item).strip()]
    filing_count = int(evidence_summary.get("filing_count") or len(filings_observed or (evidence.get("filings") or [])) or 0)
    narrative_count = int(evidence_summary.get("narrative_card_count") or len(narrative_cards) or 0)
    structured_count = int(evidence_summary.get("structured_fact_count") or len(structured_fact_cards) or 0)
    risk_flags = [str(flag).strip() for flag in (risk_panel.get("flags") or []) if str(flag).strip()]
    question_preview = conclusion.get("question_preview") or _preview(payload.get("question"), limit=140)
    answer_preview = conclusion.get("answer_preview") or _preview(payload.get("answer"), limit=1200)
    summary_rows = [
        [U["report_type"], scenario_name],
        [U["confidence_label"], _confidence_label(payload.get("confidence"), U)],
        [U["citation_count"], payload.get("citation_count") or 0],
        [U["filing_count"], filing_count],
        [U["structured_count"], structured_count],
        [U["gen_latency"], f"{payload.get('latency_ms') or '-'} ms"],
    ]
    skip_metric_ids = {"trace_availability_rate", "latency_ms", "tokens_per_query"}
    metric_rows: list[list[object]] = []
    for item in external.get("observable_metrics") or []:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("metric_id") or "").strip()
        if mid in skip_metric_ids:
            continue
        label = str(item.get("label") or item.get("metric_id") or "-")
        metric_rows.append([label, _friendly_metric_value(item.get("value") or 0, item.get("unit"))])
    evidence_rows = [
        [U["evidence_narrative"], narrative_count],
        [U["evidence_structured"], structured_count],
        [U["evidence_outline_filings"], filing_count],
        [U["risk_hints"], len(risk_flags)],
    ]
    filing_rows = [[idx, filing] for idx, filing in enumerate(filings_observed[:8], start=1)]
    quote_rows = [[item.get("title") or "-", item.get("body") or "-"] for item in narrative_cards[:5]]
    fact_rows = [
        [item.get("title") or "-", item.get("body") or "-", item.get("subtitle") or "-"]
        for item in structured_fact_cards[:6]
    ]
    limitations = str(risk_panel.get("limitations") or "").strip()
    return {
        "locale": loc,
        "ui": U,
        "scenario_name": scenario_name,
        "summary_rows": summary_rows,
        "question_preview": question_preview,
        "answer_preview": answer_preview,
        "metric_rows": metric_rows,
        "evidence_rows": evidence_rows,
        "filing_rows": filing_rows,
        "quote_rows": quote_rows,
        "fact_rows": fact_rows,
        "risk_flags": risk_flags,
        "limitations": limitations,
    }


def _format_product_summary_text(payload: dict[str, object]) -> str:
    data = _report_sections(payload)
    U = data["ui"]  # type: ignore[assignment]

    parts: list[str] = []
    parts.append(str(U["ascii_banner"]))
    parts.append("=" * 72)
    parts.append(str(data["scenario_name"]))
    parts.append("")

    parts.append(U["overview"])
    parts.append(_ascii_table([U["col_item"], U["col_result"]], data["summary_rows"]))

    parts.append("")
    parts.append(U["section_question"])
    parts.append(str(data["question_preview"]))
    parts.append("")
    parts.append(U["final_answer"])
    parts.append(str(data["answer_preview"]))

    if data["metric_rows"]:
        parts.append("")
        parts.append(U["quality_snapshot"])
        parts.append(_ascii_table([U["col_metric"], U["col_result"]], data["metric_rows"]))

    parts.append("")
    parts.append(U["evidence_overview"])
    parts.append(_ascii_table([U["col_evidence_item"], U["col_count"]], data["evidence_rows"]))

    if data["filing_rows"]:
        parts.append("")
        parts.append(U["section_filings"])
        parts.append(_ascii_table([U["col_seq"], U["col_filing_id"]], data["filing_rows"]))

    if data["quote_rows"]:
        parts.append("")
        parts.append(U["key_narrative"])
        parts.append(_ascii_table([U["col_card"], U["col_excerpt"]], data["quote_rows"]))

    if data["fact_rows"]:
        parts.append("")
        parts.append(U["key_facts"])
        parts.append(_ascii_table([U["col_metric"], U["col_value"], U["col_disclosure"]], data["fact_rows"]))

    if data["risk_flags"]:
        parts.append("")
        parts.append(U["attention"])
        for flag in data["risk_flags"]:
            parts.append(f"- {flag}")

    if data["limitations"]:
        parts.append("")
        parts.append(U["notes"])
        parts.append(str(data["limitations"]))

    parts.append("")
    return "\n".join(parts)


def _html_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


_SPLIT_KEY_POINTS_HEADING = re.compile(
    r"(?s)\s*\*\*关键要点(?:与引用依据|与引用)?\*\*\s*[:：]?\s*",
)
_SPLIT_KEY_POINTS_HEADING_EN = re.compile(
    r"(?is)\s*\*\*Key points(?:\s+and\s+citations)?\*\*\s*:?\s*",
)


def _clean_conclusion_text_for_html(raw: str, *, locale: str = "zh") -> str:
    """Strip product-unfriendly markdown: 直接回答 / Direct answer, 关键要点 / Key points, **bold**."""
    text = str(raw or "").replace("\r\n", "\n").strip()
    if not text:
        return ""
    if str(locale).lower() == "en":
        text = re.sub(r"(?i)^\s*\*\*Direct answer\*\*\s*:?\s*", "", text, count=1)
        parts = _SPLIT_KEY_POINTS_HEADING_EN.split(text, maxsplit=1)
    else:
        text = re.sub(r"^\s*\*\*直接回答\*\*\s*[:：]?\s*", "", text, count=1)
        parts = _SPLIT_KEY_POINTS_HEADING.split(text, maxsplit=1)
    text = parts[0].strip()
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _format_product_summary_html(payload: dict[str, object]) -> str:
    from tools.finance.report_locale import report_ui_strings

    loc = _payload_report_locale(payload)
    U = report_ui_strings(loc)
    vertical = payload.get("vertical_scenario") if isinstance(payload.get("vertical_scenario"), dict) else {}
    external = payload.get("external_evaluation") if isinstance(payload.get("external_evaluation"), dict) else {}
    evidence_ui = payload.get("evidence_ui") if isinstance(payload.get("evidence_ui"), dict) else {}
    conclusion = evidence_ui.get("conclusion") if isinstance(evidence_ui.get("conclusion"), dict) else {}
    evidence = evidence_ui.get("evidence") if isinstance(evidence_ui.get("evidence"), dict) else {}
    scenario_name = str(vertical.get("name") or U["default_scenario"])
    narrative_cards = [item for item in (evidence.get("narrative_cards") or []) if isinstance(item, dict)]
    structured_fact_cards = [item for item in (evidence.get("structured_fact_cards") or []) if isinstance(item, dict)]
    filings_observed = [str(item).strip() for item in (external.get("filings_observed") or []) if str(item).strip()]
    filing_rows = [[idx, filing] for idx, filing in enumerate(filings_observed[:8], start=1)]

    q_full = str(payload.get("question") or "").strip()
    if not q_full:
        q_full = str(conclusion.get("question_preview") or "").strip()

    answer_raw = str(payload.get("answer") or "").strip()
    if not answer_raw:
        answer_raw = str(conclusion.get("answer_preview") or "").strip()
    conclusion_clean = _clean_conclusion_text_for_html(answer_raw, locale=loc)

    sections: list[str] = []
    sections.append(
        "<header class='doc-header'>"
        f"<h1>{html.escape(U['doc_title'])}</h1>"
        f"<p class='subtitle'>{html.escape(scenario_name)}</p>"
        "</header>"
    )
    sections.append(
        f"<section><h2>{html.escape(U['section_question'])}</h2><div class='text-block'>"
        + html.escape(q_full)
        + "</div></section>"
    )
    sections.append(
        f"<section><h2>{html.escape(U['section_conclusion'])}</h2><div class='text-block answer'>"
        + html.escape(conclusion_clean)
        + "</div></section>"
    )

    narr_parts: list[str] = [f"<section><h2>{html.escape(U['section_narrative'])}</h2><div class='evidence-list'>"]
    if narrative_cards:
        for card in narrative_cards[:8]:
            title = html.escape(str(card.get("title") or U["quote_fallback"]))
            body_esc = html.escape(str(card.get("body") or ""))
            narr_parts.append(
                "<div class='evidence-item'>"
                f"<div class='evidence-item-title'>{title}</div>"
                f"<div class='evidence-item-body text-block'>{body_esc}</div>"
                "</div>"
            )
    else:
        narr_parts.append(f"<p class='muted'>{html.escape(U['empty_none'])}</p>")
    narr_parts.append("</div></section>")
    sections.append("".join(narr_parts))

    if structured_fact_cards:
        fact_rows = [
            [item.get("title") or "-", item.get("body") or "-", item.get("subtitle") or "-"]
            for item in structured_fact_cards[:8]
        ]
        sections.append(
            f"<section><h2>{html.escape(U['section_facts'])}</h2>"
            + _html_table([U["col_metric"], U["col_value"], U["col_disclosure"]], fact_rows)
            + "</section>"
        )
    else:
        sections.append(
            f"<section><h2>{html.escape(U['section_facts'])}</h2><p class='muted'>{html.escape(U['empty_none'])}</p></section>"
        )

    if filing_rows:
        sections.append(
            f"<section><h2>{html.escape(U['section_filings'])}</h2>"
            + _html_table([U["col_seq"], U["col_filing_id"]], filing_rows)
            + "</section>"
        )
    else:
        sections.append(
            f"<section><h2>{html.escape(U['section_filings'])}</h2><p class='muted'>{html.escape(U['empty_none'])}</p></section>"
        )

    # Parentheses required: otherwise `return """..."""` ends the statement and
    # `+ "".join(sections)` becomes a separate no-op expression (truncated HTML).
    return (
        "<!DOCTYPE html>\n<html lang=\""
        + html.escape(U["html_lang"])
        + "\">\n<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        "  <title>"
        + html.escape(U["doc_title"])
        + "</title>\n"
        "  <style>\n"
        """    :root {
      color-scheme: light;
      --bg: #f4f4f5;
      --surface: #ffffff;
      --text: #18181b;
      --muted: #71717a;
      --line: #e4e4e7;
      --head: #f4f4f5;
    }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.65;
      font-size: 15px;
    }
    .page {
      max-width: 820px;
      margin: 0 auto;
      padding: 40px 28px 56px;
    }
    .doc-header {
      margin-bottom: 36px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 1.375rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      color: var(--text);
    }
    h2 {
      margin: 0 0 14px;
      font-size: 0.8125rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 0.875rem;
      font-weight: 400;
    }
    section {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 2px;
      padding: 22px 24px;
      margin-top: 16px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 4px;
      font-size: 0.875rem;
    }
    th, td {
      border: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: var(--head);
      font-weight: 600;
      color: var(--text);
    }
    .text-block {
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--text);
    }
    .answer {
      font-size: 0.9375rem;
    }
    ul {
      margin: 0;
      padding-left: 20px;
    }
    .muted {
      color: var(--muted);
      margin: 0;
      font-size: 0.875rem;
    }
    .evidence-list {
      display: flex;
      flex-direction: column;
      gap: 0;
    }
    .evidence-item {
      padding: 16px 0;
      border-top: 1px solid var(--line);
    }
    .evidence-item:first-child {
      border-top: none;
      padding-top: 0;
    }
    .evidence-item-title {
      font-weight: 600;
      font-size: 0.875rem;
      color: var(--text);
      margin-bottom: 8px;
    }
    .evidence-item-body {
      font-size: 0.875rem;
      line-height: 1.7;
      color: #3f3f46;
    }
  </style>
</head>
<body>
  <div class="page">
"""
        + "".join(sections)
        + """
  </div>
</body>
</html>
"""
    )


def _write_product_summary_log(payload: dict[str, object]) -> str | None:
    if not any(payload.get(key) for key in ("vertical_scenario", "external_evaluation", "evidence_ui")):
        return None
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    trace_id = str(payload.get("trace_id") or "").strip()
    stem = f"ask_product_summary_{trace_id}" if trace_id else "ask_product_summary"
    path = DEFAULT_LOG_DIR / f"{stem}.log"
    path.write_text(_format_product_summary_text(payload), encoding="utf-8")
    try:
        return str(path.relative_to(AGENT_ROOT))
    except ValueError:
        return str(path)


def _write_product_summary_html(payload: dict[str, object]) -> str | None:
    if not any(payload.get(key) for key in ("vertical_scenario", "external_evaluation", "evidence_ui")):
        return None
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    trace_id = str(payload.get("trace_id") or "").strip()
    stem = f"ask_product_summary_{trace_id}" if trace_id else "ask_product_summary"
    path = DEFAULT_LOG_DIR / f"{stem}.html"
    path.write_text(_format_product_summary_html(payload), encoding="utf-8")
    try:
        return str(path.relative_to(AGENT_ROOT))
    except ValueError:
        return str(path)


def _build_ask_log_payload(payload: dict[str, object], *, include_pipeline_trace: bool) -> dict[str, object]:
    out: dict[str, object] = {
        key: payload[key]
        for key in (
            "question",
            "answer",
            "confidence",
            "sources_used",
            "citation_count",
            "limitations",
            "trace_id",
            "latency_ms",
            "report_locale",
        )
        if key in payload
    }
    if include_pipeline_trace and payload.get("pipeline_trace") is not None:
        out["pipeline_trace"] = payload["pipeline_trace"]
    summary_log = _write_product_summary_log(payload)
    if summary_log:
        out["product_summary_log"] = summary_log
    summary_html = _write_product_summary_html(payload)
    if summary_html:
        out["product_summary_html"] = summary_html
    return out


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(AGENT_ROOT / ".env")


def _ensure_path(p: Path) -> Path:
    if not p.is_file():
        raise SystemExit(f"找不到文件: {p}")
    return p.resolve()


def _api_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, trust_env=False)


async def cmd_ingest_direct(document_id: int, json_path: Path) -> None:
    _load_env()
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))
    path = _ensure_path(json_path)
    from tools.ingestion_service import process_document

    out = await process_document(
        file_path=str(path),
        file_type="sec_companyfacts",
        document_id=document_id,
    )
    _safe_print_json(out)


async def cmd_ingest_http(base_url: str, document_id: int, json_path: Path) -> None:
    path = _ensure_path(json_path)
    url = base_url.rstrip("/") + "/agent/api/documents/process"
    body = {
        "document_id": document_id,
        "file_path": str(path),
        "file_type": "sec_companyfacts",
    }
    async with _api_client(timeout=600.0) as client:
        r = await client.post(url, json=body)
    r.raise_for_status()
    _safe_print_json(r.json())


async def cmd_sql_sample(document_id: int, limit: int) -> None:
    _load_env()
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))
    from tools.node_repository import ensure_schema
    from tools.finance.financial_facts_repository import query_observations

    await ensure_schema()
    rows = await query_observations(document_id=document_id, limit=limit)
    _safe_print_json({"count": len(rows), "observations": rows})


def _cli_report_locale(value: str | None) -> str | None:
    if not value or value.strip().lower() == "auto":
        return None
    return value.strip().lower()


async def cmd_ask_direct(
    document_id: int,
    question: str,
    top_k: int,
    *,
    pipeline_trace: bool,
    report_locale: str | None,
) -> None:
    _load_env()
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))
    from tools.rag_service import answer_question

    out = await answer_question(
        question=question,
        document_ids=[document_id],
        detail_level="detailed",
        top_k=top_k,
        include_pipeline_trace=pipeline_trace,
        include_full_retrieval_debug=False,
        report_locale=_cli_report_locale(report_locale),
    )
    _safe_print_json(_build_ask_log_payload(out, include_pipeline_trace=pipeline_trace))


async def cmd_ask_http(
    base_url: str,
    document_id: int,
    question: str,
    top_k: int,
    *,
    pipeline_trace: bool,
    http_timeout: float,
    report_locale: str | None,
) -> None:
    url = base_url.rstrip("/") + "/agent/api/ask/generate"
    body = {
        "question": question,
        "document_ids": [document_id],
        "top_k": top_k,
        "detail_level": "detailed",
        "include_pipeline_trace": pipeline_trace,
        "include_full_retrieval_debug": False,
    }
    rl = _cli_report_locale(report_locale)
    if rl is not None:
        body["report_locale"] = rl
    async with _api_client(timeout=http_timeout) as client:
        r = await client.post(url, json=body)
    r.raise_for_status()
    data = r.json()
    _safe_print_json(_build_ask_log_payload(data, include_pipeline_trace=pipeline_trace))


def cmd_list_accessions(json_path: Path) -> None:
    path = _ensure_path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))
    from tools.finance.sec_company_facts import is_sec_company_facts_payload, list_accessions_from_company_facts

    if not is_sec_company_facts_payload(data):
        raise SystemExit("JSON 不是 SEC companyfacts（需 cik + facts）")
    acc = list_accessions_from_company_facts(data)
    _safe_print_json(
        {"cik": data.get("cik"), "entityName": data.get("entityName"), "count": len(acc), "accessions": acc}
    )


async def cmd_ingest_edgar(
    json_path: Path,
    document_id_start: int,
    max_filings: int,
    accessions: list[str] | None,
    delay: float,
    download_dir: Path | None,
    *,
    ingest_documents: bool = True,
) -> None:
    _load_env()
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))
    path = _ensure_path(json_path)
    from tools.finance.edgar_sync import sync_filings_from_company_facts_file

    filt = set(accessions) if accessions else None
    out = await sync_filings_from_company_facts_file(
        company_facts_path=path,
        document_id_start=document_id_start,
        max_filings=max_filings,
        accessions=filt,
        download_dir=download_dir,
        ingest_documents=ingest_documents,
        delay_seconds=delay,
    )
    _safe_print_json(out)


async def cmd_ask_multi(
    document_ids: list[int],
    question: str,
    top_k: int,
    *,
    pipeline_trace: bool,
    report_locale: str | None,
) -> None:
    _load_env()
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))
    from tools.rag_service import answer_question

    out = await answer_question(
        question=question,
        document_ids=document_ids,
        detail_level="detailed",
        top_k=top_k,
        include_pipeline_trace=pipeline_trace,
        include_full_retrieval_debug=False,
        report_locale=_cli_report_locale(report_locale),
    )
    _safe_print_json(_build_ask_log_payload(out, include_pipeline_trace=pipeline_trace))


def cmd_product_spec() -> None:
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))
    from tools.finance.product_surface import get_finance_product_spec

    _safe_print_json(get_finance_product_spec())


def _accession_meta_from_companyfacts(
    data: dict,
) -> tuple[dict[str, dict[str, str]], int | None, str | None]:
    """Map accession -> {form, filed}; plus root cik / entityName."""
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))
    from tools.finance.sec_company_facts import flatten_sec_company_facts, is_sec_company_facts_payload

    if not is_sec_company_facts_payload(data):
        return {}, None, None
    cik_root = data.get("cik")
    entity_root = data.get("entityName")
    cik_int = int(cik_root) if cik_root is not None else None
    entity = str(entity_root).strip() if entity_root else None
    rows = flatten_sec_company_facts(data)
    idx: dict[str, dict[str, str]] = {}
    for r in rows:
        a = str(r.get("accn") or "").strip()
        if not a:
            continue
        b = idx.setdefault(a, {})
        if r.get("form"):
            b["form"] = str(r["form"])
        fd = r.get("filed_date")
        if fd:
            fds = str(fd).strip()
            if "filed" not in b or fds > b.get("filed", ""):
                b["filed"] = fds
    return idx, cik_int, entity


def _parse_edgar_local_filename(stem: str) -> tuple[int, str] | None:
    """``EDGAR_<cik>_<accession>`` without extension (e.g. EDGAR_1823776_0001193125-21-100318)."""
    m = _EDGAR_LOCAL_STEM.match(stem)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


async def cmd_ingest_edgar_local(
    data_dir: Path,
    document_id_start: int,
    glob_pat: str,
    form: str,
    filed: str | None,
    entity_name: str | None,
    companyfacts_json: Path,
) -> None:
    """Ingest existing ``tools/data/EDGAR_*.htm`` (Unstructured sidecar JSON beside each file)."""
    _load_env()
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))
    from tools.ingestion_service import process_edgar_filing_document

    root = data_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")
    paths = sorted(root.glob(glob_pat))
    paths = [p for p in paths if p.is_file() and p.suffix.lower() in (".htm", ".html")]
    if not paths:
        raise SystemExit(f"未匹配到文件: {root / glob_pat}")

    acc_idx: dict[str, dict[str, str]] = {}
    cik_from_facts: int | None = None
    entity_from_facts: str | None = None
    cf_path_resolved: str | None = None
    cf = companyfacts_json.resolve()
    if cf.is_file():
        cf_data = json.loads(cf.read_text(encoding="utf-8"))
        acc_idx, cik_from_facts, entity_from_facts = _accession_meta_from_companyfacts(cf_data)
        cf_path_resolved = str(cf)

    cli_form = (form or "").strip()
    cli_filed = (filed or "").strip() or None
    cli_entity = (entity_name or "").strip() or None

    results: list[dict] = []
    doc_id = document_id_start
    skipped: list[str] = []
    for p in paths:
        parsed = _parse_edgar_local_filename(p.stem)
        if not parsed:
            skipped.append(p.name)
            continue
        cik, accession = parsed
        row_meta = acc_idx.get(accession, {})
        eff_form = cli_form or row_meta.get("form", "")
        eff_filed = cli_filed or row_meta.get("filed") or None
        eff_entity = cli_entity or entity_from_facts
        cik_warn: str | None = None
        if cik_from_facts is not None and cik != cik_from_facts:
            cik_warn = f"htm 文件名 CIK {cik} 与 companyfacts CIK {cik_from_facts} 不一致"
        out = await process_edgar_filing_document(
            str(p.resolve()),
            doc_id,
            cik=cik,
            accession=accession,
            form=eff_form,
            filed=eff_filed,
            entity_name=eff_entity,
            source_url=None,
            primary_document=p.name,
        )
        item = {"file": p.name, "document_id": doc_id, **out}
        if cik_warn:
            item["warning"] = cik_warn
        if cf_path_resolved:
            item["companyfacts_match"] = accession in acc_idx
        results.append(item)
        doc_id += 1

    _safe_print_json(
        {
            "data_dir": str(root),
            "glob": glob_pat,
            "companyfacts_json": cf_path_resolved,
            "ingested": results,
            "skipped_unmatched_name": skipped,
            "next_document_id": doc_id,
        }
    )


async def cmd_reindex_vectors(document_id: int) -> None:
    """Rebuild Qdrant + OpenSearch from Postgres rag_nodes (repairs missing dense/sparse)."""
    _load_env()
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))
    from tools.ingestion_service import reindex_document_vectors

    out = await reindex_document_vectors(document_id)
    _safe_print_json(out)


async def cmd_observations_http(base_url: str, document_id: int, limit: int) -> None:
    url = (
        base_url.rstrip("/")
        + f"/agent/api/finance/observations?document_id={document_id}&limit={limit}"
    )
    async with _api_client(timeout=60.0) as client:
        r = await client.get(url)
    r.raise_for_status()
    _safe_print_json(r.json())


def main() -> None:
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(description="SEC companyfacts 财务流水线脚本")
    parser.add_argument(
        "command",
        choices=(
            "ingest-direct",
            "ingest-http",
            "sql-sample",
            "ask-direct",
            "ask-multi",
            "ask-http",
            "product-spec",
            "list-accessions",
            "ingest-edgar",
            "download-edgar",
            "ingest-edgar-local",
            "observations-http",
            "reindex-vectors",
        ),
    )
    parser.add_argument("--document-id", type=int, default=DEFAULT_DOC_ID)
    parser.add_argument(
        "--document-id-start",
        type=int,
        default=DEFAULT_EDGAR_DOC_START,
        help="ingest-edgar：第一个 EDGAR 文档的 document_id，其后顺延",
    )
    parser.add_argument(
        "--document-ids",
        type=str,
        default="",
        help="ask-multi：逗号分隔，如 9002,9201（facts + 多份 EDGAR）；需与 ingest 时 document_id 一致；与 --group 二选一",
    )
    parser.add_argument(
        "--group",
        default="",
        help="ask-multi：使用分组文件中的命名分组（与 --document-ids 二选一；指定时忽略 --document-ids）",
    )
    parser.add_argument(
        "--group-file",
        type=Path,
        default=None,
        help="ask-multi：分组定义 JSON（默认 tools/data/document_groups.json）",
    )
    parser.add_argument("--max-filings", type=int, default=20, help="ingest-edgar：最多同步几份申报")
    parser.add_argument(
        "--accession",
        action="append",
        default=[],
        help="ingest-edgar：仅同步指定 accession（可重复）",
    )
    parser.add_argument(
        "--edgar-delay",
        type=float,
        default=0.12,
        help="ingest-edgar：两次下载之间的间隔（秒），减轻对 SEC 的压力",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=AGENT_ROOT / "tools" / "data",
        help="ingest-edgar：原始 EDGAR 文件保存目录（默认 src/agent/tools/data）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="ingest-edgar-local：存放本地 .htm 的目录（默认 tools/data）",
    )
    parser.add_argument(
        "--edgar-glob",
        default="EDGAR_*.htm",
        help="ingest-edgar-local：相对 data-dir 的 glob",
    )
    parser.add_argument(
        "--form",
        default="",
        help="ingest-edgar-local：可选，统一写入的 form（如 10-K）",
    )
    parser.add_argument(
        "--filed",
        default="",
        help="ingest-edgar-local：可选，统一 filing 日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--entity-name",
        default="",
        help="ingest-edgar-local：可选，公司名",
    )
    parser.add_argument(
        "--companyfacts-json",
        type=Path,
        default=DEFAULT_JSON,
        help="ingest-edgar-local：companyfacts JSON，按 accession 补 form/filed，根上 entityName；文件不存在则跳过对齐",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON,
        help="SEC companyfacts JSON 路径",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="与 10-K 申报相关的结构化事实有哪些？")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=30, help="sql-sample / observations-http")
    parser.add_argument("--pipeline-trace", action="store_true")
    parser.add_argument(
        "--report-locale",
        default="auto",
        choices=("auto", "zh", "en"),
        help="产品报告与 evidence_ui 文案语言：auto 按问题推断，zh/en 强制",
    )
    parser.add_argument("--http-timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.command == "ingest-direct":
        asyncio.run(cmd_ingest_direct(args.document_id, args.json_path))
    elif args.command == "ingest-http":
        asyncio.run(cmd_ingest_http(args.base_url, args.document_id, args.json_path))
    elif args.command == "sql-sample":
        asyncio.run(cmd_sql_sample(args.document_id, args.limit))
    elif args.command == "ask-direct":
        asyncio.run(
            cmd_ask_direct(
                args.document_id,
                args.question,
                args.top_k,
                pipeline_trace=args.pipeline_trace,
                report_locale=args.report_locale,
            )
        )
    elif args.command == "ask-multi":
        group_name = (args.group or "").strip()
        if group_name and (args.document_ids or "").strip():
            print("[ask-multi] 已指定 --group，忽略 --document-ids", file=sys.stderr)
        if group_name:
            if str(AGENT_ROOT) not in sys.path:
                sys.path.insert(0, str(AGENT_ROOT))
            from tools.document_groups import default_document_groups_path, load_document_groups

            gf = (args.group_file or default_document_groups_path()).resolve()
            if not gf.is_file():
                raise SystemExit(f"分组文件不存在: {gf}")
            try:
                groups = load_document_groups(gf)
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                raise SystemExit(f"无法读取分组文件 {gf}: {exc}") from exc
            if group_name not in groups:
                avail = ", ".join(sorted(groups.keys())) or "(无)"
                raise SystemExit(f"分组 {group_name!r} 不在 {gf} 中。可用分组: {avail}")
            ids = groups[group_name]
        else:
            raw = (args.document_ids or "").strip()
            if not raw:
                raise SystemExit(
                    "请提供 --document-ids（例如 9002,9100,9101）或 --group <名称>（需配置 --group-file）"
                )
            ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
        asyncio.run(
            cmd_ask_multi(
                ids,
                args.question,
                args.top_k,
                pipeline_trace=args.pipeline_trace,
                report_locale=args.report_locale,
            )
        )
    elif args.command == "list-accessions":
        cmd_list_accessions(args.json_path)
    elif args.command == "ingest-edgar":
        accs = [a.strip() for a in args.accession if a.strip()] or None
        asyncio.run(
            cmd_ingest_edgar(
                args.json_path,
                args.document_id_start,
                args.max_filings,
                accs,
                args.edgar_delay,
                args.download_dir,
                ingest_documents=True,
            )
        )
    elif args.command == "download-edgar":
        accs = [a.strip() for a in args.accession if a.strip()] or None
        asyncio.run(
            cmd_ingest_edgar(
                args.json_path,
                args.document_id_start,
                args.max_filings,
                accs,
                args.edgar_delay,
                args.download_dir,
                ingest_documents=False,
            )
        )
    elif args.command == "ingest-edgar-local":
        filed = (args.filed or "").strip() or None
        ent = (args.entity_name or "").strip() or None
        asyncio.run(
            cmd_ingest_edgar_local(
                args.data_dir,
                args.document_id_start,
                args.edgar_glob,
                (args.form or "").strip(),
                filed,
                ent,
                args.companyfacts_json,
            )
        )
    elif args.command == "ask-http":
        asyncio.run(
            cmd_ask_http(
                args.base_url,
                args.document_id,
                args.question,
                args.top_k,
                pipeline_trace=args.pipeline_trace,
                http_timeout=args.http_timeout,
                report_locale=args.report_locale,
            )
        )
    elif args.command == "product-spec":
        cmd_product_spec()
    elif args.command == "reindex-vectors":
        asyncio.run(cmd_reindex_vectors(args.document_id))
    else:
        asyncio.run(cmd_observations_http(args.base_url, args.document_id, args.limit))


if __name__ == "__main__":
    main()
