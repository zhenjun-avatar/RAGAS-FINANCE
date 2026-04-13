"""Answer-aligned verbatim quotes for evidence UI (post-hoc LLM selection + substring verification)."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import config
from tools.llm import get_llm
from tools.rag_stage_log import log_rag


def _norm_accn(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _node_metadata(node: dict[str, Any]) -> dict[str, Any]:
    metadata = node.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _node_accessions(node: dict[str, Any]) -> tuple[str, ...]:
    meta = _node_metadata(node)
    raw_values = [
        node.get("finance_accns"),
        meta.get("finance_accns"),
        node.get("sec_accession"),
        meta.get("sec_accession"),
    ]
    out: list[str] = []
    for raw in raw_values:
        values = raw if isinstance(raw, list) else [raw]
        for item in values:
            accn = _norm_accn(item)
            if accn and accn not in out:
                out.append(accn)
    return tuple(out)


def _first_accession_per_document(nodes: list[dict[str, Any]]) -> dict[int, str]:
    doc_to_accn: dict[int, str] = {}
    for node in nodes:
        doc_id = node.get("document_id")
        if doc_id is None:
            continue
        did = int(doc_id)
        if did in doc_to_accn:
            continue
        accns = _node_accessions(node)
        if accns:
            doc_to_accn[did] = accns[0]
    return doc_to_accn


def _node_relevance(item: dict[str, Any]) -> float:
    r = item.get("rerank_score")
    if r is not None:
        return float(r)
    return float(item.get("score") or 0.0)


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _is_mostly_md_table(quote: str) -> bool:
    lines = [ln for ln in quote.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    head = lines[: min(16, len(lines))]
    pipe_lines = sum(1 for ln in head if ln.count("|") >= 2)
    return pipe_lines >= max(2, (len(head) + 1) // 2)


def _verify_verbatim(quote: str, excerpt: str) -> str | None:
    """Return the matching slice if quote is a contiguous verbatim substring (allow CRLF normalization)."""
    if not quote or not excerpt:
        return None
    q_strip = quote.strip()
    if not q_strip:
        return None
    pos = excerpt.find(q_strip)
    if pos >= 0:
        return excerpt[pos : pos + len(q_strip)]
    en = excerpt.replace("\r\n", "\n")
    qn = q_strip.replace("\r\n", "\n")
    pos = en.find(qn)
    if pos < 0:
        return None
    return en[pos : pos + len(qn)]


def _accn_for_node(node: dict[str, Any], doc_accn: dict[int, str]) -> str | None:
    accns = _node_accessions(node)
    if accns:
        return accns[0]
    doc_id = node.get("document_id")
    if doc_id is None:
        return None
    return doc_accn.get(int(doc_id))


async def extract_answer_aligned_verified_quotes(
    *,
    question: str,
    answer: str,
    nodes: list[dict[str, Any]],
    locale: str,
) -> list[dict[str, Any]]:
    """
    Use an LLM to pick up to 3 excerpts that support the existing answer; each excerpt must
    appear verbatim in the provided source text (verified by substring match).
    """
    if not nodes or not (answer or "").strip():
        return []

    max_sources = max(1, int(config.ask_evidence_llm_max_sources))
    cap = max(500, int(config.ask_evidence_llm_chars_per_source))
    max_quotes = max(1, min(5, int(config.ask_evidence_llm_max_quotes)))
    max_q_chars = max(200, int(config.ask_evidence_llm_max_quote_chars))
    min_q_chars = max(20, int(config.ask_evidence_llm_min_quote_chars))

    ranked = sorted(nodes, key=_node_relevance, reverse=True)[:max_sources]
    doc_accn = _first_accession_per_document(ranked)

    sources: list[dict[str, Any]] = []
    blocks: list[str] = []
    for i, node in enumerate(ranked, start=1):
        raw = str(node.get("text") or "")
        truncated = len(raw) > cap
        excerpt = raw[:cap] if truncated else raw
        tail = "\n[Text truncated for this prompt — quote only from the text above.]" if truncated else ""
        node_id = str(node.get("node_id") or "")
        doc_id = node.get("document_id")
        title = str(node.get("title") or "").strip()
        header = f"--- SOURCE {i} ---\nnode_id: {node_id}\ndocument_id: {doc_id}\ntitle: {title or 'n/a'}"
        blocks.append(f"{header}\n\n{excerpt}{tail}")
        sources.append(
            {
                "index": i,
                "node": node,
                "excerpt": excerpt,
            }
        )

    bundle = "\n\n".join(blocks)
    loc = "en" if str(locale).lower() == "en" else "zh"

    if loc == "en":
        system = (
            "You select supporting evidence for a finance disclosure Q&A product. "
            "You MUST output a single JSON object only, no markdown fences. "
            "Schema: {\"quotes\": [{\"source_index\": <int 1-based>, \"text\": <string>}, ...]}. "
            f"Include at most {max_quotes} quotes. Each \"text\" MUST be copied exactly as a contiguous "
            "substring from the corresponding SOURCE block's body (after the header lines). "
            "Do not paraphrase, translate, fix typos, or merge non-adjacent spans. "
            "Prefer narrative sentences that directly support the given answer; skip markdown tables "
            "(lines dominated by pipe characters), boilerplate indexes, and generic section intros unless they alone support the answer. "
            f"Each quote must be at least {min_q_chars} characters and at most {max_q_chars} characters. "
            "If nothing qualifies, return {\"quotes\": []}."
        )
        user = (
            f"Question:\n{question}\n\n"
            "Answer (already finalized — use only to judge relevance; do not rewrite it):\n"
            f"{answer}\n\n"
            "Source excerpts:\n"
            f"{bundle}\n\n"
            "Return JSON only."
        )
    else:
        system = (
            "你是披露问答产品的证据抽取助手。只输出一个 JSON 对象，不要用 markdown 代码围栏。"
            "格式：{\"quotes\": [{\"source_index\": <从1开始的整数>, \"text\": <字符串>}, ...]}。"
            f"最多 {max_quotes} 条。每条 \"text\" 必须从对应 SOURCE 正文（标题行之后）原样复制连续子串，"
            "不得改写、翻译、纠错或拼接不相邻片段。"
            "优先选择与所给答案直接相关的叙述句；跳过以竖线表格为主的 markdown 表、附件索引、以及与答案无关的套话开篇。"
            f"每条长度须在 {min_q_chars}–{max_q_chars} 字符之间。若无合格片段，返回 {{\"quotes\": []}}。"
        )
        user = (
            f"问题：\n{question}\n\n"
            "答案（已定稿——仅用于判断相关性，不要改写）：\n"
            f"{answer}\n\n"
            "来源摘录：\n"
            f"{bundle}\n\n"
            "只返回 JSON。"
        )

    llm = get_llm(model_name=config.default_model, temperature=0.0)
    try:
        resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        raw_out = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as exc:
        log_rag("answer_evidence_quotes_llm_error", level="warning", error=str(exc)[:400])
        return []

    try:
        payload = json.loads(_strip_json_fence(str(raw_out)))
    except json.JSONDecodeError as exc:
        log_rag("answer_evidence_quotes_json_error", level="warning", error=str(exc)[:200])
        return []

    raw_quotes = payload.get("quotes") if isinstance(payload, dict) else None
    if not isinstance(raw_quotes, list):
        return []

    out: list[dict[str, Any]] = []
    seen_norm: set[str] = set()

    for item in raw_quotes:
        if len(out) >= max_quotes:
            break
        if not isinstance(item, dict):
            continue
        try:
            src_i = int(item.get("source_index"))
        except (TypeError, ValueError):
            continue
        text = str(item.get("text") or "").strip()
        if len(text) < min_q_chars or len(text) > max_q_chars:
            continue
        if _is_mostly_md_table(text):
            continue
        if src_i < 1 or src_i > len(sources):
            continue
        entry = sources[src_i - 1]
        excerpt = str(entry.get("excerpt") or "")
        verified = _verify_verbatim(text, excerpt)
        if not verified:
            continue
        norm_key = re.sub(r"\s+", " ", verified.lower())[:400]
        if norm_key in seen_norm:
            continue
        seen_norm.add(norm_key)

        node: dict[str, Any] = entry["node"]
        doc_id = node.get("document_id")
        if doc_id is None:
            continue
        sc = _node_relevance(node)
        card = {
            "body": verified,
            "document_id": int(doc_id),
            "node_id": str(node.get("node_id") or ""),
            "relevance_score": round(float(sc), 4),
            "relevance_level": "high" if sc >= 0.55 else ("medium" if sc >= 0.35 else "low"),
        }
        accn = _accn_for_node(node, doc_accn)
        if accn:
            card["accn"] = accn
        out.append(card)

    log_rag(
        "answer_evidence_quotes",
        quote_count=len(out),
        sources_in_prompt=len(sources),
        locale=loc,
    )
    return out
