from __future__ import annotations

import json
import re
from typing import Any, Optional

from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage

from .bocha_search import BochaSearchClient
from ..llm import get_llm


def _extract_json_object(text: str) -> dict[str, Any]:
    # Try strict JSON first, then fallback to finding the first {...} block.
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return {}


def _build_evidence_block(results: list[dict[str, Any]], *, max_sources: int = 12) -> str:
    parts: list[str] = []
    for i, r in enumerate(results[: max_sources]):
        title = r.get("title") or f"source_{i}"
        url = r.get("url") or ""
        snippet = r.get("snippet") or ""
        site = r.get("site") or ""

        parts.append(
            f"[Source {i+1}]\n标题: {title}\n站点: {site}\n链接: {url}\n摘要/片段: {snippet}\n"
        )
    return "\n".join(parts).strip()


_AD_EXCLUDE_STRONG_PATTERNS: tuple[str, ...] = (
    # contact / marketing
    "微信",
    "qq",
    "QQ",
    "电话",
    "手机号",
    "客服",
    "官网",
    "官微",
    "联系方式",
    "在线咨询",
    "私信",
    "留言",
    # institution / agency
    "机构",
    "老师",
    "招生老师",
    "报名指导",
    "报考指导",
    "辅导",
    # strong promises (often ads)
    "包过",
    "保过",
    "包上岸",
    "承诺",
    "内部名额",
    "名额",
)


def _looks_like_ad_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # 11-digit phone number
    if re.search(r"\b1\d{10}\b", t):
        return True
    # common phone patterns: 010-12345678
    if re.search(r"\b\d{3,4}-\d{7,9}\b", t):
        return True

    lower = t.lower()
    strong_hits = 0
    for p in _AD_EXCLUDE_STRONG_PATTERNS:
        if p in t or p.lower() in lower:
            # "老师" / "咨询" might appear in real Q&A; rely on multiple hits.
            strong_hits += 1
            if strong_hits >= 2:
                return True
    return False


def _filter_ads(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """
    Filter out likely institution ads/marketing.
    Returns: (kept_items, removed_count)
    """
    kept: list[dict[str, Any]] = []
    removed = 0
    for x in items:
        title = (x.get("title") or "") or ""
        snippet = (x.get("snippet") or "") or ""
        raw = title + "\n" + snippet
        if _looks_like_ad_text(raw):
            removed += 1
        else:
            kept.append(x)
    return kept, removed


_HIGH_SIGNAL_TERMS: tuple[str, ...] = (
    "报考",
    "报名",
    "备考",
    "上岸",
    "非全",
    "非全日制",
    "在职研究生",
    "学费",
    "奖学金",
    "院校",
    "专业",
    "时间",
    "材料",
    "经验贴",
)

_GOOD_DOMAINS: tuple[str, ...] = (
    "zhihu.com",
    "xiaohongshu.com",
    "tieba.baidu.com",
    "weibo.com",
    "bilibili.com",
    "douban.com",
    "kaoyan.com",
    "freekaoyan.com",
)


def _source_quality_score(item: dict[str, Any]) -> float:
    """
    Multi-source quality scoring:
    - domain prior
    - high-intent term density
    - ad-like penalty
    """
    title = str(item.get("title") or "")
    snippet = str(item.get("snippet") or "")
    site = str(item.get("site") or "")
    url = str(item.get("url") or "")
    text = f"{title}\n{snippet}"
    lower = text.lower()

    score = 0.0

    # domain prior
    domain_blob = f"{site} {url}".lower()
    if any(d in domain_blob for d in _GOOD_DOMAINS):
        score += 0.8

    # intent terms
    hits = 0
    for t in _HIGH_SIGNAL_TERMS:
        if t in text or t.lower() in lower:
            hits += 1
    score += min(1.2, hits * 0.2)

    # text richness
    score += min(0.6, len(snippet.strip()) / 300.0)

    # ad penalty
    if _looks_like_ad_text(text):
        score -= 1.5

    return round(score, 4)


def _rank_sources(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for x in items:
        row = dict(x)
        row["quality_score"] = _source_quality_score(row)
        ranked.append(row)
    ranked.sort(key=lambda r: float(r.get("quality_score") or 0.0), reverse=True)
    return ranked


def _norm_text_for_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。、“”‘’？！!?,.:;；（）()【】\[\]<>《》\-_/\\|]+", "", s)
    return s


def _simple_overlap_score(a: str, b: str) -> float:
    """
    Character trigram overlap score in [0, 1].
    """
    aa = _norm_text_for_match(a)
    bb = _norm_text_for_match(b)
    if not aa or not bb:
        return 0.0
    if aa in bb or bb in aa:
        return 1.0
    if len(aa) < 3 or len(bb) < 3:
        return 0.0
    ag = {aa[i : i + 3] for i in range(len(aa) - 2)}
    bg = {bb[i : i + 3] for i in range(len(bb) - 2)}
    if not ag or not bg:
        return 0.0
    inter = len(ag & bg)
    union = len(ag | bg)
    if union <= 0:
        return 0.0
    return inter / union


def _backfill_evidence_urls(
    leads: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    max_urls_per_lead: int = 3,
) -> None:
    """
    Fill lead['evidence_urls'] from evidence snippets by matching to source snippet/title.
    In-place update.
    """
    if not leads or not sources:
        return

    for lead in leads:
        cur_urls = [str(u).strip() for u in (lead.get("evidence_urls") or []) if str(u).strip()]
        if cur_urls:
            # keep deduped existing urls
            dedup = []
            seen = set()
            for u in cur_urls:
                if u not in seen:
                    seen.add(u)
                    dedup.append(u)
            lead["evidence_urls"] = dedup[:max_urls_per_lead]
            continue

        snippets = [str(s).strip() for s in (lead.get("evidence_snippets") or []) if str(s).strip()]
        if not snippets:
            lead["evidence_urls"] = []
            continue

        scored_urls: list[tuple[float, str]] = []
        for snip in snippets:
            best_score = 0.0
            best_url = ""
            for src in sources:
                url = str(src.get("url") or "").strip()
                if not url:
                    continue
                src_text = f"{src.get('title') or ''}\n{src.get('snippet') or ''}"
                s = _simple_overlap_score(snip, src_text)
                # bias by source quality if available
                s += max(0.0, float(src.get("quality_score") or 0.0)) * 0.05
                if s > best_score:
                    best_score = s
                    best_url = url
            # keep weak-match threshold low to improve recall
            if best_url and best_score >= 0.05:
                scored_urls.append((best_score, best_url))

        scored_urls.sort(key=lambda x: x[0], reverse=True)
        out: list[str] = []
        seen = set()
        for _, u in scored_urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
            if len(out) >= max_urls_per_lead:
                break
        lead["evidence_urls"] = out


async def extract_part_time_graduate_leads(
    *,
    query: str = "在职研究生 报考 咨询 经验贴 在职考研 非全日制 备考 评论",
    search_mode: str = "both",
    freshness: str = "oneYear",
    count_per_mode: int = 10,
    include_domains: Optional[str] = None,
    max_sources_for_llm: int = 12,
    llm_model_name: Optional[str] = None,
    llm_temperature: float = 0.0,
    exclude_ads: bool = True,
) -> dict[str, Any]:
    """
    Search public discussions / comments about part-time graduate admissions (在职研究生),
    then extract high-intent lead candidates.
    """
    client = BochaSearchClient()
    if not client.enabled():
        return {"ok": False, "error": "BOCHA_API_KEY not configured", "leads": []}

    search_mode = (search_mode or "both").lower().strip()
    if search_mode not in {"web", "ai", "both"}:
        search_mode = "both"

    web_results: list[dict[str, Any]] = []
    ai_results: list[dict[str, Any]] = []
    ai_answer: Optional[str] = None
    web_raw: Optional[dict[str, Any]] = None
    ai_raw: Optional[dict[str, Any]] = None

    if search_mode in {"web", "both"}:
        web_raw = await client.web_search(
            query=query,
            freshness=freshness,
            summary=True,
            count=count_per_mode,
            include=include_domains,
        )
        if web_raw.get("ok"):
            web_results = web_raw.get("results") or []

    if search_mode in {"ai", "both"}:
        ai_raw = await client.ai_search(
            query=query,
            freshness=freshness,
            include=include_domains,
            count=count_per_mode,
            answer=True,
            stream=False,
        )
        if ai_raw.get("ok"):
            ai_results = ai_raw.get("results") or []
            ai_answer = ai_raw.get("ai_answer")

    # Merge + de-dup by URL (best-effort).
    merged: list[dict[str, Any]] = []
    seen_url: set[str] = set()
    for r in (web_results + ai_results):
        url = (r.get("url") or "").strip()
        if url and url in seen_url:
            continue
        if url:
            seen_url.add(url)
        merged.append(r)

    # quality rank first (domain + intent terms + ad penalty)
    merged = _rank_sources(merged)

    merged_pre_ad_filter = list(merged)
    ads_filtered_out = 0
    if exclude_ads and merged:
        merged, ads_filtered_out = _filter_ads(merged)
        # keep quality order after filtering
        merged = _rank_sources(merged)

    evidence_block = _build_evidence_block(merged, max_sources=max_sources_for_llm)
    if not evidence_block:
        return {
            "ok": True,
            "query": query,
            "search_mode": search_mode,
            "sources_found": 0,
            "sources_found_pre_ad_filter": len(merged_pre_ad_filter),
            "ads_filtered_out": ads_filtered_out,
            "web_sources": len(web_results),
            "ai_sources": len(ai_results),
            "debug": {
                "bocha_web_raw_ok": bool(web_raw.get("ok")) if isinstance(web_raw, dict) else False,
                "bocha_ai_raw_ok": bool(ai_raw.get("ok")) if isinstance(ai_raw, dict) else False,
                "bocha_web_error": (
                    web_raw.get("error") if isinstance(web_raw, dict) and not web_raw.get("ok") else None
                ),
                "bocha_ai_error": (
                    ai_raw.get("error") if isinstance(ai_raw, dict) and not ai_raw.get("ok") else None
                ),
                "bocha_web_stats": web_raw.get("stats") if isinstance(web_raw, dict) else None,
                "bocha_ai_stats": ai_raw.get("stats") if isinstance(ai_raw, dict) else None,
            },
            "leads": [],
        }

    llm = get_llm(model_name=llm_model_name, temperature=llm_temperature)

    # 只要求从片段里抽取“高意向”信号，不要虚构具体个人信息。
    system_prompt = (
        "你是招生线索分析员。"
        "你的任务：从给定的公开讨论/评论片段中，识别“高意向报考在职研究生”的线索信号。"
        "只基于证据片段，不要编造。"
        "输出严格 JSON，不要输出多余文字。"
    )
    user_prompt = f"""
用户给定检索 query：{query}

下面是若干公开网页/讨论的标题与摘要片段（可能包含在职研究生报考咨询、经验贴、报名流程、院校/专业选择、上课方式、学习安排、顾虑与痛点等）。

请输出 JSON，字段含义：
{{
  "intent_rule": "高意向报考线索通常包括：明确表述“准备报考/正在咨询/已报名/即将报名”；提到具体时间窗口（如2025/2026考研、招生季）；询问院校或专业、学费/奖助/上课方式（周末/集中/远程）；表现出强行动意图（比如“要不要报”“怎么选”“需要哪些材料”“能否通过”等）。",
  "lead_candidates": [
    {{
      "intent_score": 0.0,
      "intent_label": "高意向|中意向|低意向",
      "target_keywords": ["在职研究生", "报考", "咨询", "报名", "院校/专业", "学费/奖助", "上课方式", "毕业要求/学位/证书", "时间安排"],
      "lead_type": "已咨询/正在选校|即将报名/已报名|有明确问题/材料清单|纠结观望/需要方案|其他",
      "evidence_snippets": ["必须来自片段原文或摘要原文的短句（可截断），不要编造。"],
      "evidence_urls": ["证据对应的来源链接（来自 [Evidence] 中的 链接 字段；没有就给空数组）"],
      "likely_target_majors_or_programs": ["可能的专业/项目（如MBA/MPA/心理学专硕等），没有就给空数组"],
      "timeframe_hint": "可为空；如“2026年考研/明年/近期招生季/尽快”等",
      "notes": "可为空；简述该线索为何高意向/中意向（基于证据）"
    }}
  ],
  "sources_used_count": {min(max_sources_for_llm, len(merged))},
  "bocha_ai_answer_used": {bool(ai_answer)}
}}

要求：
- lead_candidates 里尽量返回 0~5 条最相关线索；没有证据就返回空数组。
- evidence_snippets 至少 1 条，最多 3 条；并且每条不超过 120 字符（中文计字更严格时可截断）。
- intent_score 取 0.0~1.0。
"""

    try:
        resp = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=user_prompt + "\n\n[Evidence]\n" + evidence_block
                ),
            ]
        )
        raw_text = resp.content if hasattr(resp, "content") else str(resp)
        parsed = _extract_json_object(raw_text)
    except Exception as exc:
        logger.exception("[leads] llm extraction failed: {}", exc)
        return {
            "ok": True,
            "query": query,
            "search_mode": search_mode,
            "sources_found": len(merged),
            "leads": [],
            "error": str(exc),
        }

    lead_candidates = parsed.get("lead_candidates") or []
    if not isinstance(lead_candidates, list):
        lead_candidates = []

    # Normalize output keys into a stable shape for downstream use.
    leads_out: list[dict[str, Any]] = []
    for item in lead_candidates[:5]:
        if not isinstance(item, dict):
            continue
        leads_out.append(
            {
                "intent_score": float(item.get("intent_score") or 0.0),
                "intent_label": str(item.get("intent_label") or ""),
                "lead_type": str(item.get("lead_type") or ""),
                "target_keywords": item.get("target_keywords") or [],
                "evidence_snippets": item.get("evidence_snippets") or [],
                "evidence_urls": item.get("evidence_urls") or [],
                "likely_target_majors_or_programs": item.get(
                    "likely_target_majors_or_programs"
                )
                or [],
                "timeframe_hint": item.get("timeframe_hint") or "",
                "notes": item.get("notes") or "",
            }
        )

    # URL auto backfill: map snippets -> best matched sources
    _backfill_evidence_urls(leads_out, merged)

    return {
        "ok": True,
        "query": query,
        "search_mode": search_mode,
        "sources_found": len(merged),
        "sources_found_pre_ad_filter": len(merged_pre_ad_filter),
        "ads_filtered_out": ads_filtered_out,
        "web_sources": len(web_results),
        "ai_sources": len(ai_results),
        "leads": leads_out,
        "debug": {
            "evidence_used_chars": len(evidence_block),
            "bocha_web_raw_ok": bool(web_raw.get("ok")) if isinstance(web_raw, dict) else False,
            "bocha_ai_raw_ok": bool(ai_raw.get("ok")) if isinstance(ai_raw, dict) else False,
            "bocha_web_error": (
                web_raw.get("error") if isinstance(web_raw, dict) and not web_raw.get("ok") else None
            ),
            "bocha_ai_error": (
                ai_raw.get("error") if isinstance(ai_raw, dict) and not ai_raw.get("ok") else None
            ),
            "bocha_web_stats": web_raw.get("stats") if isinstance(web_raw, dict) else None,
            "bocha_ai_stats": ai_raw.get("stats") if isinstance(ai_raw, dict) else None,
            "source_quality_top": [
                {
                    "title": x.get("title"),
                    "url": x.get("url"),
                    "quality_score": x.get("quality_score"),
                }
                for x in merged[:10]
            ],
        },
    }

