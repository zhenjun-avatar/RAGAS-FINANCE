#!/usr/bin/env python3
"""
在职研究生线索提取脚本（HTTP 调用）。

在 src/agent 目录下执行（需已启动 API，并配置 BOCHA_API_KEY / LLM key）::

    cd src/agent
    ..\\venv\\Scripts\\python.exe scripts\\extract_part_time_graduate_leads.py \
      --search-mode both --freshness oneYear --count-per-mode 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict, Optional

import httpx
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(AGENT_ROOT / ".env")


def _api_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, trust_env=False)


async def cmd_extract(
    base_url: str,
    *,
    query: str,
    search_mode: str,
    freshness: str,
    count_per_mode: int,
    include_domains: Optional[str],
    exclude_ads: bool,
    max_sources_for_llm: int,
    llm_model_name: Optional[str],
    llm_temperature: float,
    http_timeout: float,
) -> None:
    url = base_url.rstrip("/") + "/agent/api/leads/extract-part-time-graduate-leads"
    body: Dict[str, Any] = {
        "query": query,
        "search_mode": search_mode,
        "freshness": freshness,
        "count_per_mode": count_per_mode,
        "include_domains": include_domains,
        "exclude_ads": exclude_ads,
        "max_sources_for_llm": max_sources_for_llm,
        "llm_model_name": llm_model_name,
        "llm_temperature": llm_temperature,
    }
    # remove nulls for cleaner payload
    body = {k: v for k, v in body.items() if v is not None}

    async with _api_client(timeout=http_timeout) as client:
        r = await client.post(url, json=body)
    r.raise_for_status()
    data = r.json()
    # Print only the useful parts (keep debug available).
    print(
        json.dumps(
            {
                "ok": data.get("ok"),
                "sources_found_pre_ad_filter": data.get("sources_found_pre_ad_filter"),
                "ads_filtered_out": data.get("ads_filtered_out"),
                "sources_found": data.get("sources_found"),
                "web_sources": data.get("web_sources"),
                "ai_sources": data.get("ai_sources"),
                "leads": data.get("leads") or [],
                "debug": data.get("debug"),
                "error": data.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="在职研究生线索提取（HTTP 调用）")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API 根地址（无尾斜杠）",
    )
    parser.add_argument(
        "--query",
        default="在职研究生 报考 咨询 经验贴 在职考研 非全日制 备考 评论",
    )
    parser.add_argument(
        "--search-mode",
        default="both",
        choices=("web", "ai", "both"),
    )
    parser.add_argument(
        "--freshness",
        default="oneYear",
        choices=("oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"),
    )
    parser.add_argument("--count-per-mode", type=int, default=10)
    parser.add_argument(
        "--include-domains",
        default=None,
        help=r"可选：站点范围（示例：\"zhihu.com|xiaohongshu.com\"；按 Bocha provider schema）",
    )
    parser.add_argument(
        "--exclude-ads",
        dest="exclude_ads",
        action="store_true",
        default=True,
        help="过滤疑似机构广告/营销内容（默认开启）",
    )
    parser.add_argument(
        "--no-exclude-ads",
        dest="exclude_ads",
        action="store_false",
        help="不进行机构广告过滤（可能含更多营销内容）",
    )
    parser.add_argument("--max-sources-for-llm", type=int, default=12)
    parser.add_argument("--llm-model-name", default=None, help="可选：覆盖服务端 LLM 模型名")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=180.0,
        help="httpx 超时秒数（包含联网检索 + LLM）",
    )

    args = parser.parse_args()
    _load_env()
    asyncio.run(
        cmd_extract(
            args.base_url,
            query=args.query,
            search_mode=args.search_mode,
            freshness=args.freshness,
            count_per_mode=args.count_per_mode,
            include_domains=args.include_domains,
            exclude_ads=args.exclude_ads,
            max_sources_for_llm=args.max_sources_for_llm,
            llm_model_name=args.llm_model_name,
            llm_temperature=args.llm_temperature,
            http_timeout=args.http_timeout,
        )
    )


if __name__ == "__main__":
    main()

