#!/usr/bin/env python3
"""并行执行 tools/data/mixed_narrative_first_questions.json 中的提问（调用本地 Ask API）。

用法（在 src/agent 下，需已启动 uvicorn）:
  python scripts/run_mixed_narrative_questions_parallel.py
  python scripts/run_mixed_narrative_questions_parallel.py --base-url http://127.0.0.1:8000
  python scripts/run_mixed_narrative_questions_parallel.py --save-full --out-dir tools/data/batch_runs
  # 叙事-only（含原 planner 四题，见 mixed_narrative_only_questions.json id 11–14）；不设 FINANCE_SQL_ROUTING_ENABLED=false 时服务端仍会查 SQL
  python scripts/run_mixed_narrative_questions_parallel.py --questions tools/data/mixed_narrative_only_questions.json

环境:
  ASK_API_BASE           默认 http://127.0.0.1:8000
  ASK_DEFAULT_TOP_K      未传 --top-k 时发往 generate 的 top_k（与服务端 Config 一致，默认 3）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))
from core.config import config

DEFAULT_QUESTIONS = AGENT_DIR / "tools" / "data" / "mixed_narrative_first_questions.json"
ASK_PATH = "/agent/api/ask/generate"


def _document_ids_from_range(low: int, high: int) -> list[int]:
    return list(range(low, high + 1))


def load_questions(path: Path) -> tuple[dict, list[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("questions") or []
    return raw, items


async def _post_one(
    client: httpx.AsyncClient,
    base: str,
    qid: int,
    question: str,
    document_ids: list[int],
    *,
    top_k: int,
    detail_level: str,
    include_pipeline_trace: bool,
) -> dict:
    url = base.rstrip("/") + ASK_PATH
    payload = {
        "question": question,
        "document_ids": document_ids,
        "top_k": top_k,
        "detail_level": detail_level,
        "report_locale": "auto",
        "include_pipeline_trace": include_pipeline_trace,
        "include_full_retrieval_debug": False,
    }
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        body: dict | None
        try:
            body = r.json()
        except Exception:
            body = None
        if r.status_code >= 400:
            return {
                "id": qid,
                "ok": False,
                "status_code": r.status_code,
                "elapsed_ms": round(elapsed_ms, 2),
                "error": body or r.text[:2000],
            }
        return {
            "id": qid,
            "ok": True,
            "status_code": r.status_code,
            "elapsed_ms": round(elapsed_ms, 2),
            "trace_id": (body or {}).get("trace_id"),
            "answer_chars": len(((body or {}).get("answer") or "")),
            "citation_count": (body or {}).get("citation_count"),
            "sources_used": (body or {}).get("sources_used"),
            "confidence": (body or {}).get("confidence"),
            "latency_ms": (body or {}).get("latency_ms"),
            "full": body,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "id": qid,
            "ok": False,
            "elapsed_ms": round(elapsed_ms, 2),
            "error": str(exc),
        }


async def run_parallel(
    *,
    base_url: str,
    questions_path: Path,
    document_ids: list[int],
    top_k: int,
    detail_level: str,
    include_pipeline_trace: bool,
    timeout_s: float,
) -> list[dict]:
    _, items = load_questions(questions_path)
    if not items:
        raise SystemExit(f"no questions in {questions_path}")

    limits = httpx.Limits(max_keepalive_connections=20, max_connections=20)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s), limits=limits) as client:
        tasks = [
            _post_one(
                client,
                base_url,
                int(item["id"]),
                str(item["question"]),
                document_ids,
                top_k=top_k,
                detail_level=detail_level,
                include_pipeline_trace=include_pipeline_trace,
            )
            for item in items
        ]
        return await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel run of mixed_narrative_first_questions.json")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="Path to mixed_narrative_first_questions.json",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ASK_API_BASE", "http://127.0.0.1:8000"),
        help="API origin (no trailing path)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"override ASK_DEFAULT_TOP_K (default from env/config: {config.ask_default_top_k})",
    )
    parser.add_argument("--detail-level", default="detailed", choices=["brief", "detailed", "comprehensive"])
    parser.add_argument("--no-pipeline-trace", action="store_true", help="omit pipeline_trace in response")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request timeout seconds")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="if set, write batch_summary.json and optional full bodies",
    )
    parser.add_argument(
        "--save-full",
        action="store_true",
        help="with --out-dir, save one JSON per question with full API response",
    )
    args = parser.parse_args()
    top_k = args.top_k if args.top_k is not None else int(config.ask_default_top_k)

    meta, items = load_questions(args.questions)
    low, high = meta.get("document_id_range") or [9500, 9570]
    document_ids = _document_ids_from_range(int(low), int(high))

    print(
        f"base_url={args.base_url} questions={len(items)} document_ids={document_ids[0]}..{document_ids[-1]} ({len(document_ids)} docs)",
        flush=True,
    )

    include_pt = not args.no_pipeline_trace
    results = asyncio.run(
        run_parallel(
            base_url=args.base_url,
            questions_path=args.questions,
            document_ids=document_ids,
            top_k=top_k,
            detail_level=args.detail_level,
            include_pipeline_trace=include_pt,
            timeout_s=args.timeout,
        )
    )

    ok_n = sum(1 for r in results if r.get("ok"))
    print(f"\nfinished: {ok_n}/{len(results)} ok\n", flush=True)
    for r in sorted(results, key=lambda x: int(x.get("id", 0))):
        rid = r.get("id")
        if r.get("ok"):
            print(
                f"  id={rid} trace={r.get('trace_id')} elapsed_ms={r.get('elapsed_ms')} "
                f"answer_chars={r.get('answer_chars')} citations={r.get('citation_count')}",
                flush=True,
            )
        else:
            err = r.get("error")
            if isinstance(err, dict):
                err = json.dumps(err, ensure_ascii=False)[:500]
            print(f"  id={rid} FAILED: {r.get('status_code')} {err}", flush=True)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        summary_path = args.out_dir / f"batch_summary_{stamp}.json"
        slim = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "full"}
            slim.append(row)
        summary_path.write_text(
            json.dumps(
                {
                    "created_at": stamp,
                    "base_url": args.base_url,
                    "document_id_range": [low, high],
                    "ok_count": ok_n,
                    "total": len(results),
                    "results": slim,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {summary_path}", flush=True)
        if args.save_full:
            for r in results:
                if not r.get("ok") or "full" not in r:
                    continue
                qid = r.get("id")
                full_path = args.out_dir / f"batch_{stamp}_q{qid}_full.json"
                full_path.write_text(
                    json.dumps(r["full"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            print(f"full responses under {args.out_dir} (batch_{stamp}_q*_full.json)", flush=True)

    if ok_n < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
