#!/usr/bin/env python3
"""调用 POST /agent/api/ask/evaluate/pending 消费 RAGAS 评测队列。

用法（在 src/agent 下，需已启动 uvicorn，且 RAGAS_ENABLED=true 等已配置）:
  python scripts/run_evaluate_pending_parallel.py
  python scripts/run_evaluate_pending_parallel.py --base-url http://127.0.0.1:8000

默认会循环请求直到某次返回 processed=failed=skipped=0（队列已空）。
可选 --concurrency N>1 时并发发 N 个 POST（单实例 API 可能与并发消费同一批 pending 任务冲突，仅在你清楚后果时使用）。

环境:
  EVALUATE_API_BASE  默认 http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import httpx

AGENT_DIR = Path(__file__).resolve().parent.parent
EVALUATE_PATH = "/agent/api/ask/evaluate/pending"


async def _post_evaluate(client: httpx.AsyncClient, base: str) -> dict:
    url = base.rstrip("/") + EVALUATE_PATH
    t0 = time.perf_counter()
    r = await client.post(url)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text[:2000]}
    return {
        "status_code": r.status_code,
        "elapsed_ms": round(elapsed_ms, 2),
        "body": body,
        "ok": r.status_code < 400,
    }


async def _round_parallel(
    client: httpx.AsyncClient, base: str, concurrency: int
) -> list[dict]:
    tasks = [_post_evaluate(client, base) for _ in range(concurrency)]
    return await asyncio.gather(*tasks)


async def drain_queue(
    *,
    base_url: str,
    concurrency: int,
    max_rounds: int,
    timeout_s: float,
) -> int:
    """重复请求直到空队列或达到 max_rounds。返回累计 processed。"""
    base = (base_url or os.environ.get("EVALUATE_API_BASE") or "http://127.0.0.1:8000").strip()
    total_processed = 0
    total_failed = 0
    total_skipped = 0
    limits = httpx.Limits(
        max_keepalive_connections=max(20, concurrency * 2),
        max_connections=max(20, concurrency * 2),
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s), limits=limits) as client:
        for round_idx in range(max_rounds):
            if concurrency <= 1:
                results = [await _post_evaluate(client, base)]
            else:
                results = await _round_parallel(client, base, concurrency)

            round_processed = 0
            round_failed = 0
            round_skipped = 0
            for res in results:
                if not res["ok"]:
                    print(
                        f"[evaluate] HTTP {res['status_code']} in {res['elapsed_ms']}ms",
                        res.get("body"),
                        file=sys.stderr,
                    )
                    continue
                b = res["body"]
                if isinstance(b, dict):
                    round_processed += int(b.get("processed") or 0)
                    round_failed += int(b.get("failed") or 0)
                    round_skipped += int(b.get("skipped") or 0)
                print(
                    f"[evaluate] round={round_idx + 1} "
                    f"processed={b.get('processed') if isinstance(b, dict) else '?'} "
                    f"failed={b.get('failed') if isinstance(b, dict) else '?'} "
                    f"skipped={b.get('skipped') if isinstance(b, dict) else '?'} "
                    f"ms={res['elapsed_ms']}"
                )

            total_processed += round_processed
            total_failed += round_failed
            total_skipped += round_skipped

            if round_processed == 0 and round_failed == 0 and round_skipped == 0:
                print("[evaluate] queue empty (all zeros in this round).")
                break
            if concurrency > 1 and round_idx == 0:
                print(
                    "[evaluate] note: concurrency>1 may duplicate work on a single API instance.",
                    file=sys.stderr,
                )
        else:
            print(f"[evaluate] stopped after {max_rounds} rounds (see --max-rounds).", file=sys.stderr)

    print(
        f"[evaluate] totals: processed={total_processed} failed={total_failed} skipped={total_skipped}"
    )
    return total_processed


def main() -> None:
    p = argparse.ArgumentParser(description="POST /agent/api/ask/evaluate/pending (async, optional parallel burst)")
    p.add_argument(
        "--base-url",
        default=os.environ.get("EVALUATE_API_BASE", "http://127.0.0.1:8000"),
        help="API base, default EVALUATE_API_BASE or http://127.0.0.1:8000",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发 POST 数；默认 1（推荐）。>1 时同一轮会同时发多个 evaluate 请求。",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=500,
        help="最多循环轮数（每轮先并发 --concurrency 次再统计），防止死循环",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="单次 HTTP 超时（秒），RAGAS 可能较慢",
    )
    args = p.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    os.chdir(AGENT_DIR)
    if str(AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(AGENT_DIR))

    asyncio.run(
        drain_queue(
            base_url=args.base_url,
            concurrency=args.concurrency,
            max_rounds=args.max_rounds,
            timeout_s=args.timeout,
        )
    )


if __name__ == "__main__":
    main()
