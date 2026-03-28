#!/usr/bin/env python3
"""
测验 gaokaoyuwen.docx：入库（直连 ingestion 或 HTTP）与可选问答。

在 src/agent 目录下执行（需已配置 .env、Postgres、Qdrant）::

    cd src/agent
    ..\\venv\\Scripts\\python.exe scripts\\test_gaokaoyuwen_docx.py ingest-direct
    ..\\venv\\Scripts\\python.exe scripts\\test_gaokaoyuwen_docx.py ingest-http
    ..\\venv\\Scripts\\python.exe scripts\\test_gaokaoyuwen_docx.py ask
    ..\\venv\\Scripts\\python.exe scripts\\test_gaokaoyuwen_docx.py ask --pipeline-trace
    ..\\venv\\Scripts\\python.exe scripts\\test_gaokaoyuwen_docx.py pipeline
    ..\\venv\\Scripts\\python.exe scripts\\test_gaokaoyuwen_docx.py full

ingest-http / ask / full / pipeline 需先启动 API::

    python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

AGENT_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = AGENT_ROOT / "tools" / "data" / "gaokaoyuwen.docx"
DEFAULT_DOC_ID = 9001

# 与 HTTP AskResponse 对齐的终端输出字段（不含 citations / key_points / retrieval_debug）
_ASK_DISPLAY_KEYS = (
    "question",
    "answer",
    "confidence",
    "sources_used",
    "citation_count",
    "limitations",
    "trace_id",
    "pipeline_trace",
)


def _print_ask_response(data: dict) -> None:
    slim = {k: data[k] for k in _ASK_DISPLAY_KEYS if k in data}
    print(json.dumps(slim, ensure_ascii=False, indent=2))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(AGENT_ROOT / ".env")


def _api_client(timeout: float) -> httpx.AsyncClient:
    # Local API calls should bypass system proxy settings.
    return httpx.AsyncClient(timeout=timeout, trust_env=False)


def _ensure_doc() -> Path:
    if not DOC_PATH.is_file():
        raise SystemExit(f"找不到样例文档: {DOC_PATH}")
    return DOC_PATH.resolve()


async def cmd_ingest_direct(document_id: int) -> None:
    _load_env()
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))
    doc = _ensure_doc()
    from tools.ingestion_service import process_document

    out = await process_document(
        file_path=str(doc),
        file_type="docx",
        document_id=document_id,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


async def cmd_ingest_http(base_url: str, document_id: int) -> None:
    doc = _ensure_doc()
    url = base_url.rstrip("/") + "/agent/api/documents/process"
    body = {
        "document_id": document_id,
        "file_path": str(doc),
        "file_type": "docx",
    }
    async with _api_client(timeout=600.0) as client:
        r = await client.post(url, json=body)
    r.raise_for_status()
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))


async def cmd_pipeline(base_url: str) -> None:
    url = base_url.rstrip("/") + "/agent/api/observability/pipeline"
    async with _api_client(timeout=30.0) as client:
        r = await client.get(url)
    r.raise_for_status()
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))


async def cmd_ask(
    base_url: str,
    document_id: int,
    question: str,
    top_k: int,
    *,
    http_timeout: float,
    pipeline_trace: bool = False,
    full_retrieval_debug: bool = False,
) -> None:
    url = base_url.rstrip("/") + "/agent/api/ask/generate"
    body = {
        "question": question,
        "document_ids": [document_id],
        "top_k": top_k,
        "detail_level": "detailed",
        "include_pipeline_trace": pipeline_trace,
        "include_full_retrieval_debug": full_retrieval_debug,
    }
    async with _api_client(timeout=http_timeout) as client:
        r = await client.post(url, json=body)
    r.raise_for_status()
    data = r.json()
    _print_ask_response(data)


async def cmd_full(
    base_url: str,
    document_id: int,
    question: str,
    top_k: int,
    *,
    http_timeout: float,
    pipeline_trace: bool = False,
    full_retrieval_debug: bool = False,
) -> None:
    await cmd_ingest_http(base_url, document_id)
    await cmd_ask(
        base_url,
        document_id,
        question,
        top_k,
        http_timeout=http_timeout,
        pipeline_trace=pipeline_trace,
        full_retrieval_debug=full_retrieval_debug,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="测验 gaokaoyuwen.docx 入库与问答")
    parser.add_argument(
        "command",
        choices=("ingest-direct", "ingest-http", "ask", "full", "pipeline"),
        help="pipeline=GET 观测配置; ingest-direct=直连入库; 其余见各命令",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API 根地址（无尾斜杠）",
    )
    parser.add_argument(
        "--document-id",
        type=int,
        default=DEFAULT_DOC_ID,
        help="与 ask 时传入的 document_ids 一致",
    )
    parser.add_argument(
        "--question",
        default="请简要概括这份文档的主题与结构。",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--pipeline-trace",
        action="store_true",
        help="ask/full 时在请求体中带 include_pipeline_trace=true，响应含 pipeline_trace",
    )
    parser.add_argument(
        "--full-retrieval-debug",
        action="store_true",
        help="与 --pipeline-trace 联用，附带 retrieval_debug 键列表",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=600.0,
        help="ask/full 调用 API 的 httpx 超时秒数（嵌入重试+检索+LLM 常超过 120s）",
    )
    args = parser.parse_args()

    if args.command == "ingest-direct":
        asyncio.run(cmd_ingest_direct(args.document_id))
    elif args.command == "ingest-http":
        asyncio.run(cmd_ingest_http(args.base_url, args.document_id))
    elif args.command == "pipeline":
        asyncio.run(cmd_pipeline(args.base_url))
    elif args.command == "ask":
        asyncio.run(
            cmd_ask(
                args.base_url,
                args.document_id,
                args.question,
                args.top_k,
                http_timeout=args.http_timeout,
                pipeline_trace=args.pipeline_trace,
                full_retrieval_debug=args.full_retrieval_debug,
            )
        )
    else:
        asyncio.run(
            cmd_full(
                args.base_url,
                args.document_id,
                args.question,
                args.top_k,
                http_timeout=args.http_timeout,
                pipeline_trace=args.pipeline_trace,
                full_retrieval_debug=args.full_retrieval_debug,
            )
        )


if __name__ == "__main__":
    main()
