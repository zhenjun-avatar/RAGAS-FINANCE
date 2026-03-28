#!/usr/bin/env python3
"""
查看本地 OpenSearch 集群与稀疏索引（与 .env 中 OPENSEARCH_* 一致）。

在 src/agent 目录下执行::

    cd src/agent
    python scripts\\inspect_opensearch.py
    python scripts\\inspect_opensearch.py health
    python scripts\\inspect_opensearch.py indices
    python scripts\\inspect_opensearch.py count
    python scripts\\inspect_opensearch.py sample --limit 5
    python scripts\\inspect_opensearch.py search -q "文言文" --document-id 9001 --limit 5
    python scripts\\inspect_opensearch.py mapping
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

AGENT_ROOT = Path(__file__).resolve().parents[1]

if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from core.config import config  # noqa: E402
from opensearchpy import OpenSearch  # noqa: E402


def _client() -> OpenSearch:
    http_auth = None
    if config.opensearch_user:
        http_auth = (config.opensearch_user, config.opensearch_password or "")
    return OpenSearch(
        hosts=[
            {
                "host": config.opensearch_host,
                "port": config.opensearch_port,
                "scheme": "https" if config.opensearch_use_ssl else "http",
            }
        ],
        http_auth=http_auth,
        use_ssl=config.opensearch_use_ssl,
        verify_certs=config.opensearch_verify_certs,
        timeout=int(config.opensearch_timeout_seconds),
    )


def _print(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_health() -> None:
    c = _client()
    _print(
        {
            "cluster_health": c.cluster.health(),
            "info": c.info(),
        }
    )


def cmd_indices() -> None:
    c = _client()
    # 文本表格，便于扫一眼
    raw = c.cat.indices(format="json")
    _print(raw)


def cmd_count(index: str) -> None:
    c = _client()
    if not c.indices.exists(index=index):
        print(f"索引不存在: {index}", file=sys.stderr)
        sys.exit(1)
    _print(c.count(index=index))


def cmd_sample(index: str, limit: int) -> None:
    c = _client()
    if not c.indices.exists(index=index):
        print(f"索引不存在: {index}", file=sys.stderr)
        sys.exit(1)
    body = {"size": limit, "query": {"match_all": {}}, "sort": [{"_id": {"order": "asc"}}]}
    resp = c.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    slim = []
    for h in hits:
        src = h.get("_source") or {}
        text = (src.get("text") or "")[:200]
        slim.append(
            {
                "_id": h.get("_id"),
                "_score": h.get("_score"),
                "node_id": src.get("node_id"),
                "document_id": src.get("document_id"),
                "level": src.get("level"),
                "title": (src.get("title") or "")[:120],
                "text_preview": text + ("…" if len(src.get("text") or "") > 200 else ""),
            }
        )
    _print({"index": index, "total_relation": resp.get("hits", {}).get("total"), "hits": slim})


def cmd_search(index: str, query: str, document_id: int | None, limit: int) -> None:
    c = _client()
    if not c.indices.exists(index=index):
        print(f"索引不存在: {index}", file=sys.stderr)
        sys.exit(1)
    filters: list[dict[str, Any]] = []
    if document_id is not None:
        filters.append({"term": {"document_id": document_id}})
    body: dict[str, Any] = {
        "size": limit,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query.strip(),
                            "fields": ["title^2", "text"],
                            "type": "best_fields",
                            "operator": "or",
                        }
                    }
                ],
                "filter": filters,
            }
        },
        "_source": ["node_id", "document_id", "level", "title", "text"],
        "sort": ["_score", {"level": {"order": "desc"}}, {"order_index": {"order": "asc"}}],
    }
    resp = c.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    slim = []
    for h in hits:
        src = h.get("_source") or {}
        text = (src.get("text") or "")[:240]
        slim.append(
            {
                "_score": h.get("_score"),
                "node_id": src.get("node_id"),
                "document_id": src.get("document_id"),
                "level": src.get("level"),
                "title": src.get("title"),
                "text_preview": text + ("…" if len(src.get("text") or "") > 240 else ""),
            }
        )
    _print(
        {
            "index": index,
            "query": query,
            "document_id_filter": document_id,
            "took_ms": resp.get("took"),
            "hits": slim,
        }
    )


def cmd_mapping(index: str) -> None:
    c = _client()
    if not c.indices.exists(index=index):
        print(f"索引不存在: {index}", file=sys.stderr)
        sys.exit(1)
    _print(c.indices.get_mapping(index=index))


def cmd_summary(index: str) -> None:
    """默认：健康 + 索引行 + 文档数 + 两条样例。"""
    c = _client()
    health = c.cluster.health()
    count_body: dict[str, Any] = {"error": None, "count": None}
    sample_hits: list[Any] = []
    if c.indices.exists(index=index):
        count_body = c.count(index=index)
        s = c.search(
            index=index,
            body={"size": 2, "query": {"match_all": {}}, "sort": [{"_score": {"order": "desc"}}]},
        )
        for h in s.get("hits", {}).get("hits", []):
            src = h.get("_source") or {}
            full_text = src.get("text") or ""
            sample_hits.append(
                {
                    "node_id": src.get("node_id"),
                    "document_id": src.get("document_id"),
                    "level": src.get("level"),
                    "title": (src.get("title") or "")[:100],
                    "text_preview": (full_text[:160] + "…") if len(full_text) > 160 else full_text,
                }
            )
    else:
        count_body = {"error": f"index missing: {index}"}

    indices_rows = [r for r in c.cat.indices(format="json") if index in (r.get("index") or "")]
    _print(
        {
            "opensearch": f"{config.opensearch_host}:{config.opensearch_port}",
            "sparse_index": index,
            "cluster_health": {
                "status": health.get("status"),
                "cluster_name": health.get("cluster_name"),
                "number_of_nodes": health.get("number_of_nodes"),
            },
            "index_row": indices_rows[0] if indices_rows else None,
            "count": count_body,
            "sample_hits": sample_hits,
        }
    )


def main() -> None:
    default_index = config.opensearch_sparse_index
    p = argparse.ArgumentParser(description="Inspect OpenSearch (RAG sparse index)")
    p.add_argument(
        "command",
        nargs="?",
        default="summary",
        choices=("summary", "health", "indices", "count", "sample", "search", "mapping"),
        help="summary=健康+索引行+count+样例；search 与线上 sparse 查询类似",
    )
    p.add_argument("--index", default=default_index, help=f"默认 {default_index}")
    p.add_argument("--limit", type=int, default=5, help="sample/search 返回条数")
    p.add_argument("-q", "--query", default="", help="search 子命令：检索语句")
    p.add_argument("--document-id", type=int, default=None, help="search 时限定 document_id")

    args = p.parse_args()
    idx = args.index

    try:
        if args.command == "summary":
            cmd_summary(idx)
        elif args.command == "health":
            cmd_health()
        elif args.command == "indices":
            cmd_indices()
        elif args.command == "count":
            cmd_count(idx)
        elif args.command == "sample":
            cmd_sample(idx, max(1, args.limit))
        elif args.command == "mapping":
            cmd_mapping(idx)
        else:
            q = (args.query or "").strip()
            if not q:
                print("search 需要 -q/--query", file=sys.stderr)
                sys.exit(2)
            cmd_search(idx, q, args.document_id, max(1, args.limit))
    except Exception as exc:
        print(f"OpenSearch 请求失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
