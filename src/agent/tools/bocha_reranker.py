"""Bocha semantic reranker client with safe fallback."""

from __future__ import annotations

import time
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from loguru import logger

from core.config import config

from .rag_stage_log import log_rag


def _extract_scored_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for key in ("data", "results", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = _extract_scored_items(value)
                if nested:
                    return nested
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _score_from_item(item: dict[str, Any]) -> float:
    for key in ("relevance_score", "rerankScore", "score"):
        raw = item.get(key)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    return 0.0


def _log_rerank_stage(stats: dict[str, Any], latency_ms: float) -> None:
    url = config.bocha_reranker_url or ""
    host = urlparse(url).netloc or None
    err = stats.get("error")
    snippet = stats.get("response_body_snippet")
    if snippet:
        snippet = str(snippet)[:200]
    else:
        snippet = None
    log_rag(
        "rerank",
        mode=stats.get("mode"),
        candidates_in=stats.get("candidates_in"),
        candidates_out=stats.get("candidates_out"),
        remote_http_called=stats.get("remote_http_called"),
        http_status=stats.get("http_status"),
        raw_scored_count=stats.get("raw_scored_count"),
        fallback=stats.get("fallback"),
        rerank_host=host,
        model=(config.bocha_reranker_model or "").strip() or None,
        latency_ms=round(latency_ms, 2),
        error=(str(err)[:400] if err else None),
        response_snippet=snippet,
    )


class BochaReranker:
    def __init__(self) -> None:
        self.enabled = bool(config.bocha_reranker_url and config.bocha_api_key)

    def describe_config(self) -> dict[str, Any]:
        m = (config.bocha_reranker_model or "").strip()
        return {
            "configured": self.enabled,
            "has_url": bool(config.bocha_reranker_url),
            "has_api_key": bool(config.bocha_api_key),
            "model": m if m else None,
            "top_n_cap": config.bocha_top_n,
            "timeout_seconds": config.bocha_timeout_seconds,
        }

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
        out_stats: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        stats = out_stats if out_stats is not None else {}
        stats.clear()
        stats.update(
            {
                "step": "bocha_rerank",
                "remote_http_called": False,
                "candidates_in": len(candidates),
                "candidates_out": 0,
                "mode": "pending",
            }
        )
        t0 = time.perf_counter()
        try:
            if not candidates:
                stats.update({"mode": "skipped_empty_candidates", "candidates_out": 0})
                return []

            if not self.enabled:
                out = candidates[:top_n]
                stats.update(
                    {
                        "mode": "skipped_not_configured",
                        "reason": "需要同时配置 BOCHA_RERANKER_URL 与 BOCHA_API_KEY",
                        "candidates_out": len(out),
                        "fallback": "truncate_fusion_order",
                    }
                )
                return out

            doc_texts = [
                str(item.get("text") or item.get("text_preview") or "") for item in candidates
            ]
            payload: dict[str, Any] = {
                "query": query,
                "documents": doc_texts,
                "top_n": min(top_n, len(doc_texts)),
            }
            model = (config.bocha_reranker_model or "").strip()
            if model:
                payload["model"] = model
            try:
                stats["remote_http_called"] = True
                async with httpx.AsyncClient(timeout=config.bocha_timeout_seconds) as client:
                    response = await client.post(
                        config.bocha_reranker_url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {config.bocha_api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    response.raise_for_status()
                    scored = _extract_scored_items(response.json())
            except httpx.HTTPStatusError as exc:
                snippet = ""
                try:
                    snippet = (exc.response.text or "")[:800]
                except Exception:
                    pass
                out = candidates[:top_n]
                stats.update(
                    {
                        "mode": "remote_error_fallback",
                        "error": str(exc),
                        "http_status": exc.response.status_code,
                        "response_body_snippet": snippet or None,
                        "candidates_out": len(out),
                        "fallback": "truncate_fusion_order",
                    }
                )
                logger.warning(
                    "[Bocha] Rerank HTTP {}，使用融合序截断: {} | body[:200]={!r}",
                    exc.response.status_code,
                    exc,
                    snippet[:200],
                )
                return out
            except Exception as exc:
                out = candidates[:top_n]
                stats.update(
                    {
                        "mode": "remote_error_fallback",
                        "error": str(exc),
                        "candidates_out": len(out),
                        "fallback": "truncate_fusion_order",
                    }
                )
                logger.warning("[Bocha] Rerank 请求失败，使用融合序截断: {}", exc)
                return out

            by_id = {str(item["node_id"]): dict(item) for item in candidates}
            reranked: list[dict[str, Any]] = []
            for item in scored:
                merged_row: Optional[dict[str, Any]] = None
                idx_raw = item.get("index")
                if idx_raw is not None:
                    try:
                        idx = int(idx_raw)
                    except (TypeError, ValueError):
                        idx = -1
                    if 0 <= idx < len(candidates):
                        merged_row = dict(candidates[idx])
                if merged_row is None:
                    node_id = str(item.get("id") or item.get("document_id") or "")
                    if node_id in by_id:
                        merged_row = dict(by_id[node_id])
                if merged_row is None:
                    continue
                merged_row["rerank_score"] = _score_from_item(item)
                reranked.append(merged_row)
            if not reranked:
                out = candidates[:top_n]
                stats.update(
                    {
                        "mode": "remote_response_unmapped",
                        "reason": "响应中无与候选 node_id 匹配的条目",
                        "raw_scored_count": len(scored),
                        "candidates_out": len(out),
                        "fallback": "truncate_fusion_order",
                    }
                )
                return out

            out = reranked[:top_n]
            stats.update(
                {
                    "mode": "remote_success",
                    "candidates_out": len(out),
                    "raw_scored_count": len(scored),
                }
            )
            return out
        finally:
            _log_rerank_stage(stats, (time.perf_counter() - t0) * 1000)


reranker = BochaReranker()
