from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from loguru import logger

from core.config import config

DEFAULT_WEB_SEARCH_URL = "https://api.bochaai.com/v1/web-search"
DEFAULT_AI_SEARCH_URL = "https://api.bochaai.com/v1/ai-search"


def _pick_first(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _normalize_result_item(item: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a single search source item into a common shape.
    Bocha payloads can vary; we keep it best-effort.
    """

    nested_source = item.get("source") if isinstance(item.get("source"), dict) else {}
    nested_page = item.get("page") if isinstance(item.get("page"), dict) else {}
    nested_doc = item.get("document") if isinstance(item.get("document"), dict) else {}

    title = _pick_first(item, ["title", "page_title", "name", "headline"])
    if title is None:
        title = _pick_first(nested_source, ["title", "name", "headline"])
    if title is None:
        title = _pick_first(nested_page, ["title", "name"])

    url = _pick_first(item, ["url", "link", "page_url", "source_url", "href"])
    if url is None:
        url = _pick_first(nested_source, ["url", "link", "href"])
    if url is None:
        url = _pick_first(nested_page, ["url", "link", "href"])
    if url is None:
        url = _pick_first(nested_doc, ["url", "link", "href"])

    snippet = _pick_first(item, ["snippet", "summary", "description", "text", "content"])
    if snippet is None:
        snippet = _pick_first(nested_source, ["snippet", "summary", "description", "text"])
    if snippet is None:
        snippet = _pick_first(nested_doc, ["snippet", "summary", "description", "text"])
    published_at = _pick_first(item, ["published_at", "date", "publish_time", "time"])
    site = _pick_first(item, ["site", "domain", "source", "host"])

    return {
        "title": (str(title).strip() if title is not None else None),
        "url": (str(url).strip() if url is not None else None),
        "snippet": (str(snippet).strip() if snippet is not None else None),
        "published_at": published_at if published_at is not None else None,
        "site": (str(site).strip() if site is not None else None),
        # keep original for debugging / future improvements
        "_raw": item,
    }


def _extract_items_from_any(data: Any) -> list[dict[str, Any]]:
    """
    Best-effort extraction of a list[dict] from unknown Bocha response schema.
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []

    for key in ("data", "results", "items", "web_results", "sources", "references"):
        v = data.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        if isinstance(v, dict):
            nested = _extract_items_from_any(v)
            if nested:
                return nested

    # common nested containers in web/ai search payloads
    for key in ("webPages", "pages"):
        v = data.get(key)
        if isinstance(v, dict):
            # e.g. {"webPages":{"value":[...]}}
            vv = v.get("value")
            if isinstance(vv, list):
                rows = [x for x in vv if isinstance(x, dict)]
                if rows:
                    return rows
            nested = _extract_items_from_any(v)
            if nested:
                return nested
        if isinstance(v, list):
            rows = [x for x in v if isinstance(x, dict)]
            if rows:
                return rows

    # last resort: scan first-level list values
    for _, v in data.items():
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return v
    # recursive fallback for nested dict values
    for _, v in data.items():
        if isinstance(v, dict):
            nested = _extract_items_from_any(v)
            if nested:
                return nested

    return []


class BochaSearchClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        web_search_url: str | None = None,
        ai_search_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = api_key or config.bocha_api_key or os.getenv("BOCHA_API_KEY", "")
        self.web_search_url = web_search_url or os.getenv(
            "BOCHA_WEB_SEARCH_URL", DEFAULT_WEB_SEARCH_URL
        )
        self.ai_search_url = ai_search_url or os.getenv(
            "BOCHA_AI_SEARCH_URL", DEFAULT_AI_SEARCH_URL
        )
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else float(config.bocha_timeout_seconds)
        )

    def enabled(self) -> bool:
        return bool((self.api_key or "").strip())

    async def web_search(
        self,
        *,
        query: str,
        freshness: str = "oneYear",
        summary: bool = True,
        count: int = 10,
        include: Optional[str] = None,
        extra_payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not self.enabled():
            return {"ok": False, "error": "BOCHA_API_KEY not configured"}

        stats: dict[str, Any] = {
            "mode": "web-search",
            "remote_http_called": False,
            "include_retry": False,
        }

        async def _do(include_value: Optional[str]) -> tuple[dict[str, Any], int]:
            payload: dict[str, Any] = {
                "query": query,
                "freshness": freshness,
                "summary": summary,
                "count": max(1, int(count)),
            }
            if include_value:
                payload["include"] = include_value
            if extra_payload:
                payload.update(extra_payload)

            t0 = None
            data: dict[str, Any] = {}
            try:
                import time

                t0 = time.perf_counter()
                stats["remote_http_called"] = True
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    resp = await client.post(
                        self.web_search_url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as exc:
                raise RuntimeError(str(exc)) from exc
            finally:
                if t0 is not None:
                    try:
                        import time

                        stats["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    except Exception:
                        pass

            items = _extract_items_from_any(data)
            normalized = [_normalize_result_item(x) for x in items]
            return {"ok": True, "results": normalized, "raw_count": len(items)}, len(items)

        include_used = include
        try:
            raw, raw_count = await _do(include_used)
        except Exception as exc:
            logger.warning("[Bocha:web_search] failed: {}", exc)
            return {"ok": False, "error": str(exc), "stats": stats}

        # Some Bocha deployments accept domains delimiter as ',' instead of '|'.
        if raw_count == 0 and include_used and "|" in include_used and "," not in include_used:
            alt = include_used.replace("|", ",")
            try:
                raw_alt, raw_count_alt = await _do(alt)
                if raw_alt.get("results") is not None and raw_count_alt > 0:
                    raw = raw_alt
                    stats["include_retry"] = True
            except Exception:
                pass

        normalized = raw.get("results") or []
        return {
            "ok": True,
            "mode": "web-search",
            "query": query,
            "count": len(normalized),
            "results": normalized,
            "stats": stats,
        }

    async def ai_search(
        self,
        *,
        query: str,
        freshness: str = "oneYear",
        include: Optional[str] = None,
        count: int = 10,
        answer: bool = True,
        stream: bool = False,
        extra_payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not self.enabled():
            return {"ok": False, "error": "BOCHA_API_KEY not configured"}

        stats: dict[str, Any] = {
            "mode": "ai-search",
            "remote_http_called": False,
            "include_retry": False,
        }

        async def _do(include_value: Optional[str]) -> tuple[dict[str, Any], int]:
            payload: dict[str, Any] = {
                "query": query,
                "freshness": freshness,
                "count": max(1, int(count)),
                "answer": bool(answer),
                "stream": bool(stream),
            }
            if include_value:
                payload["include"] = include_value
            if extra_payload:
                payload.update(extra_payload)

            t0 = None
            data: dict[str, Any] = {}
            try:
                import time

                t0 = time.perf_counter()
                stats["remote_http_called"] = True
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    resp = await client.post(
                        self.ai_search_url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as exc:
                raise RuntimeError(str(exc)) from exc
            finally:
                if t0 is not None:
                    try:
                        import time

                        stats["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    except Exception:
                        pass

            items = _extract_items_from_any(data)
            normalized = [_normalize_result_item(x) for x in items]
            ai_answer = None
            if isinstance(data, dict):
                ai_answer = data.get("answer") or data.get("summary") or data.get("result")
            return {"ok": True, "results": normalized, "count_raw": len(items), "ai_answer": ai_answer}, len(items)

        include_used = include
        try:
            raw, raw_count = await _do(include_used)
        except Exception as exc:
            logger.warning("[Bocha:ai_search] failed: {}", exc)
            return {"ok": False, "error": str(exc), "stats": stats}

        if raw_count == 0 and include_used and "|" in include_used and "," not in include_used:
            alt = include_used.replace("|", ",")
            try:
                raw_alt, raw_count_alt = await _do(alt)
                if raw_alt.get("results") is not None and raw_count_alt > 0:
                    raw = raw_alt
                    stats["include_retry"] = True
            except Exception:
                pass

        normalized = raw.get("results") or []
        ai_answer = raw.get("ai_answer")

        return {
            "ok": True,
            "mode": "ai-search",
            "query": query,
            "count": len(normalized),
            "results": normalized,
            "ai_answer": ai_answer,
            "stats": stats,
        }

