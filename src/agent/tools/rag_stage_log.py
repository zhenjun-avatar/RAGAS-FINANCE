"""Structured per-stage logs for the RAG pipeline (single JSON line per event)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator, Literal

from loguru import logger

_request_id: ContextVar[str | None] = ContextVar("rag_request_id", default=None)


def current_request_id() -> str | None:
    return _request_id.get()


@contextmanager
def rag_request_scope(request_id: str | None) -> Iterator[None]:
    token: Token | None = None
    if request_id:
        token = _request_id.set(request_id)
    try:
        yield
    finally:
        if token is not None:
            _request_id.reset(token)


def log_rag(
    stage: str,
    *,
    level: Literal["info", "warning", "error"] = "info",
    **fields: Any,
) -> None:
    payload: dict[str, Any] = {"stage": stage}
    rid = current_request_id()
    if rid:
        payload["request_id"] = rid
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    line = json.dumps(payload, ensure_ascii=False, default=str)
    if level == "warning":
        logger.warning("[RAG] {}", line)
    elif level == "error":
        logger.error("[RAG] {}", line)
    else:
        logger.info("[RAG] {}", line)
