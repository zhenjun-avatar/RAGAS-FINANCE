"""Configure loguru: stderr + optional file, single-line readable format."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from core.config import config

_AGENT_ROOT = Path(__file__).resolve().parents[1]
_configured = False


def configure_runtime_logging() -> None:
    """Idempotent. Call once at process entry (e.g. api.server) before other tools log."""
    global _configured
    if _configured:
        return
    _configured = True

    logger.remove()
    fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}"
    level = (config.log_level or "INFO").upper()

    logger.add(sys.stderr, format=fmt, level=level, enqueue=True)

    raw = (config.log_file or "").strip()
    if raw.lower() in ("", "-", "none", "false", "0"):
        raw = ""
    if raw:
        log_path = Path(raw)
        if not log_path.is_absolute():
            log_path = _AGENT_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format=fmt,
            level=level,
            rotation="20 MB",
            retention=5,
            encoding="utf-8",
            enqueue=True,
        )
