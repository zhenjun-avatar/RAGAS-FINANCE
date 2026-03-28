"""Langfuse tracing helpers with no-op fallback."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from core.config import config

try:
    from langfuse import Langfuse
except ImportError:  # pragma: no cover
    Langfuse = None  # type: ignore[misc,assignment]


def _is_enabled() -> bool:
    return bool(
        config.langfuse_enabled
        and config.langfuse_public_key
        and config.langfuse_secret_key
        and config.effective_langfuse_base_url
        and Langfuse is not None
    )


@dataclass
class TraceContext:
    trace_id: str
    root_observation: Any
    enabled: bool


class LangfuseTracer:
    """使用 core.config 显式构造 Langfuse，与仅依赖 get_client() 读环境变量 的行为对齐，避免「配置了 URL 但 SDK 读到别的值」。"""

    def __init__(self) -> None:
        self.enabled = _is_enabled()
        self.client: Any = None
        self._export_base_url: Optional[str] = None
        self._last_flush_error: Optional[str] = None

        if not self.enabled:
            return

        if os.environ.get("OTEL_SDK_DISABLED", "false").lower() == "true":
            logger.warning(
                "[Langfuse] 已设置 OTEL_SDK_DISABLED=true，OpenTelemetry 导出被禁用，"
                "Langfuse UI 将收不到 trace。请取消该环境变量或设为 false。"
            )

        base = (config.effective_langfuse_base_url or "").rstrip("/")
        try:
            self.client = Langfuse(  # type: ignore[misc]
                public_key=config.langfuse_public_key,
                secret_key=config.langfuse_secret_key,
                base_url=base or "https://cloud.langfuse.com",
            )
            self._export_base_url = base or "https://cloud.langfuse.com"
            logger.info(
                "[Langfuse] 客户端已初始化，导出 base_url={}（请在与该 URL 一致的 Langfuse 实例中查看 Traces）",
                self._export_base_url,
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("[Langfuse] 客户端初始化失败，将关闭追踪: {}", exc)
            self.enabled = False
            self.client = None
            self._export_base_url = None

    @property
    def export_base_url(self) -> Optional[str]:
        return self._export_base_url

    @property
    def last_flush_error(self) -> Optional[str]:
        return self._last_flush_error

    def diagnostics(self) -> dict[str, Any]:
        return {
            "langfuse_enabled_flag": config.langfuse_enabled,
            "langfuse_client_active": bool(self.enabled and self.client is not None),
            "export_base_url": self._export_base_url,
            "has_public_key": bool(config.langfuse_public_key),
            "has_secret_key": bool(config.langfuse_secret_key),
            "otel_sdk_disabled": os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true",
            "last_flush_error": self._last_flush_error,
        }

    def start_request(self, name: str, *, input_payload: dict[str, Any]) -> TraceContext:
        if not self.enabled or self.client is None:
            return TraceContext(trace_id="", root_observation=nullcontext(), enabled=False)
        try:
            observation = self.client.start_as_current_observation(
                as_type="span",
                name=name,
                input=input_payload,
            )
            current = observation.__enter__()
            trace_id = getattr(current, "trace_id", "") or ""
            logger.debug("[Langfuse] trace 开始 trace_id={}", trace_id)
            return TraceContext(
                trace_id=trace_id,
                root_observation=(observation, current),
                enabled=True,
            )
        except Exception as exc:
            logger.exception("[Langfuse] start_request 失败，本请求不产生 trace: {}", exc)
            return TraceContext(trace_id="", root_observation=nullcontext(), enabled=False)

    def end_request(
        self,
        ctx: TraceContext,
        *,
        output_payload: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if not ctx.enabled:
            return
        observation_cm, current = ctx.root_observation
        self._last_flush_error = None
        try:
            final_metadata = dict(metadata or {})
            if error:
                final_metadata["error"] = error
            current.update(output=output_payload, metadata=final_metadata)
        except Exception as exc:
            self._last_flush_error = f"update: {exc}"
            logger.exception("[Langfuse] end_request update 失败: {}", exc)
        finally:
            try:
                observation_cm.__exit__(None, None, None)
            except Exception as exc:
                logger.exception("[Langfuse] observation 关闭失败: {}", exc)
            try:
                self.client.flush()  # type: ignore[union-attr]
            except Exception as exc:
                self._last_flush_error = f"flush: {exc}"
                logger.exception(
                    "[Langfuse] flush 失败，数据可能未到达服务端（请检查网络、防火墙与 base_url）: {}",
                    exc,
                )
            else:
                if ctx.trace_id:
                    logger.debug(
                        "[Langfuse] trace flush OK trace_id={} base_url={}",
                        ctx.trace_id,
                        self._export_base_url,
                    )

    def span(self, name: str, *, input_payload: Optional[dict[str, Any]] = None):
        if not self.enabled or self.client is None:
            return nullcontext()
        return self.client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_payload or {},
        )

    def generation(self, name: str, *, input_payload: Optional[dict[str, Any]] = None, model: Optional[str] = None):
        if not self.enabled or self.client is None:
            return nullcontext()
        kwargs: dict[str, Any] = {"as_type": "generation", "name": name, "input": input_payload or {}}
        if model:
            kwargs["model"] = model
        return self.client.start_as_current_observation(**kwargs)

    def score_trace(self, trace_id: str, *, name: str, value: float, comment: Optional[str] = None) -> None:
        if not self.enabled or not trace_id or self.client is None:
            return
        try:
            self.client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("[Langfuse] create_score 失败 {}: {}", name, exc)


tracer = LangfuseTracer()
