"""LlamaIndex callback handler that records stage IO and latency."""

from __future__ import annotations

import time
from typing import Any, Optional

from llama_index.core.callbacks import CBEventType
from llama_index.core.callbacks.base_handler import BaseCallbackHandler


class RecordingCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._start_times: dict[str, float] = {}
        self.events: list[dict[str, Any]] = []

    def start_trace(self, trace_id: Optional[str] = None) -> None:  # pragma: no cover
        return None

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[dict[str, list[str]]] = None,
    ) -> None:  # pragma: no cover
        return None

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        self._start_times[event_id] = time.perf_counter()
        self.events.append(
            {
                "phase": "start",
                "event_id": event_id,
                "parent_id": parent_id,
                "event_type": str(event_type),
                "payload": payload or {},
            }
        )
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        start = self._start_times.pop(event_id, None)
        latency_ms = (time.perf_counter() - start) * 1000 if start else None
        self.events.append(
            {
                "phase": "end",
                "event_id": event_id,
                "event_type": str(event_type),
                "latency_ms": latency_ms,
                "payload": payload or {},
            }
        )

    def flush_events(self) -> list[dict[str, Any]]:
        snapshot = list(self.events)
        self.events.clear()
        self._start_times.clear()
        return snapshot
