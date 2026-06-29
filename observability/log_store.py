import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional
from uuid import uuid4


@dataclass
class TraceRecord:
    trace_id: str
    call_id: str
    generation_id: str
    started_at: float
    events: list[dict[str, Any]] = field(default_factory=list)
    latency: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class TraceLogStore:
    def __init__(self, max_traces: int = 500):
        self._traces: dict[str, TraceRecord] = {}
        self._order: deque[str] = deque(maxlen=max_traces)
        self._lock = Lock()

    def start_trace(self, call_id: str, generation_id: str) -> str:
        trace_id = f"tx_{uuid4().hex[:12]}"
        record = TraceRecord(
            trace_id=trace_id,
            call_id=call_id,
            generation_id=generation_id,
            started_at=time.time(),
        )
        with self._lock:
            self._traces[trace_id] = record
            self._order.append(trace_id)
            if len(self._order) > self._order.maxlen:
                old = self._order[0]
                self._traces.pop(old, None)
        return trace_id

    def add_event(self, trace_id: str, name: str, **attrs: Any) -> None:
        with self._lock:
            if trace_id not in self._traces:
                return
            self._traces[trace_id].events.append(
                {"name": name, "ts": time.time(), **attrs}
            )

    def set_latency(self, trace_id: str, **latencies: float) -> None:
        with self._lock:
            if trace_id in self._traces:
                self._traces[trace_id].latency.update(latencies)

    def finish_trace(self, trace_id: str, success: bool = True, error: Optional[str] = None) -> None:
        with self._lock:
            if trace_id in self._traces:
                self._traces[trace_id].success = success
                self._traces[trace_id].error = error

    def get(self, trace_id: str) -> Optional[dict]:
        with self._lock:
            rec = self._traces.get(trace_id)
            if not rec:
                return None
            return {
                "trace_id": rec.trace_id,
                "call_id": rec.call_id,
                "generation_id": rec.generation_id,
                "started_at": rec.started_at,
                "events": rec.events,
                "latency": rec.latency,
                "success": rec.success,
                "error": rec.error,
            }

    def list_recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            ids = list(self._order)[-limit:]
            return [
                {
                    "trace_id": self._traces[tid].trace_id,
                    "call_id": self._traces[tid].call_id,
                    "latency": self._traces[tid].latency,
                    "success": self._traces[tid].success,
                }
                for tid in reversed(ids)
                if tid in self._traces
            ]


trace_store = TraceLogStore()