import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from observability.log_store import trace_store

logger = logging.getLogger(__name__)

_tracer = None
_meter = None


def init_telemetry(app_name: str = "finance-agent") -> None:
    global _tracer, _meter
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", app_name)})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        metrics.set_meter_provider(MeterProvider(resource=resource))
        _tracer = trace.get_tracer(app_name)
        _meter = metrics.get_meter(app_name)
        logger.info("OpenTelemetry initialized")
    except ImportError:
        logger.warning("OpenTelemetry packages not installed; using in-memory trace store only")
        _tracer = None
        _meter = None


def get_tracer():
    return _tracer


class UtteranceTracer:
    def __init__(self, call_id: str, generation_id: str):
        self.call_id = call_id
        self.generation_id = generation_id
        self.trace_id = trace_store.start_trace(call_id, generation_id)
        self._timers: dict[str, float] = {}
        self.latency: dict[str, float] = {}

    def start_span(self, name: str) -> None:
        self._timers[name] = time.perf_counter()
        trace_store.add_event(self.trace_id, name, status="start")

    def end_span(self, name: str, **attrs: Any) -> float:
        start = self._timers.pop(name, time.perf_counter())
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.latency[name] = elapsed_ms
        trace_store.add_event(self.trace_id, name, status="end", duration_ms=elapsed_ms, **attrs)
        if _tracer:
            with _tracer.start_as_current_span(name) as span:
                span.set_attribute("duration_ms", elapsed_ms)
                for k, v in attrs.items():
                    span.set_attribute(k, v)
        return elapsed_ms

    def finish(self, success: bool = True, error: Optional[str] = None) -> dict[str, Any]:
        trace_store.set_latency(
            self.trace_id,
            stt_ms=self.latency.get("stt", 0),
            llm_ms=self.latency.get("llm", 0),
            tts_ms=self.latency.get("tts", 0),
            total_ms=sum(self.latency.values()),
        )
        trace_store.finish_trace(self.trace_id, success=success, error=error)
        return {
            "trace_id": self.trace_id,
            "latency": {
                "stt_ms": round(self.latency.get("stt", 0)),
                "llm_ms": round(self.latency.get("llm", 0)),
                "tts_ms": round(self.latency.get("tts", 0)),
                "total_ms": round(sum(self.latency.values())),
            },
        }


@contextmanager
def timed_span(tracer: UtteranceTracer, name: str, **attrs: Any) -> Generator[None, None, None]:
    tracer.start_span(name)
    try:
        yield
    finally:
        tracer.end_span(name, **attrs)