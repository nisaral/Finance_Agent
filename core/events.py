from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    SESSION_START = "session.start"
    SESSION_READY = "session.ready"
    AUDIO_CHUNK = "audio.chunk"
    UTTERANCE_END = "utterance.end"
    TEXT_QUERY = "text.query"
    INTERRUPT = "interrupt"
    INTERRUPT_ACK = "interrupt.ack"
    STT_PARTIAL = "stt.partial"
    STT_FINAL = "stt.final"
    STATE_TRANSITION = "state.transition"
    LLM_TOKEN = "llm.token"
    TTS_CHUNK = "tts.chunk"
    CHART_RENDER = "chart.render"
    LATENCY_BREAKDOWN = "latency.breakdown"
    TRACE_INFO = "trace.info"
    ERROR = "error"
    RESPONSE_COMPLETE = "response.complete"


def make_event(
    event_type: EventType | str,
    generation_id: str,
    call_id: str,
    **payload: Any,
) -> dict[str, Any]:
    return {
        "type": event_type if isinstance(event_type, str) else event_type.value,
        "call_id": call_id,
        "generation_id": generation_id,
        **payload,
    }


def parse_event(raw: dict) -> tuple[str, dict]:
    return raw.get("type", ""), raw