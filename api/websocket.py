import asyncio
import base64
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.config_loader import load_agent_config
from core.events import EventType, make_event
from core.session import session_manager
from graph.streaming_pipeline import run_agent_stream
from observability.tracing import UtteranceTracer, timed_span
from providers.factory import build_stt_chain

logger = logging.getLogger(__name__)
router = APIRouter()


async def _send(ws: WebSocket, event: dict) -> None:
    await ws.send_json(event)


@router.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket):
    await ws.accept()
    session = None

    try:
        while True:
            raw = await ws.receive_json()
            msg_type = raw.get("type", "")

            if msg_type == "session.start":
                portfolio = raw.get("portfolio", "")
                user_id = raw.get("user_id", "default")
                session = await session_manager.create_session(ws, portfolio, user_id)
                config = load_agent_config()
                from core.jitter_buffer import JitterBuffer
                session.jitter_buffer = JitterBuffer(window_ms=config.latency.jitter_buffer_ms)

                await _send(ws, make_event(
                    EventType.SESSION_READY,
                    session.interrupt.generation_id,
                    session.call_id,
                    agent_name=config.name,
                ))

            elif msg_type == "audio.chunk" and session:
                if session.interrupt.is_cancelled():
                    continue
                seq = raw.get("seq", 0)
                data = base64.b64decode(raw.get("data", ""))
                ready_chunks = await session.jitter_buffer.push(seq, data)
                for chunk in ready_chunks:
                    session.audio_accumulator.extend(chunk)

            elif msg_type == "utterance.end" and session:
                gen_id = session.interrupt.generation_id
                tracer = UtteranceTracer(session.call_id, gen_id)
                session.last_trace_id = tracer.trace_id

                audio = bytes(session.audio_accumulator)
                session.audio_accumulator.clear()
                remaining = await session.jitter_buffer.flush()
                for c in remaining:
                    audio += c

                query = raw.get("text", "")
                if audio and not query:
                    with timed_span(tracer, "stt"):
                        try:
                            stt_chain = build_stt_chain()
                            query, provider = await stt_chain.execute("transcribe", audio)
                            tracer.end_span("stt", provider=provider)
                        except Exception as e:
                            logger.error(f"STT failed: {e}")
                            query = "Provide an update on my portfolio."
                            tracer.end_span("stt", error=str(e))

                if not query:
                    query = "Provide an update on my portfolio."

                await _send(ws, make_event(EventType.STT_FINAL, gen_id, session.call_id, text=query))
                await _send(ws, make_event(EventType.TRACE_INFO, gen_id, session.call_id, trace_id=tracer.trace_id))

                task = asyncio.create_task(
                    run_agent_stream(session, query, gen_id, lambda e: _send(ws, e))
                )
                session.track_task(task)

            elif msg_type == "text.query" and session:
                gen_id = session.new_generation()
                query = raw.get("text", "")
                task = asyncio.create_task(
                    run_agent_stream(session, query, gen_id, lambda e: _send(ws, e))
                )
                session.track_task(task)

            elif msg_type == "interrupt" and session:
                old_gen = raw.get("generation_id", session.interrupt.generation_id)
                new_gen = session.new_generation()
                await _send(ws, make_event(
                    EventType.INTERRUPT_ACK,
                    new_gen,
                    session.call_id,
                    cancelled_generation=old_gen,
                ))
                await session_manager.set_phase(session, "INTERRUPTED")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        if session:
            session.new_generation()
            await session_manager.destroy(session.call_id)