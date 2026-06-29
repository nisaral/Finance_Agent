import asyncio
import base64
import logging
from typing import Callable, Optional

from core.config_loader import load_agent_config
from core.events import EventType, make_event
from core.session import CallSession
from graph.agent_graph import run_agent
from graph.nodes.language import _build_prompt, extract_speakable_chunk
from graph.state import AgentState
from observability.tracing import UtteranceTracer, timed_span
from providers.factory import build_llm_chain, build_tts_chain

logger = logging.getLogger(__name__)


async def run_agent_stream(
    session: CallSession,
    query: str,
    generation_id: str,
    send: Callable,
) -> None:
    config = load_agent_config()
    tracer = UtteranceTracer(session.call_id, generation_id)
    session.last_trace_id = tracer.trace_id
    cancel = session.interrupt.cancel_event

    try:
        await send(make_event(
            EventType.TRACE_INFO,
            generation_id,
            session.call_id,
            trace_id=tracer.trace_id,
        ))

        initial: AgentState = {
            "call_id": session.call_id,
            "generation_id": generation_id,
            "query": query,
            "portfolio": session.portfolio,
            "user_id": session.user_id,
            "phase": "IDLE",
            "errors": [],
            "charts": [],
            "tone_analysis": {},
        }

        with timed_span(tracer, "graph"):
            result = await run_agent(initial, thread_id=session.call_id)
            if cancel.is_set():
                return

        phase = result.get("phase", "RESPONDING")
        await session_manager_set_phase(session, phase)

        if result.get("charts"):
            await send(make_event(
                EventType.CHART_RENDER,
                generation_id,
                session.call_id,
                config=result["charts"][0],
            ))

        if result.get("safe_mode"):
            await send(make_event(
                EventType.ERROR,
                generation_id,
                session.call_id,
                code="NEWS_DEGRADED",
                message=config.safe_mode.news_fallback_message,
                safe_mode=True,
            ))

        prompt = _build_prompt(result)
        llm_chain = build_llm_chain()
        tts_chain = build_tts_chain()

        buffer = ""
        tts_started = False

        tracer.start_span("llm")
        first_token = True
        llm_provider = type(llm_chain.providers[0]).__name__ if llm_chain.providers else "unknown"
        async for token in _stream_llm_with_fallback(llm_chain, prompt):
                if cancel.is_set() or not session.interrupt.check(generation_id):
                    return
                if first_token:
                    tracer.end_span("llm", provider=llm_provider)
                    first_token = False
                buffer += token
                await send(make_event(EventType.LLM_TOKEN, generation_id, session.call_id, text=token))

                chunk, buffer = extract_speakable_chunk(
                    buffer, min_words=config.latency.tts_sentence_min_words
                )
                if chunk:
                    if not tts_started:
                        tracer.start_span("tts")
                        tts_started = True
                    async for audio in await_tts_chunks(tts_chain, chunk, cancel):
                        if cancel.is_set():
                            return
                        await send(make_event(
                            EventType.TTS_CHUNK,
                            generation_id,
                            session.call_id,
                            data=base64.b64encode(audio).decode(),
                            encoding="pcm_s16le",
                        ))

        if first_token:
            tracer.end_span("llm", provider=llm_provider, empty=True)

        if buffer.strip() and not cancel.is_set():
            if not tts_started:
                tracer.start_span("tts")
            async for audio in await_tts_chunks(tts_chain, buffer.strip(), cancel):
                if cancel.is_set():
                    return
                await send(make_event(
                    EventType.TTS_CHUNK,
                    generation_id,
                    session.call_id,
                    data=base64.b64encode(audio).decode(),
                    encoding="pcm_s16le",
                ))

        if tts_started:
            tracer.end_span("tts", provider=tts_chain.last_provider)

        summary = tracer.finish(success=True)
        await send(make_event(
            EventType.LATENCY_BREAKDOWN,
            generation_id,
            session.call_id,
            **summary,
        ))
        await send(make_event(
            EventType.RESPONSE_COMPLETE,
            generation_id,
            session.call_id,
            trace_id=tracer.trace_id,
        ))

    except asyncio.CancelledError:
        tracer.finish(success=False, error="cancelled")
        raise
    except Exception as e:
        logger.exception("Stream pipeline error")
        tracer.finish(success=False, error=str(e))
        await send(make_event(
            EventType.ERROR,
            generation_id,
            session.call_id,
            code="PIPELINE_ERROR",
            message=str(e),
        ))


async def _stream_llm_with_fallback(llm_chain, prompt: str):
    last_error = None
    for provider in llm_chain.providers:
        try:
            async for token in provider.stream(prompt):
                yield token
            return
        except Exception as e:
            last_error = e
            logger.warning(f"LLM provider {type(provider).__name__} failed: {e}")
    if last_error:
        raise last_error


async def await_tts_chunks(tts_chain, text: str, cancel: asyncio.Event):
    stream, _ = tts_chain.get_stream("synthesize_stream", text)
    async for chunk in stream:
        if cancel.is_set():
            break
        yield chunk


async def session_manager_set_phase(session: CallSession, phase: str) -> None:
    from core.session import session_manager
    await session_manager.set_phase(session, phase)