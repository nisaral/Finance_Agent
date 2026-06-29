import asyncio

import pytest

from core.config_loader import load_agent_config, reload_config
from core.events import EventType, make_event
from core.interrupt import InterruptController
from core.jitter_buffer import JitterBuffer
from core.session import CallSession, InMemorySessionStore, SessionManager


@pytest.mark.asyncio
async def test_jitter_buffer_reorders_packets():
    buf = JitterBuffer(window_ms=10)
    await buf.push(1, b"aaa")
    ready = await buf.push(0, b"bbb")
    assert b"".join(ready) == b"bbbaaa"
    assert await buf.flush() == []


def test_interrupt_controller_new_generation():
    ctrl = InterruptController()
    old = ctrl.generation_id
    new = ctrl.new_generation()
    assert new != old
    assert ctrl.is_cancelled() is False
    ctrl.cancel_all()
    assert ctrl.is_cancelled() is True


def test_config_loader():
    reload_config()
    cfg = load_agent_config()
    assert cfg.name == "Global Market Assistant"
    assert cfg.safe_mode.enabled is True
    assert cfg.latency.jitter_buffer_ms == 75


def test_event_envelope():
    ev = make_event(EventType.STT_FINAL, "gen-1", "call-1", text="hello")
    assert ev["type"] == "stt.final"
    assert ev["generation_id"] == "gen-1"
    assert ev["text"] == "hello"


@pytest.mark.asyncio
async def test_in_memory_session_store():
    store = InMemorySessionStore()
    session = CallSession(call_id="c1", portfolio="AAPL:10")
    await store.create("c1", session)
    got = await store.get("c1")
    assert got is not None
    assert got.portfolio == "AAPL:10"
    await store.update_phase("c1", "ANALYZING")
    assert (await store.get("c1")).phase == "ANALYZING"
    await store.delete("c1")
    assert await store.get("c1") is None


@pytest.mark.asyncio
async def test_session_manager_create():
    mgr = SessionManager(store=InMemorySessionStore())
    session = await mgr.create_session(ws=None, portfolio="MSFT:5", user_id="test")
    assert session.call_id
    assert session.portfolio == "MSFT:5"
    await mgr.destroy(session.call_id)