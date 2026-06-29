import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from fastapi import WebSocket

from core.interrupt import InterruptController
from core.jitter_buffer import JitterBuffer

logger = logging.getLogger(__name__)


@dataclass
class CallSession:
    call_id: str
    user_id: str = "default"
    portfolio: str = ""
    websocket: Optional[WebSocket] = None
    interrupt: InterruptController = field(default_factory=InterruptController)
    jitter_buffer: JitterBuffer = field(default_factory=JitterBuffer)
    audio_accumulator: bytearray = field(default_factory=bytearray)
    active_tasks: set[asyncio.Task] = field(default_factory=set)
    phase: str = "IDLE"
    last_trace_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def new_generation(self) -> str:
        for task in list(self.active_tasks):
            task.cancel()
        self.active_tasks.clear()
        self.audio_accumulator.clear()
        return self.interrupt.new_generation()

    def track_task(self, task: asyncio.Task) -> None:
        self.active_tasks.add(task)
        task.add_done_callback(lambda t: self.active_tasks.discard(t))


class SessionStore(ABC):
    @abstractmethod
    async def create(self, call_id: str, session: CallSession) -> None: ...

    @abstractmethod
    async def get(self, call_id: str) -> Optional[CallSession]: ...

    @abstractmethod
    async def delete(self, call_id: str) -> None: ...

    @abstractmethod
    async def update_phase(self, call_id: str, phase: str) -> None: ...


class InMemorySessionStore(SessionStore):
    def __init__(self):
        self._sessions: dict[str, CallSession] = {}

    async def create(self, call_id: str, session: CallSession) -> None:
        self._sessions[call_id] = session

    async def get(self, call_id: str) -> Optional[CallSession]:
        return self._sessions.get(call_id)

    async def delete(self, call_id: str) -> None:
        self._sessions.pop(call_id, None)

    async def update_phase(self, call_id: str, phase: str) -> None:
        if call_id in self._sessions:
            self._sessions[call_id].phase = phase


class RedisSessionStore(SessionStore):
    """Production swap-in: horizontal scaling + sticky-session LB."""

    def __init__(self, redis_url: str):
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._local_cache: dict[str, CallSession] = {}

    def _key(self, call_id: str) -> str:
        return f"voice:session:{call_id}"

    async def create(self, call_id: str, session: CallSession) -> None:
        self._local_cache[call_id] = session
        meta = {
            "call_id": call_id,
            "user_id": session.user_id,
            "portfolio": session.portfolio,
            "phase": session.phase,
            "generation_id": session.interrupt.generation_id,
        }
        await self._redis.set(self._key(call_id), json.dumps(meta), ex=3600)

    async def get(self, call_id: str) -> Optional[CallSession]:
        if call_id in self._local_cache:
            return self._local_cache[call_id]
        raw = await self._redis.get(self._key(call_id))
        if not raw:
            return None
        meta = json.loads(raw)
        session = CallSession(
            call_id=meta["call_id"],
            user_id=meta.get("user_id", "default"),
            portfolio=meta.get("portfolio", ""),
        )
        session.phase = meta.get("phase", "IDLE")
        session.interrupt.generation_id = meta.get("generation_id", str(uuid4()))
        self._local_cache[call_id] = session
        return session

    async def delete(self, call_id: str) -> None:
        self._local_cache.pop(call_id, None)
        await self._redis.delete(self._key(call_id))

    async def update_phase(self, call_id: str, phase: str) -> None:
        session = await self.get(call_id)
        if session:
            session.phase = phase
            await self.create(call_id, session)


class SessionManager:
    def __init__(self, store: Optional[SessionStore] = None):
        store_type = os.getenv("SESSION_STORE", "memory").lower()
        if store:
            self.store = store
        elif store_type == "redis" and os.getenv("REDIS_URL"):
            self.store = RedisSessionStore(os.getenv("REDIS_URL"))
            logger.info("SessionManager using Redis-backed store")
        else:
            self.store = InMemorySessionStore()
            logger.info("SessionManager using in-memory store (swap to Redis for horizontal scaling)")

    async def create_session(
        self,
        ws: WebSocket,
        portfolio: str,
        user_id: str = "default",
    ) -> CallSession:
        call_id = str(uuid4())
        session = CallSession(
            call_id=call_id,
            user_id=user_id,
            portfolio=portfolio,
            websocket=ws,
        )
        await self.store.create(call_id, session)
        return session

    async def get(self, call_id: str) -> Optional[CallSession]:
        return await self.store.get(call_id)

    async def set_phase(self, session: CallSession, phase: str, from_phase: Optional[str] = None) -> None:
        old = session.phase
        session.phase = phase
        await self.store.update_phase(session.call_id, phase)
        if session.websocket:
            from core.events import EventType, make_event

            await session.websocket.send_json(
                make_event(
                    EventType.STATE_TRANSITION,
                    session.interrupt.generation_id,
                    session.call_id,
                    from_state=from_phase or old,
                    to_state=phase,
                )
            )

    async def destroy(self, call_id: str) -> None:
        session = await self.store.get(call_id)
        if session:
            session.new_generation()
        await self.store.delete(call_id)


session_manager = SessionManager()