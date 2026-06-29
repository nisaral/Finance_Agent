import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class AudioPacket:
    seq: int
    data: bytes
    received_at: float = field(default_factory=time.monotonic)


class JitterBuffer:
    """Re-order inbound audio packets before STT ingestion."""

    def __init__(self, window_ms: int = 75):
        self.window_ms = window_ms
        self._packets: dict[int, AudioPacket] = {}
        self._next_expected = 0
        self._lock = asyncio.Lock()

    async def push(self, seq: int, data: bytes) -> list[bytes]:
        async with self._lock:
            self._packets[seq] = AudioPacket(seq=seq, data=data)
            if self.window_ms > 0:
                await asyncio.sleep(self.window_ms / 1000.0)
            return await self._drain_in_order()

    async def flush(self) -> list[bytes]:
        async with self._lock:
            in_order = await self._drain_in_order()
            remaining = [self._packets[s].data for s in sorted(self._packets)]
            self._packets.clear()
            self._next_expected = 0
            return in_order + remaining

    async def _drain_in_order(self) -> list[bytes]:
        ready: list[bytes] = []
        while self._next_expected in self._packets:
            pkt = self._packets.pop(self._next_expected)
            ready.append(pkt.data)
            self._next_expected += 1
        return ready