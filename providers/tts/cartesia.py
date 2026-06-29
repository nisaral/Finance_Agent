import base64
import json
import logging
import os
import uuid
from typing import AsyncIterator

import aiohttp

from providers.base import TTSProvider

logger = logging.getLogger(__name__)


class CartesiaTTS(TTSProvider):
    def __init__(self, model: str = "sonic-3", voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091"):
        self.api_key = os.getenv("CARTESIA_API_KEY", "")
        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY required")
        self.model = model
        self.voice_id = voice_id
        self.cartesia_version = "2026-03-01"
        self.ws_url = f"wss://api.cartesia.ai/tts/websocket?cartesia_version={self.cartesia_version}"

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        context_id = str(uuid.uuid4())
        headers = {
            "Cartesia-Version": self.cartesia_version,
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model_id": self.model,
            "transcript": text,
            "voice": {"mode": "id", "id": self.voice_id},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
            "context_id": context_id,
            "continue": False,
            "max_buffer_delay_ms": 100,
        }

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.ws_url, headers=headers) as ws:
                await ws.send_str(json.dumps(payload))
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data = json.loads(msg.data)
                    if data.get("type") == "chunk" and data.get("data"):
                        yield base64.b64decode(data["data"])
                    elif data.get("type") == "done" or data.get("done"):
                        break
                    elif data.get("type") == "error":
                        raise RuntimeError(data.get("message", "Cartesia TTS error"))