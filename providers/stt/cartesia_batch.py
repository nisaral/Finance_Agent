import logging
import os

import aiohttp

from providers.base import STTProvider

logger = logging.getLogger(__name__)
CARTESIA_VERSION = "2026-03-01"


class CartesiaBatchSTT(STTProvider):
    def __init__(self, model: str = "ink-whisper"):
        self.api_key = os.getenv("CARTESIA_API_KEY", "")
        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY required")
        self.model = model
        self.url = "https://api.cartesia.ai/stt"

    async def transcribe(self, audio: bytes) -> str:
        headers = {
            "Cartesia-Version": CARTESIA_VERSION,
            "Authorization": f"Bearer {self.api_key}",
        }
        form = aiohttp.FormData()
        form.add_field("file", audio, filename="audio.webm", content_type="application/octet-stream")
        form.add_field("model", self.model)
        form.add_field("language", "en")

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=headers, data=form) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Cartesia STT HTTP {resp.status}: {body}")
                data = await resp.json()
                return (data.get("text") or "").strip()