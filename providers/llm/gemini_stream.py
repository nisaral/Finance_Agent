import logging
import os
from typing import AsyncIterator

import google.generativeai as genai

from providers.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiStreamLLM(LLMProvider):
    def __init__(self, model: str = "gemini-1.5-flash", api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required")
        self.model_name = model

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text