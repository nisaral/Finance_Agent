import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import AsyncIterator, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ProviderChain(Generic[T]):
    def __init__(self, providers: list[T], threshold: int = 3):
        self.providers = providers
        self.failures: dict[str, int] = defaultdict(int)
        self.threshold = threshold
        self.last_provider: Optional[str] = None

    async def execute(self, method: str, *args, **kwargs):
        last_error = None
        for provider in self.providers:
            name = type(provider).__name__
            if self.failures[name] >= self.threshold:
                continue
            try:
                result = await getattr(provider, method)(*args, **kwargs)
                self.failures[name] = 0
                self.last_provider = name
                return result, name
            except Exception as e:
                self.failures[name] += 1
                last_error = e
                logger.warning(f"{name}.{method} failed: {e}")
        raise RuntimeError(f"All providers failed: {last_error}")

    def get_stream(self, method: str, *args, **kwargs):
        """Return async iterator without awaiting (for LLM/TTS streams)."""
        last_error = None
        for provider in self.providers:
            name = type(provider).__name__
            if self.failures[name] >= self.threshold:
                continue
            try:
                self.last_provider = name
                return getattr(provider, method)(*args, **kwargs), name
            except Exception as e:
                self.failures[name] += 1
                last_error = e
                logger.warning(f"{name}.{method} stream failed: {e}")
        raise RuntimeError(f"All providers failed: {last_error}")


class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio: bytes) -> str: ...


class TTSProvider(ABC):
    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]: ...


class LLMProvider(ABC):
    @abstractmethod
    async def stream(self, prompt: str) -> AsyncIterator[str]: ...