import logging
import os

from core.config_loader import load_agent_config
from providers.base import ProviderChain
from providers.llm.gemini_stream import GeminiStreamLLM
from providers.stt.cartesia_batch import CartesiaBatchSTT
from providers.tts.cartesia import CartesiaTTS

logger = logging.getLogger(__name__)


def _try_provider(factory):
    try:
        return factory()
    except ValueError as e:
        logger.warning(str(e))
        return None


def build_stt_chain() -> ProviderChain:
    config = load_agent_config()
    model = config.providers["stt"]["primary"].get("model", "ink-whisper")
    providers = []
    primary = _try_provider(lambda: CartesiaBatchSTT(model=model))
    if primary:
        providers.append(primary)
    if not providers:
        raise ValueError("No STT providers configured — set CARTESIA_API_KEY in .env")
    return ProviderChain(providers)


def build_tts_chain() -> ProviderChain:
    config = load_agent_config()
    chain_cfg = config.providers["tts"]
    providers = []
    primary = _try_provider(
        lambda: CartesiaTTS(
            model=chain_cfg["primary"].get("model", "sonic-3"),
            voice_id=chain_cfg["primary"].get("voice_id", "a0e99841-438c-4a64-b679-ae501e7d6091"),
        )
    )
    if primary:
        providers.append(primary)
    for fb in chain_cfg.get("fallback", []):
        if fb["provider"] == "cartesia":
            alt = _try_provider(
                lambda f=fb: CartesiaTTS(
                    model=f.get("model", "sonic-3"),
                    voice_id=f.get("voice_id", "a0e99841-438c-4a64-b679-ae501e7d6091"),
                )
            )
            if alt and alt not in providers:
                providers.append(alt)
    if not providers:
        raise ValueError("No TTS providers configured — set CARTESIA_API_KEY in .env")
    return ProviderChain(providers)


def build_llm_chain() -> ProviderChain:
    config = load_agent_config()
    model = config.providers["llm"]["primary"].get("model", "gemini-2.0-flash")
    providers = []
    keys = []
    for env_name in ("GEMINI_API_KEY", "GEMINI_API_KEY_FALLBACK"):
        key = os.getenv(env_name, "").strip()
        if key and key not in keys:
            keys.append(key)
    keys.sort(key=lambda k: 0 if k.startswith("AIza") else 1)
    for key in keys:
        llm = _try_provider(lambda k=key: GeminiStreamLLM(model=model, api_key=k))
        if llm:
            providers.append(llm)
    if not providers:
        raise ValueError("No LLM providers configured — set GEMINI_API_KEY in .env")
    return ProviderChain(providers)