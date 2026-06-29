import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass
class ToneConfig:
    style: str = "professional"
    empathy_multiplier: float = 1.0
    max_words_per_response: int = 200
    avoid_symbols_in_speech: bool = True


@dataclass
class RulesConfig:
    disclaimer_required: bool = True
    disclaimer_text: str = ""
    prohibited_topics: list[str] = field(default_factory=list)
    require_portfolio_verification: bool = True


@dataclass
class SafeModeConfig:
    enabled: bool = True
    news_fallback_message: str = ""
    market_data_stale_threshold_seconds: int = 300


@dataclass
class LatencyConfig:
    stt_endpointing_ms: int = 300
    tts_sentence_min_words: int = 8
    max_tts_chunk_chars: int = 200
    jitter_buffer_ms: int = 75


@dataclass
class AgentConfig:
    name: str
    tone: ToneConfig
    compliance_mode: bool
    rules: RulesConfig
    safe_mode: SafeModeConfig
    latency: LatencyConfig
    providers: dict[str, Any]


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_agent_config() -> AgentConfig:
    raw = _load_json(CONFIG_DIR / "agent_config.json")
    agent = raw["agent"]
    providers = _load_json(CONFIG_DIR / "providers.json")
    return AgentConfig(
        name=agent["name"],
        tone=ToneConfig(**agent["tone"]),
        compliance_mode=agent.get("compliance_mode", False),
        rules=RulesConfig(**agent["rules"]),
        safe_mode=SafeModeConfig(**agent["safe_mode"]),
        latency=LatencyConfig(**agent["latency"]),
        providers=providers,
    )


def reload_config() -> AgentConfig:
    load_agent_config.cache_clear()
    return load_agent_config()