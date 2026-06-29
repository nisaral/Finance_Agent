import operator
from typing import Annotated, Literal, Optional, TypedDict


Phase = Literal[
    "IDLE",
    "PORTFOLIO_VERIFICATION",
    "FETCHING_MARKET",
    "NEWS_ANALYSIS",
    "ANALYZING",
    "VISUALIZATION",
    "RESPONDING",
    "INTERRUPTED",
    "ERROR",
]


class AgentState(TypedDict, total=False):
    call_id: str
    generation_id: str
    phase: Phase
    query: str
    portfolio: str
    user_id: str
    market_data: dict
    news: list
    news_available: bool
    retrieved: list
    analysis_result: dict
    charts: list
    narrative: str
    narrative_for_audio: str
    tone_analysis: dict
    confidence: float
    safe_mode: bool
    interrupted: bool
    intent: str
    errors: Annotated[list, operator.add]
    error: str