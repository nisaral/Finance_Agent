import logging
import os
import re
from datetime import datetime

import google.generativeai as genai

from core.config_loader import load_agent_config
from graph.state import AgentState

logger = logging.getLogger(__name__)

def _gemini_keys() -> list[str]:
    keys = []
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_FALLBACK"):
        val = os.getenv(name, "").strip()
        if val and val not in keys:
            keys.append(val)
    return keys


def _build_prompt(state: AgentState) -> str:
    config = load_agent_config()
    analysis = state.get("analysis_result", {})
    total = analysis.get("total_value", 0)
    exposure = analysis.get("exposure", {})
    exposure_str = "; ".join(f"{k}: {v:.1f}%" for k, v in exposure.items()) or "N/A"
    news_summary = " | ".join(
        (n.get("article", "")[:120] for n in (state.get("news") or [])[:3])
    ) or "No recent news"

    safe_note = ""
    if state.get("safe_mode"):
        safe_note = f"\nIMPORTANT: {config.safe_mode.news_fallback_message}\nFocus on portfolio prices only.\n"

    disclaimer = ""
    if config.compliance_mode and config.rules.disclaimer_required:
        disclaimer = f"\nEnd with: {config.rules.disclaimer_text}\n"

    return f"""You are a financial voice assistant. Respond in {config.tone.max_words_per_response} words or fewer.
{safe_note}
Portfolio: {state.get('portfolio', '')}
Total value: ${total:,.2f}
Exposure: {exposure_str}
News: {news_summary}
Query: {state.get('query', '')}
Date: {datetime.utcnow().strftime('%B %d, %Y')}
Provide actionable, quantitative analysis. Avoid $ and % symbols in spoken text.
{disclaimer}"""


async def language_node(state: AgentState) -> AgentState:
    state["phase"] = "RESPONDING"
    keys = sorted(_gemini_keys(), key=lambda k: 0 if k.startswith("AIza") else 1)
    if not keys:
        state["narrative"] = "LLM not configured."
        state["narrative_for_audio"] = state["narrative"]
        return state

    if state.get("safe_mode") and not state.get("news_available"):
        state["narrative_for_audio"] = load_agent_config().safe_mode.news_fallback_message

    prompt = _build_prompt(state)
    last_error = None
    for api_key in keys:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            state["narrative"] = response.text.strip()
            state["narrative_for_audio"] = state["narrative"]
            last_error = None
            break
        except Exception as e:
            last_error = e
            logger.warning(f"Gemini key failed, trying fallback: {e}")

    if last_error:
        logger.error(f"Language node error: {last_error}")
        state["narrative"] = "Unable to generate analysis. Please try again."
        state["narrative_for_audio"] = state["narrative"]
        state["errors"] = state.get("errors", []) + [f"llm_error:{last_error}"]
    return state


async def failure_recovery_node(state: AgentState) -> AgentState:
    if not state.get("market_data"):
        from graph.nodes.market import api_node
        state = await api_node(state)
    if not state.get("news_available"):
        state["safe_mode"] = True
    state["phase"] = "RESPONDING"
    return state


SENTENCE_END = re.compile(r"[.!?]\s|$")


def extract_speakable_chunk(buffer: str, min_words: int = 8) -> tuple[str, str]:
    if SENTENCE_END.search(buffer) or len(buffer.split()) >= min_words:
        match = SENTENCE_END.search(buffer)
        if match:
            end = match.end()
            return buffer[:end].strip(), buffer[end:].strip()
        return buffer.strip(), ""
    return "", buffer