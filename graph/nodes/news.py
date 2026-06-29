import logging
import os
from datetime import datetime

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config_loader import load_agent_config
from graph.nodes.shared import analyze_sentiment, cache, documents, metadata, save_news
from graph.state import AgentState

logger = logging.getLogger(__name__)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_newsapi(symbol: str) -> list:
    if not NEWSAPI_KEY:
        raise ValueError("NEWSAPI_KEY not configured")
    url = "https://newsapi.org/v2/everything"
    params = {"q": symbol, "apiKey": NEWSAPI_KEY, "pageSize": 5, "language": "en", "sortBy": "relevancy"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise RuntimeError(f"NewsAPI HTTP {resp.status}")
            data = await resp.json()
            if data.get("status") != "ok":
                raise RuntimeError(data.get("message", "NewsAPI error"))
            articles = [a["description"] for a in data.get("articles", []) if a.get("description")]
            return articles or [f"No news available for {symbol}"]


async def scraping_node(state: AgentState) -> AgentState:
    state["phase"] = "NEWS_ANALYSIS"
    config = load_agent_config()
    state["news"] = []
    symbols = [s.split(":")[0] for s in state.get("portfolio", "").split(",") if s and ":" in s]
    if not symbols:
        state["news_available"] = False
        return state

    try:
        news = []
        for symbol in symbols:
            cache_key = f"news_{symbol}"
            if cache_key in cache:
                news.extend(cache[cache_key])
                continue
            articles_raw = await fetch_newsapi(symbol)
            articles = [
                {"article": t, "sentiment": analyze_sentiment(t), "timestamp": datetime.utcnow().isoformat()}
                for t in articles_raw
            ]
            save_news(symbol, articles)
            cache[cache_key] = articles
            news.extend(articles)

        state["news"] = news
        state["news_available"] = bool(news)
        for i, item in enumerate(news):
            doc_id = f"doc_{state.get('user_id', 'default')}_{len(documents) + i}"
            metadata[doc_id] = {"content": item["article"], "user_id": state.get("user_id", "default"), "type": "document"}
            documents.append(item["article"])
    except Exception as e:
        logger.error(f"News fetch failed: {e}")
        state["news"] = []
        state["news_available"] = False
        state["safe_mode"] = config.safe_mode.enabled
        state["errors"] = state.get("errors", []) + ["NEWS_DEGRADED"]
    return state