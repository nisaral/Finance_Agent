import logging

import numpy as np
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential

from graph.nodes.shared import documents, metadata
from graph.state import AgentState

logger = logging.getLogger(__name__)
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_asset_metadata(symbols: list) -> dict:
    meta = {}
    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).info
            asset_class = "Stock"
            if info.get("quoteType") == "ETF":
                asset_class = "ETF"
            meta[symbol] = {
                "asset_class": asset_class,
                "sector": info.get("sector", "Unknown"),
                "name": info.get("longName", symbol),
                "region": info.get("country", "Unknown"),
            }
        except Exception:
            meta[symbol] = {"asset_class": "Unknown", "sector": "Unknown", "region": "Unknown"}
    return meta


async def retriever_node(state: AgentState) -> AgentState:
    state["retrieved"] = []
    state["confidence"] = 0.0
    query = state.get("query", "")
    if not documents or not query:
        return state
    try:
        all_texts = documents + [query]
        matrix = tfidf_vectorizer.fit_transform(all_texts)
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
        top_k = min(5, len(documents))
        indices = np.argsort(sims)[-top_k:][::-1]
        state["confidence"] = float(max(sims)) if len(sims) else 0.0
        state["retrieved"] = [documents[i] for i in indices if i < len(documents)]
    except Exception as e:
        logger.error(f"Retriever error: {e}")
    return state


async def analysis_node(state: AgentState) -> AgentState:
    state["phase"] = "ANALYZING"
    state["analysis_result"] = {
        "total_value": 0.0,
        "exposure": {},
        "regional_sector_exposure": {},
        "earnings_surprises": {},
    }
    portfolio = {}
    for s in state.get("portfolio", "").split(","):
        s = s.strip()
        if not s or ":" not in s:
            continue
        sym, qty = s.split(":", 1)
        try:
            portfolio[sym.strip()] = float(qty.strip())
        except ValueError:
            continue

    if not portfolio or not state.get("market_data"):
        return state

    try:
        meta = await fetch_asset_metadata(list(portfolio.keys()))
        total = 0.0
        asset_vals: dict = {}
        regional_vals: dict = {}

        for sym, shares in portfolio.items():
            info = state["market_data"].get("stocks", {}).get(sym, {})
            price = float(info.get("price", 100.0))
            value = shares * price
            total += value
            ac = meta.get(sym, {}).get("asset_class", "Unknown")
            asset_vals[ac] = asset_vals.get(ac, 0.0) + value
            rs = f"{meta.get(sym, {}).get('region', 'Unknown')}_{meta.get(sym, {}).get('sector', 'Unknown')}"
            regional_vals[rs] = regional_vals.get(rs, 0.0) + value

        if total > 0:
            state["analysis_result"] = {
                "total_value": total,
                "exposure": {k: (v / total) * 100 for k, v in asset_vals.items()},
                "regional_sector_exposure": {k: (v / total) * 100 for k, v in regional_vals.items()},
                "earnings_surprises": {},
            }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        state["errors"] = state.get("errors", []) + [f"analysis_error:{e}"]
    return state