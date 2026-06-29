import logging
from typing import Dict, List

import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from graph.nodes.shared import cache
from graph.state import AgentState

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_yfinance_data(symbols: List[str]) -> Dict[str, Dict]:
    result = {}
    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).info
            price = info.get("regularMarketPrice", info.get("previousClose", 0))
            volume = info.get("volume", 0)
            result[symbol] = {
                "price": float(price) if price else 100.0,
                "volume": int(volume) if volume else 100000,
                "asset_type": "stocks",
            }
        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            result[symbol] = {"price": 100.0, "volume": 100000, "asset_type": "stocks"}
    return result


async def api_node(state: AgentState) -> AgentState:
    state["phase"] = "FETCHING_MARKET"
    state["market_data"] = {"stocks": {}, "indices": {}, "currencies": {}, "commodities": {}}
    symbols = []
    for s in state.get("portfolio", "").split(","):
        s = s.strip()
        if not s or ":" not in s:
            continue
        symbols.append(s.split(":")[0].strip())

    if not symbols:
        state["error"] = "No valid symbols in portfolio"
        return state

    try:
        cache_key = ",".join(sorted(symbols))
        if cache_key in cache:
            state["market_data"] = cache[cache_key]
            return state

        yf_data = await fetch_yfinance_data(symbols)
        for symbol in symbols:
            if symbol in yf_data:
                at = yf_data[symbol].get("asset_type", "stocks")
                state["market_data"][at][symbol] = {
                    "price": float(yf_data[symbol]["price"]),
                    "volume": int(yf_data[symbol].get("volume", 0)),
                }
        cache[cache_key] = state["market_data"]
    except Exception as e:
        logger.error(f"Market fetch error: {e}")
        state["errors"] = state.get("errors", []) + [f"market_error:{e}"]
        state["market_data"] = {"stocks": {s: {"price": 100.0, "volume": 100000} for s in symbols}}
    return state