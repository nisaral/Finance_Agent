import logging
from graph.state import AgentState

logger = logging.getLogger(__name__)


async def verify_portfolio_node(state: AgentState) -> AgentState:
    state["phase"] = "PORTFOLIO_VERIFICATION"
    portfolio = state.get("portfolio", "")
    if not portfolio.strip():
        state["error"] = "No portfolio provided"
        state["errors"] = state.get("errors", []) + ["portfolio_missing"]
        return state

    for entry in portfolio.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            state["error"] = f"Invalid format: '{entry}' — use SYMBOL:QUANTITY"
            state["errors"] = state.get("errors", []) + ["portfolio_invalid"]
            return state
        symbol, qty = entry.split(":", 1)
        try:
            float(qty.strip())
        except ValueError:
            state["error"] = f"Invalid quantity in '{entry}'"
            state["errors"] = state.get("errors", []) + ["portfolio_invalid"]
            return state

    state["phase"] = "FETCHING_MARKET"
    return state