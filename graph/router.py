from graph.state import AgentState


def route_intent(state: AgentState) -> AgentState:
    query = (state.get("query") or "").lower()
    if not state.get("portfolio"):
        state["intent"] = "verify"
        state["phase"] = "PORTFOLIO_VERIFICATION"
    elif "visual" in query or "chart" in query:
        state["intent"] = "visualize"
        state["phase"] = "VISUALIZATION"
    elif "news" in query or "headline" in query:
        state["intent"] = "news"
        state["phase"] = "NEWS_ANALYSIS"
    else:
        state["intent"] = "general"
        state["phase"] = "FETCHING_MARKET"
    return state


def route_by_intent(state: AgentState) -> str:
    return state.get("intent", "general")


def should_visualize(state: AgentState) -> bool:
    return state.get("intent") == "visualize" or "visual" in (state.get("query") or "").lower()