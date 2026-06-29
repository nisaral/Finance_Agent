from graph.state import AgentState


async def visualization_node(state: AgentState) -> AgentState:
    state["phase"] = "VISUALIZATION"
    state["charts"] = state.get("charts", [])
    exposure = state.get("analysis_result", {}).get("exposure", {})
    if exposure:
        state["charts"] = [{
            "type": "pie",
            "data": {
                "labels": list(exposure.keys()),
                "datasets": [{"label": "Sector Exposure", "data": list(exposure.values())}],
            },
            "options": {"plugins": {"title": {"display": True, "text": "Sector Exposure Breakdown"}}},
        }]
    return state