import pytest

from graph.agent_graph import agent_graph, run_agent
from graph.router import route_intent, route_by_intent
from graph.state import AgentState


def test_langgraph_compiles():
    nodes = list(agent_graph.get_graph().nodes.keys())
    assert "fetch_market" in nodes
    assert "generate_response" in nodes
    assert "visualize" in nodes


def test_route_intent_visualization():
    state: AgentState = {"query": "visualize my portfolio", "portfolio": "AAPL:10"}
    result = route_intent(state)
    assert result["intent"] == "visualize"
    assert route_by_intent(result) == "visualize"


def test_route_intent_news():
    state: AgentState = {"query": "latest news on AAPL", "portfolio": "AAPL:10"}
    result = route_intent(state)
    assert result["intent"] == "news"


@pytest.mark.asyncio
async def test_agent_graph_market_only(all_live_keys):
    if not all_live_keys:
        pytest.skip("GEMINI_API_KEY, CARTESIA_API_KEY, NEWSAPI_KEY required for live graph test")
    state: AgentState = {
        "query": "portfolio update",
        "portfolio": "AAPL:10",
        "user_id": "test",
        "errors": [],
        "charts": [],
    }
    result = await run_agent(state, thread_id="test_thread_graph")
    assert result.get("analysis_result") or result.get("market_data")
    assert result.get("phase") == "RESPONDING" or result.get("narrative")