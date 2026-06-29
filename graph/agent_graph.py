from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from graph.nodes.analysis import analysis_node, retriever_node
from graph.nodes.language import language_node
from graph.nodes.market import api_node
from graph.nodes.news import scraping_node
from graph.nodes.portfolio import verify_portfolio_node
from graph.nodes.visualization import visualization_node
from graph.router import route_by_intent, route_intent, should_visualize
from graph.state import AgentState

_checkpointer = MemorySaver()


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("route_intent", route_intent)
    graph.add_node("verify_portfolio", verify_portfolio_node)
    graph.add_node("fetch_market", api_node)
    graph.add_node("fetch_news", scraping_node)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("analyze", analysis_node)
    graph.add_node("visualize", visualization_node)
    graph.add_node("generate_response", language_node)
    graph.set_entry_point("route_intent")

    graph.add_conditional_edges("route_intent", route_by_intent, {
        "verify": "verify_portfolio",
        "news": "fetch_market",
        "visualize": "fetch_market",
        "general": "fetch_market",
    })

    graph.add_edge("verify_portfolio", "fetch_market")
    graph.add_edge("fetch_market", "fetch_news")
    graph.add_edge("fetch_news", "retrieve")
    graph.add_edge("retrieve", "analyze")

    graph.add_conditional_edges("analyze", should_visualize, {
        True: "visualize",
        False: "generate_response",
    })

    graph.add_edge("visualize", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile(checkpointer=_checkpointer)


agent_graph = build_agent_graph()


async def run_agent(state: AgentState, thread_id: str) -> AgentState:
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent_graph.ainvoke(state, config)
    return result