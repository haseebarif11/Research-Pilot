from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import ResearchPilotState
from agent.router import router_node
from agent.retriever import retriever_node
from agent.web_search import web_search_node
from agent.reasoner import reasoner_node
from agent.synthesizer import synthesizer_node

def route_after_router(state: ResearchPilotState) -> Literal["retriever", "web_search", "synthesizer"]:
    route = state.get("route", "hybrid")
    if route == "direct":
        return "synthesizer"
    elif route == "web_search":
        return "web_search"
    else:
        # For "local_retrieval", "hybrid", or "reasoning", start with local retrieval
        return "retriever"

def route_after_retriever(state: ResearchPilotState) -> Literal["web_search", "reasoner", "synthesizer"]:
    route = state.get("route", "hybrid")
    local_docs = state.get("local_docs", [])
    
    # Dynamic fallback: if local_retrieval was requested but yielded no chunks, fall back to web search
    if route == "local_retrieval" and not local_docs:
        return "web_search"

    if route == "reasoning":
        return "reasoner"
    elif route == "hybrid":
        return "web_search"
    else:
        return "synthesizer"

def route_after_web_search(state: ResearchPilotState) -> Literal["reasoner", "synthesizer"]:
    route = state.get("route", "hybrid")
    iteration_count = state.get("iteration_count", 0)
    if route == "reasoning" and iteration_count < 3:
        return "reasoner"
    return "synthesizer"

def create_research_pilot_graph(checkpointer: Any = None):
    workflow = StateGraph(ResearchPilotState)

    # Register nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add conditional branching
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retriever": "retriever",
            "web_search": "web_search",
            "synthesizer": "synthesizer"
        }
    )

    workflow.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {
            "web_search": "web_search",
            "reasoner": "reasoner",
            "synthesizer": "synthesizer"
        }
    )

    workflow.add_conditional_edges(
        "web_search",
        route_after_web_search,
        {
            "reasoner": "reasoner",
            "synthesizer": "synthesizer"
        }
    )

    workflow.add_edge("reasoner", "synthesizer")
    workflow.add_edge("synthesizer", END)

    memory = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=memory)

# Pre-compiled singleton
_compiled_app = None

def get_agent_app():
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = create_research_pilot_graph()
    return _compiled_app

def run_agent_query(query: str, thread_id: str = "default_session") -> ResearchPilotState:
    """
    Invoke the graph and return the final state.

    For a brand-new thread_id a full initial state is created.
    For an existing thread_id the persisted checkpoint is resumed: only the
    new user message is added so that prior conversation history is preserved.
    """
    app = get_agent_app()
    config = {"configurable": {"thread_id": thread_id}}

    # Check whether a checkpoint already exists for this thread
    existing = app.get_state(config)
    if existing and existing.values:
        # Thread exists — build on top of the persisted state.
        # Pull the accumulated messages list from the checkpoint and append
        # the new user turn; reset per-turn working fields so the graph
        # starts fresh for this question.
        prior_messages: list = list(existing.values.get("messages", []))
        prior_messages.append({"role": "user", "content": query})
        delta_state: ResearchPilotState = {
            "query": query,
            "messages": prior_messages,
            "route": "",
            "route_reasoning": "",
            "local_docs": [],
            "web_results": [],
            "reasoning_steps": [],
            "decision_trace": [],
            "final_answer": "",
            "citations": [],
            "sources_used": [],
            "iteration_count": 0,
        }
        result_state = app.invoke(delta_state, config=config)
    else:
        # First turn — create the full initial state.
        initial_state: ResearchPilotState = {
            "query": query,
            "messages": [{"role": "user", "content": query}],
            "route": "",
            "route_reasoning": "",
            "local_docs": [],
            "web_results": [],
            "reasoning_steps": [],
            "decision_trace": [],
            "final_answer": "",
            "citations": [],
            "sources_used": [],
            "iteration_count": 0,
        }
        result_state = app.invoke(initial_state, config=config)

    return result_state

