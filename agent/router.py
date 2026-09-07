import json
import re
from typing import Dict, Any
from agent.state import ResearchPilotState
from agent.llm import get_llm_client

ROUTER_SYSTEM_PROMPT = """You are the expert Router Node of ResearchPilot, an autonomous Agentic RAG system.
Your job is to analyze the user's question and decide the best execution strategy.

Strategies:
1. "local_retrieval": The query asks about internal documents, uploaded research papers, system architecture, benchmarks, or specific internal facts.
2. "web_search": The query asks about recent events, current news, live web data, or general topics not covered by internal files.
3. "reasoning": The query is complex, comparative, multi-part, or requires step-by-step analytical decomposition.
4. "hybrid": The query would benefit from both checking internal documents AND corroborating with web search.
5. "direct": Simple greetings, courtesies, or questions about ResearchPilot itself.

Return ONLY a JSON object with this exact structure:
{
  "route": "local_retrieval" | "web_search" | "reasoning" | "hybrid" | "direct",
  "reasoning": "Brief explanation of why this path was chosen",
  "search_query": "Clean keyword search query for retrieval/web"
}
"""

def router_node(state: ResearchPilotState, client=None) -> Dict[str, Any]:
    query = state.get("query", "")
    messages = state.get("messages", [])
    trace = state.get("decision_trace", []).copy()

    if client is None:
        client = get_llm_client()

    # Include up to 4 prior turns as conversation context (excluding the current user message)
    history_turns = messages[:-1][-4:] if len(messages) > 1 else []
    history_text = ""
    if history_turns:
        lines = ["Conversation history (most recent turns):"]
        for m in history_turns:
            role = m.get("role", "user").capitalize()
            lines.append(f"  {role}: {m.get('content', '')[:200]}")
        history_text = "\n".join(lines) + "\n\n"

    prompt = (
        f"{history_text}"
        f"User Question: {query}\n\n"
        "Classify this query and provide the JSON routing decision."
    )

    try:
        response_text = client.generate(prompt=prompt, system=ROUTER_SYSTEM_PROMPT, json_mode=True)
        # Extract JSON block if wrapped in markdown
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            data = json.loads(response_text)
            
        route = data.get("route", "hybrid").lower()
        reason = data.get("reasoning", "Routing based on query intent.")
    except Exception as e:
        # Heuristic fallback if json parse fails
        q_lower = query.lower()
        if any(w in q_lower for w in ["compare", "difference", "versus", "vs", "why", "break down"]):
            route = "reasoning"
            reason = "Detected comparative/analytical phrasing; routing to multi-step reasoning."
        elif any(w in q_lower for w in ["latest", "recent", "today", "news", "current"]):
            route = "web_search"
            reason = "Query references live/recent events; routing to web search."
        elif any(w in q_lower for w in ["document", "paper", "pdf", "section", "table", "benchmark"]):
            route = "local_retrieval"
            reason = "Query specifically references document corpus; routing to local retrieval."
        else:
            route = "hybrid"
            reason = "Hybrid query: cross-referencing local document knowledge with external search."

    trace.append(f"🧭 **Router Node**: Selected route `[{route.upper()}]` — *{reason}*")

    return {
        "route": route,
        "route_reasoning": reason,
        "decision_trace": trace,
    }

