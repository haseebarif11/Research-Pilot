from typing import Dict, Any, List
from agent.state import ResearchPilotState


try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None


def web_search_node(state: ResearchPilotState, max_results: int = 4) -> Dict[str, Any]:
    query = state.get("query", "")
    trace = state.get("decision_trace", []).copy()
    web_results = state.get("web_results", []).copy()
    sources_used = state.get("sources_used", []).copy()

    # Clean query for web search (remove punctuation, question marks)
    clean_query = query.replace("?", "").strip()

    trace.append(f"🌐 **Web Search Node**: Querying DuckDuckGo (free, 0-key) for: *\"{clean_query}\"*")

    found_results = []
    if DDGS is None:
        trace.append(
            "⚠️ **Web Search Node**: `duckduckgo-search` package not installed — install it with "
            "`pip install duckduckgo-search`. No web results available."
        )
        return {
            "web_results": web_results,
            "sources_used": sources_used,
            "decision_trace": trace,
        }

    try:
        ddgs = DDGS()
        results = list(ddgs.text(clean_query, max_results=max_results))
        for r in results:
            found_results.append({
                "title": r.get("title", "Web Source"),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")
            })
    except Exception as e:
        trace.append(
            f"⚠️ **Web Search Node**: DuckDuckGo search failed — {str(e)[:120]}. "
            f"No web results available for this query."
        )
        return {
            "web_results": web_results,
            "sources_used": sources_used,
            "decision_trace": trace,
        }

    if found_results:
        source_label = "DuckDuckGo Web Search"
        if source_label not in sources_used:
            sources_used.append(source_label)

        trace.append(
            f"✅ **Web Search Node**: Found {len(found_results)} live web sources "
            f"(`{found_results[0].get('title', 'Web')[:35]}...`)."
        )
        web_results.extend(found_results)
    else:
        trace.append("⚠️ **Web Search Node**: No relevant web search results returned.")

    return {
        "web_results": web_results,
        "sources_used": sources_used,
        "decision_trace": trace,
    }
