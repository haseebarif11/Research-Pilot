from typing import Dict, Any, List
from agent.state import ResearchPilotState
from ingestion.vector_store import ChromaVectorStore

def retriever_node(state: ResearchPilotState, store: ChromaVectorStore = None) -> Dict[str, Any]:
    query = state.get("query", "")
    trace = state.get("decision_trace", []).copy()
    local_docs = state.get("local_docs", []).copy()
    sources_used = state.get("sources_used", []).copy()

    if store is None:
        store = ChromaVectorStore()

    # Search ChromaDB
    hits = store.query(query_text=query, n_results=4, score_threshold=0.55)

    if hits:
        top_score = hits[0]["score"]
        sources = list({h["metadata"].get("source", "doc") for h in hits})
        for s in sources:
            source_label = f"Local Document ({s})"
            if source_label not in sources_used:
                sources_used.append(source_label)

        trace.append(
            f"📚 **Retrieval Node**: Retrieved {len(hits)} relevant chunks from `{', '.join(sources)}` "
            f"(Top similarity confidence: {int(top_score * 100)}%)."
        )
        local_docs.extend(hits)
    else:
        doc_count = store.count()
        if doc_count == 0:
            trace.append("ℹ️ **Retrieval Node**: No documents currently indexed in ChromaDB knowledge base.")
        else:
            trace.append(f"⚠️ **Retrieval Node**: Low semantic match in {doc_count} local chunks; triggering web search expansion.")

    return {
        "local_docs": local_docs,
        "sources_used": sources_used,
        "decision_trace": trace,
    }
