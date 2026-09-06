import json
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from agent.state import ResearchPilotState
from agent.router import router_node
from agent.retriever import retriever_node
from agent.web_search import web_search_node
from agent.reasoner import reasoner_node
from agent.synthesizer import synthesizer_node
from agent.graph import route_after_retriever, run_agent_query
from agent.llm import LocalOllamaClient, OllamaUnavailableError


@pytest.fixture
def base_state() -> ResearchPilotState:
    return {
        "query": "According to the research paper, what is the accuracy of agentic RAG?",
        "messages": [{"role": "user", "content": "According to the research paper, what is the accuracy of agentic RAG?"}],
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


def test_ollama_unavailable_raises_error():
    """Verify that when Ollama is unavailable, LocalOllamaClient raises OllamaUnavailableError."""
    client = LocalOllamaClient()
    client._available_cache = False

    with pytest.raises(OllamaUnavailableError) as exc_info:
        client.generate("Test prompt")

    assert "Local Ollama daemon is not running" in str(exc_info.value)


def test_router_node_with_mocked_llm(base_state):
    """Verify router_node parses structured JSON from LLM without live network."""
    mock_client = MagicMock(spec=LocalOllamaClient)
    mock_client.generate.return_value = json.dumps({
        "route": "local_retrieval",
        "reasoning": "Query asks about internal benchmark accuracy.",
        "search_query": "accuracy of agentic rag"
    })

    result = router_node(base_state, client=mock_client)

    assert result["route"] == "local_retrieval"
    assert "benchmark accuracy" in result["route_reasoning"]
    assert any("ROUTER NODE" in step.upper() for step in result["decision_trace"])
    mock_client.generate.assert_called_once()


def test_retriever_node_finds_hits(base_state):
    """Verify retriever_node adds hits and sources_used using score_threshold 0.45."""
    mock_store = MagicMock()
    mock_store.query.return_value = [
        {
            "id": "doc_1",
            "text": "Agentic RAG achieved 89.4% accuracy.",
            "score": 0.88,
            "metadata": {"source": "ai_research_2025.md", "page": 1}
        }
    ]

    result = retriever_node(base_state, store=mock_store)

    assert len(result["local_docs"]) == 1
    assert "Local Document (ai_research_2025.md)" in result["sources_used"]
    assert any("88%" in step for step in result["decision_trace"])
    mock_store.query.assert_called_once_with(
        query_text=base_state["query"],
        n_results=4,
        score_threshold=0.45
    )


def test_retriever_node_dynamic_fallback_trace(base_state):
    """Verify retriever appends dynamic fallback message when local_retrieval finds 0 chunks."""
    base_state["route"] = "local_retrieval"
    mock_store = MagicMock()
    mock_store.query.return_value = []
    mock_store.count.return_value = 5

    result = retriever_node(base_state, store=mock_store)

    assert len(result["local_docs"]) == 0
    assert any("Dynamic Fallback" in step for step in result["decision_trace"])


def test_route_after_retriever_is_pure_readonly(base_state):
    """Verify route_after_retriever does NOT mutate state['decision_trace']."""
    base_state["route"] = "local_retrieval"
    base_state["local_docs"] = []
    base_state["decision_trace"] = ["initial_trace"]

    initial_trace_len = len(base_state["decision_trace"])
    target_node = route_after_retriever(base_state)

    assert target_node == "web_search"
    # Pure function invariant: must NOT mutate state
    assert len(base_state["decision_trace"]) == initial_trace_len
    assert base_state["decision_trace"] == ["initial_trace"]


def test_web_search_node_with_mocked_ddgs(base_state):
    """Verify web_search_node populates results from mocked DDGS."""
    mock_results = [
        {
            "title": "Agentic RAG Overview",
            "href": "https://example.com/agentic-rag",
            "body": "Detailed benchmark results on multi-agent architectures."
        }
    ]

    with patch("ddgs.DDGS") as mock_ddgs_class:
        mock_instance = MagicMock()
        mock_instance.text.return_value = mock_results
        mock_ddgs_class.return_value = mock_instance

        result = web_search_node(base_state, max_results=2)

        assert len(result["web_results"]) == 1
        assert result["web_results"][0]["title"] == "Agentic RAG Overview"
        assert "DuckDuckGo Web Search" in result["sources_used"]


def test_web_search_node_error_does_not_fabricate(base_state):
    """Verify web_search_node on failure does NOT inject fabricated Wikipedia results."""
    with patch("ddgs.DDGS") as mock_ddgs_class:
        mock_instance = MagicMock()
        mock_instance.text.side_effect = ConnectionError("DDGS service unreachable")
        mock_ddgs_class.return_value = mock_instance

        result = web_search_node(base_state, max_results=2)

        # Must NOT contain fabricated Wikipedia snippet
        assert result["web_results"] == []
        assert any("DuckDuckGo search failed" in step for step in result["decision_trace"])


def test_reasoner_node_increments_iteration_count(base_state):
    """Verify reasoner_node increments iteration_count and creates reasoning steps."""
    mock_client = MagicMock(spec=LocalOllamaClient)
    mock_client.generate.side_effect = [
        json.dumps({"sub_questions": ["What is Agentic RAG?", "What is its accuracy?"]}),
        "Agentic RAG uses dynamic routing.",
        "It achieves 89.4% factual accuracy."
    ]

    result = reasoner_node(base_state, client=mock_client)

    assert result["iteration_count"] == 1
    assert len(result["reasoning_steps"]) == 2
    assert "Multi-Step Reasoning Trace" in result["sources_used"]


def test_reasoner_node_iteration_cap(base_state):
    """Verify reasoner_node caps iterations at 3 and does not invoke LLM if limit is exceeded."""
    base_state["iteration_count"] = 3
    mock_client = MagicMock(spec=LocalOllamaClient)

    result = reasoner_node(base_state, client=mock_client)

    assert result["iteration_count"] == 4
    assert any("Maximum reasoning loop iterations reached" in step for step in result["decision_trace"])
    mock_client.generate.assert_not_called()


def test_synthesizer_node_with_citations_and_messages(base_state):
    """Verify synthesizer_node formats citations, grounds answer, and records assistant message."""
    base_state["local_docs"] = [
        {
            "id": "doc_1",
            "text": "Agentic RAG achieved 89.4% factual accuracy compared to 61.2% for naive RAG.",
            "score": 0.9,
            "metadata": {"source": "ai_research_2025.md", "page": 1}
        }
    ]

    mock_client = MagicMock(spec=LocalOllamaClient)
    mock_client.generate.return_value = "Agentic RAG achieves 89.4% accuracy [1]."

    result = synthesizer_node(base_state, client=mock_client)

    assert "89.4%" in result["final_answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source_type"] == "local_doc"
    assert result["citations"][0]["source_name"] == "ai_research_2025.md"
    # Assistant turn should be added to messages
    assert any(m.get("role") == "assistant" for m in result["messages"])


def test_conversation_memory_accumulates_across_turns():
    """Verify run_agent_query preserves messages across multiple turns with the same thread_id."""
    thread_id = "test_memory_thread_123"

    mock_router_resp = json.dumps({"route": "direct", "reasoning": "Greeting", "search_query": ""})
    mock_synth_resp = "Hello! I am ResearchPilot [1]."

    with patch("agent.router.LocalOllamaClient") as mock_router_client_cls, \
         patch("agent.synthesizer.LocalOllamaClient") as mock_synth_client_cls:
        
        mock_r_inst = MagicMock()
        mock_r_inst.generate.return_value = mock_router_resp
        mock_router_client_cls.return_value = mock_r_inst

        mock_s_inst = MagicMock()
        mock_s_inst.generate.return_value = mock_synth_resp
        mock_synth_client_cls.return_value = mock_s_inst

        # Turn 1
        turn1_state = run_agent_query("Hello!", thread_id=thread_id)
        assert len(turn1_state["messages"]) >= 2
        assert turn1_state["messages"][0]["role"] == "user"
        assert turn1_state["messages"][0]["content"] == "Hello!"

        # Turn 2
        turn2_state = run_agent_query("What was my first message?", thread_id=thread_id)
        assert len(turn2_state["messages"]) >= 4
        # Turn 1 user message is preserved in memory
        assert turn2_state["messages"][0]["content"] == "Hello!"
        # Turn 2 user message is appended
        assert turn2_state["messages"][2]["content"] == "What was my first message?"
