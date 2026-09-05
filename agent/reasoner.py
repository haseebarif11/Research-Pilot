import json
import re
from typing import Dict, Any, List
from agent.state import ResearchPilotState, ReasoningStep
from agent.llm import LocalOllamaClient

DECOMPOSE_PROMPT = """You are the Reasoning Node of ResearchPilot.
Given a complex, comparative, or multi-step question, break it down into 2 to 3 atomic sub-questions that must be investigated in sequence to form a complete, well-reasoned answer.

User Question: {query}

Return ONLY a JSON object with this exact structure:
{{
  "sub_questions": [
    "First atomic sub-question",
    "Second atomic sub-question"
  ]
}}
"""

STEP_ANSWER_PROMPT = """You are answering an intermediate step of a multi-part reasoning problem.
Sub-Question: {sub_question}

Context available:
{context}

Provide a concise, direct answer to this sub-question using the context. Keep it under 3 sentences.
"""

def reasoner_node(state: ResearchPilotState, client: LocalOllamaClient = None) -> Dict[str, Any]:
    query = state.get("query", "")
    trace = state.get("decision_trace", []).copy()
    reasoning_steps = state.get("reasoning_steps", []).copy()
    local_docs = state.get("local_docs", [])
    web_results = state.get("web_results", [])
    sources_used = state.get("sources_used", []).copy()

    if client is None:
        client = LocalOllamaClient()

    trace.append("🧠 **Reasoning Node**: Initiating multi-step query decomposition...")

    # Step 1: Decompose
    sub_questions = []
    try:
        raw_res = client.generate(
            prompt=DECOMPOSE_PROMPT.format(query=query),
            json_mode=True
        )
        json_match = re.search(r"\{.*\}", raw_res, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            sub_questions = data.get("sub_questions", [])
    except Exception:
        pass

    if not sub_questions:
        # Fallback heuristic decomposition
        sub_questions = [
            f"What are the baseline definitions and components relevant to: {query[:50]}?",
            f"What are the key comparisons, metrics, or evidence relating to: {query[:50]}?"
        ]

    trace.append(f"🧠 **Reasoning Node**: Decomposed into {len(sub_questions)} sequential sub-questions.")

    # Context pool
    context_snippets = []
    for d in local_docs[:3]:
        context_snippets.append(f"Doc ({d['metadata'].get('source', 'doc')}): {d['text'][:250]}")
    for w in web_results[:3]:
        context_snippets.append(f"Web ({w.get('title', 'web')}): {w.get('snippet', '')[:250]}")
    context_str = "\n".join(context_snippets) if context_snippets else "General knowledge context."

    # Step 2: Answer each sub-question
    for i, sq in enumerate(sub_questions):
        prompt = STEP_ANSWER_PROMPT.format(sub_question=sq, context=context_str)
        step_ans = client.generate(prompt=prompt)
        
        step_item: ReasoningStep = {
            "step_num": i + 1,
            "sub_question": sq,
            "thought": f"Investigating sub-component {i + 1} across gathered evidence.",
            "answer": step_ans.strip(),
            "sources": [d["metadata"].get("source", "doc") for d in local_docs[:2]]
        }
        reasoning_steps.append(step_item)
        trace.append(f"🔍 **Reasoning Step {i + 1}**: Resolved *\"{sq[:60]}...\"*")

    if "Multi-Step Reasoning Trace" not in sources_used:
        sources_used.append("Multi-Step Reasoning Trace")

    return {
        "reasoning_steps": reasoning_steps,
        "sources_used": sources_used,
        "decision_trace": trace,
    }
