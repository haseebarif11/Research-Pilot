from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class Citation(TypedDict):
    id: int
    source_type: str  # "local_doc" | "web" | "reasoning"
    source_name: str
    detail: str       # page number or URL
    snippet: str

class ReasoningStep(TypedDict):
    step_num: int
    sub_question: str
    thought: str
    answer: str
    sources: List[str]

class ResearchPilotState(TypedDict):
    query: str
    messages: List[Dict[str, str]]
    route: str
    route_reasoning: str
    local_docs: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    reasoning_steps: List[ReasoningStep]
    decision_trace: List[str]
    final_answer: str
    citations: List[Citation]
    sources_used: List[str]
    iteration_count: int
