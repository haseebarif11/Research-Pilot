import time
import sys
from pathlib import Path

# Add root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("1. Testing agent.state import...")
from agent.state import ResearchPilotState

print("2. Testing router_node...")
from agent.router import router_node
state: ResearchPilotState = {
    "query": "According to the research paper, what is the accuracy of agentic RAG?",
    "messages": [],
    "route": "",
    "route_reasoning": "",
    "local_docs": [],
    "web_results": [],
    "reasoning_steps": [],
    "decision_trace": [],
    "final_answer": "",
    "citations": [],
    "sources_used": [],
    "iteration_count": 0
}
t0 = time.time()
res_router = router_node(state)
print(f"   Router result: route={res_router.get('route')}, time={time.time()-t0:.2f}s")
state.update(res_router)

print("3. Testing retriever_node...")
from agent.retriever import retriever_node
t0 = time.time()
res_retriever = retriever_node(state)
print(f"   Retriever result: local_docs={len(res_retriever.get('local_docs', []))}, time={time.time()-t0:.2f}s")
state.update(res_retriever)

print("4. Testing web_search_node...")
from agent.web_search import web_search_node
t0 = time.time()
res_web = web_search_node(state, max_results=2)
print(f"   Web search result: web_results={len(res_web.get('web_results', []))}, time={time.time()-t0:.2f}s")
state.update(res_web)

print("5. Testing synthesizer_node...")
from agent.synthesizer import synthesizer_node
t0 = time.time()
res_synth = synthesizer_node(state)
print(f"   Synthesizer result: citations={len(res_synth.get('citations', []))}, time={time.time()-t0:.2f}s")
print("   Sample answer:", res_synth.get("final_answer")[:150])

print("\nALL NODES COMPLETED SUCCESSFULLY!")
