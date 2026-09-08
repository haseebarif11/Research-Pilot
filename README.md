---
title: ResearchPilot
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# 🚀 ResearchPilot: 100% Local, Free, Open-Source Agentic RAG

> **Zero paid APIs. Zero account signups. Zero credit cards required.**  
> Built entirely with open-source tools: **LangGraph**, **Ollama**, **ChromaDB**, **sentence-transformers**, **DuckDuckGo Search**, **FastAPI**, and **Streamlit**.

---

## 🌟 Core Concept

Standard RAG (Retrieval-Augmented Generation) is rigid: it blindly retrieves arbitrary document chunks and feeds them to an LLM, often hallucinating when documents don't have the answer or failing when fresh web information or multi-step logic is needed.

**ResearchPilot is an autonomous Agentic RAG system.** It dynamically analyzes the user's query and decides whether to:
1. **Retrieve from a local document knowledge base** (vector search in persistent ChromaDB with cosine similarity score filtering).
2. **Search the live web** for current real-time information via DuckDuckGo (free, 0-key, zero signup).
3. **Reason through multi-step questions** by decomposing complex queries into intermediate sub-questions and solving them sequentially.
4. **Combine multiple sources** into an authoritative final answer with **grounded citations (`[1]`, `[2]`)** and an explicit **decision trace**.

---

## 🛠️ Tech Stack ($0 Cost, 100% Local)

| Component | Tool / Technology | License & Cost |
| :--- | :--- | :--- |
| **Agent Framework** | [LangGraph](https://github.com/langchain-ai/langgraph) | MIT (Free, Open-Source) |
| **LLM Inference** | [Ollama](https://ollama.com) (`llama3.1` or `qwen2.5`) | Runs 100% locally on your machine, $0 |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Apache 2.0 (Local embedded persistent DB) |
| **Local Embeddings** | [sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (`all-MiniLM-L6-v2`) | Apache 2.0 (Runs locally on CPU/GPU) |
| **Web Search Tool** | [duckduckgo-search](https://pypi.org/project/duckduckgo-search/) | Free, no API key, no account signup |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | MIT |
| **Interactive UI** | [Streamlit](https://streamlit.io/) | Apache 2.0 |

---

## 🏛️ System Architecture

```
User Query
    │
    ▼
┌────────────────────────────────────────────────────────┐
│                      Router Node                       │
│     (Classifies intent using local Ollama model)       │
└────────────────────────────────────────────────────────┘
          │                   │                 │
    ┌─────┴───────┐     ┌─────┴──────┐    ┌─────┴──────────┐
    ▼             ▼     ▼            ▼    ▼                ▼
[Local Docs]  [Web Search]     [Multi-Step Reasoning]  [Hybrid]
ChromaDB      DuckDuckGo       Decomposition + Sub-QA  Both Sources
(Cosine Sim)  (Free, 0-Key)    (Ollama Intermediate)   Cross-Checked
    │             │                  │                 │
    └─────────────┴─────────┬────────┴─────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│                     Synthesis Node                     │
│  - Merges cross-source evidence                        │
│  - Grounds claims with numbered citation footnotes     │
│  - Renders complete chronological decision trace       │
└────────────────────────────────────────────────────────┘
                            │
                            ▼
               Final Answer + Footnotes + Trace
```

### Key Differentiators vs Plain RAG
- **Transparent Decision Trace**: Shows you exactly how the agent thought (`"Searching local docs -> not enough info -> searching web -> synthesizing"`).
- **Verifiable Citation Footnotes**: Every single answer highlights whether the source was a local document chunk, a web URL, or an intermediate reasoning step.
- **Dynamic Fallback**: If local documents don't match the query, it automatically escalates to DuckDuckGo search.
- **Conversation Memory**: Context is preserved across multiple chat turns.

---

## 📋 Quick Setup & Installation

### Step 1: Install Ollama (Local LLM Server)
1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Open your terminal and pull a local model (Llama 3.1 or Qwen 2.5):
   ```bash
   ollama pull llama3.1
   ```
   *(Or for lower VRAM: `ollama pull qwen2.5:7b` or `ollama pull llama3.2`)*

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch ResearchPilot
Run the interactive Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

*(Optional: You can also run the backend FastAPI REST API with `uvicorn api.main:app --reload`)*

---

## 📁 Repository Structure

```
research pilot/
├── agent/
│   ├── state.py         # ResearchPilotState typed dictionary
│   ├── llm.py           # Local Ollama client with smart fallback
│   ├── router.py        # Intent classification node
│   ├── retriever.py     # ChromaDB vector search with confidence scoring
│   ├── web_search.py    # DuckDuckGo 0-key web search node
│   ├── reasoner.py      # Query decomposer & intermediate reasoning
│   ├── synthesizer.py   # Final grounded synthesis with citations
│   └── graph.py         # Compiled LangGraph workflow with MemorySaver
├── ingestion/
│   ├── chunker.py       # Multi-format parser (PDF, DOCX, TXT, MD) + sliding chunking
│   ├── embedder.py      # Local sentence-transformers (all-MiniLM-L6-v2) singleton
│   └── vector_store.py  # Persistent ChromaDB collection manager
├── api/
│   └── main.py          # FastAPI server (chat, document upload, status)
├── frontend/
│   └── app.py           # Streamlit application with live decision trace visualizer
├── eval/
│   ├── questions.py     # 10 benchmark queries across all categories
│   └── benchmark.py     # Agentic RAG vs Plain RAG local comparative test
├── sample_docs/
│   └── ai_research_2025.md # Included sample research paper for instant testing
├── app.py               # Root Streamlit entrypoint
├── requirements.txt     # Pinned lightweight dependencies
└── README.md
```

---

## 🧪 Automated Benchmark: Agentic RAG vs Plain RAG

ResearchPilot includes a local evaluation harness comparing Agentic RAG vs Plain RAG across 10 benchmark questions without relying on paid external evaluation APIs:

```bash
python eval/benchmark.py
```

### Measured Metrics:
1. **Factual Accuracy**: Presence of target key facts and accurate metrics.
2. **Hallucination Risk**: Measured when queries require real-time information or deep reasoning where naive RAG guesses blindly.
3. **Latency**: End-to-end processing time per query.
4. **Source Attribution Grounding**: Verification of structured citation footnotes.

---

## 🔒 Privacy & Local Execution Guarantee
- **100% Offline Capable** (Ollama backend): Your documents and questions never leave your device.
- **No Telemetry / No Paid Keys**: Embeddings and inference run locally on CPU/GPU with the Ollama backend.
- **HF Inference backend**: Prompts are sent to the Hugging Face Inference API; no data is stored by HF beyond the request lifecycle.

---

## ☁️ Deploying to Hugging Face Spaces

ResearchPilot can be deployed to [HF Spaces](https://huggingface.co/spaces) using the **Streamlit SDK** — no Docker required.

### 1. Create the Space

On HF, create a new Space with:
- **SDK**: Streamlit
- **App file**: `app.py` *(already at repo root — no changes needed)*

Add the following front-matter to the Space's `README.md`:

```yaml
---
title: ResearchPilot
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---
```

### 2. Set Space Secrets

In your Space → **Settings → Variables and secrets**, add:

| Secret name | Value | Required? |
| :--- | :--- | :--- |
| `LLM_BACKEND` | `hf_inference` | ✅ Yes |
| `HF_TOKEN` | `hf_...` (your token) | ✅ Yes |
| `HF_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | Optional (this is the default) |

**Getting a free HF token** — no credit card required:
1. Sign up at [huggingface.co](https://huggingface.co) (free).
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Create a token with **Read** permissions.

### 3. Supported Free Serverless Models

Both models work with the free HF serverless tier (no billing, no model-card gating):

| Model | Speed | Quality | `HF_MODEL` value |
| :--- | :--- | :--- | :--- |
| Qwen 2.5 7B Instruct | Medium | Higher | `Qwen/Qwen2.5-7B-Instruct` ← **default** |
| Llama 3.2 3B Instruct | ⚡ Fast | Good | `meta-llama/Llama-3.2-3B-Instruct` |

### 4. Local Development with .env

To test the HF backend locally without exporting env vars manually:

```bash
cp .env.example .env
# Edit .env: set LLM_BACKEND=hf_inference and paste your HF_TOKEN
streamlit run app.py
```

`python-dotenv` is included in `requirements.txt` and `app.py` auto-loads `.env` at startup.

