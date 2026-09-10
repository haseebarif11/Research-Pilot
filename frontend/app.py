import os
import sys
from pathlib import Path
import streamlit as st

# ── Streamlit Cloud secrets → os.environ bridge ──────────────────────────────
# On Streamlit Cloud, secrets set in the dashboard live in st.secrets.
# We recursively extract all secrets and normalize common keys into os.environ.
try:
    def _extract_secrets(mapping):
        for k, v in mapping.items():
            if isinstance(v, dict) or hasattr(v, "items"):
                _extract_secrets(v)
            else:
                os.environ.setdefault(str(k), str(v))
                upper_k = str(k).upper()
                if upper_k in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN", "LLM_BACKEND", "HF_MODEL"):
                    os.environ[upper_k] = str(v)
    _extract_secrets(st.secrets)
except Exception:
    pass

# Auto-select hf_inference backend if an HF token is available and no backend was specified
if not os.environ.get("LLM_BACKEND"):
    for _token_key in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(_token_key):
            os.environ["LLM_BACKEND"] = "hf_inference"
            break
# ─────────────────────────────────────────────────────────────────────────────

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.graph import run_agent_query
from agent.llm import get_llm_client, reset_llm_client, LocalOllamaClient, OllamaUnavailableError, HFInferenceError
from ingestion.chunker import DocumentChunker
from ingestion.embedder import EmbeddingModelError
from ingestion.vector_store import ChromaVectorStore


st.set_page_config(
    page_title="ResearchPilot | 100% Free Local Agentic RAG",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Theme enhancements */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header hero */
    .hero-container {
        padding: 1.5rem 1.5rem 1.2rem 1.5rem;
        border-radius: 12px;
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(48, 54, 61, 0.9);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        margin-bottom: 0.6rem;
    }
    .pill-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        background: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.3);
    }
    .pill-badge-green {
        background: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.3);
    }

    /* Source Tags */
    .source-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem 0.3rem 0.2rem 0;
        background: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }

    /* Citation Box */
    .citation-card {
        background: #161b22;
        border-left: 3px solid #58a6ff;
        padding: 0.6rem 0.8rem;
        border-radius: 0 6px 6px 0;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #8b949e;
    }
    .citation-card strong {
        color: #58a6ff;
    }

    /* Trace timeline box */
    .trace-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.8rem;
        font-family: monospace;
        font-size: 0.85rem;
        color: #7ee787;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Custom CSS injected fresh each run (st.markdown is idempotent)
    st.markdown("""
    <style>
        /* Theme enhancements */
        .stApp {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            color: #e6edf3;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        
        /* Header hero */
        .hero-container {
            padding: 1.5rem 1.5rem 1.2rem 1.5rem;
            border-radius: 12px;
            background: rgba(22, 27, 34, 0.85);
            border: 1px solid rgba(48, 54, 61, 0.9);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
            margin-bottom: 1.5rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #58a6ff, #bc8cff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            color: #8b949e;
            font-size: 0.95rem;
            margin-bottom: 0.6rem;
        }
        .pill-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.4rem;
            background: rgba(56, 139, 253, 0.15);
            color: #58a6ff;
            border: 1px solid rgba(56, 139, 253, 0.3);
        }
        .pill-badge-green {
            background: rgba(46, 160, 67, 0.15);
            color: #3fb950;
            border: 1px solid rgba(46, 160, 67, 0.3);
        }

        /* Source Tags */
        .source-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
            margin: 0.2rem 0.3rem 0.2rem 0;
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
        }

        /* Citation Box */
        .citation-card {
            background: #161b22;
            border-left: 3px solid #58a6ff;
            padding: 0.6rem 0.8rem;
            border-radius: 0 6px 6px 0;
            margin-top: 0.5rem;
            font-size: 0.85rem;
            color: #8b949e;
        }
        .citation-card strong {
            color: #58a6ff;
        }

        /* Trace timeline box */
        .trace-box {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 0.8rem;
            font-family: monospace;
            font-size: 0.85rem;
            color: #7ee787;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize singletons in session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = ChromaVectorStore()

    if "chunker" not in st.session_state:
        st.session_state.chunker = DocumentChunker()

    if "llm_client" not in st.session_state:
        st.session_state.llm_client = get_llm_client()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    vector_store = st.session_state.vector_store
    chunker = st.session_state.chunker
    llm_client = st.session_state.llm_client

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
        st.title("ResearchPilot")
        st.caption("100% Free & Local Agentic RAG")

        st.markdown("---")
        st.subheader("⚙️ LLM Engine Settings")

        ollama_probe = LocalOllamaClient()
        ollama_is_up = ollama_probe.is_available()
        hf_token_val = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or ""

        # Determine default provider
        current_backend = os.environ.get("LLM_BACKEND", "").strip().lower()
        if current_backend == "hf_inference":
            provider_idx = 0
        elif current_backend == "ollama":
            provider_idx = 1
        else:
            provider_idx = 1 if ollama_is_up else 0

        provider_choice = st.radio(
            "Inference Provider",
            ["☁️ Hugging Face (Free Cloud API)", "💻 Local Ollama"],
            index=provider_idx,
            help="Hugging Face provides free serverless inference in the cloud. Ollama runs locally on your machine."
        )

        if "Hugging Face" in provider_choice:
            os.environ["LLM_BACKEND"] = "hf_inference"

            token_input = st.text_input(
                "Hugging Face Token",
                value=hf_token_val,
                type="password",
                placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                help="Get a free token at https://huggingface.co/settings/tokens"
            )
            if token_input != hf_token_val:
                os.environ["HF_TOKEN"] = token_input.strip()
                reset_llm_client()
                hf_token_val = token_input.strip()

            hf_model_options = [
                "Qwen/Qwen2.5-7B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
            ]
            saved_hf_model = os.environ.get("HF_MODEL", hf_model_options[0])
            model_index = hf_model_options.index(saved_hf_model) if saved_hf_model in hf_model_options else 0
            chosen_hf_model = st.selectbox("Active HF Model", hf_model_options, index=model_index)
            os.environ["HF_MODEL"] = chosen_hf_model

            if hf_token_val:
                st.success("🟢 HF Inference API: Configured")
            else:
                st.warning("⚠️ Enter a free HF Token above (or in Secrets) to enable cloud inference.")

            if st.button("⚡ Test Connection"):
                with st.spinner("Pinging Hugging Face model..."):
                    try:
                        active_client = get_llm_client("hf_inference")
                        if hasattr(active_client, "test_connection"):
                            ok, msg = active_client.test_connection()
                            if ok:
                                st.success(f"✅ {msg}")
                            else:
                                st.error(f"❌ {msg}")
                        else:
                            st.info("Client initialized.")
                    except Exception as e:
                        st.error(f"❌ Test failed: {e}")

        else:
            os.environ["LLM_BACKEND"] = "ollama"
            if ollama_is_up:
                st.success("🟢 Ollama Daemon: Connected")
                installed_models = ollama_probe.list_installed_models()
                if installed_models:
                    sel_model = st.selectbox("Active Ollama Model", installed_models, index=0)
                    os.environ["OLLAMA_MODEL"] = sel_model
                else:
                    st.warning("No models found in Ollama. Pull one with: `ollama pull llama3.1`")
            else:
                st.info("🟡 Ollama: Offline (Local Heuristic Fallback Active)")
                st.caption("To run locally: install Ollama and execute `ollama run llama3.1`.")
                if st.button("🔄 Re-check Ollama"):
                    ollama_probe._available_cache = None
                    reset_llm_client()
                    st.rerun()

        # Keep session state client updated
        st.session_state.llm_client = get_llm_client()
        llm_client = st.session_state.llm_client


        doc_count = vector_store.count()
        st.markdown(f"**Vector Knowledge Base**: `{doc_count}` chunks")

        st.markdown("---")
        st.subheader("📁 Ingest Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT, or MD",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True
        )

        if uploaded_files:
            upload_dir = ROOT_DIR / "data" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            for up in uploaded_files:
                file_path = upload_dir / up.name
                with open(file_path, "wb") as f:
                    f.write(up.getbuffer())
                
                with st.spinner(f"Indexing {up.name}..."):
                    try:
                        chunks = chunker.chunk_document(str(file_path))
                        added = vector_store.add_chunks(chunks)
                        st.success(f"Indexed {up.name} ({added} chunks)")
                    except (OllamaUnavailableError, HFInferenceError) as e:
                        st.error(f"LLM backend error — check your LLM_BACKEND and credentials ({e})")
                    except EmbeddingModelError as e:
                        st.error(f"Embedding model failed to load — check your internet connection or sentence-transformers install ({e})")

        # Sample Document Button
        if st.button("📥 Load 2025 AI Research Sample Doc"):
            sample_path = ROOT_DIR / "sample_docs" / "ai_research_2025.md"
            if sample_path.exists():
                try:
                    chunks = chunker.chunk_document(str(sample_path))
                    vector_store.add_chunks(chunks)
                    st.success("Loaded sample research doc into ChromaDB!")
                    st.rerun()
                except (OllamaUnavailableError, HFInferenceError) as e:
                    st.error(f"LLM backend error — check your LLM_BACKEND and credentials ({e})")
                except EmbeddingModelError as e:
                    st.error(f"Embedding model failed to load — check your internet connection or sentence-transformers install ({e})")

        # Document list
        sources = vector_store.list_sources()
        if sources:
            st.markdown("**Indexed Documents:**")
            for s in sources:
                st.caption(f"📄 `{s['source']}` — {s['chunk_count']} chunks")

        if st.button("🗑️ Clear Vector Database"):
            vector_store.clear()
            st.warning("Vector database cleared.")
            st.rerun()

        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Main Area Header
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🚀 ResearchPilot: Autonomous Agentic RAG</div>
        <div class="hero-subtitle">
            Dynamically routes between local vector storage, free DuckDuckGo live search, and multi-step reasoning before synthesizing answers with grounded citations.
        </div>
        <div>
            <span class="pill-badge pill-badge-green">✔ $0 Cost (100% Free)</span>
            <span class="pill-badge">✔ Zero API Keys</span>
            <span class="pill-badge">✔ LangGraph Agent</span>
            <span class="pill-badge">✔ Numpy Vector Store</span>
            <span class="pill-badge">✔ DuckDuckGo Search</span>
            <span class="pill-badge">✔ Local Ollama LLM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick prompts
    col1, col2, col3 = st.columns(3)
    quick_query = None
    if col1.button("📊 Compare Agentic vs Plain RAG"):
        quick_query = "What are the benchmark accuracy and hallucination rate differences between Agentic RAG and Plain RAG in the research paper?"
    if col2.button("🌐 Latest AI Hardware Developments"):
        quick_query = "What are the latest public developments and specifications for Nvidia Blackwell B200 AI GPUs?"
    if col3.button("🧠 Multi-Step Reasoner Breakdown"):
        quick_query = "Compare the architectural differences, latency tradeoffs, and source attribution between naive RAG and agentic RAG step by step."

    # Display message history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                # Sources Used header
                sources_used = msg.get("sources_used", [])
                if sources_used:
                    st.markdown("**Sources Consulted:**")
                    badges_html = "".join([f'<span class="source-tag">{s}</span>' for s in sources_used])
                    st.markdown(badges_html, unsafe_allow_html=True)

                # Decision Trace Accordion
                trace = msg.get("decision_trace", [])
                if trace:
                    with st.expander("🔍 Agent Decision Trace & Execution Path", expanded=False):
                        for step in trace:
                            st.markdown(step)

                # Final Answer
                st.markdown(msg["content"])

                # Citations list
                citations = msg.get("citations", [])
                if citations:
                    with st.expander(f"📚 Grounded Citations ({len(citations)})", expanded=False):
                        for c in citations:
                            st.markdown(f"""
                            <div class="citation-card">
                                <strong>[{c['id']}] {c['source_type'].upper()}: {c['source_name']}</strong> ({c['detail']})<br/>
                                <em>"{c['snippet']}"</em>
                            </div>
                            """, unsafe_allow_html=True)

    # User input box
    prompt = st.chat_input("Ask ResearchPilot anything (internal docs, web search, or multi-step reasoning)...")
    if quick_query:
        prompt = quick_query

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with Agentic RAG LangGraph
        with st.chat_message("assistant"):
            result = None
            error_msg = None
            with st.status("🧠 ResearchPilot Agent is thinking...", expanded=True) as status_box:
                status_box.write("1. 🧭 Routing intent (Local Docs vs Web Search vs Reasoning)...")
                
                # Run query
                try:
                    result = run_agent_query(prompt, thread_id="streamlit_user_session")
                except (OllamaUnavailableError, HFInferenceError) as e:
                    status_box.update(label="❌ LLM Backend Error", state="error", expanded=False)
                    error_msg = f"LLM backend error — check your LLM_BACKEND and credentials ({e})"
                except EmbeddingModelError as e:
                    status_box.update(label="❌ Embedding Model Error", state="error", expanded=False)
                    error_msg = f"Embedding model failed to load — check your internet connection or sentence-transformers install ({e})"

                if result:
                    route = result.get("route", "hybrid")
                    status_box.write(f"2. 🔀 Strategy determined: **{route.upper()}**")
                    
                    if result.get("local_docs"):
                        status_box.write(f"3. 📚 Retrieved {len(result['local_docs'])} local document chunks from ChromaDB.")
                    if result.get("web_results"):
                        status_box.write(f"4. 🌐 Searched DuckDuckGo and collected {len(result['web_results'])} live web sources.")
                    if result.get("reasoning_steps"):
                        status_box.write(f"5. 💡 Decomposed into {len(result['reasoning_steps'])} intermediate reasoning steps.")
                        
                    status_box.write("6. ✍️ Synthesizing grounded answer with citation footnotes...")
                    status_box.update(label="✅ Answer Synthesized!", state="complete", expanded=False)

            if error_msg:
                st.error(error_msg)

            if result:
                # Sources Used Badges
                sources_used = result.get("sources_used", [])
                if sources_used:
                    st.markdown("**Sources Consulted:**")
                    badges_html = "".join([f'<span class="source-tag">{s}</span>' for s in sources_used])
                    st.markdown(badges_html, unsafe_allow_html=True)

                # Decision Trace
                trace = result.get("decision_trace", [])
                if trace:
                    with st.expander("🔍 Agent Decision Trace & Execution Path", expanded=True):
                        for step in trace:
                            st.markdown(step)

                # Final Answer
                final_ans = result.get("final_answer", "")
                st.markdown(final_ans)

                # Citations
                citations = result.get("citations", [])
                if citations:
                    with st.expander(f"📚 Grounded Citations ({len(citations)})", expanded=True):
                        for c in citations:
                            st.markdown(f"""
                            <div class="citation-card">
                                <strong>[{c['id']}] {c['source_type'].upper()}: {c['source_name']}</strong> ({c['detail']})<br/>
                                <em>"{c['snippet']}"</em>
                            </div>
                            """, unsafe_allow_html=True)

                # Append assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_ans,
                    "sources_used": sources_used,
                    "decision_trace": trace,
                    "citations": citations
                })

if __name__ == "__main__":
    main()
