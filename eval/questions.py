BENCHMARK_QUESTIONS = [
    {
        "id": 1,
        "category": "Local Document Factual",
        "question": "According to the research document, what was the factual accuracy achieved by Agentic RAG compared to standard Naive RAG on multi-hop questions?",
        "expected_keywords": ["89.4%", "61.2%"],
        "target_source": "local_doc"
    },
    {
        "id": 2,
        "category": "Local Document Factual",
        "question": "What cosine similarity distance threshold is used to flag low confidence chunks in the benchmark?",
        "expected_keywords": ["0.45"],
        "target_source": "local_doc"
    },
    {
        "id": 3,
        "category": "Local Document Factual",
        "question": "By how much did Agentic RAG reduce ungrounded hallucination rates in the 2025 benchmark consortium?",
        "expected_keywords": ["23.8%", "4.2%"],
        "target_source": "local_doc"
    },
    {
        "id": 4,
        "category": "Web Search Real-Time",
        "question": "What is the primary purpose and architecture of Ollama for running local LLMs?",
        "expected_keywords": ["ollama", "local", "model"],
        "target_source": "web"
    },
    {
        "id": 5,
        "category": "Web Search Real-Time",
        "question": "What are the latest open-source Qwen 2.5 and Llama 3.1 model families developed for?",
        "expected_keywords": ["llama", "qwen", "weights"],
        "target_source": "web"
    },
    {
        "id": 6,
        "category": "Multi-Step Reasoning",
        "question": "Compare the tradeoffs between Plain RAG and Agentic RAG in terms of latency versus factual accuracy step by step.",
        "expected_keywords": ["latency", "accuracy", "tradeoff"],
        "target_source": "reasoning"
    },
    {
        "id": 7,
        "category": "Multi-Step Reasoning",
        "question": "Analyze why dynamic routing and intermediate query decomposition reduce hallucinations in research agents.",
        "expected_keywords": ["routing", "decomposition", "hallucination"],
        "target_source": "reasoning"
    },
    {
        "id": 8,
        "category": "Hybrid Knowledge",
        "question": "How does integrating DuckDuckGo zero-key search with local ChromaDB vector retrieval prevent knowledge staleness?",
        "expected_keywords": ["duckduckgo", "chromadb", "retrieval"],
        "target_source": "hybrid"
    },
    {
        "id": 9,
        "category": "Out-of-Domain Web",
        "question": "What is the latest status of NASA Artemis lunar exploration program?",
        "expected_keywords": ["artemis", "nasa", "lunar"],
        "target_source": "web"
    },
    {
        "id": 10,
        "category": "Direct Intent / Meta",
        "question": "Hello ResearchPilot, what local components do you use to answer questions?",
        "expected_keywords": ["ollama", "chromadb", "langgraph"],
        "target_source": "direct"
    }
]
