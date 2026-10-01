# Contributing to Research-Pilot

Thank you for taking the time to improve Research-Pilot! This guide covers everything you need to get started.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Running the Test Suite](#running-the-test-suite)
3. [Project Structure](#project-structure)
4. [Coding Guidelines](#coding-guidelines)
5. [Submitting a Pull Request](#submitting-a-pull-request)
6. [Reporting Issues](#reporting-issues)

---

## Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/haseebarif11/Research-Pilot.git
cd Research-Pilot

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies (including dev extras)
pip install -r requirements.txt

# 4. Copy the environment template and fill in your values
cp .env.example .env
```

> **Ollama (local LLM):** Install [Ollama](https://ollama.ai) and pull the default model with `ollama pull llama3.2`.  
> **HuggingFace (cloud LLM):** Set `HF_API_TOKEN` in your `.env` file.

---

## Running the Test Suite

All tests live under `tests/` and use `pytest`:

```bash
# Run the full suite
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_nodes.py -v

# Run tests matching a keyword
python -m pytest tests/ -k "web_search" -v

# Show coverage report (requires pytest-cov)
pip install pytest-cov
python -m pytest tests/ --cov=agent --cov-report=term-missing
```

The test suite mocks all LLM backends and the DDGS web-search client — **no live API calls or running Ollama daemon are required** to pass the tests.

---

## Project Structure

```
Research-Pilot/
├── agent/
│   ├── graph.py          # LangGraph workflow definition & entry point
│   ├── llm.py            # LLM client abstraction (Ollama + HuggingFace)
│   ├── router.py         # Intent-routing node
│   ├── retriever.py      # Local vector-DB retrieval node
│   ├── web_search.py     # DuckDuckGo web-search node
│   ├── reasoner.py       # Multi-step reasoning / decomposition node
│   ├── synthesizer.py    # Final answer synthesis node
│   └── state.py          # Shared TypedDict state schema
├── ingestion/            # Document ingestion & embedding pipeline
├── api/                  # FastAPI REST endpoints
├── frontend/             # Streamlit or HTML/JS UI
├── eval/                 # Offline evaluation harness
├── tests/                # Pytest test suite
├── sample_docs/          # Example documents for quick demos
├── requirements.txt
├── .env.example
└── README.md
```

---

## Coding Guidelines

- **Python version:** 3.9+  
- **Type hints:** All public functions must have full type annotations.  
- **Docstrings:** Use Google-style docstrings for new modules and public functions.  
- **Imports:** Standard library → third-party → local. Each group separated by a blank line.  
- **Soft dependencies:** Wrap optional imports in `try/except ImportError` and log a clear warning when the package is absent (see `agent/web_search.py` for the pattern).  
- **Error propagation:** Infrastructure errors (e.g., `OllamaUnavailableError`) must be re-raised so callers can handle them explicitly. Only transient or recoverable errors should be swallowed with a fallback.  
- **Commits:** Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).

---

## Submitting a Pull Request

1. **Fork** the repository and create a feature branch off `main`:
   ```bash
   git checkout -b feat/my-improvement
   ```
2. **Write tests** for every new behaviour or bug fix — the CI pipeline will reject untested changes.
3. **Ensure all tests pass** locally before pushing.
4. **Open a PR** against `main` with:
   - A clear title following Conventional Commits.
   - A description explaining *what* changed and *why*.
   - Screenshots / trace output for user-facing changes.

---

## Reporting Issues

Please use the [GitHub Issues](https://github.com/haseebarif11/Research-Pilot/issues) tracker.

When filing a bug report, include:
- Python version (`python --version`)
- Operating system
- Steps to reproduce
- Expected vs. actual behaviour
- Full traceback (if applicable)

---

*Happy hacking! 🚀*
