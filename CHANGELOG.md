# Changelog

All notable changes to **Research-Pilot** are documented here.  
This project follows [Semantic Versioning](https://semver.org/) and [Conventional Commits](https://www.conventionalcommits.org/).

---

## [Unreleased]

### Fixed
- **`agent/synthesizer.py`** — `OllamaUnavailableError` is now re-raised instead of being swallowed by the generic exception handler. Callers (CLI, UI, integration tests) receive the error directly so they can surface a clear user-facing message rather than silently returning a degraded heuristic response.
- **`agent/web_search.py`** — `duckduckgo-search` is now an optional dependency. The module loads successfully even when the package is absent; a descriptive warning is logged and the web-search node short-circuits gracefully.
- **`tests/test_nodes.py`** — Corrected mock patch paths for DDGS to target `agent.web_search.DDGS` (the actual runtime name) rather than the original package path, eliminating false test failures when the optional dependency is not installed.

### Added
- **`CONTRIBUTING.md`** — New contributor guide covering development setup, test execution, project structure, coding guidelines, and PR/issue workflows.
- **`CHANGELOG.md`** — This file.

---

## [0.4.0] — 2025-01-15

### Added
- GitHub Actions CI workflow (`test.yml`) — automated pytest run on every push and pull request.
- README badges for CI status, Python version, and license.
- HuggingFace serverless endpoint fallback with a clear error message for 403 token-permission issues.

### Fixed
- Comprehensive fixes for cloud and local LLM execution paths (`agent/llm.py`).
- Replaced unsupported HF model references with verified chat-capable models; prioritised the free `hf-inference` provider.

---

## [0.3.0] — 2024-12-10

### Added
- Multi-step reasoning node (`agent/reasoner.py`) with LLM-powered query decomposition.
- Conversation memory — prior turns are included as context across multi-turn queries.
- `eval/` harness for offline quality benchmarking.

### Changed
- State schema (`agent/state.py`) extended with `reasoning_steps`, `iteration_count`, and `citations` fields.

---

## [0.2.0] — 2024-11-05

### Added
- Web-search node (`agent/web_search.py`) backed by DuckDuckGo.
- FastAPI REST layer (`api/`) exposing `/query` and `/ingest` endpoints.
- Streamlit frontend (`frontend/`) for interactive chat.

### Changed
- LangGraph workflow (`agent/graph.py`) updated with conditional routing via `route_after_retriever`.

---

## [0.1.0] — 2024-10-20

### Added
- Initial implementation of the LangGraph agentic RAG pipeline.
- Local vector-DB retrieval node (`agent/retriever.py`).
- Intent-routing node (`agent/router.py`) with Ollama/HF LLM backend.
- Document ingestion & embedding pipeline (`ingestion/`).
- Basic Pytest test suite (`tests/test_nodes.py`).
