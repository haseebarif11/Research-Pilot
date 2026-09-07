import os
import json
import re
import requests
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama daemon is unreachable or returns an error."""


class HFInferenceError(RuntimeError):
    """Raised when the Hugging Face Inference API is unreachable or returns an error."""


# ---------------------------------------------------------------------------
# Local Ollama backend
# ---------------------------------------------------------------------------

class LocalOllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", default_model: str = "llama3.1"):
        self.base_url = os.environ.get("OLLAMA_BASE_URL", base_url).rstrip("/")
        self.default_model = os.environ.get("OLLAMA_MODEL", default_model)
        self._available_cache = None

    def is_available(self) -> bool:
        """Check if local Ollama daemon is reachable using ultra-fast socket probe."""
        if self._available_cache is not None:
            return self._available_cache
        try:
            import socket
            s = socket.socket()
            s.settimeout(0.15)
            # 11434 default port
            port = 11434
            host = "127.0.0.1"
            if "://" in self.base_url:
                host_part = self.base_url.split("://")[1]
                if ":" in host_part:
                    host, p_str = host_part.split(":")
                    port = int(p_str)
                else:
                    host = host_part
            res = s.connect_ex((host, port))
            s.close()
            self._available_cache = (res == 0)
        except Exception:
            self._available_cache = False
        return self._available_cache

    def list_installed_models(self) -> List[str]:
        """List models installed in local Ollama."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if r.status_code == 200:
                data = r.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate(self, prompt: str, system: Optional[str] = None, model: Optional[str] = None, json_mode: bool = False) -> str:
        """
        Generate completion using Ollama HTTP API.

        Raises:
            OllamaUnavailableError: if Ollama is not reachable or returns a non-200 response.
        """
        if not self.is_available():
            raise OllamaUnavailableError(
                "Local Ollama daemon is not running. "
                "Start it with: ollama run llama3.1"
            )

        active_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": active_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            }
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        try:
            r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=90)
        except requests.exceptions.ConnectionError as e:
            # Daemon disappeared between the socket probe and the POST
            self._available_cache = False
            raise OllamaUnavailableError(
                f"Connection to Ollama lost mid-request: {e}"
            ) from e

        if r.status_code != 200:
            raise OllamaUnavailableError(
                f"Ollama API returned HTTP {r.status_code}: {r.text[:200]}"
            )

        return r.json().get("response", "").strip()


# ---------------------------------------------------------------------------
# Hugging Face Inference API backend
# ---------------------------------------------------------------------------

_DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

class HFInferenceClient:
    """
    LLM client backed by the Hugging Face serverless Inference API.

    Compatible interface with LocalOllamaClient so every node and the
    frontend can use either backend interchangeably.

    Requirements:
        pip install huggingface_hub>=0.23.0

    Environment variables:
        HF_TOKEN  — required; free token from https://huggingface.co/settings/tokens
        HF_MODEL  — optional; defaults to Qwen/Qwen2.5-7B-Instruct
    """

    def __init__(self):
        self._token = os.environ.get("HF_TOKEN", "")
        self._model = os.environ.get("HF_MODEL", _DEFAULT_HF_MODEL)
        self._client = None  # lazy-initialised

    def _get_client(self):
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
            except ImportError as e:
                raise HFInferenceError(
                    "huggingface_hub is not installed. "
                    "Run: pip install huggingface_hub>=0.23.0"
                ) from e
            if not self._token:
                raise HFInferenceError(
                    "HF_TOKEN environment variable is not set. "
                    "Generate a free token at https://huggingface.co/settings/tokens "
                    "and set it as a Space Secret (or in your .env file locally)."
                )
            self._client = InferenceClient(token=self._token)
        return self._client

    # ------------------------------------------------------------------
    # Interface parity with LocalOllamaClient
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """HF Inference API is a cloud service — always report as available."""
        return True

    def list_installed_models(self) -> List[str]:
        """Return the configured HF model as a single-element list."""
        return [self._model]

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a completion via the HF Inference API chat endpoint.

        Uses the messages API (system + user roles) which works correctly
        with all instruction-tuned models without manual prompt templating.

        Raises:
            HFInferenceError: on auth failure, network error, or API error.
        """
        client = self._get_client()
        active_model = model or self._model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat_completion(
                model=active_model,
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
                top_p=0.9,
            )
            text = response.choices[0].message.content or ""
            return text.strip()
        except HFInferenceError:
            raise
        except Exception as e:
            raise HFInferenceError(
                f"Hugging Face Inference API call failed: {e}"
            ) from e


# ---------------------------------------------------------------------------
# Factory — returns the right client based on LLM_BACKEND env var
# ---------------------------------------------------------------------------

_client_singleton: Optional[object] = None


def get_llm_client():
    """
    Return the configured LLM client singleton.

    Reads LLM_BACKEND environment variable:
        "ollama"       → LocalOllamaClient  (default, for local development)
        "hf_inference" → HFInferenceClient  (for Hugging Face Spaces / cloud)

    Both clients expose the same .generate(prompt) interface.
    """
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton

    backend = os.environ.get("LLM_BACKEND", "ollama").strip().lower()
    if backend == "hf_inference":
        _client_singleton = HFInferenceClient()
    else:
        _client_singleton = LocalOllamaClient()

    return _client_singleton
