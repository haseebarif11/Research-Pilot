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

_DEFAULT_HF_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
_DEFAULT_HF_MODEL = _DEFAULT_HF_MODELS[0]


def _resolve_hf_token() -> str:
    """Check common environment variable names and Streamlit secrets for an HF token."""
    for var in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN", "hf_token"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    # Also check streamlit secrets if available
    try:
        import streamlit as st
        for var in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN", "hf_token"):
            if var in st.secrets:
                return str(st.secrets[var]).strip()
    except Exception:
        pass
    return ""


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

    def __init__(self, token: Optional[str] = None, model: Optional[str] = None):
        self._explicit_token = token
        self._model = model or os.environ.get("HF_MODEL", _DEFAULT_HF_MODEL)
        self._client = None
        self._backend_name = "hf_inference"

    @property
    def token(self) -> str:
        return self._explicit_token or _resolve_hf_token()

    def _get_client(self, provider: Optional[str] = "hf-inference"):
        token = self.token
        if not token:
            raise HFInferenceError(
                "HF_TOKEN is not set. Please provide a free Hugging Face token in the sidebar "
                "or set HF_TOKEN in Streamlit Secrets / .env."
            )
        try:
            from huggingface_hub import InferenceClient
        except ImportError as e:
            raise HFInferenceError(
                "huggingface_hub is not installed. Run: pip install huggingface_hub>=0.23.0"
            ) from e

        try:
            return InferenceClient(token=token, provider=provider)
        except (TypeError, ValueError):
            return InferenceClient(token=token)

    # ------------------------------------------------------------------
    # Interface parity with LocalOllamaClient
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """HF Inference API is available if an HF token is configured."""
        return bool(self.token)

    def list_installed_models(self) -> List[str]:
        """Return the available HF models list."""
        current = os.environ.get("HF_MODEL", self._model)
        models = [current]
        for m in _DEFAULT_HF_MODELS:
            if m not in models:
                models.append(m)
        return models

    def test_connection(self) -> tuple[bool, str]:
        """Test whether the HF Inference API is working with the current token and model."""
        try:
            res = self.generate("Say 'connected' in one word.", model=self._model)
            return True, f"Success! Model response: {res[:80]}"
        except Exception as e:
            return False, str(e)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a completion via the HF Inference API chat endpoint.

        Tries the free serverless provider first, then router, with fallback
        to classic text-generation endpoints if chat endpoint is not available.

        Raises:
            HFInferenceError: on auth failure, network error, or API error.
        """
        primary_model = model or os.environ.get("HF_MODEL", self._model)
        
        # Build candidate models list: primary first, followed by defaults
        candidate_models = [primary_model]
        for m in _DEFAULT_HF_MODELS:
            if m not in candidate_models:
                candidate_models.append(m)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        errors_log = []
        for candidate in candidate_models:
            # Attempt A: chat_completion with free serverless provider
            for prov in ("hf-inference", None):
                try:
                    c = self._get_client(provider=prov)
                    response = c.chat_completion(
                        model=candidate,
                        messages=messages,
                        max_tokens=1024,
                        temperature=0.2,
                        top_p=0.9,
                    )
                    text = response.choices[0].message.content or ""
                    if text.strip():
                        return text.strip()
                except Exception as e:
                    errors_log.append(f"{candidate} ({prov or 'router'}): {e}")

            # Attempt B: text_generation fallback
            try:
                c = self._get_client(provider="hf-inference")
                full_prompt = f"{system}\n\nUser: {prompt}\nAssistant:" if system else prompt
                output = c.text_generation(
                    full_prompt,
                    model=candidate,
                    max_new_tokens=1024,
                    temperature=0.2,
                    return_full_text=False,
                )
                if output and output.strip():
                    return output.strip()
            except Exception as e:
                errors_log.append(f"{candidate} (text_gen): {e}")

            # Attempt C: Direct classic serverless HTTP request
            try:
                import requests
                headers = {"Authorization": f"Bearer {self.token}"}
                api_url = f"https://api-inference.huggingface.co/models/{candidate}"
                full_prompt = f"{system}\n\nUser: {prompt}\nAssistant:" if system else prompt
                r = requests.post(
                    api_url,
                    headers=headers,
                    json={
                        "inputs": full_prompt,
                        "parameters": {"max_new_tokens": 1024, "temperature": 0.2, "return_full_text": False}
                    },
                    timeout=20
                )
                if r.status_code == 200:
                    res_json = r.json()
                    if isinstance(res_json, list) and len(res_json) > 0:
                        gen_text = res_json[0].get("generated_text", "")
                        if gen_text:
                            return gen_text.strip()
                    elif isinstance(res_json, dict) and "generated_text" in res_json:
                        return res_json["generated_text"].strip()
            except Exception as e:
                errors_log.append(f"{candidate} (direct_http): {e}")

        # Check for 403 permission error across logged errors
        all_errs = " ".join(errors_log)
        if "403" in all_errs and ("Inference Providers" in all_errs or "permissions" in all_errs or "Forbidden" in all_errs):
            raise HFInferenceError(
                "403 Forbidden: Your Hugging Face token is missing the 'Make calls to Inference Providers' permission. "
                "Fix: Go to https://huggingface.co/settings/tokens -> create or edit a token -> select type 'Write' "
                "(or under Inference check 'Make calls to Inference Providers') -> paste new token."
            )

        last_detail = errors_log[-1] if errors_log else "Unknown error"
        raise HFInferenceError(
            f"Hugging Face Inference API failed across models {candidate_models}.\n"
            f"Details: {last_detail}"
        )




# ---------------------------------------------------------------------------
# Factory — returns the right client based on LLM_BACKEND or auto-detection
# ---------------------------------------------------------------------------

_client_singleton: Optional[object] = None


def get_llm_client(force_backend: Optional[str] = None):
    """
    Return the configured LLM client singleton.

    Auto-detects the best backend if LLM_BACKEND is not set:
      1. If force_backend is provided, use it.
      2. If LLM_BACKEND is set ("hf_inference" or "ollama"), respect it.
      3. If HF_TOKEN is present -> defaults to HFInferenceClient.
      4. If local Ollama daemon is reachable -> defaults to LocalOllamaClient.
      5. Otherwise defaults to HFInferenceClient (cloud-ready).
    """
    global _client_singleton

    # Determine desired backend
    backend = (force_backend or os.environ.get("LLM_BACKEND", "")).strip().lower()

    if not backend:
        # Smart auto-detection
        if _resolve_hf_token():
            backend = "hf_inference"
        elif LocalOllamaClient().is_available():
            backend = "ollama"
        else:
            backend = "hf_inference"

    # Re-instantiate if backend changed or singleton not initialized
    current_backend = getattr(_client_singleton, "_backend_name", None)
    if _client_singleton is None or current_backend != backend:
        if backend == "hf_inference":
            _client_singleton = HFInferenceClient()
        else:
            _client_singleton = LocalOllamaClient()
        _client_singleton._backend_name = backend

    return _client_singleton


def reset_llm_client():
    """Reset the LLM client singleton so subsequent calls re-initialize."""
    global _client_singleton
    _client_singleton = None

