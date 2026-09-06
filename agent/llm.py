import os
import json
import re
import requests
from typing import Dict, Any, List, Optional


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama daemon is unreachable or returns an error."""


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
