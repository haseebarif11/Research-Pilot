"""
ResearchPilot Entrypoint
Allows running: streamlit run app.py

On Hugging Face Spaces, set LLM_BACKEND and HF_TOKEN as Space Secrets.
For local development, copy .env.example to .env and fill in your values.
"""
import os
import sys
from pathlib import Path

# Load .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Synchronize Streamlit secrets into os.environ if running on Streamlit Cloud
try:
    import streamlit as st
    def _sync_secrets(mapping):
        for k, v in mapping.items():
            if isinstance(v, dict) or hasattr(v, "items"):
                _sync_secrets(v)
            else:
                os.environ.setdefault(str(k), str(v))
                upper_k = str(k).upper()
                if upper_k in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN", "LLM_BACKEND", "HF_MODEL"):
                    os.environ[upper_k] = str(v)
    _sync_secrets(st.secrets)
except Exception:
    pass

# Auto-set LLM_BACKEND to hf_inference if HF token is present and backend is unset
if not os.environ.get("LLM_BACKEND"):
    for _token_var in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(_token_var):
            os.environ["LLM_BACKEND"] = "hf_inference"
            break

# Insert root
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# Run the frontend application by calling its main() function directly.
# Previously this used importlib.reload() to force re-execution of top-level
# st.* calls on every Streamlit rerun, but that caused StreamlitDuplicateElementId
# because widgets were registered once on import and again on reload.
# The correct pattern is to wrap all UI code in a main() function and call it here.
import frontend.app as _frontend_app
_frontend_app.main()
