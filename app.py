"""
ResearchPilot Entrypoint
Allows running: streamlit run app.py

On Hugging Face Spaces, set LLM_BACKEND and HF_TOKEN as Space Secrets.
For local development, copy .env.example to .env and fill in your values.
"""
import sys
from pathlib import Path

# Load .env file for local development (no-op on HF Spaces where Secrets are
# injected as real environment variables by the platform).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; env vars must be set manually

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
