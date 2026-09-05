"""
ResearchPilot Entrypoint
Allows running: streamlit run app.py
"""
import sys
from pathlib import Path

# Insert root
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Run the frontend application
from frontend.app import *
