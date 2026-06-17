"""
streamlit_app.py
-----------------
Streamlit Cloud entry point.

Streamlit Cloud requires a .py file at the repo root.
This module simply starts the Flask dashboard server
and then serves a redirect page.

For Streamlit Cloud deployment:
  - Main file path: streamlit_app.py
  - Python version: 3.12
  - Packages: see requirements-streamlit.txt

Architecture note:
  Streamlit Cloud runs a single Python process. This file boots
  our Flask server on a background thread (port 8501) and then
  uses st.components.v1.iframe to embed the full dashboard.
"""
import os
import sys
import threading
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Streamlit imports
import streamlit as st

st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "https://churn-intelligence-api.onrender.com")
PORT    = int(os.getenv("FLASK_PORT", "8502"))

@st.cache_resource
def _start_flask_server():
    """Start the Flask frontend server on a background thread (once per session)."""
    os.environ["API_URL"]        = API_URL
    os.environ["FRONTEND_PORT"]  = str(PORT)

    def _run():
        try:
            from frontend.server import app
            app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        except Exception as e:
            print(f"Flask server error: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(2)   # Give Flask time to bind
    return f"http://localhost:{PORT}"


# ── Main UI ────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp > header {display: none;}
        .block-container {padding: 0 !important; max-width: 100% !important;}
        iframe {border: none; width: 100%; height: 100vh;}
    </style>
    """, unsafe_allow_html=True)

    dashboard_url = _start_flask_server()

    # Embed the Flask dashboard in a full-screen iframe
    st.components.v1.iframe(dashboard_url, height=900, scrolling=True)

    # Fallback link
    st.markdown(
        f'<p style="text-align:center;color:#64748b;font-size:12px">'
        f'If the dashboard does not load, '
        f'<a href="{dashboard_url}" target="_blank">open it here</a>.</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
