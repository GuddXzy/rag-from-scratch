"""Launch the Streamlit frontend from project root."""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run([sys.executable, "-m", "streamlit", "run", "src/frontend/streamlit_app.py"])
