"""Application Entry Point: Hybrid RAG PDF Assistant.

Runs the FastAPI backend and serves the modern HTML/CSS/JS frontend.
"""

import sys
from pathlib import Path

# Safe stdout encoding on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure root directory is on python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print(" [*] Starting Hybrid RAG PDF Assistant on http://127.0.0.1:8000")
    print(" [*] Interactive API Docs available at: http://127.0.0.1:8000/docs")
    print("=" * 65)
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
