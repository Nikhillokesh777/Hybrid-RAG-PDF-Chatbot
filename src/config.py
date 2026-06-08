"""Central configuration — all constants and environment settings live here."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR: Path        = Path(__file__).resolve().parent.parent
FAISS_INDEX_DIR: Path = ROOT_DIR / "faiss_index"
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── API ───────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_FALLBACK_MODEL: str      = "models/gemini-1.5-flash-latest"
GEMINI_ANSWER_MAX_TOKENS: int   = 1700
GEMINI_SUMMARY_MAX_TOKENS: int  = 500
GEMINI_FALLBACK_MAX_TOKENS: int = 400
GEMINI_ANSWER_TEMPERATURE: float   = 0.3
GEMINI_FALLBACK_TEMPERATURE: float = 0.4

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE: int = 64

# ── Chunking ──────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE: int    = 1000
DEFAULT_CHUNK_OVERLAP: int = 200

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K: int                  = 3
DEFAULT_SIMILARITY_THRESHOLD: float = 1.2
PREVIEW_LENGTH: int                 = 300

# ── Memory ────────────────────────────────────────────────────────────────────
# Last N turns injected into every Gemini prompt for conversational continuity.
MEMORY_CONTEXT_WINDOW: int = 3

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE: str   = "PDF Question Answering System"
APP_CAPTION: str = "Hybrid RAG · Semantic search · Gemini · Conversational memory"
