"""Core processing modules for the PDF RAG application."""

from src.chunker import chunk_text
from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    GEMINI_FALLBACK_MODEL,
    GOOGLE_API_KEY,
    MEMORY_CONTEXT_WINDOW,
)
from src.memory_manager import MemoryManager
from src.pdf_processor import PDFExtractionResult, extract_text_from_pdf
from src.retrieval import (
    RetrievalResult,
    RetrievalStatus,
    RetrievedChunk,
    retrieve,
)
from src.text_cleaner import clean_chunk, clean_chunks, clean_text
from src.ui_components import (
    render_answer,
    render_chat_controls,
    render_chat_history,
    render_index_status,
    render_page_config,
    render_retrieval_panel,
    render_sidebar,
    render_stats,
    stream_text,
)
from src.vector_store import VectorStore, get_or_build_vector_store

__all__ = [
    # Config
    "GOOGLE_API_KEY",
    "GEMINI_FALLBACK_MODEL",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_TOP_K",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "MEMORY_CONTEXT_WINDOW",
    # PDF
    "extract_text_from_pdf",
    "PDFExtractionResult",
    # Text
    "clean_text",
    "clean_chunk",
    "clean_chunks",
    # Chunking
    "chunk_text",
    # Vector store
    "VectorStore",
    "get_or_build_vector_store",
    # Retrieval
    "retrieve",
    "RetrievalResult",
    "RetrievalStatus",
    "RetrievedChunk",
    # Memory
    "MemoryManager",
    # UI
    "render_page_config",
    "render_sidebar",
    "render_stats",
    "render_retrieval_panel",
    "render_answer",
    "render_chat_history",
    "render_chat_controls",
    "render_index_status",
    "stream_text",
]
