"""Vector store: embedding generation, FAISS index build, save, and load."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Directory where FAISS indexes are persisted between sessions
INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model — loaded once at module level and reused across calls.
# 'all-MiniLM-L6-v2' produces 384-dimensional float vectors.
# It is fast, lightweight, and performs well on semantic similarity tasks.
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the shared embedding model, loading it on first call."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _embedding_model


# ── Index file naming ────────────────────────────────────────────────────────

def _chunks_fingerprint(chunks: list[str]) -> str:
    """
    Produce a short hash that uniquely identifies a specific list of chunks.

    This fingerprint is used to name the saved index files so that a
    different document always gets a fresh index rather than loading a
    stale one from a previous upload.
    """
    content = "\n".join(chunks).encode("utf-8")
    return hashlib.md5(content).hexdigest()[:16]


def _index_paths(fingerprint: str) -> tuple[Path, Path]:
    """Return (faiss_index_path, chunks_json_path) for a given fingerprint."""
    return (
        INDEX_DIR / f"{fingerprint}.index",
        INDEX_DIR / f"{fingerprint}.chunks.json",
    )


# ── Core dataclass ───────────────────────────────────────────────────────────

@dataclass
class VectorStore:
    """
    Holds everything needed for semantic retrieval.

    Attributes:
        index:       FAISS IndexFlatL2 built from chunk embeddings.
        embeddings:  float32 numpy array of shape (n_chunks, embedding_dim).
        chunks:      The original text chunks in the same order as embeddings.
        fingerprint: MD5 hash of chunks, used for cache file naming.
    """

    index: faiss.Index
    embeddings: np.ndarray
    chunks: list[str]
    fingerprint: str


# ── Public API ───────────────────────────────────────────────────────────────

def build_vector_store(chunks: list[str]) -> VectorStore:
    """
    Encode chunks into embeddings and build a FAISS L2 index.

    Steps:
        1. Embed all chunks → float32 numpy array of shape (n, 384)
        2. Create a FAISS IndexFlatL2 (exact nearest-neighbour, L2 distance)
        3. Add all embeddings to the index
        4. Return a VectorStore containing the index, embeddings, and chunks

    Args:
        chunks: Non-empty list of cleaned text chunks.

    Returns:
        A populated VectorStore ready for retrieval.

    Raises:
        ValueError: If chunks is empty.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    model = get_embedding_model()

    # encode() returns a (n_chunks, 384) float32 numpy array
    embeddings: np.ndarray = model.encode(
        chunks,
        batch_size=64,          # process in batches to avoid OOM on large docs
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    # IndexFlatL2 performs exact brute-force L2 search.
    # For hundreds of chunks this is instant; no training step needed.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    fingerprint = _chunks_fingerprint(chunks)
    return VectorStore(
        index=index,
        embeddings=embeddings,
        chunks=chunks,
        fingerprint=fingerprint,
    )


def save_vector_store(store: VectorStore) -> None:
    """
    Persist a VectorStore to disk so it survives app restarts.

    Saves two files under faiss_index/:
        <fingerprint>.index        — binary FAISS index (faiss.write_index)
        <fingerprint>.chunks.json  — the original chunk texts as JSON

    Args:
        store: A populated VectorStore returned by build_vector_store().
    """
    index_path, chunks_path = _index_paths(store.fingerprint)

    try:
        faiss.write_index(store.index, str(index_path))
        chunks_path.write_text(
            json.dumps(store.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise OSError(f"Failed to save vector store to disk: {exc}") from exc


def load_vector_store(chunks: list[str]) -> VectorStore | None:
    """
    Attempt to load a previously saved VectorStore that matches these chunks.

    Returns None if no matching index exists on disk, which signals the caller
    to build a fresh one.

    Args:
        chunks: The same chunk list that was used when the index was saved.

    Returns:
        A VectorStore if a cached index was found, otherwise None.
    """
    fingerprint = _chunks_fingerprint(chunks)
    index_path, chunks_path = _index_paths(fingerprint)

    if not index_path.exists() or not chunks_path.exists():
        return None

    try:
        index = faiss.read_index(str(index_path))
        saved_chunks: list[str] = json.loads(
            chunks_path.read_text(encoding="utf-8")
        )

        # Rebuild the embeddings array from the index's stored vectors
        # so the VectorStore is fully hydrated without re-encoding.
        n_vectors = index.ntotal
        dimension = index.d
        embeddings = faiss.rev_swig_ptr(
            index.get_xb(), n_vectors * dimension
        ).reshape(n_vectors, dimension).copy()
        embeddings = embeddings.astype(np.float32)

        return VectorStore(
            index=index,
            embeddings=embeddings,
            chunks=saved_chunks,
            fingerprint=fingerprint,
        )
    except Exception:
        # Corrupted or incompatible index file — discard and rebuild
        _safe_remove(index_path)
        _safe_remove(chunks_path)
        return None


def get_or_build_vector_store(chunks: list[str]) -> tuple[VectorStore, bool]:
    """
    Load a cached VectorStore if available, otherwise build and save a new one.

    This is the primary entry point used by the Streamlit app. It abstracts
    the load → miss → build → save flow into a single call.

    Args:
        chunks: Cleaned text chunks from the current document.

    Returns:
        (store, was_cached) where was_cached=True means the index was
        loaded from disk rather than freshly computed.
    """
    cached = load_vector_store(chunks)
    if cached is not None:
        return cached, True

    store = build_vector_store(chunks)
    save_vector_store(store)
    return store, False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_remove(path: Path) -> None:
    """Delete a file without raising if it does not exist."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
