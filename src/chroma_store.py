"""ChromaDB embedded vector store: persistence, indexing, and similarity retrieval."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    CHROMA_DISTANCE_METRIC,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
)

logger = logging.getLogger(__name__)

# Embedding model singleton
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# ChromaDB client singleton
_chroma_client: Any | None = None


def get_chroma_client():
    """Return the persistent ChromaDB client located on the D: drive."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb

        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        logger.info("ChromaDB PersistentClient initialized at: %s", CHROMA_DB_DIR)
    return _chroma_client


def get_or_create_collection(
    name: str = CHROMA_COLLECTION_NAME,
    distance_metric: str = CHROMA_DISTANCE_METRIC,
):
    """Retrieve or create the ChromaDB collection configured for cosine distance."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": distance_metric},
    )


def compute_fingerprint(chunks: list[str]) -> str:
    """Generate a hash uniquely identifying a list of document chunks."""
    content = ("v2\n" + "\n".join(chunks)).encode("utf-8")
    return hashlib.md5(content).hexdigest()[:16]


@dataclass
class ChromaIndexAdapter:
    """
    Adapter providing .ntotal and .d properties so components that inspected
    the FAISS index continue to function seamlessly with ChromaDB.
    """

    collection: Any
    dimension: int = 384
    chunk_count: int = 0

    @property
    def ntotal(self) -> int:
        if self.chunk_count > 0:
            return self.chunk_count
        return self.collection.count()

    @property
    def total_collection_vectors(self) -> int:
        return self.collection.count()

    @property
    def d(self) -> int:
        return self.dimension


@dataclass
class ChromaVectorStore:
    """
    Persistent ChromaDB vector store wrapper.

    Attributes:
        collection: The ChromaDB Collection object.
        chunks: The list of raw text chunks associated with the active session.
        fingerprint: Unique identifier for the chunk set.
        dimension: Embedding vector dimension (384 for all-MiniLM-L6-v2).
        index: Adapter exposing .ntotal and .d for legacy UI compatibility.
    """

    collection: Any
    chunks: list[str]
    fingerprint: str
    dimension: int = 384
    index: ChromaIndexAdapter = field(init=False)

    def __post_init__(self) -> None:
        self.index = ChromaIndexAdapter(
            collection=self.collection,
            dimension=self.dimension,
            chunk_count=len(self.chunks),
        )


def build_chroma_vector_store(
    chunks: list[str],
    doc_name: str = "document",
) -> ChromaVectorStore:
    """
    Encode chunks and persist them into ChromaDB.

    Args:
        chunks: Cleaned document text chunks.
        doc_name: Document name for chunk metadata tagging.

    Returns:
        ChromaVectorStore populated and persisted in ChromaDB.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    fp = compute_fingerprint(chunks)
    collection = get_or_create_collection()
    model = get_embedding_model()

    chunk_ids = [f"{fp}_{i}" for i in range(len(chunks))]

    logger.info("Encoding %d chunks with %s...", len(chunks), EMBEDDING_MODEL_NAME)
    embeddings: np.ndarray = model.encode(
        chunks,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    metadatas = [
        {
            "fingerprint": fp,
            "chunk_index": i,
            "doc_name": doc_name,
            "char_count": len(c),
        }
        for i, c in enumerate(chunks)
    ]

    # Batch upsert into ChromaDB
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=chunk_ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info(
        "ChromaDB collection '%s' now holds %d items.",
        collection.name,
        collection.count(),
    )
    return ChromaVectorStore(
        collection=collection,
        chunks=chunks,
        fingerprint=fp,
    )


def load_chroma_vector_store(chunks: list[str]) -> ChromaVectorStore | None:
    """
    Check if chunks for this fingerprint already exist in ChromaDB.
    If fully indexed, return ChromaVectorStore without re-embedding.
    """
    if not chunks:
        return None

    fp = compute_fingerprint(chunks)
    collection = get_or_create_collection()

    chunk_ids = [f"{fp}_{i}" for i in range(len(chunks))]
    try:
        # Check first and last chunks, or all if small
        check_ids = chunk_ids if len(chunk_ids) <= 100 else [chunk_ids[0], chunk_ids[-1]]
        existing = collection.get(ids=check_ids, include=[])
        if len(existing.get("ids", [])) == len(check_ids):
            # For verification, check full count matching this fingerprint
            match_records = collection.get(
                where={"fingerprint": fp},
                include=[],
            )
            if len(match_records.get("ids", [])) == len(chunks):
                logger.info(
                    "Found complete cached ChromaDB index for fingerprint %s (%d chunks).",
                    fp,
                    len(chunks),
                )
                return ChromaVectorStore(
                    collection=collection,
                    chunks=chunks,
                    fingerprint=fp,
                )
    except Exception as exc:
        logger.warning("Error checking ChromaDB cache: %s", exc)

    return None


def get_or_build_vector_store(
    chunks: list[str],
    doc_name: str = "document",
) -> tuple[ChromaVectorStore, bool]:
    """
    Primary entry point: check ChromaDB cache, or build and persist new chunks.

    Returns:
        (store, was_cached)
    """
    cached = load_chroma_vector_store(chunks)
    if cached is not None:
        return cached, True

    store = build_chroma_vector_store(chunks, doc_name=doc_name)
    return store, False
