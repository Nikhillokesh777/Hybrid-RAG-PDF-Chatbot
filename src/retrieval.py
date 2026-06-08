"""Semantic retrieval pipeline: query embedding → FAISS search → ranked results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from src.vector_store import VectorStore, get_embedding_model

# ── Constants ────────────────────────────────────────────────────────────────

# Default L2 distance threshold.
# L2 distance is unbounded, but in practice with 'all-MiniLM-L6-v2':
#   < 0.5  → highly relevant
#   0.5–1.0 → moderately relevant
#   > 1.0  → likely irrelevant
#
# This default is intentionally conservative. Users can tune it via the UI.
DEFAULT_SIMILARITY_THRESHOLD: float = 1.2

# Number of characters shown in the chunk preview inside the UI expander
PREVIEW_LENGTH: int = 300


# ── Status enum ──────────────────────────────────────────────────────────────

class RetrievalStatus(Enum):
    """Describes the outcome of a retrieval call."""

    SUCCESS          = auto()  # At least one chunk passed the threshold
    BELOW_THRESHOLD  = auto()  # All chunks exceeded the L2 distance threshold
    EMPTY_STORE      = auto()  # The vector store contained no vectors
    EMPTY_QUERY      = auto()  # The question string was blank


# ── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A single chunk returned by the retrieval pipeline.

    Attributes:
        rank:        1-based relevance rank (1 = most similar to query).
        text:        Full cleaned chunk text.
        preview:     First PREVIEW_LENGTH characters — used in the UI so the
                     expander does not dump an entire wall of text.
        l2_distance: Raw L2 distance from the query vector. Lower = more
                     semantically similar.
        passed_threshold: True when l2_distance is below the configured limit.
    """

    rank: int
    text: str
    preview: str
    l2_distance: float
    passed_threshold: bool


@dataclass
class RetrievalResult:
    """
    Complete output of one retrieval query.

    Attributes:
        status:   RetrievalStatus enum — tells the caller exactly what happened.
        chunks:   All retrieved RetrievedChunk objects (including filtered ones).
        context:  Chunks that passed the threshold, joined with double newlines.
                  Empty string when status is not SUCCESS.
        query:    The original question string.
        hit_count: Number of chunks that passed the threshold.

    Future-ready fields (unused in Phase 3, wired in Phase 4+):
        source_doc_ids: Placeholder for multi-PDF document tracking.
        conversation_id: Placeholder for conversational memory integration.
    """

    status: RetrievalStatus
    chunks: list[RetrievedChunk]
    context: str
    query: str
    hit_count: int

    # Phase 4+ placeholders — already typed so adding multi-PDF / memory
    # support later requires no dataclass restructuring.
    source_doc_ids: list[str] = field(default_factory=list)
    conversation_id: str | None = None

    @property
    def has_context(self) -> bool:
        """True when the result carries usable document context."""
        return self.status == RetrievalStatus.SUCCESS and bool(self.context)


# ── Core retrieval function ──────────────────────────────────────────────────

def retrieve(
    query: str,
    store: VectorStore,
    k: int = 3,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> RetrievalResult:
    """
    Retrieve the top-k semantically relevant chunks for a natural language query.

    Pipeline:
        1. Validate inputs → return early with descriptive status if invalid
        2. Embed query → (1, 384) float32 vector via SentenceTransformer
        3. FAISS index.search(k) → distances + indices arrays
        4. Map indices → chunk texts, compute previews
        5. Apply similarity_threshold — chunks with L2 distance above the
           threshold are included in result.chunks but excluded from context
        6. Return RetrievalResult with full ranking visibility

    Threshold logic:
        L2 distance measures how far apart two vectors are in 384-dim space.
        A lower distance = more semantically similar.
        If ALL retrieved chunks exceed the threshold, status is BELOW_THRESHOLD
        and context is empty, triggering the general knowledge fallback in app.py.

    Args:
        query:                User's natural language question.
        store:                Populated VectorStore from vector_store.py.
        k:                    Maximum chunks to retrieve (clamped to store size).
        similarity_threshold: Maximum L2 distance a chunk may have to be
                              included in the context passed to Gemini.

    Returns:
        RetrievalResult with status, ranked chunks, and filtered context.
    """
    # ── Guard: blank query ───────────────────────────────────────────────────
    if not query or not query.strip():
        return RetrievalResult(
            status=RetrievalStatus.EMPTY_QUERY,
            chunks=[],
            context="",
            query=query,
            hit_count=0,
        )

    # ── Guard: empty store ───────────────────────────────────────────────────
    if store.index.ntotal == 0:
        return RetrievalResult(
            status=RetrievalStatus.EMPTY_STORE,
            chunks=[],
            context="",
            query=query,
            hit_count=0,
        )

    model = get_embedding_model()

    # ── Step 1: embed the query ──────────────────────────────────────────────
    query_vector: np.ndarray = model.encode(
        [query.strip()],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # ── Step 2: FAISS search ─────────────────────────────────────────────────
    safe_k = min(k, store.index.ntotal)
    distances, indices = store.index.search(query_vector, safe_k)

    # ── Step 3: build ranked chunk list ─────────────────────────────────────
    all_chunks: list[RetrievedChunk] = []

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0 or idx >= len(store.chunks):
            continue  # FAISS sentinel — fewer results than k

        text = store.chunks[idx]
        l2 = float(dist)
        passed = l2 <= similarity_threshold

        all_chunks.append(
            RetrievedChunk(
                rank=rank,
                text=text,
                preview=_make_preview(text),
                l2_distance=l2,
                passed_threshold=passed,
            )
        )

    # ── Step 4: build context from passing chunks only ───────────────────────
    passing = [c for c in all_chunks if c.passed_threshold]
    context = "\n\n".join(c.text for c in passing)

    status = RetrievalStatus.SUCCESS if passing else RetrievalStatus.BELOW_THRESHOLD

    return RetrievalResult(
        status=status,
        chunks=all_chunks,       # full list for UI visibility (all ranks shown)
        context=context,         # only passing chunks go to Gemini
        query=query,
        hit_count=len(passing),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_preview(text: str, length: int = PREVIEW_LENGTH) -> str:
    """
    Return a short readable preview of a chunk for display in the UI.

    Tries to end the preview on a sentence boundary rather than mid-word.
    """
    if len(text) <= length:
        return text

    truncated = text[:length]

    # Prefer ending at the last sentence boundary within the preview
    last_sentence_end = max(
        truncated.rfind(". "),
        truncated.rfind("? "),
        truncated.rfind("! "),
    )

    if last_sentence_end > length // 2:
        return truncated[: last_sentence_end + 1] + " ..."

    # Fall back to last word boundary
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space] + " ..."

    return truncated + "..."
