"""Reusable Streamlit UI components for the PDF RAG application."""

from __future__ import annotations

import time

import streamlit as st

from src.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
)
from src.memory_manager import MemoryManager
from src.retrieval import RetrievalResult, RetrievalStatus


# ── Page config ───────────────────────────────────────────────────────────────

def render_page_config(title: str, caption: str) -> None:
    """Configure the Streamlit page and render the main header."""
    st.set_page_config(
        page_title=title,
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title(f"📄 {title}")
    st.caption(caption)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(gemini_model_name: str) -> tuple[int, float]:
    """
    Render the sidebar settings panel.

    Returns:
        (top_k, similarity_threshold)
    """
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("Retrieval")
        top_k = st.slider(
            "Top-K chunks",
            min_value=1, max_value=10, value=DEFAULT_TOP_K,
            help=(
                "How many chunks FAISS retrieves per question. "
                "More chunks = more context for Gemini."
            ),
        )
        similarity_threshold = st.slider(
            "Similarity threshold (L2)",
            min_value=0.1, max_value=2.0,
            value=DEFAULT_SIMILARITY_THRESHOLD, step=0.05,
            help=(
                "Maximum L2 distance allowed for a chunk to reach Gemini. "
                "Lower = stricter.  <0.5 very close · 0.5–1.0 moderate · >1.0 distant."
            ),
        )

        st.divider()
        st.subheader("ℹ️ Model Info")
        st.caption(f"🔤 Embeddings: `{EMBEDDING_MODEL_NAME}`")
        st.caption(f"🤖 Gemini: `{gemini_model_name}`")

    return top_k, similarity_threshold


# ── Document stats strip ──────────────────────────────────────────────────────

def render_stats(
    page_count: int,
    char_count: int,
    chunk_count: int,
    avg_chunk_chars: int,
    file_count: int,
) -> None:
    """Render the five-column document statistics strip."""
    cols = st.columns(5)
    cols[0].metric("📄 Files",      file_count)
    cols[1].metric("📑 Pages",      page_count)
    cols[2].metric("🔤 Characters", f"{char_count:,}")
    cols[3].metric("🧩 Chunks",     chunk_count)
    cols[4].metric("📏 Avg Chunk",  f"{avg_chunk_chars} ch")


# ── Retrieval panel ───────────────────────────────────────────────────────────

def render_retrieval_panel(
    result: RetrievalResult,
    similarity_threshold: float,
    elapsed_ms: float,
) -> None:
    """
    Expandable panel showing every retrieved chunk with L2 scores and
    pass / filter badges so the user can see exactly what reached Gemini.
    """
    passing = sum(1 for c in result.chunks if c.passed_threshold)
    total   = len(result.chunks)

    label = (
        f"🔍 Retrieved Chunks  ·  {passing}/{total} passed threshold "
        f"(≤ {similarity_threshold:.2f})  ·  {elapsed_ms:.0f} ms"
    )

    with st.expander(label, expanded=False):
        if not result.chunks:
            st.info("No chunks were retrieved.")
            return

        for chunk in result.chunks:
            colour = "green" if chunk.passed_threshold else "red"
            badge  = "✅ sent to Gemini" if chunk.passed_threshold else "❌ filtered out"

            st.markdown(
                f"**Rank {chunk.rank}** &nbsp;|&nbsp; "
                f"L2 distance: "
                f"<span style='color:{colour}; font-weight:bold'>"
                f"{chunk.l2_distance:.4f}</span> "
                f"&nbsp;&nbsp;{badge}",
                unsafe_allow_html=True,
            )
            st.caption(f"{len(chunk.text):,} characters in this chunk")
            st.text(chunk.preview)

            if chunk.rank < total:
                st.divider()


# ── Answer + citations ────────────────────────────────────────────────────────

def render_answer(
    answer: str,
    from_document: bool,
    result: RetrievalResult,
    elapsed_ms: float,
) -> None:
    """
    Render the Gemini answer with source badge, streaming effect,
    response time, and collapsible source citations.
    """
    if from_document:
        st.subheader("💬 Answer — From Document")
        st.info("📎 This answer is grounded in your uploaded document.")
    else:
        st.subheader("💬 Answer — General Knowledge")
        st.warning(
            "⚠️ The document did not contain sufficient information. "
            "Answering from general knowledge."
        )

    stream_text(answer)
    st.caption(f"⏱️ Generated in {elapsed_ms:.0f} ms · Est. tokens: {len(answer) // 4:,}")

    # Citations — only shown for document-grounded answers
    used = [c for c in result.chunks if c.passed_threshold and from_document]
    if used:
        with st.expander(f"📚 Source Citations ({len(used)} chunks used)"):
            for chunk in used:
                st.markdown(
                    f"**Citation {chunk.rank}** &nbsp;|&nbsp; "
                    f"L2: `{chunk.l2_distance:.4f}`"
                )
                st.text(chunk.preview)
                if chunk.rank < len(used):
                    st.divider()


# ── Chat history ──────────────────────────────────────────────────────────────

def render_chat_history(memory: MemoryManager) -> None:
    """Render all prior turns as chat bubbles above the input box."""
    if memory.is_empty:
        return

    st.subheader("💬 Conversation History")

    for turn in memory.turns:
        with st.chat_message("user"):
            st.markdown(turn.question)

        with st.chat_message("assistant"):
            icon = "📄" if turn.source == "document" else "🌐"
            st.markdown(turn.answer)
            st.caption(
                f"{icon} {turn.source.replace('_', ' ').title()} · "
                f"{turn.timestamp} · "
                f"Docs: {', '.join(turn.doc_names) or 'none'}"
            )


# ── Chat controls ─────────────────────────────────────────────────────────────

def render_chat_controls(memory: MemoryManager) -> None:
    """Render Clear Chat, Export JSON, and Export TXT buttons."""
    if memory.is_empty:
        return

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            memory.clear()
            st.rerun()

    with col2:
        st.download_button(
            label="⬇️ Export JSON",
            data=memory.export_as_json(),
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True,
        )

    with col3:
        st.download_button(
            label="⬇️ Export TXT",
            data=memory.export_as_text(),
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ── Index status ──────────────────────────────────────────────────────────────

def render_index_status(ntotal: int, dimension: int, was_cached: bool) -> None:
    """Render the one-line FAISS index status caption."""
    label = "⚡ Loaded from cache" if was_cached else "🔨 Built and saved"
    st.caption(
        f"🗄️ Semantic index ready · {ntotal:,} vectors · "
        f"{dimension}-dim · {label}"
    )


# ── Streaming writer ──────────────────────────────────────────────────────────

def stream_text(text: str, delay: float = 0.012) -> None:
    """
    Render text with a word-by-word streaming effect.

    Writes each word into a Streamlit placeholder with a blinking cursor
    to simulate live token generation. Replace with st.write_stream() +
    a Gemini streaming response for true server-sent streaming.
    """
    placeholder = st.empty()
    displayed   = ""

    for word in text.split():
        displayed += word + " "
        placeholder.markdown(displayed + "▌")
        time.sleep(delay)

    placeholder.markdown(displayed.strip())
