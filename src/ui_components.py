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
from src.styles import get_custom_css


# ── Page config & Hero Banner ─────────────────────────────────────────────────

def render_page_config(title: str, caption: str) -> None:
    """Configure the Streamlit page, inject custom CSS, and render the hero banner."""
    st.set_page_config(
        page_title=title,
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Inject design system stylesheet
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Render modern hero header banner
    st.markdown(
        f"""
        <div class="hero-wrapper">
            <div class="hero-badge">
                <span class="badge-dot"></span>
                Hybrid RAG · Neural Search Engine
            </div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Welcome & Empty State ─────────────────────────────────────────────────────

def render_empty_state() -> None:
    """Render a professional welcome card when no PDF document is uploaded."""
    st.markdown(
        """
        <div class="empty-state-card">
            <div class="empty-state-icon">📄✨</div>
            <h2 class="empty-state-title">Welcome to Document Intelligence</h2>
            <p class="empty-state-desc">
                Upload one or multiple PDF documents above to build an ultra-fast semantic index. 
                Ask complex questions and receive grounded, cited answers backed by Google Gemini.
            </p>
            <div class="features-grid">
                <div class="feature-pill">
                    <div class="feature-pill-title">📑 Multi-PDF Ingestion</div>
                    <div class="feature-pill-desc">
                        Seamlessly merge, sanitize, and chunk text across multiple files with sentence-aware boundaries.
                    </div>
                </div>
                <div class="feature-pill">
                    <div class="feature-pill-title">⚡ Sub-millisecond Search</div>
                    <div class="feature-pill-desc">
                        384-dimensional dense vectors powered by FAISS L2 exact nearest-neighbor search.
                    </div>
                </div>
                <div class="feature-pill">
                    <div class="feature-pill-title">🤖 Dual-Path Reasoning</div>
                    <div class="feature-pill-desc">
                        Strict document grounding with automated LLM Judge verification and general knowledge fallback.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(gemini_model_name: str) -> tuple[int, float]:
    """
    Render the modern sidebar settings panel.

    Returns:
        (top_k, similarity_threshold)
    """
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-card-title">⚙️ Control Center</div>
                <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                    Configure semantic retrieval sensitivity and context size.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Semantic Retrieval Settings**")
        top_k = st.slider(
            "Top-K Chunks",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            help="Number of nearest chunks retrieved by FAISS per question.",
        )

        similarity_threshold = st.slider(
            "Similarity Threshold (L2 Distance)",
            min_value=0.1,
            max_value=2.0,
            value=DEFAULT_SIMILARITY_THRESHOLD,
            step=0.05,
            help=(
                "Maximum Euclidean distance allowed for a chunk to reach Gemini. "
                "Lower is stricter: <0.5 very close · 0.5–1.0 moderate · >1.0 distant."
            ),
        )

        # Visual threshold explainer
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 0.75rem; margin-top: 0.4rem; font-size: 0.78rem;">
                <div style="color: #94a3b8; margin-bottom: 0.25rem;">Current threshold cutoff: <b style="color: #6366f1;">{similarity_threshold:.2f}</b></div>
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <span style="color: #10b981;">● &lt;0.5 High</span>
                    <span style="color: #f59e0b;">● 0.5-1.0 Med</span>
                    <span style="color: #f43f5e;">● &gt;1.0 Low</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Engine Architecture Card
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-card-title">⚡ Engine Specs</div>
                <div style="display: flex; flex-direction: column; gap: 0.5rem; font-size: 0.8rem;">
                    <div>
                        <span style="color: #64748b;">Embeddings:</span><br/>
                        <code style="color: #38bdf8;">{EMBEDDING_MODEL_NAME}</code>
                    </div>
                    <div>
                        <span style="color: #64748b;">Dimension:</span><br/>
                        <code style="color: #a78bfa;">384-d dense float32</code>
                    </div>
                    <div>
                        <span style="color: #64748b;">Generative LLM:</span><br/>
                        <code style="color: #34d399;">{gemini_model_name}</code>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return top_k, similarity_threshold


# ── Document stats strip ──────────────────────────────────────────────────────

def render_stats(
    page_count: int,
    char_count: int,
    chunk_count: int,
    avg_chunk_chars: int,
    file_count: int,
) -> None:
    """Render the responsive, glassmorphic 5-column document statistics strip."""
    st.markdown(
        f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="icon-wrapper">📁</div>
                <div class="metric-label">Documents</div>
                <div class="metric-value">{file_count}</div>
            </div>
            <div class="metric-card">
                <div class="icon-wrapper">📑</div>
                <div class="metric-label">Pages</div>
                <div class="metric-value">{page_count}</div>
            </div>
            <div class="metric-card">
                <div class="icon-wrapper">🔤</div>
                <div class="metric-label">Characters</div>
                <div class="metric-value">{char_count:,}</div>
            </div>
            <div class="metric-card">
                <div class="icon-wrapper">🧩</div>
                <div class="metric-label">Chunks</div>
                <div class="metric-value">{chunk_count:,}</div>
            </div>
            <div class="metric-card">
                <div class="icon-wrapper">📏</div>
                <div class="metric-label">Avg Chunk Size</div>
                <div class="metric-value">{avg_chunk_chars} <span style="font-size:0.8rem; font-weight:normal; color:#64748b;">ch</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Retrieval panel ───────────────────────────────────────────────────────────

def render_retrieval_panel(
    result: RetrievalResult,
    similarity_threshold: float,
    elapsed_ms: float,
) -> None:
    """
    Expandable panel showing every retrieved chunk with visual confidence bars,
    L2 distance scores, and pass / filter status badges.
    """
    passing = sum(1 for c in result.chunks if c.passed_threshold)
    total   = len(result.chunks)

    label = (
        f"🔍 Semantic Retrieval Diagnostics  ·  {passing}/{total} chunks passed "
        f"(≤ {similarity_threshold:.2f})  ·  ⏱️ {elapsed_ms:.1f} ms"
    )

    with st.expander(label, expanded=False):
        if not result.chunks:
            st.info("No candidate chunks found in the vector space.")
            return

        for chunk in result.chunks:
            # Calculate normalized similarity percentage for visual bar (0 to 100%)
            # Where L2 = 0 is 100% confidence and L2 >= 2.0 is 0%
            confidence = max(0.0, min(100.0, (1.0 - (chunk.l2_distance / 2.0)) * 100.0))
            
            bar_color = "#10b981" if chunk.passed_threshold else "#f43f5e"
            badge_class = "badge-doc" if chunk.passed_threshold else "badge-gen"
            badge_text = "✓ PASSED TO GEMINI" if chunk.passed_threshold else "✗ FILTERED OUT"

            st.markdown(
                f"""
                <div class="retrieval-card">
                    <div class="retrieval-header">
                        <div>
                            <span style="font-weight: 700; color: #f1f5f9; font-size: 0.95rem;">Rank #{chunk.rank}</span>
                            <span style="color: #64748b; font-size: 0.8rem; margin-left: 0.5rem;">L2 Distance: <b>{chunk.l2_distance:.4f}</b></span>
                        </div>
                        <span class="attribution-pill {badge_class}">{badge_text}</span>
                    </div>
                    <div style="font-size: 0.76rem; color: #94a3b8; display: flex; justify-content: space-between;">
                        <span>Relevance Confidence</span>
                        <span>{confidence:.1f}%</span>
                    </div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width: {confidence}%; background: {bar_color};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Excerpt ({len(chunk.text):,} total characters):")
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
    Render the Gemini answer with an attribution banner, streaming effect,
    inference duration chips, and expandable source citations.
    """
    if from_document:
        st.markdown(
            f"""
            <div class="attribution-banner attribution-document">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>📎</span>
                    <span><strong>Grounded Response:</strong> Verified against uploaded document context</span>
                </div>
                <span class="attribution-pill badge-doc">Verified Document</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="attribution-banner attribution-general">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>🌐</span>
                    <span><strong>General Knowledge Fallback:</strong> Document did not contain sufficient information</span>
                </div>
                <span class="attribution-pill badge-gen">General Knowledge</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Word-by-word streaming animation
    stream_text(answer)

    st.markdown(
        f"""
        <div style="display: flex; gap: 1rem; margin-top: 0.6rem; font-size: 0.78rem; color: #64748b;">
            <span>⏱️ Latency: <b>{elapsed_ms:.0f} ms</b></span>
            <span>📊 Est. Tokens: <b>{len(answer) // 4:,}</b></span>
            <span>🤖 Model: <b>Gemini 1.5</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Citations accordion — only displayed when grounded in document
    used = [c for c in result.chunks if c.passed_threshold and from_document]
    if used:
        with st.expander(f"📚 Grounded Citations ({len(used)} source chunks)", expanded=False):
            for chunk in used:
                st.markdown(
                    f"**Source Chunk #{chunk.rank}** &nbsp;·&nbsp; "
                    f"L2 Distance: `{chunk.l2_distance:.4f}`"
                )
                st.text(chunk.preview)
                if chunk.rank < len(used):
                    st.divider()


# ── Chat history ──────────────────────────────────────────────────────────────

def render_chat_history(memory: MemoryManager) -> None:
    """Render all prior turns with custom styling and origin badges."""
    if memory.is_empty:
        return

    st.markdown("### 💬 Conversation Thread")

    for turn in memory.turns:
        with st.chat_message("user", avatar="👤"):
            st.markdown(turn.question)

        with st.chat_message("assistant", avatar="⚡"):
            if turn.source == "document":
                st.markdown(
                    """<div style="font-size: 0.75rem; color: #10b981; font-weight: 600; margin-bottom: 0.3rem;">
                    📎 GROUNDED DOCUMENT ANSWER</div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """<div style="font-size: 0.75rem; color: #f59e0b; font-weight: 600; margin-bottom: 0.3rem;">
                    🌐 GENERAL KNOWLEDGE FALLBACK</div>""",
                    unsafe_allow_html=True,
                )
            
            st.markdown(turn.answer)
            st.caption(
                f"{turn.timestamp} · "
                f"Active Documents: {', '.join(turn.doc_names) or 'None'}"
            )


# ── Chat controls ─────────────────────────────────────────────────────────────

def render_chat_controls(memory: MemoryManager) -> None:
    """Render Clear Chat, Export JSON, and Export TXT action buttons."""
    if memory.is_empty:
        return

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("🗑️ Clear Thread", use_container_width=True):
            memory.clear()
            st.rerun()

    with col2:
        st.download_button(
            label="⬇️ Export JSON",
            data=memory.export_as_json(),
            file_name="rag_chat_history.json",
            mime="application/json",
            use_container_width=True,
        )

    with col3:
        st.download_button(
            label="⬇️ Export TXT",
            data=memory.export_as_text(),
            file_name="rag_chat_history.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ── Index status ──────────────────────────────────────────────────────────────

def render_index_status(ntotal: int, dimension: int, was_cached: bool) -> None:
    """Render a clean FAISS index status badge."""
    label = "⚡ Loaded instantly from disk cache" if was_cached else "🔨 Built & persisted to disk"
    icon = "💾" if was_cached else "⚙️"
    st.markdown(
        f"""
        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 10px; padding: 0.6rem 1rem; margin: 0.8rem 0; font-size: 0.82rem; color: #c7d2fe; display: flex; align-items: center; gap: 0.5rem;">
            <span>{icon}</span>
            <span><b>FAISS Index Active:</b> {ntotal:,} vectors ({dimension}-dim) · {label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Streaming writer ──────────────────────────────────────────────────────────

def stream_text(text: str, delay: float = 0.01) -> None:
    """Render text with a smooth word-by-word typewriter effect."""
    placeholder = st.empty()
    displayed = ""

    for word in text.split():
        displayed += word + " "
        placeholder.markdown(displayed + "▌")
        time.sleep(delay)

    placeholder.markdown(displayed.strip())
