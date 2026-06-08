"""
Hybrid RAG PDF Question Answering System
Stable production architecture: Retriever → Gemini

Pipeline per question:
    1. Embed question
    2. FAISS top-k semantic search
    3. Retrieved chunks sent directly to Gemini (no compression, no filtering)
    4. Gemini generates a grounded natural language answer
    5. Answer streamed to UI with citations and timing
"""

from __future__ import annotations

import logging
import time

import google.generativeai as genai
import streamlit as st

from src import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    GEMINI_FALLBACK_MODEL,
    GOOGLE_API_KEY,
    MemoryManager,
    PDFExtractionResult,
    RetrievalResult,
    RetrievalStatus,
    VectorStore,
    chunk_text,
    clean_chunks,
    clean_text,
    extract_text_from_pdf,
    get_or_build_vector_store,
    render_answer,
    render_chat_controls,
    render_chat_history,
    render_index_status,
    render_page_config,
    render_retrieval_panel,
    render_sidebar,
    render_stats,
    retrieve,
)
from src.config import (
    APP_CAPTION,
    APP_TITLE,
    GEMINI_ANSWER_MAX_TOKENS,
    GEMINI_ANSWER_TEMPERATURE,
    GEMINI_FALLBACK_MAX_TOKENS,
    GEMINI_FALLBACK_TEMPERATURE,
    GEMINI_SUMMARY_MAX_TOKENS,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Page + API guard ──────────────────────────────────────────────────────────
render_page_config(APP_TITLE, APP_CAPTION)

if not GOOGLE_API_KEY:
    st.error(
        "GOOGLE_API_KEY is missing. "
        "Add it to your .env file and restart the app."
    )
    st.stop()

# ── Gemini setup ──────────────────────────────────────────────────────────────
genai.configure(api_key=GOOGLE_API_KEY)

try:
    _model_name = next(
        m.name for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    )
except Exception:
    _model_name = GEMINI_FALLBACK_MODEL

gemini = genai.GenerativeModel(_model_name)
logger.info("Gemini model: %s", _model_name)

# ── Session state ─────────────────────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager()
if "vector_store" not in st.session_state:
    st.session_state.vector_store: VectorStore | None = None
if "active_doc_names" not in st.session_state:
    st.session_state.active_doc_names: list[str] = []

memory: MemoryManager = st.session_state.memory

# ── Sidebar ───────────────────────────────────────────────────────────────────
top_k, similarity_threshold = render_sidebar(_model_name)


# =========================================================
# Cached PDF pipeline
# =========================================================

@st.cache_data(show_spinner=False)
def _process_single_pdf(pdf_bytes: bytes) -> tuple[PDFExtractionResult, list[str]]:
    """Extract → clean → chunk one PDF. Result cached by file content hash."""
    extraction = extract_text_from_pdf(pdf_bytes)
    cleaned    = clean_text(extraction.text)
    raw_chunks = chunk_text(cleaned, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    chunks     = clean_chunks(raw_chunks)
    return extraction, chunks


def _merge_pdfs(
    files: list,
) -> tuple[list[PDFExtractionResult], list[str], int, int]:
    """Process all uploaded PDFs and merge their chunks into one flat list."""
    extractions: list[PDFExtractionResult] = []
    all_chunks:  list[str]                 = []
    total_pages  = 0
    total_chars  = 0

    for f in files:
        extraction, chunks = _process_single_pdf(f.getvalue())
        extractions.append(extraction)
        all_chunks.extend(chunks)
        total_pages += extraction.page_count
        total_chars += extraction.character_count

    return extractions, all_chunks, total_pages, total_chars


# =========================================================
# Gemini answer generation
# =========================================================

def _judge(context: str, question: str) -> tuple[bool, bool]:
    """
    Ask Gemini whether the retrieved context is sufficient to answer
    the question.

    Returns:
        (can_answer, judge_failed)
    """
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Respond with only YES if the context contains sufficient "
        "information to answer the question, otherwise respond with NO."
    )
    try:
        r = gemini.generate_content(prompt)
        return r.text.strip().upper() == "YES", False
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return False, True


def _generate_document_answer(
    context: str,
    question: str,
    conversation_history: str,
) -> str:
    """
    Generate a Gemini answer grounded in the retrieved document chunks.

    The retrieved chunks are passed to Gemini exactly as returned by FAISS —
    no compression, no sentence filtering, no summarization. This preserves
    full semantic continuity and gives Gemini the most complete context.
    """
    history_block = f"\n\n{conversation_history}\n\n" if conversation_history else "\n\n"
    prompt = (
        "Answer the question using only the information in the context below.\n"
        "If the answer is not present in the context, say so clearly.\n"
        f"{history_block}"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )
    try:
        r = gemini.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": GEMINI_ANSWER_MAX_TOKENS,
                "temperature":       GEMINI_ANSWER_TEMPERATURE,
            },
        )
        return r.text
    except Exception as exc:
        logger.error("Document answer generation failed: %s", exc)
        return f"Error generating answer: {exc}"


def _generate_general_answer(
    question: str,
    conversation_history: str,
) -> str:
    """Fall back to Gemini general knowledge when the document lacks context."""
    history_block = f"\n\n{conversation_history}\n\n" if conversation_history else "\n\n"
    prompt = (
        "The uploaded document does not contain sufficient information "
        "to answer this question.\n\n"
        "Provide a clear, informative response based on general knowledge "
        f"in 5-8 well-structured sentences.{history_block}"
        f"Question:\n{question}"
    )
    try:
        r = gemini.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": GEMINI_FALLBACK_MAX_TOKENS,
                "temperature":       GEMINI_FALLBACK_TEMPERATURE,
            },
        )
        return r.text
    except Exception as exc:
        logger.error("General answer generation failed: %s", exc)
        return f"Error generating answer: {exc}"


def _generate_summary(text: str) -> str:
    """Generate a concise Gemini summary of the document."""
    try:
        r = gemini.generate_content(
            f"Provide a concise, well-structured summary of this document:\n\n{text}",
            generation_config={
                "max_output_tokens": GEMINI_SUMMARY_MAX_TOKENS,
                "temperature":       GEMINI_ANSWER_TEMPERATURE,
            },
        )
        return r.text
    except Exception as exc:
        logger.error("Summary generation failed: %s", exc)
        return f"Error generating summary: {exc}"


# =========================================================
# Main UI
# =========================================================

# ── Multi-PDF upload ──────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload at least one PDF to get started.")
    st.stop()

# ── Process PDFs ──────────────────────────────────────────────────────────────
with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
    extractions, all_chunks, total_pages, total_chars = _merge_pdfs(uploaded_files)

doc_names = [f.name for f in uploaded_files]
st.session_state.active_doc_names = doc_names

had_error = False
for f, extraction in zip(uploaded_files, extractions):
    for err in extraction.errors:
        st.error(f"**{f.name}**: {err}")
        had_error = True
    for warn in extraction.warnings:
        st.warning(f"**{f.name}**: {warn}")

if had_error or not all_chunks:
    st.error("One or more files could not be processed. Please check your PDFs.")
    st.stop()

st.success(
    f"✅ {len(uploaded_files)} file(s) processed — "
    f"{len(all_chunks)} chunks ready."
)

# ── Stats dashboard ───────────────────────────────────────────────────────────
avg_chunk = int(sum(len(c) for c in all_chunks) / len(all_chunks)) if all_chunks else 0
render_stats(total_pages, total_chars, len(all_chunks), avg_chunk, len(uploaded_files))
st.caption(f"📊 Estimated tokens: {total_chars // 4:,}")

with st.expander("🔎 Sample Chunk Preview"):
    st.caption(f"Chunk 1 of {len(all_chunks)} · {len(all_chunks[0])} characters")
    st.text(all_chunks[0][:800] + ("..." if len(all_chunks[0]) > 800 else ""))

# ── Summary ───────────────────────────────────────────────────────────────────
if st.button("📝 Generate Document Summary"):
    with st.spinner("Generating summary..."):
        summary = _generate_summary(" ".join(all_chunks)[:12000])
    st.subheader("Document Summary")
    st.write(summary)

st.divider()

# ── Vector store ──────────────────────────────────────────────────────────────
with st.spinner("Preparing semantic index..."):
    try:
        t0 = time.perf_counter()
        vector_store, was_cached = get_or_build_vector_store(all_chunks)
        st.session_state.vector_store = vector_store
        logger.info(
            "Index ready: %d vectors in %.0f ms (cached=%s)",
            vector_store.index.ntotal,
            (time.perf_counter() - t0) * 1000,
            was_cached,
        )
    except Exception as exc:
        st.error(f"Failed to build vector store: {exc}")
        logger.exception("Vector store build failed.")
        st.stop()

render_index_status(vector_store.index.ntotal, vector_store.index.d, was_cached)

# ── Conversation history ──────────────────────────────────────────────────────
render_chat_history(memory)
render_chat_controls(memory)

if not memory.is_empty:
    st.divider()

# ── Question input ────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your document(s)...")

if not question:
    st.stop()

# Echo user message
with st.chat_message("user"):
    st.markdown(question)

# ── Step 1: Semantic retrieval ────────────────────────────────────────────────
with st.spinner("Searching document semantically..."):
    t_retrieval = time.perf_counter()
    try:
        result: RetrievalResult = retrieve(
            query=question,
            store=vector_store,
            k=top_k,
            similarity_threshold=similarity_threshold,
        )
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        logger.exception("Retrieval error.")
        st.stop()

    retrieval_ms = (time.perf_counter() - t_retrieval) * 1000
    logger.info(
        "Retrieval: %d chunks in %.0f ms (status=%s)",
        result.hit_count, retrieval_ms, result.status.name,
    )

# Show retrieval panel (what FAISS found, scores, pass/filter badges)
render_retrieval_panel(result, similarity_threshold, retrieval_ms)

# Threshold / empty-store warnings
if result.status == RetrievalStatus.EMPTY_STORE:
    st.error("Vector store is empty. Please re-upload your PDFs.")
    st.stop()

if result.status == RetrievalStatus.BELOW_THRESHOLD:
    st.warning(
        f"No retrieved chunk was within the similarity threshold "
        f"({similarity_threshold:.2f}). "
        "Try raising the threshold in the sidebar, or rephrase your question."
    )

# ── Step 2: Judge + Gemini answer ─────────────────────────────────────────────
conversation_history = memory.build_context_string()
t_gen = time.perf_counter()

if result.has_context:
    can_answer, judge_failed = _judge(result.context, question)
    if judge_failed:
        st.warning(
            "Could not verify context relevance — falling back to general knowledge."
        )
else:
    can_answer   = False
    judge_failed = False

from_document = result.has_context and can_answer

with st.chat_message("assistant"):
    with st.spinner("Generating answer..."):
        if from_document:
            # Pass the full retrieved chunks directly to Gemini —
            # no sentence filtering, no compression, no summarization.
            answer = _generate_document_answer(
                result.context, question, conversation_history
            )
        else:
            answer = _generate_general_answer(question, conversation_history)

    gen_ms = (time.perf_counter() - t_gen) * 1000
    logger.info(
        "Answer generated in %.0f ms (from_document=%s)", gen_ms, from_document
    )

    render_answer(
        answer=answer,
        from_document=from_document,
        result=result,
        elapsed_ms=gen_ms,
    )

# ── Step 3: Save turn to memory ───────────────────────────────────────────────
memory.add_turn(
    question=question,
    answer=answer,
    source="document" if from_document else "general_knowledge",
    doc_names=doc_names,
)
logger.info("Turn saved. Total turns: %d", memory.turn_count)
