"""
FastAPI Server for PDF Question Answering System
Serves the modern HTML/CSS/JS frontend and exposes high-performance REST/SSE endpoints
powered by the persistent ChromaDB RAG pipeline on D: drive.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.chunker import chunk_text
from src.config import (
    CHROMA_DB_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    GEMINI_ANSWER_MAX_TOKENS,
    GEMINI_ANSWER_TEMPERATURE,
    GEMINI_FALLBACK_MAX_TOKENS,
    GEMINI_FALLBACK_MODEL,
    GEMINI_FALLBACK_TEMPERATURE,
    GEMINI_SUMMARY_MAX_TOKENS,
    GOOGLE_API_KEY,
)
from src.memory_manager import MemoryManager
from src.pdf_processor import PDFExtractionResult, extract_text_from_pdf
from src.retrieval import RetrievalResult, RetrievalStatus, retrieve
from src.text_cleaner import clean_chunks, clean_text
from src.vector_store import get_or_build_vector_store

# ── Setup & Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_server")

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"

# ── Gemini Setup & Multi-Model Pool ──────────────────────────────────────────
_model_name = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
if not _model_name.startswith("models/"):
    _model_name = f"models/{_model_name}"

CANDIDATE_MODELS: list[str] = [
    _model_name,
    "models/gemini-flash-latest",
    "models/gemini-3.5-flash",
    "models/gemini-3.7-flash",
    "models/gemini-flash-lite-latest",
    "models/gemini-2.5-flash",
]

if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai

        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Google Generative AI configured with primary model: %s", _model_name)
    except Exception as exc:
        logger.warning("Gemini configuration warning: %s", exc)


def generate_content_with_failover(
    prompt: str,
    generation_config: dict | None = None,
) -> tuple[str, str]:
    """
    Generate content with Gemini, automatically failing over across models
    if a 429 quota, 404 deprecation, or rate limit error occurs.

    Returns:
        (response_text, model_name_used)
    """
    global _model_name
    import google.generativeai as genai

    seen = set()
    models_to_try = [m for m in CANDIDATE_MODELS if m and not (m in seen or seen.add(m))]

    last_exc = None
    for model_name in models_to_try:
        try:
            m = genai.GenerativeModel(model_name)
            resp = m.generate_content(prompt, generation_config=generation_config)
            _model_name = model_name
            return resp.text, model_name
        except Exception as exc:
            err_str = str(exc)
            last_exc = exc
            logger.warning(
                "Gemini model '%s' failed (%s). Attempting failover to next candidate...",
                model_name,
                err_str[:120],
            )
            continue

    logger.error("All Gemini candidate models failed. Last exception: %s", last_exc)
    raise HTTPException(
        status_code=500,
        detail=f"Gemini generation error across all models: {last_exc}",
    )


# ── Global Server Session State ───────────────────────────────────────────────
class ServerState:
    vector_store: Any = None
    all_chunks: list[str] = []
    active_doc_names: list[str] = []
    stats: dict[str, Any] = {}
    memory: MemoryManager = MemoryManager()


state = ServerState()

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PDF RAG Neural Search",
    description="ChromaDB Persistent Vector Store & Gemini 2.5 Flash",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static directory
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)
    similarity_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD, ge=0.1, le=2.0)


# ── Overview Query Detection ──────────────────────────────────────────────────
def _is_summary_or_overview_query(question: str) -> bool:
    q = question.lower().strip()
    keywords = [
        "summary", "summarize", "summarise", "overview", "highlight",
        "main topic", "what is this document", "about this document",
        "outline", "key finding", "key point", "executive summary",
        "topics covered", "what does this document discuss", "tell me about this document",
        "main conclusions", "key data points",
    ]
    return any(kw in q for kw in keywords)


def _build_overview_context(chunks: list[str], max_chars: int = 11000) -> str:
    if not chunks:
        return ""
    if len(chunks) <= 6:
        return "\n\n".join(chunks)

    selected = [chunks[0], chunks[1]]
    step = max(1, (len(chunks) - 2) // 4)
    for i in range(2, len(chunks) - 1, step):
        if len(selected) < 6:
            selected.append(chunks[i])

    if chunks[-1] not in selected:
        selected.append(chunks[-1])

    return "\n\n".join(selected)[:max_chars]


def _judge_context(context: str, question: str, best_dist: float = 1.0) -> bool:
    """
    Evaluate whether retrieved context contains sufficient info.
    Quota optimization: If best chunk has strong cosine similarity (dist <= 0.65),
    it is strong evidence; skip the extra LLM call to save quota.
    """
    if not context or not context.strip():
        return False
    if best_dist <= 0.65:
        return True

    prompt = f"""You are a strict retrieval evaluator.
Determine whether the provided Context contains sufficient information to answer the Question.
Respond with EXACTLY one word: YES or NO.

Context:
{context[:2500]}

Question: {question}
Answer:"""
    try:
        ans, _ = generate_content_with_failover(
            prompt,
            generation_config={"max_output_tokens": 10, "temperature": 0.0},
        )
        return "yes" in ans.strip().lower()
    except Exception:
        return True  # Fail-safe to document context


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    """Serve the single-page HTML5 client application."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(str(index_path))


@app.get("/api/status")
async def get_status():
    """Return system readiness, ChromaDB vector count, and active documents."""
    total_vectors = 0
    if state.vector_store:
        total_vectors = getattr(state.vector_store.index, "ntotal", 0)

    return {
        "status": "ready",
        "model": _model_name,
        "chroma_db_dir": str(CHROMA_DB_DIR),
        "total_vectors": total_vectors,
        "active_docs": state.active_doc_names,
        "stats": state.stats,
    }


@app.post("/api/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """
    Accept one or multiple PDF files, extract, clean, chunk,
    and persist into ChromaDB on D: drive.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    all_chunks: list[str] = []
    total_pages = 0
    total_chars = 0
    doc_names: list[str] = []

    for f in files:
        content = await f.read()
        extraction: PDFExtractionResult = extract_text_from_pdf(content)
        if not extraction.has_text:
            logger.warning("No text found in file: %s", f.filename)
            continue

        cleaned = clean_text(extraction.text)
        raw_chunks = chunk_text(cleaned, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        valid_chunks = clean_chunks(raw_chunks)

        all_chunks.extend(valid_chunks)
        total_pages += extraction.page_count
        total_chars += extraction.character_count
        doc_names.append(f.filename or "uploaded_file.pdf")

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No readable text could be extracted from the uploaded PDF(s).")

    # Index into ChromaDB
    t0 = time.perf_counter()
    doc_label = doc_names[0] if doc_names else "document"
    vector_store, was_cached = get_or_build_vector_store(all_chunks, doc_name=doc_label)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    state.vector_store = vector_store
    state.all_chunks = all_chunks
    state.active_doc_names = doc_names

    avg_chunk = int(sum(len(c) for c in all_chunks) / len(all_chunks)) if all_chunks else 0
    state.stats = {
        "file_count": len(doc_names),
        "total_pages": total_pages,
        "total_chars": total_chars,
        "total_chunks": len(all_chunks),
        "avg_chunk_size": avg_chunk,
        "was_cached": was_cached,
        "index_ms": round(elapsed_ms, 1),
    }

    logger.info("Upload processed %d files (%d chunks) in %.1fms (cached=%s)", len(doc_names), len(all_chunks), elapsed_ms, was_cached)

    return {
        "success": True,
        "doc_names": doc_names,
        "stats": state.stats,
    }


@app.post("/api/query")
async def query_rag(req: QueryRequest):
    """
    Execute semantic search in ChromaDB, evaluate relevance,
    and generate a grounded Gemini response.
    """
    if not state.vector_store:
        raise HTTPException(status_code=400, detail="No active document. Please upload a PDF first.")

    question = req.question.strip()
    is_overview = _is_summary_or_overview_query(question)

    retrieval_res: RetrievalResult = retrieve(
        query=question,
        store=state.vector_store,
        k=req.top_k,
        similarity_threshold=req.similarity_threshold,
    )

    # Overview inquiry handling
    best_dist = retrieval_res.chunks[0].l2_distance if retrieval_res.chunks else 1.0
    if is_overview and len(state.all_chunks) > 0:
        overview_ctx = _build_overview_context(state.all_chunks)
        effective_context = overview_ctx
        has_sufficient_context = True
    else:
        effective_context = retrieval_res.context
        has_sufficient_context = (
            _judge_context(effective_context, question, best_dist=best_dist)
            if retrieval_res.has_context
            else False
        )

    # Multi-turn memory injection
    convo_history = state.memory.build_context_string()

    # Prompt assembly
    if has_sufficient_context and effective_context:
        source_type = "document"
        citations = state.active_doc_names or ["Uploaded Document"]
        prompt = f"""You are a precise, professional AI document assistant.
Answer the question thoroughly and accurately based STRICTLY on the document context provided below.
If the context contains relevant information, answer directly with clear formatting, bullet points where appropriate, and crisp explanations.

Context:
{effective_context}

{convo_history}
Question: {question}
Answer:"""
        temp = GEMINI_ANSWER_TEMPERATURE
        max_tokens = GEMINI_ANSWER_MAX_TOKENS
    else:
        source_type = "general"
        citations = []
        prompt = f"""You are a knowledgeable AI assistant.
The document does not appear to contain sufficient details to directly answer the user's question.
Provide a clear, helpful answer from your general knowledge, and briefly mention that this information is drawn from general knowledge rather than the uploaded document.

{convo_history}
Question: {question}
Answer:"""
        temp = GEMINI_FALLBACK_TEMPERATURE
        max_tokens = GEMINI_FALLBACK_MAX_TOKENS

    # Generate answer with multi-model failover protection
    answer, used_model = generate_content_with_failover(
        prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": temp},
    )

    # Track in conversation memory
    state.memory.add_turn(
        question=question,
        answer=answer,
        source=source_type,
        doc_names=citations,
    )

    # Format chunks for client diagnostics
    retrieved_chunks = [
        {
            "rank": c.rank,
            "preview": c.preview,
            "l2_distance": round(c.l2_distance, 4),
            "passed_threshold": c.passed_threshold,
            "cosine_similarity": round(c.cosine_similarity, 4),
        }
        for c in retrieval_res.chunks
    ]

    return {
        "answer": answer,
        "source_type": source_type,
        "citations": citations,
        "retrieved_chunks": retrieved_chunks,
        "model": used_model,
    }


@app.post("/api/summary")
async def generate_document_summary():
    """Generate a concise executive summary of the uploaded document."""
    if not state.all_chunks:
        raise HTTPException(status_code=400, detail="No active document found to summarize.")

    overview_text = " ".join(state.all_chunks)[:14000]
    prompt = f"Provide a concise, well-structured executive summary of this document:\n\n{overview_text}"

    ans, used_model = generate_content_with_failover(
        prompt,
        generation_config={
            "max_output_tokens": GEMINI_SUMMARY_MAX_TOKENS,
            "temperature": GEMINI_ANSWER_TEMPERATURE,
        },
    )
    return {"summary": ans, "model": used_model}


@app.get("/api/history")
async def get_chat_history():
    """Return the current conversation thread."""
    return state.memory.turns


@app.delete("/api/history")
async def clear_chat_history():
    """Clear the active conversation memory."""
    state.memory.clear()
    return {"message": "Chat history cleared successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
