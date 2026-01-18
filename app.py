import os
import re
import streamlit as st
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv

# =========================================================
# Configuration
# =========================================================

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    models = genai.list_models()
    supported_models = [
        m.name for m in models
        if "generateContent" in m.supported_generation_methods
    ]
    MODEL_NAME = supported_models[0]
except Exception:
    MODEL_NAME = "models/gemini-flash-latest"

model = genai.GenerativeModel(MODEL_NAME)

# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(page_title="PDF Question Answering System", layout="wide")
st.title("PDF Question Answering System")
st.caption("Document-grounded answers with graceful fallback to general knowledge")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# =========================================================
# Utility Functions
# =========================================================

def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text content from a PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"

    return text


def chunk_text(text: str, chunk_size: int = 600) -> list[str]:
    """Splits text into fixed-size chunks."""
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def retrieve_relevant_chunks(chunks: list[str], question: str, k: int = 3) -> list[str]:
    """Selects top-k relevant chunks using keyword overlap."""
    question_terms = set(re.findall(r"\w+", question.lower()))
    scored = []

    for chunk in chunks:
        chunk_terms = set(re.findall(r"\w+", chunk.lower()))
        score = len(question_terms & chunk_terms)
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored[:k] if score > 0]


# =========================================================
# LLM Interaction Functions
# =========================================================

def context_contains_answer(context: str, question: str) -> bool:
    """Determines whether the provided context contains sufficient information."""
    prompt = f"""
Context:
{context}

Question:
{question}

Task:
Respond with only one word:
YES if the context contains sufficient information.
NO if it does not.
"""
    response = model.generate_content(prompt)
    return response.text.strip().upper() == "YES"


def generate_document_answer(context: str, question: str) -> str:
    """Answers strictly using the document context."""
    prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{question}

If the answer cannot be determined, state that it is not present in the document.
"""
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 1700, "temperature": 0.3}
    )
    return response.text


def generate_general_answer(question: str) -> str:
    """Provides a general knowledge explanation when document context is insufficient."""
    prompt = f"""
The document does not contain the required information.

Provide a clear and informative explanation based on general knowledge.
Limit the response to 5–8 well-structured sentences.

Question:
{question}
"""
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 400, "temperature": 0.4}
    )
    return response.text


def generate_summary(text: str) -> str:
    """Generates a concise summary of the document."""
    prompt = f"""
Provide a concise and well-structured summary of the following document:

{text}
"""
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 500, "temperature": 0.3}
    )
    return response.text


# =========================================================
# Application Logic
# =========================================================

if uploaded_file:
    with st.spinner("Extracting document text..."):
        document_text = extract_text_from_pdf(uploaded_file)

    if not document_text.strip():
        st.error("No readable text found in the document.")
        st.stop()

    st.success("Document loaded successfully.")

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            summary = generate_summary(document_text[:12000])
        st.subheader("Document Summary")
        st.write(summary)

    st.divider()

    chunks = chunk_text(document_text)
    question = st.text_input("Enter your question")

    if question:
        with st.spinner("Retrieving relevant content..."):
            relevant_chunks = retrieve_relevant_chunks(chunks, question)
            context = "\n\n".join(relevant_chunks)

        if context and context_contains_answer(context, question):
            with st.spinner("Generating document-based answer..."):
                answer = generate_document_answer(context, question)
            st.subheader("Answer (From Document)")
        else:
            with st.spinner("Generating general knowledge answer..."):
                answer = generate_general_answer(question)
            st.subheader("Answer (General Knowledge)")

        st.write(answer)
