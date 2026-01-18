<<<<<<< HEAD
# Hybrid-RAG-PDF-Chatbot
PDF question answering system using a hybrid RAG approach with document-based answers and general knowledge fallback.
=======
# Hybrid RAG PDF Chatbot

A web-based Question Answering system built using a **Hybrid Retrieval-Augmented Generation (RAG)** approach.  
The application allows users to upload PDF documents and ask natural language questions, generating accurate answers grounded in document content with a fallback to general knowledge when needed.

---

## 🚀 Features
- PDF upload and text extraction
- Text chunking for efficient retrieval
- Top-k semantic retrieval
- Hybrid decision logic (document-based vs general knowledge)
- Answer generation using LLMs
- Summary generation
- Interactive web UI using Streamlit

---

## 🧠 Architecture Overview
1. PDF Loader extracts text
2. Chunking splits text into smaller segments
3. Retriever fetches top-k relevant chunks
4. Judge decides answer source (PDF or LLM knowledge)
5. Generator produces final response
6. Summary module provides document overview

---

## 🛠️ Tech Stack
- Python
- Streamlit
- Google Gemini / LLM API
- PyPDF2

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/your-username/hybrid-rag-pdf-chatbot.git
cd hybrid-rag-pdf-chatbot
pip install -r requirements.txt
- generate your own api key and store in GOOGLE_API_KEY= 
streamlit run app.py
>>>>>>> 883309d (Initial commit: Hybrid RAG PDF Chatbot)
