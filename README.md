# Hybrid RAG PDF Assistant

An intelligent, document-grounded Question Answering application powered by **FastAPI**, **ChromaDB**, **Sentence-Transformers**, and **Google Gemini (2.5 Flash)**.

The system features a modern, responsive single-page web interface with real-time semantic search, citation provenance, sliding-window conversation memory, and executive document summarization.

---

## 📌 Features
- **Modern Responsive Web UI**: Glassmorphic dark theme, dynamic metrics, and chat history.
- **Persistent Vector Store**: ChromaDB storage with cosine similarity search and MD5 fingerprinting.
- **Sentence-Aware Chunking**: Context preservation with configurable chunk sizes and overlap.
- **LLM Judge & Fallback**: Dual-path routing—answers from document if context is sufficient; otherwise falls back safely to general knowledge with clear alerts.
- **Full REST API**: High-performance asynchronous FastAPI backend with interactive Swagger documentation (`/docs`).

---

## 🛠️ Technologies Used
- **Backend**: FastAPI, Uvicorn, Python 3.13
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla, Zero Heavy Frameworks)
- **Vector Databases**: ChromaDB (Primary), FAISS
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM**: Google Gemini API (`gemini-2.5-flash`)
- **Document Processing**: PyPDF2

---

## 📂 Project Structure
```text
pdf_rag_app/
│
├── main.py                     # Primary application launcher
├── server.py                   # FastAPI REST server & static file provider
├── requirements.txt            # Project dependencies
├── README.md                   # Quickstart guide
├── ARCHITECTURE.md             # In-depth architectural design & flows
├── .env.example                # Environment variable template
│
├── frontend/                   # Modern Single-Page Web Client
│   ├── index.html              # HTML5 application structure
│   ├── style.css               # Modern glassmorphic styles & animations
│   └── app.js                  # Frontend client logic & REST API bindings
│
├── src/                        # Headless Python Core RAG Engine
│   ├── chroma_store.py         # ChromaDB persistence & similarity retrieval
│   ├── vector_store.py         # FAISS vector store
│   ├── chunker.py              # Text chunking logic
│   ├── pdf_processor.py        # PDF text extraction
│   ├── text_cleaner.py         # Text normalization & cleaning
│   ├── retrieval.py            # Semantic retrieval & LLM Judge
│   ├── memory_manager.py       # Multi-turn conversation buffer
│   └── config.py               # Centralized configuration & hyper-parameters
│
├── chroma_db/                  # Persistent ChromaDB storage directory
└── archive/
    └── streamlit_legacy/       # Archived original Streamlit prototype
```

---

## ▶️ How to Run Locally

### 1️⃣ Clone & Setup Virtual Environment
```bash
git clone https://github.com/Nikhillokesh777/Hybrid-RAG-PDF-Chatbot.git
cd Hybrid-RAG-PDF-Chatbot

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Configure Gemini API Key
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3️⃣ Launch the Application
```bash
python main.py
```
Or with Uvicorn directly:
```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

- **Web Application**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Docs (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
