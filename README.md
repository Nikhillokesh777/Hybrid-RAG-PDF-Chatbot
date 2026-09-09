# Hybrid RAG PDF Chatbot

This project is a web-based PDF Question Answering system.  
Users can upload a PDF file and ask questions. The system tries to answer using the document content first. If the answer is not found in the document, it falls back to general knowledge.

The application is built using Python and Streamlit.

---

## 📌 Features
- Upload PDF files
- Extract text from PDFs
- Split text into chunks
- Retrieve relevant chunks based on the question
- Decide whether the answer comes from the document or general knowledge
- Display the final answer in a web interface

---

## 🧩 How the System Works (High Level)
1. User uploads a PDF
2. Text is extracted from the PDF
3. Text is divided into smaller chunks
4. Relevant chunks are selected based on the question
5. A decision is made:
   - If the answer exists in the document → use document data
   - Else → use general knowledge
6. The answer is displayed to the user

---

## 🛠️ Technologies Used
- Python
- Streamlit
- Google Gemini API
- PyPDF2
- python-dotenv

---

## 📂 Project Structure
```text
Hybrid-RAG-PDF-Chatbot/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── README.md               # Quickstart guide
├── ARCHITECTURE.md         # Full implementation architecture & system flow
├── .env.example            # Environment variable template
├── faiss_index/            # Persistent vector database storage
└── src/                    # Core modules (processing, chunking, retrieval, memory, UI)
```

For detailed architecture diagrams, component breakdown, and end-to-end data flows, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## ▶️ How to Run the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Nikhillokesh777/Hybrid-RAG-PDF-Chatbot.git
cd Hybrid-RAG-PDF-Chatbot
```

python -m venv venv 

venv\Scripts\activate

pip install -r requirements.txt


---

**Create Google Gemini API Key:**
#Go to Google AI Studio

Sign in with your Google account

Generate a new API key

Copy the API key

GOOGLE_API_KEY=your_api_key_here

streamlit run app.py
