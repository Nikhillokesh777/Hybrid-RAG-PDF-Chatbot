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
rag-pdf-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore

---

## ▶️ How to Run the Project Locally

### 1️⃣ Clone the Repository

git clone https://github.com/your-username/hybrid-rag-pdf-chatbot.git
cd hybrid-rag-pdf-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

**Create Google Gemini API Key:**
Go to Google AI Studio
Sign in with your Google account
Generate a new API key
Copy the API key
GOOGLE_API_KEY=your_api_key_here
streamlit run app.py
