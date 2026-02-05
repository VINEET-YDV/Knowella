# Knowella

Deployed Link - https://knowella-y5inzrprwdpskxhat8tkq6.streamlit.app/

# ⚡ Offline RAG Chatbot (FAISS + MiniLM + Groq API + Streamlit)

A fast, lightweight Retrieval-Augmented Generation (RAG) chatbot using:

- **FAISS** — Local vector search  
- **all-MiniLM-L6-v2** — SentenceTransformer embeddings  
- **Groq API** — Ultra-fast LLM inference (llama) 
- **Streamlit** — Simple, interactive web UI  
- **PDF/TXT ingestion** — Upload and index documents dynamically  

Runs locally or fully in **Streamlit Cloud**, since generation uses the Groq API.

---

## 🚀 Features

- 🧠 **RAG-style answers** grounded ONLY in your uploaded documents  
- 🔍 **FAISS vector search** for fast & offline retrieval  
- ✨ **Groq API** for extremely fast LLM responses  
- 📄 Upload **PDF** or **TXT** files  
- 💾 Persistent indexing using:
  - `faiss.index`
  - `documents.json`
  - `doc_sources.json`
- 🌐 Deployable on **Streamlit Cloud**  
- 🖥 Works locally without cloud (except LLM generation)  

---

## 📦 Tech Stack

| Component | Technology |
|----------|------------|
| UI | Streamlit |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS |
| LLM | Groq API (Mixtral) |
| File Parsing | PyPDF |
