import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import pypdf
import requests
import os

# ---------------------------------------------------------
# 1. Setup Groq API
# ---------------------------------------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # Store inside Streamlit Cloud Secrets
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"  # or llama3-70b if preferred

# ---------------------------------------------------------
# 2. Embedding Model
# ---------------------------------------------------------
EMBED_DIM = 384
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# 3. Load / Initialize FAISS
# ---------------------------------------------------------
INDEX_FILE = "faiss.index"
DOCS_FILE = "documents.json"
SRC_FILE = "doc_sources.json"

if Path(INDEX_FILE).exists():
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(EMBED_DIM)

if Path(DOCS_FILE).exists():
    documents = json.load(open(DOCS_FILE, "r", encoding="utf-8"))
else:
    documents = []

if Path(SRC_FILE).exists():
    doc_sources = json.load(open(SRC_FILE, "r", encoding="utf-8"))
else:
    doc_sources = []

# ---------------------------------------------------------
# 4. Utility Functions
# ---------------------------------------------------------
def save_metadata():
    json.dump(documents, open(DOCS_FILE, "w", encoding="utf-8"))
    json.dump(doc_sources, open(SRC_FILE, "w", encoding="utf-8"))

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def load_pdf(file):
    reader = pypdf.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def add_document(doc_id, text):
    global documents, doc_sources

    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks).astype("float32")

    index.add(embeddings)
    documents.extend(chunks)
    doc_sources.extend([doc_id] * len(chunks))

def retrieve_context(query, k=4):
    q_emb = embed_model.encode([query]).astype("float32")
    distances, idxs = index.search(q_emb, k)
    valid = [i for i in idxs[0] if i < len(documents)]
    return [documents[i] for i in valid]

def groq_chatcompletion(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    res = requests.post(GROQ_API_URL, headers=headers, json=data)

    if res.status_code != 200:
        return f"Groq API Error: {res.text}"

    return res.json()["choices"][0]["message"]["content"]

def generate_answer(question, context):
    prompt = f"""
You are a retrieval-augmented assistant.
Answer the question using ONLY the context below.

Question:
{question}

Context:
{context}

If the answer is not in the context, say: "Not found in documents."
"""
    return groq_chatcompletion(prompt)

# ---------------------------------------------------------
# 5. Streamlit UI
# ---------------------------------------------------------
st.title("⚡ RAG Chatbot (FAISS + MiniLM + Groq API)")
st.write("Runs vector search locally, LLM via Groq API.")

# Uploading documents
st.subheader("📄 Upload PDFs or TXT files")

uploaded_files = st.file_uploader(
    "Choose files", type=["pdf", "txt"], accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Indexing..."):
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = load_pdf(file)
            else:
                text = file.read().decode("utf-8")

            add_document(file.name, text)

        faiss.write_index(index, INDEX_FILE)
        save_metadata()

    st.success("Documents successfully indexed!")

st.markdown("---")

# Query
st.subheader("💬 Ask a question")

user_query = st.text_input("Your question")

if st.button("Ask"):
    if len(documents) == 0:
        st.error("Upload documents first.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            retrieved = retrieve_context(user_query)
            context = "\n\n".join(retrieved)

        with st.spinner("Generating answer using Groq..."):
            answer = generate_answer(user_query, context)

        st.markdown("### 🔍 Retrieved Context")
        st.info(context if context else "No relevant chunks found.")

        st.markdown("### 🤖 Answer")
        st.write(answer)
