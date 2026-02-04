import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from pathlib import Path
import json
import pypdf

# ---------------------------------------------------------
# 1. Initialize Embedding Model
# ---------------------------------------------------------
EMBED_DIM = 384  # all-MiniLM-L6-v2 dimension
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# 2. Load / Initialize FAISS Index
# ---------------------------------------------------------
INDEX_FILE = "faiss.index"
DOCS_FILE = "documents.json"
SRC_FILE = "doc_sources.json"

# Load FAISS index if exists
if Path(INDEX_FILE).exists():
    index = faiss.read_index(INDEX_FILE)
    st.sidebar.success("FAISS index loaded.")
else:
    index = faiss.IndexFlatL2(EMBED_DIM)
    st.sidebar.info("Created a new FAISS index.")

# Load documents metadata
if Path(DOCS_FILE).exists():
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)
else:
    documents = []

if Path(SRC_FILE).exists():
    with open(SRC_FILE, "r", encoding="utf-8") as f:
        doc_sources = json.load(f)
else:
    doc_sources = []

# ---------------------------------------------------------
# 3. Utilities
# ---------------------------------------------------------
def save_metadata():
    """Saves documents + sources to disk so FAISS remains in sync."""
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f)

    with open(SRC_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_sources, f)

def chunk_text(text, chunk_size=500, overlap=50):
    """Split large text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def load_pdf(file):
    """Extract raw text from a PDF file."""
    reader = pypdf.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def add_document(doc_id, text):
    """Add a document to FAISS + metadata."""
    global documents, doc_sources
    chunks = chunk_text(text)

    embeddings = embed_model.encode(chunks).astype("float32")
    index.add(embeddings)

    documents.extend(chunks)
    doc_sources.extend([doc_id] * len(chunks))

def retrieve_context(query, k=4):
    """Return top-k relevant document chunks."""
    q_emb = embed_model.encode([query]).astype("float32")
    distances, idxs = index.search(q_emb, k)

    valid_idxs = [i for i in idxs[0] if i < len(documents)]
    return [documents[i] for i in valid_idxs]

def generate_answer(question, context):
    """Generate answer using local Ollama model."""
    prompt = f"""
You are an offline document assistant.
Answer using ONLY this context.

Question:
{question}

Context:
{context}

If answer is not found in the context, respond: "Not found in documents."
"""
    response = ollama.generate(
        model="mistral",
        prompt=prompt
    )
    return response["response"]

# ---------------------------------------------------------
# 4. Streamlit UI
# ---------------------------------------------------------
st.title("🔒 Offline RAG Chatbot (FAISS + MiniLM + Ollama)")
st.write("Runs 100% locally. No internet or paid APIs required.")

# --- Document Upload ---
st.subheader("📄 Upload Documents for Indexing")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = load_pdf(file)
            else:
                text = file.read().decode("utf-8")

            add_document(file.name, text)

        # Save FAISS index + metadata
        faiss.write_index(index, INDEX_FILE)
        save_metadata()

    st.success("Documents successfully indexed!")

st.markdown("---")

# --- Query Interface ---
st.subheader("💬 Ask a Question About the Documents")

user_query = st.text_input("Enter your question:")

if st.button("Ask"):
    if len(documents) == 0:
        st.error("Please upload documents first.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            retrieved = retrieve_context(user_query)
            context = "\n\n".join(retrieved)

        with st.spinner("Generating answer..."):
            answer = generate_answer(user_query, context)

        st.markdown("### 🔍 Retrieved Context")
        st.info(context if context else "No relevant context found.")

        st.markdown("### 🤖 Answer")
        st.write(answer)
