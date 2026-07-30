import os
import time
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Global state built once at startup
index = None
chunks = []


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    result = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        result.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return result


def get_embedding(text, task_type):
    result = client.models.embed_content(model="gemini-embedding-001", contents=text)
    return result.embeddings[0].values


@app.on_event("startup")
def build_index():
    global index, chunks
    pdf_path = os.getenv("PDF_PATH", "document.pdf")

    reader = PdfReader(pdf_path)
    full_text = "".join(page.extract_text() + "\n" for page in reader.pages)

    chunks = chunk_text(full_text)

    embeddings = []
    for c in chunks:
        embeddings.append(get_embedding(c, "retrieval_document"))
        time.sleep(1)  # gentle pacing to avoid rate limits during startup

    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print(f"Index built with {index.ntotal} chunks from {pdf_path}")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    if index is None:
        raise HTTPException(status_code=503, detail="Index not ready yet")

    query_embedding = get_embedding(request.question, "retrieval_query")
    query_vector = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_vector, request.top_k)
    retrieved = [chunks[i] for i in indices[0]]

    context = "\n\n".join(retrieved)
    prompt = f"""Answer the question using only the context below. If the context does not contain the answer, say you don't have enough information.

Context:
{context}

Question: {request.question}

Answer:"""

    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)

    return QueryResponse(answer=response.text, sources=[s[:150] for s in retrieved])


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": index is not None}


app.mount("/", StaticFiles(directory="static", html=True), name="static")