# backend/rag_service_no_faiss.py
"""
RAG backend that DOES NOT use FAISS.
Instead it computes embeddings for chunks at startup (SentenceTransformer)
and does an in-memory cosine-similarity retrieval using numpy.

This avoids native faiss/OpenMP issues (mutex.cc) that can hang uvicorn
on some macOS/arch builds.
"""
import os
import json
import time
import logging
from typing import List, Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# -------- CONFIG --------
HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS_JSON_PATH = os.path.join(HERE, "ml_book_chunks.json")
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_MODEL_NAME = "gemma-3-12b-it"   # change if needed

TOP_K_DEFAULT = 3
MAX_CONTEXT_CHARS = 15000
REQUEST_TIMEOUT = 60  # seconds

# -------- Logging --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service_no_faiss")

# -------- FastAPI app --------
app = FastAPI(title="RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Data holders (populated at startup) --------
chunks: List[str] = []
embeddings: Optional[np.ndarray] = None  # shape (n_chunks, dim)
embed_model = None

# -------- Pydantic request body --------
class Query(BaseModel):
    question: str
    top_k: int = TOP_K_DEFAULT

# -------- Utility functions --------
def safe_load_chunks(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunks JSON not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Chunks JSON must be a list of strings.")
    return data

def compute_embeddings_for_chunks(model, chunks_list: List[str], batch_size: int = 64) -> np.ndarray:
    # returns numpy array of shape (n_chunks, dim), dtype float32
    emb_list = []
    n = len(chunks_list)
    for i in range(0, n, batch_size):
        batch = chunks_list[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        emb_list.append(emb.astype("float32"))
    return np.vstack(emb_list)

def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms

def retrieve_top_k(query: str, top_k: int = TOP_K_DEFAULT) -> List[int]:
    """
    Embed the query and return indices of top_k most similar chunks (cosine similarity).
    """
    global embeddings, embed_model
    if embeddings is None or embed_model is None:
        raise RuntimeError("Embeddings not loaded.")
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    # normalize query and embeddings
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    # cosine similarity via dot product (embeddings normalized)
    sims = np.dot(embeddings, q_emb.T).squeeze()  # shape (n_chunks,)
    if sims.ndim == 0:
        sims = np.array([sims])
    # get top_k indices
    top_k = min(len(sims), max(1, top_k))
    top_indices = np.argpartition(-sims, top_k-1)[:top_k]
    # sort them by score descending
    top_indices = top_indices[np.argsort(-sims[top_indices])]
    return top_indices.tolist()

def truncate_context(chunks_list: List[str], max_chars: int = MAX_CONTEXT_CHARS) -> List[str]:
    out = []
    total = 0
    for c in chunks_list:
        if total + len(c) <= max_chars:
            out.append(c)
            total += len(c)
        else:
            remaining = max_chars - total
            if remaining > 100:
                out.append(c[:remaining])
            break
    return out

def build_prompt(question: str, retrieved_chunks_texts: List[str]) -> str:
    safe_chunks = truncate_context(retrieved_chunks_texts)
    context = "\n\n".join(safe_chunks)
    prompt = f"""You are a college AI tutor.
Answer the question strictly from the provided syllabus context.
If the answer is not present in the context, say: "This topic is not covered in the syllabus."

Context:
{context}

Question:
{question}

Answer clearly and simply (suitable for a 3rd-year engineering student)."""
    return prompt

def ask_lmstudio(prompt: str) -> str:
    payload = {
        "model": LM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful college tutor."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    try:
        r = requests.post(LMSTUDIO_URL, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        # Bubble up helpful message
        try:
            detail = e.response.json()
        except Exception:
            detail = str(e)
        logger.exception("LM Studio returned HTTPError")
        raise HTTPException(status_code=502, detail=f"LM Studio HTTP error: {detail}")
    except requests.exceptions.RequestException as e:
        logger.exception("LM Studio request failed.")
        raise HTTPException(status_code=502, detail=f"LM Studio request failed: {e}")

# -------- Startup event: load chunks and compute embeddings --------
@app.on_event("startup")
def startup_load():
    global chunks, embeddings, embed_model
    logger.info("Startup: loading chunks and embedding model (no faiss).")
    try:
        chunks = safe_load_chunks(CHUNKS_JSON_PATH)
    except Exception as e:
        logger.exception("Failed to load chunks JSON.")
        # keep server up but mark chunks empty
        chunks = []
        return

    logger.info(f"Loaded {len(chunks)} chunks from JSON.")
    # load embed model
    try:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded SentenceTransformer model.")
    except Exception as e:
        logger.exception("Failed to load embedder.")
        raise

    # compute embeddings (in-memory)
    try:
        embeddings = compute_embeddings_for_chunks(embed_model, chunks, batch_size=64)
        embeddings = normalize_embeddings(embeddings)
        logger.info(f"Computed embeddings shape: {embeddings.shape}")
    except Exception as e:
        logger.exception("Failed to compute embeddings.")
        embeddings = None

# -------- Health endpoint --------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_count": len(chunks),
        "embeddings_loaded": embeddings is not None,
        "model": LM_MODEL_NAME
    }

# -------- /ask endpoint --------
class QueryIn(BaseModel):
    question: str
    top_k: int = TOP_K_DEFAULT

@app.post("/ask")
def ask(query: QueryIn):
    q = query.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not loaded; backend startup failed.")
    top_k = max(1, min(query.top_k, 20))
    try:
        idxs = retrieve_top_k(q, top_k=top_k)
    except Exception as e:
        logger.exception("Retrieval failed.")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    retrieved_chunks = [chunks[i] for i in idxs]
    prompt = build_prompt(q, retrieved_chunks)
    # Ask the local LLM via LM Studio
    answer = ask_lmstudio(prompt)
    return {"answer": answer, "retrieved_count": len(retrieved_chunks)}
