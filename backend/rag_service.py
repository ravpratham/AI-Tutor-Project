# backend/rag_service.py
import json
import numpy as np
import faiss
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import logging
import os

# -------- CONFIG --------
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_MODEL_NAME = "gemma-3-12b-it"          # <-- change this to your LM Studio model name
HERE = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(HERE, "ml_book.index")
CHUNKS_JSON_PATH = os.path.join(HERE, "ml_book_chunks.json")

TOP_K = 3                  # default number of chunks to retrieve
MAX_CONTEXT_CHARS = 15000  # simple guard to keep prompt size reasonable
REQUEST_TIMEOUT = 60       # seconds to wait for LM Studio responses

# -------- Logging --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")

# -------- FastAPI app --------
app = FastAPI(title="RAG Service")

# Allow local frontend to call the API. Adjust origins for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Load resources --------
logger.info("Loading FAISS index and chunks...")
try:
    index = faiss.read_index(FAISS_INDEX_PATH)
    chunks = json.load(open(CHUNKS_JSON_PATH, "r", encoding="utf-8"))
    logger.info(f"Loaded index and {len(chunks)} chunks.")
except Exception as e:
    logger.exception("Failed to load index/chunks. Make sure the files exist.")
    raise

# Use the same embedder used to make the index
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------- Pydantic request body --------
class Query(BaseModel):
    question: str
    top_k: int = TOP_K

# -------- Helper functions --------
def retrieve_chunks(query: str, top_k: int = TOP_K):
    """Return a list of top_k chunk texts relevant to the query."""
    q_emb = embed_model.encode([query])
    distances, indices = index.search(np.array(q_emb).astype('float32'), top_k)
    idx_list = indices[0].tolist()
    # filter invalid (faiss may return -1)
    idx_list = [i for i in idx_list if i >= 0 and i < len(chunks)]
    return [chunks[i] for i in idx_list]

def truncate_context(chunks_list, max_chars=MAX_CONTEXT_CHARS):
    """
    Join chunks but ensure the total character length doesn't exceed max_chars.
    Uses greedy concatenation of top chunks and truncates last chunk if necessary.
    """
    out = []
    total = 0
    for c in chunks_list:
        if total + len(c) <= max_chars:
            out.append(c)
            total += len(c)
        else:
            remaining = max_chars - total
            if remaining > 100:  # keep at least a little context from the last chunk
                out.append(c[:remaining])
            break
    return out

def build_prompt(question: str, retrieved_chunks: list[str]):
    """
    Build a safe prompt that instructs the LLM to only use the provided context,
    and to decline if the topic isn't covered.
    """
    # truncate context if too big
    safe_chunks = truncate_context(retrieved_chunks)
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

def ask_lmstudio(prompt: str):
    """Call the local LM Studio chat completions endpoint."""
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
        # defensive access
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        # Handle specific 400 error for context length overflow
        if e.response is not None and e.response.status_code == 400:
            try:
                error_data = e.response.json()
                error_msg = str(error_data)
                # Check if it's a context length overflow error
                if "context length" in error_msg.lower() or "context overflow" in error_msg.lower() or "not enough" in error_msg.lower():
                    logger.error(f"Context length overflow error from LM Studio: {error_msg}")
                    raise HTTPException(
                        status_code=413,
                        detail="Context length overflow: The input is too long for the model's context window. "
                               "Please reduce the number of context chunks (top_k) or load the model with a larger context length in LM Studio."
                    )
                else:
                    logger.exception(f"LM Studio returned 400 error: {error_msg}")
                    raise HTTPException(status_code=400, detail=f"LM Studio error: {error_msg}")
            except ValueError:
                # If response is not JSON, use the raw text
                error_text = e.response.text if e.response else str(e)
                logger.exception(f"LM Studio returned 400 error with non-JSON response: {error_text}")
                # Check if error text contains context length info
                if "context length" in error_text.lower() or "context overflow" in error_text.lower():
                    raise HTTPException(
                        status_code=413,
                        detail="Context length overflow: The input is too long for the model's context window. "
                               "Please reduce the number of context chunks (top_k) or load the model with a larger context length in LM Studio."
                    )
                raise HTTPException(status_code=400, detail=f"LM Studio error: {error_text}")
        else:
            status_code = e.response.status_code if e.response else 502
            logger.exception(f"LM Studio HTTP error ({status_code}): {str(e)}")
            raise HTTPException(status_code=502, detail=f"LM Studio error {status_code}: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.exception("LM Studio request failed.")
        raise HTTPException(status_code=502, detail="LM Studio error: " + str(e))
    except Exception as e:
        logger.exception("Unexpected response format from LM Studio.")
        raise HTTPException(status_code=502, detail="Invalid response from LM Studio")

# -------- API endpoint --------
@app.post("/ask")
def ask(query: Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    top_k = max(1, min(query.top_k, 10))
    retrieved = retrieve_chunks(query.question, top_k=top_k)
    if len(retrieved) == 0:
        return {"answer": "No relevant content found in the knowledge base.", "retrieved_count": 0}
    prompt = build_prompt(query.question, retrieved)
    answer = ask_lmstudio(prompt)
    return {"answer": answer, "retrieved_count": len(retrieved)}

@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(chunks)}


