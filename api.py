# api.py

import os
import uuid
import tempfile
import shutil
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVector, models

# --- main.py se existing functions/values import kiye, kuch bhi duplicate nahi likha ---
from main import (
    load_pdf,
    chunk_text,
    get_document_embedding,
    answer_question,
    sparse_model,
    COLLECTION_NAME,
)

app = FastAPI(title="InsightRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qdrant se ek hi baar connect karo (poori API life mein reuse hoga)
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))


# ================================
# Request/Response models
# ================================
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    pages: list[int] = []
    confidence: float
    answered: bool


# ================================
# Collection ko safely ensure karna — agar already hai to kuch mat karo,
# nahi hai to naya banao (delete kabhi nahi karta, isliye multi-PDF safe hai)
# ================================
def ensure_collection_exists(sample_text: str):
    if client.collection_exists(COLLECTION_NAME):
        return

    sample_embedding = get_document_embedding(sample_text)
    vector_size = len(sample_embedding)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )


# ================================
# Naye chunks ko EXISTING collection mein ADD karta hai (delete nahi karta) —
# ye main.py ke store_in_qdrant() se alag hai, isliye 2nd, 3rd PDF upload
# karne par pehle wala data safe rehta hai. UUID ids use kiye taaki
# alag-alag uploads ke chunks kabhi overwrite na hon.
# ================================
def add_chunks_to_qdrant(chunks: list, source_name: str) -> int:
    if not chunks:
        return 0

    ensure_collection_exists(chunks[0]["text"])

    texts = [c["text"] for c in chunks]
    sparse_embeddings = list(sparse_model.embed(texts))

    points = []
    for chunk, sparse_emb in zip(chunks, sparse_embeddings):
        dense_emb = get_document_embedding(chunk["text"])
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_emb,
                    "sparse": SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                },
                payload={"text": chunk["text"], "page": chunk["page"], "source": source_name},
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


# ================================
# Health check
# ================================
@app.get("/health")
def health():
    return {"status": "ok"}


# ================================
# PDF upload — jitni baar chaho call karo, purana data delete nahi hoga,
# naya PDF pehle wale ke saath hi mix ho jayega (combined knowledge base)
# ================================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sirf PDF files allowed hain")

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save karne mein error: {e}")

    try:
        pages = load_pdf(tmp_path)

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="PDF se text extract nahi hua (scanned image PDF ho sakta hai)"
            )

        chunks = chunk_text(pages)
        added_count = add_chunks_to_qdrant(chunks, source_name=file.filename)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing fail hui: {e}")
    finally:
        os.remove(tmp_path)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_added": added_count,
    }


# ================================
# Question-answering — koi limit nahi, jitni baar chaho call karo.
# Ye seedha main.py ke answer_question() ko use karta hai, koi naya
# retrieval logic nahi likha.
# ================================
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question khaali nahi ho sakta")

    if not client.collection_exists(COLLECTION_NAME):
        raise HTTPException(
            status_code=400,
            detail="Abhi tak koi PDF upload nahi hua. Pehle /upload se PDF daalo."
        )

    try:
        result = answer_question(client, question, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer generate karne mein error: {e}")

    return AskResponse(
        answer=result["answer"],
        pages=result["pages"],
        confidence=result["confidence"],
        answered=result["answered"],
    )


# ================================
# Saved eval_results.csv se average scores nikal ke return karta hai —
# koi naya calculation nahi, sirf jo pehle se calculate ho chuka hai wo padh raha hai
# ================================
@app.get("/eval-stats")
def eval_stats():
    csv_path = "eval_results.csv"

    if not os.path.exists(csv_path):
        return {"available": False}

    try:
        df = pd.read_csv(csv_path)
        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

        stats = {}
        for metric in metric_names:
            if metric in df.columns:
                avg = df[metric].mean(skipna=True)
                stats[metric] = round(float(avg), 2) if pd.notna(avg) else None

        return {
            "available": True,
            "num_questions": len(df),
            "scores": stats,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ================================
# (Optional) Sab data clear karke fresh shuru karne ke liye — sirf tab
# use karo jab jaan-boojh kar sab PDFs hata ke naye se start karna ho
# ================================
@app.delete("/reset")
def reset_collection():
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    return {"status": "collection cleared"}


# ================================
# Static frontend (static/index.html yahan se serve hoga)
# ================================
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")