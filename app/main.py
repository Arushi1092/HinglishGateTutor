from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil, tempfile, os

from app.ingest import ingest_pdf, create_collection
from app.retriever import hybrid_search, build_bm25_index, prepare_query, get_qdrant_client, COLLECTION
from app.generator import generate_answer

app = FastAPI(title="Hindi-GATE Tutor", version="1.0")

# Global BM25 state
bm25_index, all_chunks = None, []

def already_ingested(filename: str) -> bool:
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION):
        return False
    
    # Check if any point has this source in payload
    # Note: For large collections, this should be an optimized search
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    result = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="source", match=MatchValue(value=filename))
            ]
        ),
        limit=1,
        with_payload=False
    )
    return len(result[0]) > 0

# ── STARTUP ──────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global bm25_index, all_chunks

    # ✅ ensure collection exists (without deleting data)
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION):
        create_collection()

    # ✅ build BM25 from stored chunks
    bm25_index, all_chunks = build_bm25_index()

    print(f"✅ BM25 index built: {len(all_chunks)} chunks")


# ── POST /ingest ──────────────────────────────────────
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):

    # ✅ safer file check
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    # 🔒 prevent duplicate ingestion
    if already_ingested(file.filename):
        return {
            "filename": file.filename,
            "ingested_chunks": 0,
            "status": "already_ingested"
        }

    # save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # ingest into Qdrant
        n = ingest_pdf(tmp_path, file.filename)

        # 🔄 rebuild BM25 index after new data
        global bm25_index, all_chunks
        bm25_index, all_chunks = build_bm25_index()

        print(f"✅ Ingested {n} chunks from {file.filename}")

        return {
            "filename": file.filename,
            "ingested_chunks": n,
            "status": "success"
        }

    finally:
        os.unlink(tmp_path)


# ── POST /ask ──────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/ask")
async def ask(req: AskRequest):
    # Use retriever's prepare_query for language detection
    q_info = prepare_query(req.question)

    chunks = hybrid_search(
        q_info["original"],
        top_k=req.top_k
    )

    # 🚫 avoid hallucination if nothing found
    if not chunks:
        raise HTTPException(
            404,
            "No relevant documents found. Please ingest PDFs first."
        )

    print(f"🔍 Retrieved {len(chunks)} chunks")

    answer = generate_answer(req.question, chunks, q_info["lang"])

    return {
        "question": req.question,
        "language_detected": q_info["lang"],
        "answer": answer,
        "sources": list(set(c["source"] for c in chunks))
    }


# ── GET /health ────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_indexed": len(all_chunks)
    }
