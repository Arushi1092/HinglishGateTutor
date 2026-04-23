from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil, tempfile, os

from app.ingest import ingest_pdf, create_collection
from app.retriever import hybrid_search, build_bm25_index, prepare_query, get_qdrant_client, COLLECTION
from app.generator import generate_answer, rewrite_query

app = FastAPI(title="Hindi-GATE Tutor", version="1.0")

# Global BM25 state
bm25_index, all_chunks = None, []


# ── HELPER ────────────────────────────────────────────
def already_ingested(filename: str) -> bool:
    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION):
        return False

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

    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION):
        create_collection()

    bm25_index, all_chunks = build_bm25_index()

    print(f"✅ BM25 index built: {len(all_chunks)} chunks")


# ── POST /ingest ──────────────────────────────────────
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    if already_ingested(file.filename):
        return {
            "filename": file.filename,
            "ingested_chunks": 0,
            "status": "already_ingested"
        }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        n = ingest_pdf(tmp_path, file.filename)

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
    history: list = []


@app.post("/ask")
async def ask(req: AskRequest):
    
    # 1. Rewrite query using history (better approach)
    search_query = rewrite_query(req.question, req.history)

    if search_query != req.question:
        print(f"🔄 Rewritten (LLM): {req.question} -> {search_query}")

    # 2. Language detection
    q_info = prepare_query(req.question)

    # 3. Retrieval using rewritten query
    chunks = hybrid_search(search_query, top_k=req.top_k)

    if not chunks:
        raise HTTPException(
            404,
            "No relevant documents found. Please ingest PDFs first."
        )

    print(f"🔍 Retrieved {len(chunks)} chunks")

    # 4. Generate answer with history
    answer = generate_answer(
        req.question,
        chunks,
        q_info["lang"],
        history=req.history
    )

    return {
        "question": req.question,
        "language_detected": q_info["lang"],
        "answer": answer,
        "sources": list(set(c["source"] for c in chunks))
    }


# ── MANAGE SOURCES ─────────────────────────────────────
@app.get("/sources")
async def list_sources():
    # Get unique sources from the current all_chunks list
    sources = list(set(c["source"] for c in all_chunks))
    return {"sources": sorted(sources)}

@app.delete("/source/{filename}")
async def delete_source(filename: str):
    global bm25_index, all_chunks
    client = get_qdrant_client()
    
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # 1. Delete from Qdrant
        client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=filename))
                ]
            )
        )
        
        # 2. Rebuild local index to sync
        bm25_index, all_chunks = build_bm25_index()
        
        print(f"🗑️ Deleted source: {filename}")
        return {"status": "success", "message": f"Deleted {filename}"}
    except Exception as e:
        raise HTTPException(500, f"Deletion failed: {str(e)}")


# ── RESET DATABASE ────────────────────────────────────
@app.post("/reset")
async def reset():
    global bm25_index, all_chunks
    try:
        # Recreate collection (this deletes all data)
        create_collection(force_recreate=True)
        
        # Reset global state
        bm25_index = None
        all_chunks = []
        
        print("🚨 Database Reset Triggered")
        return {"status": "success", "message": "All data cleared"}
    except Exception as e:
        raise HTTPException(500, f"Reset failed: {str(e)}")


# ── GET /health ────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_indexed": len(all_chunks)
    }