from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil, tempfile, os

from app.ingest import ingest_pdf, ingest_text, create_collection
from app.retriever import hybrid_search, build_bm25_index, prepare_query, get_qdrant_client, COLLECTION, already_ingested
from app.generator import generate_answer, rewrite_query
from app.wiki import fetch_wikipedia_content
import json

app = FastAPI(title="Hindi-GATE Tutor", version="1.0")

GOLDEN_DATA_PATH = "data/golden_qa.json"

# ── EVALUATION HELPERS ────────────────────────────────
def judge_response(question, golden, generated, sources):
    from app.generator import client
    prompt = f"""
    Compare the Generated Answer against the Golden Answer for the Question.
    Question: {question}
    Golden Answer: {golden}
    Generated Answer: {generated}
    Sources: {sources}

    Return JSON: {{"accuracy": score_1_to_5, "reasoning": "why"}}
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except:
        return None

# Global BM25 state
bm25_index, all_chunks = None, []


# ── STARTUP ──────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global bm25_index, all_chunks

    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION):
        create_collection()

    # Pre-load models into memory
    from app.retriever import get_embedder, get_reranker
    print("⏳ Loading embedding model...")
    get_embedder()
    print("⏳ Loading reranker model...")
    get_reranker()

    bm25_index, all_chunks = build_bm25_index()

    print(f"✅ BM25 index built: {len(all_chunks)} chunks")
    print("🚀 Server ready")


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


# ── POST /wiki ────────────────────────────────────────
class WikiRequest(BaseModel):
    title: str
    lang: str = "en"


@app.post("/wiki")
async def ingest_wiki(req: WikiRequest):
    source_name = f"wiki:{req.title}"
    
    if already_ingested(source_name):
        return {
            "title": req.title,
            "ingested_chunks": 0,
            "status": "already_ingested"
        }

    text = fetch_wikipedia_content(req.title, req.lang)
    if not text:
        raise HTTPException(404, f"Wikipedia page '{req.title}' not found")

    try:
        n = ingest_text(text, source_name)

        global bm25_index, all_chunks
        bm25_index, all_chunks = build_bm25_index()

        print(f"✅ Ingested {n} chunks from Wikipedia: {req.title}")

        return {
            "title": req.title,
            "ingested_chunks": n,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


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
    # FIX (faithfulness NaN + relevancy lock): generator must be told to answer
    # ONLY from the retrieved chunks. Ensure your generate_answer() system prompt
    # contains: "Answer ONLY based on the context provided. Do not add any
    # information not present in the context. If unsure, say so in Hindi."
    answer = generate_answer(
        req.question,
        chunks,
        q_info["lang"],
        history=req.history
    )

    # 5. Optional: Real-time Evaluation if in Golden Dataset
    eval_result = None
    if os.path.exists(GOLDEN_DATA_PATH):
        try:
            with open(GOLDEN_DATA_PATH, "r", encoding="utf-8") as f:
                golden_data = json.load(f)
                for item in golden_data:
                    if item["question"].lower().strip() == req.question.lower().strip():
                        eval_result = judge_response(
                            req.question, 
                            item["golden_answer"], 
                            answer, 
                            list(set(c["source"] for c in chunks))
                        )
                        break
        except:
            pass

    return {
        "question": req.question,
        "language_detected": q_info["lang"],
        "answer": answer,
        "sources": list(set(c["source"] for c in chunks)),
        "contexts": [c["text"] for c in chunks],  # FIX (NaN faithfulness): expose raw context list for RAGAS evaluation
        "chunks": chunks, # Return full chunks for evaluation
        "evaluation": eval_result
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

@app.get("/golden_qa")
async def get_golden_qa():
    if os.path.exists(GOLDEN_DATA_PATH):
        with open(GOLDEN_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []