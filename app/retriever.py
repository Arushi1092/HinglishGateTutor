from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from rank_bm25 import BM25Okapi
import os

# ----------------------------
# Config
# ----------------------------
COLLECTION = "gate_docs"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

# ----------------------------
# Language Detection
# ----------------------------
def detect_language(text: str) -> str:
    if len(text.strip()) < 5:
        return "en"
    try:
        lang = detect(text)
        return "hi" if lang == "hi" else "en"
    except LangDetectException:
        return "en"

# ----------------------------
# Query Preparation
# ----------------------------
def prepare_query(user_query: str) -> dict:
    lang = detect_language(user_query)
    return {
        "original": user_query,
        "lang": lang,
        "search_queries": [user_query],
        "respond_in": lang
    }

# ----------------------------
# Shared Resources (Singletons)
# ----------------------------
_client = None
_embedder = None

def get_qdrant_client():
    global _client

    if _client is None:
        # Use local persistent storage on disk. No Docker required!
        try:
            _client = QdrantClient(path="qdrant_data")
            print("📦 Qdrant connected (Local Mode: qdrant_data/)")
        except Exception as e:
            print(f"❌ CRITICAL: Could not initialize local Qdrant: {e}")
            raise e

    return _client

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

# ----------------------------
# Initialize (Lazy loading)
# ----------------------------
# We will use get_embedder() and get_qdrant_client() inside functions

# ----------------------------
# Helpers
# ----------------------------
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

def get_all_chunks():
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION):
        return []
    
    # Scroll through all points with larger limit for speed
    chunks = []
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=COLLECTION,
            limit=1000,
            with_payload=True,
            offset=next_page
        )
        for p in points:
            chunks.append({
                "text": p.payload.get("text", ""),
                "source": p.payload.get("source", "unknown")
            })
        if next_page is None:
            break
    return chunks

# ----------------------------
# Dense Search
# ----------------------------
def dense_search(query: str, top_k: int = 5):
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION):
        return []

    embedder = get_embedder()
    query_vec = embedder.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        limit=top_k
    ).points

    return [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "score": r.score
        }
        for r in results
    ]

# ----------------------------
# BM25 Index Builder
# ----------------------------
def build_bm25_index(chunks=None):
    global _all_chunks, bm25_index

    if chunks is None:
        chunks = get_all_chunks()
    
    _all_chunks = chunks
    if not _all_chunks:
        bm25_index = None
        return None, []

    tokenized = [c["text"].lower().split() for c in _all_chunks]
    bm25_index = BM25Okapi(tokenized)
    return bm25_index, _all_chunks

# ----------------------------
# BM25 Search
# ----------------------------
def bm25_search(query: str, top_k: int = 10):
    global bm25_index

    if bm25_index is None:
        build_bm25_index()
        if bm25_index is None:
            return []

    scores = bm25_index.get_scores(query.lower().split())

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [
        {
            "text": _all_chunks[i]["text"],
            "source": _all_chunks[i]["source"],
            "score": float(scores[i])
        }
        for i in top_indices
    ]

# ----------------------------
# RRF Fusion
# ----------------------------
def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    scores = {}

    for rank, r in enumerate(dense_results):
        key = r["text"]
        scores.setdefault(key, {"text": r["text"], "source": r["source"], "rrf": 0})
        scores[key]["rrf"] += 1 / (k + rank + 1)

    for rank, r in enumerate(bm25_results):
        key = r["text"]
        scores.setdefault(key, {"text": r["text"], "source": r["source"], "rrf": 0})
        scores[key]["rrf"] += 1 / (k + rank + 1)

    return sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)


# We can also add a simple reranking step using a cross-encoder for the top candidates
def rerank(query: str, candidates: list[dict], top_k: int = 5):
    if not candidates:
        return []

    reranker = get_reranker()

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        {
            **c,
            "rerank_score": float(score)
        }
        for score, c in ranked[:top_k]
    ]

# ----------------------------
# Hybrid Search
# ----------------------------
def hybrid_search(query: str, top_k: int = 5):
    q_info = prepare_query(query)

    dense = dense_search(q_info["original"], top_k=20)
    sparse = bm25_search(q_info["original"], top_k=20)

    fused = reciprocal_rank_fusion(dense, sparse)

    # take top 20 candidates for reranking
    candidates = fused[:20]

    # final smart selection
    return rerank(q_info["original"], candidates, top_k=top_k)
