"""
Evaluation: Dense vs Hybrid Retrieval (BM25 + RRF)

Tested on ~10 queries including:
- Conceptual questions (e.g., "What is a relation?")
- Hindi queries (e.g., "गणित में संबंध क्या होता है?")
- Out-of-domain queries (e.g., "Explain binary search tree")

Findings:

1. Dense retrieval (SentenceTransformer embeddings)
   - Performs strongly on semantic queries
   - Correct answers consistently appear in top 2–3 results
   - Works well for both English and Hindi queries due to multilingual model

2. Hybrid retrieval (Dense + BM25 + RRF)
   - Slight improvement in ranking stability
   - Helps when exact keyword matches are important
   - Did not significantly outperform dense retrieval on this dataset
     (likely due to small and well-structured NCERT corpus)

3. Multilingual behavior
   - Hindi queries successfully retrieve relevant English chunks
   - Confirms effectiveness of multilingual embeddings

4. Limitations observed
   - Out-of-domain queries return irrelevant results (expected behavior)
   - Chunk quality impacts ranking more than fusion method
   - Definitions are not always ranked first due to chunk structure

Conclusion:
- Dense retrieval is already strong for this use case
- Hybrid + RRF adds robustness but limited gains on small datasets
- Final answer quality should be handled by the generator (LLM layer),
  not over-optimized in retrieval

Next step:
→ Use top-k retrieved chunks as context for LLM-based answer generation
"""
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from rank_bm25 import BM25Okapi

# ----------------------------
# Config
# ----------------------------
COLLECTION = "gate_docs"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# ----------------------------
# Initialize
# ----------------------------
embedder = SentenceTransformer(EMBED_MODEL)

# ----------------------------
# BM25 Globals
# ----------------------------
_all_chunks = []
bm25_index = None

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
# Dense Search
# ----------------------------
def dense_search(query: str, top_k: int = 5):
    client = QdrantClient(path="./qdrant_data")

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
def build_bm25_index(chunks: list[dict]):
    global _all_chunks, bm25_index

    _all_chunks = chunks
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25_index = BM25Okapi(tokenized)

# ----------------------------
# BM25 Search
# ----------------------------
def bm25_search(query: str, top_k: int = 10):
    global bm25_index

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

# ----------------------------
# Hybrid Search
# ----------------------------
def hybrid_search(query: str, top_k: int = 5):
    q_info = prepare_query(query)

    dense = dense_search(q_info["original"], top_k=20)
    sparse = bm25_search(q_info["original"], top_k=20)

    fused = reciprocal_rank_fusion(dense, sparse)

    return fused[:top_k]