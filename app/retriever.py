from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
client = QdrantClient(path="./qdrant_data")  

def dense_search(query: str, top_k: int = 10) -> list[dict]:
    query_vec = embedder.encode(query).tolist()
    results = client.query_points(
    collection_name="gate_docs",
    query=query_vec,
    limit=top_k
).points
    return [
        {"text": r.payload["text"], "source": r.payload["source"], "score": r.score}
        for r in results
    ]