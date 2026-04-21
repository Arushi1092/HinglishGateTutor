import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from app.retriever import build_bm25_index
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid, os

COLLECTION = "gate_docs"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 500   # characters (not tokens — simpler)
OVERLAP = 100

embedder = SentenceTransformer(EMBED_MODEL)
client = QdrantClient(path="./qdrant_data")

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += CHUNK_SIZE - OVERLAP
    return chunks

def create_collection():  # deletes the old data each run 
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

def ingest_pdf(pdf_path: str, source_name: str):
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    chunk_dicts = [{"text": c, "source": source_name} for c in chunks] #converts chunks to structured data
    build_bm25_index(chunk_dicts) #Take all chunks → tokenize → build keyword index  bm25=keyword based search
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={"text": chunk, "source": source_name}
        )
        for chunk, emb in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Ingested {len(points)} chunks from {source_name}")
    return len(points)

if __name__ == "__main__":
    create_collection()
    for fname in os.listdir("data/raw"):
        if fname.endswith(".pdf"):
            ingest_pdf(f"data/raw/{fname}", fname)