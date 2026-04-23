import fitz  # PyMuPDF
from app.retriever import build_bm25_index, get_qdrant_client, get_embedder
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid, os

COLLECTION = "gate_docs"
CHUNK_SIZE = 500   # characters
OVERLAP = 100

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

def create_collection(force_recreate=False):
    client = get_qdrant_client()
    if client.collection_exists(COLLECTION):
        if force_recreate:
            client.delete_collection(COLLECTION)
        else:
            return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

def ingest_pdf(pdf_path: str, source_name: str):
    client = get_qdrant_client()
    embedder = get_embedder()
    
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    
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
    create_collection(force_recreate=True)
    raw_data_dir = "data/raw"
    if os.path.exists(raw_data_dir):
        for fname in os.listdir(raw_data_dir):
            if fname.endswith(".pdf"):
                ingest_pdf(os.path.join(raw_data_dir, fname), fname)
    else:
        print(f"Directory {raw_data_dir} not found.")
