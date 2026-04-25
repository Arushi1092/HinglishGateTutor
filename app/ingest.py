import fitz  # PyMuPDF
from app.retriever import build_bm25_index, get_qdrant_client, get_embedder
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff, ScalarQuantization, ScalarType
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
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        quantization_config=ScalarQuantization(
            scalar=ScalarType.INT8,
            quantile=0.99,
            always_ram=True
        )
    )

def ingest_text(text: str, source_name: str):
    import torch
    client = get_qdrant_client()
    embedder = get_embedder()
    
    # 🚀 Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(embedder.device) != device:
        embedder.to(device)
    
    chunks = chunk_text(text)
    total_chunks = len(chunks)
    batch_size = 128
    
    print(f"📦 Processing {total_chunks} chunks for {source_name} using {device} (Batch Size: {batch_size})")

    for i in range(0, total_chunks, batch_size):
        batch_texts = chunks[i : i + batch_size]
        
        batch_embeddings = embedder.encode(
            batch_texts, 
            batch_size=batch_size, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={"text": chunk, "source": source_name}
            )
            for chunk, emb in zip(batch_texts, batch_embeddings)
        ]
        
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"✅ Indexed {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")

    print(f"🎉 Successfully ingested {source_name}")
    return total_chunks

def ingest_pdf(pdf_path: str, source_name: str):
    print(f"📖 Extracting text from PDF: {source_name}")
    text = extract_text(pdf_path)
    return ingest_text(text, source_name)

if __name__ == "__main__":
    create_collection(force_recreate=True)
    raw_data_dir = "data/raw"
    if os.path.exists(raw_data_dir):
        for fname in os.listdir(raw_data_dir):
            if fname.endswith(".pdf"):
                ingest_pdf(os.path.join(raw_data_dir, fname), fname)
    else:
        print(f"Directory {raw_data_dir} not found.")
