import fitz  # PyMuPDF
import nltk
from app.retriever import build_bm25_index, get_qdrant_client, get_embedder, already_ingested
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,   # ✅ NEW
    ScalarType
)
import uuid, os

# ----------------------------
# COLLECTION name
# ----------------------------
COLLECTION = "gate_docs"

# ----------------------------
# CHUNK CONFIG
# ----------------------------
CHUNK_SENTENCES = 12
OVERLAP_SENTENCES = 3


# ----------------------------
# Extract text from PDF
# ----------------------------
def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


# ----------------------------
# Convert text → chunks
# ----------------------------
def chunk_text(text: str) -> list[str]:
    sentences = nltk.sent_tokenize(text)

    # Keep all sentences (don't delete math formulas!)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]

    chunks = []
    step = max(1, CHUNK_SENTENCES - OVERLAP_SENTENCES)

    for i in range(0, len(sentences), step):
        window = sentences[i : i + CHUNK_SENTENCES]

        # only keep meaningful chunks
        if len(window) >= 2:
            chunks.append(" ".join(window))

    return chunks


# ----------------------------
# Create Qdrant Collection
# ----------------------------
def create_collection(force_recreate=False):
    client = get_qdrant_client()

    if client.collection_exists(COLLECTION):
        if force_recreate:
            client.delete_collection(COLLECTION)
        else:
            return

    client.create_collection(
        collection_name=COLLECTION,
    
        # Vector settings
        vectors_config=VectorParams(
            size=384,                  # embedding size for MiniLM-L12
            distance=Distance.COSINE   # similarity metric
        ),

        # Index (ANN search)
       # hnsw_config=HnswConfigDiff(
        #    m=16,
         #   ef_construct=100
        #),

        # ✅ FIXED quantization (NEW API)
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True
            )
        )
    )


# ----------------------------
# Ingest raw text
# ----------------------------
def ingest_text(text: str, source_name: str):
    import torch
    from qdrant_client.models import PointStruct

    client = get_qdrant_client()
    embedder = get_embedder()

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move model if needed (only once)
    if str(embedder.device) != device:
        embedder.to(device)

    # Split into chunks
    chunks = chunk_text(text)
    total_chunks = len(chunks)
    
    if total_chunks == 0:
        print(f"⚠️ No chunks extracted for {source_name}")
        return 0

    print(f"📦 Processing {total_chunks} chunks for {source_name} using {device}")

    # Convert text → embeddings in optimized batches
    # SentenceTransformers.encode handles batching internally and is quite efficient
    batch_size = 64 if device == "cpu" else 128
    
    print(f"🧠 Generating embeddings...")
    # Standard stable encoding (multi-process is unstable in Docker/Windows)
    embeddings = embedder.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Prepare points for Qdrant
    print(f"📤 Uploading to Qdrant...")
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={
                "text": chunk,
                "source": source_name
            }
        )
        for chunk, emb in zip(chunks, embeddings)
    ]

    # Use upload_points for high-performance parallel upload
    client.upload_points(
        collection_name=COLLECTION,
        points=points,
        batch_size=batch_size,
        parallel=max(1, os.cpu_count() // 2),  # Use half of available cores for upload
        wait=False
    )

    print(f"✅ Finished indexing {source_name}")
    return total_chunks


# ----------------------------
# Ingest PDF
# ----------------------------
def ingest_pdf(pdf_path: str, source_name: str):
    if already_ingested(source_name):
        print(f"⚠️ {source_name} already ingested, skipping")
        return 0
    print(f"📖 Extracting text from {source_name}")
    text = extract_text(pdf_path)
    return ingest_text(text, source_name)


# ----------------------------
# Run manually
# ----------------------------
if __name__ == "__main__":
    create_collection(force_recreate=True)

    raw_data_dir = "data/raw"

    if os.path.exists(raw_data_dir):
        for fname in os.listdir(raw_data_dir):
            if fname.endswith(".pdf"):
                ingest_pdf(os.path.join(raw_data_dir, fname), fname)
    else:
        print(f"Directory {raw_data_dir} not found.")