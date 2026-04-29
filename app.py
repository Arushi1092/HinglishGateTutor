import streamlit as st
import os
import uuid
import nltk
import torch
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScalarQuantization, ScalarQuantizationConfig, ScalarType, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from langdetect import detect
from groq import Groq
from dotenv import load_dotenv

# --- INITIALIZATION & CONFIG ---
load_dotenv()
st.set_page_config(page_title="Hindi-GATE Tutor", page_icon="🎓", layout="wide")

COLLECTION = "gate_docs"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- RESOURCE CACHING ---
@st.cache_resource
def get_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    # On HF Spaces, we use 'qdrant_data' for local persistence during the session
    client = QdrantClient(path="qdrant_data")
    return embedder, reranker, client

embedder, reranker, q_client = get_resources()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- LOGIC: INGESTION ---
def chunk_text(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    chunks = []
    for i in range(0, len(sentences), 9): # 12 sentences with overlap
        window = sentences[i : i + 12]
        if len(window) >= 2:
            chunks.append(" ".join(window))
    return chunks

def ingest_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    chunks = chunk_text(text)
    
    if not q_client.collection_exists(COLLECTION):
        q_client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            quantization_config=ScalarQuantization(scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True))
        )
    
    embeddings = embedder.encode(chunks, batch_size=32, convert_to_numpy=True)
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=emb.tolist(), payload={"text": c, "source": uploaded_file.name})
        for c, emb in zip(chunks, embeddings)
    ]
    q_client.upload_points(collection_name=COLLECTION, points=points)
    return len(chunks)

# --- LOGIC: RETRIEVAL ---
def hybrid_search(query, top_k=5):
    if not q_client.collection_exists(COLLECTION): return []
    
    # Dense
    query_vec = embedder.encode(query).tolist()
    dense_res = q_client.query_points(collection_name=COLLECTION, query=query_vec, limit=20).points
    
    # BM25 (Keyword)
    all_points = q_client.scroll(collection_name=COLLECTION, limit=1000, with_payload=True)[0]
    if not all_points: return []
    corpus = [p.payload["text"] for p in all_points]
    bm25 = BM25Okapi([t.lower().split() for t in corpus])
    bm25_scores = bm25.get_scores(query.lower().split())
    
    # Fusion (RRF)
    scores = {}
    for rank, r in enumerate(dense_res):
        txt = r.payload["text"]
        scores[txt] = scores.get(txt, 0) + 1/(60 + rank + 1)
    
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    for rank, idx in enumerate(top_bm25_idx):
        txt = corpus[idx]
        scores[txt] = scores.get(txt, 0) + 1.2/(60 + rank + 1) # Boost keyword match
        
    sorted_txt = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Rerank
    pairs = [(query, t[0]) for t in sorted_txt]
    rr_scores = reranker.predict(pairs)
    final = sorted(zip(rr_scores, sorted_txt), key=lambda x: x[0], reverse=True)
    
    return [{"text": f[1][0], "source": next(p.payload["source"] for p in all_points if p.payload["text"] == f[1][0])} for f in final[:top_k]]

# --- LOGIC: GENERATION ---
def ask_ai(query, context_list, history):
    lang = "hi" if detect(query) == "hi" else "en"
    context = "\n---\n".join([c["text"] for c in context_list])
    
    sys_prompt = f"You are an expert GATE/JEE tutor. Answer based ONLY on context. Use Chain of Thought for math. " \
                 f"If missing, say you don't know. Language: {lang}. End with 'Want a similar practice problem?'"
    
    messages = [{"role": "system", "content": sys_prompt}]
    for h in history[-3:]:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    
    resp = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, temperature=0.1)
    return resp.choices[0].message.content

# --- UI ---
st.title("🎓 Hindi-GATE Tutor")
st.markdown("Your AI partner for GATE/JEE Preparation in Hindi & English.")

with st.sidebar:
    st.header("📚 Study Material")
    uploaded = st.file_uploader("Upload Syllabus PDF", type="pdf")
    if uploaded and st.button("Index Book"):
        with st.spinner("Reading & Indexing..."):
            n = ingest_pdf(uploaded)
            st.success(f"Indexed {n} chunks!")

if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = hybrid_search(prompt)
            history = [[m["content"], ""] for m in st.session_state.messages if m["role"] == "user"] # Simplified
            ans = ask_ai(prompt, context, [])
            st.markdown(ans)
            if context:
                st.caption(f"Sources: {', '.join(set(c['source'] for c in context))}")
            st.session_state.messages.append({"role": "assistant", "content": ans})
