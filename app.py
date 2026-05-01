import streamlit as st
import os
import uuid
import nltk
import torch
import fitz  # PyMuPDF
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScalarQuantization, ScalarQuantizationConfig, ScalarType, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from langdetect import detect
from groq import AsyncGroq
from dotenv import load_dotenv

# --- INITIALIZATION & CONFIG ---
load_dotenv()
st.set_page_config(page_title="Hindi-GATE Tutor", page_icon="🎓", layout="wide")

COLLECTION = "gate_docs"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

HYDE_PROMPT = """Write a 3-sentence technical textbook answer to the following question.
Focus on being factual and providing core details that would likely appear in a textbook.

Question: {query}
Hypothetical Answer:"""

# --- RESOURCE CACHING ---
@st.cache_resource
def get_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    # On HF Spaces, we use 'qdrant_data' for local persistence
    client = QdrantClient(path="qdrant_data")
    return embedder, reranker, client

embedder, reranker, q_client = get_resources()
# Using AsyncGroq for better concurrency
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# --- LOGIC: HYDE ---
async def generate_hypothetical_answer(query: str) -> str:
    try:
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
            max_tokens=150,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return query

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
def hybrid_search(query, top_k=5, hypothetical_answer=None):
    if not q_client.collection_exists(COLLECTION): return []
    
    # Dense Search (Use HyDE if available)
    search_text = hypothetical_answer if hypothetical_answer else query
    query_vec = embedder.encode(search_text).tolist()
    dense_res = q_client.query_points(collection_name=COLLECTION, query=query_vec, limit=20).points
    
    # BM25 (Keyword) - always use original query
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
    
    # Rerank (Cross-Encoder)
    pairs = [(query, t[0]) for t in sorted_txt]
    rr_scores = reranker.predict(pairs)
    final = sorted(zip(rr_scores, sorted_txt), key=lambda x: x[0], reverse=True)
    
    return [{"text": f[1][0], "source": next(p.payload["source"] for p in all_points if p.payload["text"] == f[1][0])} for f in final[:top_k]]

# --- LOGIC: GENERATION ---
async def ask_ai(query, context_list, history):
    # Language Detection
    try:
        lang = detect(query)
    except:
        lang = "en"
    
    context = "\n---\n".join([f"[Source: {c['source']}] {c['text']}" for c in context_list])
    
    if lang == "hi":
        sys_prompt = "आप एक मददगार और सख्त GATE/JEE शिक्षक हैं। केवल दिए गए संदर्भ के आधार पर उत्तर दें। उत्तर के अंत में यह लिखें: 'क्या आप एक समान अभ्यास प्रश्न चाहते हैं?'"
    else:
        sys_prompt = "You are an expert GATE/JEE tutor. Answer based ONLY on context. Use step-by-step reasoning for math. " \
                     "If info is missing, say you don't know. End with 'Want a similar practice problem?'"
    
    messages = [{"role": "system", "content": sys_prompt}]
    for h in history[-3:]:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": f"### Context:\n{context}\n\n### Question: {query}"})
    
    resp = await groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, temperature=0.1)
    return resp.choices[0].message.content

# --- UI ---
st.title("🎓 Hindi-GATE Tutor")
st.caption("Advanced RAG System with HyDE & Hybrid Search")

with st.sidebar:
    st.header("📚 Study Material")
    uploaded = st.file_uploader("Upload Syllabus PDF", type="pdf")
    if uploaded and st.button("Index Book"):
        with st.spinner("Reading & Indexing..."):
            n = ingest_pdf(uploaded)
            st.success(f"Indexed {n} chunks!")
    
    st.divider()
    st.toggle("Use HyDE (Better Search)", value=True, key="use_hyde")

if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Handling async in Streamlit
async def main_chat():
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching & Thinking..."):
                # 1. HyDE
                hypo = None
                if st.session_state.use_hyde:
                    hypo = await generate_hypothetical_answer(prompt)
                
                # 2. Retrieval
                context = hybrid_search(prompt, hypothetical_answer=hypo)
                
                # 3. History
                history = []
                for i in range(0, len(st.session_state.messages)-1, 2):
                    if i+1 < len(st.session_state.messages):
                        history.append([st.session_state.messages[i]["content"], st.session_state.messages[i+1]["content"]])
                
                # 4. Generate
                ans = await ask_ai(prompt, context, history)
                
                st.markdown(ans)
                if context:
                    with st.expander("View Sources"):
                        for c in context:
                            st.write(f"**{c['source']}**: {c['text'][:200]}...")
                
                st.session_state.messages.append({"role": "assistant", "content": ans})

if __name__ == "__main__":
    asyncio.run(main_chat())
