# 📚 Hindi-GATE Tutor: Multilingual RAG Assistant

An AI-powered tutoring system designed for GATE/JEE preparation. This application uses a **Retrieval-Augmented Generation (RAG)** pipeline to provide grounded, context-aware answers from technical textbooks in both English and Hindi.

## 🚀 Technical Features
- **Hybrid Retrieval:** Combines semantic search (Dense) with keyword matching (BM25) for high-precision retrieval.
- **Advanced Reranking:** Utilizes a Cross-Encoder to re-score candidates, ensuring the most relevant context is fed to the LLM.
- **Conversational Memory:** Maintains context through multi-turn dialogues with LLM-driven **Query Rewriting**.
- **Optimized Ingestion:** Batched PDF processing with GPU acceleration and HNSW indexing for scalability.
- **Dynamic Library:** Manage your knowledge base with the ability to ingest or delete specific documents.

## 🧠 Tech Stack
- **Backend:** FastAPI (Asynchronous Python)
- **Frontend:** Streamlit
- **Vector DB:** Qdrant (with INT8 Quantization)
- **LLM:** Llama-3 (via Groq Cloud)
- **Embeddings:** `paraphrase-multilingual-mpnet-base-v2`
- **Search Logic:** Rank-BM25 + RRF (Reciprocal Rank Fusion)

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+
- Groq API Key (Place in `.env` file)

### 2. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Application
You will need two terminals:

**Terminal 1 (Backend):**
```bash
uvicorn app.main:app --reload
```

**Terminal 2 (Frontend UI):**
```bash
streamlit run app_ui.py
```

## 📸 Project Structure
- `app/`: Core logic (Ingestion, Retrieval, Generation)
- `app_ui.py`: Streamlit dashboard
- `data/raw/`: Storage for source PDFs
- `qdrant_data/`: Local vector database (Git ignored)

---
*Developed for technical academic assistance and competitive exam preparation.*
