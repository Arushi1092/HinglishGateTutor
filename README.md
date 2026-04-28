# 📚 Hindi-GATE Tutor: Multilingual RAG Assistant

An AI-powered tutoring system designed for GATE/JEE preparation. This application uses a **Retrieval-Augmented Generation (RAG)** pipeline to provide grounded, context-aware answers from technical textbooks in both English and Hindi.

## 🚀 Technical Features
- **Hybrid Retrieval:** Combines semantic search (Dense) with keyword matching (BM25) for high-precision retrieval.
- **Advanced Reranking:** Utilizes a Cross-Encoder to re-score candidates, ensuring the most relevant context is fed to the LLM.
- **Conversational Memory:** Maintains context through multi-turn dialogues with LLM-driven **Query Rewriting**.
- **Optimized Ingestion:** High-speed processing using **multi-process CPU encoding** and **parallel Qdrant uploads**.
- **Startup Pre-loading:** Models are pre-cached and loaded at server startup for zero-latency first queries.
- **Dynamic Library:** Manage your knowledge base with the ability to ingest or delete specific documents.

## 🧠 Tech Stack
- **Backend:** FastAPI (Asynchronous Python)
- **Frontend:** Streamlit
- **Vector DB:** Qdrant (with INT8 Quantization)
- **LLM:** Llama-3 (via Groq Cloud)
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (Fast, Multilingual)
- **Search Logic:** Rank-BM25 + RRF (Reciprocal Rank Fusion)

## 🛠️ Installation & Setup

### Option 1: Docker (Recommended)
This is the easiest way to run the full stack including the Qdrant vector database.

1. Ensure you have Docker and Docker Compose installed.
2. Create a `.env` file with your API keys:
   ```env
   GROQ_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ```
3. Run with Docker Compose:
   ```bash
   docker-compose up --build
   ```
4. Access the UI at `http://localhost:8501` and the API at `http://localhost:8000`.

### Option 2: Local Setup
1. Prerequisites: Python 3.10+, Groq API Key.
2. Setup Environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the Application (Two terminals):
   - **Backend:** `uvicorn app.main:app --reload`
   - **Frontend:** `streamlit run app_ui.py`

## 📸 Project Structure
- `app/`: Core logic (Ingestion, Retrieval, Generation)
- `app_ui.py`: Streamlit dashboard
- `data/raw/`: Storage for source PDFs
- `qdrant_data/`: Local vector database (Git ignored)

---
*Developed for technical academic assistance and competitive exam preparation.*
