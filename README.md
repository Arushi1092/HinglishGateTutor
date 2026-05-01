---
title: Hindi-GATE-Tutor
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.2
python_version: 3.11
app_file: app.py
pinned: false
---

# 🎓 Multilingual Smart AI Tutor: Advanced RAG System

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-deploy-button.svg)](https://huggingface.co/spaces/arushi1092/HindiEnglish_GATEJEE_Tutor)

An AI-powered **Multilingual Tutor** (supporting **Hindi + English**) for GATE and JEE preparation using an **Advanced Retrieval-Augmented Generation (RAG)** pipeline.

This system allows students to upload study material and get **conceptual, context-aware explanations** in their preferred language, optimized for high-fidelity retrieval and low-latency responses.

---

## 🚀 Key Enhancements (Production-Ready)

* 🧠 **HyDE (Hypothetical Document Embedding):** Improved retrieval for vague student queries by generating textbook-style hypothetical answers before vector search.
* ⚡ **Async Architecture:** Fully refactored API using `FastAPI` and `AsyncGroq` for high-concurrency handling of multiple student sessions.
* 🛡️ **Rate Limiting:** Integrated `slowapi` to ensure system stability and protect against API abuse.
* 📊 **RAGAS Evaluation:** Automated pipeline to measure **Faithfulness**, **Answer Relevancy**, and **Context Precision**.
* 🧪 **Automated Testing:** Unit test suite for language detection, chunking logic, and retrieval using `pytest`.

---

## 🏗️ Tech Stack

* **Frontend:** Streamlit
* **API:** FastAPI (Asynchronous)
* **LLM:** Llama 3.3 70B (via Groq)
* **Vector DB:** Qdrant (local persistent storage)
* **Search:** Advanced Hybrid (Dense Vector + BM25 + HyDE Expansion)
* **Evaluation:** RAGAS (Retrieval-Augmented Generation Assessment)
* **Parsing:** PyMuPDF & NLTK

---

## 📊 Evaluation Results (RAGAS)

| Metric | Target Score | Current Score |
| :--- | :--- | :--- |
| **Faithfulness** | ≥ 0.80 | *(Run `ragas_eval.py` to update)* |
| **Answer Relevancy** | ≥ 0.85 | 0.95 |
| **Context Precision** | ≥ 0.70 | 0.35 |

---

## 🧠 Advanced Pipeline Flow

1. **Query Rewriting:** LLM converts conversational follow-up questions into standalone technical queries.
2. **HyDE Expansion:** System generates a "hypothetical answer" to bridge the semantic gap between questions and textbook content.
3. **Hybrid Search:** Parallel retrieval using **Qdrant Vector Search** (semantic) and **BM25** (keyword).
4. **Reciprocal Rank Fusion (RRF):** Fuses search results for optimal ranking.
5. **Cross-Encoder Reranking:** Top 20 candidates are re-scored for maximum precision.
6. **Bilingual Generation:** Llama 3.3 generates a step-by-step technical explanation in the student's detected language.

---

## ⚙️ Installation & Setup

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/Arushi1092/HindiGateTutor.git
   ```
2. **Setup Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Environment Variables:**
   Create a `.env` file with your `GROQ_API_KEY`.
4. **Run the App:**
   ```bash
   # Start Backend
   python -m uvicorn app.main:app
   
   # Start UI
   streamlit run app_ui.py
   ```

---

## 👩‍💻 Author

Built with ❤️ by **Arushi** to empower students by bridging the language gap in technical education.
