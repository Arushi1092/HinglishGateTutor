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

# 🎓 Hindi-GATE Tutor: Advanced Bilingual RAG System

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-deploy-button.svg)](https://huggingface.co/spaces/arushi1092/Hindi-GATE-Tutor)

An AI-powered **multilingual (Hindi + English) tutor** for GATE and JEE preparation using an **Advanced Retrieval-Augmented Generation (RAG)** pipeline.

This system allows students to upload study material and get **conceptual, context-aware explanations** in their preferred language, optimized for high-fidelity retrieval and low-latency responses.

---

## 🚀 Key Enhancements (Latest)

* 🧠 **HyDE (Hypothetical Document Embedding):** Improved retrieval for vague student queries by generating textbook-style hypothetical answers before vector search.
* ⚡ **Async Architecture:** Fully refactored API using `FastAPI` and `AsyncGroq` for high-concurrency handling of multiple student sessions.
* 🛡️ **Rate Limiting:** Integrated `slowapi` to ensure system stability and protect against API abuse.
* 📊 **RAGAS Evaluation:** Automated pipeline to measure **Faithfulness**, **Answer Relevancy**, and **Context Precision**.
* 🧪 **Automated Testing:** Unit test suite for language detection, chunking logic, and retrieval.

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

## ⚙️ Features

* 📄 **Bilingual PDF Ingestion:** Process technical GATE/JEE papers and textbooks.
* 🌐 **Language-Agile:** Automatically detects and responds in English or Hindi.
* 🔍 **Hybrid Retrieval:** Combines semantic vector search with keyword-based BM25 for maximum accuracy.
* 🤖 **Smart Query Rewriting:** Uses conversation history to refine follow-up questions into standalone search queries.

---

## 📊 Evaluation & Metrics

The system is continuously evaluated using the **RAGAS framework** to ensure high educational standards:
- **Faithfulness:** Ensures answers are strictly grounded in provided context.
- **Answer Relevancy:** Measures how well the response addresses the student's query.
- **Context Precision:** Optimizes the retrieval of the most relevant study material.

---

## ⚙️ Installation & Setup

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/your-username/HindiGateTutor.git
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
   python -m uvicorn app.main:app
   streamlit run app_ui.py
   ```

---

## 👩‍💻 Author

Built with ❤️ to empower students by bridging the language gap in technical education.
