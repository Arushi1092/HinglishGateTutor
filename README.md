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

# 🎓 Hindi-GATE Tutor

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-deploy-button.svg)](https://huggingface.co/spaces/arushi1092/Hindi-GATE-Tutor)

An AI-powered **multilingual (Hindi + English) tutor** for GATE and JEE preparation using a **Retrieval-Augmented Generation (RAG)** pipeline.

This system allows students to upload study material and get **conceptual, context-aware explanations** in their preferred language.

---

## 🚀 Features

* 📄 Upload PDFs (GATE papers, textbooks, notes)
* 🌐 Hindi + English query support
* 🔍 Hybrid retrieval (Dense embeddings + BM25)
* 🧠 Semantic understanding using sentence-transformers
* 🤖 LLM-powered explanations (Groq - Llama 3.3 70B)
* 📊 Context-aware and step-by-step answers

---

## 🏗️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Llama 3.3 70B (via Groq)
* **Vector DB:** Qdrant (local persistent storage)
* **Embeddings:** Sentence Transformers
* **Search:** Hybrid (Vector + BM25)
* **Parsing:** PyMuPDF

---

## ⚙️ How to Use

1. Open the app
2. Upload your syllabus / notes PDF
3. Wait for indexing
4. Ask questions in Hindi or English
5. Get structured explanations

---

## 🧠 How It Works

1. Documents are split into chunks
2. Each chunk is converted into embeddings
3. Stored in Qdrant vector database
4. Hybrid retrieval fetches relevant chunks
5. LLM generates answers based on retrieved context

---

## ⚠️ Limitations

* Answers depend on quality of uploaded documents
* Large PDFs may increase latency
* Requires API key for Groq

---

## 🔮 Future Improvements

* Better chunking strategy
* Answer evaluation (RAGAS)
* UI enhancements
* Multi-document comparison

---

## 👩‍💻 Author

Built to help students learn concepts more effectively in their preferred language ❤️
