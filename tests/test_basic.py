import pytest
import nltk
from app.retriever import detect_language, prepare_query
from app.ingest import chunk_text

# Setup NLTK
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

def test_detect_language_english():
    assert detect_language("What is the capital of France?") == "en"

def test_detect_language_hindi():
    # "नमस्ते आप कैसे हैं?"
    assert detect_language("नमस्ते आप कैसे हैं?") == "hi"

def test_detect_language_short():
    assert detect_language("Hi") == "en"

def test_prepare_query():
    q = "What is memory?"
    info = prepare_query(q)
    assert info["original"] == q
    assert info["lang"] == "en"
    assert info["respond_in"] == "en"

def test_prepare_query_hindi():
    q = "मेमोरी क्या है?"
    info = prepare_query(q)
    assert info["lang"] == "hi"
    assert info["respond_in"] == "hi"

def test_chunk_text():
    text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6. Sentence 7. Sentence 8. Sentence 9. Sentence 10. Sentence 11. Sentence 12. Sentence 13. Sentence 14."
    chunks = chunk_text(text)
    assert len(chunks) >= 1
    # Check if first chunk contains the expected sentences
    assert "Sentence 1." in chunks[0]
    assert "Sentence 12." in chunks[0]
