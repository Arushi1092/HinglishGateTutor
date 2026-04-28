from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
import os

def download():
    print("Downloading SentenceTransformer model...")
    SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    print("Downloading CrossEncoder model...")
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    print("Downloading NLTK data...")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    
    print("Models and data downloaded successfully.")

if __name__ == "__main__":
    download()

