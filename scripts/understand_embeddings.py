from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

sentences = [
    "What is time complexity of binary search?",
    "Binary search has O(log n) time complexity",
    "बाइनरी सर्च की समय जटिलता क्या है?",   # Hindi version
    "What is photosynthesis?",               # unrelated
]

embeddings = model.encode(sentences)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = embeddings[0]
for i, s in enumerate(sentences):
    print(f"{cosine_sim(query, embeddings[i]):.3f}  |  {s}")