from app.retriever import dense_search

queries = [
    "What is a relation in mathematics?",
    "गणित में संबंध क्या होता है?",
    "Explain binary search tree",
]

for q in queries:
    print(f"\nQuery: {q}")
    results = dense_search(q, top_k=3)

    for r in results:
        clean_text = r["text"][:200].replace("\n", " ")
        print(f"  [{r['score']:.3f}] {clean_text}...")