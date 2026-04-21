from app.retriever import hybrid_search, prepare_query
from app.generator import generate_answer

queries = [
    "What is a relation in mathematics?",
    "गणित में संबंध क्या होता है?",
    "Explain binary search tree"
]

for query in queries:
    print("\n" + "="*50)
    print(f"Query: {query}")

    q_info = prepare_query(query)
    chunks = hybrid_search(query)

    answer = generate_answer(query, chunks, q_info["respond_in"])

    print("\nAnswer:\n")
    print(answer)