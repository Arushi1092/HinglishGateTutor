import requests
import sys

def ask(question, top_k=5):
    url = "http://localhost:8000/ask"
    payload = {
        "question": question,
        "top_k": top_k
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question\"")
        sys.exit(1)
    
    question = sys.argv[1]
    result = ask(question)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nQuestion: {result['question']}")
        print(f"Language: {result['language_detected']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {', '.join(result['sources'])}")
