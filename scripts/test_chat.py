import requests
import json

def chat():
    url = "http://localhost:8000/ask"
    history = []
    
    print("--- HindiGateTutor Chat Mode ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break
            
        payload = {
            "question": question,
            "top_k": 5,
            "history": history
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            answer = data["answer"]
            print(f"\nTutor: {answer}\n")
            print(f"(Sources: {', '.join(data['sources'])})\n")
            
            # Update history for the next turn
            history.append({"q": question, "a": answer})
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat()
