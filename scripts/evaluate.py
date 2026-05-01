import json
import requests
import os
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq

# Load environment
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

BASE_URL = "http://localhost:8000"
GOLDEN_DATA_PATH = "data/golden_qa.json"

JUDGE_PROMPT = """
You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.
Your goal is to compare a 'Generated Answer' against a 'Golden Answer' and provide a score.

Evaluation Criteria:
1. Accuracy (1-5): How factually correct is the answer compared to the golden answer?
2. Completeness (1-5): Does it cover all the key points mentioned in the golden answer?
3. Faithfulness (1-5): Is the answer grounded in the retrieved sources? (Look for source citations if available).

Input:
Question: {question}
Golden Answer: {golden_answer}
Generated Answer: {generated_answer}
Sources Used: {sources}

Output ONLY a JSON object with this structure:
{{
  "accuracy": 0,
  "completeness": 0,
  "faithfulness": 0,
  "reasoning": "Brief explanation of the scores"
}}
"""

def get_judge_score(question, golden, generated, sources) -> Dict:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question,
                golden_answer=golden,
                generated_answer=generated,
                sources=", ".join(sources)
            )}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error judging: {e}")
        return {"accuracy": 0, "completeness": 0, "faithfulness": 0, "reasoning": "Judge failed"}

def run_evaluation():
    if not os.path.exists(GOLDEN_DATA_PATH):
        print(f"Error: {GOLDEN_DATA_PATH} not found.")
        return

    with open(GOLDEN_DATA_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    results = []
    total_accuracy = 0
    
    print(f"🚀 Starting Evaluation of {len(golden_data)} questions...\n")

    for item in golden_data:
        question = item["question"]
        golden = item["golden_answer"]
        
        print(f"❓ Testing: {question}")
        
        # 1. Get Answer from RAG
        try:
            resp = requests.post(f"{BASE_URL}/ask", json={"question": question, "top_k": 5})
            if resp.status_code != 200:
                print(f"   ❌ Backend error: {resp.text}")
                continue
            
            data = resp.json()
            generated = data["answer"]
            sources = data["sources"]
            
            # 2. Judge the answer
            score = get_judge_score(question, golden, generated, sources)
            
            results.append({
                "question": question,
                "golden": golden,
                "generated": generated,
                "score": score
            })
            
            total_accuracy += score["accuracy"]
            print(f"   ✅ Score: {score['accuracy']}/5 | {score['reasoning'][:100]}...")

        except Exception as e:
            print(f"   ❌ Connection error: {e}")

    # 3. Final Report
    avg_accuracy = (total_accuracy / len(results)) if results else 0
    print(f"\n📊 FINAL EVALUATION REPORT")
    print(f"Total Questions: {len(golden_data)}")
    print(f"Successful Tests: {len(results)}")
    print(f"Average Accuracy: {avg_accuracy:.2f}/5")
    
    # Save results
    with open("data/eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "average_accuracy": avg_accuracy,
            "details": results
        }, f, indent=2)
    print(f"\nResults saved to data/eval_results.json")

if __name__ == "__main__":
    # Check if backend is running first
    try:
        requests.get(f"{BASE_URL}/health")
        run_evaluation()
    except:
        print(f"Error: Backend is not running at {BASE_URL}. Please start it with 'uvicorn app.main:app'")
