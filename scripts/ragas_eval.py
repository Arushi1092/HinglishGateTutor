import json
import requests
import os
from datasets import Dataset
from ragas import evaluate
from dotenv import load_dotenv
from ragas.llms import llm_factory
from openai import OpenAI

# Metrics
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.metrics.collections.context_precision import ContextPrecision

# Embeddings
from ragas.embeddings import embedding_factory

# ----------------------------
# Load environment
# ----------------------------
load_dotenv()

BASE_URL = "http://localhost:8000"

# ----------------------------
# API Key Check
# ----------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Check your .env file.")

# ----------------------------
# LLM Setup
# ----------------------------
client = OpenAI(api_key=api_key)

llm = llm_factory(
    "gpt-4o-mini",   # fast + cheap
    client=client
)

# ----------------------------
# Embeddings Setup (MODERN FIX)
# ----------------------------
embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=client,          # ✅ same client
    interface="modern"      # ✅ CRITICAL FIX
)

# ----------------------------
# Main Evaluation Function
# ----------------------------
def run_eval(qa_file: str = "data/golden_qa.json", sample_step: int = 10):

    if not os.path.exists(qa_file):
        print(f"❌ File not found: {qa_file}")
        return

    with open(qa_file, encoding="utf-8") as f:
        pairs = json.load(f)

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    selected_pairs = pairs[::sample_step]
    print(f"🚀 Starting evaluation on {len(selected_pairs)} samples...\n")

    # ----------------------------
    # Fetch responses from your API
    # ----------------------------
    for idx, pair in enumerate(selected_pairs, 1):
        try:
            print(f"➡️ [{idx}/{len(selected_pairs)}] {pair['question'][:100]}")

            resp = requests.post(
                f"{BASE_URL}/ask",
                json={
                    "question": pair["question"],
                    "history": [],
                    "top_k": 5,
                },
                timeout=120,  # ✅ avoid timeout issues
            )

            if resp.status_code != 200:
                print(f"❌ Failed: {pair['question']} → {resp.text}")
                continue

            resp_json = resp.json()

            answer = resp_json.get("answer", "")

            # ----------------------------
            # CRITICAL: ensure we get chunk TEXT
            # ----------------------------
            # FIX (NaN faithfulness): prefer the new "contexts" key (List[str])
            # added to /ask response. Fall back to extracting from "chunks" dicts.
            # NaN happens when contexts is [] or contains empty/whitespace strings —
            # RAGAS cannot decompose statements with no context to check against.
            raw_contexts = resp_json.get("contexts")  # clean List[str] from updated main.py

            if chunks and isinstance(chunks[0], dict):
                    # handle both "text" and "chunk" key names defensively
                    contexts = [c.get("text") or c.get("chunk") or "" for c in chunks]
            else:
                print("⚠️ No valid chunk text returned → skipping")
                continue

            ground_truth = pair.get("ground_truth") or pair.get("golden_answer", "")

            if not answer or not ground_truth:
                print("⚠️ Missing answer or ground truth")
                continue

            data["question"].append(pair["question"])
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(ground_truth)

        except Exception as e:
            print(f"⚠️ Error: {pair['question']} → {str(e)}")

    if not data["question"]:
        print("❌ No valid data collected for evaluation.")
        return

    # ----------------------------
    # Debug sample
    # ----------------------------
    print("\n🔍 Sample check:")
    print("Q:", data["question"][0])
    print("A:", data["answer"][0][:120])
    # FIX: show all context entries + lengths so empty/short contexts are visible
    print(f"C: {len(data['contexts'][0])} context chunks")
    for i, ctx in enumerate(data["contexts"][0]):
        print(f"   [{i}] ({len(ctx)} chars) {ctx[:80]}...")
    print("GT:", data["ground_truth"][0])

    dataset = Dataset.from_dict(data)

    print("\n🚀 Running RAGAS evaluation...\n")

    # ----------------------------
    # Run Evaluation
    # ----------------------------
    try:
        result = evaluate(
            dataset,
            metrics=[
                Faithfulness(llm=llm),
                AnswerRelevancy(llm=llm, embeddings=embeddings),
                ContextPrecision(llm=llm),
            ]
        )

        print("📊 RAGAS Results:")
        print(result)

        # Save results
        df = result.to_pandas()
        os.makedirs("eval", exist_ok=True)
        df.to_csv("eval/results.csv", index=False)

        print("\n✅ Results saved to eval/results.csv")

    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    run_eval()