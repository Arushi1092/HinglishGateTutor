import json
import requests
import os
from datasets import Dataset
from ragas import evaluate
from dotenv import load_dotenv

# RAGAS metrics
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

# RAGAS Wrappers for Langchain
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Langchain providers
try:
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("❌ Missing dependencies! Run: pip install langchain-groq langchain-huggingface")
    exit(1)

# ----------------------------
# Load environment
# ----------------------------
load_dotenv()

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ----------------------------
# LLM & Embeddings Setup (FREE/GROQ FIX)
# ----------------------------
# Using Groq for evaluation to avoid OpenAI quota issues
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env. Evaluation requires an LLM (OpenAI or Groq).")

print("⏳ Initializing Groq (Llama-3-8b) for evaluation...")
langchain_llm = ChatGroq(model="llama-3-8b-8192", groq_api_key=groq_api_key)
llm = LangchainLLMWrapper(langchain_llm)

# Use a local multilingual model for RAGAS embeddings (FREE & Fast)
print("⏳ Loading local embedding model for evaluation (paraphrase-multilingual-MiniLM-L12-v2)...")
langchain_embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

# ----------------------------
# Main Evaluation Function
# ----------------------------
def run_eval(qa_file: str = "data/golden_qa.json", sample_step: int = 10, use_hyde: bool = False):

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

    # Sample a few pairs for evaluation
    selected_pairs = pairs[::sample_step]
    print(f"🚀 Starting evaluation on {len(selected_pairs)} samples (HyDE: {use_hyde})...\n")

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
                    "use_hyde": use_hyde
                },
                timeout=120,
            )

            if resp.status_code != 200:
                print(f"❌ Failed: {pair['question']} → {resp.text}")
                continue

            resp_json = resp.json()
            answer = resp_json.get("answer", "")
            raw_contexts = resp_json.get("contexts")

            if raw_contexts and isinstance(raw_contexts, list):
                contexts = raw_contexts
            else:
                # fallback: extract from chunks dicts
                chunks = resp_json.get("chunks", [])
                contexts = [c.get("text", "") for c in chunks]

            # Clean contexts
            contexts = [c.strip() for c in contexts if c and len(c.strip()) > 30]

            if not contexts:
                print("⚠️ No valid context returned → skipping")
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

    dataset = Dataset.from_dict(data)
    print("\n🚀 Running RAGAS evaluation (using Groq + Local Embeddings)...\n")

    # ----------------------------
    # Run Evaluation
    # ----------------------------
    try:
        # Initialize metrics with Groq LLM
        faithfulness_m = Faithfulness(llm=llm)
        answer_relevancy_m = AnswerRelevancy(llm=llm, embeddings=embeddings)
        context_precision_m = ContextPrecision(llm=llm)

        from ragas import RunConfig
        config = RunConfig(max_workers=2) # Slow but steady to avoid Groq rate limits

        result = evaluate(
            dataset,
            metrics=[
                faithfulness_m,
                answer_relevancy_m,
                context_precision_m,
            ],
            run_config=config
        )

        print("📊 RAGAS Results:")
        print(result)

        # Save results
        df = result.to_pandas()
        os.makedirs("eval", exist_ok=True)
        df.to_csv("eval/results.csv", index=False)
        print("\n✅ Results saved to eval/results.csv")

        # Summary and bad question tracking
        available_cols = df.columns.tolist()
        print("\n📊 Per-metric summary:")
        for col in ["faithfulness", "answer_relevancy", "context_precision"]:
            if col in available_cols:
                mean_val = df[col].mean()
                print(f"   {col}: mean={mean_val:.3f}")

        print("\n🚨 Questions needing attention (faithfulness < 0.6 or relevancy < 0.6):")
        if all(c in available_cols for c in ["faithfulness", "answer_relevancy", "question"]):
            bad = df[(df["faithfulness"] < 0.6) | (df["answer_relevancy"] < 0.6)]
            if bad.empty:
                print("   None — all above threshold ✅")
            else:
                cols_to_show = ["question", "faithfulness", "answer_relevancy"]
                print(bad[cols_to_show].to_string(index=False))
                bad.to_csv("eval/failing_questions.csv", index=False)
                print("\n   Saved to eval/failing_questions.csv")

    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the Hindi-GATE Tutor.")
    parser.add_argument("--hyde", action="store_true", help="Use HyDE for retrieval during evaluation.")
    parser.add_argument("--sample_step", type=int, default=10, help="Sample every Nth question from the golden dataset.")
    parser.add_argument("--qa_file", type=str, default="data/golden_qa.json", help="Path to the golden QA dataset.")
    
    args = parser.parse_args()
    
    run_eval(qa_file=args.qa_file, sample_step=args.sample_step, use_hyde=args.hyde)
