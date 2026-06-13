# evaluation/evaluate.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import pandas as pd
from datasets import load_dataset
from agents.retrieval_agent import retrieve
from agents.query_agent import decompose_query
from config import CONFIDENCE_THRESHOLD

print("=" * 60)
print("AgentRAG Evaluation vs Baseline RAG")
print("=" * 60)

# ── Load dataset ─────────────────────────────────────────────
print("\n[1/4] Loading evaluation dataset...")
dataset      = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
test_samples = list(dataset)[:20]
print(f"      Using {len(test_samples)} questions")

# ── Answer matching — NO API needed ─────────────────────────
def keyword_correct(retrieved_docs: list, ground_truth: str) -> bool:
    """
    Checks if retrieved documents contain signals matching ground truth.
    No LLM call needed — pure keyword matching on retrieved passages.
    """
    combined = " ".join([d["text"].lower() for d in retrieved_docs])
    gt       = ground_truth.lower().strip()

    if gt == "yes":
        positive = ["significant", "effective", "associated", "improves",
                    "reduces", "increases", "beneficial", "supports",
                    "demonstrated", "found that", "shown to", "evidence"]
        negative = ["not significant", "no evidence", "no association",
                    "not effective", "failed to", "no difference"]
        has_pos  = any(p in combined for p in positive)
        has_neg  = any(n in combined for n in negative)
        return has_pos and not has_neg

    elif gt == "no":
        negative = ["not significant", "no evidence", "no association",
                    "not effective", "failed to", "no difference",
                    "no significant", "did not", "does not"]
        return any(n in combined for n in negative)

    elif gt == "maybe":
        maybe = ["unclear", "insufficient", "limited evidence",
                 "further research", "inconclusive", "mixed results",
                 "some evidence", "may ", "might ", "possible",
                 "potentially", "not conclusive", "more research"]
        return any(m in combined for m in maybe)

    return False


def avg_confidence(docs: list) -> float:
    if not docs:
        return 0.0
    return round(sum(d["score"] for d in docs) / len(docs), 4)


# ── Baseline RAG — single query, single retrieval ────────────
def baseline_rag_retrieval(question: str) -> dict:
    result = retrieve(question)
    return {
        "docs"      : result["results"],
        "confidence": result["confidence"],
        "confident" : result["is_confident"]
    }


# ── AgentRAG Retrieval — decompose + multi retrieve ──────────
def agentrag_retrieval(question: str) -> dict:
    try:
        sub_queries = decompose_query(question)
        time.sleep(2)  # Small delay after LLM call
    except Exception:
        sub_queries = [question]

    all_docs   = []
    all_conf   = []
    reformulations = 0

    for q in sub_queries:
        result = retrieve(q)
        # Count reformulation triggers
        if not result["is_confident"]:
            reformulations += 1
        all_docs.extend(result["results"][:2])
        all_conf.append(result["confidence"])

    mean_conf = round(sum(all_conf) / len(all_conf), 4) if all_conf else 0.0

    return {
        "docs"          : all_docs,
        "confidence"    : mean_conf,
        "sub_queries"   : sub_queries,
        "reformulations": reformulations,
        "confident"     : mean_conf >= CONFIDENCE_THRESHOLD
    }


# ── Run Evaluation ────────────────────────────────────────────
print("\n[2/4] Running evaluation...")
print("      Retrieval-based metrics — no API rate limits\n")

results = []

for i, sample in enumerate(test_samples):
    question     = sample["question"]
    ground_truth = sample["final_decision"]

    print(f"  [{i+1:02d}/20] {question[:55]}...")

    # Baseline
    base = baseline_rag_retrieval(question)
    base_correct = keyword_correct(base["docs"], ground_truth)

    # AgentRAG
    agnt = agentrag_retrieval(question)
    agnt_correct = keyword_correct(agnt["docs"], ground_truth)

    # Confidence delta
    conf_delta = round(agnt["confidence"] - base["confidence"], 4)

    print(f"         GT: {ground_truth:<6} | "
          f"Base: {'✅' if base_correct else '❌'} ({base['confidence']:.3f}) | "
          f"AgentRAG: {'✅' if agnt_correct else '❌'} ({agnt['confidence']:.3f}) | "
          f"Δ {conf_delta:+.3f} | "
          f"Sub-queries: {len(agnt['sub_queries'])}")

    results.append({
        "question"           : question,
        "ground_truth"       : ground_truth,
        "baseline_correct"   : base_correct,
        "baseline_confidence": base["confidence"],
        "agentrag_correct"   : agnt_correct,
        "agentrag_confidence": agnt["confidence"],
        "sub_queries_count"  : len(agnt["sub_queries"]),
        "reformulations"     : agnt["reformulations"],
        "conf_delta"         : conf_delta
    })

# ── Compute Metrics ───────────────────────────────────────────
print("\n[3/4] Computing metrics...")
df = pd.DataFrame(results)

base_acc   = df["baseline_correct"].mean() * 100
agnt_acc   = df["agentrag_correct"].mean() * 100
base_conf  = df["baseline_confidence"].mean()
agnt_conf  = df["agentrag_confidence"].mean()
acc_delta  = agnt_acc - base_acc
conf_delta = agnt_conf - base_conf

avg_subq   = df["sub_queries_count"].mean()
total_ref  = df["reformulations"].sum()
ref_rate   = (df["reformulations"] > 0).mean() * 100

# Cases where AgentRAG helped
agentrag_helped  = len(df[(df["agentrag_correct"]) & (~df["baseline_correct"])])
baseline_helped  = len(df[(df["baseline_correct"]) & (~df["agentrag_correct"])])
both_correct     = len(df[df["agentrag_correct"] & df["baseline_correct"]])
both_wrong       = len(df[~df["agentrag_correct"] & ~df["baseline_correct"]])

# ── Save Results ──────────────────────────────────────────────
print("\n[4/4] Saving...")
os.makedirs("evaluation", exist_ok=True)
df.to_csv("evaluation/results.csv", index=False)

summary = {
    "total_questions"         : len(results),
    "baseline_accuracy"       : round(base_acc,  2),
    "agentrag_accuracy"       : round(agnt_acc,  2),
    "accuracy_improvement"    : round(acc_delta, 2),
    "baseline_avg_confidence" : round(base_conf, 4),
    "agentrag_avg_confidence" : round(agnt_conf, 4),
    "confidence_improvement"  : round(conf_delta,4),
    "avg_sub_queries"         : round(avg_subq,  2),
    "reformulation_rate_pct"  : round(ref_rate,  2),
    "agentrag_helped_cases"   : agentrag_helped,
    "baseline_helped_cases"   : baseline_helped,
    "both_correct"            : both_correct,
    "both_wrong"              : both_wrong
}

with open("evaluation/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ── Final Report ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"\n{'Metric':<38} {'Baseline':>10} {'AgentRAG':>10}")
print("-" * 60)
print(f"{'Answer Accuracy (%)':<38} {base_acc:>9.1f}% {agnt_acc:>9.1f}%")
print(f"{'Avg Retrieval Confidence':<38} {base_conf:>10.4f} {agnt_conf:>10.4f}")
print(f"{'Accuracy Improvement':<38} {'':>10} {acc_delta:>+9.1f}%")
print(f"{'Confidence Improvement':<38} {'':>10} {conf_delta:>+9.4f}")
print(f"\n{'AgentRAG-specific Metrics':<38}")
print("-" * 60)
print(f"{'Avg Sub-queries per Question':<38} {avg_subq:>19.2f}")
print(f"{'Reformulation Triggered Rate':<38} {ref_rate:>18.1f}%")
print(f"{'Cases AgentRAG fixed baseline error':<38} {agentrag_helped:>19}")
print(f"{'Cases baseline beat AgentRAG':<38} {baseline_helped:>19}")
print(f"{'Both correct':<38} {both_correct:>19}")
print(f"{'Both wrong':<38} {both_wrong:>19}")
print("\n" + "=" * 60)
print("Saved: evaluation/results.csv + evaluation/summary.json")
print("=" * 60)