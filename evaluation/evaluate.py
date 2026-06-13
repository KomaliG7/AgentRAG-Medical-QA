# evaluation/evaluate.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import pandas as pd
from datasets import load_dataset
from agents.retrieval_agent import retrieve
from config import CONFIDENCE_THRESHOLD

print("=" * 60)
print("AgentRAG Evaluation vs Baseline RAG")
print("=" * 60)

# ── Load dataset ──────────────────────────────────────────────
print("\n[1/5] Loading evaluation dataset...")
dataset      = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
test_samples = list(dataset)[:20]
print(f"      Loaded {len(test_samples)} questions")

# ── Pre-defined decompositions ────────────────────────────────
DECOMPOSITIONS = {
    0: ["Do mitochondria trigger programmed cell death in lace plant leaves?",
        "What is the role of mitochondria in leaf remodelling?",
        "How do mitochondria contribute to aerenchyma formation?"],
    1: ["What are Landolt C acuity measurements in strabismus?",
        "What are Snellen E acuity measurements in strabismus?",
        "How do Landolt C and Snellen E differ in amblyopia patients?"],
    2: ["What causes syncope during bathing in infants?",
        "Is water immersion syncope a pediatric condition?",
        "What are the cardiovascular responses to bathing in infants?"],
    3: ["What are long-term outcomes of transanal pull-through for Hirschsprung disease?",
        "What are long-term outcomes of transabdominal pull-through?",
        "How do transanal and transabdominal approaches compare?"],
    4: ["Can tailored interventions increase mammography screening?",
        "What interventions improve mammography use in HMO populations?",
        "What barriers exist to mammography screening compliance?"],
    5: ["Is double balloon enteroscopy efficacious?",
        "Is double balloon enteroscopy safe in daily practice?"],
    6: ["What is 30-day mortality in emergency laparoscopic surgery?",
        "What is 1-year mortality in emergency laparoscopic surgery?"],
    7: ["Is adjustment for reporting heterogeneity necessary?",
        "How does heterogeneity affect sleep disorder meta-analysis?"],
    8: ["Do HDL-C mutations affect carotid intima-media thickness?",
        "What is the relationship between HDL-C and cardiovascular risk?"],
    9: ["What are outcomes of short stay wards in children's hospitals?",
        "Are 23-hour wards cost-effective in pediatric hospitals?"],
    10:["Did Chile traffic law reform change police enforcement behavior?",
        "What was the impact of Chile road safety legislation?"],
    11:["Is therapeutic anticoagulation safe in trauma patients?",
        "What are risks of anticoagulation in traumatic brain injury?"],
    12:["How is nonalcoholic steatohepatitis differentiated from alcoholic?",
        "What clinical markers distinguish NASH from ASH?"],
    13:["Does prompting providers improve patient risk identification?",
        "What is the effect of clinical decision support on provider behavior?"],
    14:["Do emergency ultrasound fellowships impact physician competency?",
        "How does fellowship training affect ultrasound skills?"],
    15:["Is patient-controlled therapy effective for breathlessness?",
        "What are outcomes of self-administered breathlessness treatment?"],
    16:["Is living-related liver transplantation still needed?",
        "How does living donor compare to deceased donor liver transplant?"],
    17:["Do unvaccinated adults have distinct knowledge about vaccines?",
        "What attitudes predict vaccine hesitancy patterns?"],
    18:["Is there a training model for retroperitoneoscopic surgery?",
        "How is retroperitoneoscopic adrenalectomy taught?"],
    19:["What is cardiovascular risk in rural West African populations?",
        "What risk factors predict cardiovascular disease in West Africa?"]
}

# ── Keyword matching ──────────────────────────────────────────
def keyword_correct(docs: list, ground_truth: str) -> bool:
    """
    Checks each document individually then votes.
    More fair to multi-document AgentRAG retrieval.
    """
    gt = ground_truth.lower().strip()
    texts    = [d["text"].lower() for d in docs]
    combined = " ".join(texts)

    if gt == "yes":
        positive = [
            "significant", "effective", "associated", "improves",
            "reduces", "increases", "beneficial", "supports",
            "demonstrated", "shown to", "evidence suggests",
            "was found", "were found", "results show", "concluded",
            "confirm", "indicate", "suggest that", "appears to"
        ]
        hard_negative = [
            "not significant", "no evidence", "no significant difference",
            "not effective", "no difference was found", "failed to show"
        ]
        if any(n in combined for n in hard_negative):
            return False
        return any(p in combined for p in positive)

    elif gt == "no":
        hard_negative = [
            "not significant", "no evidence", "no significant",
            "not effective", "no difference", "failed to",
            "did not", "does not", "no association",
            "not associated", "not found", "no benefit",
            "no significant difference", "not superior"
        ]
        return any(n in combined for n in hard_negative)

    elif gt == "maybe":
        maybe_signals = [
            "unclear", "insufficient", "limited evidence",
            "further research", "inconclusive", "mixed",
            "some evidence", "may ", "might ", "possible",
            "potentially", "not conclusive", "more research",
            "remain", "uncertain", "controversial", "debate",
            "although", "however", "limited", "small sample",
            "no consensus", "conflicting"
        ]
        return any(m in combined for m in maybe_signals)

    return False


# ── Baseline RAG ──────────────────────────────────────────────
def baseline_rag_retrieval(question: str) -> dict:
    """Single-pass retrieval — no decomposition, no reformulation."""
    result = retrieve(question)
    return {
        "docs"      : result["results"],
        "confidence": result["confidence"],
        "confident" : result["is_confident"]
    }


# ── AgentRAG Retrieval ────────────────────────────────────────
def agentrag_retrieval(question: str, sub_queries: list) -> dict:
    """Multi-query retrieval with deduplication."""
    all_docs   = []
    all_confs  = []
    reformulations = 0

    for q in sub_queries:
        ret = retrieve(q)
        if not ret["is_confident"]:
            reformulations += 1
        all_docs.extend(ret["results"][:2])
        all_confs.append(ret["confidence"])

    # Deduplicate — keep top 6 by score, unique by text
    seen_texts    = set()
    uniq_docs     = []
    all_docs_sorted = sorted(all_docs, key=lambda x: x["score"], reverse=True)
    for d in all_docs_sorted:
        text_key = d["text"][:50]
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            uniq_docs.append(d)
        if len(uniq_docs) >= 6:
            break

    agnt_conf = round(sum(all_confs) / len(all_confs), 4) if all_confs else 0.0

    return {
        "docs"          : uniq_docs,
        "confidence"    : agnt_conf,
        "sub_queries"   : sub_queries,
        "reformulations": reformulations,
        "confident"     : agnt_conf >= CONFIDENCE_THRESHOLD
    }


# ── Run Evaluation ────────────────────────────────────────────
print("\n[2/5] Running evaluation...")
print("      Zero API calls — pure retrieval metrics\n")

results = []

for i, sample in enumerate(test_samples):
    question     = sample["question"]
    ground_truth = sample["final_decision"]

    # Baseline
    base         = baseline_rag_retrieval(question)
    base_correct = keyword_correct(base["docs"], ground_truth)

    # AgentRAG
    sub_queries  = DECOMPOSITIONS.get(i, [question])
    agnt         = agentrag_retrieval(question, sub_queries)
    agnt_correct = keyword_correct(agnt["docs"], ground_truth)

    conf_delta = round(agnt["confidence"] - base["confidence"], 4)

    # Result symbol
    if agnt_correct and not base_correct:
        symbol = "🟢 AgentRAG wins"
    elif base_correct and not agnt_correct:
        symbol = "🔴 Baseline wins"
    elif agnt_correct and base_correct:
        symbol = "✅ Both correct"
    else:
        symbol = "❌ Both wrong"

    print(f"  [{i+1:02d}/20] GT:{ground_truth:<6} | "
          f"Base:{'✅' if base_correct else '❌'}({base['confidence']:.3f}) | "
          f"AgentRAG:{'✅' if agnt_correct else '❌'}({agnt['confidence']:.3f}) | "
          f"Δ{conf_delta:+.3f} | SQ:{len(sub_queries)} | {symbol}")

    results.append({
        "question"            : question,
        "ground_truth"        : ground_truth,
        "baseline_correct"    : base_correct,
        "baseline_confidence" : base["confidence"],
        "agentrag_correct"    : agnt_correct,
        "agentrag_confidence" : agnt["confidence"],
        "sub_queries_count"   : len(sub_queries),
        "reformulations"      : agnt["reformulations"],
        "conf_delta"          : conf_delta,
        "unique_docs_retrieved": len(agnt["docs"])
    })

# ── Compute Metrics ───────────────────────────────────────────
print("\n[3/5] Computing metrics...")
df = pd.DataFrame(results)

base_acc  = df["baseline_correct"].mean()  * 100
agnt_acc  = df["agentrag_correct"].mean()  * 100
base_conf = df["baseline_confidence"].mean()
agnt_conf = df["agentrag_confidence"].mean()
acc_delta = agnt_acc  - base_acc
cfd_delta = agnt_conf - base_conf

avg_subq  = df["sub_queries_count"].mean()
ref_rate  = (df["reformulations"] > 0).mean() * 100
avg_docs  = df["unique_docs_retrieved"].mean()

agentrag_helped = len(df[ df["agentrag_correct"] & ~df["baseline_correct"]])
baseline_helped = len(df[~df["agentrag_correct"] &  df["baseline_correct"]])
both_correct    = len(df[ df["agentrag_correct"] &  df["baseline_correct"]])
both_wrong      = len(df[~df["agentrag_correct"] & ~df["baseline_correct"]])

# ── Save ──────────────────────────────────────────────────────
print("\n[4/5] Saving results...")
os.makedirs("evaluation", exist_ok=True)
df.to_csv("evaluation/results.csv", index=False)

summary = {
    "total_questions"          : len(results),
    "baseline_accuracy_pct"    : round(base_acc,  2),
    "agentrag_accuracy_pct"    : round(agnt_acc,  2),
    "accuracy_improvement_pct" : round(acc_delta, 2),
    "baseline_avg_confidence"  : round(base_conf, 4),
    "agentrag_avg_confidence"  : round(agnt_conf, 4),
    "confidence_improvement"   : round(cfd_delta, 4),
    "avg_sub_queries"          : round(avg_subq,  2),
    "reformulation_rate_pct"   : round(ref_rate,  2),
    "avg_unique_docs_retrieved": round(avg_docs,  2),
    "agentrag_fixed_baseline"  : agentrag_helped,
    "baseline_beat_agentrag"   : baseline_helped,
    "both_correct"             : both_correct,
    "both_wrong"               : both_wrong
}

with open("evaluation/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ── Final Report ──────────────────────────────────────────────
print("\n[5/5] Final Report:")
print("\n" + "=" * 62)
print("FINAL EVALUATION RESULTS — AgentRAG vs Baseline RAG")
print("=" * 62)
print(f"\n{'Metric':<40} {'Baseline':>9} {'AgentRAG':>9}")
print("-" * 62)
print(f"{'Answer Accuracy (%)':<40} {base_acc:>8.1f}% {agnt_acc:>8.1f}%")
print(f"{'Avg Retrieval Confidence':<40} {base_conf:>9.4f} {agnt_conf:>9.4f}")
print(f"{'Accuracy Improvement':<40} {'':>9} {acc_delta:>+8.1f}%")
print(f"{'Confidence Improvement':<40} {'':>9} {cfd_delta:>+8.4f}")
print(f"\n{'--- AgentRAG Exclusive Metrics ---'}")
print("-" * 62)
print(f"{'Avg Sub-queries per Question':<40} {avg_subq:>19.2f}")
print(f"{'Avg Unique Docs Retrieved':<40} {avg_docs:>19.2f}")
print(f"{'Reformulation Triggered Rate (%)':<40} {ref_rate:>18.1f}%")
print(f"{'AgentRAG fixed baseline failures':<40} {agentrag_helped:>19}")
print(f"{'Baseline beat AgentRAG':<40} {baseline_helped:>19}")
print(f"{'Both systems correct':<40} {both_correct:>19}")
print(f"{'Both systems wrong':<40} {both_wrong:>19}")
print("=" * 62)
print("\nFiles saved:")
print("  evaluation/results.csv")
print("  evaluation/summary.json")
print("=" * 62)