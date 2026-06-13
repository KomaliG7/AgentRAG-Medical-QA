# pipeline/agentrag.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.query_agent     import decompose_query, reformulate_query
from agents.retrieval_agent import retrieve_multi, retrieve
from agents.reasoning_agent import synthesize_answer
from agents.critic_agent    import validate_and_label
from config import MAX_REFORMULATION_ATTEMPTS, CONFIDENCE_THRESHOLD, UNKNOWN


def run_agentrag(user_query: str, verbose: bool = True) -> dict:
    """
    Full AgentRAG pipeline.
    Takes a user query and returns a fully validated, cited, labeled answer.
    """

    def log(msg):
        if verbose:
            print(msg)

    log("\n" + "=" * 60)
    log(f"AgentRAG Processing: {user_query}")
    log("=" * 60)

    # ── STEP 1: Query Decomposition ──────────────────────────────
    log("\n[1/5] Query Agent — Decomposing query...")
    sub_queries = decompose_query(user_query)
    log(f"      Sub-queries: {len(sub_queries)}")
    for i, q in enumerate(sub_queries, 1):
        log(f"        {i}. {q}")

    # ── STEP 2: Retrieval with Self-Correction Loop ──────────────
    log("\n[2/5] Retrieval Agent — Searching knowledge base...")
    all_retrievals   = []
    all_temporal_flags = []

    for q_idx, query in enumerate(sub_queries):
        log(f"\n      Sub-query {q_idx+1}: '{query}'")
        current_query = query
        retrieval     = None

        for attempt in range(1, MAX_REFORMULATION_ATTEMPTS + 1):
            retrieval = retrieve(current_query)
            confidence = retrieval["confidence"]
            log(f"        Attempt {attempt} — Confidence: {confidence}")

            if retrieval["is_confident"]:
                log(f"        ✅ Confident retrieval achieved")
                break
            else:
                if attempt < MAX_REFORMULATION_ATTEMPTS:
                    log(f"        ⚠️  Low confidence. Reformulating query...")
                    failed_docs   = [r["text"] for r in retrieval["results"]]
                    current_query = reformulate_query(
                        current_query, failed_docs, attempt
                    )
                    log(f"        Reformulated: '{current_query}'")
                else:
                    log(f"        ❌ Max attempts reached. Proceeding with best available.")

        all_retrievals.append(retrieval)

        # Collect temporal flags
        if retrieval["temporal_flags"]:
            all_temporal_flags.extend(retrieval["temporal_flags"])

    # ── STEP 3: Check Overall Confidence ────────────────────────
    log("\n[3/5] Checking overall retrieval confidence...")
    avg_confidence = sum(r["confidence"] for r in all_retrievals) / len(all_retrievals)
    log(f"      Average confidence: {round(avg_confidence, 4)}")

    if avg_confidence < CONFIDENCE_THRESHOLD * 0.5:
        log("      ❌ Confidence too low across all sub-queries.")
        return {
            "query"            : user_query,
            "answer"           : "AgentRAG could not retrieve sufficient evidence to answer this question reliably.",
            "label"            : UNKNOWN,
            "reason"           : "Retrieval confidence too low across all sub-queries",
            "sources"          : [],
            "contradictions"   : [],
            "temporal_flags"   : all_temporal_flags,
            "sub_queries"      : sub_queries,
            "avg_confidence"   : avg_confidence
        }

    # ── STEP 4: Reasoning + Contradiction Detection ──────────────
    log("\n[4/5] Reasoning Agent — Synthesizing answer...")
    reasoning_output = synthesize_answer(
        user_query,
        sub_queries,
        all_retrievals
    )

    if reasoning_output["contradictions"]:
        log(f"      ⚠️  Contradictions detected:")
        for c in reasoning_output["contradictions"]:
            log(f"        {c}")
    else:
        log("      No contradictions detected")

    # ── STEP 5: Critic Validation + Confidence Label ─────────────
    log("\n[5/5] Critic Agent — Validating and labeling...")
    final_output = validate_and_label(reasoning_output, avg_confidence)
    log(f"      Label : {final_output['label']}")
    log(f"      Reason: {final_output['reason']}")

    # ── FINAL RESULT ─────────────────────────────────────────────
    return {
        "query"          : user_query,
        "answer"         : final_output["answer"],
        "label"          : final_output["label"],
        "reason"         : final_output["reason"],
        "sources"        : final_output["sources"],
        "contradictions" : final_output["contradictions"],
        "temporal_flags" : all_temporal_flags,
        "sub_queries"    : sub_queries,
        "avg_confidence" : round(avg_confidence, 4)
    }


def format_output(result: dict) -> str:
    """Formats the final result for clean display."""
    lines = []
    lines.append("\n" + "╔" + "═" * 58 + "╗")
    lines.append("║" + " AGENTRAG RESULT ".center(58) + "║")
    lines.append("╠" + "═" * 58 + "╣")

    # Confidence Label
    lines.append(f"║  Confidence : {result['label']}".ljust(59) + "║")
    lines.append(f"║  Reason     : {result['reason'][:42]}".ljust(59) + "║")
    lines.append("╠" + "═" * 58 + "╣")

    # Answer
    lines.append("║  ANSWER:".ljust(59) + "║")
    answer_lines = [result["answer"][i:i+54] for i in range(0, min(len(result["answer"]), 540), 54)]
    for line in answer_lines:
        lines.append(f"║  {line}".ljust(59) + "║")
    lines.append("╠" + "═" * 58 + "╣")

    # Sources
    lines.append("║  SOURCES:".ljust(59) + "║")
    for i, src in enumerate(result["sources"][:3], 1):
        lines.append(f"║  [{i}] PubMed {src['pubmed_id']} | Score: {src['score']} | {src['year']}".ljust(59) + "║")

    # Contradictions
    if result["contradictions"]:
        lines.append("╠" + "═" * 58 + "╣")
        lines.append("║  ⚠️  CONTRADICTIONS DETECTED:".ljust(59) + "║")
        for c in result["contradictions"]:
            lines.append(f"║  {c[:54]}".ljust(59) + "║")

    # Temporal Flags
    if result["temporal_flags"]:
        lines.append("╠" + "═" * 58 + "╣")
        lines.append("║  ⏰ TEMPORAL FLAGS:".ljust(59) + "║")
        for t in result["temporal_flags"][:2]:
            lines.append(f"║  {t[:54]}".ljust(59) + "║")

    lines.append("╚" + "═" * 58 + "╝")
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with a real medical question
    test_query = "What are the effects of metformin in elderly patients with type 2 diabetes?"

    result = run_agentrag(test_query, verbose=True)
    print(format_output(result))