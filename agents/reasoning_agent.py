# agents/reasoning_agent.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from config import GROQ_API_KEY, MODEL_NAME

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)


def detect_contradictions(docs: list[dict]) -> list[str]:
    """
    Checks if retrieved documents contradict each other.
    """
    if len(docs) < 2:
        return []

    passages = "\n\n".join([
        f"Source {i+1} (PubMed {d['pubmed_id']}): {d['text'][:300]}"
        for i, d in enumerate(docs[:4])
    ])

    prompt = f"""
You are a medical evidence analyst.

Analyze these retrieved medical documents for contradictions:

{passages}

Instructions:
- If documents clearly contradict each other on a medical claim, state it
- Format: "CONTRADICTION: Source X claims [A] but Source Y claims [B]"
- If no contradictions found, return exactly: "NO_CONTRADICTIONS"
- Maximum 2 contradictions
- Be specific, not vague
"""
    response = llm.invoke(prompt)
    raw = response.content.strip()

    if "NO_CONTRADICTIONS" in raw:
        return []

    contradictions = []
    for line in raw.split("\n"):
        if "CONTRADICTION:" in line:
            contradictions.append(line.strip())

    return contradictions


def synthesize_answer(
    original_query: str,
    sub_queries    : list[str],
    all_retrievals : list[dict]
) -> dict:
    """
    Synthesizes a cited answer from all retrieved documents.
    """
    # Build context from all retrievals
    context_parts = []
    all_docs = []

    for i, (query, retrieval) in enumerate(zip(sub_queries, all_retrievals)):
        for doc in retrieval["results"][:2]:
            context_parts.append(
                f"[Source {len(all_docs)+1} | PubMed {doc['pubmed_id']}]\n{doc['text'][:400]}"
            )
            all_docs.append(doc)

    context = "\n\n".join(context_parts)

    # Detect contradictions
    contradictions = detect_contradictions(all_docs)

    prompt = f"""
You are a medical research synthesizer.

Original question: "{original_query}"

Retrieved evidence:
{context}

Instructions:
- Answer the question using ONLY the provided sources
- Cite sources using [Source N] notation after each claim
- If sources contradict each other, acknowledge both sides
- Be concise but complete
- If evidence is insufficient, say so clearly

Provide your answer now:
"""
    response = llm.invoke(prompt)
    answer   = response.content.strip()

    return {
        "original_query" : original_query,
        "answer"         : answer,
        "sources"        : all_docs,
        "contradictions" : contradictions
    }


if __name__ == "__main__":
    print("Testing Reasoning Agent...")

    fake_retrieval = {
        "results": [
            {"text": "Metformin is first-line therapy for Type 2 diabetes. It reduces HbA1c by 1-2%.",
             "pubmed_id": "12345", "score": 0.8, "year": 2022, "is_stale": False},
            {"text": "In elderly patients, metformin requires dose adjustment due to renal function decline.",
             "pubmed_id": "67890", "score": 0.75, "year": 2021, "is_stale": False}
        ]
    }

    result = synthesize_answer(
        "What is the recommended treatment for Type 2 diabetes?",
        ["What is the recommended treatment for Type 2 diabetes?"],
        [fake_retrieval]
    )

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nContradictions: {result['contradictions']}")
    print("\n✅ Reasoning Agent working")