# agents/critic_agent.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from config import GROQ_API_KEY, MODEL_NAME, VERIFIED, PARTIAL, INFERRED, UNKNOWN

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)


def validate_and_label(reasoning_output: dict, retrieval_confidence: float) -> dict:
    """
    Validates answer against sources.
    Assigns VERIFIED / PARTIAL / INFERRED / UNKNOWN label.
    """
    answer  = reasoning_output["answer"]
    sources = reasoning_output["sources"]

    source_text = "\n\n".join([
        f"[Source {i+1}]: {s['text'][:300]}"
        for i, s in enumerate(sources[:4])
    ])

    prompt = f"""
You are a medical fact-checker and answer validator.

Answer to validate:
"{answer}"

Source documents used:
{source_text}

Retrieval confidence score: {retrieval_confidence}

Task: Evaluate how well the answer is supported by the sources.

Respond with EXACTLY one of these verdicts and a one-line reason:

VERDICT: VERIFIED
Reason: [The answer is fully supported by 2 or more sources]

VERDICT: PARTIAL  
Reason: [The answer is partially supported but has gaps]

VERDICT: INFERRED
Reason: [The answer is reasoned but not directly stated in sources]

VERDICT: UNKNOWN
Reason: [The sources do not support the answer at all]
"""
    response = llm.invoke(prompt)
    raw      = response.content.strip()

    # Parse verdict
    label  = UNKNOWN
    reason = "Could not validate"

    for line in raw.split("\n"):
        if "VERDICT:" in line:
            if "VERIFIED"  in line: label = VERIFIED
            elif "PARTIAL" in line: label = PARTIAL
            elif "INFERRED"in line: label = INFERRED
            elif "UNKNOWN" in line: label = UNKNOWN
        if "Reason:" in line:
            reason = line.replace("Reason:", "").strip()

    return {
        "label"     : label,
        "reason"    : reason,
        "answer"    : answer,
        "sources"   : sources,
        "contradictions": reasoning_output.get("contradictions", [])
    }


if __name__ == "__main__":
    print("Testing Critic Agent...")

    fake_reasoning = {
        "answer": "Metformin is first-line therapy for Type 2 diabetes [Source 1]. Dose adjustment is needed for elderly patients [Source 2].",
        "sources": [
            {"text": "Metformin is first-line therapy for Type 2 diabetes.", "pubmed_id": "111", "score": 0.85, "year": 2022, "is_stale": False},
            {"text": "Elderly patients require metformin dose adjustment.", "pubmed_id": "222", "score": 0.80, "year": 2021, "is_stale": False}
        ],
        "contradictions": []
    }

    result = validate_and_label(fake_reasoning, retrieval_confidence=0.82)
    print(f"\nLabel  : {result['label']}")
    print(f"Reason : {result['reason']}")
    print("\n✅ Critic Agent working")