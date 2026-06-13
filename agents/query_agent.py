# agents/query_agent.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from config import GROQ_API_KEY, MODEL_NAME

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)

def decompose_query(user_query: str) -> list[str]:
    """
    Detects if query is compound and splits into atomic sub-queries.
    Returns list of sub-queries.
    """
    prompt = f"""
You are a query decomposition specialist.

Analyze this question: "{user_query}"

If it is a simple single question, return it as-is.
If it contains multiple parts or requires chaining facts, decompose it.

Rules:
- Return ONLY a numbered list of sub-queries
- Maximum 3 sub-queries
- Each sub-query must be self-contained and searchable
- No explanations, no preamble

Example input: "What are the side effects of metformin and how does it compare to insulin for elderly patients?"
Example output:
1. What are the side effects of metformin?
2. What are the treatment outcomes of insulin for elderly diabetes patients?
3. How does metformin compare to insulin in elderly patients?
"""
    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Parse numbered list into clean list
    lines = raw.split("\n")
    queries = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            # Remove numbering like "1." or "1)"
            clean = line.split(".", 1)[-1].strip()
            clean = clean.split(")", 1)[-1].strip()
            if clean:
                queries.append(clean)

    # Fallback: if parsing fails, return original
    if not queries:
        queries = [user_query]

    return queries


def reformulate_query(original_query: str, failed_docs: list[str], attempt: int) -> str:
    """
    Rewrites a failed query using signals from low-scoring retrieved docs.
    """
    context = "\n".join(failed_docs[:2]) if failed_docs else "No relevant documents found."

    prompt = f"""
You are a query reformulation specialist.

Original query: "{original_query}"
Attempt number: {attempt}

The previous retrieval returned these poorly matching documents:
{context}

Rewrite the query to be more specific and likely to retrieve relevant medical information.
- Use different medical terminology
- Be more specific
- Focus on the core medical concept

Return ONLY the rewritten query. No explanation.
"""
    response = llm.invoke(prompt)
    return response.content.strip()


if __name__ == "__main__":
    print("Testing Query Agent...")
    print("\n--- Decomposition Test ---")
    queries = decompose_query(
        "What are the side effects of metformin and how does it compare to insulin for elderly patients?"
    )
    for i, q in enumerate(queries, 1):
        print(f"  Sub-query {i}: {q}")

    print("\n--- Reformulation Test ---")
    reformed = reformulate_query(
        "diabetes treatment elderly",
        ["document about heart surgery", "document about pediatric care"],
        attempt=1
    )
    print(f"  Reformulated: {reformed}")
    print("\n✅ Query Agent working")