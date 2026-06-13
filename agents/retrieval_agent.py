# agents/retrieval_agent.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from config import (TOP_K_DOCUMENTS, CONFIDENCE_THRESHOLD,
                    TEMPORAL_THRESHOLD_YEAR)

# Load index and data once at import
print("[Retrieval Agent] Loading FAISS index...")
index     = faiss.read_index("data/pubmedqa.index")

with open("data/documents.pkl", "rb") as f:
    documents = pickle.load(f)

with open("data/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("[Retrieval Agent] Ready.")


def retrieve(query: str, top_k: int = TOP_K_DOCUMENTS) -> dict:
    """
    Retrieves top-k documents for a query.
    Returns dict with docs, scores, confidence, temporal flags.
    """
    # Embed query
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    # Search FAISS
    scores, indices = index.search(query_vec, top_k)
    scores   = scores[0].tolist()
    indices  = indices[0].tolist()

    # Build results
    results = []
    temporal_flags = []

    for score, idx in zip(scores, indices):
        if idx == -1:
            continue

        doc  = documents[idx]
        meta = metadata[idx]
        year = meta.get("year", 2020)

        # Temporal check
        is_stale = year < TEMPORAL_THRESHOLD_YEAR
        if is_stale:
            temporal_flags.append(
                f"⚠️ Document from {year} may be outdated (threshold: {TEMPORAL_THRESHOLD_YEAR})"
            )

        results.append({
            "text"     : doc,
            "score"    : round(score, 4),
            "pubmed_id": meta.get("pubmed_id", "N/A"),
            "year"     : year,
            "is_stale" : is_stale
        })

    # Confidence = average of top scores
    top_scores = [r["score"] for r in results]
    confidence = round(np.mean(top_scores), 4) if top_scores else 0.0
    confident  = confidence >= CONFIDENCE_THRESHOLD

    return {
        "query"          : query,
        "results"        : results,
        "confidence"     : confidence,
        "is_confident"   : confident,
        "temporal_flags" : temporal_flags
    }


def retrieve_multi(queries: list[str]) -> list[dict]:
    """Retrieves for multiple sub-queries independently."""
    return [retrieve(q) for q in queries]


if __name__ == "__main__":
    print("\nTesting Retrieval Agent...")
    result = retrieve("metformin side effects elderly diabetes")
    print(f"\nQuery     : {result['query']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Confident : {result['is_confident']}")
    print(f"\nTop 3 results:")
    for i, r in enumerate(result["results"][:3], 1):
        print(f"  [{i}] Score: {r['score']} | {r['text'][:100]}...")
    print("\n✅ Retrieval Agent working")