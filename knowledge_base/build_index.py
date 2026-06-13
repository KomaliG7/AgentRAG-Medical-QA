# knowledge_base/build_index.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import json

print("=" * 50)
print("AgentRAG — Building Knowledge Base")
print("=" * 50)

# ── Step 1: Load PubMedQA Dataset ──────────────────
print("\n[1/5] Loading PubMedQA dataset...")
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
print(f"      Loaded {len(dataset)} records")

# ── Step 2: Extract Documents ──────────────────────
print("\n[2/5] Extracting documents...")
documents = []
metadata  = []

for item in dataset:
    # Each item has contexts (list of passages), question, answer
    contexts = item["context"]["contexts"]
    question = item["question"]
    answer   = item["final_decision"]
    pubmed_id = str(item["pubid"])

    for i, passage in enumerate(contexts):
        if passage.strip():
            documents.append(passage.strip())
            metadata.append({
                "pubmed_id"  : pubmed_id,
                "question"   : question,
                "answer"     : answer,
                "passage_idx": i,
                "year"       : 2020   # PubMedQA default year tag
            })

print(f"      Extracted {len(documents)} passages")

# ── Step 3: Embed Documents ────────────────────────
print("\n[3/5] Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("      Embedding documents (this takes 2-5 minutes)...")
embeddings = embedder.encode(
    documents,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"      Embedding shape: {embeddings.shape}")

# ── Step 4: Build FAISS Index ──────────────────────
print("\n[4/5] Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)          # Inner product = cosine sim

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)
index.add(embeddings)
print(f"      Index built with {index.ntotal} vectors")

# ── Step 5: Save Everything ────────────────────────
print("\n[5/5] Saving index and metadata...")
os.makedirs("data", exist_ok=True)

faiss.write_index(index, "data/pubmedqa.index")

with open("data/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

with open("data/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("\n" + "=" * 50)
print("✅ Knowledge Base Built Successfully")
print(f"   Index  : data/pubmedqa.index")
print(f"   Docs   : data/documents.pkl")
print(f"   Meta   : data/metadata.pkl")
print(f"   Total  : {len(documents)} searchable passages")
print("=" * 50)