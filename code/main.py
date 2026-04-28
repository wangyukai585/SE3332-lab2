"""
Lab2: Multi-hop QA with RAG
Basic system: BM25 retrieval over chunked context + GPT-4o-mini generation
"""

import json
import re
import string
import collections
import os
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from rank_bm25 import BM25Okapi

# ─── Configuration ───────────────────────────────────────────────────────────

API_KEY = "YOUR_OPENAI_API_KEY_HERE"  # ← replace with your actual OpenAI API key
MODEL = "gpt-4o-mini"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "hotpotqa_longbench.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results", "outputs.json")

CHUNK_SIZE = 150      # words per chunk
CHUNK_OVERLAP = 30    # word overlap between adjacent chunks
TOP_K = 5             # number of chunks to retrieve

client = OpenAI(api_key=API_KEY)

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_dataset(file_path, num_samples=None):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data[:num_samples] if num_samples else data

# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_context(context, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split context into overlapping word-level chunks, respecting passage boundaries."""
    # Split on passage markers, keeping the title line as part of each passage
    raw_passages = re.split(r'\n(?=Passage \d+:\n)', context)
    chunks = []
    for passage in raw_passages:
        passage = passage.strip()
        if not passage:
            continue
        words = passage.split()
        if len(words) <= chunk_size:
            chunks.append(passage)
        else:
            for start in range(0, len(words), chunk_size - overlap):
                chunk_words = words[start:start + chunk_size]
                if chunk_words:
                    chunks.append(" ".join(chunk_words))
    return chunks

# ─── BM25 Retrieval ───────────────────────────────────────────────────────────

def retrieve_chunks(query, chunks, top_k=TOP_K):
    """Return the top_k most relevant chunks for the query using BM25."""
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    # Filter out chunks with zero score only if some have positive scores
    if scores[top_indices[0]] > 0:
        top_indices = [i for i in top_indices if scores[i] > 0][:top_k]
    return [chunks[i] for i in top_indices]

# ─── Model Interaction ────────────────────────────────────────────────────────

def query_llm(messages, max_tokens=64):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [LLM error] {e}")
        return None

# ─── RAG Answer ───────────────────────────────────────────────────────────────

def answer_with_rag(question, context):
    chunks = chunk_context(context)
    retrieved = retrieve_chunks(question, chunks)
    retrieved_text = "\n\n".join(retrieved)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise question-answering assistant. "
                "Answer strictly based on the provided context. "
                "Be concise: give only the answer phrase, no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{retrieved_text}\n\n"
                f"Question: {question}\n\n"
                "Instructions:\n"
                "- For Yes/No questions answer only 'Yes' or 'No'.\n"
                "- For all other questions give only the exact answer phrase "
                "(a name, date, place, number, or short phrase) with no full sentence."
            ),
        },
    ]
    return query_llm(messages)

# ─── Evaluation ───────────────────────────────────────────────────────────────

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_em(gold, pred):
    return int(normalize_answer(gold) == normalize_answer(pred))

def compute_f1(gold, pred):
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()
    if not gold_toks or not pred_toks:
        return int(gold_toks == pred_toks)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def evaluate(results):
    em_scores = [compute_em(r["golden_answer"], r["predicted_answer"]) for r in results]
    f1_scores = [compute_f1(r["golden_answer"], r["predicted_answer"]) for r in results]
    avg_em = np.mean(em_scores)
    avg_f1 = np.mean(f1_scores)
    for r, em, f1 in zip(results, em_scores, f1_scores):
        r["em"] = em
        r["f1"] = round(f1, 4)
    return avg_em, avg_f1

# ─── Save ─────────────────────────────────────────────────────────────────────

def save_outputs(results, output_path):
    output_data = [{"id": r["id"], "pred_answer": r["predicted_answer"]} for r in results]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(output_data)} predictions → {output_path}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def run(num_samples=200):
    print(f"Loading dataset ({num_samples} samples)...")
    dataset = load_dataset(DATA_PATH, num_samples=num_samples)
    print(f"Loaded {len(dataset)} samples.")

    results = []
    for item in tqdm(dataset, desc="Basic RAG"):
        pred = answer_with_rag(item["question"], item["context"])
        if pred is None:
            pred = ""
        results.append({
            "id": item["id"],
            "question": item["question"],
            "predicted_answer": pred,
            "golden_answer": item["answer"],
        })

    avg_em, avg_f1 = evaluate(results)
    print(f"\nFinal Average EM: {avg_em:.4f}  |  Final Average F1: {avg_f1:.4f}")

    save_outputs(results, OUTPUT_PATH)
    return results, avg_em, avg_f1

if __name__ == "__main__":
    run(num_samples=200)
