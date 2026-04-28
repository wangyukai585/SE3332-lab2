"""
Lab2: Multi-hop QA with RAG – Improved System
Strategy: Query Decomposition + Iterative RAG
  1. Use GPT-4o-mini to decompose the multi-hop question into 2-3 atomic sub-questions.
  2. Answer each sub-question in sequence with BM25 retrieval, feeding prior sub-answers
     as additional context for subsequent retrievals (chain-of-thought grounding).
  3. Synthesize the final answer from all sub-answers.
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
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results", "outputs_improved.json")

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
TOP_K = 5

client = OpenAI(api_key=API_KEY)

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_dataset(file_path, num_samples=None):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data[:num_samples] if num_samples else data

# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_context(context, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    if scores[top_indices[0]] > 0:
        top_indices = [i for i in top_indices if scores[i] > 0][:top_k]
    return [chunks[i] for i in top_indices]

# ─── LLM Calls ────────────────────────────────────────────────────────────────

def query_llm(messages, max_tokens=256):
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

# ─── Step 1: Decompose ────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM = (
    "You are an expert at decomposing complex multi-hop questions into simpler atomic sub-questions. "
    "Each sub-question should be answerable from a single piece of text. "
    "Output ONLY the sub-questions, one per line, no numbering, no extra text."
)

def decompose_question(question):
    """Return a list of 2-3 atomic sub-questions for the given multi-hop question."""
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Multi-hop question: {question}\n\n"
                "Break this into 2-3 simpler sub-questions that together lead to the answer. "
                "Make each sub-question specific and independently answerable. "
                "Output only the sub-questions, one per line."
            ),
        },
    ]
    result = query_llm(messages, max_tokens=200)
    if not result:
        return [question]
    sub_qs = [q.strip().lstrip("-•·0123456789.) ") for q in result.splitlines() if q.strip()]
    # Deduplicate and keep at most 3
    seen = set()
    unique = []
    for q in sub_qs:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    return unique[:3] if unique else [question]

# ─── Step 2: Iterative Sub-Question Answering ─────────────────────────────────

def answer_sub_question(sub_question, chunks, prior_context=""):
    """Answer a single sub-question using BM25 retrieval + optional prior context."""
    # Enrich query with prior evidence to help retrieval of subsequent hops
    enriched_query = (sub_question + " " + prior_context).strip()
    retrieved = retrieve_chunks(enriched_query, chunks)
    retrieved_text = "\n\n".join(retrieved)

    context_block = retrieved_text
    if prior_context:
        context_block = f"Previously established facts:\n{prior_context}\n\nRetrieved passages:\n{retrieved_text}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise answering assistant. "
                "Answer based strictly on the provided context. "
                "Give only the answer phrase, no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context_block}\n\n"
                f"Question: {sub_question}\n\n"
                "Give only the exact answer phrase (name, date, place, number, or short phrase). "
                "If you cannot determine the answer from the context, write 'Unknown'."
            ),
        },
    ]
    return query_llm(messages, max_tokens=64)

# ─── Step 3: Synthesis ────────────────────────────────────────────────────────

def synthesize_answer(original_question, sub_qa_pairs):
    """Combine sub-question answers to produce the final answer."""
    sub_qa_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in sub_qa_pairs)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise question-answering assistant. "
                "Use the provided sub-question answers to answer the original question. "
                "Give only the final answer phrase, no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Sub-questions and their answers:\n{sub_qa_text}\n\n"
                f"Original question: {original_question}\n\n"
                "Based on the sub-answers above, provide the final answer. "
                "For Yes/No questions answer only 'Yes' or 'No'. "
                "For all other questions give only the exact answer phrase "
                "(a name, date, place, number, or short phrase), no full sentence."
            ),
        },
    ]
    return query_llm(messages, max_tokens=64)

# ─── Full Pipeline ────────────────────────────────────────────────────────────

def answer_with_decomposition(question, context):
    chunks = chunk_context(context)

    # Decompose
    sub_questions = decompose_question(question)

    # Iteratively answer each sub-question
    sub_qa_pairs = []
    prior_context = ""
    for sq in sub_questions:
        ans = answer_sub_question(sq, chunks, prior_context)
        if ans and ans.lower() != "unknown":
            prior_context += f"{sq}: {ans}. "
        sub_qa_pairs.append((sq, ans or "Unknown"))

    # Synthesize
    final_answer = synthesize_answer(question, sub_qa_pairs)
    return final_answer

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
    for item in tqdm(dataset, desc="Query-Decomp RAG"):
        pred = answer_with_decomposition(item["question"], item["context"])
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
