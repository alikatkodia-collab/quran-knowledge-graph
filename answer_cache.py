"""
Answer Cache — stores past Q&A pairs and retrieves relevant ones for new questions.

Uses embedding similarity to find previous answers that are relevant to the
current question. Matching answers are injected into the system prompt so
Claude can reuse them instead of making full tool-use loops.

Storage: data/answer_cache.json
Embeddings: same all-MiniLM-L6-v2 model used elsewhere in the project.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

# ── file path ─────────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "data"
CACHE_FILE = CACHE_DIR / "answer_cache.json"

# ── embedding model (lazy singleton) ──────────────────────────────────────────

_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ── cache I/O ─────────────────────────────────────────────────────────────────

def _load_cache() -> list[dict]:
    if not CACHE_FILE.exists():
        return []
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_cache(entries: list[dict]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)


# ── public API ────────────────────────────────────────────────────────────────

def save_answer(question: str, answer: str, verses: dict | None = None):
    """
    Save a Q&A pair to the cache with its embedding.

    Args:
        question: the user's question
        answer:   Claude's full response text
        verses:   dict of {verse_id: {text, arabic}} cited in the answer
    """
    if not question.strip() or not answer.strip():
        return
    if len(answer) < 50:
        return  # skip trivially short answers

    model = _get_model()
    emb = model.encode(question, normalize_embeddings=True).tolist()

    entries = _load_cache()

    # dedupe: if an almost-identical question exists (similarity > 0.95), update it
    for entry in entries:
        sim = float(np.dot(emb, entry["embedding"]))
        if sim > 0.95:
            entry["answer"] = answer
            entry["verses"] = verses or {}
            entry["timestamp"] = time.time()
            _save_cache(entries)
            return

    entries.append({
        "question": question,
        "answer": answer,
        "verses": verses or {},
        "embedding": emb,
        "timestamp": time.time(),
    })

    # cap at 500 entries — evict oldest if needed
    if len(entries) > 500:
        entries.sort(key=lambda e: e["timestamp"])
        entries = entries[-500:]

    _save_cache(entries)
    print(f"  [cache] saved answer ({len(entries)} total)")


def search_cache(question: str, top_k: int = 3, threshold: float = 0.6) -> list[dict]:
    """
    Find past answers relevant to the current question.

    Returns up to top_k entries with similarity >= threshold.
    Each entry has: question, answer, verses, similarity.
    """
    entries = _load_cache()
    if not entries:
        return []

    model = _get_model()
    q_emb = model.encode(question, normalize_embeddings=True)

    scored = []
    for entry in entries:
        sim = float(np.dot(q_emb, entry["embedding"]))
        if sim >= threshold:
            scored.append({
                "question": entry["question"],
                "answer": entry["answer"],
                "verses": entry.get("verses", {}),
                "similarity": round(sim, 3),
            })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def build_cache_context(question: str, top_k: int = 3, threshold: float = 0.6) -> str | None:
    """
    Search the cache and format matching answers into a context block
    that can be prepended to the system prompt.

    Returns None if no relevant cached answers found.
    """
    hits = search_cache(question, top_k=top_k, threshold=threshold)
    if not hits:
        return None

    parts = [
        "PREVIOUSLY ANSWERED QUESTIONS (use these to inform your response — "
        "you may reuse verse citations and analysis from them, but still verify "
        "accuracy and add new information if needed):\n"
    ]

    for i, hit in enumerate(hits, 1):
        sim_pct = int(hit["similarity"] * 100)
        parts.append(f"--- Previous Q&A #{i} (relevance: {sim_pct}%) ---")
        parts.append(f"Q: {hit['question']}")
        # Truncate long answers to keep prompt lean
        answer = hit["answer"]
        if len(answer) > 1500:
            answer = answer[:1500] + "... [truncated]"
        parts.append(f"A: {answer}\n")

    return "\n".join(parts)


def cache_stats() -> dict:
    """Return cache statistics."""
    entries = _load_cache()
    if not entries:
        return {"total_entries": 0}

    timestamps = [e["timestamp"] for e in entries]
    avg_len = sum(len(e["answer"]) for e in entries) / len(entries)

    return {
        "total_entries": len(entries),
        "oldest": time.strftime("%Y-%m-%d %H:%M", time.localtime(min(timestamps))),
        "newest": time.strftime("%Y-%m-%d %H:%M", time.localtime(max(timestamps))),
        "avg_answer_length": int(avg_len),
    }
