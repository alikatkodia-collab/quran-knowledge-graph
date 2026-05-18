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

# ── embedding model (lazy singleton via shared registry) ─────────────────────

def _get_model():
    """Return the shared all-MiniLM-L6-v2 instance from model_registry."""
    from model_registry import get_minilm
    return get_minilm()


# ── cache I/O ─────────────────────────────────────────────────────────────────
#
# Each /chat request used to read answer_cache.json twice (once in
# search_cache, once in save_answer) and write it once. At 1500+ entries
# that totals ~1.9s of synchronous JSON I/O per request — measured tax
# of ~95s across the 50-question baseline run on 2026-05-17. See
# data/research/server_degradation_diagnosis_2026-05-19/DIAGNOSIS.md
# (bonus finding section).
#
# Fix: in-memory cache keyed by file mtime. First call reads from disk
# and populates the singleton. Subsequent calls return the singleton
# unless the file mtime has changed since the last read (another
# process / overnight seeder modified the file). Writes update the
# singleton in-place so the next read is a no-op.
#
# Trade-off: a second process modifying the file concurrently could
# race with a save here (last-writer-wins on the whole file). Acceptable
# for a non-transactional augmentation cache — the cap-at-5000 already
# acknowledges this isn't a database — and the alternative (locking)
# would re-introduce the I/O tax it's meant to eliminate.

_entries_cache: list[dict] | None = None
_cache_mtime: float = 0.0


def _reset_memory_cache_for_tests() -> None:
    """Test hook: drop the in-memory singleton. Not part of the public API."""
    global _entries_cache, _cache_mtime
    _entries_cache = None
    _cache_mtime = 0.0


def _load_cache() -> list[dict]:
    global _entries_cache, _cache_mtime

    if not CACHE_FILE.exists():
        _entries_cache = None
        _cache_mtime = 0.0
        return []

    current_mtime = CACHE_FILE.stat().st_mtime
    if _entries_cache is not None and current_mtime == _cache_mtime:
        return _entries_cache

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _entries_cache = json.load(f)
        _cache_mtime = current_mtime
        return _entries_cache
    except (json.JSONDecodeError, IOError):
        _entries_cache = None
        _cache_mtime = 0.0
        return []


def _save_cache(entries: list[dict]):
    global _entries_cache, _cache_mtime
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
    # Sync the in-memory view to what we just wrote so the next read
    # is served from memory.
    _entries_cache = entries
    _cache_mtime = CACHE_FILE.stat().st_mtime


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

    # dedupe: if a near-identical question exists (similarity > 0.98), update it
    # 0.98 threshold lets semantic siblings ("Ramadan fasting" vs "fasting in Ramadan exceptions")
    # coexist while still catching exact/trivial rephrases.
    for entry in entries:
        sim = float(np.dot(emb, entry["embedding"]))
        if sim > 0.98:
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

    # cap at 5000 entries — evict oldest if needed.
    # Previous cap of 500 was tight; we're seeding aggressively now.
    if len(entries) > 5000:
        entries.sort(key=lambda e: e["timestamp"])
        entries = entries[-5000:]

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
