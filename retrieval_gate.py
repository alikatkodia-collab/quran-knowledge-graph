"""
Retrieval Quality Gating — Phase 2 of hallucination reduction.

Cross-encoder reranking, quality assessment, lost-in-middle reordering,
and Corrective RAG fallback.

Used by chat.py to post-process tool results before feeding them to Claude.

Reranker model is multilingual by default (BAAI/bge-reranker-v2-m3) to
match the BGE-M3 retrieval path and to handle Arabic queries. The legacy
ms-marco-MiniLM-L-6-v2 (English-only) was actively HURTING our QRCD
numbers (hit@10 fell from 0.64 -> 0.32 on Arabic queries in the
ablation eval). Override via env: RERANKER_MODEL=...

Also: set RERANKER_MODEL=none (or RERANK_DISABLED=1) to skip reranking
entirely. The ablation showed vector-only ranks better than legacy
rerank on QRCD; multilingual rerank should fix that, but the kill switch
is here for safety.
"""

import os
from sentence_transformers import CrossEncoder

import config as cfg

# ── cross-encoder reranker (loaded once) ─────────────────────────────────────

_DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", _DEFAULT_RERANKER).strip()
RERANK_DISABLED = (RERANKER_MODEL.lower() in ("none", "off", "disabled") or
                    os.environ.get("RERANK_DISABLED", "0") == "1")

_reranker = None

def _get_reranker():
    global _reranker
    if RERANK_DISABLED:
        return None
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank_verses(query: str, verses: list[dict], top_k: int = 20) -> list[dict]:
    """
    Rerank a list of verse dicts by cross-encoder relevance to the query.
    Each verse dict must have a "text" key.
    Returns the top_k verses sorted by relevance, with relevance_score added.
    """
    if not verses or not query:
        return verses

    model = _get_reranker()
    if model is None:   # rerank explicitly disabled
        return verses[:top_k]

    pairs = [(query, v.get("text", "")) for v in verses]
    scores = model.predict(pairs)

    for v, s in zip(verses, scores):
        v["relevance_score"] = float(s)

    verses.sort(key=lambda v: v["relevance_score"], reverse=True)
    return verses[:top_k]


def assess_quality(verses: list[dict], threshold: float = 0.3) -> str:
    """
    Return 'good', 'marginal', or 'poor' based on top relevance scores.
    Assumes verses already have relevance_score from rerank_verses().
    """
    if not verses:
        return "poor"
    top_score = max(v.get("relevance_score", 0) for v in verses)
    if top_score >= threshold:
        return "good"
    elif top_score >= 0.1:
        return "marginal"
    return "poor"


def lost_in_middle_reorder(verses: list[dict]) -> list[dict]:
    """
    Reorder verses so the most relevant are at the START and END of the list.
    LLMs attend best to positions at the beginning and end of context.

    Strategy: alternate placing items at front and back.
    Input:  [1st, 2nd, 3rd, 4th, 5th, 6th]  (ranked by relevance)
    Output: [1st, 3rd, 5th, 6th, 4th, 2nd]   (best at edges, worst in middle)
    """
    if len(verses) <= 2:
        return verses

    front = []
    back = []
    for i, v in enumerate(verses):
        if i % 2 == 0:
            front.append(v)
        else:
            back.append(v)
    back.reverse()
    return front + back


def gate_tool_result(query: str, tool_name: str, result: dict) -> dict:
    """
    Post-process a tool result: rerank verses, assess quality, reorder.
    Returns the modified result dict with reranked verses and quality metadata.

    Only applies to tools that return verse lists (search_keyword, semantic_search,
    traverse_topic).
    """
    if not query:
        return result

    if "error" in result:
        return result

    reranked = False

    if tool_name == "search_keyword" and "by_surah" in result:
        # Flatten all verses, rerank, then re-group
        all_verses = []
        for surah_verses in result["by_surah"].values():
            all_verses.extend(surah_verses)

        if all_verses:
            all_verses = rerank_verses(query, all_verses, top_k=30)
            quality = assess_quality(all_verses)
            all_verses = lost_in_middle_reorder(all_verses)

            # Re-group by surah
            by_surah = {}
            for v in all_verses:
                vid = v.get("verse_id", "")
                try:
                    surah_num = vid.split(":")[0]
                except Exception:
                    surah_num = "?"
                # Use a simple key since we lost the original surah names
                key = f"Surah {surah_num}"
                by_surah.setdefault(key, []).append(v)

            result["by_surah"] = by_surah
            result["total_verses"] = len(all_verses)
            result["retrieval_quality"] = quality
            reranked = True

    elif tool_name == "semantic_search" and "by_surah" in result:
        all_verses = []
        for surah_verses in result["by_surah"].values():
            all_verses.extend(surah_verses)

        if all_verses:
            all_verses = rerank_verses(query, all_verses, top_k=30)
            quality = assess_quality(all_verses)
            all_verses = lost_in_middle_reorder(all_verses)

            by_surah = {}
            for v in all_verses:
                vid = v.get("verse_id", "")
                try:
                    surah_num = vid.split(":")[0]
                except Exception:
                    surah_num = "?"
                key = f"Surah {surah_num}"
                by_surah.setdefault(key, []).append(v)

            result["by_surah"] = by_surah
            result["total_verses"] = len(all_verses)
            result["retrieval_quality"] = quality
            reranked = True

    elif tool_name == "traverse_topic" and "direct_matches" in result:
        directs = result["direct_matches"]
        if directs:
            directs = rerank_verses(query, directs, top_k=30)
            quality = assess_quality(directs)
            result["direct_matches"] = lost_in_middle_reorder(directs)
            result["retrieval_quality"] = quality
            reranked = True

    if not reranked:
        result["retrieval_quality"] = "n/a"

    return result
