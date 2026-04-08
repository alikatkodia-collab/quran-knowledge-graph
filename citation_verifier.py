"""
Citation Verifier — Phase 3 of hallucination reduction.

Verifies that each citation in Claude's response actually supports the claim
it's attached to. Uses NLI cross-encoder for entailment checking.

Components:
  1. Claim decomposition — split response into atomic claims with citations
  2. NLI verification — check if cited verse entails the claim
  3. Response-level scoring — aggregate into precision/recall metrics
"""

import json
import re

from sentence_transformers import CrossEncoder

# ── NLI model (loaded once) ──────────────────────────────────────────────────

_nli_model = None

def _get_nli():
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-xsmall")
    return _nli_model


# ── claim decomposition ──────────────────────────────────────────────────────

_CITE_REF = re.compile(r'\[(\d+:\d+)\]')

def decompose_claims(text: str) -> list[dict]:
    """
    Split response text into atomic claims, each with its cited verse references.
    Uses regex-based extraction (no LLM call needed).

    Returns list of {"claim": str, "citations": list[str]}
    """
    claims = []
    # Split into sentences (rough but fast)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue  # skip short fragments

        refs = _CITE_REF.findall(sentence)
        if refs:
            # Remove the citation brackets to get the clean claim text
            clean = _CITE_REF.sub('', sentence).strip()
            clean = re.sub(r'\s{2,}', ' ', clean)
            if len(clean) > 15:
                claims.append({"claim": clean, "citations": list(set(refs))})

    return claims


# ── NLI verification ─────────────────────────────────────────────────────────

def verify_citation(claim: str, verse_text: str) -> dict:
    """
    Check if verse_text entails the claim using NLI.
    Returns {"label": "entailment"|"neutral"|"contradiction", "score": float}
    """
    model = _get_nli()
    # NLI format: (premise, hypothesis) — does the verse (premise) support the claim (hypothesis)?
    scores = model.predict([(verse_text, claim)])
    labels = ["contradiction", "entailment", "neutral"]
    idx = int(scores[0].argmax())
    return {"label": labels[idx], "score": float(scores[0].max())}


# ── full response verification ───────────────────────────────────────────────

def verify_response(text: str, verse_texts: dict) -> dict:
    """
    Verify all citations in a response.

    Args:
        text: Claude's full response text
        verse_texts: dict mapping verse_id -> {"text": str, "arabic": str}
                     (from _fetch_verses)

    Returns dict with:
        citation_precision: fraction of citations verified as entailment
        total_claims: number of claims found
        total_citations_checked: number of citation-claim pairs checked
        flagged: list of claims where citation doesn't support the claim
    """
    claims = decompose_claims(text)

    total_checked = 0
    supported = 0
    flagged = []

    for c in claims:
        for ref in c["citations"]:
            vdata = verse_texts.get(ref)
            if not vdata:
                continue

            vtext = vdata.get("text", "") if isinstance(vdata, dict) else str(vdata)
            if not vtext:
                continue

            result = verify_citation(c["claim"], vtext)
            total_checked += 1

            if result["label"] == "entailment":
                supported += 1
            else:
                flagged.append({
                    "claim": c["claim"][:120],
                    "ref": ref,
                    "nli_label": result["label"],
                    "nli_score": round(result["score"], 3),
                })

    precision = supported / total_checked if total_checked > 0 else 1.0

    return {
        "citation_precision": round(precision, 4),
        "total_claims": len(claims),
        "total_citations_checked": total_checked,
        "supported": supported,
        "flagged_count": len(flagged),
        "flagged": flagged[:5],  # limit to top 5 to avoid huge payloads
    }
