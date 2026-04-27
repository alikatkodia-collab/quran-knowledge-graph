"""
Citation Verifier — Phase 3 of hallucination reduction.

Verifies that each citation in the response actually supports the claim
it's attached to. Two backends are supported via env var
`CITATION_VERIFIER_MODEL`:

  - "nli"        (default) cross-encoder/nli-deberta-v3-xsmall
                 — fast, good for short verse + short claim
  - "minicheck"  MiniCheck-Flan-T5-Large (Tang et al., EMNLP 2024)
                 — purpose-built for grounding-document fact-checking;
                 GPT-4-class faithfulness at ~400x lower cost than GPT-4

Output shape is identical for both backends so callers (app.py, app_free.py,
reasoning_memory.QueryRecorder.log_citation_checks) don't need to change.

Components:
  1. Claim decomposition — split response into atomic claims with citations
  2. Backend dispatch — NLI or MiniCheck
  3. Response-level scoring — aggregate into precision/recall metrics
"""

import json
import os
import re

from sentence_transformers import CrossEncoder

# ── backend selection ────────────────────────────────────────────────────────
# Read once at module load; can be overridden by CITATION_VERIFIER_MODEL env var
DEFAULT_BACKEND = os.environ.get("CITATION_VERIFIER_MODEL", "nli").strip().lower()
MINICHECK_MODEL_NAME = os.environ.get("MINICHECK_MODEL", "flan-t5-large").strip()
MINICHECK_THRESHOLD  = float(os.environ.get("MINICHECK_THRESHOLD", "0.5"))


# ── NLI model (loaded once) ──────────────────────────────────────────────────

_nli_model = None

def _get_nli():
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-xsmall")
    return _nli_model


# ── MiniCheck model (loaded once, lazily) ────────────────────────────────────

_minicheck_model = None

def _get_minicheck():
    global _minicheck_model
    if _minicheck_model is None:
        try:
            from minicheck.minicheck import MiniCheck
            _minicheck_model = MiniCheck(
                model_name=MINICHECK_MODEL_NAME,
                cache_dir="./ckpts/minicheck",
            )
        except ImportError as e:
            raise RuntimeError(
                "MiniCheck not installed. Run: "
                'pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"'
            ) from e
    return _minicheck_model


# ── claim cleanup helpers ────────────────────────────────────────────────────

# Framing prefixes that confuse MiniCheck — it treats them as meta-claims about
# the corpus rather than direct restatements. NLI handles them OK so this
# cleanup only fires when calling MiniCheck.
_FRAMING_PREFIX = re.compile(
    r"^(the quran (?:teaches|states|says|tells|reveals|declares|describes|emphasizes|notes|warns|reminds|asserts|affirms|explains|establishes) (?:that |us that |how |about )?|"
    r"according to (?:the )?quran[,:]?\s*|"
    r"this verse (?:teaches|tells us|reminds us|reveals|states|emphasizes|describes) (?:that |how |about )?|"
    r"the verse (?:teaches|tells us|reminds us|reveals|states|emphasizes|describes) (?:that |how |about )?)",
    re.IGNORECASE,
)

def _strip_framing(claim: str) -> str:
    """Strip 'The Quran teaches that...' style prefixes for MiniCheck only."""
    out = _FRAMING_PREFIX.sub("", claim).strip()
    if out:
        # Capitalize first letter
        out = out[0].upper() + out[1:]
    return out or claim


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

def verify_citation_nli(claim: str, verse_text: str) -> dict:
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


def verify_citation_minicheck(claim: str, verse_text: str) -> dict:
    """
    Check if verse_text supports the claim using MiniCheck-FT5.
    Returns the same shape as verify_citation_nli for caller compatibility.

    MiniCheck returns (pred_label, prob, ...) where pred_label = 1 means
    'supported', 0 = 'not supported'. We map this to NLI-compatible labels:
      label=1 (prob >= threshold) -> "entailment"
      label=0 (prob < threshold)  -> "neutral" (not "contradiction" — MiniCheck
                                                doesn't distinguish those two)
    """
    model = _get_minicheck()
    cleaned_claim = _strip_framing(claim)
    pred_label, raw_prob, _, _ = model.score(docs=[verse_text], claims=[cleaned_claim])
    prob = float(raw_prob[0]) if raw_prob else 0.0
    label = "entailment" if prob >= MINICHECK_THRESHOLD else "neutral"
    return {"label": label, "score": prob}


def verify_citation(claim: str, verse_text: str, backend: str = None) -> dict:
    """
    Dispatch to the active citation-verifier backend.
    backend can be passed explicitly or read from CITATION_VERIFIER_MODEL.
    """
    use = (backend or DEFAULT_BACKEND).lower()
    if use == "minicheck":
        return verify_citation_minicheck(claim, verse_text)
    return verify_citation_nli(claim, verse_text)


# ── full response verification ───────────────────────────────────────────────

def verify_response(text: str, verse_texts: dict, backend: str = None) -> dict:
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
    use = (backend or DEFAULT_BACKEND).lower()

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

            result = verify_citation(c["claim"], vtext, backend=use)
            total_checked += 1

            if result["label"] == "entailment":
                supported += 1
            else:
                flagged.append({
                    "claim": c["claim"][:120],
                    "ref": ref,
                    "nli_label": result["label"],   # kept name for back-compat
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
        "verifier_backend": use,
    }
