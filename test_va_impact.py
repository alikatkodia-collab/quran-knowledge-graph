"""
VerseAnalysis impact test.

Picks 10 cached questions that heavily cite our 20 enriched verses, then
re-runs each question in two modes:
  A) Baseline  — current system, no VerseAnalysis awareness
  B) Enriched  — system prompt includes VerseAnalysis JSON for the key verses

Compares answers on citation count, unique citations, answer length, and
a qualitative readout.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import json
import re
import time
from pathlib import Path
import requests


def load_env():
    for line in (Path(__file__).parent / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip().strip('"').strip("'")
load_env()

OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "openai/gpt-oss-120b:free"

VA_DIR = Path("data/verse_analysis")
ENRICHED_VERSES = {p.stem.replace("_", ":") for p in VA_DIR.glob("*.json")}
print(f"[setup] {len(ENRICHED_VERSES)} enriched verses loaded")

# Pick 10 cached questions that cite >=3 of our enriched verses
cache = json.loads(Path("data/answer_cache.json").read_text(encoding="utf-8"))
candidates = []
for e in cache:
    ans = e.get("answer", "")
    cited = {f"{s}:{v}" for s, v in re.findall(r"\[(\d+):(\d+)\]", ans)}
    overlap = cited & ENRICHED_VERSES
    if len(overlap) >= 3:
        candidates.append({
            "question": e["question"],
            "cached_answer": ans,
            "cited_verses": sorted(cited),
            "overlap": sorted(overlap),
            "overlap_count": len(overlap),
        })

candidates.sort(key=lambda x: -x["overlap_count"])
test_questions = candidates[:10]
print(f"[setup] {len(candidates)} candidate questions; picking top 10")
for i, q in enumerate(test_questions, 1):
    print(f"  {i:2}. {q['question'][:70]:70} (overlap: {q['overlap_count']})")

if len(test_questions) < 5:
    print("[fatal] not enough overlap — need more enriched verses")
    sys.exit(1)


def load_va_for(verse_ids):
    """Load VerseAnalysis JSONs for given verse IDs; return a compact digest."""
    chunks = []
    for vid in verse_ids:
        p = VA_DIR / f"{vid.replace(':', '_')}.json"
        if not p.exists():
            continue
        try:
            va = json.loads(p.read_text(encoding="utf-8"))
            # Compact digest — strip verbose fields, keep what's useful for synthesis
            compact = {
                "verse": vid,
                "speaker": va.get("semantic", {}).get("speaker", {}).get("value"),
                "addressee": va.get("semantic", {}).get("addressee", {}).get("value"),
                "speech_acts": va.get("semantic", {}).get("speech_acts", {}).get("value"),
                "divine_attributes": va.get("semantic", {}).get("divine_attributes_invoked", {}).get("value"),
                "eschatological": va.get("semantic", {}).get("eschatological", {}).get("value"),
                "legal_ruling": va.get("semantic", {}).get("legal_ruling", {}).get("value"),
                "narrative_thread": va.get("semantic", {}).get("narrative_thread", {}).get("value"),
                "genre": va.get("rhetorical", {}).get("genre", {}).get("value"),
                "rhetorical_devices": va.get("rhetorical", {}).get("rhetorical_devices", {}).get("value"),
                "negations": va.get("logical_form", {}).get("negations", {}).get("value"),
                "exceptions": va.get("logical_form", {}).get("exceptions", {}).get("value"),
                "modals": va.get("logical_form", {}).get("modals", {}).get("value"),
                "candidate_predicates": va.get("logical_form", {}).get("candidate_predicates", {}).get("value"),
                "named_entities": va.get("semantic", {}).get("named_entities", {}),
                "semantic_domains": va.get("entity_links", {}).get("semantic_domains", {}).get("value"),
                "typed_edge_candidates": va.get("edge_hints", {}).get("typed_edge_candidates", []),
            }
            chunks.append(compact)
        except Exception as e:
            print(f"  [va-load] {vid} failed: {e}")
    return chunks


SYSTEM_BASELINE = (
    "You are a Quran scholar answering questions about Rashad Khalifa's translation of the Quran. "
    "Cite verses inline as [surah:verse]. Organise your answer into thematic sections with ## headers. "
    "Include at least 10 verse citations. The user wants the full picture, organised thematically."
)

SYSTEM_ENRICHED_TMPL = (
    "You are a Quran scholar answering questions about Rashad Khalifa's translation of the Quran. "
    "Cite verses inline as [surah:verse]. Organise your answer into thematic sections with ## headers. "
    "Include at least 10 verse citations. The user wants the full picture, organised thematically.\n\n"
    "=== VerseAnalysis metadata for central verses ===\n"
    "The following verses in your knowledge base have structured semantic/logical/rhetorical metadata. "
    "Use this to enrich your answer with precise observations about speaker, addressee, speech acts, "
    "divine attributes invoked, logical structure, genre, and typed relationships. Cite these verses "
    "when relevant and lean on the metadata for depth.\n\n"
    "{va_dump}\n"
    "=== end VerseAnalysis metadata ===\n"
)


def call_model(system, user_question, max_tokens=4096):
    t0 = time.time()
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_question},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        },
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8085",
            "X-Title": "QKG VA impact test",
        },
        timeout=240,
    )
    elapsed = time.time() - t0
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"] or ""
    return content, elapsed


def score(answer: str):
    cites = re.findall(r"\[(\d+):(\d+)\]", answer)
    unique = set(cites)
    return {
        "total_citations": len(cites),
        "unique_citations": len(unique),
        "chars": len(answer),
        "headers": answer.count("##"),
    }


def main():
    results = []
    for i, q in enumerate(test_questions, 1):
        qtext = q["question"]
        enriched_to_include = q["overlap"][:10]  # cap to avoid bloating prompt
        va_chunks = load_va_for(enriched_to_include)
        va_dump = "\n---\n".join(
            json.dumps(c, indent=1, ensure_ascii=False) for c in va_chunks
        )
        enriched_system = SYSTEM_ENRICHED_TMPL.format(va_dump=va_dump)

        print(f"\n[{i}/10] {qtext}")
        print(f"  overlap: {', '.join(enriched_to_include[:8])}")

        # Baseline
        print(f"  [baseline] calling...")
        try:
            a0, t0 = call_model(SYSTEM_BASELINE, qtext)
            s0 = score(a0)
            print(f"  [baseline] {t0:.0f}s · {s0['unique_citations']}u cites · {s0['chars']}c · {s0['headers']} headers")
        except Exception as e:
            print(f"  [baseline] ERROR: {e}")
            a0, t0, s0 = "", 0, {}

        time.sleep(2)

        # Enriched
        print(f"  [enriched] calling...")
        try:
            a1, t1 = call_model(enriched_system, qtext)
            s1 = score(a1)
            print(f"  [enriched] {t1:.0f}s · {s1['unique_citations']}u cites · {s1['chars']}c · {s1['headers']} headers")
        except Exception as e:
            print(f"  [enriched] ERROR: {e}")
            a1, t1, s1 = "", 0, {}

        # Cached (for reference)
        sc = score(q["cached_answer"])

        results.append({
            "question": qtext,
            "overlap_verses": enriched_to_include,
            "cached": {"answer": q["cached_answer"], **sc},
            "baseline": {"answer": a0, "elapsed": round(t0, 1), **s0},
            "enriched": {"answer": a1, "elapsed": round(t1, 1), **s1},
        })
        time.sleep(2)

    out = Path("va_impact_results.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRaw results: {out}")

    # Summary report
    report = []
    report.append("# VerseAnalysis Impact A/B Report\n")
    report.append(f"**Method:** 10 cached questions, each cites >=3 of our 20 enriched verses. "
                  f"Re-answered by gpt-oss-120b in two modes:")
    report.append(f"- **Baseline:** plain system prompt, no VerseAnalysis access")
    report.append(f"- **Enriched:** same system prompt + VerseAnalysis JSON for the relevant verses injected as context\n")

    report.append("## Summary\n")
    report.append("| Metric | Cached | Baseline | Enriched |")
    report.append("|---|---|---|---|")
    for field in ["unique_citations", "total_citations", "chars", "headers"]:
        c = sum(r["cached"].get(field, 0) for r in results) / len(results)
        b = sum(r["baseline"].get(field, 0) for r in results) / len(results)
        e = sum(r["enriched"].get(field, 0) for r in results) / len(results)
        report.append(f"| avg {field} | {c:.1f} | {b:.1f} | {e:.1f} |")
    avg_t = sum(r["baseline"].get("elapsed", 0) for r in results) / len(results)
    avg_t_e = sum(r["enriched"].get("elapsed", 0) for r in results) / len(results)
    report.append(f"| avg wall-clock | — | {avg_t:.1f}s | {avg_t_e:.1f}s |\n")

    report.append("## Per-question breakdown\n")
    report.append("| # | Question | Cached cites | Baseline cites | Enriched cites | Enriched - Baseline |")
    report.append("|---|---|---|---|---|---|")
    for i, r in enumerate(results, 1):
        qshort = r["question"][:60]
        cc = r["cached"].get("unique_citations", 0)
        bc = r["baseline"].get("unique_citations", 0)
        ec = r["enriched"].get("unique_citations", 0)
        delta = ec - bc
        sign = "+" if delta > 0 else ""
        report.append(f"| {i} | {qshort}... | {cc} | {bc} | {ec} | {sign}{delta} |")

    Path("va_impact_report.md").write_text("\n".join(report), encoding="utf-8")
    print("Report: va_impact_report.md")


if __name__ == "__main__":
    main()
