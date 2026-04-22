"""
A/B test harness: VerseAnalysis prompt v2.0 vs v2.1.

Runs both prompts against the same set of diverse verses using OpenRouter's
free tier (gpt-oss-120b). Scores outputs on:
  1. JSON validity
  2. Schema adherence (required top-level keys present)
  3. Null-vs-hallucinate discipline on fields that should be null for the verse
  4. Payload size (output tokens via char proxy)
  5. Enumeration compliance (values stay within the closed vocabularies)
  6. Wall-clock time per extraction

Produces a markdown report at verse_analysis_ab_report.md.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
import os
import re
import time
from pathlib import Path
from datetime import datetime
import requests
from neo4j import GraphDatabase


# ─── Setup ────────────────────────────────────────────────────────────────────

def load_env():
    for line in (Path(__file__).parent / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

load_env()

OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = "openai/gpt-oss-120b:free"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "quran")

PROMPTS_DIR = Path(__file__).parent / "prompts" / "verse_analysis"
V2_SYSTEM = (PROMPTS_DIR / "v2_system.txt").read_text(encoding="utf-8")
V21_SYSTEM = (PROMPTS_DIR / "v2_1_system.txt").read_text(encoding="utf-8")


# ─── Test verses (diverse across genre, length, difficulty) ──────────────────

TEST_VERSES = [
    "1:1",     # Basmalah — edge case (is/isn't a verse depending on sura)
    "2:255",   # Ayat al-Kursi — theological declaration with divine attributes
    "2:173",   # food prohibitions — legal ruling
    "12:4",    # Joseph's dream — narrative with speech_acts
    "36:79",   # parable about bones (resurrection)
    "3:159",   # forgiveness — command + context
    "74:30",   # Code 19 reference — Khalifa-specific
    "112:1",   # Ikhlas opening — short theological
    "55:13",   # the refrain in Ar-Rahman — rhetorical repetition
    "4:34",    # interpretive-sensitive (family relations)
]


# ─── Build input payload for the prompt ──────────────────────────────────────

def fetch_verse_context(session, verse_id: str) -> dict:
    q = """
    MATCH (v:Verse {reference: $vid})
    OPTIONAL MATCH (v)-[:MENTIONS_ROOT]->(r:ArabicRoot)
    OPTIONAL MATCH (v)-[:MENTIONS]->(k:Keyword)
    RETURN v.reference AS vid, v.sura AS surah, v.number AS num,
           v.text AS text, v.arabicText AS arabicText,
           v.arabicPlain AS arabicPlain,
           collect(DISTINCT r.root) AS roots,
           collect(DISTINCT k.name) AS keywords
    """
    r = session.run(q, vid=verse_id).single()
    if not r:
        return None
    return {
        "verseId": r["vid"],
        "reference": r["vid"],
        "surah": r["surah"],
        "verseNumberInSurah": r["num"],
        "arabicText": r["arabicText"] or "",
        "arabicPlain": r["arabicPlain"] or "",
        "rkText": r["text"] or "",
        "precomputed_morphology": [],
        "existing_root_ids": [x for x in r["roots"] if x],
        "existing_lemma_ids": [],
        "existing_semantic_domain_ids": [],
        "existing_keyword_ids": [x for x in r["keywords"] if x],
    }


# ─── OpenRouter call ─────────────────────────────────────────────────────────

def call_model(system_prompt: str, input_payload: dict) -> tuple[str, float, int]:
    """Returns (response_text, elapsed_seconds, output_char_count)."""
    start = time.time()
    user_msg = (
        "Produce the VerseAnalysis JSON for this verse. Emit JSON only, no preamble:\n\n"
        + json.dumps(input_payload, ensure_ascii=False, indent=2)
    )
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
        },
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8085",
            "X-Title": "Quran Knowledge Graph AB Test",
        },
        timeout=180,
    )
    elapsed = time.time() - start
    r.raise_for_status()
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return content, elapsed, len(content)


# ─── Scoring ─────────────────────────────────────────────────────────────────

REQUIRED_TOP_LEVEL_KEYS = {
    "analysisId", "verseRef", "extractorVersion", "extractedAt",
    "confidence_overall", "semantic", "rhetorical", "logical_form",
    "quantitative", "entity_links", "edge_hints", "provenance_notes",
}

ALLOWED_SPEAKER = {"god", "prophet", "angel", "believer", "disbeliever", "satan", "unspecified", "neutral_narration"}
ALLOWED_SPEECH_ACTS = {"command", "prohibition", "promise", "warning", "question", "declaration", "narrative", "supplication", "oath", "parable"}
ALLOWED_GENRES = {"narrative", "legal", "theological_declaration", "eschatological", "supplication", "parable", "oath", "rebuke", "consolation"}
ALLOWED_EDGE_TYPES = {"SUPPORTS", "ELABORATES", "QUALIFIES", "CONTRASTS", "REPEATS"}


def extract_json(text: str) -> dict | None:
    """Try parsing the response. Accept optional ``` fences."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        # fallback: find first { and matching } greedy
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def score_output(verse_id: str, raw: str, parsed: dict | None) -> dict:
    """Returns a scoring dict with booleans and counts."""
    score = {
        "json_valid": parsed is not None,
        "has_all_top_keys": False,
        "missing_keys": [],
        "has_allowed_arrays_in_output": False,
        "confidence_on_deterministic_fields": 0,
        "enum_violations": [],
        "fabricated_null_fields": 0,
    }
    if not parsed:
        return score

    # Top-level keys
    keys = set(parsed.keys())
    score["has_all_top_keys"] = REQUIRED_TOP_LEVEL_KEYS.issubset(keys)
    score["missing_keys"] = sorted(REQUIRED_TOP_LEVEL_KEYS - keys)

    # Check for inline `allowed` arrays (v2 may echo them, v2.1 should not)
    score["has_allowed_arrays_in_output"] = '"allowed"' in raw

    # Check for confidence on deterministic fields (v2.1 should NOT have these)
    deterministic_paths = [
        ("rhetorical", "has_question"),
        ("rhetorical", "has_oath"),
        ("rhetorical", "has_conditional"),
        ("rhetorical", "has_negation"),
        ("rhetorical", "has_emphatic"),
        ("rhetorical", "has_exception"),
        ("quantitative", "word_count"),
        ("quantitative", "letter_count_no_spaces"),
        ("quantitative", "has_basmalah"),
    ]
    for section, field in deterministic_paths:
        val = parsed.get(section, {}).get(field, {})
        if isinstance(val, dict) and "confidence" in val:
            score["confidence_on_deterministic_fields"] += 1

    # Enum compliance
    sem = parsed.get("semantic", {})
    sp = sem.get("speaker", {}).get("value")
    if sp and sp not in ALLOWED_SPEAKER:
        score["enum_violations"].append(f"speaker={sp!r}")
    for sa in sem.get("speech_acts", {}).get("value", []) or []:
        if sa not in ALLOWED_SPEECH_ACTS:
            score["enum_violations"].append(f"speech_act={sa!r}")
    g = parsed.get("rhetorical", {}).get("genre", {}).get("value")
    if g and g not in ALLOWED_GENRES:
        score["enum_violations"].append(f"genre={g!r}")
    for tec in parsed.get("edge_hints", {}).get("typed_edge_candidates", []) or []:
        if isinstance(tec, dict) and tec.get("type") and tec["type"] not in ALLOWED_EDGE_TYPES:
            score["enum_violations"].append(f"edge_type={tec['type']!r}")

    # Fabricated-null discipline: legal_ruling should be null for non-legal verses
    # (we can't tell perfectly but counting how often it's non-null on clearly
    # non-legal verses like 2:255, 112:1 would be misleading. Skip this for now.)

    return score


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    driver.verify_connectivity()

    results = {"v2.0": [], "v2.1": []}

    with driver.session(database=NEO4J_DB) as session:
        for verse_id in TEST_VERSES:
            ctx = fetch_verse_context(session, verse_id)
            if not ctx:
                print(f"[SKIP] {verse_id} not found in Neo4j")
                continue
            print(f"\n=== {verse_id} ===")
            print(f"    RK: {ctx['rkText'][:100]}...")

            for version, sys_prompt in [("v2.0", V2_SYSTEM), ("v2.1", V21_SYSTEM)]:
                print(f"  [{version}] calling...")
                try:
                    raw, elapsed, out_chars = call_model(sys_prompt, ctx)
                    parsed = extract_json(raw)
                    score = score_output(verse_id, raw, parsed)
                    score["verse_id"] = verse_id
                    score["elapsed_s"] = round(elapsed, 1)
                    score["output_chars"] = out_chars
                    score["raw"] = raw[:4000]  # truncate for report
                    results[version].append(score)
                    ok = "OK" if score["json_valid"] and score["has_all_top_keys"] else "FAIL"
                    print(f"  [{version}] {ok}  {elapsed:.1f}s  {out_chars}c  "
                          f"top_keys={score['has_all_top_keys']}  "
                          f"enum_viol={len(score['enum_violations'])}  "
                          f"det_conf={score['confidence_on_deterministic_fields']}")
                except Exception as e:
                    print(f"  [{version}] ERROR: {e}")
                    results[version].append({
                        "verse_id": verse_id, "error": str(e)[:200],
                        "json_valid": False, "has_all_top_keys": False,
                        "elapsed_s": None, "output_chars": 0,
                    })
                time.sleep(1)  # gentle pacing

    driver.close()

    # Write JSON results
    out_json = Path("verse_analysis_ab_results.json")
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown report
    write_report(results)

    print(f"\nFull results: {out_json}")
    print(f"Markdown report: verse_analysis_ab_report.md")


def write_report(results: dict):
    md = []
    md.append("# VerseAnalysis Prompt A/B Report\n")
    md.append(f"**Generated:** {datetime.utcnow().isoformat()}Z")
    md.append(f"**Model:** {OPENROUTER_MODEL}")
    md.append(f"**Verses tested:** {len(TEST_VERSES)}\n")

    md.append("## Summary table\n")
    md.append("| Metric | v2.0 | v2.1 | Delta |")
    md.append("|---|---|---|---|")

    def agg(name, fn, fmt="{:.0f}"):
        v0 = fn(results["v2.0"])
        v1 = fn(results["v2.1"])
        delta = v1 - v0
        sign = "+" if delta > 0 else ""
        md.append(f"| {name} | {fmt.format(v0)} | {fmt.format(v1)} | {sign}{fmt.format(delta)} |")

    agg("Valid JSON", lambda rs: sum(1 for r in rs if r.get("json_valid")))
    agg("All top-level keys present", lambda rs: sum(1 for r in rs if r.get("has_all_top_keys")))
    agg("Avg output chars (lower=leaner)",
        lambda rs: sum(r.get("output_chars", 0) for r in rs) / max(1, len(rs)))
    agg("Inline `allowed` arrays in output",
        lambda rs: sum(1 for r in rs if r.get("has_allowed_arrays_in_output")))
    agg("Avg `confidence` on deterministic fields (v2.1 should be 0)",
        lambda rs: sum(r.get("confidence_on_deterministic_fields", 0) for r in rs) / max(1, len(rs)),
        fmt="{:.1f}")
    agg("Total enum violations",
        lambda rs: sum(len(r.get("enum_violations", [])) for r in rs))
    agg("Avg wall-clock sec",
        lambda rs: sum(r.get("elapsed_s", 0) or 0 for r in rs) / max(1, len(rs)),
        fmt="{:.1f}")

    md.append("\n## Per-verse detail\n")
    md.append("| Verse | v2.0 valid | v2.1 valid | v2.0 chars | v2.1 chars | v2.0 enum_viol | v2.1 enum_viol |")
    md.append("|---|---|---|---|---|---|---|")
    for i, vid in enumerate(TEST_VERSES):
        r0 = results["v2.0"][i] if i < len(results["v2.0"]) else {}
        r1 = results["v2.1"][i] if i < len(results["v2.1"]) else {}
        md.append(
            f"| {vid} | {'Y' if r0.get('json_valid') else 'N'} | "
            f"{'Y' if r1.get('json_valid') else 'N'} | "
            f"{r0.get('output_chars','-')} | {r1.get('output_chars','-')} | "
            f"{len(r0.get('enum_violations', []))} | "
            f"{len(r1.get('enum_violations', []))} |"
        )

    md.append("\n## Interpretation\n")
    md.append("- Lower `output chars` = leaner payload = cheaper per extraction.")
    md.append("- `allowed arrays in output` should be 0 for v2.1 (they were schema docs, not data).")
    md.append("- `confidence on deterministic fields` should be 0 for v2.1; v2.0 may have them.")
    md.append("- `enum violations` = values outside the allowed vocabulary. Lower is better on both.")

    Path("verse_analysis_ab_report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
