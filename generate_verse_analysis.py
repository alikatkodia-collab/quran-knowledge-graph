"""
Generate VerseAnalysis v2.1 metadata for a curated set of high-value verses.
Saves one JSON file per verse at data/verse_analysis/{verseId}.json.
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
from datetime import datetime
import requests
from neo4j import GraphDatabase


def load_env():
    for line in (Path(__file__).parent / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip().strip('"').strip("'")
load_env()

OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "openai/gpt-oss-120b:free"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "quran")

SYSTEM = (Path(__file__).parent / "prompts" / "verse_analysis" / "v2_1_system.txt").read_text(encoding="utf-8")
OUT_DIR = Path("data/verse_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 20 diverse high-value verses — iconic, legal, narrative, theological mix
# Each is heavily cited in our existing 500-entry answer cache.
VERSES = [
    "2:255",   # Ayat al-Kursi
    "24:35",   # Light verse
    "112:1",   # Ikhlas
    "2:286",   # last verse of Al-Baqarah
    "55:33",   # challenge to humans and jinn
    "2:177",   # true righteousness
    "2:185",   # fasting / Ramadan
    "2:173",   # food prohibitions
    "2:178",   # equivalence in retribution
    "73:20",   # night prayer accommodations
    "3:81",    # covenant of prophets
    "3:7",     # clear vs allegorical verses
    "17:9",    # Quran guides to righteousness
    "59:13",   # fear of believers vs fear of God
    "84:22",   # those who refuse to prostrate
    "68:17",   # parable of the garden owners
    "2:153",   # seek help through patience + prayer
    "3:146",   # patience of prophets and followers
    "22:32",   # reverence for God's rites
    "1:1",     # basmalah
]


def fetch_verse(session, verse_id):
    q = """
    MATCH (v:Verse {reference: $vid})
    OPTIONAL MATCH (v)-[:MENTIONS_ROOT]->(r:ArabicRoot)
    OPTIONAL MATCH (v)-[:MENTIONS]->(k:Keyword)
    RETURN v.reference AS vid, v.sura AS surah, v.number AS num,
           v.text AS text, v.arabicText AS arabicText, v.arabicPlain AS arabicPlain,
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


def call(verse_ctx):
    user = (
        "Produce the VerseAnalysis JSON for this verse. Emit JSON only:\n\n"
        + json.dumps(verse_ctx, ensure_ascii=False, indent=2)
    )
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
        },
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8085",
            "X-Title": "QKG VerseAnalysis generation",
        },
        timeout=180,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"] or ""
    return content


def extract_json(text):
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    driver.verify_connectivity()

    done, failed, skipped = 0, 0, 0

    with driver.session(database=NEO4J_DB) as session:
        for i, vid in enumerate(VERSES, 1):
            out = OUT_DIR / f"{vid.replace(':', '_')}.json"
            if out.exists():
                print(f"[{i:2}/20] {vid} already exists, skipping")
                skipped += 1
                continue
            ctx = fetch_verse(session, vid)
            if not ctx:
                print(f"[{i:2}/20] {vid} not found")
                failed += 1
                continue
            t0 = time.time()
            try:
                raw = call(ctx)
                parsed = extract_json(raw)
                if not parsed:
                    print(f"[{i:2}/20] {vid} unparseable, saving raw")
                    out.write_text(raw, encoding="utf-8")
                    failed += 1
                    continue
                out.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
                elapsed = time.time() - t0
                kb = out.stat().st_size / 1024
                print(f"[{i:2}/20] {vid} OK {elapsed:.0f}s {kb:.1f}KB")
                done += 1
            except Exception as e:
                print(f"[{i:2}/20] {vid} ERROR: {str(e)[:200]}")
                failed += 1
            time.sleep(1)

    driver.close()
    print(f"\nDone: {done}, skipped: {skipped}, failed: {failed}")
    print(f"Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
