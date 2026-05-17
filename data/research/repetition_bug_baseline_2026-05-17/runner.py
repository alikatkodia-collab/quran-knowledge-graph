"""
Bounded characterization: verbatim-explanation-duplicate bug on qwen3:14b.
N=50 thematic questions, single sequential pass against local app_free.py.

Subcommands:
  run      Run all 50 questions, save raw SSE per question to raw/q_NN.txt.
  analyze  Parse raw files, produce results.json.
  report   Read results.json, produce REPORT.md.
"""
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import requests

BASE = "http://localhost:8085"
OUT_DIR = Path(__file__).resolve().parent
RAW_DIR = OUT_DIR / "raw"
RUN_LOG = OUT_DIR / "run_log.jsonl"
RESULTS = OUT_DIR / "results.json"
REPORT = OUT_DIR / "REPORT.md"
EXPECTED_MODEL = "qwen3:14b"
PER_Q_TIMEOUT = 300       # 5 min
MAX_WALL_SEC = 4 * 3600   # 4 h
MAX_CONSEC_TIMEOUTS = 5
INTER_REQUEST_PAUSE = 5.0

ABSTRACT = [
    "What does the Quran say about charity?",
    "What does the Quran teach about forgiveness?",
    "What does the Quran say about gratitude?",
    "What does the Quran say about sin?",
    "What does the Quran say about patience?",
    "What does the Quran teach about humility?",
    "What does the Quran say about repentance?",
    "How does the Quran describe true belief?",
    "What does the Quran say about hypocrisy?",
    "What does the Quran teach about justice?",
    "What does the Quran say about the consequences of arrogance?",
    "How does the Quran describe the path of righteousness?",
    "What does the Quran teach about trusting in God?",
    "What does the Quran say about doubt and certainty?",
    "How does the Quran describe sincere worship?",
    "What does the Quran say about Satan and how he misleads people?",
    "What does the Quran teach about being led astray?",
    "What does the Quran say about the whisperings of Satan?",
    "How does the Quran describe the consequences of following Satan?",
    "What does the Quran say about wealth and worldly attachment?",
    "What does the Quran teach about death and the soul?",
    "What does the Quran say about reflection and contemplation?",
    "How does the Quran describe the mercy of God?",
    "What does the Quran say about prayer?",
    "What does the Quran teach about honoring parents?",
]
BROAD = [
    "What are the main themes of the Quran?",
    "How does the Quran describe itself?",
    "What is the Quran's view of revelation?",
    "How does the Quran describe the relationship between God and humanity?",
    "What does the Quran say about its own miraculous nature?",
    "How does the Quran describe the Day of Judgment?",
    "What is the Quran's view of other scriptures?",
    "What is the central message of Surah Ar-Rahman?",
    "What is the main theme of Surah Yasin?",
    "What is the message of Surah Al-Kahf?",
]
COMPLEMENTARY = [
    "What does the Quran say about fear of God?",
    "What does the Quran teach about kindness?",
    "How does the Quran describe the believers' relationship to each other?",
    "What does the Quran say about the unseen?",
    "What does the Quran say about envy and jealousy?",
    "What does the Quran teach about lying and truthfulness?",
    "How does the Quran describe paradise?",
    "How does the Quran describe hell?",
    "What does the Quran say about generosity?",
    "What does the Quran teach about peace?",
    "What does the Quran say about anger and self-control?",
    "What does the Quran say about wisdom?",
    "What does the Quran say about freedom and slavery?",
    "What does the Quran teach about justice in commerce?",
    "What does the Quran say about the resurrection?",
]

QUESTIONS = [(q, "ABSTRACT") for q in ABSTRACT] \
          + [(q, "BROAD") for q in BROAD] \
          + [(q, "COMPLEMENTARY") for q in COMPLEMENTARY]
assert len(QUESTIONS) == 50, len(QUESTIONS)


def stop(reason: str) -> None:
    sys.stderr.write(f"\n*** STOP: {reason}\n")
    sys.exit(2)


# ── runner ─────────────────────────────────────────────────────────────────

def run_one(idx: int, question: str) -> dict:
    """POST one /chat request and stream the raw SSE to disk. Returns metadata."""
    raw_path = RAW_DIR / f"q_{idx:03d}.txt"
    payload = {"message": question, "history": [], "local_only": True}
    t0 = time.time()
    status = "ok"
    model_served = None
    err = None
    first_event_seen = False
    try:
        r = requests.post(f"{BASE}/chat", json=payload, stream=True,
                          timeout=(10, PER_Q_TIMEOUT))
        r.raise_for_status()
        with open(raw_path, "w", encoding="utf-8") as f:
            for line in r.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                f.write(line + "\n")
                if line.startswith("data: ") and not first_event_seen:
                    try:
                        ev = json.loads(line[6:])
                    except Exception:
                        ev = {}
                    if ev.get("t") == "tool" and ev.get("name") == "Model":
                        model_served = ev.get("summary")
                        first_event_seen = True
                if time.time() - t0 > PER_Q_TIMEOUT:
                    status = "timeout"
                    break
    except requests.exceptions.Timeout:
        status = "timeout"
    except Exception as e:
        status = "error"
        err = str(e)[:300]

    elapsed = round(time.time() - t0, 1)
    size = raw_path.stat().st_size if raw_path.exists() else 0
    return {
        "q_id": f"q_{idx:03d}",
        "idx": idx,
        "question": question,
        "elapsed_sec": elapsed,
        "raw_size_bytes": size,
        "status": status,
        "model_served": model_served,
        "error": err,
    }


def cmd_run() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # check RALPH_STOP
    ralph = Path(__file__).resolve().parents[2] / "RALPH_STOP"
    if not ralph.exists():
        stop(f"RALPH_STOP missing at {ralph}")
    # check server up
    try:
        h = requests.get(BASE + "/", timeout=5)
        if h.status_code != 200:
            stop(f"Server returned {h.status_code} at {BASE}/")
    except Exception as e:
        stop(f"Server not reachable at {BASE}: {e}")

    log_f = open(RUN_LOG, "w", encoding="utf-8")
    wall_t0 = time.time()
    consec_timeouts = 0
    completed = 0
    for i, (q, cat) in enumerate(QUESTIONS, start=1):
        if time.time() - wall_t0 > MAX_WALL_SEC:
            sys.stderr.write(f"\n*** wall-time exceeded after {i-1} questions, stopping\n")
            break
        sys.stdout.write(f"[{i:02d}/50] ({cat}) {q[:70]}...\n")
        sys.stdout.flush()
        meta = run_one(i, q)
        meta["category"] = cat
        log_f.write(json.dumps(meta) + "\n")
        log_f.flush()
        sys.stdout.write(f"     -> {meta['status']} · {meta['elapsed_sec']}s · "
                         f"{meta['raw_size_bytes']}B · model={meta.get('model_served')}\n")
        sys.stdout.flush()
        if meta["status"] == "ok":
            completed += 1
            consec_timeouts = 0
        elif meta["status"] == "timeout":
            consec_timeouts += 1
            if consec_timeouts >= MAX_CONSEC_TIMEOUTS:
                sys.stderr.write("\n*** 5 consecutive timeouts — stopping\n")
                break
        else:
            consec_timeouts = 0
        # Model verification: skip empty/None (timeout/error before first frame)
        if meta.get("model_served") and meta["model_served"] != EXPECTED_MODEL:
            log_f.close()
            stop(f"wrong model: {meta['model_served']!r} != {EXPECTED_MODEL!r}")
        time.sleep(INTER_REQUEST_PAUSE)
    log_f.close()
    sys.stdout.write(f"\nRun complete. {completed} ok / {len(QUESTIONS)} attempted in "
                     f"{round(time.time()-wall_t0,1)}s.\n")


# ── analyze ────────────────────────────────────────────────────────────────

def parse_sse(text: str) -> dict:
    """Parse raw SSE file: return {answer, tools, model_served, error}."""
    answer = []
    tools = []
    model_served = None
    error = None
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
        except Exception:
            continue
        t = ev.get("t")
        if t == "text":
            answer.append(ev.get("d", ""))
        elif t == "tool":
            tools.append({"name": ev.get("name", ""),
                          "args": (ev.get("args") or "")[:200],
                          "summary": (ev.get("summary") or "")[:200]})
            if ev.get("name") == "Model" and model_served is None:
                model_served = ev.get("summary")
        elif t == "error":
            error = (ev.get("d") or "")[:300]
    return {
        "answer": "".join(answer),
        "tools": tools,
        "model_served": model_served,
        "error": error,
    }


_VERSE_BLOCK_RE = re.compile(
    r"\*\*\[(\d+):(\d+)\]\*\*[:\s]*(.+?)(?=\n\s*[-*]\s*\*\*\[|\n\s*\n|\n##\s|\Z)",
    re.DOTALL,
)
_H2_RE = re.compile(r"(?m)^##\s+(.+)$")
_WS_RE = re.compile(r"\s+")


def _norm(s: str, n: int = 150) -> str:
    return _WS_RE.sub(" ", s).strip()[:n]


def find_h2_sections(answer: str) -> list[tuple[str, int, int]]:
    """Return [(heading, start, end), ...] for each H2-bounded section body."""
    hits = list(_H2_RE.finditer(answer))
    sections = []
    for i, m in enumerate(hits):
        body_start = m.end()
        body_end = hits[i + 1].start() if i + 1 < len(hits) else len(answer)
        sections.append((m.group(1).strip(), body_start, body_end))
    return sections


def extract_explanations(answer: str) -> dict[str, list[tuple[int, str]]]:
    """verse_id -> [(section_idx, explanation_text<=300c), ...]."""
    sections = find_h2_sections(answer)
    if not sections:
        # Treat whole answer as one section (section 0).
        sections = [("(no-h2)", 0, len(answer))]
    out: dict[str, list[tuple[int, str]]] = {}
    for sec_idx, (_, start, end) in enumerate(sections):
        body = answer[start:end]
        for m in _VERSE_BLOCK_RE.finditer(body):
            vid = f"{m.group(1)}:{m.group(2)}"
            exp = m.group(3)[:300]
            out.setdefault(vid, []).append((sec_idx, exp))
    return out


def classify_pair(e1: str, e2: str) -> str:
    a = _norm(e1, 150)
    b = _norm(e2, 150)
    if not a or not b:
        return "skip"
    if a == b:
        return "verbatim"
    if SequenceMatcher(None, a, b).ratio() >= 0.80:
        return "near_verbatim"
    return "distinct"


def analyze_answer(answer: str) -> dict:
    sections = find_h2_sections(answer)
    explanations = extract_explanations(answer)
    duplicate_pairs = []
    saw_verbatim = False
    saw_near = False
    for vid, hits in explanations.items():
        # only consider when verse appears in >=2 distinct sections
        sec_ids = {s for s, _ in hits}
        if len(sec_ids) < 2:
            continue
        # pairwise within unique-section subset (keep first hit per section)
        per_section: dict[int, str] = {}
        for s, e in hits:
            per_section.setdefault(s, e)
        items = sorted(per_section.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                s1, e1 = items[i]
                s2, e2 = items[j]
                verdict = classify_pair(e1, e2)
                if verdict in ("verbatim", "near_verbatim"):
                    duplicate_pairs.append({
                        "verse": vid, "sections": [s1, s2],
                        "match": verdict,
                    })
                    if verdict == "verbatim":
                        saw_verbatim = True
                    else:
                        saw_near = True
    if saw_verbatim:
        verdict = "BUG_VERBATIM"
    elif saw_near:
        verdict = "BUG_NEAR"
    else:
        verdict = "CLEAN"
    return {
        "h2_count": len(sections),
        "answer_length_chars": len(answer),
        "verdict": verdict,
        "duplicate_pairs": duplicate_pairs,
    }


def cmd_analyze() -> None:
    # Re-read run_log.jsonl for the metadata (status/elapsed/category).
    if not RUN_LOG.exists():
        stop(f"missing {RUN_LOG}")
    meta_by_idx = {}
    with open(RUN_LOG, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            meta_by_idx[m["idx"]] = m

    results = []
    for i, (q, cat) in enumerate(QUESTIONS, start=1):
        meta = meta_by_idx.get(i, {})
        raw_path = RAW_DIR / f"q_{i:03d}.txt"
        entry = {
            "q_id": f"q_{i:03d}",
            "question": q,
            "category": cat,
            "elapsed_seconds": meta.get("elapsed_sec"),
            "raw_file": f"raw/q_{i:03d}.txt",
            "status": meta.get("status", "missing"),
            "model_served": meta.get("model_served"),
            "tool_call_count": 0,
            "answer_length_chars": 0,
            "h2_count": 0,
            "verdict": "MISSING",
            "duplicate_pairs": [],
        }
        if not raw_path.exists() or meta.get("status") == "error":
            entry["verdict"] = "ERROR" if meta.get("status") == "error" else "MISSING"
            entry["error"] = meta.get("error")
            results.append(entry)
            continue
        parsed = parse_sse(raw_path.read_text(encoding="utf-8", errors="replace"))
        entry["model_served"] = parsed["model_served"] or meta.get("model_served")
        entry["tool_call_count"] = sum(1 for t in parsed["tools"]
                                       if t["name"] not in ("Model",))
        ana = analyze_answer(parsed["answer"])
        entry["h2_count"] = ana["h2_count"]
        entry["answer_length_chars"] = ana["answer_length_chars"]
        if meta.get("status") == "timeout":
            entry["verdict"] = "TIMEOUT"
        elif parsed["error"]:
            entry["verdict"] = "ERROR"
            entry["error"] = parsed["error"]
        elif entry["answer_length_chars"] == 0:
            entry["verdict"] = "ERROR"
            entry["error"] = "empty answer"
        else:
            entry["verdict"] = ana["verdict"]
            entry["duplicate_pairs"] = ana["duplicate_pairs"]
        results.append(entry)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sys.stdout.write(f"Wrote {RESULTS}\n")


# ── report ─────────────────────────────────────────────────────────────────

def _quartiles(vals: list[int]) -> list[tuple[int, int]]:
    if not vals:
        return []
    s = sorted(vals)
    n = len(s)
    q1 = s[n // 4]
    q2 = s[n // 2]
    q3 = s[(3 * n) // 4]
    return [(0, q1), (q1, q2), (q2, q3), (q3, max(s) + 1)]


def cmd_report() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    n_total = len(results)
    n_ok = sum(1 for r in results if r["verdict"] in ("CLEAN", "BUG_NEAR", "BUG_VERBATIM"))
    n_timeout = sum(1 for r in results if r["verdict"] == "TIMEOUT")
    n_error = sum(1 for r in results if r["verdict"] == "ERROR")
    n_missing = sum(1 for r in results if r["verdict"] == "MISSING")
    verdict_ct = Counter(r["verdict"] for r in results)
    n_bug_v = verdict_ct["BUG_VERBATIM"]
    n_bug_n = verdict_ct["BUG_NEAR"]
    bug_v_rate = (n_bug_v / n_ok * 100.0) if n_ok else 0.0
    bug_comb_rate = ((n_bug_v + n_bug_n) / n_ok * 100.0) if n_ok else 0.0

    md = []
    md.append("# Repetition-Bug Baseline — qwen3:14b, N=50\n")
    md.append(f"Captured {time.strftime('%Y-%m-%d %H:%M')}.")
    md.append("Server: `app_free.py --model qwen3:14b` + `SEMANTIC_SEARCH_INDEX=verse_embedding_m3`, "
              "`RERANKER_MODEL=BAAI/bge-reranker-v2-m3`. `/chat` payload: `{local_only: true, history: []}`.\n")
    md.append("## Headline")
    md.append("")
    md.append(f"- **Verbatim bug rate:** {n_bug_v}/{n_ok} = **{bug_v_rate:.1f}%** of completed answers")
    md.append(f"- **Combined (verbatim + near):** {n_bug_v + n_bug_n}/{n_ok} = **{bug_comb_rate:.1f}%**")
    md.append("")
    md.append("## Run summary")
    md.append("")
    md.append(f"- Total questions attempted: {n_total}")
    md.append(f"- Completed (CLEAN + BUG_*): {n_ok}")
    md.append(f"- Timed out: {n_timeout}")
    md.append(f"- Errored: {n_error}")
    md.append(f"- Missing (not run): {n_missing}")
    elapsed_ok = [r["elapsed_seconds"] for r in results
                  if r["verdict"] in ("CLEAN", "BUG_NEAR", "BUG_VERBATIM") and r["elapsed_seconds"]]
    if elapsed_ok:
        md.append(f"- Median elapsed time per completed question: {statistics.median(elapsed_ok):.0f}s")
        md.append(f"- Total wall-time over completed questions: {sum(elapsed_ok):.0f}s "
                  f"({sum(elapsed_ok)/60.0:.1f} min)")

    md.append("\n## Verdict counts\n")
    md.append("| verdict | count |")
    md.append("|---------|------:|")
    for k in ("CLEAN", "BUG_NEAR", "BUG_VERBATIM", "TIMEOUT", "ERROR", "MISSING"):
        md.append(f"| {k} | {verdict_ct.get(k, 0)} |")

    md.append("\n## By category\n")
    md.append("| category | n | CLEAN | BUG_NEAR | BUG_VERBATIM | verbatim % | combined % |")
    md.append("|----------|--:|------:|---------:|-------------:|-----------:|-----------:|")
    for cat in ("ABSTRACT", "BROAD", "COMPLEMENTARY"):
        rs = [r for r in results if r["category"] == cat
              and r["verdict"] in ("CLEAN", "BUG_NEAR", "BUG_VERBATIM")]
        n = len(rs)
        c = sum(1 for r in rs if r["verdict"] == "CLEAN")
        bn = sum(1 for r in rs if r["verdict"] == "BUG_NEAR")
        bv = sum(1 for r in rs if r["verdict"] == "BUG_VERBATIM")
        vr = (bv / n * 100.0) if n else 0.0
        cr = ((bv + bn) / n * 100.0) if n else 0.0
        md.append(f"| {cat} | {n} | {c} | {bn} | {bv} | {vr:.1f}% | {cr:.1f}% |")

    md.append("\n## By answer-length quartile\n")
    ok_rs = [r for r in results if r["verdict"] in ("CLEAN", "BUG_NEAR", "BUG_VERBATIM")]
    lens = [r["answer_length_chars"] for r in ok_rs]
    qbounds = _quartiles(lens)
    md.append("| quartile | char-range | n | BUG_VERBATIM | verbatim % |")
    md.append("|----------|-----------:|--:|-------------:|-----------:|")
    for qi, (lo, hi) in enumerate(qbounds, start=1):
        bucket = [r for r in ok_rs if lo <= r["answer_length_chars"] < hi]
        bv = sum(1 for r in bucket if r["verdict"] == "BUG_VERBATIM")
        n = len(bucket)
        rate = (bv / n * 100.0) if n else 0.0
        md.append(f"| Q{qi} | {lo}-{hi-1} | {n} | {bv} | {rate:.1f}% |")

    md.append("\n## By tool-call count\n")
    tc = [r["tool_call_count"] for r in ok_rs]
    if tc:
        tbins = [(0, 3), (3, 5), (5, 7), (7, 100)]
        md.append("| tool-calls | n | BUG_VERBATIM | verbatim % |")
        md.append("|------------|--:|-------------:|-----------:|")
        for lo, hi in tbins:
            bucket = [r for r in ok_rs if lo <= r["tool_call_count"] < hi]
            bv = sum(1 for r in bucket if r["verdict"] == "BUG_VERBATIM")
            n = len(bucket)
            rate = (bv / n * 100.0) if n else 0.0
            label = f"{lo}-{hi-1}" if hi < 100 else f"{lo}+"
            md.append(f"| {label} | {n} | {bv} | {rate:.1f}% |")

    md.append("\n## Top 10 duplicated verses across corpus\n")
    verse_ct: Counter = Counter()
    for r in results:
        for dp in r.get("duplicate_pairs", []):
            verse_ct[dp["verse"]] += 1
    md.append("| verse | duplicate-pair count |")
    md.append("|-------|--------------------:|")
    for vid, n in verse_ct.most_common(10):
        md.append(f"| {vid} | {n} |")

    md.append("\n## Per-question table\n")
    md.append("| # | category | verdict | h2 | chars | tools | elapsed (s) | question |")
    md.append("|--:|----------|---------|---:|------:|------:|------------:|----------|")
    for r in results:
        idx = int(r["q_id"].split("_")[1])
        md.append(f"| {idx} | {r['category']} | {r['verdict']} | {r['h2_count']} | "
                  f"{r['answer_length_chars']} | {r['tool_call_count']} | "
                  f"{r['elapsed_seconds']} | {r['question'][:60]} |")

    md.append("\n## Plain-English summary\n")
    md.append(
        f"On {n_ok} qwen3:14b completions, the verbatim-explanation-duplicate "
        f"bug fired in **{bug_v_rate:.1f}%** of answers. Including near-verbatim, "
        f"the combined rate is **{bug_comb_rate:.1f}%**. See the breakdowns above "
        f"for correlations with category, answer length, and tool-call depth."
    )

    REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.write(f"Wrote {REPORT}\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": cmd_run, "analyze": cmd_analyze, "report": cmd_report}[cmd]()
