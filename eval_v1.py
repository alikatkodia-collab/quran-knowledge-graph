"""
eval_v1.py — Hand-curated end-to-end eval against live /chat.

Captures: which tools fired (and how many times each), citation count,
unique citations, elapsed time, char count, answer text. Saved to
data/eval_v1_results.json + a markdown summary.

Run: python eval_v1.py
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import requests

PORT = 8085
BASE = f"http://localhost:{PORT}"
OUT_JSON = Path("data/eval_v1_results.json")
OUT_MD = Path("data/eval_v1_results.md")

QUESTIONS_GENERAL = [
    "Tell me about paradise.",
    "Tell me about Sin",
    "Tell me about hell.",
    "What are some common themes in the Quran?",
    "Tell me about hypocrites.",
    "Tell me about charity.",
    "Tell me about meditation.",
    "Tell me about reverence.",
]

QUESTIONS_SURAH = [
    "Summarize Surah Al-Fatihah (1) and its main insights.",
    "Summarize Surah Al-Baqarah (2) — its central themes and key sections.",
    "Summarize Surah Yasin (36) and its main insights.",
    "Summarize Surah Ar-Rahman (55) and its main insights.",
    "Summarize Surah Al-Ikhlas (112) and its main insights.",
]


def ask(message, timeout=900):
    t0 = time.time()
    payload = {
        "message": message,
        "history": [],
        "deep_dive": False,
        "full_coverage": True,
    }
    try:
        r = requests.post(f"{BASE}/chat", json=payload, stream=True, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": str(e)[:300], "elapsed": time.time() - t0}

    full = ""
    tools_fired = []   # list of (tool_name, args_summary)
    fallback = False
    error = None
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
        except Exception:
            continue
        t = ev.get("t")
        if t == "tool":
            tools_fired.append({
                "name": ev.get("name", ""),
                "args": (ev.get("args") or "")[:120],
                "summary": (ev.get("summary") or "")[:120],
            })
            if ev.get("name") == "Fallback":
                fallback = True
        elif t == "text":
            full += ev.get("d", "")
        elif t == "error":
            error = ev.get("d", "")[:300]
            break
        elif t == "done":
            break

    cites = re.findall(r"\[(\d+):(\d+)\]", full)
    unique_cites = sorted(set(f"{s}:{v}" for s, v in cites))
    tool_names = [t["name"] for t in tools_fired]
    return {
        "ok": True if not error else False,
        "elapsed_sec": round(time.time() - t0, 1),
        "n_tool_calls": len(tools_fired),
        "tool_call_breakdown": dict(Counter(tool_names)),
        "tools_fired": tools_fired,
        "n_cites_total": len(cites),
        "n_cites_unique": len(unique_cites),
        "unique_cites": unique_cites,
        "answer_chars": len(full),
        "answer": full,
        "fallback": fallback,
        "error": error,
    }


def run_batch(questions, label):
    print(f"\n=== {label} ({len(questions)} questions) ===")
    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q}")
        r = ask(q)
        if not r.get("ok"):
            print(f"  ERROR: {r.get('error', 'unknown')}")
        else:
            print(f"  -> {r['elapsed_sec']}s · {r['n_tool_calls']} tools · "
                  f"{r['n_cites_unique']} unique cites · {r['answer_chars']}c")
            print(f"     tool calls: {r['tool_call_breakdown']}")
        results.append({"question": q, **r})
        # incremental save
        all_so_far = (results_general if label == "general" else []) + results if label == "surah" else results
    return results


def render_md(general, surah):
    md = ["# Eval v1 — Live `/chat` baseline\n",
          f"Captured {time.strftime('%Y-%m-%d %H:%M')}.",
          "Server flags: `SEMANTIC_SEARCH_INDEX=verse_embedding_m3 RERANKER_MODEL=BAAI/bge-reranker-v2-m3`\n"]

    def section(title, results):
        md.append(f"\n## {title}\n")
        # summary table
        md.append("| # | Question | Tools | Cites (uniq) | Chars | Time |")
        md.append("|---|----------|-------|--------------|-------|------|")
        for i, r in enumerate(results, 1):
            qshort = r["question"][:50]
            tools = sum(r.get("tool_call_breakdown", {}).values())
            md.append(f"| {i} | {qshort} | {tools} | {r.get('n_cites_unique',0)} | "
                      f"{r.get('answer_chars',0)} | {r.get('elapsed_sec','-')}s |")
        md.append("")
        md.append("### Tool-call breakdown (across all questions in this batch)")
        agg = Counter()
        for r in results:
            for tool, n in r.get("tool_call_breakdown", {}).items():
                agg[tool] += n
        md.append("| Tool | Calls |")
        md.append("|------|-------|")
        for tool, n in agg.most_common():
            md.append(f"| {tool} | {n} |")
        md.append("")
        md.append("### Per-question detail")
        for i, r in enumerate(results, 1):
            md.append(f"\n#### {i}. {r['question']}")
            md.append(f"- {r.get('elapsed_sec', 0)}s · "
                      f"{r.get('n_tool_calls', 0)} tool calls · "
                      f"{r.get('n_cites_unique', 0)} unique cites · "
                      f"{r.get('answer_chars', 0)} chars")
            tcb = r.get("tool_call_breakdown", {})
            if tcb:
                md.append(f"- tool calls: {tcb}")
            uc = r.get("unique_cites", [])
            if uc:
                md.append(f"- top citations: {uc[:15]}")
            ans = r.get("answer") or ""
            if ans:
                excerpt = ans[:1500].replace("\n", "\n  ")
                md.append(f"\n  ```\n  {excerpt}{'...' if len(ans) > 1500 else ''}\n  ```")
        md.append("")

    section("General questions", general)
    section("Whole-surah questions", surah)
    return "\n".join(md)


if __name__ == "__main__":
    # Verify server
    try:
        requests.get(BASE, timeout=5).raise_for_status()
    except Exception as e:
        print(f"FATAL: server not reachable: {e}")
        sys.exit(1)

    print("Eval v1 — running live /chat...")
    results_general = run_batch(QUESTIONS_GENERAL, "general")
    OUT_JSON.write_text(json.dumps({"general": results_general}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n  general saved to {OUT_JSON}")

    results_surah = run_batch(QUESTIONS_SURAH, "surah")
    OUT_JSON.write_text(json.dumps({"general": results_general, "surah": results_surah},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  surah saved to {OUT_JSON}")

    md = render_md(results_general, results_surah)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"\n  markdown report: {OUT_MD}")

    # Final headline
    n = len(results_general) + len(results_surah)
    total_cites = sum(r.get("n_cites_unique", 0) for r in results_general + results_surah)
    total_tools = sum(r.get("n_tool_calls", 0) for r in results_general + results_surah)
    elapsed_total = sum(r.get("elapsed_sec", 0) for r in results_general + results_surah)
    print(f"\n=== Final ===")
    print(f"  {n} questions · {total_tools} tool calls · "
          f"{total_cites} unique cites · {elapsed_total:.0f}s wall clock")
