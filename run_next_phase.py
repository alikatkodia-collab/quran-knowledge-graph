"""
Auto-phase runner.

Finds the highest numbered existing Phase N+1 file (overnight_seed_phaseN.py).
If the next phase has a handwritten bank, run it. Otherwise, ask OpenRouter
to generate 150 fresh questions (avoiding overlap with the current cache and
with earlier phase banks), write it to a new phase file, and run it.

Usage:
  python run_next_phase.py --port 8085 --hours 4
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests

import overnight_seed as engine


ROOT = Path(__file__).parent
PHASE_FILE_RE = re.compile(r"^overnight_seed_phase(\d+)\.py$")


def _load_env():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def find_phase_files() -> dict[int, Path]:
    out = {}
    for p in ROOT.glob("overnight_seed_phase*.py"):
        m = PHASE_FILE_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def load_cache_questions(cache_path: str = "data/answer_cache.json") -> set[str]:
    try:
        cache = json.loads((ROOT / cache_path).read_text(encoding="utf-8"))
        return {e.get("question", "").strip().lower() for e in cache}
    except Exception:
        return set()


def load_phase_bank_questions(phase_file: Path) -> list[str]:
    """Extract PHASEN_RAW list contents from a phase file."""
    text = phase_file.read_text(encoding="utf-8")
    # crude eval-free extraction
    m = re.search(r"PHASE\d+_RAW\s*=\s*\[(.*?)\n\]", text, re.S)
    if not m:
        return []
    body = m.group(1)
    out = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith('"') and line.endswith('",'):
            out.append(line[1:-2])
        elif line.startswith('"') and line.endswith('"'):
            out.append(line[1:-1])
    return out


def generate_bank_via_openrouter(phase_num: int, existing: set[str],
                                   avoid_samples: list[str], count: int = 150) -> list[str]:
    """Ask OpenRouter gpt-oss-120b to generate `count` fresh Quran questions."""
    _load_env()
    key = os.environ["OPENROUTER_API_KEY"]

    # Sample 40 existing questions to show the model what we already have
    sample = random.sample(list(existing), min(40, len(existing)))
    avoid_ex = random.sample(avoid_samples, min(40, len(avoid_samples))) if avoid_samples else []

    system = (
        "You are a research assistant helping seed an answer cache for an AI "
        "Quran-exploration app that uses Rashad Khalifa's translation. Your users "
        "are Submitters (followers of the Messenger of the Covenant). Generate fresh, "
        "high-quality Quran exploration questions that will teach the AI something new. "
        "Each question should be specific enough to yield strong citations from a graph "
        "of 6,234 verses. Avoid duplicates or near-duplicates of what is already cached. "
        "Output ONE question per line, no numbering, no markdown, no preamble."
    )
    user = (
        f"Generate {count} fresh questions for Phase {phase_num}. They must be DIFFERENT "
        f"(semantically and lexically) from the examples below.\n\n"
        f"=== Already cached (sample — AVOID these) ===\n"
        + "\n".join(f"- {q}" for q in sample)
        + "\n\n=== Earlier phase banks (sample — AVOID these too) ===\n"
        + "\n".join(f"- {q}" for q in avoid_ex[:40])
        + "\n\n=== Rules ===\n"
        "1. One question per line.\n"
        "2. Under 120 characters each.\n"
        "3. No numbering, no markdown, no bullets.\n"
        "4. Mix specific-verse, cross-verse thematic, character studies, Arabic-word, "
        "practice/application, eschatology, rhetoric, and comparative questions.\n"
        "5. Favor questions that haven't been asked of this dataset before.\n"
        f"6. Produce exactly {count} lines.\n"
    )

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json={
            "model": "openai/gpt-oss-120b:free",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.9,
            "max_tokens": 8000,
        },
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8085",
            "X-Title": "QKG phase-bank generator",
        },
        timeout=180,
    )
    r.raise_for_status()
    body = r.json()["choices"][0]["message"]["content"] or ""

    # parse: one line per question, strip bullets/numbering/quotes
    lines = []
    for raw in body.splitlines():
        ln = raw.strip()
        ln = re.sub(r"^[\-*\d+.)\s]+", "", ln)  # strip bullets/numbering
        ln = ln.strip().strip('"').strip("'")
        if ln and len(ln) > 10 and len(ln) < 200 and "?" in ln:
            lines.append(ln)

    # dedupe and filter against cache/existing banks
    seen_lc = {q.lower() for q in existing} | {q.lower() for q in avoid_samples}
    out = []
    for q in lines:
        if q.lower() not in seen_lc:
            out.append(q)
            seen_lc.add(q.lower())
    return out


def write_phase_file(phase_num: int, questions: list[str]) -> Path:
    path = ROOT / f"overnight_seed_phase{phase_num}.py"
    body = [
        f'"""',
        f'Phase {phase_num} seeding — auto-generated question bank ({len(questions)} questions).',
        f'Generated at {time.strftime("%Y-%m-%d %H:%M:%S")} via OpenRouter gpt-oss-120b.',
        f'"""',
        "import sys",
        "try:",
        '    sys.stdout.reconfigure(encoding="utf-8", errors="replace")',
        "except Exception:",
        "    pass",
        "",
        "import json",
        "from pathlib import Path",
        "import overnight_seed as engine",
        "",
        "",
        f"PHASE{phase_num}_RAW = [",
    ]
    for q in questions:
        body.append(f'    "{q.replace(chr(34), chr(39))}",')
    body.extend([
        "]",
        "",
        "",
        "def filter_new(questions, cache_path=\"data/answer_cache.json\"):",
        "    try:",
        "        cache = json.loads(Path(cache_path).read_text(encoding=\"utf-8\"))",
        "        seen = {e.get(\"question\", \"\").strip().lower() for e in cache}",
        "    except Exception:",
        "        seen = set()",
        "    return [q for q in questions if q.strip().lower() not in seen]",
        "",
        "",
        "if __name__ == \"__main__\":",
        f"    fresh = filter_new(PHASE{phase_num}_RAW)",
        f"    print(f\"[phase{phase_num}] total={{len(PHASE{phase_num}_RAW)}}, after dedup={{len(fresh)}}\")",
        "    sf = Path(\"overnight_seed.state.json\")",
        "    if sf.exists():",
        "        try:",
        "            st = json.loads(sf.read_text(encoding=\"utf-8\"))",
        "            st[\"done\"] = [q for q in st.get(\"done\", []) if q in fresh]",
        "            st[\"failed\"] = []",
        "            sf.write_text(json.dumps(st, indent=2), encoding=\"utf-8\")",
        f"            print(f\"[phase{phase_num}] state pruned to {{len(st['done'])}} matching done entries\")",
        "        except Exception as e:",
        f"            print(f\"[phase{phase_num}] state prune failed: {{e}}\")",
        "    engine.QUESTIONS = fresh",
        "    engine.main()",
        "",
    ])
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--phase", type=int, default=None,
                    help="Explicitly run this phase (skip auto-discovery)")
    ap.add_argument("--generate-count", type=int, default=150)
    args = ap.parse_args()

    phases = find_phase_files()
    if args.phase is not None:
        phase_num = args.phase
    else:
        phase_num = (max(phases.keys()) if phases else 5) + 1

    if phase_num in phases:
        print(f"[run_next] phase {phase_num} file already exists: {phases[phase_num].name}")
        phase_file = phases[phase_num]
    else:
        print(f"[run_next] generating phase {phase_num} bank via OpenRouter ...")
        existing = load_cache_questions()
        # seed avoid_samples from earlier phase banks
        avoid_samples = []
        for n in sorted(phases.keys()):
            avoid_samples.extend(load_phase_bank_questions(phases[n]))
        qs = generate_bank_via_openrouter(
            phase_num, existing, avoid_samples, count=args.generate_count
        )
        print(f"[run_next] generated {len(qs)} fresh questions (after dedup)")
        if not qs:
            print("[run_next] FATAL — no questions generated, aborting")
            sys.exit(1)
        phase_file = write_phase_file(phase_num, qs)
        print(f"[run_next] wrote {phase_file.name}")

    # Run it
    sys.argv = ["phase", "--port", str(args.port), "--hours", str(args.hours)]
    # import the file and call its main
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"phase{phase_num}", phase_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


if __name__ == "__main__":
    main()
