#!/usr/bin/env python3
"""
Master runner — runs all AutoResearch processes end-to-end.

Usage:
    python autoresearch/run_all.py [--hours 8]

This script runs:
  1. Graph construction optimization (until converged)
  2. Retrieval pipeline optimization (until converged)
  3. Infinite deduction loop (for remaining time)

No human intervention needed. Commits and logs results automatically.
"""

import argparse
import json
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTORESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(name, args, timeout_seconds=None):
    """Run a Python script and stream output."""
    cmd = [sys.executable, os.path.join(AUTORESEARCH_DIR, name)] + args
    print(f"\n{'='*70}")
    print(f"  RUNNING: {name} {' '.join(args)}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            timeout=timeout_seconds,
            capture_output=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"\n  {name} timed out after {timeout_seconds}s")
        return True
    except Exception as e:
        print(f"\n  {name} failed: {e}")
        return False


def git_commit_and_push(message):
    """Commit all changes and push."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=BASE_DIR, capture_output=True, timeout=30)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=BASE_DIR, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "push", "-u", "origin", "HEAD"],
            cwd=BASE_DIR, capture_output=True, timeout=60,
        )
        print(f"  Committed and pushed: {message[:60]}...")
    except Exception as e:
        print(f"  Git error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run all AutoResearch processes")
    parser.add_argument("--hours", type=float, default=8,
                        help="Total hours to run (default: 8)")
    parser.add_argument("--skip-graph", action="store_true",
                        help="Skip graph optimization (already converged)")
    parser.add_argument("--skip-retrieval", action="store_true",
                        help="Skip retrieval optimization (already converged)")
    parser.add_argument("--deduction-only", action="store_true",
                        help="Only run the deduction loop")
    args = parser.parse_args()

    total_seconds = args.hours * 3600
    start = time.time()

    print("=" * 70)
    print("  QURAN KNOWLEDGE GRAPH — FULL AutoResearch PIPELINE")
    print("=" * 70)
    print(f"  Total time budget: {args.hours} hours")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Phase 1: Graph optimization (10% of time or skip if converged)
    if not args.skip_graph and not args.deduction_only:
        phase1_time = int(total_seconds * 0.10)
        print(f"\n  Phase 1: Graph optimization ({phase1_time//60} min budget)")
        run_script("loop.py", [
            "--max-experiments", "500",
            "--strategy", "mixed",
            "--no-commit",
        ], timeout_seconds=phase1_time)
        git_commit_and_push("autoresearch: graph optimization phase")

    # Phase 2: Retrieval optimization (5% of time or skip if converged)
    if not args.skip_retrieval and not args.deduction_only:
        phase2_time = int(total_seconds * 0.05)
        print(f"\n  Phase 2: Retrieval optimization ({phase2_time//60} min budget)")
        run_script("retrieval_loop.py", [
            "--max-experiments", "2000",
        ], timeout_seconds=phase2_time)
        git_commit_and_push("autoresearch: retrieval optimization phase")

    # Phase 3: Infinite deduction loop (remaining time)
    remaining = total_seconds - (time.time() - start)
    if remaining > 60:
        remaining_hours = remaining / 3600
        print(f"\n  Phase 3: Infinite deduction loop ({remaining_hours:.1f}h remaining)")
        run_script("infinite_deduction.py", [
            "--max-hours", str(round(remaining_hours, 2)),
        ], timeout_seconds=int(remaining) + 60)
        git_commit_and_push("autoresearch: deduction discovery phase")

    # Final summary
    total_time = time.time() - start
    print(f"\n{'='*70}")
    print(f"  ALL PHASES COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Print results summary
    for fname in ["best_params.json", "best_retrieval_params.json", "best_deductions.json"]:
        fpath = os.path.join(AUTORESEARCH_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            if isinstance(data, dict) and "score" in data:
                print(f"  {fname}: score = {data['score']}")
            elif isinstance(data, list):
                print(f"  {fname}: {len(data)} entries")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
