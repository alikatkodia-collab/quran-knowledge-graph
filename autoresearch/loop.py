#!/usr/bin/env python3
"""
AutoResearch Loop for Quran Knowledge Graph Optimization.

Inspired by Karpathy's autoresearch: autonomously varies parameters in train.py,
rebuilds the graph, evaluates quality, keeps improvements, discards regressions.

Usage:
    python autoresearch/loop.py [--max-experiments 200] [--strategy random]

The loop modifies PARAMS in train.py, rebuilds to a temp dir, evaluates,
and if the composite score improves, overwrites data/ and commits.
"""

import argparse
import copy
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTORESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(AUTORESEARCH_DIR, "train.py")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_TSV = os.path.join(AUTORESEARCH_DIR, "results.tsv")
BEST_PARAMS_JSON = os.path.join(AUTORESEARCH_DIR, "best_params.json")

# Parameter search space: (min, max, type, step_size)
SEARCH_SPACE = {
    "min_df":              (1, 10, "int", 1),
    "max_df":              (50, 1000, "int", 25),
    "min_tfidf_score":     (0.01, 0.15, "float", 0.005),
    "max_features":        (5000, 100000, "int", 5000),
    "max_edges_per_verse": (4, 30, "int", 2),
    "max_verse_freq":      (50, 1000, "int", 25),
    "min_token_length":    (2, 4, "int", 1),
    "sublinear_tf":        (False, True, "bool", None),
    "ngram_max":           (1, 2, "int", 1),
    "norm":                (["l1", "l2"], None, "choice", None),
    "edge_weight_method":  (["geometric_mean", "harmonic_mean", "min"], None, "choice", None),
    "lemma_verb_first":    (False, True, "bool", None),
}


def read_current_params():
    """Read PARAMS dict from train.py by executing it."""
    # Import train module to get current params
    sys.path.insert(0, AUTORESEARCH_DIR)
    # Force reimport
    if "train" in sys.modules:
        del sys.modules["train"]
    import train
    return copy.deepcopy(train.PARAMS)


def write_params_to_train(params):
    """Write new PARAMS dict into train.py file."""
    with open(TRAIN_PY, "r", encoding="utf-8") as f:
        content = f.read()

    # Build the PARAMS string
    lines = ["PARAMS = {"]
    for key, val in params.items():
        if isinstance(val, str):
            lines.append(f'    "{key}": "{val}",')
        elif isinstance(val, bool):
            lines.append(f'    "{key}": {str(val)},')
        elif isinstance(val, list):
            lines.append(f'    "{key}": {json.dumps(val)},')
        elif isinstance(val, (int, float)):
            lines.append(f'    "{key}": {val},')
        else:
            lines.append(f'    "{key}": {repr(val)},')
    lines.append("}")
    new_params_block = "\n".join(lines)

    # Replace the PARAMS block using regex
    pattern = r"PARAMS = \{[^}]+\}"
    new_content = re.sub(pattern, new_params_block, content, count=1)

    with open(TRAIN_PY, "w", encoding="utf-8") as f:
        f.write(new_content)


def mutate_params(params, n_mutations=None):
    """Randomly mutate 1-3 parameters."""
    if n_mutations is None:
        n_mutations = random.choice([1, 1, 1, 2, 2, 3])

    new_params = copy.deepcopy(params)
    keys = list(SEARCH_SPACE.keys())
    chosen = random.sample(keys, min(n_mutations, len(keys)))

    mutations = []
    for key in chosen:
        spec = SEARCH_SPACE[key]
        old_val = new_params.get(key)

        if spec[2] == "int":
            lo, hi, _, step = spec
            # Perturb by +-step * random(1-3)
            delta = random.choice([-1, 1]) * step * random.randint(1, 3)
            new_val = max(lo, min(hi, old_val + delta))
        elif spec[2] == "float":
            lo, hi, _, step = spec
            delta = random.choice([-1, 1]) * step * random.uniform(0.5, 3.0)
            new_val = round(max(lo, min(hi, old_val + delta)), 4)
        elif spec[2] == "bool":
            new_val = not old_val
        elif spec[2] == "choice":
            options = spec[0]
            new_val = random.choice([o for o in options if o != old_val] or options)
        else:
            continue

        mutations.append(f"{key}: {old_val} -> {new_val}")
        new_params[key] = new_val

    return new_params, mutations


def mutate_params_focused(params, weak_metric):
    """Targeted mutations based on the weakest metric."""
    new_params = copy.deepcopy(params)
    mutations = []

    if weak_metric == "cluster_coherence":
        # More edges, lower thresholds to connect clusters
        choices = [
            ("max_edges_per_verse", lambda v: min(30, v + random.choice([2, 4]))),
            ("min_tfidf_score", lambda v: max(0.01, round(v - random.uniform(0.005, 0.01), 4))),
            ("max_verse_freq", lambda v: min(1000, v + random.choice([25, 50, 100]))),
            ("max_df", lambda v: min(1000, v + random.choice([25, 50, 100]))),
        ]
    elif weak_metric == "retrieval_recall":
        choices = [
            ("min_df", lambda v: max(1, v - 1)),
            ("min_tfidf_score", lambda v: max(0.01, round(v - random.uniform(0.005, 0.01), 4))),
            ("min_token_length", lambda v: max(2, v - 1)),
        ]
    elif weak_metric == "edge_density":
        choices = [
            ("max_edges_per_verse", lambda v: v + random.choice([-2, 2])),
            ("max_verse_freq", lambda v: v + random.choice([-25, 25])),
        ]
    elif weak_metric == "keyword_coverage":
        choices = [
            ("min_df", lambda v: max(1, v - 1)),
            ("max_df", lambda v: min(1000, v + random.choice([25, 50]))),
            ("min_tfidf_score", lambda v: max(0.01, round(v - 0.005, 4))),
        ]
    elif weak_metric == "vocabulary_efficiency":
        choices = [
            ("max_features", lambda v: v + random.choice([-5000, 5000])),
            ("max_df", lambda v: v + random.choice([-25, 25])),
            ("min_df", lambda v: max(1, v + random.choice([-1, 1]))),
        ]
    else:
        # Fallback: random mutation
        return mutate_params(params)

    # Apply 1-2 focused mutations
    n = random.choice([1, 2])
    selected = random.sample(choices, min(n, len(choices)))
    for key, mutator in selected:
        old_val = new_params[key]
        new_val = mutator(old_val)
        mutations.append(f"{key}: {old_val} -> {new_val}")
        new_params[key] = new_val

    return new_params, mutations


def run_build_and_evaluate(params):
    """Build graph with given params in a temp dir, evaluate, return results."""
    tmpdir = tempfile.mkdtemp(prefix="autoresearch_")
    try:
        # Write params to train.py
        write_params_to_train(params)

        # Force reimport
        if "train" in sys.modules:
            del sys.modules["train"]

        sys.path.insert(0, AUTORESEARCH_DIR)
        import train
        # Reload to pick up new params
        import importlib
        importlib.reload(train)

        # Build
        stats = train.build(output_dir=tmpdir)

        # Evaluate
        if "evaluate" in sys.modules:
            del sys.modules["evaluate"]
        from evaluate import evaluate, load_graph_csvs
        results = evaluate(data_dir=tmpdir)

        return {
            "success": True,
            "stats": stats,
            "scores": results,
            "tmpdir": tmpdir,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tmpdir": tmpdir,
        }


def copy_csvs_to_data(tmpdir):
    """Copy generated CSVs from temp dir to data/."""
    for fname in ["verse_nodes.csv", "keyword_nodes.csv", "verse_keyword_rels.csv", "verse_related_rels.csv"]:
        src = os.path.join(tmpdir, fname)
        dst = os.path.join(DATA_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)


def git_commit(experiment_id, score, mutations, params):
    """Commit the improvement."""
    msg = (
        f"autoresearch experiment #{experiment_id}: score {score:.2f}\n\n"
        f"Mutations: {'; '.join(mutations)}\n"
        f"Params: {json.dumps(params, indent=2)}"
    )
    try:
        subprocess.run(["git", "add", "-A"], cwd=BASE_DIR, capture_output=True, timeout=30)
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=BASE_DIR, capture_output=True, timeout=30
        )
    except Exception:
        pass


def git_reset():
    """Revert train.py changes (but not data/ since we use temp dirs)."""
    try:
        subprocess.run(
            ["git", "checkout", "--", "autoresearch/train.py"],
            cwd=BASE_DIR, capture_output=True, timeout=10
        )
    except Exception:
        pass


def log_result(experiment_id, score, best_score, mutations, params, elapsed, kept):
    """Append to results.tsv."""
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w") as f:
            f.write("experiment\ttimestamp\tscore\tbest_score\tkept\tmutations\telapsed_sec\tparams\n")

    with open(RESULTS_TSV, "a") as f:
        f.write(f"{experiment_id}\t{datetime.now().isoformat()}\t{score:.4f}\t{best_score:.4f}\t"
                f"{kept}\t{'; '.join(mutations)}\t{elapsed:.1f}\t{json.dumps(params)}\n")


def find_weakest_metric(scores):
    """Find the metric with the lowest weighted contribution."""
    from evaluate import WEIGHTS
    worst_name = None
    worst_contribution = float("inf")
    for name, weight in WEIGHTS.items():
        contribution = scores.get(name, 0) * weight
        if contribution < worst_contribution:
            worst_contribution = contribution
            worst_name = name
    return worst_name


def print_scores(scores, prefix=""):
    """Pretty print score breakdown."""
    for name, val in scores.items():
        if name != "composite_score":
            print(f"  {prefix}{name:30s} {val:7.2f}")
    print(f"  {prefix}{'COMPOSITE':30s} {scores['composite_score']:7.2f}")


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AutoResearch loop for Quran Knowledge Graph")
    parser.add_argument("--max-experiments", type=int, default=200,
                        help="Maximum number of experiments to run")
    parser.add_argument("--strategy", choices=["random", "focused", "mixed"], default="mixed",
                        help="Mutation strategy: random, focused on weakest metric, or mixed")
    parser.add_argument("--no-commit", action="store_true",
                        help="Don't git commit improvements")
    args = parser.parse_args()

    print("=" * 70)
    print("  QURAN KNOWLEDGE GRAPH — AutoResearch Optimization Loop")
    print("=" * 70)
    print(f"  Strategy:        {args.strategy}")
    print(f"  Max experiments: {args.max_experiments}")
    print(f"  Build time:      ~6s per experiment")
    print(f"  Estimated:       ~{args.max_experiments * 8 // 60} minutes total")
    print("=" * 70)

    # Get baseline score
    print("\n[baseline] Evaluating current graph...")
    current_params = read_current_params()

    sys.path.insert(0, AUTORESEARCH_DIR)
    from evaluate import evaluate
    baseline_scores = evaluate()
    best_score = baseline_scores["composite_score"]
    best_params = copy.deepcopy(current_params)

    print(f"\n[baseline] Score breakdown:")
    print_scores(baseline_scores)
    print(f"\n  BASELINE SCORE: {best_score:.4f}")

    improvements_kept = 0
    experiments_run = 0
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"  Starting optimization loop...")
    print(f"{'='*70}\n")

    for i in range(1, args.max_experiments + 1):
        experiments_run = i
        t0 = time.time()

        # Choose mutation strategy
        if args.strategy == "random":
            new_params, mutations = mutate_params(best_params)
        elif args.strategy == "focused":
            weak = find_weakest_metric(baseline_scores)
            new_params, mutations = mutate_params_focused(best_params, weak)
        else:  # mixed
            if random.random() < 0.6:
                weak = find_weakest_metric(baseline_scores)
                new_params, mutations = mutate_params_focused(best_params, weak)
            else:
                new_params, mutations = mutate_params(best_params)

        # Skip if params didn't actually change
        if new_params == best_params:
            new_params, mutations = mutate_params(best_params)

        print(f"[exp {i:3d}/{args.max_experiments}] Mutations: {', '.join(mutations)}")

        # Build and evaluate
        result = run_build_and_evaluate(new_params)

        if not result["success"]:
            elapsed = time.time() - t0
            print(f"  FAILED ({elapsed:.1f}s): {result.get('error', 'unknown')}")
            log_result(i, 0, best_score, mutations, new_params, elapsed, False)
            # Restore best params
            write_params_to_train(best_params)
            # Cleanup temp dir
            shutil.rmtree(result["tmpdir"], ignore_errors=True)
            continue

        new_score = result["scores"]["composite_score"]
        elapsed = time.time() - t0
        delta = new_score - best_score

        if new_score > best_score:
            # IMPROVEMENT! Keep it.
            improvements_kept += 1
            print(f"  IMPROVED! {best_score:.4f} -> {new_score:.4f} (+{delta:.4f}) [{elapsed:.1f}s]")
            print_scores(result["scores"], prefix="  ")

            best_score = new_score
            best_params = copy.deepcopy(new_params)
            baseline_scores = result["scores"]

            # Copy CSVs to data/
            copy_csvs_to_data(result["tmpdir"])

            # Save best params
            with open(BEST_PARAMS_JSON, "w") as f:
                json.dump({"score": best_score, "params": best_params, "experiment": i}, f, indent=2)

            # Git commit
            if not args.no_commit:
                git_commit(i, new_score, mutations, new_params)

            log_result(i, new_score, best_score, mutations, new_params, elapsed, True)
        else:
            print(f"  no improvement: {new_score:.4f} (best: {best_score:.4f}, delta: {delta:+.4f}) [{elapsed:.1f}s]")
            # Restore best params in train.py
            write_params_to_train(best_params)
            log_result(i, new_score, best_score, mutations, new_params, elapsed, False)

        # Cleanup temp dir
        shutil.rmtree(result["tmpdir"], ignore_errors=True)

        # Periodic summary
        if i % 10 == 0:
            total_time = time.time() - start_time
            rate = i / total_time * 3600
            print(f"\n  --- Progress: {i}/{args.max_experiments} experiments, "
                  f"{improvements_kept} improvements, "
                  f"best: {best_score:.4f}, "
                  f"rate: {rate:.0f} exp/hr ---\n")

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Experiments run:    {experiments_run}")
    print(f"  Improvements kept:  {improvements_kept}")
    print(f"  Starting score:     {baseline_scores['composite_score']:.4f}" if improvements_kept == 0
          else f"  Final score:        {best_score:.4f}")
    print(f"  Total time:         {total_time/60:.1f} minutes")
    print(f"  Rate:               {experiments_run/total_time*3600:.0f} experiments/hour")
    print(f"\n  Best params saved to: {BEST_PARAMS_JSON}")
    print(f"  Full results log:     {RESULTS_TSV}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
