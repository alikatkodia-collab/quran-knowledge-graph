"""
Autoresearch Loop — Automated pipeline optimization.

Uses Optuna (Bayesian TPE) to search over pipeline_config.yaml parameters
and maximize the QIS composite score. Runs overnight unattended.

Usage:
    py autoresearch.py                    # run with defaults (50 trials)
    py autoresearch.py --trials 20        # limit to 20 trials
    py autoresearch.py --budget 50.00     # stop after ~$50 estimated spend
    py autoresearch.py --full             # use all 218 questions (expensive)

Each trial:
    1. Optuna proposes parameter values
    2. pipeline_config.yaml is rewritten
    3. evaluate.py runs on a 15-question subset
    4. Composite score is returned to Optuna
    5. Best config so far is saved to best_config.yaml

Cost: ~$2-3 per trial (15 questions). 50 trials ≈ $100-150.
Time: ~20-30 min per trial. 50 trials ≈ 17-25 hours.
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# ── setup ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "pipeline_config.yaml"
BEST_CONFIG_PATH = PROJECT_ROOT / "best_config.yaml"
LOG_PATH = PROJECT_ROOT / "autoresearch_log.jsonl"

# Core evaluation subset — stratified across difficulty and tool usage patterns
# 8 standard + 4 edge cases + 3 unanswerable = 15 questions
CORE_SUBSET = [
    # Standard — easy (keyword-heavy)
    "q01_forgiveness",      # broad topic, many keyword hits
    "q12_ablution",         # narrow topic, tests precision
    # Standard — medium (multi-tool)
    "q03_charity",          # keyword + semantic overlap
    "q05_nineteen",         # Khalifa-specific, semantic search needed
    "q06_marriage",         # keyword diversity test
    "q16_arabic_root_rhm",  # Arabic root tool test
    # Standard — hard (cross-surah, broad)
    "q09_justice",          # semantic search critical
    "q13_afterlife",        # very broad, many surahs
    "q14_women_rights",     # multi-keyword, sensitive topic
    "q10_previous_scriptures",  # keyword variety test
    # Edge cases
    "q08_abraham_path",     # find_path tool test
    "q15_messenger_covenant",  # Khalifa interpretation, contested
    # Unanswerable (tests abstention — should produce no citations)
    # These will be added once we confirm which IDs exist in the expanded set
]

# Cost estimation: ~$0.15-0.20 per question (Sonnet + tool calls)
COST_PER_QUESTION = 0.18


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def log_trial(trial_data):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(trial_data, ensure_ascii=False) + "\n")


# ── find unanswerable question IDs ──────────────────────────────────────────

def get_unanswerable_ids():
    """Find 3 unanswerable question IDs from the test dataset."""
    dataset_path = PROJECT_ROOT / "test_dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    unanswerable = [q["id"] for q in dataset if q.get("expected_citations") == []]
    # Pick 3 spread across the range
    if len(unanswerable) >= 3:
        step = len(unanswerable) // 3
        return [unanswerable[0], unanswerable[step], unanswerable[2 * step]]
    return unanswerable[:3]


# ── objective function ───────────────────────────────────────────────────────

def create_objective(subset_ids: str, original_config: dict):
    """Create an Optuna objective function that evaluates a config."""

    def objective(trial):
        cfg = copy.deepcopy(original_config)

        # ── Parameters to optimize ──────────────────────────────────────

        # LLM parameters (highest impact)
        cfg["llm"]["temperature"] = trial.suggest_float("temperature", 0.0, 1.0, step=0.1)
        cfg["llm"]["max_tokens"] = trial.suggest_categorical("max_tokens", [2048, 3072, 4096])

        # Retrieval parameters (high impact)
        cfg["retrieval"]["semantic_search"]["default_top_k"] = trial.suggest_int("top_k", 10, 60, step=5)
        cfg["retrieval"]["traverse_topic"]["seed_limit"] = trial.suggest_int("seed_limit", 10, 50, step=5)
        cfg["retrieval"]["traverse_topic"]["hop1_limit"] = trial.suggest_int("hop1_limit", 20, 80, step=10)
        cfg["retrieval"]["traverse_topic"]["hop2_limit"] = trial.suggest_int("hop2_limit", 10, 60, step=10)

        # Scoring thresholds (medium impact)
        cfg["scoring"]["related_to_min_score"] = trial.suggest_float("min_score", 0.05, 0.30, step=0.05)

        # get_verse limits
        cfg["retrieval"]["get_verse"]["neighbour_limit"] = trial.suggest_int("neighbour_limit", 6, 20, step=2)
        cfg["retrieval"]["get_verse"]["keyword_limit"] = trial.suggest_int("keyword_limit", 8, 20, step=2)

        # ── Write config and run eval ───────────────────────────────────

        save_config(cfg)

        # Force config module to reload
        try:
            import config as config_mod
            config_mod.reload()
        except Exception:
            pass

        # Run evaluate.py as subprocess (clean import state)
        import subprocess
        trial_out = PROJECT_ROOT / f"trial_{trial.number}.json"
        t0 = time.time()

        try:
            result = subprocess.run(
                [sys.executable, "-u", "evaluate.py",
                 "--ids", subset_ids,
                 "--out", str(trial_out)],
                capture_output=True, text=True, timeout=1800,  # 30 min max
                cwd=str(PROJECT_ROOT),
            )

            if result.returncode != 0:
                print(f"  Trial {trial.number} FAILED: {result.stderr[-200:]}")
                # Log failure and return worst score
                log_trial({
                    "trial": trial.number,
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "error": result.stderr[-500:],
                    "params": trial.params,
                })
                return 0.0

            # Parse result
            with open(trial_out, encoding="utf-8") as f:
                report = json.load(f)

            score = report["composite_score"]
            elapsed = time.time() - t0

            print(f"  Trial {trial.number}: QIS={score:.4f} ({elapsed:.0f}s)")
            print(f"    params: temp={trial.params.get('temperature')}, "
                  f"top_k={trial.params.get('top_k')}, "
                  f"max_tokens={trial.params.get('max_tokens')}, "
                  f"seed={trial.params.get('seed_limit')}, "
                  f"hop1={trial.params.get('hop1_limit')}")

            # Log success
            log_trial({
                "trial": trial.number,
                "timestamp": datetime.now().isoformat(),
                "status": "ok",
                "score": score,
                "elapsed_s": round(elapsed, 1),
                "params": trial.params,
                "aggregate": report.get("aggregate", {}),
            })

            # Clean up trial output
            try:
                trial_out.unlink()
            except Exception:
                pass

            return score

        except subprocess.TimeoutExpired:
            print(f"  Trial {trial.number} TIMEOUT (30 min)")
            log_trial({
                "trial": trial.number,
                "timestamp": datetime.now().isoformat(),
                "status": "timeout",
                "params": trial.params,
            })
            return 0.0
        except Exception as e:
            print(f"  Trial {trial.number} EXCEPTION: {e}")
            return 0.0

    return objective


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Autoresearch Loop — optimize pipeline config")
    parser.add_argument("--trials", type=int, default=50, help="Max number of trials (default 50)")
    parser.add_argument("--budget", type=float, default=150.0, help="Max estimated cost in USD (default $150)")
    parser.add_argument("--full", action="store_true", help="Use all 218 questions (expensive)")
    args = parser.parse_args()

    print("=" * 70)
    print("AUTORESEARCH LOOP — Quran Knowledge Graph Pipeline Optimizer")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Max trials: {args.trials}")
    print(f"Budget cap: ${args.budget:.2f}")
    print()

    # Save original config as backup
    original_config = load_config()
    backup_path = PROJECT_ROOT / "pipeline_config_backup.yaml"
    save_config_to = lambda cfg, path: yaml.dump(cfg, open(path, "w", encoding="utf-8"),
                                                  default_flow_style=False, allow_unicode=True,
                                                  sort_keys=False)
    save_config_to(original_config, backup_path)
    print(f"Original config backed up to {backup_path.name}")

    # Build subset IDs
    if args.full:
        # Use all questions
        with open(PROJECT_ROOT / "test_dataset.json", encoding="utf-8") as f:
            all_qs = json.load(f)
        subset_ids = ",".join(q["id"] for q in all_qs)
        n_questions = len(all_qs)
    else:
        # Use core subset + unanswerable
        unanswerable = get_unanswerable_ids()
        all_subset = CORE_SUBSET + unanswerable
        subset_ids = ",".join(all_subset)
        n_questions = len(all_subset)

    cost_per_trial = n_questions * COST_PER_QUESTION
    max_trials_by_budget = int(args.budget / cost_per_trial)
    effective_trials = min(args.trials, max_trials_by_budget)

    print(f"Questions per trial: {n_questions}")
    print(f"Estimated cost per trial: ${cost_per_trial:.2f}")
    print(f"Effective trials (budget-limited): {effective_trials}")
    print(f"Estimated total cost: ${effective_trials * cost_per_trial:.2f}")
    print(f"Estimated total time: {effective_trials * 25 / 60:.1f} hours")
    print(f"Subset IDs: {subset_ids[:80]}...")
    print("=" * 70)
    print()

    # Create Optuna study
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="quran-graph-optimization",
    )

    objective = create_objective(subset_ids, original_config)

    best_score = 0.0
    total_cost = 0.0

    # Custom callback to save best config and enforce budget
    def callback(study, trial):
        nonlocal best_score, total_cost

        total_cost += cost_per_trial

        if trial.value and trial.value > best_score:
            best_score = trial.value
            # Save best config
            best_cfg = copy.deepcopy(original_config)
            for key, val in trial.params.items():
                if key == "temperature":
                    best_cfg["llm"]["temperature"] = val
                elif key == "max_tokens":
                    best_cfg["llm"]["max_tokens"] = val
                elif key == "top_k":
                    best_cfg["retrieval"]["semantic_search"]["default_top_k"] = val
                elif key == "seed_limit":
                    best_cfg["retrieval"]["traverse_topic"]["seed_limit"] = val
                elif key == "hop1_limit":
                    best_cfg["retrieval"]["traverse_topic"]["hop1_limit"] = val
                elif key == "hop2_limit":
                    best_cfg["retrieval"]["traverse_topic"]["hop2_limit"] = val
                elif key == "min_score":
                    best_cfg["scoring"]["related_to_min_score"] = val
                elif key == "neighbour_limit":
                    best_cfg["retrieval"]["get_verse"]["neighbour_limit"] = val
                elif key == "keyword_limit":
                    best_cfg["retrieval"]["get_verse"]["keyword_limit"] = val

            save_config_to(best_cfg, BEST_CONFIG_PATH)
            print(f"\n  *** NEW BEST: QIS={best_score:.4f} (saved to {BEST_CONFIG_PATH.name}) ***\n")

        # Budget check
        if total_cost >= args.budget:
            print(f"\n  Budget cap reached (${total_cost:.2f} >= ${args.budget:.2f}). Stopping.")
            study.stop()

        # Progress summary every 5 trials
        if (trial.number + 1) % 5 == 0:
            print(f"\n--- Progress: {trial.number + 1} trials, "
                  f"best={best_score:.4f}, "
                  f"est. cost=${total_cost:.2f} ---\n")

    # Run optimization
    try:
        study.optimize(objective, n_trials=effective_trials, callbacks=[callback])
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving results...")

    # ── Final report ─────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("AUTORESEARCH COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"Trials completed: {len(study.trials)}")
    print(f"Estimated cost: ${total_cost:.2f}")
    print(f"\nBest QIS score: {study.best_value:.4f}")
    print(f"Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Restore best config as the active config
    if BEST_CONFIG_PATH.exists():
        import shutil
        shutil.copy2(BEST_CONFIG_PATH, CONFIG_PATH)
        print(f"\nBest config applied to {CONFIG_PATH.name}")
    else:
        # No improvement found — restore original
        save_config(original_config)
        print(f"\nNo improvement found. Original config restored.")

    print(f"\nFull trial log: {LOG_PATH.name}")
    print(f"Original config backup: {backup_path.name}")
    print(f"Best config: {BEST_CONFIG_PATH.name}")

    # Summary of top 5 trials
    print(f"\nTop 5 trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    for t in sorted_trials[:5]:
        print(f"  Trial {t.number}: QIS={t.value:.4f} — {t.params}")

    print("=" * 70)


if __name__ == "__main__":
    main()
