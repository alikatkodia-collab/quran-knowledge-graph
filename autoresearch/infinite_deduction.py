#!/usr/bin/env python3
"""
Infinite Deduction Loop — AutoResearch for Syllogistic Insights.

Runs continuously, discovering novel deductions from the Quran Knowledge Graph.
Each round:
  1. Extract propositions from verses (with varying extraction strategies)
  2. Run syllogistic deduction engine with mutated parameters
  3. Score deductions for novelty
  4. Keep the best new deductions, append to cumulative log
  5. Mutate extraction/deduction parameters and repeat

The "metric" being optimized: number and quality of novel deductions found.
"""

import copy
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTORESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AUTORESEARCH_DIR)

from deduction_engine import (
    PropositionExtractor, SyllogisticEngine, NoveltyScorer,
    load_graph, Deduction
)

# Output files
DEDUCTIONS_LOG = os.path.join(AUTORESEARCH_DIR, "all_deductions.jsonl")
BEST_DEDUCTIONS = os.path.join(AUTORESEARCH_DIR, "best_deductions.json")
ROUND_LOG = os.path.join(AUTORESEARCH_DIR, "deduction_rounds.tsv")

# ══════════════════════════════════════════════════════════════════════════════
# Tunable deduction parameters
# ══════════════════════════════════════════════════════════════════════════════

DEDUCTION_PARAMS = {
    # Extraction
    "max_verses_per_round": 1000,       # how many verses to process per round
    "verse_sample_strategy": "random",  # "random", "by_surah", "high_connectivity"

    # Traversal depth
    "transitive_max_chains": 300,
    "shared_subject_max": 300,
    "thematic_bridge_max": 300,
    "max_neighbors_hop1": 10,
    "max_neighbors_hop2": 6,
    "max_neighbors_hop3": 4,

    # Novelty thresholds
    "min_novelty_to_keep": 60.0,  # minimum novelty score to save a deduction
    "min_bridge_keywords": 2,     # minimum bridge keywords for thematic bridges

    # Focus areas (surah ranges to emphasize)
    "focus_surahs": [],           # empty = all surahs
}


def mutate_deduction_params(params):
    """Mutate parameters for the next round."""
    new = copy.deepcopy(params)
    mutations = []

    # Vary sampling strategy
    if random.random() < 0.3:
        old = new["verse_sample_strategy"]
        new["verse_sample_strategy"] = random.choice(["random", "by_surah", "high_connectivity"])
        if new["verse_sample_strategy"] != old:
            mutations.append(f"strategy: {old} -> {new['verse_sample_strategy']}")

    # Vary verse count
    if random.random() < 0.3:
        old = new["max_verses_per_round"]
        new["max_verses_per_round"] = random.choice([500, 750, 1000, 1500, 2000, 3000, 6234])
        if new["max_verses_per_round"] != old:
            mutations.append(f"verses: {old} -> {new['max_verses_per_round']}")

    # Vary traversal limits
    if random.random() < 0.3:
        key = random.choice(["transitive_max_chains", "shared_subject_max", "thematic_bridge_max"])
        old = new[key]
        new[key] = random.choice([100, 200, 300, 500, 800])
        if new[key] != old:
            mutations.append(f"{key}: {old} -> {new[key]}")

    # Vary novelty threshold
    if random.random() < 0.2:
        old = new["min_novelty_to_keep"]
        new["min_novelty_to_keep"] = random.choice([40, 50, 60, 70, 80])
        if new["min_novelty_to_keep"] != old:
            mutations.append(f"min_novelty: {old} -> {new['min_novelty_to_keep']}")

    # Vary hop limits
    if random.random() < 0.3:
        key = random.choice(["max_neighbors_hop1", "max_neighbors_hop2", "max_neighbors_hop3"])
        old = new[key]
        new[key] = max(2, min(20, old + random.choice([-2, -1, 1, 2])))
        if new[key] != old:
            mutations.append(f"{key}: {old} -> {new[key]}")

    # Focus on specific surahs sometimes
    if random.random() < 0.2:
        if new["focus_surahs"]:
            new["focus_surahs"] = []
            mutations.append("focus: specific -> all")
        else:
            # Pick 5-20 random surahs to focus on
            n = random.randint(5, 20)
            new["focus_surahs"] = sorted(random.sample(range(1, 115), n))
            mutations.append(f"focus: all -> surahs {new['focus_surahs'][:5]}...")

    if not mutations:
        mutations.append("no changes")

    return new, mutations


def sample_verses(graph, params):
    """Select which verses to process this round."""
    all_ids = list(graph["verses"].keys())
    max_v = min(params["max_verses_per_round"], len(all_ids))

    strategy = params["verse_sample_strategy"]
    focus = params.get("focus_surahs", [])

    if focus:
        focus_set = set(str(s) for s in focus)
        all_ids = [vid for vid in all_ids if graph["verses"][vid].get("surah", "") in focus_set]
        max_v = min(max_v, len(all_ids))

    if strategy == "random":
        return random.sample(all_ids, max_v) if len(all_ids) > max_v else all_ids

    elif strategy == "by_surah":
        # Cycle through surahs, picking some from each
        by_surah = defaultdict(list)
        for vid in all_ids:
            s = graph["verses"][vid].get("surah", "0")
            by_surah[s].append(vid)
        result = []
        per_surah = max(1, max_v // len(by_surah))
        for s in sorted(by_surah.keys()):
            result.extend(random.sample(by_surah[s], min(per_surah, len(by_surah[s]))))
        return result[:max_v]

    elif strategy == "high_connectivity":
        # Prefer verses with many connections
        scored = [(vid, len(graph["related"].get(vid, []))) for vid in all_ids]
        scored.sort(key=lambda x: -x[1])
        return [vid for vid, _ in scored[:max_v]]

    return random.sample(all_ids, max_v)


def run_one_round(graph, extractor, params, seen_conclusions, round_num):
    """Run one round of deduction discovery."""
    t0 = time.time()

    # Sample verses
    verse_ids = sample_verses(graph, params)

    # Extract propositions
    all_props = []
    props_by_verse = {}
    for vid in verse_ids:
        text = graph["verses"][vid]["text"]
        props = extractor.extract(vid, text)
        if props:
            props_by_verse[vid] = props
            all_props.extend(props)

    if not all_props:
        return [], 0

    # Run deduction engine
    engine = SyllogisticEngine(graph, props_by_verse, all_props)

    deductions = []
    deductions.extend(engine.find_transitive_chains(max_chains=params["transitive_max_chains"]))
    deductions.extend(engine.find_shared_subject_syntheses(max_results=params["shared_subject_max"]))
    deductions.extend(engine.find_thematic_bridges(max_results=params["thematic_bridge_max"]))

    # Score novelty
    scorer = NoveltyScorer(graph, all_props)
    scored = scorer.score_all(deductions)

    # Filter by novelty threshold and dedup against previous rounds
    novel = []
    for d in scored:
        if d.novelty_score < params["min_novelty_to_keep"]:
            continue
        # Dedup: check if this conclusion is already known
        conclusion_key = (
            tuple(sorted(d.premise_verses)),
            d.rule,
        )
        if conclusion_key not in seen_conclusions:
            seen_conclusions.add(conclusion_key)
            novel.append(d)

    elapsed = time.time() - t0
    return novel, elapsed


def save_deductions(deductions, append=True):
    """Save deductions to JSONL log."""
    mode = "a" if append else "w"
    with open(DEDUCTIONS_LOG, mode, encoding="utf-8") as f:
        for d in deductions:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "novelty_score": d.novelty_score,
                "coherence_score": d.coherence_score,
                "rule": d.rule,
                "premise_verses": d.premise_verses,
                "bridge_keywords": d.bridge_keywords,
                "conclusion": d.conclusion,
                "graph_distance": d.graph_distance,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_best_deductions(all_best, new_deductions, max_keep=100):
    """Maintain a ranked list of the best deductions found."""
    for d in new_deductions:
        all_best.append({
            "novelty_score": d.novelty_score,
            "rule": d.rule,
            "premise_verses": d.premise_verses,
            "bridge_keywords": d.bridge_keywords,
            "conclusion": d.conclusion,
            "graph_distance": d.graph_distance,
        })
    all_best.sort(key=lambda x: -x["novelty_score"])
    return all_best[:max_keep]


# ══════════════════════════════════════════════════════════════════════════════
# Main Infinite Loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int, default=0,
                        help="Max rounds (0 = infinite)")
    parser.add_argument("--max-hours", type=float, default=0,
                        help="Max hours to run (0 = infinite)")
    args = parser.parse_args()

    print("=" * 70)
    print("  INFINITE DEDUCTION LOOP — Syllogistic AutoResearch")
    print("=" * 70)

    max_rounds = args.max_rounds or float("inf")
    max_seconds = args.max_hours * 3600 if args.max_hours > 0 else float("inf")

    print(f"  Rounds: {'infinite' if max_rounds == float('inf') else max_rounds}")
    print(f"  Time limit: {'none' if max_seconds == float('inf') else f'{args.max_hours}h'}")

    print("\nLoading graph...")
    graph = load_graph()
    print(f"  {len(graph['verses'])} verses, {len(graph['keyword_verses'])} keywords")

    print("Loading spaCy model...")
    extractor = PropositionExtractor()
    print("  Ready.")

    # Initialize
    params = copy.deepcopy(DEDUCTION_PARAMS)
    seen_conclusions = set()
    all_best = []
    total_deductions = 0
    total_novel = 0
    start_time = time.time()

    # Load existing best if any
    if os.path.exists(BEST_DEDUCTIONS):
        with open(BEST_DEDUCTIONS) as f:
            all_best = json.load(f)
        print(f"  Loaded {len(all_best)} previous best deductions")

    # Init round log
    if not os.path.exists(ROUND_LOG):
        with open(ROUND_LOG, "w") as f:
            f.write("round\ttimestamp\tnovel_found\ttotal_novel\telapsed\tstrategy\tmutations\n")

    print(f"\n{'='*70}")
    print(f"  Starting infinite deduction loop...")
    print(f"{'='*70}\n")

    round_num = 0
    while round_num < max_rounds:
        round_num += 1
        elapsed_total = time.time() - start_time

        if elapsed_total > max_seconds:
            print(f"\n  Time limit reached ({args.max_hours}h)")
            break

        # Mutate parameters
        params, mutations = mutate_deduction_params(params)

        # Run one round
        novel, elapsed = run_one_round(graph, extractor, params, seen_conclusions, round_num)
        total_novel += len(novel)

        # Save novel deductions
        if novel:
            save_deductions(novel)
            all_best = update_best_deductions(all_best, novel)
            with open(BEST_DEDUCTIONS, "w", encoding="utf-8") as f:
                json.dump(all_best, f, indent=2, ensure_ascii=False)

        # Log
        with open(ROUND_LOG, "a") as f:
            f.write(f"{round_num}\t{datetime.now().isoformat()}\t{len(novel)}\t{total_novel}\t"
                    f"{elapsed:.1f}\t{params['verse_sample_strategy']}\t{'; '.join(mutations)}\n")

        # Print progress
        if novel:
            print(f"[round {round_num:4d}] {len(novel):3d} novel deductions "
                  f"(total: {total_novel}) [{elapsed:.1f}s] "
                  f"mutations: {', '.join(mutations)}")
            # Print top deduction from this round
            best = max(novel, key=lambda d: d.novelty_score)
            print(f"  TOP: [{best.rule}] novelty={best.novelty_score:.1f}")
            print(f"       {best.conclusion[:150]}")
        elif round_num % 10 == 0:
            print(f"[round {round_num:4d}] 0 novel (total: {total_novel}) [{elapsed:.1f}s]")

        # Periodic summary
        if round_num % 25 == 0:
            hours = elapsed_total / 3600
            rate = total_novel / max(1, elapsed_total) * 3600
            print(f"\n  === PROGRESS: {round_num} rounds, {total_novel} novel deductions, "
                  f"{hours:.2f}h elapsed, {rate:.0f} deductions/hr ===\n")

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  DEDUCTION LOOP COMPLETE")
    print(f"{'='*70}")
    print(f"  Rounds:              {round_num}")
    print(f"  Novel deductions:    {total_novel}")
    print(f"  Time:                {total_time/60:.1f} minutes")
    print(f"  Rate:                {total_novel/max(1,total_time)*3600:.0f} deductions/hr")
    print(f"  Best deductions:     {BEST_DEDUCTIONS}")
    print(f"  Full log:            {DEDUCTIONS_LOG}")
    print(f"{'='*70}")

    # Print top 10
    if all_best:
        print(f"\n  TOP 10 MOST NOVEL DEDUCTIONS:")
        for i, d in enumerate(all_best[:10]):
            print(f"\n  #{i+1} [novelty: {d['novelty_score']:.1f}] [{d['rule']}]")
            print(f"     Premises: {d['premise_verses']}")
            print(f"     Bridge: {d['bridge_keywords'][:5]}")
            print(f"     {d['conclusion'][:200]}")


if __name__ == "__main__":
    main()
