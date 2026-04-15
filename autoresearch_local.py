"""
Autoresearch Loop (Local) — Free overnight optimization using Ollama.

Uses a local LLM (Qwen 2.5 14B via Ollama) instead of Claude API.
The eval is simplified since local models don't support Anthropic tool_use:
  - Instead of the full agentic loop, we test retrieval quality directly
  - Query Neo4j with each tool, feed context to the local LLM, check citations
  - Much faster per trial (~5 min vs ~25 min) but lower eval fidelity

Usage:
    py autoresearch_local.py                # run with defaults
    py autoresearch_local.py --trials 100   # run 100 trials overnight
    py autoresearch_local.py --model qwen2.5:14b-instruct-q6_K

This is FREE — all inference runs locally on your machine.
"""

import argparse
import copy
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "pipeline_config.yaml"
BEST_CONFIG_PATH = PROJECT_ROOT / "best_config_local.yaml"
LOG_PATH = PROJECT_ROOT / "autoresearch_local_log.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"

# ── config helpers ───────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

def log_trial(data):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

# ── Ollama chat ──────────────────────────────────────────────────────────────

def ollama_chat(model: str, system: str, user: str, temperature: float = 0.3) -> str:
    """Send a chat request to Ollama and return the response text."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 2048},
        }, timeout=180)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"

# ── simplified eval (no tool_use, direct retrieval) ──────────────────────────

_BRACKET_REF = re.compile(r'\[(\d+:\d+)\]')
_ST_MODEL = None

def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL

def eval_question_local(question: str, expected_citations: set, model: str,
                        session, system_prompt: str) -> dict:
    """
    Evaluate one question using local LLM + direct Neo4j retrieval.

    1. Run keyword search + semantic search in Neo4j
    2. Build context from retrieved verses
    3. Send to local LLM with system prompt
    4. Score the response (citation recall, precision, grounding)
    """
    from chat import tool_search_keyword, tool_semantic_search
    from build_graph import tokenize_and_lemmatize
    import config as cfg

    # Step 1: Retrieve context using the current config's parameters
    # Extract keywords from question
    words = question.lower().split()
    important_words = [w for w in words if len(w) > 3 and w not in
                       {'what', 'does', 'how', 'that', 'this', 'with', 'from',
                        'about', 'which', 'where', 'when', 'they', 'their',
                        'there', 'have', 'been', 'were', 'will', 'would',
                        'could', 'should', 'quran', 'says', 'say', 'describe',
                        'described', 'tell'}]

    retrieved_verses = {}

    # Keyword search for top 2 keywords
    for kw in important_words[:2]:
        try:
            result = tool_search_keyword(session, kw)
            if "by_surah" in result:
                for verses in result["by_surah"].values():
                    for v in verses[:5]:
                        vid = v.get("verse_id", "")
                        if vid:
                            retrieved_verses[vid] = v.get("text", "")
        except Exception:
            pass

    # Semantic search
    try:
        result = tool_semantic_search(session, question, top_k=cfg.semantic_default_top_k())
        if "by_surah" in result:
            for verses in result["by_surah"].values():
                for v in verses[:5]:
                    vid = v.get("verse_id", "")
                    if vid:
                        retrieved_verses[vid] = v.get("text", "")
    except Exception:
        pass

    # Step 2: Build context
    context_lines = []
    for vid, text in list(retrieved_verses.items())[:30]:
        context_lines.append(f"[{vid}]: {text}")
    context = "\n".join(context_lines)

    # Step 3: Send to local LLM
    user_prompt = (
        f"RETRIEVED VERSES:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using ONLY the verses above. Cite every verse you reference "
        f"using [surah:verse] format (e.g. [2:255]). If the verses don't address "
        f"the question, say so."
    )

    answer = ollama_chat(model, system_prompt, user_prompt)

    # Step 4: Score
    produced = set(_BRACKET_REF.findall(answer))

    if expected_citations:
        recall = len(produced & expected_citations) / len(expected_citations)
        precision = len(produced & expected_citations) / len(produced) if produced else 0.0
    else:
        # Unanswerable — good if no citations produced
        recall = 1.0 if not produced else 0.0
        precision = 1.0 if not produced else 0.0

    paragraphs = [p for p in answer.split('\n\n') if len(p.strip()) > 50]
    cited_paras = sum(1 for p in paragraphs if _BRACKET_REF.search(p))
    grounding = cited_paras / len(paragraphs) if paragraphs else 0.0

    # Answer relevance via embedding similarity (cached model)
    try:
        import numpy as np
        st_model = _get_st_model()
        vecs = st_model.encode([question, answer], normalize_embeddings=True)
        relevance = float(np.dot(vecs[0], vecs[1]))
    except Exception:
        relevance = 0.0

    return {
        "citation_recall": recall,
        "citation_precision": precision,
        "grounding_rate": grounding,
        "answer_relevance": relevance,
        "citations_produced": len(produced),
        "citations_expected": len(expected_citations),
        "answer_length": len(answer),
    }


# ── subset ───────────────────────────────────────────────────────────────────

CORE_SUBSET_IDS = [
    "q01_forgiveness", "q03_charity", "q05_nineteen",
    "q09_justice", "q13_afterlife", "q10_previous_scriptures",
]

def load_subset():
    with open(PROJECT_ROOT / "test_dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)

    # Core 6 + 1 unanswerable = 7 questions (~8-10 min/trial)
    unanswerable = [q for q in dataset if q.get("expected_citations") == []]
    selected_ids = set(CORE_SUBSET_IDS)

    subset = [q for q in dataset if q["id"] in selected_ids]
    subset.extend(unanswerable[:1])

    return subset


# ── objective ────────────────────────────────────────────────────────────────

def create_objective(subset: list, original_config: dict, model: str):
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pw = os.getenv("NEO4J_PASSWORD", "")
    neo4j_db = os.getenv("NEO4J_DATABASE", "quran")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))
    driver.verify_connectivity()

    def objective(trial):
        cfg = copy.deepcopy(original_config)

        # Parameters to optimize (same as API version)
        cfg["llm"]["max_tokens"] = trial.suggest_categorical("max_tokens", [2048, 3072, 4096])
        cfg["retrieval"]["semantic_search"]["default_top_k"] = trial.suggest_int("top_k", 10, 60, step=5)
        cfg["retrieval"]["traverse_topic"]["seed_limit"] = trial.suggest_int("seed_limit", 10, 50, step=5)
        cfg["retrieval"]["traverse_topic"]["hop1_limit"] = trial.suggest_int("hop1_limit", 20, 80, step=10)
        cfg["retrieval"]["traverse_topic"]["hop2_limit"] = trial.suggest_int("hop2_limit", 10, 60, step=10)
        cfg["scoring"]["related_to_min_score"] = trial.suggest_float("min_score", 0.05, 0.30, step=0.05)
        cfg["retrieval"]["get_verse"]["neighbour_limit"] = trial.suggest_int("neighbour_limit", 6, 20, step=2)
        cfg["retrieval"]["get_verse"]["keyword_limit"] = trial.suggest_int("keyword_limit", 8, 20, step=2)

        # Write config and reload
        save_config(cfg)
        try:
            import config as config_mod
            config_mod.reload()
        except Exception:
            pass

        # Get system prompt
        try:
            import config as config_mod
            system_prompt = config_mod.system_prompt()
        except Exception:
            system_prompt = "You are a Quran scholar. Cite verses using [surah:verse] format."

        # Evaluate each question
        weights = cfg["evaluation"]["composite_weights"]
        totals = {m: 0.0 for m in weights}
        t0 = time.time()

        with driver.session(database=neo4j_db) as session:
            for q in subset:
                expected = set(q.get("expected_citations", []))
                scores = eval_question_local(
                    q["question"], expected, model, session, system_prompt
                )
                for m in weights:
                    if m in scores:
                        totals[m] += scores[m]

        n = len(subset)
        avg = {m: totals[m] / n for m in weights}
        composite = sum(weights[m] * avg[m] for m in weights)
        elapsed = time.time() - t0

        print(f"  Trial {trial.number}: QIS={composite:.4f} ({elapsed:.0f}s) "
              f"top_k={trial.params.get('top_k')}, "
              f"seed={trial.params.get('seed_limit')}, "
              f"hop1={trial.params.get('hop1_limit')}")

        log_trial({
            "trial": trial.number,
            "timestamp": datetime.now().isoformat(),
            "score": round(composite, 4),
            "elapsed_s": round(elapsed, 1),
            "params": trial.params,
            "aggregate": {m: round(avg[m], 4) for m in weights},
        })

        return composite

    return objective, driver


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Autoresearch Loop (Local/Free)")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials (default 100)")
    parser.add_argument("--model", type=str, default="qwen2.5:14b-instruct-q6_K",
                        help="Ollama model name")
    args = parser.parse_args()

    print("=" * 70)
    print("AUTORESEARCH LOOP (LOCAL) — Free Overnight Optimization")
    print("=" * 70)
    print(f"Model: {args.model} (Ollama)")
    print(f"Trials: {args.trials}")
    print(f"Cost: $0.00 (all local)")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Verify Ollama is running
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama models available: {models}")
        if args.model not in models:
            print(f"WARNING: {args.model} not found. Available: {models}")
    except Exception as e:
        print(f"ERROR: Ollama not reachable ({e}). Start it with: ollama serve")
        sys.exit(1)

    # Save original config
    original_config = load_config()
    backup_path = PROJECT_ROOT / "pipeline_config_backup.yaml"
    with open(backup_path, "w", encoding="utf-8") as f:
        yaml.dump(original_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    subset = load_subset()
    print(f"Eval subset: {len(subset)} questions")
    print(f"Estimated time per trial: ~3-5 min")
    print(f"Estimated total time: {args.trials * 4 / 60:.1f} hours")
    print("=" * 70)
    print()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="quran-graph-local-optimization",
    )

    objective, driver = create_objective(subset, original_config, args.model)
    best_score = 0.0

    def callback(study, trial):
        nonlocal best_score
        if trial.value and trial.value > best_score:
            best_score = trial.value
            # Save best config
            best_cfg = copy.deepcopy(original_config)
            for key, val in trial.params.items():
                if key == "max_tokens":
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

            with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(best_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            print(f"\n  *** NEW BEST: QIS={best_score:.4f} ***\n")

        if (trial.number + 1) % 10 == 0:
            print(f"\n--- Progress: {trial.number + 1} trials, best={best_score:.4f} ---\n")

    try:
        study.optimize(objective, n_trials=args.trials, callbacks=[callback])
    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving results...")

    # Final report
    print("\n" + "=" * 70)
    print("AUTORESEARCH (LOCAL) COMPLETE")
    print("=" * 70)
    print(f"Trials: {len(study.trials)}")
    print(f"Best QIS: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Restore best config
    if BEST_CONFIG_PATH.exists():
        import shutil
        shutil.copy2(BEST_CONFIG_PATH, CONFIG_PATH)
        print(f"\nBest config applied to {CONFIG_PATH.name}")
    else:
        save_config(original_config)
        print(f"\nOriginal config restored.")

    print(f"Log: {LOG_PATH.name}")
    driver.close()
    print("=" * 70)


if __name__ == "__main__":
    main()
