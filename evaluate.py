"""
Quran Knowledge Graph — Evaluation Harness

Runs test questions through the pipeline and computes a composite grounding
metric.  This is the scalar signal that the autoresearch optimizer maximises.

Usage:
    py evaluate.py                       # run all questions, print report
    py evaluate.py --ids q01,q05         # run specific question IDs
    py evaluate.py --out results.json    # save full results to file

Metrics (per question):
    citation_recall    — fraction of expected citations found in answer
    citation_precision — fraction of produced citations that are in expected set
    grounding_rate     — fraction of non-empty paragraphs containing >=1 citation
    answer_relevance   — cosine similarity between question and answer embeddings

Composite score = weighted average of the four metrics (weights in pipeline_config.yaml).
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

load_dotenv(Path(__file__).parent / ".env", override=True)

# ── helpers ──────────────────────────────────────────────────────────────────

_BRACKET_REF = re.compile(r'\[(\d+:\d+)\]')

def extract_citations(text: str) -> set[str]:
    """Pull all [surah:verse] references from a response."""
    return set(_BRACKET_REF.findall(text))

def grounding_rate(text: str) -> float:
    """Fraction of non-empty paragraphs that contain at least one citation."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return 0.0
    cited = sum(1 for p in paragraphs if _BRACKET_REF.search(p))
    return cited / len(paragraphs)

def citation_recall(produced: set[str], expected: set[str]) -> float:
    if not expected:
        return 1.0
    return len(produced & expected) / len(expected)

def citation_precision(produced: set[str], expected: set[str]) -> float:
    if not produced:
        return 0.0
    return len(produced & expected) / len(produced)

def answer_relevance(question: str, answer: str) -> float:
    """Cosine similarity between question and answer embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer(cfg.embedding_model())
        vecs = model.encode([question, answer], normalize_embeddings=True)
        return float(np.dot(vecs[0], vecs[1]))
    except Exception:
        return 0.0

# ── run one question through the pipeline ────────────────────────────────────

def run_question(question: str, session, client) -> str:
    """Run a single question through the agentic loop and return the final text."""
    from chat import TOOLS, SYSTEM_PROMPT, dispatch_tool

    msgs = [{"role": "user", "content": question}]
    full_text = ""
    max_turns = 15  # safety cap on tool-use turns

    for _ in range(max_turns):
        response = client.messages.create(
            model=cfg.llm_model(),
            max_tokens=cfg.llm_max_tokens(),
            temperature=cfg.llm_temperature(),
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=msgs,
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                full_text += block.text

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_str = dispatch_tool(session, block.name, block.input, user_query=question)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

        msgs.append({"role": "assistant", "content": response.content})
        msgs.append({"role": "user", "content": tool_results})

    return full_text

# ── evaluate ─────────────────────────────────────────────────────────────────

def evaluate(ids: list[str] | None = None, output_path: str | None = None):
    """Run evaluation on all (or selected) test questions."""
    import anthropic
    from neo4j import GraphDatabase

    # Load test dataset
    dataset_path = cfg.eval_dataset_path()
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if ids:
        id_set = set(ids)
        dataset = [q for q in dataset if q["id"] in id_set]

    if not dataset:
        print("No matching questions found.")
        return

    # Connect
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pw = os.getenv("NEO4J_PASSWORD", "")
    neo4j_db = os.getenv("NEO4J_DATABASE", "quran")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))
    driver.verify_connectivity()
    client = anthropic.Anthropic(api_key=api_key)

    weights = cfg.eval_weights()
    results = []
    totals = {m: 0.0 for m in weights}

    print(f"Evaluating {len(dataset)} questions...")
    print(f"Config: {cfg.llm_model()}, max_tokens={cfg.llm_max_tokens()}")
    print(f"Weights: {weights}")
    print("=" * 70)

    with driver.session(database=neo4j_db) as session:
        for i, q in enumerate(dataset):
            qid = q["id"]
            question = q["question"]
            expected = set(q["expected_citations"])

            print(f"\n[{i+1}/{len(dataset)}] {qid}: {question[:60]}...")
            t0 = time.time()

            answer = run_question(question, session, client)
            elapsed = time.time() - t0

            produced = extract_citations(answer)
            cr = citation_recall(produced, expected)
            cp = citation_precision(produced, expected)
            gr = grounding_rate(answer)
            ar = answer_relevance(question, answer)

            composite = (
                weights.get("citation_recall", 0) * cr +
                weights.get("citation_precision", 0) * cp +
                weights.get("grounding_rate", 0) * gr +
                weights.get("answer_relevance", 0) * ar
            )

            result = {
                "id": qid,
                "question": question,
                "difficulty": q.get("difficulty", ""),
                "elapsed_s": round(elapsed, 1),
                "citations_produced": sorted(produced),
                "citations_expected": sorted(expected),
                "citation_recall": round(cr, 4),
                "citation_precision": round(cp, 4),
                "grounding_rate": round(gr, 4),
                "answer_relevance": round(ar, 4),
                "composite": round(composite, 4),
                "answer_length": len(answer),
            }
            results.append(result)

            for m in weights:
                totals[m] += result[m]

            print(f"  recall={cr:.2f}  precision={cp:.2f}  grounding={gr:.2f}  relevance={ar:.2f}  composite={composite:.3f}  ({elapsed:.1f}s)")
            print(f"  citations: {len(produced)} produced, {len(expected)} expected, {len(produced & expected)} overlap")

    # Aggregate
    n = len(results)
    avg = {m: round(totals[m] / n, 4) for m in weights}
    avg_composite = round(sum(weights[m] * avg[m] for m in weights), 4)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print(f"  Questions evaluated: {n}")
    for m in weights:
        print(f"  avg {m}: {avg[m]:.4f}  (weight {weights[m]})")
    print(f"  COMPOSITE SCORE: {avg_composite:.4f}")
    print("=" * 70)

    # Save
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_snapshot": {
            "llm_model": cfg.llm_model(),
            "llm_max_tokens": cfg.llm_max_tokens(),
            "embedding_model": cfg.embedding_model(),
            "system_prompt_source": cfg.raw()["system_prompt"],
        },
        "weights": weights,
        "aggregate": avg,
        "composite_score": avg_composite,
        "per_question": results,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nFull results saved to {output_path}")

    driver.close()
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Quran Knowledge Graph pipeline")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated question IDs to run")
    parser.add_argument("--out", type=str, default="eval_results.json", help="Output file path")
    args = parser.parse_args()

    ids = [x.strip() for x in args.ids.split(",")] if args.ids else None
    evaluate(ids=ids, output_path=args.out)
