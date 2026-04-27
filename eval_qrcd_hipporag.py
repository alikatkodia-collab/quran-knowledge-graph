"""
A/B test: HippoRAG-style traversal vs plain BGE-M3 vector retrieval on QRCD.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
URI = os.getenv("NEO4J_URI"); USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD"); DB = os.getenv("NEO4J_DATABASE", "quran")

os.environ.setdefault("SEMANTIC_SEARCH_INDEX", "verse_embedding_m3")

QRCD = Path("data/qrcd_test.jsonl")


def expand(surah, vrange):
    out = set()
    for chunk in str(vrange).split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            try:
                for v in range(int(a), int(b) + 1):
                    out.add(f"{surah}:{v}")
            except ValueError:
                pass
        else:
            try:
                int(chunk); out.add(f"{surah}:{chunk}")
            except ValueError:
                pass
    return out


def hit_at_k(ids, gold, k):
    return any(r in gold for r in ids[:k])

def recall_at_k(ids, gold, k):
    return sum(1 for r in ids[:k] if r in gold) / len(gold) if gold else 0

def first_hit_rank(ids, gold):
    for i, r in enumerate(ids, 1):
        if r in gold:
            return i
    return None


def main():
    items = [json.loads(l) for l in QRCD.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_q = defaultdict(lambda: set())
    for it in items:
        by_q[it["question"]] |= expand(it["surah"], it["verses"])
    questions = [{"question": q, "gold": g} for q, g in by_q.items()]
    print(f"Loaded {len(questions)} unique questions")

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("Neo4j OK")

    from hipporag_traverse import hipporag_search, _vector_seed
    K_VALUES = [5, 10, 20]

    summaries = {"vector_only": defaultdict(int), "hipporag": defaultdict(int)}
    summaries["vector_only"]["sum_rr"] = 0.0
    summaries["hipporag"]["sum_rr"] = 0.0
    for k in K_VALUES:
        summaries["vector_only"][f"sum_recall@{k}"] = 0.0
        summaries["hipporag"][f"sum_recall@{k}"] = 0.0

    n_evaluated = 0

    with driver.session(database=DB) as s:
        for i, q in enumerate(questions, 1):
            qtext = q["question"]
            gold = q["gold"]

            # Vanilla vector
            vec_seeds = _vector_seed(s, qtext, top_k=20)
            vec_ids = [v[0] for v in vec_seeds]

            # HippoRAG
            hr = hipporag_search(s, qtext, top_k_seed_verses=30,
                                 top_k_past_queries=5, final_top_k=20)
            hr_ids = [r["verse_id"] for r in hr.get("results", [])]

            n_evaluated += 1
            for k in K_VALUES:
                summaries["vector_only"][f"hit@{k}"] += int(hit_at_k(vec_ids, gold, k))
                summaries["vector_only"][f"sum_recall@{k}"] += recall_at_k(vec_ids, gold, k)
                summaries["hipporag"][f"hit@{k}"] += int(hit_at_k(hr_ids, gold, k))
                summaries["hipporag"][f"sum_recall@{k}"] += recall_at_k(hr_ids, gold, k)

            r1 = first_hit_rank(vec_ids, gold)
            r2 = first_hit_rank(hr_ids, gold)
            summaries["vector_only"]["sum_rr"] += (1.0 / r1) if r1 else 0
            summaries["hipporag"]["sum_rr"] += (1.0 / r2) if r2 else 0

            print(f"[{i}/{len(questions)}] vec_hit@10={hit_at_k(vec_ids, gold, 10)} "
                  f"hr_hit@10={hit_at_k(hr_ids, gold, 10)}")

    n = max(n_evaluated, 1)
    print("\n=== Comparison ===")
    print(f"{'metric':<14} {'vector':<10} {'hipporag':<10} {'delta':<8}")
    for k in K_VALUES:
        v = summaries["vector_only"][f"hit@{k}"] / n
        h = summaries["hipporag"][f"hit@{k}"] / n
        print(f"{'hit@'+str(k):<14} {v:<10.4f} {h:<10.4f} {h-v:+.4f}")
    for k in K_VALUES:
        v = summaries["vector_only"][f"sum_recall@{k}"] / n
        h = summaries["hipporag"][f"sum_recall@{k}"] / n
        print(f"{'recall@'+str(k):<14} {v:<10.4f} {h:<10.4f} {h-v:+.4f}")
    v_mrr = summaries["vector_only"]["sum_rr"] / n
    h_mrr = summaries["hipporag"]["sum_rr"] / n
    print(f"{'mrr':<14} {v_mrr:<10.4f} {h_mrr:<10.4f} {h_mrr-v_mrr:+.4f}")

    out = Path("data/qrcd_hipporag_compare.json")
    out.write_text(json.dumps({
        "n": n,
        "vector_only": {
            **{f"hit@{k}": summaries["vector_only"][f"hit@{k}"]/n for k in K_VALUES},
            **{f"recall@{k}": summaries["vector_only"][f"sum_recall@{k}"]/n for k in K_VALUES},
            "mrr": v_mrr,
        },
        "hipporag": {
            **{f"hit@{k}": summaries["hipporag"][f"hit@{k}"]/n for k in K_VALUES},
            **{f"recall@{k}": summaries["hipporag"][f"sum_recall@{k}"]/n for k in K_VALUES},
            "mrr": h_mrr,
        },
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    driver.close()


if __name__ == "__main__":
    main()
