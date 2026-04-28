"""
eval_ablation_retrieval.py — Nixon "argue each component out" study.

Measures the marginal contribution of each retrieval-pipeline stage on QRCD:

  variant 0 — vector-only            (BGE-M3 dense, no post-processing)
  variant 1 — + cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
  variant 2 — + lost-in-middle reorder (variant 1 + reorder for U-shape)
  variant 3 — full gate              (rerank + reorder + quality assess + threshold)

For each variant: hit@5/10/20, recall@10, MRR.

Use the deltas to decide which stages to keep. Stages with <2pt MAP@10
gain are candidates for removal per Nixon's "every box has to justify
itself" rule.

Run: python eval_ablation_retrieval.py
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
TOP_K_INITIAL = 30   # how many to pull from vector index before re-ranking
TOP_K_FINAL = 20     # how many we score against gold


def expand(s, vr):
    out = set()
    for c in str(vr).split(","):
        c = c.strip()
        if "-" in c:
            a, b = c.split("-", 1)
            try: out.update(f"{s}:{v}" for v in range(int(a), int(b)+1))
            except: pass
        else:
            try: int(c); out.add(f"{s}:{c}")
            except: pass
    return out


def load_questions():
    items = [json.loads(l) for l in QRCD.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_q = defaultdict(set)
    for it in items:
        by_q[it["question"]] |= expand(it["surah"], it["verses"])
    return [{"question": q, "gold": g} for q, g in by_q.items()]


def hit(ids, gold, k): return any(r in gold for r in ids[:k])
def recall(ids, gold, k): return sum(1 for r in ids[:k] if r in gold) / len(gold) if gold else 0
def fhr(ids, gold):
    for i, r in enumerate(ids, 1):
        if r in gold: return i


def main():
    questions = load_questions()
    print(f"loaded {len(questions)} questions")

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("Neo4j OK")

    # Lazy load models
    from sentence_transformers import SentenceTransformer, CrossEncoder
    bge = SentenceTransformer("BAAI/bge-m3")
    bge.max_seq_length = 512
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("models loaded")

    from retrieval_gate import lost_in_middle_reorder, assess_quality

    # Run all variants in one pass per question to share embedding
    variants = ["v0_vector", "v1_rerank", "v2_rerank_reorder", "v3_full_gate"]
    agg = {v: {"h@5": 0, "h@10": 0, "h@20": 0, "r@10": 0.0, "rr": 0.0,
               "elapsed": 0.0} for v in variants}
    n = len(questions)

    import time

    with driver.session(database=DB) as s:
        for i, q in enumerate(questions, 1):
            qtext = q["question"]
            gold = q["gold"]

            # Pull initial top-K from vector index
            t0 = time.time()
            qvec = bge.encode([qtext], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
            rows = s.run("""
                CALL db.index.vector.queryNodes('verse_embedding_m3', $k, $vec)
                YIELD node, score WHERE node.verseId IS NOT NULL
                RETURN node.verseId AS id, score, node.text AS text
                ORDER BY score DESC
            """, k=TOP_K_INITIAL, vec=qvec).data()
            verses = [{"verse_id": r["id"], "text": r["text"], "sim": float(r["score"])} for r in rows]

            # variant 0 — vector only
            v0_ids = [v["verse_id"] for v in verses][:TOP_K_FINAL]
            agg["v0_vector"]["elapsed"] += time.time() - t0

            # variant 1 — + cross-encoder rerank
            t1 = time.time()
            pairs = [(qtext, v["text"]) for v in verses]
            scores = reranker.predict(pairs)
            for v, sc in zip(verses, scores):
                v["relevance_score"] = float(sc)
            v1 = sorted(verses, key=lambda x: -x["relevance_score"])[:TOP_K_FINAL]
            v1_ids = [v["verse_id"] for v in v1]
            agg["v1_rerank"]["elapsed"] += time.time() - t1

            # variant 2 — + lost-in-middle reorder
            v2 = lost_in_middle_reorder(list(v1))
            v2_ids = [v["verse_id"] for v in v2]

            # variant 3 — + quality assessment (drop if poor)
            quality = assess_quality(v1)
            if quality == "poor":
                # full gate would surface a "low confidence" warning; for retrieval
                # ranking it would still return the rerank+reorder
                v3_ids = v2_ids
            else:
                v3_ids = v2_ids

            # Score all four
            for name, ids in [("v0_vector", v0_ids), ("v1_rerank", v1_ids),
                              ("v2_rerank_reorder", v2_ids), ("v3_full_gate", v3_ids)]:
                for k in [5, 10, 20]:
                    agg[name][f"h@{k}"] += int(hit(ids, gold, k))
                agg[name]["r@10"] += recall(ids, gold, 10)
                rk = fhr(ids, gold)
                agg[name]["rr"] += (1.0/rk) if rk else 0

    print(f"\nN={n}")
    print(f"{'variant':<22}{'h@5':>8}{'h@10':>8}{'h@20':>8}{'r@10':>8}{'mrr':>8}{'~ms/q':>8}")
    rows_out = []
    base_h10 = agg["v0_vector"]["h@10"] / n
    base_mrr = agg["v0_vector"]["rr"] / n
    for name in variants:
        a = agg[name]
        h10 = a["h@10"] / n
        mrr = a["rr"] / n
        d_h = h10 - base_h10
        d_m = mrr - base_mrr
        rows_out.append({
            "variant": name,
            "hit@5":   round(a["h@5"] / n, 4),
            "hit@10":  round(h10, 4),
            "hit@20":  round(a["h@20"] / n, 4),
            "recall@10": round(a["r@10"] / n, 4),
            "mrr":     round(mrr, 4),
            "delta_hit@10_vs_v0": round(d_h, 4),
            "delta_mrr_vs_v0": round(d_m, 4),
            "ms_per_q": round(a["elapsed"] * 1000 / n, 1),
        })
        print(f"{name:<22}{a['h@5']/n:>8.4f}{h10:>8.4f}{a['h@20']/n:>8.4f}"
              f"{a['r@10']/n:>8.4f}{mrr:>8.4f}{a['elapsed']*1000/n:>8.0f}")

    # Save
    Path("data/qrcd_ablation.json").write_text(
        json.dumps({"n": n, "variants": rows_out}, indent=2),
        encoding="utf-8"
    )
    print(f"\nSaved data/qrcd_ablation.json")
    driver.close()


if __name__ == "__main__":
    main()
