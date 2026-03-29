"""
Parameterized Quran Knowledge Graph builder — the AGENT MODIFIES THIS FILE.

This is the equivalent of Karpathy's train.py: the file that the optimization
loop mutates to find better configurations.

All tunable parameters are in the PARAMS dict at the top.
The build() function rebuilds the graph CSVs using these parameters.
"""

import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

csv.field_size_limit(sys.maxsize)

# ══════════════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS — the optimization loop mutates these values
# ══════════════════════════════════════════════════════════════════════════════

PARAMS = {
    # TF-IDF vectorizer settings
    "min_df": 2,              # min document frequency (int: absolute count)
    "max_df": 300,            # max document frequency (int: absolute count)
    "min_tfidf_score": 0.04,  # minimum TF-IDF score to create a MENTIONS edge
    "max_features": 50000,    # max vocabulary size

    # Graph topology
    "max_edges_per_verse": 12,   # cap RELATED_TO edges per verse
    "max_verse_freq": 300,       # skip keywords in more than this many verses for RELATED_TO
    "edge_weight_method": "geometric_mean",  # "geometric_mean", "harmonic_mean", or "min"

    # Stopwords
    "extra_stopwords": [],       # additional words to exclude
    "min_token_length": 3,       # minimum token character length

    # Lemmatization
    "lemma_verb_first": True,    # Try verb lemma before noun

    # Sublinear TF
    "sublinear_tf": False,       # Use sublinear tf scaling (1 + log(tf))

    # Ngram range
    "ngram_min": 1,              # minimum n-gram size
    "ngram_max": 1,              # maximum n-gram size (1=unigrams, 2=bigrams)

    # Norm
    "norm": "l2",                # normalization: "l1", "l2", or None
}

# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Stopwords ────────────────────────────────────────────────────────────────

QURAN_STOPWORDS = {
    "god", "lord", "indeed", "surely", "verily", "thus", "therefore",
    "henceforth", "none", "nothing", "anyone", "everyone", "whoever",
    "whatever", "wherever", "whenever", "said", "says",
    "shall", "will", "upon", "unto", "thee", "thy", "thou", "thine",
    "hath", "doth", "ye", "yea", "nay", "thereof", "therein", "thereby",
    "herein", "hereby", "wherein", "whereby", "also", "even", "still",
    "yet", "well", "away", "back", "never", "ever", "always", "already",
    "truly", "certainly", "absolutely", "completely", "totally", "simply",
}

nltk_stops = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def get_all_stopwords():
    extra = set(w.lower() for w in PARAMS["extra_stopwords"])
    return nltk_stops | QURAN_STOPWORDS | extra


def lemmatize_token(token: str) -> str:
    if PARAMS["lemma_verb_first"]:
        verb = lemmatizer.lemmatize(token, pos='v')
        if verb != token:
            return verb
        return lemmatizer.lemmatize(token, pos='n')
    else:
        noun = lemmatizer.lemmatize(token, pos='n')
        if noun != token:
            return noun
        return lemmatizer.lemmatize(token, pos='v')


def tokenize_and_lemmatize(text: str) -> list:
    all_stops = get_all_stopwords()
    min_len = PARAMS["min_token_length"]
    tokens = re.findall(r'[a-z]+', text.lower())
    result = []
    for t in tokens:
        if len(t) < min_len:
            continue
        if t in all_stops:
            continue
        lemma = lemmatize_token(t)
        if lemma in all_stops:
            continue
        result.append(lemma)
    return result


class LemmaAnalyzer:
    def __call__(self, text: str) -> list:
        return tokenize_and_lemmatize(text)


# ── Build pipeline ───────────────────────────────────────────────────────────

def load_verses():
    with open(os.path.join(DATA_DIR, "verses.json"), encoding="utf-8") as f:
        return json.load(f)


def build(output_dir=None):
    """
    Full graph build pipeline. Returns timing and stats.
    Writes CSVs to output_dir (defaults to data/).
    """
    if output_dir is None:
        output_dir = DATA_DIR
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()

    # 1. Load verses
    verses = load_verses()
    texts = [v["text"] for v in verses]

    # 2. TF-IDF
    vectorizer = TfidfVectorizer(
        analyzer=LemmaAnalyzer(),
        max_df=PARAMS["max_df"],
        min_df=PARAMS["min_df"],
        max_features=PARAMS["max_features"],
        sublinear_tf=PARAMS["sublinear_tf"],
        ngram_range=(PARAMS["ngram_min"], PARAMS["ngram_max"]),
        norm=PARAMS["norm"],
    )
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # 3. Write verse nodes
    with open(os.path.join(output_dir, "verse_nodes.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["verseId", "surah", "verseNum", "surahName", "text"])
        writer.writeheader()
        for v in verses:
            writer.writerow({
                "verseId": v["verse_id"],
                "surah": v["surah"],
                "verseNum": v["verse"],
                "surahName": v["surah_name"],
                "text": v["text"].replace("\n", " "),
            })

    # 4. Write keyword nodes
    with open(os.path.join(output_dir, "keyword_nodes.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["keyword"])
        writer.writeheader()
        for kw in feature_names:
            writer.writerow({"keyword": kw})

    # 5. Write MENTIONS edges
    keyword_to_verses = defaultdict(list)
    total_mentions = 0
    min_score = PARAMS["min_tfidf_score"]

    with open(os.path.join(output_dir, "verse_keyword_rels.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["verseId", "keyword", "score"])
        writer.writeheader()
        cx = matrix.tocsr()
        for i, verse in enumerate(verses):
            row = cx.getrow(i)
            for idx, score in zip(row.indices, row.data):
                if score >= min_score:
                    kw = feature_names[idx]
                    writer.writerow({
                        "verseId": verse["verse_id"],
                        "keyword": kw,
                        "score": round(float(score), 6),
                    })
                    keyword_to_verses[kw].append((verse["verse_id"], float(score)))
                    total_mentions += 1

    # 6. Write RELATED_TO edges
    max_edges = PARAMS["max_edges_per_verse"]
    max_freq = PARAMS["max_verse_freq"]
    weight_method = PARAMS["edge_weight_method"]

    pair_scores = defaultdict(float)
    for kw, verse_list in keyword_to_verses.items():
        if len(verse_list) > max_freq:
            continue
        for i in range(len(verse_list)):
            for j in range(i + 1, len(verse_list)):
                v1_id, s1 = verse_list[i]
                v2_id, s2 = verse_list[j]
                if weight_method == "geometric_mean":
                    shared_score = (s1 * s2) ** 0.5
                elif weight_method == "harmonic_mean":
                    shared_score = 2 * s1 * s2 / (s1 + s2) if (s1 + s2) > 0 else 0
                elif weight_method == "min":
                    shared_score = min(s1, s2)
                else:
                    shared_score = (s1 * s2) ** 0.5
                key = (min(v1_id, v2_id), max(v1_id, v2_id))
                pair_scores[key] += shared_score

    # Cap edges per verse
    verse_candidates = defaultdict(list)
    for (v1, v2), score in pair_scores.items():
        verse_candidates[v1].append((score, v2))
        verse_candidates[v2].append((score, v1))

    accepted_pairs = set()
    for verse_id, candidates in verse_candidates.items():
        candidates.sort(reverse=True)
        for score, other_id in candidates[:max_edges]:
            key = (min(verse_id, other_id), max(verse_id, other_id))
            accepted_pairs.add(key)

    total_related = 0
    with open(os.path.join(output_dir, "verse_related_rels.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["verseId1", "verseId2", "score"])
        writer.writeheader()
        for (v1, v2) in accepted_pairs:
            writer.writerow({
                "verseId1": v1,
                "verseId2": v2,
                "score": round(pair_scores[(v1, v2)], 6),
            })
            total_related += 1

    elapsed = time.time() - t0

    stats = {
        "verses": len(verses),
        "keywords": len(feature_names),
        "mentions_edges": total_mentions,
        "related_edges": total_related,
        "build_time_seconds": round(elapsed, 2),
        "params": dict(PARAMS),
    }
    return stats


if __name__ == "__main__":
    stats = build()
    print(json.dumps(stats, indent=2))
