"""
Deduction Analyzer — categorizes, scores, and builds a meta-knowledge graph
from raw syllogistic deductions.

Three layers:
  1. Theological categorization (keyword-based clustering into themes)
  2. Quality scoring (filters noise from signal)
  3. Meta-knowledge graph (deductions as nodes, shared themes/verses as edges)
  4. Insight synthesis (groups related deductions into higher-order conclusions)
"""

import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

csv.field_size_limit(sys.maxsize)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUTORESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# Theological Categories
# ══════════════════════════════════════════════════════════════════════════════

THEOLOGICAL_CATEGORIES = {
    "monotheism_and_gods_nature": {
        "keywords": {"god", "lord", "creator", "eternal", "living", "omniscient", "almighty",
                     "merciful", "gracious", "wise", "light", "throne", "sovereign", "one",
                     "omnipotent", "king", "supreme", "glory"},
        "description": "God's nature, attributes, oneness, and sovereignty",
    },
    "prophecy_and_revelation": {
        "keywords": {"prophet", "messenger", "revelation", "scripture", "quran", "book",
                     "message", "recite", "reveal", "send", "sign", "proof", "miracle",
                     "torah", "gospel", "psalm", "inspire", "angel"},
        "description": "Prophetic mission, revelation, scripture, and divine communication",
    },
    "moral_law_and_ethics": {
        "keywords": {"righteous", "good", "evil", "justice", "honest", "truth", "lie",
                     "cheat", "steal", "murder", "adultery", "orphan", "parent", "kind",
                     "patience", "charity", "humble", "arrogant", "oppression", "fair"},
        "description": "Moral principles, ethical conduct, and commandments",
    },
    "worship_and_ritual": {
        "keywords": {"prayer", "salat", "contact", "fast", "ramadan", "pilgrimage", "hajj",
                     "worship", "prostrate", "bow", "ablution", "mosque", "shrine", "rite",
                     "sacrifice", "observe", "reverence", "commemorate"},
        "description": "Acts of worship, prayer, fasting, pilgrimage, and ritual practice",
    },
    "afterlife_and_judgment": {
        "keywords": {"judgment", "resurrection", "paradise", "hell", "fire", "garden",
                     "reward", "punish", "hereafter", "day", "scale", "record", "account",
                     "eternal", "doom", "retribution", "heaven", "reckoning", "trumpet"},
        "description": "Day of Judgment, paradise, hell, resurrection, and accountability",
    },
    "creation_and_cosmology": {
        "keywords": {"create", "heaven", "earth", "sky", "mountain", "sea", "water",
                     "animal", "plant", "human", "clay", "drop", "bone", "universe",
                     "star", "moon", "sun", "night", "livestock", "rain"},
        "description": "Creation of universe, humans, nature, and natural phenomena",
    },
    "prophetic_narratives": {
        "keywords": {"moses", "abraham", "jesus", "noah", "joseph", "david", "solomon",
                     "pharaoh", "mary", "adam", "lot", "jonah", "isaac", "ishmael",
                     "jacob", "aaron", "job", "elijah", "zachariah"},
        "description": "Stories of prophets and historical narratives",
    },
    "social_law": {
        "keywords": {"marry", "divorce", "wife", "husband", "inherit", "woman", "man",
                     "witness", "contract", "debt", "trade", "usury", "orphan", "wealth",
                     "property", "dowry", "guardian", "family", "child"},
        "description": "Family law, marriage, divorce, inheritance, and social contracts",
    },
    "covenant_and_obedience": {
        "keywords": {"covenant", "obey", "follow", "stray", "rebel", "transgress",
                     "submit", "devote", "idol", "disbelieve", "reject", "deny",
                     "believe", "faith", "trust", "sincere", "hypocrite"},
        "description": "Divine covenant, faith, obedience, and disbelief",
    },
    "divine_mercy_and_forgiveness": {
        "keywords": {"forgive", "mercy", "merciful", "repent", "redeem", "pardon",
                     "compassion", "grace", "bless", "save", "atone", "absolve",
                     "relent", "clemency"},
        "description": "God's mercy, forgiveness, repentance, and redemption",
    },
    "warfare_and_struggle": {
        "keywords": {"fight", "war", "strive", "kill", "enemy", "battle", "army",
                     "victory", "defeat", "flee", "migrate", "exile", "oppress",
                     "defend", "martyr", "jihad", "persecute"},
        "description": "Armed struggle, persecution, migration, and defense",
    },
    "knowledge_and_wisdom": {
        "keywords": {"knowledge", "wisdom", "learn", "teach", "understand", "intellect",
                     "reason", "reflect", "ponder", "sign", "lesson", "aware",
                     "cognizant", "comprehend", "educate"},
        "description": "Pursuit of knowledge, wisdom, reflection, and understanding",
    },
    "mathematical_miracle": {
        "keywords": {"nineteen", "number", "count", "mathematical", "code", "initial",
                     "letter", "gematrical"},
        "description": "Khalifa's mathematical miracle of 19 and Quranic numerology",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Categorizer
# ══════════════════════════════════════════════════════════════════════════════

def categorize_deduction(deduction, verses_text):
    """Assign theological categories to a deduction based on keywords."""
    # Combine all text: conclusion + bridge keywords + verse texts
    text_parts = [
        deduction.get("conclusion", "").lower(),
        " ".join(deduction.get("bridge_keywords", [])).lower(),
    ]
    for vid in deduction.get("premise_verses", []):
        if vid in verses_text:
            text_parts.append(verses_text[vid].lower())

    combined = " ".join(text_parts)
    words = set(re.findall(r'[a-z]+', combined))

    # Score each category
    category_scores = {}
    for cat_name, cat_info in THEOLOGICAL_CATEGORIES.items():
        overlap = words & cat_info["keywords"]
        if overlap:
            category_scores[cat_name] = len(overlap)

    if not category_scores:
        return [("uncategorized", 0)]

    # Return top categories (sorted by score)
    sorted_cats = sorted(category_scores.items(), key=lambda x: -x[1])
    return sorted_cats[:3]  # top 3 categories


# ══════════════════════════════════════════════════════════════════════════════
# Quality Scorer
# ══════════════════════════════════════════════════════════════════════════════

def score_quality(deduction, verses_text):
    """Score deduction quality 0-100 based on multiple factors."""
    scores = []

    # Factor 1: Bridge keyword specificity (rare keywords = more meaningful)
    bridges = deduction.get("bridge_keywords", [])
    COMMON_WORDS = {"the", "and", "of", "to", "in", "for", "is", "are", "was", "that", "this",
                    "god", "lord", "shall", "will", "upon"}
    specific_bridges = [b for b in bridges if b.lower() not in COMMON_WORDS and len(b) > 3]
    specificity = min(100, len(specific_bridges) * 30)
    scores.append(("specificity", specificity, 0.20))

    # Factor 2: Cross-surah diversity (connecting different parts of the Quran)
    surahs = set()
    for vid in deduction.get("premise_verses", []):
        surahs.add(vid.split(":")[0])
    diversity = min(100, (len(surahs) - 1) * 40)
    scores.append(("surah_diversity", diversity, 0.15))

    # Factor 3: Conclusion meaningfulness (length and structure)
    conclusion = deduction.get("conclusion", "")
    # Penalize very short or very long conclusions
    clen = len(conclusion)
    if 50 < clen < 300:
        meaning = 80
    elif 30 < clen < 500:
        meaning = 50
    else:
        meaning = 20
    scores.append(("meaningfulness", meaning, 0.15))

    # Factor 4: Bridge chain coherence (do the bridges form a logical progression?)
    if len(bridges) >= 2:
        # Check if bridges are diverse (not repeating the same word)
        unique_ratio = len(set(bridges)) / len(bridges)
        coherence = unique_ratio * 100
    else:
        coherence = 30
    scores.append(("coherence", coherence, 0.20))

    # Factor 5: Verse text relevance (do the connected verses actually share concepts?)
    verse_ids = deduction.get("premise_verses", [])
    if len(verse_ids) >= 2:
        first_text = verses_text.get(verse_ids[0], "").lower()
        last_text = verses_text.get(verse_ids[-1], "").lower()
        first_words = set(re.findall(r'[a-z]{4,}', first_text))
        last_words = set(re.findall(r'[a-z]{4,}', last_text))
        shared = first_words & last_words - COMMON_WORDS
        relevance = min(100, len(shared) * 20)
    else:
        relevance = 50
    scores.append(("relevance", relevance, 0.15))

    # Factor 6: Rule type bonus
    rule = deduction.get("rule", "")
    rule_bonus = {
        "thematic_bridge_3hop": 70,
        "transitive_chain": 60,
        "shared_subject_synthesis": 40,
        "shared_subject_multi_predicate": 30,
    }.get(rule, 50)
    scores.append(("rule_bonus", rule_bonus, 0.15))

    composite = sum(s * w for _, s, w in scores)
    return round(composite, 2), {name: round(s, 1) for name, s, _ in scores}


# ══════════════════════════════════════════════════════════════════════════════
# Meta-Knowledge Graph Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_meta_graph(categorized_deductions, min_quality=50):
    """
    Build a meta-knowledge graph where:
    - Nodes = theological themes (categories)
    - Edges = deductions that bridge two themes
    - Edge weight = number and quality of bridging deductions
    """
    # Filter by quality
    good = [d for d in categorized_deductions if d["quality_score"] >= min_quality]

    # Build theme-to-theme edges
    theme_edges = defaultdict(lambda: {"count": 0, "total_quality": 0, "examples": []})
    theme_verse_count = defaultdict(int)

    for d in good:
        cats = d["categories"]
        if len(cats) >= 2:
            # Create edges between category pairs
            for i in range(len(cats)):
                for j in range(i + 1, len(cats)):
                    c1 = min(cats[i][0], cats[j][0])
                    c2 = max(cats[i][0], cats[j][0])
                    key = (c1, c2)
                    theme_edges[key]["count"] += 1
                    theme_edges[key]["total_quality"] += d["quality_score"]
                    if len(theme_edges[key]["examples"]) < 3:
                        theme_edges[key]["examples"].append({
                            "verses": d["premise_verses"],
                            "bridge": d["bridge_keywords"][:4],
                            "conclusion": d["conclusion"][:150],
                        })

        for cat_name, _ in cats:
            theme_verse_count[cat_name] += len(d["premise_verses"])

    # Build graph structure
    nodes = []
    for cat_name, cat_info in THEOLOGICAL_CATEGORIES.items():
        if theme_verse_count.get(cat_name, 0) > 0:
            nodes.append({
                "id": cat_name,
                "label": cat_name.replace("_", " ").title(),
                "description": cat_info["description"],
                "verse_count": theme_verse_count[cat_name],
            })

    edges = []
    for (c1, c2), data in sorted(theme_edges.items(), key=lambda x: -x[1]["count"]):
        edges.append({
            "source": c1,
            "target": c2,
            "weight": data["count"],
            "avg_quality": round(data["total_quality"] / data["count"], 1),
            "examples": data["examples"],
        })

    return {"nodes": nodes, "edges": edges}


# ══════════════════════════════════════════════════════════════════════════════
# Insight Synthesizer
# ══════════════════════════════════════════════════════════════════════════════

def synthesize_insights(categorized_deductions, min_quality=60):
    """
    Group related high-quality deductions into synthesized insights.
    Each insight combines multiple deductions about the same theme.
    """
    good = [d for d in categorized_deductions if d["quality_score"] >= min_quality]

    # Group by primary category
    by_category = defaultdict(list)
    for d in good:
        if d["categories"]:
            primary = d["categories"][0][0]
            by_category[primary].append(d)

    insights = []
    for category, deductions in by_category.items():
        # Sub-group by shared verses
        verse_groups = defaultdict(list)
        for d in deductions:
            # Use the first and last verse as the group key
            key = (d["premise_verses"][0], d["premise_verses"][-1])
            verse_groups[key].append(d)

        # Find clusters (groups with 3+ deductions about same verse pair)
        for (v1, v2), group in verse_groups.items():
            if len(group) >= 2:
                # Collect all bridge keywords
                all_bridges = set()
                for d in group:
                    all_bridges.update(d.get("bridge_keywords", []))

                avg_quality = sum(d["quality_score"] for d in group) / len(group)
                best = max(group, key=lambda d: d["quality_score"])

                insights.append({
                    "category": category,
                    "category_label": THEOLOGICAL_CATEGORIES.get(category, {}).get("description", category),
                    "verse_pair": [v1, v2],
                    "num_supporting_deductions": len(group),
                    "avg_quality": round(avg_quality, 1),
                    "bridge_keywords": sorted(all_bridges)[:10],
                    "best_conclusion": best["conclusion"][:300],
                    "all_conclusions": [d["conclusion"][:200] for d in group[:5]],
                })

    # Sort by number of supporting deductions * quality
    insights.sort(key=lambda x: -x["num_supporting_deductions"] * x["avg_quality"])
    return insights


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def load_verses_text():
    """Load verse texts for analysis."""
    verses = {}
    with open(os.path.join(DATA_DIR, "verse_nodes.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            verses[row["verseId"]] = row["text"]
    return verses


def analyze_all():
    """Run the full analysis pipeline on all deductions."""
    print("Loading data...")
    verses_text = load_verses_text()

    deductions = []
    with open(os.path.join(AUTORESEARCH_DIR, "all_deductions.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                deductions.append(json.loads(line.strip()))

    print(f"  {len(deductions)} deductions to analyze")
    print(f"  {len(verses_text)} verse texts loaded")

    # Step 1: Categorize and score
    print("\nCategorizing and scoring...")
    categorized = []
    for i, d in enumerate(deductions):
        cats = categorize_deduction(d, verses_text)
        quality, breakdown = score_quality(d, verses_text)
        categorized.append({
            **d,
            "categories": cats,
            "quality_score": quality,
            "quality_breakdown": breakdown,
        })
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(deductions)}")

    # Quality distribution
    quality_scores = [d["quality_score"] for d in categorized]
    print(f"\n  Quality distribution:")
    print(f"    High (>70):   {sum(1 for q in quality_scores if q > 70):,}")
    print(f"    Medium (50-70): {sum(1 for q in quality_scores if 50 <= q <= 70):,}")
    print(f"    Low (<50):    {sum(1 for q in quality_scores if q < 50):,}")

    # Category distribution
    cat_counts = Counter()
    for d in categorized:
        for cat_name, _ in d["categories"]:
            cat_counts[cat_name] += 1
    print(f"\n  Top categories:")
    for cat, count in cat_counts.most_common(15):
        print(f"    {cat:40s} {count:6,}")

    # Step 2: Build meta-knowledge graph
    print("\nBuilding meta-knowledge graph...")
    meta_graph = build_meta_graph(categorized, min_quality=50)
    print(f"  {len(meta_graph['nodes'])} theme nodes")
    print(f"  {len(meta_graph['edges'])} theme-to-theme edges")
    print(f"\n  Strongest theme connections:")
    for edge in meta_graph["edges"][:10]:
        print(f"    {edge['source']:35s} ↔ {edge['target']:35s} "
              f"({edge['weight']} deductions, avg quality {edge['avg_quality']})")

    # Step 3: Synthesize insights
    print("\nSynthesizing insights...")
    insights = synthesize_insights(categorized, min_quality=55)
    print(f"  {len(insights)} synthesized insights")

    # Save results
    print("\nSaving results...")

    # Save categorized deductions (top 500 by quality)
    top_categorized = sorted(categorized, key=lambda x: -x["quality_score"])[:500]
    with open(os.path.join(AUTORESEARCH_DIR, "categorized_deductions.json"), "w", encoding="utf-8") as f:
        json.dump(top_categorized, f, indent=2, ensure_ascii=False)

    # Save meta-knowledge graph
    with open(os.path.join(AUTORESEARCH_DIR, "meta_knowledge_graph.json"), "w", encoding="utf-8") as f:
        json.dump(meta_graph, f, indent=2, ensure_ascii=False)

    # Save synthesized insights
    with open(os.path.join(AUTORESEARCH_DIR, "synthesized_insights.json"), "w", encoding="utf-8") as f:
        json.dump(insights[:200], f, indent=2, ensure_ascii=False)

    print(f"\n  categorized_deductions.json — top 500 by quality")
    print(f"  meta_knowledge_graph.json   — theme-to-theme connections")
    print(f"  synthesized_insights.json   — {min(200, len(insights))} grouped insights")

    return {
        "categorized": categorized,
        "meta_graph": meta_graph,
        "insights": insights,
    }


if __name__ == "__main__":
    analyze_all()
