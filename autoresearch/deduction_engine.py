"""
Syllogistic Deduction Engine for the Quran Knowledge Graph.

Treats each verse as an axiom and computes transitive deductions
to surface novel or non-obvious insights.

Architecture:
  1. Proposition Extraction — extract (subject, relation, object) triples from verses
  2. Axiom Registry — store all verse propositions as formal axioms
  3. Syllogistic Rules — apply deductive inference across verse pairs/chains
  4. Novelty Scoring — rank deductions by how surprising/non-obvious they are
  5. Infinite Loop — continuously discover, score, and log deductions

Key insight: The knowledge graph already encodes thematic proximity.
A 2-3 hop path between two verses that are NOT directly connected
represents a hidden logical chain — a syllogistic deduction waiting
to be surfaced.
"""

import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import spacy

csv.field_size_limit(sys.maxsize)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Proposition:
    """A subject-relation-object triple extracted from a verse."""
    subject: str
    relation: str
    object: str
    verse_id: str
    confidence: float = 1.0

    def __str__(self):
        return f"{self.subject} —[{self.relation}]→ {self.object}"

    def as_tuple(self):
        return (self.subject.lower(), self.relation.lower(), self.object.lower())


@dataclass
class Deduction:
    """A novel conclusion derived from combining two or more verse-axioms."""
    premise_verses: list          # verse IDs that form the premises
    premise_propositions: list    # the Proposition objects used
    conclusion: str               # natural language conclusion
    conclusion_triple: tuple      # (subject, relation, object)
    bridge_keywords: list         # keywords connecting the premises
    rule: str                     # which deductive rule was applied
    novelty_score: float = 0.0   # how novel/surprising is this?
    coherence_score: float = 0.0 # how logically coherent?
    graph_distance: int = 0      # hops between premise verses in graph

    def __str__(self):
        premises = " + ".join(f"[{v}]" for v in self.premise_verses)
        return f"{premises} ⟹ {self.conclusion} (novelty: {self.novelty_score:.2f})"


# ══════════════════════════════════════════════════════════════════════════════
# Proposition Extractor
# ══════════════════════════════════════════════════════════════════════════════

class PropositionExtractor:
    """Extract subject-relation-object triples from verse text using spaCy."""

    # Canonical subjects in the Quran
    CANONICAL_SUBJECTS = {
        "god": "GOD", "allah": "GOD", "lord": "GOD", "he": None,  # context-dependent
        "believers": "BELIEVERS", "righteous": "BELIEVERS",
        "disbelievers": "DISBELIEVERS", "wicked": "DISBELIEVERS",
        "humans": "HUMANS", "people": "HUMANS", "human": "HUMANS", "man": "HUMANS",
        "moses": "MOSES", "abraham": "ABRAHAM", "jesus": "JESUS",
        "noah": "NOAH", "joseph": "JOSEPH", "pharaoh": "PHARAOH",
        "angels": "ANGELS", "devil": "DEVIL", "satan": "DEVIL",
        "messengers": "MESSENGERS", "prophets": "PROPHETS",
        "heaven": "PARADISE", "paradise": "PARADISE", "garden": "PARADISE",
        "hell": "HELL", "fire": "HELL", "hellfire": "HELL",
    }

    # Relation categories
    RELATION_PATTERNS = {
        "commands": r"\b(shall|must|observe|worship|believe|obey|fear|follow|pray|fast|give)\b",
        "prohibits": r"\b(prohibit|forbid|forbidden|shall not|do not|never|avoid)\b",
        "promises": r"\b(reward|paradise|garden|forgive|mercy|bless|save|redeem)\b",
        "warns": r"\b(punish|hell|fire|retribution|doom|wrath|torment|curse)\b",
        "creates": r"\b(create|made|fashion|form|bring|originate)\b",
        "describes": r"\b(is|are|was|were|has|have|possess)\b",
        "decrees": r"\b(decree|ordain|prescribe|appoint|decide|judge)\b",
        "reveals": r"\b(reveal|send down|inspire|proclaim|recite|scripture)\b",
    }

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, verse_id: str, text: str) -> list:
        """Extract propositions from a verse. Returns list of Proposition."""
        doc = self.nlp(text)
        propositions = []

        # Method 1: SVO extraction from dependency parse
        for sent in doc.sents:
            svos = self._extract_svo(sent)
            for subj, verb, obj in svos:
                # Canonicalize subject
                canon_subj = self._canonicalize(subj)
                # Categorize relation
                relation = self._categorize_relation(verb, sent.text)
                # Build proposition
                if canon_subj and relation:
                    prop = Proposition(
                        subject=canon_subj,
                        relation=relation,
                        object=obj,
                        verse_id=verse_id,
                        confidence=0.8,
                    )
                    propositions.append(prop)

        # Method 2: Pattern-based extraction for imperative/declarative forms
        pattern_props = self._extract_patterns(verse_id, text)
        propositions.extend(pattern_props)

        # Deduplicate
        seen = set()
        unique = []
        for p in propositions:
            key = p.as_tuple()
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique

    def _extract_svo(self, sent):
        """Extract subject-verb-object triples from a spaCy sentence."""
        svos = []
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                # Find subject
                subj = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj = self._get_span_text(child)
                        break

                # Find object
                obj = None
                for child in token.children:
                    if child.dep_ in ("dobj", "attr", "pobj", "oprd"):
                        obj = self._get_span_text(child)
                        break
                    elif child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                obj = f"{child.text} {self._get_span_text(grandchild)}"
                                break

                if subj and obj:
                    svos.append((subj, token.lemma_, obj))
        return svos

    def _get_span_text(self, token):
        """Get the full noun phrase for a token."""
        subtree = list(token.subtree)
        # Limit to reasonable length
        if len(subtree) > 8:
            subtree = subtree[:8]
        return " ".join(t.text for t in subtree)

    def _canonicalize(self, subject_text: str) -> str:
        """Map subject text to canonical entity."""
        lower = subject_text.lower().strip()
        for key, canon in self.CANONICAL_SUBJECTS.items():
            if key in lower:
                return canon if canon else subject_text
        # Return cleaned version for unknown subjects
        if len(lower) > 2:
            return lower.title()
        return None

    def _categorize_relation(self, verb: str, context: str) -> str:
        """Categorize a verb into a relation type."""
        context_lower = context.lower()
        for rel_type, pattern in self.RELATION_PATTERNS.items():
            if re.search(pattern, verb, re.IGNORECASE):
                return rel_type
            if re.search(pattern, context_lower):
                return rel_type
        return "states"  # default relation

    def _extract_patterns(self, verse_id: str, text: str) -> list:
        """Pattern-based proposition extraction for common Quranic structures."""
        props = []
        text_lower = text.lower()

        # "GOD [verb]s [object]"
        m = re.search(r'god\s+(create|forgive|punish|reward|decree|prohibit|command|reveal|bless|guide)s?\s+(.{3,50}?)[\.,;]', text_lower)
        if m:
            props.append(Proposition(
                subject="GOD",
                relation=m.group(1) + "s",
                object=m.group(2).strip(),
                verse_id=verse_id,
                confidence=0.9,
            ))

        # "Those who [verb] shall [consequence]"
        m = re.search(r'those who\s+(.{3,40}?)\s+(?:shall|will)\s+(.{3,60}?)[\.,;]', text_lower)
        if m:
            props.append(Proposition(
                subject=f"those who {m.group(1).strip()}",
                relation="will_receive",
                object=m.group(2).strip(),
                verse_id=verse_id,
                confidence=0.85,
            ))

        # "He prohibits/forbids [object]"
        m = re.search(r'(?:he|god)\s+(?:only\s+)?(?:prohibit|forbid)s?\s+(.{3,80}?)[\.,;]', text_lower)
        if m:
            props.append(Proposition(
                subject="GOD",
                relation="prohibits",
                object=m.group(1).strip(),
                verse_id=verse_id,
                confidence=0.9,
            ))

        # "[Subject] is/are [predicate]"
        m = re.search(r'(god|believers|disbelievers|humans?|people)\s+(?:is|are)\s+(.{3,50}?)[\.,;]', text_lower)
        if m:
            subj = self._canonicalize(m.group(1))
            if subj:
                props.append(Proposition(
                    subject=subj,
                    relation="is",
                    object=m.group(2).strip(),
                    verse_id=verse_id,
                    confidence=0.8,
                ))

        return props


# ══════════════════════════════════════════════════════════════════════════════
# Graph Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_graph():
    """Load graph data from CSVs."""
    verses = {}
    with open(os.path.join(DATA_DIR, "verse_nodes.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            verses[row["verseId"]] = {
                "text": row["text"],
                "surah": row["surah"],
                "surahName": row["surahName"],
            }

    keyword_verses = defaultdict(list)
    verse_keywords = defaultdict(list)
    with open(os.path.join(DATA_DIR, "verse_keyword_rels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid, kw, score = row["verseId"], row["keyword"], float(row["score"])
            keyword_verses[kw].append((vid, score))
            verse_keywords[vid].append((kw, score))

    related = defaultdict(list)
    with open(os.path.join(DATA_DIR, "verse_related_rels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v1, v2, score = row["verseId1"], row["verseId2"], float(row["score"])
            related[v1].append((v2, score))
            related[v2].append((v1, score))

    return {
        "verses": verses,
        "keyword_verses": keyword_verses,
        "verse_keywords": verse_keywords,
        "related": related,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Syllogistic Deduction Rules
# ══════════════════════════════════════════════════════════════════════════════

class SyllogisticEngine:
    """
    Apply deductive inference rules across verse propositions.

    Core rules:
    1. TRANSITIVE PREDICATE: A→B, B→C ⟹ A→C
       "GOD commands prayer" + "Prayer leads to righteousness" ⟹ "GOD's command leads to righteousness"

    2. SHARED SUBJECT SYNTHESIS: A→B, A→C ⟹ A→(B∧C)
       "GOD forgives sins" + "GOD is merciful" ⟹ "GOD forgives sins because He is merciful"

    3. CONTRAPOSITIVE: A→B ⟹ ¬B→¬A
       "Believers will be saved" ⟹ "Those not saved are not believers"

    4. THEMATIC BRIDGE: If verse A and verse C are connected through verse B (2+ hops),
       the bridge keywords reveal an implicit logical connection.

    5. UNIVERSAL-PARTICULAR: "All X do Y" + "Z is X" ⟹ "Z does Y"
    """

    def __init__(self, graph, propositions_by_verse, all_propositions):
        self.graph = graph
        self.props_by_verse = propositions_by_verse
        self.all_props = all_propositions
        # Index propositions by subject and object for fast lookup
        self.by_subject = defaultdict(list)
        self.by_object = defaultdict(list)
        self.by_relation = defaultdict(list)
        for p in all_propositions:
            self.by_subject[p.subject.lower()].append(p)
            self.by_object[p.object.lower()[:30]].append(p)
            self.by_relation[p.relation].append(p)

    def find_transitive_chains(self, max_chains=500):
        """Rule 1: A→B, B→C ⟹ A→C. Find via graph 2-hop paths."""
        deductions = []

        for verse_id, neighbors in self.graph["related"].items():
            if verse_id not in self.props_by_verse:
                continue
            props_a = self.props_by_verse[verse_id]

            for neighbor_id, edge_score in neighbors[:10]:  # top 10 neighbors
                if neighbor_id not in self.props_by_verse:
                    continue

                # Find 2-hop connections through this neighbor
                for hop2_id, hop2_score in self.graph["related"].get(neighbor_id, [])[:10]:
                    if hop2_id == verse_id or hop2_id not in self.props_by_verse:
                        continue

                    # Check if verse_id and hop2_id are NOT directly connected
                    direct_neighbors = set(n for n, _ in self.graph["related"].get(verse_id, []))
                    if hop2_id in direct_neighbors:
                        continue  # Skip — already directly connected, not novel

                    props_c = self.props_by_verse[hop2_id]

                    # Find bridge keywords
                    kw_a = set(kw for kw, _ in self.graph["verse_keywords"].get(verse_id, []))
                    kw_b = set(kw for kw, _ in self.graph["verse_keywords"].get(neighbor_id, []))
                    kw_c = set(kw for kw, _ in self.graph["verse_keywords"].get(hop2_id, []))
                    bridge_ab = kw_a & kw_b
                    bridge_bc = kw_b & kw_c

                    if not bridge_ab or not bridge_bc:
                        continue

                    # Construct deduction from best proposition pair
                    for pa in props_a[:2]:
                        for pc in props_c[:2]:
                            conclusion = self._make_transitive_conclusion(pa, pc, bridge_ab, bridge_bc)
                            if conclusion:
                                d = Deduction(
                                    premise_verses=[verse_id, neighbor_id, hop2_id],
                                    premise_propositions=[pa, pc],
                                    conclusion=conclusion,
                                    conclusion_triple=(pa.subject, "relates_to", pc.object),
                                    bridge_keywords=list(bridge_ab | bridge_bc),
                                    rule="transitive_chain",
                                    graph_distance=2,
                                )
                                deductions.append(d)

                    if len(deductions) >= max_chains:
                        return deductions

        return deductions

    def find_shared_subject_syntheses(self, max_results=500):
        """Rule 2: A→B, A→C ⟹ A→(B∧C). Same subject, combine predicates."""
        deductions = []

        for subject, props in self.by_subject.items():
            if len(props) < 2:
                continue

            # Group by verse to avoid trivial within-verse combinations
            by_verse = defaultdict(list)
            for p in props:
                by_verse[p.verse_id].append(p)

            verse_ids = list(by_verse.keys())
            for i in range(len(verse_ids)):
                for j in range(i + 1, min(i + 5, len(verse_ids))):
                    v1, v2 = verse_ids[i], verse_ids[j]

                    # Check graph distance
                    direct = set(n for n, _ in self.graph["related"].get(v1, []))
                    if v2 not in direct:
                        # 2-hop check
                        reachable = set()
                        for n, _ in self.graph["related"].get(v1, []):
                            for n2, _ in self.graph["related"].get(n, []):
                                reachable.add(n2)
                        if v2 not in reachable:
                            continue

                    for p1 in by_verse[v1][:2]:
                        for p2 in by_verse[v2][:2]:
                            if p1.relation == p2.relation and p1.object != p2.object:
                                conclusion = f"{p1.subject} {p1.relation} both {p1.object} AND {p2.object}"
                                d = Deduction(
                                    premise_verses=[v1, v2],
                                    premise_propositions=[p1, p2],
                                    conclusion=conclusion,
                                    conclusion_triple=(p1.subject, p1.relation, f"{p1.object} + {p2.object}"),
                                    bridge_keywords=[],
                                    rule="shared_subject_synthesis",
                                    graph_distance=1 if v2 in direct else 2,
                                )
                                deductions.append(d)

                            elif p1.relation != p2.relation:
                                conclusion = f"{p1.subject}: [{v1}] says '{p1.relation} {p1.object}', [{v2}] says '{p2.relation} {p2.object}' — combined: {p1.subject} {p1.relation} {p1.object} while also {p2.relation} {p2.object}"
                                d = Deduction(
                                    premise_verses=[v1, v2],
                                    premise_propositions=[p1, p2],
                                    conclusion=conclusion,
                                    conclusion_triple=(p1.subject, "combined", f"{p1.object} & {p2.object}"),
                                    bridge_keywords=[],
                                    rule="shared_subject_multi_predicate",
                                    graph_distance=1 if v2 in direct else 2,
                                )
                                deductions.append(d)

                    if len(deductions) >= max_results:
                        return deductions

        return deductions

    def find_thematic_bridges(self, max_results=500):
        """
        Rule 4: Find 3-hop paths that connect thematically distant verses.
        The bridge reveals a hidden logical chain.
        """
        deductions = []

        # Sample starting verses (pick from different surahs)
        verse_ids = list(self.props_by_verse.keys())
        if len(verse_ids) > 200:
            import random
            verse_ids = random.sample(verse_ids, 200)

        for start_id in verse_ids:
            if start_id not in self.graph["related"]:
                continue

            start_surah = self.graph["verses"].get(start_id, {}).get("surah", "")

            # 3-hop BFS
            hop1 = [(n, s) for n, s in self.graph["related"].get(start_id, [])[:8]]
            for h1_id, h1_score in hop1:
                hop2 = [(n, s) for n, s in self.graph["related"].get(h1_id, [])[:6]
                        if n != start_id]
                for h2_id, h2_score in hop2:
                    hop3 = [(n, s) for n, s in self.graph["related"].get(h2_id, [])[:4]
                            if n != start_id and n != h1_id]
                    for h3_id, h3_score in hop3:
                        # Only interesting if start and end are in different surahs
                        end_surah = self.graph["verses"].get(h3_id, {}).get("surah", "")
                        if start_surah == end_surah:
                            continue

                        # Must NOT be directly connected
                        direct = set(n for n, _ in self.graph["related"].get(start_id, []))
                        if h3_id in direct:
                            continue

                        if start_id not in self.props_by_verse or h3_id not in self.props_by_verse:
                            continue

                        # Build the bridge chain keywords
                        kw_start = set(kw for kw, _ in self.graph["verse_keywords"].get(start_id, []))
                        kw_h1 = set(kw for kw, _ in self.graph["verse_keywords"].get(h1_id, []))
                        kw_h2 = set(kw for kw, _ in self.graph["verse_keywords"].get(h2_id, []))
                        kw_end = set(kw for kw, _ in self.graph["verse_keywords"].get(h3_id, []))

                        chain = []
                        b1 = kw_start & kw_h1
                        b2 = kw_h1 & kw_h2
                        b3 = kw_h2 & kw_end
                        if b1: chain.extend(list(b1)[:2])
                        if b2: chain.extend(list(b2)[:2])
                        if b3: chain.extend(list(b3)[:2])

                        if len(chain) < 2:
                            continue

                        pa = self.props_by_verse[start_id][0]
                        pc = self.props_by_verse[h3_id][0]

                        conclusion = (
                            f"[{start_id}] ({pa}) connects to [{h3_id}] ({pc}) "
                            f"through thematic chain: {' → '.join(chain)}. "
                            f"Path: [{start_id}] → [{h1_id}] → [{h2_id}] → [{h3_id}]"
                        )

                        d = Deduction(
                            premise_verses=[start_id, h1_id, h2_id, h3_id],
                            premise_propositions=[pa, pc],
                            conclusion=conclusion,
                            conclusion_triple=(pa.subject, "bridges_to", pc.subject),
                            bridge_keywords=chain,
                            rule="thematic_bridge_3hop",
                            graph_distance=3,
                        )
                        deductions.append(d)

                        if len(deductions) >= max_results:
                            return deductions

        return deductions

    def _make_transitive_conclusion(self, prop_a, prop_c, bridge_ab, bridge_bc):
        """Construct a natural language transitive conclusion."""
        if not prop_a or not prop_c:
            return None

        bridge_word = list(bridge_ab)[0] if bridge_ab else "theme"
        bridge_word2 = list(bridge_bc)[0] if bridge_bc else "concept"

        return (
            f"BECAUSE {prop_a.subject} {prop_a.relation} {prop_a.object} [{prop_a.verse_id}], "
            f"and this connects via '{bridge_word}' → '{bridge_word2}', "
            f"THEREFORE {prop_a.subject}'s {prop_a.relation} relates to {prop_c.object} [{prop_c.verse_id}]"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Novelty Scorer
# ══════════════════════════════════════════════════════════════════════════════

class NoveltyScorer:
    """Score how novel/surprising a deduction is."""

    def __init__(self, graph, all_propositions):
        self.graph = graph
        # Index known facts for novelty checking
        self.known_triples = set()
        for p in all_propositions:
            self.known_triples.add(p.as_tuple())

        # Surah name map for nice output
        self.surah_names = {}
        for vid, data in graph["verses"].items():
            s = data.get("surah", "")
            self.surah_names[s] = data.get("surahName", f"Surah {s}")

    def score(self, deduction: Deduction) -> float:
        """Compute novelty score (0-100)."""
        scores = []

        # Factor 1: Graph distance (farther = more novel)
        dist_score = min(100, deduction.graph_distance * 30)
        scores.append(("distance", dist_score, 0.25))

        # Factor 2: Cross-surah (different surahs = more novel)
        surahs = set()
        for vid in deduction.premise_verses:
            v = self.graph["verses"].get(vid, {})
            surahs.add(v.get("surah", ""))
        cross_surah_score = 100 if len(surahs) > 1 else 20
        scores.append(("cross_surah", cross_surah_score, 0.20))

        # Factor 3: Bridge keyword diversity
        n_bridges = len(deduction.bridge_keywords)
        bridge_score = min(100, n_bridges * 25)
        scores.append(("bridge_diversity", bridge_score, 0.15))

        # Factor 4: Conclusion not already a known triple
        ct = deduction.conclusion_triple
        known = (ct[0].lower(), ct[1].lower(), ct[2].lower()[:30]) in self.known_triples
        novelty_score = 20 if known else 100
        scores.append(("not_known", novelty_score, 0.25))

        # Factor 5: Different subjects in premises (connecting different entities)
        subjects = set()
        for p in deduction.premise_propositions:
            subjects.add(p.subject)
        subject_diversity = 100 if len(subjects) > 1 else 40
        scores.append(("subject_diversity", subject_diversity, 0.15))

        composite = sum(s * w for _, s, w in scores)
        return round(composite, 2)

    def score_all(self, deductions: list) -> list:
        """Score all deductions and sort by novelty."""
        for d in deductions:
            d.novelty_score = self.score(d)
            # Coherence is based on edge weights along the path
            if d.graph_distance > 0:
                d.coherence_score = round(len(d.bridge_keywords) / max(1, d.graph_distance) * 50, 2)
        deductions.sort(key=lambda d: d.novelty_score, reverse=True)
        return deductions


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction_pipeline(max_verses=None):
    """Run the full proposition extraction and deduction pipeline."""
    print("Loading graph...")
    graph = load_graph()
    print(f"  {len(graph['verses'])} verses, "
          f"{len(graph['keyword_verses'])} keywords, "
          f"{sum(len(v) for v in graph['related'].values()) // 2} edges")

    print("\nExtracting propositions from verses...")
    extractor = PropositionExtractor()

    all_propositions = []
    props_by_verse = {}
    verse_ids = list(graph["verses"].keys())
    if max_verses:
        verse_ids = verse_ids[:max_verses]

    for i, vid in enumerate(verse_ids):
        text = graph["verses"][vid]["text"]
        props = extractor.extract(vid, text)
        if props:
            props_by_verse[vid] = props
            all_propositions.extend(props)
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(verse_ids)} verses, {len(all_propositions)} propositions so far")

    print(f"  Total: {len(all_propositions)} propositions from {len(props_by_verse)} verses")

    print("\nRunning syllogistic deduction engine...")
    engine = SyllogisticEngine(graph, props_by_verse, all_propositions)

    print("  Finding transitive chains...")
    transitive = engine.find_transitive_chains(max_chains=300)
    print(f"    {len(transitive)} transitive deductions")

    print("  Finding shared-subject syntheses...")
    shared = engine.find_shared_subject_syntheses(max_results=300)
    print(f"    {len(shared)} shared-subject deductions")

    print("  Finding thematic bridges...")
    bridges = engine.find_thematic_bridges(max_results=300)
    print(f"    {len(bridges)} thematic bridge deductions")

    all_deductions = transitive + shared + bridges
    print(f"\n  Total deductions: {len(all_deductions)}")

    print("\nScoring for novelty...")
    scorer = NoveltyScorer(graph, all_propositions)
    scored = scorer.score_all(all_deductions)

    return {
        "graph": graph,
        "propositions": all_propositions,
        "props_by_verse": props_by_verse,
        "deductions": scored,
        "engine": engine,
        "scorer": scorer,
    }


def print_top_deductions(deductions, n=30):
    """Print the most novel deductions."""
    print(f"\n{'='*80}")
    print(f"  TOP {n} MOST NOVEL DEDUCTIONS")
    print(f"{'='*80}")

    for i, d in enumerate(deductions[:n]):
        print(f"\n  #{i+1} [novelty: {d.novelty_score:.1f}] [{d.rule}]")
        print(f"  Premises: {', '.join(f'[{v}]' for v in d.premise_verses)}")
        print(f"  Bridge:   {', '.join(d.bridge_keywords) if d.bridge_keywords else 'N/A'}")
        print(f"  Conclusion: {d.conclusion[:200]}")
        print(f"  {'─'*76}")


if __name__ == "__main__":
    result = run_extraction_pipeline()
    print_top_deductions(result["deductions"], n=30)

    # Save deductions to JSON
    _autoresearch_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(_autoresearch_dir, "deductions.json")
    deductions_data = []
    for d in result["deductions"][:200]:
        deductions_data.append({
            "rank": len(deductions_data) + 1,
            "novelty_score": d.novelty_score,
            "coherence_score": d.coherence_score,
            "rule": d.rule,
            "premise_verses": d.premise_verses,
            "bridge_keywords": d.bridge_keywords,
            "conclusion": d.conclusion,
            "graph_distance": d.graph_distance,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deductions_data, f, indent=2, ensure_ascii=False)
    print(f"\nTop 200 deductions saved to {output_path}")
