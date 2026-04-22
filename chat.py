"""
Quran Graph Chat — conversational LLM agent with graph tools.

Claude drives the exploration. Instead of a fixed keyword→query→answer pipeline,
Claude decides which tools to call, how deep to traverse, and when it has enough
context to answer well.

Usage:
    py chat.py

Tools available to Claude:
    search_keyword(keyword)              → verses mentioning a keyword, by surah
    get_verse(verse_id)                  → verse text + connected verses + shared keywords
    traverse_topic(keywords, hops)       → multi-keyword search + graph traversal
    find_path(verse_id_1, verse_id_2)    → shortest thematic path between two verses
    explore_surah(surah_number)          → all verses in a surah + cross-surah connections
"""

import json
import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
import anthropic
import config as cfg

# ── env loading ────────────────────────────────────────────────────────────────

def _load_env(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ[k.strip()] = v.strip()

load_dotenv()
_load_env()

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_TOKEN = os.getenv("ANTHROPIC_OAUTH_TOKEN", "")
MODEL          = cfg.llm_model()

# ── reuse lemmatizer ───────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_graph import tokenize_and_lemmatize

# ── semantic model (loaded once, reused across requests) ───────────────────────

_sem_model = None

def _get_sem_model():
    global _sem_model
    if _sem_model is None:
        from sentence_transformers import SentenceTransformer
        _sem_model = SentenceTransformer(cfg.embedding_model())
    return _sem_model

# ── graph tool implementations ─────────────────────────────────────────────────

def tool_search_keyword(session, keyword: str) -> dict:
    """Find ALL verses mentioning a keyword, grouped by surah."""
    lemmas = tokenize_and_lemmatize(keyword)
    if not lemmas:
        return {"error": f"'{keyword}' is a stopword or too short — try a different word"}
    kw = lemmas[0]

    exists = session.run("MATCH (k:Keyword {keyword: $kw}) RETURN k.keyword AS kw", kw=kw).single()
    if not exists:
        # Fuzzy fallback: suggest similar keywords
        similar = [r["kw"] for r in session.run(
            "MATCH (k:Keyword) WHERE k.keyword STARTS WITH $prefix RETURN k.keyword AS kw LIMIT $lim",
            prefix=kw[:cfg.search_keyword_fuzzy_prefix()], lim=cfg.search_keyword_fuzzy_limit()
        )]
        return {
            "error": f"Keyword '{kw}' not in graph",
            "suggestions": similar,
            "tip": "Try one of the suggested keywords above"
        }

    # No LIMIT — return every verse that mentions this keyword
    rows = list(session.run("""
        MATCH (k:Keyword {keyword: $kw})<-[r:MENTIONS]-(v:Verse)
        RETURN v.surah AS surah, v.surahName AS surahName,
               v.verseId AS verseId, v.text AS text, v.arabicText AS arabic,
               r.score AS score
        ORDER BY r.score DESC
    """, kw=kw))

    by_surah = {}
    for row in rows:
        sname = f"Surah {row['surah']}: {row['surahName']}"
        by_surah.setdefault(sname, []).append({
            "verse_id": row["verseId"],
            "score": round(row["score"], 4),
            "text": row["text"],
            "arabic_text": row["arabic"] or "",
        })

    return {
        "keyword": kw,
        "total_verses": len(rows),
        "by_surah": by_surah
    }


def tool_get_verse(session, verse_id: str) -> dict:
    """Get a verse's full text, its keywords, and all directly connected verses."""
    verse_id = verse_id.strip().replace(" ", ":")
    row = session.run("MATCH (v:Verse {verseId: $id}) RETURN v", id=verse_id).single()
    if not row:
        return {"error": f"Verse [{verse_id}] not found. Format: surah:verse e.g. 2:255"}

    v = row["v"]
    keywords = [r["kw"] for r in session.run("""
        MATCH (v:Verse {verseId: $id})-[r:MENTIONS]->(k:Keyword)
        RETURN k.keyword AS kw ORDER BY r.score DESC LIMIT $lim
    """, id=verse_id, lim=cfg.get_verse_keyword_limit())]

    neighbours = list(session.run("""
        MATCH (v:Verse {verseId: $id})-[r:RELATED_TO]-(other:Verse)
        RETURN other.verseId AS otherId, other.surahName AS surahName,
               other.text AS text, r.score AS score
        ORDER BY r.score DESC LIMIT $lim
    """, id=verse_id, lim=cfg.get_verse_neighbour_limit()))

    # Batch: fetch shared keywords for ALL neighbours in one query
    neighbour_ids = [r["otherId"] for r in neighbours]
    shared_map = {}
    if neighbour_ids:
        shared_rows = session.run("""
            UNWIND $otherIds AS oid
            MATCH (v1:Verse {verseId: $v1})-[:MENTIONS]->(k:Keyword)<-[:MENTIONS]-(v2:Verse {verseId: oid})
            WITH oid, collect(k.keyword)[..$skLim] AS kws
            RETURN oid, kws
        """, v1=verse_id, otherIds=neighbour_ids, skLim=cfg.get_verse_shared_kw_limit())
        for sr in shared_rows:
            shared_map[sr["oid"]] = sr["kws"]

    connected = [{
        "verse_id": r["otherId"],
        "surah_name": r["surahName"],
        "text": r["text"],
        "shared_keywords": shared_map.get(r["otherId"], []),
        "connection_score": round(r["score"], 4)
    } for r in neighbours]

    # Fetch top Arabic roots for this verse
    arabic_roots = [{"root": r["root"], "gloss": r["gloss"] or "", "forms": r["forms"] or []}
                    for r in session.run("""
        MATCH (v:Verse {verseId: $id})-[m:MENTIONS_ROOT]->(r:ArabicRoot)
        RETURN r.root AS root, r.gloss AS gloss, m.forms AS forms
        ORDER BY m.count DESC LIMIT 10
    """, id=verse_id)]

    # Typed edges summary (SUPPORTS, ELABORATES, etc.)
    typed_rows = list(session.run("""
        MATCH (v:Verse {verseId: $id})-[r]-(other:Verse)
        WHERE type(r) IN ['SUPPORTS','ELABORATES','QUALIFIES','CONTRASTS','REPEATS']
        RETURN other.verseId AS otherId, type(r) AS relType,
               r.score AS score, r.confidence AS confidence
        ORDER BY r.score DESC LIMIT 15
    """, id=verse_id))
    typed_edges = {}
    for tr in typed_rows:
        typed_edges.setdefault(tr["relType"], []).append({
            "verse_id": tr["otherId"],
            "score": round(tr["score"], 4) if tr["score"] else None,
        })

    return {
        "verse_id": verse_id,
        "surah": v["surah"],
        "surah_name": v["surahName"],
        "text": v["text"],
        "arabic_text": v.get("arabicText", "") or "",
        "arabic_roots": arabic_roots,
        "keywords": keywords,
        "connected_verses": connected,
        "typed_edges": typed_edges,
    }


def tool_traverse_topic(session, keywords: list[str], hops: int = 1) -> dict:
    """Multi-keyword search + graph traversal to map a topic."""
    hops = max(1, min(hops, 2))
    lemmas = list(set(
        lemma
        for kw in keywords
        for lemma in tokenize_and_lemmatize(kw)
    ))
    if not lemmas:
        return {"error": "All keywords were stopwords. Use more specific terms."}

    # Direct matches — no limit, return ALL verses matching any of the keywords
    direct = list(session.run("""
        UNWIND $keywords AS kw
        MATCH (k:Keyword {keyword: kw})<-[r:MENTIONS]-(v:Verse)
        WITH v, sum(r.score) AS total, collect(kw) AS matched
        RETURN v.verseId AS verseId, v.surahName AS surahName,
               v.text AS text, total, matched
        ORDER BY total DESC
    """, keywords=lemmas))

    direct_ids = [r["verseId"] for r in direct]
    direct_results = [{
        "verse_id": r["verseId"],
        "surah_name": r["surahName"],
        "text": r["text"],
        "matched_keywords": r["matched"],
        "score": round(r["total"], 4)
    } for r in direct]

    seed_ids = direct_ids[:cfg.traverse_seed_limit()]

    # 1-hop — top 60 thematically connected verses not already in direct matches
    hop1 = list(session.run("""
        UNWIND $seedIds AS sid
        MATCH (seed:Verse {verseId: sid})-[r:RELATED_TO]-(n:Verse)
        WHERE NOT n.verseId IN $allDirectIds
        WITH n, sum(r.score) AS score, collect(sid) AS via
        RETURN n.verseId AS verseId, n.surahName AS surahName,
               n.text AS text, score, via
        ORDER BY score DESC LIMIT $lim
    """, seedIds=seed_ids, allDirectIds=direct_ids, lim=cfg.traverse_hop1_limit()))

    hop1_ids = [r["verseId"] for r in hop1]
    hop1_results = [{
        "verse_id": r["verseId"],
        "surah_name": r["surahName"],
        "text": r["text"],
        "connected_via": r["via"],
        "score": round(r["score"], 4)
    } for r in hop1]

    hop2_results = []
    if hops >= 2:
        exclude = set(direct_ids + hop1_ids)
        hop2 = list(session.run("""
            UNWIND $h1Ids AS h1
            MATCH (h1v:Verse {verseId: h1})-[:RELATED_TO]-(h2:Verse)
            WHERE NOT h2.verseId IN $exclude
            WITH h2, count(*) AS connections
            RETURN h2.verseId AS verseId, h2.surahName AS surahName,
                   h2.text AS text, connections
            ORDER BY connections DESC LIMIT $lim
        """, h1Ids=hop1_ids, exclude=list(exclude), lim=cfg.traverse_hop2_limit()))
        hop2_results = [{
            "verse_id": r["verseId"],
            "surah_name": r["surahName"],
            "text": r["text"]
        } for r in hop2]

    return {
        "keywords_used": lemmas,
        "direct_matches": direct_results,
        "hop_1_connections": hop1_results,
        "hop_2_connections": hop2_results,
        "total_verses_found": len(direct_results) + len(hop1_results) + len(hop2_results)
    }


def tool_find_path(session, verse_id_1: str, verse_id_2: str) -> dict:
    """Find the shortest thematic path between two verses through the graph."""
    v1 = verse_id_1.strip().replace(" ", ":")
    v2 = verse_id_2.strip().replace(" ", ":")

    for vid in [v1, v2]:
        if not session.run("MATCH (v:Verse {verseId: $id}) RETURN v", id=vid).single():
            return {"error": f"Verse [{vid}] not found"}

    result = session.run("""
        MATCH (v1:Verse {verseId: $v1}), (v2:Verse {verseId: $v2}),
              path = shortestPath((v1)-[:RELATED_TO*..%d]-(v2))""" % cfg.find_path_max_depth() + """
        RETURN path, length(path) AS hops LIMIT 1
    """, v1=v1, v2=v2).single()

    if not result:
        return {"error": f"No path found between [{v1}] and [{v2}] within {cfg.find_path_max_depth()} hops"}

    nodes = result["path"].nodes

    # Batch: fetch bridge keywords for ALL consecutive pairs in one query
    pairs = [{"a": nodes[i]["verseId"], "b": nodes[i + 1]["verseId"]}
             for i in range(len(nodes) - 1)]
    bridge_map = {}
    if pairs:
        bridge_rows = session.run("""
            UNWIND $pairs AS pair
            MATCH (a:Verse {verseId: pair.a})-[:MENTIONS]->(k:Keyword)<-[:MENTIONS]-(b:Verse {verseId: pair.b})
            WITH pair.a AS fromId, pair.b AS toId, collect(k.keyword)[..$bkLim] AS kws
            RETURN fromId, toId, kws
        """, pairs=pairs, bkLim=cfg.find_path_bridge_kw_limit())
        for br in bridge_rows:
            bridge_map[(br["fromId"], br["toId"])] = br["kws"]

    steps = []
    for i, node in enumerate(nodes):
        step = {
            "step": i + 1,
            "verse_id": node["verseId"],
            "surah_name": node["surahName"],
            "text": node["text"]
        }
        if i < len(nodes) - 1:
            step["bridge_keywords"] = bridge_map.get(
                (node["verseId"], nodes[i + 1]["verseId"]), [])
        steps.append(step)

    return {
        "from": v1,
        "to": v2,
        "hops": result["hops"],
        "path": steps
    }


def tool_explore_surah(session, surah_number: int) -> dict:
    """Get all verses in a surah and its top cross-surah thematic connections."""
    verses = list(session.run("""
        MATCH (v:Verse {surah: $s})
        RETURN v.verseId AS verseId, v.text AS text, v.verseNum AS verseNum
        ORDER BY v.verseNum
    """, s=surah_number))

    if not verses:
        return {"error": f"Surah {surah_number} not found (valid range: 1-114)"}

    surah_name = session.run(
        "MATCH (v:Verse {surah: $s}) RETURN v.surahName AS name LIMIT 1", s=surah_number
    ).single()["name"]

    verse_ids = [v["verseId"] for v in verses]
    cross = list(session.run("""
        UNWIND $vids AS vid
        MATCH (v:Verse {verseId: vid})-[:RELATED_TO]-(other:Verse)
        WHERE other.surah <> $s
        WITH other.surah AS otherSurah, other.surahName AS otherName, count(*) AS connections
        RETURN otherSurah, otherName, connections
        ORDER BY connections DESC LIMIT $lim
    """, vids=verse_ids, s=surah_number, lim=cfg.explore_surah_cross_limit()))

    return {
        "surah": surah_number,
        "surah_name": surah_name,
        "verse_count": len(verses),
        "verses": [{"verse_id": v["verseId"], "text": v["text"]} for v in verses],
        "top_cross_surah_connections": [{
            "surah": r["otherSurah"],
            "surah_name": r["otherName"],
            "connections": r["connections"]
        } for r in cross]
    }


def tool_search_arabic_root(session, root: str) -> dict:
    """Find all verses containing a specific Arabic tri-literal root."""
    root = root.strip()

    # Check if root exists
    row = session.run(
        "MATCH (r:ArabicRoot {root: $root}) RETURN r.root AS root, r.gloss AS gloss, r.verseCount AS vc",
        root=root,
    ).single()

    if not row:
        # Try Buckwalter lookup
        row = session.run(
            "MATCH (r:ArabicRoot {rootBW: $bw}) RETURN r.root AS root, r.gloss AS gloss, r.verseCount AS vc",
            bw=root,
        ).single()

    if not row:
        # Fuzzy: suggest roots starting with same first letter
        similar = [r["root"] + (" (" + r["gloss"] + ")" if r["gloss"] else "")
                   for r in session.run(
            "MATCH (r:ArabicRoot) WHERE r.root STARTS WITH $prefix RETURN r.root, r.gloss LIMIT 8",
            prefix=root[:1] if root else "",
        )]
        return {"error": f"Root '{root}' not found", "suggestions": similar}

    actual_root = row["root"]
    gloss = row["gloss"] or ""

    verses = list(session.run("""
        MATCH (r:ArabicRoot {root: $root})<-[m:MENTIONS_ROOT]-(v:Verse)
        RETURN v.verseId AS verseId, v.surah AS surah, v.surahName AS surahName,
               v.text AS text, v.arabicText AS arabic,
               m.count AS count, m.forms AS forms
        ORDER BY m.count DESC, v.surah, v.verseNum
    """, root=actual_root))

    by_surah = {}
    for r in verses:
        sname = f"Surah {r['surah']}: {r['surahName']}"
        by_surah.setdefault(sname, []).append({
            "verse_id": r["verseId"],
            "text": r["text"],
            "arabic_text": r["arabic"] or "",
            "root_count": r["count"],
            "forms_used": r["forms"] or [],
        })

    return {
        "root": actual_root,
        "gloss": gloss,
        "total_verses": len(verses),
        "by_surah": by_surah,
    }


def tool_compare_arabic_usage(session, root: str) -> dict:
    """Compare how different forms of an Arabic root are used across the Quran."""
    root = root.strip()

    row = session.run(
        "MATCH (r:ArabicRoot {root: $root}) RETURN r.root AS root, r.gloss AS gloss",
        root=root,
    ).single()
    if not row:
        row = session.run(
            "MATCH (r:ArabicRoot {rootBW: $bw}) RETURN r.root AS root, r.gloss AS gloss",
            bw=root,
        ).single()
    if not row:
        return {"error": f"Root '{root}' not found. Use search_arabic_root for suggestions."}

    actual_root = row["root"]
    gloss = row["gloss"] or ""

    verses = list(session.run("""
        MATCH (r:ArabicRoot {root: $root})<-[m:MENTIONS_ROOT]-(v:Verse)
        RETURN v.verseId AS verseId, v.surah AS surah, v.surahName AS surahName,
               v.text AS text, v.arabicText AS arabic, m.forms AS forms
        ORDER BY v.surah, v.verseNum
    """, root=actual_root))

    # Group by surface form
    by_form = {}
    for r in verses:
        for form in (r["forms"] or []):
            by_form.setdefault(form, []).append({
                "verse_id": r["verseId"],
                "surah_name": r["surahName"],
                "text": r["text"],
                "arabic_text": r["arabic"] or "",
            })

    # Limit forms and verses per form
    try:
        max_forms = cfg.raw().get("arabic", {}).get("compare_arabic_usage", {}).get("max_forms_per_root", 20)
        max_per_form = cfg.raw().get("arabic", {}).get("compare_arabic_usage", {}).get("max_verses_per_form", 10)
    except Exception:
        max_forms, max_per_form = 20, 10

    form_summary = []
    for form, vlist in sorted(by_form.items(), key=lambda x: -len(x[1]))[:max_forms]:
        form_summary.append({
            "form": form,
            "total_occurrences": len(vlist),
            "sample_verses": vlist[:max_per_form],
        })

    return {
        "root": actual_root,
        "gloss": gloss,
        "total_verses": len(verses),
        "total_forms": len(by_form),
        "forms": form_summary,
    }


def tool_query_typed_edges(session, verse_id: str, edge_type: str = None) -> dict:
    """Query verses connected by a specific relationship type (SUPPORTS, ELABORATES, etc.)."""
    verse_id = verse_id.strip().replace(" ", ":")
    row = session.run("MATCH (v:Verse {verseId: $id}) RETURN v", id=verse_id).single()
    if not row:
        return {"error": f"Verse [{verse_id}] not found. Format: surah:verse e.g. 2:255"}

    valid_types = ['SUPPORTS', 'ELABORATES', 'QUALIFIES', 'CONTRASTS', 'REPEATS']

    if edge_type:
        edge_type = edge_type.upper()
        if edge_type not in valid_types:
            return {"error": f"Unknown edge type '{edge_type}'. Valid: {valid_types}"}
        rows = list(session.run("""
            MATCH (v:Verse {verseId: $id})-[r:""" + edge_type + """]-(other:Verse)
            RETURN other.verseId AS otherId, other.surahName AS surahName,
                   other.text AS text, other.arabicText AS arabic,
                   r.score AS score, r.confidence AS confidence,
                   $etype AS relType
            ORDER BY r.score DESC LIMIT 12
        """, id=verse_id, etype=edge_type))
    else:
        rows = list(session.run("""
            MATCH (v:Verse {verseId: $id})-[r]-(other:Verse)
            WHERE type(r) IN ['SUPPORTS','ELABORATES','QUALIFIES','CONTRASTS','REPEATS']
            RETURN other.verseId AS otherId, other.surahName AS surahName,
                   other.text AS text, other.arabicText AS arabic,
                   type(r) AS relType, r.score AS score
            ORDER BY r.score DESC LIMIT 20
        """, id=verse_id))

    by_type = {}
    for r in rows:
        rtype = r["relType"]
        by_type.setdefault(rtype, []).append({
            "verse_id": r["otherId"],
            "surah_name": r["surahName"],
            "text": r["text"],
            "arabic_text": r["arabic"] or "",
            "score": round(r["score"], 4) if r["score"] else None,
        })

    return {
        "verse_id": verse_id,
        "edge_type_filter": edge_type,
        "total_results": len(rows),
        "by_type": by_type,
    }


def tool_semantic_search(session, query: str, top_k: int = 40) -> dict:
    """
    Find verses conceptually similar to a query using vector embeddings,
    then enrich each hit with connected graph structure in the same pass.

    Returns per verse:
      - similarity score
      - text + surah context
      - related verses (RELATED_TO edges, top 5)
      - Arabic roots present (top 5)
      - typed edges (SUPPORTS / ELABORATES / QUALIFIES / CONTRASTS / REPEATS)

    This is a VectorCypherRetriever pattern — one tool call gets semantic
    hits plus their graph context, reducing the need for follow-up get_verse
    calls on every hit.
    """
    model = _get_sem_model()
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()

    results = session.run("""
        CALL db.index.vector.queryNodes('verse_embedding', $k, $vec)
        YIELD node, score
        WHERE node.verseId IS NOT NULL
        WITH node, score ORDER BY score DESC
        // Graph enrichment in the same pass (VectorCypherRetriever pattern)
        OPTIONAL MATCH (node)-[:RELATED_TO]-(related:Verse)
        WITH node, score, collect(DISTINCT related.reference)[0..5] AS related_verses
        OPTIONAL MATCH (node)-[:MENTIONS_ROOT]->(root:ArabicRoot)
        WITH node, score, related_verses,
             collect(DISTINCT root.root)[0..5] AS arabic_roots
        OPTIONAL MATCH (node)-[typed:SUPPORTS|ELABORATES|QUALIFIES|CONTRASTS|REPEATS]-(te:Verse)
        WITH node, score, related_verses, arabic_roots,
             [x IN collect(DISTINCT {type: type(typed), target: te.reference})
              WHERE x.type IS NOT NULL AND x.target IS NOT NULL][0..5] AS typed_edges
        RETURN node.verseId AS verseId, node.surahName AS surahName,
               node.surah AS surah, node.text AS text, score,
               related_verses, arabic_roots, typed_edges
    """, k=top_k, vec=vec).data()

    by_surah = {}
    for r in results:
        sname = f"Surah {r['surah']}: {r['surahName']}"
        entry = {
            "verse_id": r["verseId"],
            "similarity": round(r["score"], 4),
            "text": r["text"],
        }
        # Only include enrichment fields when non-empty — keeps payload lean
        if r["related_verses"]:
            entry["related_verses"] = r["related_verses"]
        if r["arabic_roots"]:
            entry["arabic_roots"] = r["arabic_roots"]
        if r["typed_edges"]:
            entry["typed_edges"] = r["typed_edges"]
        by_surah.setdefault(sname, []).append(entry)

    return {
        "query": query,
        "total_verses": len(results),
        "note": ("Results ranked by semantic similarity. Each hit includes its "
                 "related verses, Arabic roots present, and any typed edges "
                 "(SUPPORTS/ELABORATES/QUALIFIES/CONTRASTS/REPEATS) in one pass."),
        "by_surah": by_surah,
    }


# ── etymology tools ─────────────────────────────────────────────────────────────

def tool_lookup_word(session, word: str) -> dict:
    """Look up an Arabic word — root, lemma, pattern, morphology, occurrences."""
    results = session.run("""
        MATCH (w:WordToken)
        WHERE w.arabicClean CONTAINS $word
           OR w.translitBW CONTAINS $word
           OR w.lemma CONTAINS $word
        WITH w LIMIT 200
        OPTIONAL MATCH (w)-[:HAS_LEMMA]->(l:Lemma)
        OPTIONAL MATCH (l)-[:DERIVES_FROM]->(r:ArabicRoot)
        OPTIONAL MATCH (w)-[:FOLLOWS_PATTERN]->(p:MorphPattern)
        RETURN w.tokenId AS tokenId, w.verseId AS verseId,
               w.arabicText AS arabicText, w.pos AS pos,
               w.morphFeatures AS morphFeatures, w.wazn AS wazn,
               l.lemma AS lemma, l.glossEn AS gloss,
               r.root AS root, r.gloss AS rootGloss,
               p.pattern AS pattern, p.label AS patternLabel,
               p.meaningTendency AS meaningTendency
        ORDER BY w.verseId
    """, word=word).data()

    if not results:
        return {"word": word, "found": False, "message": "Word not found in the Quran"}

    # Group by lemma
    by_lemma = {}
    for r in results:
        lem = r['lemma'] or r['arabicText']
        if lem not in by_lemma:
            by_lemma[lem] = {
                "lemma": lem,
                "root": r['root'] or '',
                "rootGloss": r['rootGloss'] or r['gloss'] or '',
                "pattern": r['pattern'] or r['wazn'] or '',
                "patternLabel": r['patternLabel'] or '',
                "meaningTendency": r['meaningTendency'] or '',
                "pos": r['pos'],
                "occurrences": [],
            }
        if len(by_lemma[lem]['occurrences']) < 50:
            by_lemma[lem]['occurrences'].append({
                "verse_id": r['verseId'],
                "token_id": r['tokenId'],
                "arabic": r['arabicText'],
                "morphology": r['morphFeatures'],
            })

    return {
        "word": word,
        "found": True,
        "total_occurrences": len(results),
        "lemmas": list(by_lemma.values()),
    }


def tool_explore_root_family(session, root: str) -> dict:
    """Full derivative tree of a root — all lemmas grouped by pattern."""
    results = session.run("""
        MATCH (r:ArabicRoot)
        WHERE r.root = $root OR r.rootBW = $root
        OPTIONAL MATCH (l:Lemma)-[:DERIVES_FROM]->(r)
        OPTIONAL MATCH (w:WordToken)-[:HAS_LEMMA]->(l)
        OPTIONAL MATCH (w)-[:FOLLOWS_PATTERN]->(p:MorphPattern)
        OPTIONAL MATCH (r)-[:IN_DOMAIN]->(d:SemanticDomain)
        RETURN r.root AS root, r.gloss AS rootGloss, r.verseCount AS rootVerseCount,
               l.lemma AS lemma, l.glossEn AS lemmaGloss, l.pos AS lemmaPos,
               l.verseCount AS lemmaVerseCount,
               w.verseId AS verseId, w.arabicText AS arabicText,
               p.pattern AS pattern, p.label AS patternLabel,
               d.domainId AS domainId, d.nameEn AS domainName
        ORDER BY l.lemma, w.verseId
    """, root=root).data()

    if not results:
        return {"root": root, "found": False, "message": "Root not found"}

    root_info = {
        "root": results[0]['root'],
        "gloss": results[0]['rootGloss'] or '',
        "verse_count": results[0]['rootVerseCount'] or 0,
    }

    # Collect domains
    domains = {}
    for r in results:
        if r['domainId']:
            domains[r['domainId']] = r['domainName']

    # Group by lemma then pattern
    lemma_map = {}
    for r in results:
        lem = r['lemma']
        if not lem:
            continue
        if lem not in lemma_map:
            lemma_map[lem] = {
                "lemma": lem,
                "gloss": r['lemmaGloss'] or '',
                "pos": r['lemmaPos'] or '',
                "verse_count": r['lemmaVerseCount'] or 0,
                "pattern": r['pattern'] or '',
                "pattern_label": r['patternLabel'] or '',
                "sample_verses": [],
            }
        if r['verseId'] and len(lemma_map[lem]['sample_verses']) < 5:
            # Avoid duplicate verses
            existing = {v['verse_id'] for v in lemma_map[lem]['sample_verses']}
            if r['verseId'] not in existing:
                lemma_map[lem]['sample_verses'].append({
                    "verse_id": r['verseId'],
                    "arabic": r['arabicText'] or '',
                })

    return {
        "root": root_info,
        "found": True,
        "semantic_domains": domains,
        "total_lemmas": len(lemma_map),
        "lemmas": sorted(lemma_map.values(), key=lambda x: -x['verse_count']),
    }


def tool_get_verse_words(session, verse_id: str) -> dict:
    """Word-by-word breakdown of a verse."""
    results = session.run("""
        MATCH (w:WordToken {verseId: $vid})
        OPTIONAL MATCH (w)-[:HAS_LEMMA]->(l:Lemma)
        OPTIONAL MATCH (l)-[:DERIVES_FROM]->(r:ArabicRoot)
        OPTIONAL MATCH (w)-[:FOLLOWS_PATTERN]->(p:MorphPattern)
        RETURN w.tokenId AS tokenId, w.wordPos AS wordPos,
               w.arabicText AS arabicText, w.pos AS pos,
               w.morphFeatures AS morphFeatures, w.wazn AS wazn,
               l.lemma AS lemma, l.glossEn AS gloss,
               r.root AS root, r.gloss AS rootGloss,
               p.pattern AS pattern, p.label AS patternLabel
        ORDER BY w.wordPos
    """, vid=verse_id).data()

    if not results:
        return {"verse_id": verse_id, "found": False, "message": "Verse not found or no word tokens"}

    # Also get the verse text
    verse = session.run("""
        MATCH (v:Verse {verseId: $vid})
        RETURN v.text AS text, v.surahName AS surahName, v.surah AS surah
    """, vid=verse_id).single()

    words = []
    for r in results:
        words.append({
            "position": r['wordPos'],
            "arabic": r['arabicText'],
            "root": r['root'] or '',
            "root_gloss": r['rootGloss'] or r['gloss'] or '',
            "lemma": r['lemma'] or '',
            "pos": r['pos'],
            "pattern": r['pattern'] or r['wazn'] or '',
            "pattern_label": r['patternLabel'] or '',
            "morphology": r['morphFeatures'] or '',
        })

    return {
        "verse_id": verse_id,
        "found": True,
        "surah_name": verse['surahName'] if verse else '',
        "translation": verse['text'] if verse else '',
        "word_count": len(words),
        "words": words,
    }


def tool_search_semantic_field(session, domain: str) -> dict:
    """Find all roots and words in a semantic domain."""
    results = session.run("""
        MATCH (d:SemanticDomain)
        WHERE d.domainId = $domain
           OR d.nameEn CONTAINS $domain
           OR d.nameAr CONTAINS $domain
        WITH d LIMIT 1
        OPTIONAL MATCH (r:ArabicRoot)-[:IN_DOMAIN]->(d)
        OPTIONAL MATCH (l:Lemma)-[:DERIVES_FROM]->(r)
        RETURN d.domainId AS domainId, d.nameEn AS nameEn,
               d.nameAr AS nameAr, d.description AS description,
               r.root AS root, r.gloss AS rootGloss, r.verseCount AS rootVerseCount,
               l.lemma AS lemma, l.glossEn AS lemmaGloss, l.verseCount AS lemmaVerseCount
        ORDER BY r.verseCount DESC, l.verseCount DESC
    """, domain=domain.lower()).data()

    if not results or not results[0]['domainId']:
        # Try fuzzy match
        all_domains = session.run("""
            MATCH (d:SemanticDomain)
            RETURN d.domainId AS id, d.nameEn AS name, d.nameAr AS nameAr
        """).data()
        return {
            "domain": domain,
            "found": False,
            "available_domains": [{"id": d['id'], "name": d['name'], "nameAr": d['nameAr']} for d in all_domains],
        }

    domain_info = {
        "domainId": results[0]['domainId'],
        "nameEn": results[0]['nameEn'],
        "nameAr": results[0]['nameAr'],
        "description": results[0]['description'],
    }

    # Group by root
    root_map = {}
    for r in results:
        if not r['root']:
            continue
        if r['root'] not in root_map:
            root_map[r['root']] = {
                "root": r['root'],
                "gloss": r['rootGloss'] or '',
                "verse_count": r['rootVerseCount'] or 0,
                "lemmas": [],
            }
        if r['lemma']:
            existing = {l['lemma'] for l in root_map[r['root']]['lemmas']}
            if r['lemma'] not in existing:
                root_map[r['root']]['lemmas'].append({
                    "lemma": r['lemma'],
                    "gloss": r['lemmaGloss'] or '',
                    "verse_count": r['lemmaVerseCount'] or 0,
                })

    return {
        "domain": domain_info,
        "found": True,
        "total_roots": len(root_map),
        "roots": sorted(root_map.values(), key=lambda x: -x['verse_count']),
    }


def tool_lookup_wujuh(session, root: str) -> dict:
    """Show all distinct meanings (wujuh) of a polysemous word/root."""
    # First check if we have wujuh data stored as node properties
    # For now, load from the CSV data that was imported
    import csv
    from pathlib import Path
    wujuh_csv = Path(__file__).parent / "data" / "wujuh_entries.csv"

    if not wujuh_csv.exists():
        return {"root": root, "found": False, "message": "Wujuh data not loaded"}

    # Search for matching root
    matches = []
    with open(wujuh_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['root'] == root or root in row.get('lemma', ''):
                matches.append(row)

    if not matches:
        # Try to find verses with this root to provide context even without wujuh data
        verse_data = session.run("""
            MATCH (r:ArabicRoot)
            WHERE r.root = $root OR r.rootBW = $root
            OPTIONAL MATCH (v:Verse)-[:MENTIONS_ROOT]->(r)
            RETURN r.root AS root, r.gloss AS gloss,
                   v.verseId AS verseId, v.text AS text
            ORDER BY v.verseId LIMIT 10
        """, root=root).data()

        if verse_data:
            return {
                "root": root,
                "found": False,
                "message": f"No wujuh (polysemy) data for this root yet. Root '{verse_data[0]['root']}' ({verse_data[0]['gloss']}) appears in {len(verse_data)} sample verses.",
                "sample_verses": [{"verse_id": v['verseId'], "text": v['text']} for v in verse_data if v['verseId']],
            }
        return {"root": root, "found": False, "message": "Root not found"}

    senses = []
    for m in matches:
        sample_verses = json.loads(m.get('sampleVerses', '[]'))
        # Fetch verse texts for each sample
        verse_texts = {}
        if sample_verses:
            vt_results = session.run("""
                UNWIND $vids AS vid
                MATCH (v:Verse {verseId: vid})
                RETURN v.verseId AS verseId, v.text AS text
            """, vids=sample_verses[:5]).data()
            verse_texts = {v['verseId']: v['text'] for v in vt_results}

        senses.append({
            "sense_id": m['senseId'],
            "meaning_en": m['meaningEn'],
            "meaning_ar": m.get('meaningAr', ''),
            "sample_verses": [
                {"verse_id": vid, "text": verse_texts.get(vid, '')}
                for vid in sample_verses[:5]
            ],
        })

    return {
        "root": root,
        "lemma": matches[0].get('lemma', ''),
        "found": True,
        "total_senses": len(senses),
        "senses": senses,
    }


def tool_search_morphological_pattern(session, pattern: str = None,
                                       pos: str = None,
                                       verb_form: str = None) -> dict:
    """Find words by morphological pattern, POS, or verbal form."""
    conditions = []
    params = {}

    if pattern:
        conditions.append("(p.pattern = $pattern OR p.patternBW = $pattern)")
        params['pattern'] = pattern
    if pos:
        conditions.append("w.pos = $pos")
        params['pos'] = pos.upper()
    if verb_form:
        # verb_form could be "IV", "4", "Form IV", etc.
        vf_num = verb_form.replace('Form ', '').replace('form ', '')
        roman_to_num = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
                        'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10', 'XI': '11'}
        vf_num = roman_to_num.get(vf_num, vf_num)
        conditions.append("w.morphFeatures CONTAINS $vf_str")
        params['vf_str'] = f'"vf": "{vf_num}"'

    if not conditions:
        return {"error": "Provide at least one of: pattern, pos, verb_form"}

    where_clause = " AND ".join(conditions)

    if pattern:
        query = f"""
            MATCH (w:WordToken)-[:FOLLOWS_PATTERN]->(p:MorphPattern)
            WHERE {where_clause}
            OPTIONAL MATCH (w)-[:HAS_LEMMA]->(l:Lemma)
            OPTIONAL MATCH (l)-[:DERIVES_FROM]->(r:ArabicRoot)
            RETURN DISTINCT w.arabicText AS arabic, w.pos AS pos,
                   l.lemma AS lemma, r.root AS root, r.gloss AS gloss,
                   p.pattern AS pattern, p.label AS patternLabel,
                   count(w) AS occurrences
            ORDER BY occurrences DESC
            LIMIT 100
        """
    else:
        query = f"""
            MATCH (w:WordToken)
            WHERE {where_clause}
            OPTIONAL MATCH (w)-[:HAS_LEMMA]->(l:Lemma)
            OPTIONAL MATCH (l)-[:DERIVES_FROM]->(r:ArabicRoot)
            OPTIONAL MATCH (w)-[:FOLLOWS_PATTERN]->(p:MorphPattern)
            RETURN DISTINCT w.arabicText AS arabic, w.pos AS pos,
                   l.lemma AS lemma, r.root AS root, r.gloss AS gloss,
                   p.pattern AS pattern, p.label AS patternLabel,
                   count(w) AS occurrences
            ORDER BY occurrences DESC
            LIMIT 100
        """

    results = session.run(query, **params).data()

    if not results:
        return {"found": False, "message": "No words match the given criteria",
                "pattern": pattern, "pos": pos, "verb_form": verb_form}

    # Group by root
    by_root = {}
    for r in results:
        root_key = r['root'] or 'unknown'
        if root_key not in by_root:
            by_root[root_key] = {
                "root": r['root'] or '',
                "gloss": r['gloss'] or '',
                "words": [],
            }
        by_root[root_key]['words'].append({
            "arabic": r['arabic'],
            "lemma": r['lemma'] or '',
            "pos": r['pos'],
            "pattern": r['pattern'] or '',
            "occurrences": r['occurrences'],
        })

    return {
        "found": True,
        "pattern": pattern,
        "pos": pos,
        "verb_form": verb_form,
        "total_distinct_words": len(results),
        "by_root": sorted(by_root.values(), key=lambda x: -sum(w['occurrences'] for w in x['words'])),
    }


# ── tool schema (Anthropic tool_use format) ────────────────────────────────────

TOOLS = [
    {
        "name": "search_keyword",
        "description": (
            "Search for all Quran verses that mention a specific keyword. "
            "Results are grouped by surah and ranked by relevance (TF-IDF score). "
            "Use this to find where a concept appears across the Quran."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "A single word to search for (e.g. 'covenant', 'prayer', 'abraham')"
                }
            },
            "required": ["keyword"]
        }
    },
    {
        "name": "get_verse",
        "description": (
            "Get a specific verse by its ID, along with its keywords and all directly "
            "connected verses in the knowledge graph. Shows what the verse is thematically "
            "linked to and which keywords they share. Use this to deeply explore a single verse."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verse_id": {
                    "type": "string",
                    "description": "Verse ID in surah:verse format, e.g. '2:255' or '36:1'"
                }
            },
            "required": ["verse_id"]
        }
    },
    {
        "name": "traverse_topic",
        "description": (
            "Explore a topic using multiple keywords + graph traversal. "
            "Finds direct keyword matches first, then expands outward through "
            "thematic connections (1-2 hops) to find related verses that wouldn't "
            "show up in a keyword search. Best for broad topic exploration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of keywords describing the topic (e.g. ['covenant', 'abraham', 'prophet'])"
                },
                "hops": {
                    "type": "integer",
                    "description": "How many graph hops to traverse: 1 (faster, tighter focus) or 2 (broader, more connections). Default: 1",
                    "default": 1
                }
            },
            "required": ["keywords"]
        }
    },
    {
        "name": "find_path",
        "description": (
            "Find the shortest thematic path between two verses through the knowledge graph. "
            "Shows the chain of connected verses and the keywords that bridge each step. "
            "Useful for discovering unexpected connections between concepts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verse_id_1": {
                    "type": "string",
                    "description": "Starting verse ID, e.g. '2:255'"
                },
                "verse_id_2": {
                    "type": "string",
                    "description": "Ending verse ID, e.g. '112:1'"
                }
            },
            "required": ["verse_id_1", "verse_id_2"]
        }
    },
    {
        "name": "explore_surah",
        "description": (
            "Get all verses in a specific surah (chapter) and a map of its strongest "
            "thematic connections to other surahs. Use this to understand a surah's "
            "content and how it relates to the rest of the Quran."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "surah_number": {
                    "type": "integer",
                    "description": "Surah number between 1 and 114"
                }
            },
            "required": ["surah_number"]
        }
    },
    {
        "name": "semantic_search",
        "description": (
            "Find verses that are CONCEPTUALLY similar to a query using vector embeddings. "
            "Unlike search_keyword which requires exact word matches, this finds verses that "
            "express the same idea even with completely different vocabulary. "
            "Use this to catch verses about redemption, divine mercy, being freed from sin, "
            "etc. when searching for 'forgiveness' — or any concept where the idea matters "
            "more than the exact words. Always use this alongside keyword search for full coverage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A phrase or sentence describing the concept to search for, e.g. 'God forgiving and accepting repentance' or 'patience in hardship'"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most similar verses to return (default 40, max 80)",
                    "default": 40
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "query_typed_edges",
        "description": (
            "Find verses connected to a given verse by a specific relationship type: "
            "SUPPORTS (independent evidence), ELABORATES (expands with detail), "
            "QUALIFIES (adds condition/exception), CONTRASTS (complementary perspective), "
            "REPEATS (near-verbatim across surahs). Use this after get_verse to understand "
            "HOW verses relate, not just THAT they relate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verse_id": {
                    "type": "string",
                    "description": "Verse reference e.g. '2:255'"
                },
                "edge_type": {
                    "type": "string",
                    "enum": ["SUPPORTS", "ELABORATES", "QUALIFIES", "CONTRASTS", "REPEATS"],
                    "description": "Optional: filter to one relationship type. Omit to get all typed edges."
                }
            },
            "required": ["verse_id"]
        }
    },
    {
        "name": "search_arabic_root",
        "description": (
            "Find all Quran verses containing a specific Arabic tri-literal root. "
            "Arabic roots connect words that share a common origin — e.g. root k-t-b "
            "(ك ت ب) yields kitab/book, kataba/wrote, maktub/written. Use this to trace "
            "how an Arabic concept appears across the entire Quran with all its derived forms. "
            "Accepts Arabic script (e.g. 'رحم') or Buckwalter transliteration (e.g. 'rHm')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "Arabic root in Arabic letters (e.g. 'رحم', 'كتب', 'علم') or Buckwalter (e.g. 'rHm', 'ktb', 'Elm')"
                }
            },
            "required": ["root"]
        }
    },
    {
        "name": "compare_arabic_usage",
        "description": (
            "Compare how different forms of an Arabic root are used across the Quran. "
            "Shows each derived word form and the verses where it appears, revealing how "
            "the same root carries different meanings in different contexts. For example, "
            "root r-H-m (رحم) yields raHman (most gracious), raHim (merciful), raHmah (mercy) — "
            "each used in different theological contexts. Use this for linguistic analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "Arabic root to analyse (e.g. 'رحم' or 'rHm')"
                }
            },
            "required": ["root"]
        }
    },
    {
        "name": "lookup_word",
        "description": (
            "Look up any Arabic word in the Quran. Returns the word's trilateral root, "
            "lemma, morphological pattern (wazn), part of speech, grammatical features, "
            "and all verse occurrences. Accepts Arabic script or Buckwalter transliteration. "
            "Use this when a user asks about a specific Arabic word."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "Arabic word to look up (e.g. 'رحيم' or 'rHym')"
                }
            },
            "required": ["word"]
        }
    },
    {
        "name": "explore_root_family",
        "description": (
            "Show the full derivative tree of an Arabic root — all lemmas derived from it, "
            "grouped by morphological pattern, with semantic domain membership and sample verses. "
            "Reveals how a single root generates a family of related words. "
            "Use this to explore the semantic architecture of a root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "Arabic root to explore (e.g. 'رحم' or 'rHm')"
                }
            },
            "required": ["root"]
        }
    },
    {
        "name": "get_verse_words",
        "description": (
            "Get a complete word-by-word grammatical breakdown of a Quranic verse. "
            "Returns each word with its Arabic text, root, lemma, morphological pattern, "
            "English gloss, part of speech, and grammatical features (person, gender, number, case). "
            "Use this when a user wants to understand the grammar or etymology of a specific verse."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verse_id": {
                    "type": "string",
                    "description": "Verse ID in surah:verse format (e.g. '1:1')"
                }
            },
            "required": ["verse_id"]
        }
    },
    {
        "name": "search_semantic_field",
        "description": (
            "Find all Arabic roots and words that belong to a semantic domain (e.g. 'mercy', "
            "'knowledge', 'creation'). Returns the domain's roots with their derived lemmas "
            "and verse counts. Based on al-Isfahani's classical categorization. "
            "Use this to explore how the Quran expresses a concept through multiple roots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Semantic domain name or Arabic name (e.g. 'mercy', 'رحمة', 'knowledge')"
                }
            },
            "required": ["domain"]
        }
    },
    {
        "name": "lookup_wujuh",
        "description": (
            "Show all distinct meanings (wujuh) of a polysemous root across the Quran. "
            "Based on the classical Islamic discipline of al-wujuh wa al-naza'ir. "
            "Returns each sense with its meaning and sample verses. "
            "Use this when exploring how the same word carries different meanings in different contexts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "Arabic root to look up wujuh for (e.g. 'هدي' or 'hdy')"
                }
            },
            "required": ["root"]
        }
    },
    {
        "name": "search_morphological_pattern",
        "description": (
            "Find Quranic words by morphological pattern (wazn), part of speech, or verbal form. "
            "For example, find all words on the فَعِيل pattern (intensive adjectives like رحيم, عليم), "
            "or all Form IV verbs (أَفْعَلَ — causative). Results grouped by root. "
            "Use this to study how morphological patterns shape meaning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Morphological pattern in Arabic (e.g. 'فَعِيل', 'فَعُول') or Buckwalter"
                },
                "pos": {
                    "type": "string",
                    "description": "Part of speech filter (e.g. 'V.PERF', 'N', 'ADJ', 'ACT_PCPL')"
                },
                "verb_form": {
                    "type": "string",
                    "description": "Verbal form number or Roman numeral (e.g. '4', 'IV', 'X')"
                }
            }
        }
    }
]

SYSTEM_PROMPT = cfg.system_prompt()


# ── agentic tool-use loop ──────────────────────────────────────────────────────

def dispatch_tool(session, tool_name: str, tool_input: dict, user_query: str = None) -> str:
    """Call the appropriate graph function, apply retrieval gating, return JSON."""
    try:
        if tool_name == "search_keyword":
            result = tool_search_keyword(session, **tool_input)
        elif tool_name == "get_verse":
            result = tool_get_verse(session, **tool_input)
        elif tool_name == "traverse_topic":
            result = tool_traverse_topic(session, **tool_input)
        elif tool_name == "find_path":
            result = tool_find_path(session, **tool_input)
        elif tool_name == "explore_surah":
            result = tool_explore_surah(session, **tool_input)
        elif tool_name == "semantic_search":
            result = tool_semantic_search(session, **tool_input)
        elif tool_name == "query_typed_edges":
            result = tool_query_typed_edges(session, **tool_input)
        elif tool_name == "search_arabic_root":
            result = tool_search_arabic_root(session, **tool_input)
        elif tool_name == "compare_arabic_usage":
            result = tool_compare_arabic_usage(session, **tool_input)
        elif tool_name == "lookup_word":
            result = tool_lookup_word(session, **tool_input)
        elif tool_name == "explore_root_family":
            result = tool_explore_root_family(session, **tool_input)
        elif tool_name == "get_verse_words":
            result = tool_get_verse_words(session, **tool_input)
        elif tool_name == "search_semantic_field":
            result = tool_search_semantic_field(session, **tool_input)
        elif tool_name == "lookup_wujuh":
            result = tool_lookup_wujuh(session, **tool_input)
        elif tool_name == "search_morphological_pattern":
            result = tool_search_morphological_pattern(session, **tool_input)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        # Retrieval quality gating: rerank + assess + reorder
        if user_query and tool_name in ("search_keyword", "semantic_search", "traverse_topic"):
            try:
                from retrieval_gate import gate_tool_result
                result = gate_tool_result(user_query, tool_name, result)
            except Exception as e:
                result["gate_error"] = str(e)

    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result, ensure_ascii=False)


def run_agent_turn(
    user_message: str,
    conversation: list,
    session,
    client: anthropic.Anthropic,
) -> str:
    """
    Run one full agent turn: may involve multiple tool calls before final response.
    Returns the final text response.
    """
    conversation.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=cfg.llm_max_tokens(),
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=conversation,
        )

        # Collect any text blocks for display
        text_parts = [b.text for b in response.content if b.type == "text" and b.text.strip()]

        # If no tool calls, we're done
        if response.stop_reason != "tool_use":
            final_text = "\n".join(text_parts) if text_parts else "(no response)"
            conversation.append({"role": "assistant", "content": response.content})
            return final_text

        # Process tool calls
        tool_results = []
        tool_names = []
        for block in response.content:
            if block.type == "tool_use":
                tool_names.append(f"{block.name}({json.dumps(block.input)})")
                print(f"  [tool] {block.name}({json.dumps(block.input)[:80]}{'...' if len(json.dumps(block.input)) > 80 else ''})")
                result_str = dispatch_tool(session, block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

        # Add assistant turn + tool results to conversation
        conversation.append({"role": "assistant", "content": response.content})
        conversation.append({"role": "user", "content": tool_results})
        # Loop back for Claude to process tool results


# ── main loop ──────────────────────────────────────────────────────────────────

def main():
    print("Quran Knowledge Graph — Conversational Agent")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print("Claude can freely explore the graph using tools.")
    print()

    if not ANTHROPIC_KEY and not ANTHROPIC_TOKEN:
        print("Set ANTHROPIC_API_KEY or ANTHROPIC_OAUTH_TOKEN in your .env file first.")
        sys.exit(1)

    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("  Connected OK")
    except Exception as e:
        print(f"  Connection failed: {e}")
        sys.exit(1)

    if ANTHROPIC_TOKEN:
        client = anthropic.Anthropic(auth_token=ANTHROPIC_TOKEN)
    else:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    conversation = []

    print("\nType your question (or 'quit' / 'clear'):\n")

    with driver.session() as session:
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break
            if question.lower() == "clear":
                conversation.clear()
                print("  Conversation cleared.\n")
                continue

            print()
            answer = run_agent_turn(question, conversation, session, client)
            print(f"\nClaude:\n{answer}\n")
            print("-" * 60)

    driver.close()


if __name__ == "__main__":
    main()
