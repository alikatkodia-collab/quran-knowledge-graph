"""
Quran Graph — FULL Hallucination-Reduced Version

All phases enabled:
  - Phase 2: Retrieval gate (cross-encoder reranking)
  - Phase 3: NLI citation verification
  - Phase 5: Semantic entropy uncertainty check (5 Haiku probes)
  - Citation density check + re-generation

Model: claude-sonnet-4-5 (from pipeline_config.yaml)
Port: 8083

Usage:
    py app_full.py
"""

import asyncio
import json
import os
import re
import sys
import threading
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import anthropic
from neo4j import GraphDatabase

# ── env ────────────────────────────────────────────────────────────────────────

def _load_env():
    path = Path(__file__).parent / ".env"
    if not path.exists():
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

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from chat import TOOLS, SYSTEM_PROMPT, dispatch_tool
from answer_cache import save_answer, build_cache_context
from tool_compressor import compress_tool_result

# ── clients ────────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB       = os.getenv("NEO4J_DATABASE", "quran")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_TOKEN = os.getenv("ANTHROPIC_OAUTH_TOKEN", "")
CLAUDE_MODEL   = cfg.llm_model()  # claude-sonnet-4-5

print(f"[FULL] Model: {CLAUDE_MODEL}")
print("[FULL] Hallucination reduction: ALL PHASES ENABLED")

print("Connecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
try:
    driver.verify_connectivity()
    print(f"  Neo4j OK  (database: {NEO4J_DB})")
except Exception as e:
    print(f"  Neo4j unavailable: {e}\n  Graph tools will return errors until Neo4j is started.")

if ANTHROPIC_TOKEN:
    ai = anthropic.Anthropic(auth_token=ANTHROPIC_TOKEN)
    print("  Auth: OAuth token")
else:
    ai = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    print("  Auth: API key")

# ── helpers ────────────────────────────────────────────────────────────────────

_BRACKET_REF = re.compile(r'(\d+:\d+)')
_BRACKET_CONTEXT = re.compile(r'\[[\d:,\s]+\]')

def _extract_verse_refs(text: str) -> set:
    refs = set()
    for block in _BRACKET_CONTEXT.findall(text):
        refs.update(_BRACKET_REF.findall(block))
    return refs

def _fetch_verses(session, verse_ids: set) -> dict:
    if not verse_ids:
        return {}
    result = session.run(
        "UNWIND $ids AS vid "
        "MATCH (v:Verse {reference: vid}) "
        "RETURN v.reference AS id, v.text AS text, v.arabicText AS arabic",
        ids=list(verse_ids),
    )
    return {r["id"]: {"text": r["text"], "arabic": r["arabic"] or ""} for r in result}


def _check_citation_density(text: str) -> dict:
    """Check if response has enough citations. Returns density info."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 20]
    if not sentences:
        return {"density": 1.0, "cited": 0, "total": 0, "low": False}
    cited = sum(1 for s in sentences if _BRACKET_CONTEXT.search(s))
    density = cited / len(sentences) if sentences else 0
    return {
        "density": round(density, 2),
        "cited": cited,
        "total": len(sentences),
        "low": density < 0.3 and len(sentences) >= 3,
    }

# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI()

TOOL_LABELS = {
    "search_keyword":              "Searching keywords",
    "get_verse":                   "Looking up verse",
    "traverse_topic":              "Traversing topic",
    "find_path":                   "Finding path",
    "explore_surah":               "Exploring surah",
    "semantic_search":             "Semantic search",
    "search_arabic_root":          "Searching Arabic root",
    "compare_arabic_usage":        "Comparing Arabic usage",
    "query_typed_edges":           "Querying typed edges",
    "lookup_word":                 "Looking up word",
    "explore_root_family":         "Exploring root family",
    "get_verse_words":             "Analyzing verse words",
    "search_semantic_field":       "Searching semantic field",
    "lookup_wujuh":                "Looking up word meanings",
    "search_morphological_pattern": "Searching patterns",
}


# ── graph extraction for 3D visualiser ────────────────────────────────────────
# (identical to app.py — imported as shared logic)

def _graph_for_tool(name: str, inp: dict, result: dict):
    nodes = {}
    links = []
    active = []

    def vnode(vid, sname="", text="", arabic=""):
        nid = f"v:{vid}"
        if nid not in nodes:
            try:
                surah = int(vid.split(":")[0])
            except Exception:
                surah = 0
            nodes[nid] = {"id": nid, "type": "verse", "label": f"[{vid}]",
                          "verseId": vid, "surah": surah,
                          "surahName": sname, "text": (text or "")[:200],
                          "arabicText": (arabic or "")[:300]}
        return nid

    def knode(kw):
        nid = f"k:{kw}"
        if nid not in nodes:
            nodes[nid] = {"id": nid, "type": "keyword", "label": kw}
        return nid

    def link(src, tgt, ltype):
        links.append({"source": src, "target": tgt, "type": ltype})

    try:
        if name == "get_verse" and "verse_id" in result:
            vid = result["verse_id"]
            v = vnode(vid, result.get("surah_name", ""), result.get("text", ""), result.get("arabic_text", ""))
            active.append(v)
            for kw in result.get("keywords", [])[:10]:
                k = knode(kw); link(v, k, "mentions")
            for cv in result.get("connected_verses", [])[:cfg.vis("get_verse_max_connected")]:
                cv_n = vnode(cv["verse_id"], cv.get("surah_name",""), cv.get("text",""))
                link(v, cv_n, "related")
                for kw in cv.get("shared_keywords", [])[:cfg.vis("get_verse_max_kw_per_neighbour")]:
                    k = knode(kw); link(cv_n, k, "mentions")

        elif name == "search_keyword" and "keyword" in result:
            kw = result["keyword"]
            k = knode(kw); active.append(k)
            count = 0
            _max_kw = cfg.vis("search_keyword_max_nodes")
            for surah_verses in result.get("by_surah", {}).values():
                for v_data in surah_verses[:3]:
                    if count >= _max_kw: break
                    v = vnode(v_data["verse_id"], "", v_data.get("text",""), v_data.get("arabic_text",""))
                    link(v, k, "mentions"); count += 1
                if count >= _max_kw: break

        elif name == "traverse_topic":
            direct_ids = {v["verse_id"] for v in result.get("direct_matches", [])}
            for v_data in result.get("direct_matches", []):
                v = vnode(v_data["verse_id"], v_data.get("surah_name",""), v_data.get("text",""))
                active.append(v)
                for kw in v_data.get("matched_keywords", []):
                    k = knode(kw); link(v, k, "mentions")
            for v_data in result.get("hop_1_connections", []):
                v = vnode(v_data["verse_id"], v_data.get("surah_name",""), v_data.get("text",""))
                for src_id in v_data.get("connected_via", [])[:2]:
                    if src_id in direct_ids:
                        link(f"v:{src_id}", v, "related")
            for v_data in result.get("hop_2_connections", []):
                vnode(v_data["verse_id"], v_data.get("surah_name",""), v_data.get("text",""))

        elif name == "find_path" and "path" in result:
            path = result["path"]
            for i, step in enumerate(path):
                v = vnode(step["verse_id"], step.get("surah_name",""), step.get("text",""))
                if i == 0 or i == len(path) - 1:
                    active.append(v)
                if i > 0:
                    link(f"v:{path[i-1]['verse_id']}", v, "related")
                for kw in step.get("bridge_keywords", [])[:cfg.vis("find_path_max_bridge_kw")]:
                    k = knode(kw); link(v, k, "mentions")

        elif name == "explore_surah" and "verses" in result:
            sname = result.get("surah_name", "")
            for v_data in result.get("verses", [])[:cfg.vis("explore_surah_max_verses")]:
                v = vnode(v_data["verse_id"], sname, v_data.get("text",""))
            active = list(nodes.keys())[:5]

        elif name == "search_arabic_root" and "root" in result:
            root = result["root"]
            rnid = f"r:{root}"
            nodes[rnid] = {"id": rnid, "type": "arabicRoot", "label": root,
                           "gloss": result.get("gloss", "")}
            active.append(rnid)
            count = 0
            for surah_verses in result.get("by_surah", {}).values():
                for v_data in surah_verses[:3]:
                    if count >= 15: break
                    v = vnode(v_data["verse_id"], "", v_data.get("text",""), v_data.get("arabic_text",""))
                    link(v, rnid, "mentions_root"); count += 1
                if count >= 15: break

        elif name == "query_typed_edges" and "by_type" in result:
            vid = result.get("verse_id", "")
            v_center = vnode(vid)
            active.append(v_center)
            for rtype, edges in result.get("by_type", {}).items():
                ltype = rtype.lower()
                for e in edges[:8]:
                    v = vnode(e["verse_id"], e.get("surah_name",""), e.get("text",""), e.get("arabic_text",""))
                    link(v_center, v, ltype)

        elif name == "compare_arabic_usage" and "root" in result:
            root = result["root"]
            rnid = f"r:{root}"
            nodes[rnid] = {"id": rnid, "type": "arabicRoot", "label": root,
                           "gloss": result.get("gloss", "")}
            active.append(rnid)
            for form_data in result.get("forms", [])[:8]:
                fnid = f"f:{form_data['form']}"
                nodes[fnid] = {"id": fnid, "type": "arabicForm", "label": form_data["form"]}
                link(rnid, fnid, "derives")
                for v_data in form_data.get("sample_verses", [])[:3]:
                    v = vnode(v_data["verse_id"], v_data.get("surah_name",""), v_data.get("text",""), v_data.get("arabic_text",""))
                    link(fnid, v, "appears_in")

        elif name == "lookup_word" and result.get("found"):
            for lem_data in result.get("lemmas", [])[:5]:
                lnid = f"l:{lem_data['lemma']}"
                nodes[lnid] = {"id": lnid, "type": "lemma",
                               "label": lem_data['lemma'],
                               "gloss": lem_data.get('rootGloss', '')}
                active.append(lnid)
                if lem_data.get('root'):
                    rnid = f"r:{lem_data['root']}"
                    nodes[rnid] = {"id": rnid, "type": "arabicRoot",
                                   "label": lem_data['root'],
                                   "gloss": lem_data.get('rootGloss', '')}
                    link(lnid, rnid, "derives_from")
                for occ in lem_data.get('occurrences', [])[:8]:
                    v = vnode(occ['verse_id']); link(lnid, v, "appears_in")

        elif name == "explore_root_family" and result.get("found"):
            root_info = result.get("root", {})
            rnid = f"r:{root_info.get('root', '')}"
            nodes[rnid] = {"id": rnid, "type": "arabicRoot",
                           "label": root_info.get('root', ''),
                           "gloss": root_info.get('gloss', '')}
            active.append(rnid)
            for dom_id, dom_name in result.get("semantic_domains", {}).items():
                dnid = f"d:{dom_id}"
                nodes[dnid] = {"id": dnid, "type": "semanticDomain", "label": dom_name}
                link(rnid, dnid, "in_domain")
            for lem_data in result.get("lemmas", [])[:15]:
                lnid = f"l:{lem_data['lemma']}"
                nodes[lnid] = {"id": lnid, "type": "lemma",
                               "label": lem_data['lemma'],
                               "gloss": lem_data.get('gloss', '')}
                link(rnid, lnid, "derives")
                if lem_data.get('pattern'):
                    pnid = f"p:{lem_data['pattern']}"
                    nodes[pnid] = {"id": pnid, "type": "morphPattern",
                                   "label": lem_data.get('pattern_label') or lem_data['pattern']}
                    link(lnid, pnid, "follows_pattern")
                for sv in lem_data.get('sample_verses', [])[:2]:
                    v = vnode(sv['verse_id']); link(lnid, v, "appears_in")

        elif name == "get_verse_words" and result.get("found"):
            vid = result["verse_id"]
            v = vnode(vid, result.get("surah_name", ""), result.get("translation", ""))
            active.append(v)
            for w in result.get("words", []):
                if w.get('root'):
                    rnid = f"r:{w['root']}"
                    nodes[rnid] = {"id": rnid, "type": "arabicRoot",
                                   "label": w['root'],
                                   "gloss": w.get('root_gloss', '')}
                    link(v, rnid, "mentions_root")

        elif name == "search_semantic_field" and result.get("found"):
            dom = result.get("domain", {})
            dnid = f"d:{dom.get('domainId', '')}"
            nodes[dnid] = {"id": dnid, "type": "semanticDomain",
                           "label": dom.get('nameEn', '')}
            active.append(dnid)
            for root_data in result.get("roots", [])[:15]:
                rnid = f"r:{root_data['root']}"
                nodes[rnid] = {"id": rnid, "type": "arabicRoot",
                               "label": root_data['root'],
                               "gloss": root_data.get('gloss', '')}
                link(dnid, rnid, "contains_root")
                for lem in root_data.get('lemmas', [])[:3]:
                    lnid = f"l:{lem['lemma']}"
                    nodes[lnid] = {"id": lnid, "type": "lemma",
                                   "label": lem['lemma'],
                                   "gloss": lem.get('gloss', '')}
                    link(rnid, lnid, "derives")

        elif name == "lookup_wujuh" and result.get("found"):
            root = result.get("root", "")
            rnid = f"r:{root}"
            nodes[rnid] = {"id": rnid, "type": "arabicRoot", "label": root}
            active.append(rnid)
            for sense in result.get("senses", []):
                snid = f"s:{sense['sense_id']}"
                nodes[snid] = {"id": snid, "type": "lemma",
                               "label": sense['meaning_en'][:30]}
                link(rnid, snid, "has_sense")
                for sv in sense.get("sample_verses", [])[:3]:
                    v = vnode(sv['verse_id'], "", sv.get('text', ''))
                    link(snid, v, "example_in")

        elif name == "search_morphological_pattern" and result.get("found"):
            pat = result.get("pattern", "")
            if pat:
                pnid = f"p:{pat}"
                nodes[pnid] = {"id": pnid, "type": "morphPattern", "label": pat}
                active.append(pnid)
            for root_data in result.get("by_root", [])[:10]:
                if root_data.get('root'):
                    rnid = f"r:{root_data['root']}"
                    nodes[rnid] = {"id": rnid, "type": "arabicRoot",
                                   "label": root_data['root'],
                                   "gloss": root_data.get('gloss', '')}
                    if pat:
                        link(pnid, rnid, "used_by")
                    for w in root_data.get('words', [])[:3]:
                        lnid = f"l:{w.get('lemma', w['arabic'])}"
                        nodes[lnid] = {"id": lnid, "type": "lemma",
                                       "label": w.get('lemma', w['arabic'])}
                        link(rnid, lnid, "derives")

    except Exception as e:
        print(f"  [graph] extract error ({name}): {e}")
        return None

    return {"nodes": list(nodes.values()), "links": links, "active": active} if nodes else None


class ChatRequest(BaseModel):
    message: str
    history: list


@app.get("/")
async def index():
    html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/stats")
async def stats():
    html = (Path(__file__).parent / "stats.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/verses")
async def all_verses():
    with driver.session(database=NEO4J_DB) as s:
        result = s.run(
            "MATCH (v:Verse) RETURN v.reference AS id, v.sura AS surah, v.number AS num "
            "ORDER BY v.sura, v.number"
        )
        verses = [{"id": r["id"], "surah": r["surah"]} for r in result if r["id"]]
    return {"verses": verses}


@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        _agent_stream(req.message, req.history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _agent_stream(message: str, history: list):
    import queue as tqueue
    q: tqueue.SimpleQueue = tqueue.SimpleQueue()

    def run():
        try:
            msgs = []
            for m in history:
                r = m.get("role")    if isinstance(m, dict) else m["role"]
                c = m.get("content") if isinstance(m, dict) else m["content"]
                if r in ("user", "assistant") and c:
                    msgs.append({"role": r, "content": str(c)})
            msgs.append({"role": "user", "content": message})

            full_text = ""
            system_prompt = SYSTEM_PROMPT

            # ── answer cache: check for relevant past answers ──
            try:
                cache_ctx = build_cache_context(message, top_k=3, threshold=0.6)
                if cache_ctx:
                    system_prompt = system_prompt + "\n\n" + cache_ctx
                    q.put({"t": "tool", "name": "Answer cache", "args": message[:60],
                           "summary": "Found relevant cached answers"})
            except Exception as ce:
                print(f"  [cache] lookup error: {ce}")

            # ── Phase 5: Semantic entropy uncertainty check ──
            try:
                from uncertainty import assess_uncertainty
                uc = assess_uncertainty(message, ai, n_probes=5)
                q.put({"t": "uncertainty", "d": uc})
                if uc.get("should_abstain"):
                    system_prompt = SYSTEM_PROMPT + (
                        "\n\nIMPORTANT: HIGH UNCERTAINTY detected (entropy: %.2f). "
                        "Be extra cautious. Clearly state when you are unsure. "
                        "Prefer saying 'I don't have enough information' over guessing." % uc["entropy"])
            except Exception as ue:
                print(f"  [uncertainty] error: {ue}")

            with driver.session(database=NEO4J_DB) as session:
                while True:
                    resp = ai.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=cfg.llm_max_tokens(),
                        system=system_prompt,
                        tools=TOOLS,
                        messages=msgs,
                    )

                    for block in resp.content:
                        if block.type == "text" and block.text.strip():
                            full_text += block.text
                            q.put({"t": "text", "d": block.text})

                    if resp.stop_reason != "tool_use":
                        break

                    tool_results = []
                    for block in resp.content:
                        if block.type != "tool_use":
                            continue

                        label = TOOL_LABELS.get(block.name, block.name)
                        args_s = json.dumps(block.input)
                        if len(args_s) > 80:
                            args_s = args_s[:77] + "..."

                        result_str = dispatch_tool(session, block.name, block.input, user_query=message)

                        try:
                            res = json.loads(result_str)
                            if "error" in res:
                                summary = f"Error: {res['error']}"
                            elif "total_verses" in res:
                                summary = f"Found {res['total_verses']} verses for '{res.get('keyword','')}'"
                            elif "verse_id" in res:
                                summary = f"[{res['verse_id']}]: {res.get('text','')[:90]}..."
                            elif "hops" in res:
                                summary = f"Path in {res['hops']} hops"
                            elif "verse_count" in res:
                                summary = f"{res.get('surah_name','')} — {res['verse_count']} verses"
                            elif "total_verses_found" in res:
                                summary = f"Found {res['total_verses_found']} verses"
                            else:
                                summary = "Done"
                        except Exception:
                            summary = "Done"

                        q.put({"t": "tool", "name": label, "args": args_s, "summary": summary})

                        # graph update for 3D visualiser
                        try:
                            res_dict = json.loads(result_str)
                            gu = _graph_for_tool(block.name, block.input, res_dict)
                            if gu:
                                q.put({"t": "graph_update",
                                       "nodes": gu["nodes"],
                                       "links": gu["links"],
                                       "active": gu["active"]})
                        except Exception:
                            pass

                        # etymology panel
                        _ETYMOLOGY_TOOLS = {"lookup_word", "explore_root_family",
                                            "get_verse_words", "search_semantic_field",
                                            "lookup_wujuh", "search_morphological_pattern"}
                        if block.name in _ETYMOLOGY_TOOLS:
                            try:
                                ep = json.loads(result_str)
                                if ep.get("found") or ep.get("words") or ep.get("lemmas"):
                                    q.put({"t": "etymology_panel",
                                           "tool": block.name,
                                           "result": ep})
                            except Exception:
                                pass

                        compressed = compress_tool_result(block.name, result_str)
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     compressed,
                        })

                    msgs.append({"role": "assistant", "content": resp.content})
                    msgs.append({"role": "user",      "content": tool_results})

                # ── Citation density check — re-generate if too few citations ──
                density = _check_citation_density(full_text)
                if density["low"]:
                    q.put({"t": "warning", "d": f"Low citation density ({density['density']:.0%}). Re-generating with stricter grounding..."})
                    retry_prompt = (
                        "Your previous response had low citation density "
                        f"({density['cited']}/{density['total']} sentences cited). "
                        "Please rewrite your answer ensuring EVERY factual claim "
                        "has a specific verse citation [surah:verse]. "
                        "If you cannot cite a verse for a claim, remove that claim."
                    )
                    msgs.append({"role": "user", "content": retry_prompt})
                    retry_resp = ai.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=cfg.llm_max_tokens(),
                        system=system_prompt,
                        tools=TOOLS,
                        messages=msgs,
                    )
                    retry_text = ""
                    for block in retry_resp.content:
                        if block.type == "text" and block.text.strip():
                            retry_text += block.text
                    if retry_text:
                        q.put({"t": "retry", "d": retry_text})
                        full_text = retry_text

                # Fetch verse texts for all refs in the response
                refs   = _extract_verse_refs(full_text)
                verses = _fetch_verses(session, refs)

                # ── Phase 3: NLI citation verification ──
                try:
                    from citation_verifier import verify_response
                    verification = verify_response(full_text, verses)
                    q.put({"t": "verification", "d": verification})
                except Exception as ve:
                    print(f"  [verify] error: {ve}")

                # ── save to answer cache ──
                try:
                    save_answer(message, full_text, verses)
                except Exception as ce:
                    print(f"  [cache] save error: {ce}")

                q.put({"t": "done", "verses": verses})

        except Exception as e:
            q.put({"t": "error", "d": str(e)})
        finally:
            q.put(None)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    while True:
        try:
            event = q.get_nowait()
        except Exception:
            await asyncio.sleep(0.05)
            continue
        if event is None:
            break
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


if __name__ == "__main__":
    import uvicorn
    print("\n[FULL] Quran Graph — Full Hallucination Reduction: http://localhost:8083\n")
    webbrowser.open("http://localhost:8083")
    uvicorn.run(app, host="0.0.0.0", port=8083, log_level="info")
