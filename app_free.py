"""
Quran Graph — FREE Version (Local Ollama)

Model: qwen2.5:14b-instruct (or any local model with tool-use support)
Cost: $0.00 — runs entirely on your machine
Port: 8085

Uses Ollama's tool-use API for agentic graph exploration,
same 15 tools as the paid versions.

Usage:
    py app_free.py
    py app_free.py --model qwen2.5:32b-instruct-q3_K_M
"""

import argparse
import asyncio
import json
import os
import re
import sys
import threading
import webbrowser
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
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
from chat import TOOLS as ANTHROPIC_TOOLS, SYSTEM_PROMPT, dispatch_tool
from answer_cache import save_answer, build_cache_context
from tool_compressor import compress_tool_result

# ── Ollama config ─────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:14b-instruct-q6_K"
MAX_TOKENS = 2048
MAX_TOOL_TURNS = 8  # safety cap on agentic loop

# ── convert Anthropic tool schemas to Ollama format ───────────────────────────

def _convert_tools(anthropic_tools: list) -> list:
    """Convert Anthropic tool format to Ollama/OpenAI function-calling format."""
    ollama_tools = []
    for t in anthropic_tools:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
        })
    return ollama_tools

OLLAMA_TOOLS = _convert_tools(ANTHROPIC_TOOLS)

# Parse --model early so it's available at module level
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model", default=DEFAULT_MODEL)
_parser.add_argument("--port", type=int, default=8085)
_args, _ = _parser.parse_known_args()
OLLAMA_MODEL = _args.model
OLLAMA_PORT = _args.port

# ── clients ────────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB       = os.getenv("NEO4J_DATABASE", "quran")

print("Connecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
try:
    driver.verify_connectivity()
    print(f"  Neo4j OK  (database: {NEO4J_DB})")
except Exception as e:
    print(f"  Neo4j unavailable: {e}")

# ── Ollama chat ───────────────────────────────────────────────────────────────

def ollama_chat(model: str, messages: list, tools: list | None = None,
                temperature: float = 0.3) -> dict:
    """
    Send a chat request to Ollama. Returns the full response dict.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": MAX_TOKENS,
        },
    }
    if tools:
        payload["tools"] = tools

    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


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


# ── graph extraction (simplified — reuse from app.py) ─────────────────────────

def _graph_for_tool(name: str, inp: dict, result: dict):
    """Extract graph nodes + links from a tool result for the 3D visualiser."""
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

    def lnk(src, tgt, ltype):
        links.append({"source": src, "target": tgt, "type": ltype})

    try:
        if name == "search_keyword" and "keyword" in result:
            kw = result["keyword"]
            k = knode(kw); active.append(k)
            count = 0
            for surah_verses in result.get("by_surah", {}).values():
                for v_data in surah_verses[:3]:
                    if count >= 15: break
                    v = vnode(v_data["verse_id"], "", v_data.get("text",""))
                    lnk(v, k, "mentions"); count += 1
                if count >= 15: break

        elif name == "get_verse" and "verse_id" in result:
            vid = result["verse_id"]
            v = vnode(vid, result.get("surah_name",""), result.get("text",""))
            active.append(v)
            for kw in result.get("keywords", [])[:10]:
                k = knode(kw); lnk(v, k, "mentions")
            for cv in result.get("connected_verses", [])[:8]:
                cv_n = vnode(cv["verse_id"], cv.get("surah_name",""), cv.get("text",""))
                lnk(v, cv_n, "related")

        elif name == "semantic_search" and "results" in result:
            for v_data in result.get("results", [])[:10]:
                v = vnode(v_data["verse_id"], "", v_data.get("text",""))
                active.append(v)

        elif name == "traverse_topic":
            for v_data in result.get("direct_matches", []):
                v = vnode(v_data["verse_id"], v_data.get("surah_name",""), v_data.get("text",""))
                active.append(v)
                for kw in v_data.get("matched_keywords", []):
                    k = knode(kw); lnk(v, k, "mentions")

        elif name == "search_arabic_root" and "root" in result:
            root = result["root"]
            rnid = f"r:{root}"
            nodes[rnid] = {"id": rnid, "type": "arabicRoot", "label": root}
            active.append(rnid)
            count = 0
            for surah_verses in result.get("by_surah", {}).values():
                for v_data in surah_verses[:3]:
                    if count >= 15: break
                    v = vnode(v_data["verse_id"], "", v_data.get("text",""))
                    lnk(v, rnid, "mentions_root"); count += 1
                if count >= 15: break

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
async def stats_page():
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


@app.get("/cache-stats")
async def get_cache_stats():
    from answer_cache import cache_stats
    return cache_stats()


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
            # Build messages for Ollama
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            for m in history:
                r = m.get("role") if isinstance(m, dict) else m["role"]
                c = m.get("content") if isinstance(m, dict) else m["content"]
                if r in ("user", "assistant") and c:
                    msgs.append({"role": r, "content": str(c)})

            # ── answer cache: check for relevant past answers ──
            system_content = SYSTEM_PROMPT
            try:
                cache_ctx = build_cache_context(message, top_k=3, threshold=0.6)
                if cache_ctx:
                    system_content = SYSTEM_PROMPT + "\n\n" + cache_ctx
                    msgs[0] = {"role": "system", "content": system_content}
                    q.put({"t": "tool", "name": "Answer cache", "args": message[:60],
                           "summary": "Found relevant cached answers"})
            except Exception as ce:
                print(f"  [cache] lookup error: {ce}")

            msgs.append({"role": "user", "content": message})

            full_text = ""
            turn = 0

            with driver.session(database=NEO4J_DB) as session:
                while turn < MAX_TOOL_TURNS:
                    turn += 1

                    resp = ollama_chat(
                        model=OLLAMA_MODEL,
                        messages=msgs,
                        tools=OLLAMA_TOOLS,
                        temperature=0.3,
                    )

                    msg = resp.get("message", {})
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])

                    # Emit any text content
                    if content and content.strip():
                        full_text += content
                        q.put({"t": "text", "d": content})

                    # If no tool calls, we're done
                    if not tool_calls:
                        break

                    # Add assistant message (with tool_calls) to history
                    msgs.append(msg)

                    # Process each tool call
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        tool_name = func.get("name", "")
                        tool_args = func.get("arguments", {})

                        # Ensure arguments is a dict
                        if isinstance(tool_args, str):
                            try:
                                tool_args = json.loads(tool_args)
                            except json.JSONDecodeError:
                                tool_args = {}

                        label = TOOL_LABELS.get(tool_name, tool_name)
                        args_s = json.dumps(tool_args)
                        if len(args_s) > 80:
                            args_s = args_s[:77] + "..."

                        # Execute the tool
                        result_str = dispatch_tool(session, tool_name, tool_args, user_query=message)

                        # Summary for UI
                        try:
                            res = json.loads(result_str)
                            if "error" in res:
                                summary = f"Error: {res['error']}"
                            elif "total_verses" in res:
                                summary = f"Found {res['total_verses']} verses"
                            elif "verse_id" in res:
                                summary = f"[{res['verse_id']}]"
                            elif "hops" in res:
                                summary = f"Path in {res['hops']} hops"
                            elif "verse_count" in res:
                                summary = f"{res.get('surah_name','')} — {res['verse_count']} verses"
                            else:
                                summary = "Done"
                        except Exception:
                            summary = "Done"

                        q.put({"t": "tool", "name": label, "args": args_s, "summary": summary})

                        # Graph update for 3D visualiser
                        try:
                            res_dict = json.loads(result_str)
                            gu = _graph_for_tool(tool_name, tool_args, res_dict)
                            if gu:
                                q.put({"t": "graph_update",
                                       "nodes": gu["nodes"],
                                       "links": gu["links"],
                                       "active": gu["active"]})
                        except Exception:
                            pass

                        # Etymology panel
                        _ETYMOLOGY_TOOLS = {"lookup_word", "explore_root_family",
                                            "get_verse_words", "search_semantic_field",
                                            "lookup_wujuh", "search_morphological_pattern"}
                        if tool_name in _ETYMOLOGY_TOOLS:
                            try:
                                ep = json.loads(result_str)
                                if ep.get("found") or ep.get("words") or ep.get("lemmas"):
                                    q.put({"t": "etymology_panel",
                                           "tool": tool_name,
                                           "result": ep})
                            except Exception:
                                pass

                        # Send compressed tool result back to Ollama
                        compressed = compress_tool_result(tool_name, result_str)
                        msgs.append({
                            "role": "tool",
                            "content": compressed,
                        })

                # Fetch verse texts for tooltips
                refs = _extract_verse_refs(full_text)
                verses = _fetch_verses(session, refs)

                # Save to answer cache
                try:
                    save_answer(message, full_text, verses)
                except Exception as ce:
                    print(f"  [cache] save error: {ce}")

                q.put({"t": "done", "verses": verses})

        except requests.ConnectionError:
            q.put({"t": "error", "d": "Cannot connect to Ollama. Make sure it's running: ollama serve"})
        except requests.Timeout:
            q.put({"t": "error", "d": "Ollama request timed out (300s). Try a smaller model or shorter question."})
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
    # Verify Ollama is reachable
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if OLLAMA_MODEL not in models:
            matches = [m for m in models if OLLAMA_MODEL.split(":")[0] in m]
            if matches:
                print(f"  Model '{OLLAMA_MODEL}' not found, using '{matches[0]}'")
                OLLAMA_MODEL = matches[0]
            else:
                print(f"  WARNING: Model '{OLLAMA_MODEL}' not found. Available: {models}")
    except Exception as e:
        print(f"  WARNING: Ollama not reachable ({e}). Start it with: ollama serve")

    print(f"\n[FREE] Model: {OLLAMA_MODEL}")
    print(f"[FREE] Cost: $0.00 (local)")
    print(f"[FREE] Quran Graph: http://localhost:{OLLAMA_PORT}\n")

    import uvicorn
    webbrowser.open(f"http://localhost:{OLLAMA_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=OLLAMA_PORT, log_level="info")
