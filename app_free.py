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

# Force UTF-8 stdout on Windows so stray unicode glyphs don't crash print()
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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
from chat import TOOLS as ANTHROPIC_TOOLS, dispatch_tool

# Use a lean system prompt for local models — fewer tokens, faster inference
_free_prompt_path = Path(__file__).parent / "prompts" / "system_prompt_free.txt"
SYSTEM_PROMPT = _free_prompt_path.read_text(encoding="utf-8").strip()
from answer_cache import save_answer, build_cache_context
from tool_compressor import compress_tool_result
from reasoning_memory import ReasoningMemory

# ── Ollama config ─────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:8b"
MAX_TOKENS = 4096  # allow longer, more thorough answers
MAX_TOOL_TURNS = 8  # safety cap on agentic loop

# ── Lean tool schemas for local models ────────────────────────────────────────
# Full Anthropic schemas are ~9,600 chars (15 tools with long descriptions).
# Local 8-14B models choke on that. These minimal schemas benchmarked at
# ~14s per turn vs 60-300s+ with the full set.

OLLAMA_TOOLS = [
    {"type": "function", "function": {
        "name": "search_keyword",
        "description": "Find all verses mentioning a keyword (exact match)",
        "parameters": {"type": "object",
                       "properties": {"keyword": {"type": "string", "description": "keyword to search"}},
                       "required": ["keyword"]}}},
    {"type": "function", "function": {
        "name": "semantic_search",
        "description": "Find verses conceptually related to a query via embeddings",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "natural language query"},
                                      "top_k": {"type": "integer", "description": "max results (default 20)"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "traverse_topic",
        "description": "Multi-keyword search + graph traversal. Use for broad topics to pull in connected verses via shared keywords",
        "parameters": {"type": "object",
                       "properties": {"keywords": {"type": "array", "items": {"type": "string"},
                                                   "description": "2-5 related keywords"},
                                      "hops": {"type": "integer", "description": "traversal depth (default 1)"}},
                       "required": ["keywords"]}}},
    {"type": "function", "function": {
        "name": "get_verse",
        "description": "Get a specific verse, its text, connections, and context",
        "parameters": {"type": "object",
                       "properties": {"verse_id": {"type": "string", "description": "e.g. 2:255"}},
                       "required": ["verse_id"]}}},
    {"type": "function", "function": {
        "name": "explore_surah",
        "description": "Overview of a surah: themes, key verses, cross-surah links",
        "parameters": {"type": "object",
                       "properties": {"surah_number": {"type": "integer", "description": "1-114"}},
                       "required": ["surah_number"]}}},
]

# Parse --model early so it's available at module level
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model", default=DEFAULT_MODEL)
_parser.add_argument("--port", type=int, default=8085)
_args, _ = _parser.parse_known_args()
OLLAMA_MODEL = _args.model
OLLAMA_PORT = _args.port

# OpenRouter for deep-dive / cache seeding (larger free models)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder:free").strip()
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

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
    print(f"\n  ❌ Neo4j unavailable: {e}")
    print(f"\n  Fix: Open Neo4j Desktop and START your 'quran' database.")
    print(f"  Then re-run this script.\n")
    sys.exit(1)

# Initialize reasoning memory (creates schema indexes if missing)
reasoning_memory = ReasoningMemory(driver, db=NEO4J_DB)
try:
    reasoning_memory.ensure_schema()
    print(f"  ReasoningMemory schema OK")
except Exception as e:
    print(f"  [ReasoningMemory] schema setup failed: {e}")

# ── Ollama chat ───────────────────────────────────────────────────────────────

def openrouter_chat(model: str, messages: list, tools: list | None = None,
                    temperature: float = 0.3, num_predict: int | None = None) -> dict:
    """
    Send a chat request to OpenRouter (OpenAI-compatible API).
    Returns a dict with the same shape as ollama_chat (has `message` with
    `content` and optional `tool_calls`).

    OpenRouter expects OpenAI-style tool format, which matches what we already
    build in OLLAMA_TOOLS.
    """
    # OpenRouter messages use OpenAI's format — tool message 'content' is fine,
    # but tool calls sit on assistant messages the same way. Our code already
    # follows this structure, so we can pass messages through unchanged.
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": num_predict or MAX_TOKENS,
    }
    if tools:
        payload["tools"] = tools

    r = requests.post(
        OPENROUTER_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8085",  # OpenRouter courtesy header
            "X-Title": "Quran Knowledge Graph",
        },
        timeout=900,
    )
    r.raise_for_status()
    data = r.json()

    # Normalise OpenAI-style response into our internal "like ollama" shape
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {}) or {}
    return {
        "message": {
            "role": "assistant",
            "content": msg.get("content", "") or "",
            "tool_calls": msg.get("tool_calls", []) or [],
        }
    }


def ollama_chat(model: str, messages: list, tools: list | None = None,
                temperature: float = 0.3, num_ctx: int = 24576,
                num_predict: int | None = None, think: bool = False) -> dict:
    """
    Send a chat request to Ollama. Returns the full response dict.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict or MAX_TOKENS,
            "num_ctx": num_ctx,
        },
        "think": think,
    }
    if tools:
        payload["tools"] = tools

    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
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
    deep_dive: bool = False        # use the 14B model
    full_coverage: bool = False    # disable verse truncation, cite every retrieved verse
    model_override: str | None = None  # optional: force a specific OpenRouter model for this request
    local_only: bool = False       # skip OpenRouter entirely, force local model (useful when quota hit)


# ── model status tracking ─────────────────────────────────────────────────────

_model_status = {"state": "connecting", "model": ""}  # connecting | ready | error


@app.get("/")
async def index():
    html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    # Inject a dynamic model badge + polling script
    badge_html = (
        f'<span id="model-badge" style="'
        f'font-size:0.7em;color:#f59e0b;background:#1e293b;'
        f'padding:3px 10px;border-radius:12px;white-space:nowrap;'
        f'border:1px solid #92400e;">'
        f'⏳ connecting to {OLLAMA_MODEL}...</span>'
    )
    poll_script = """
<script>
(function pollModelStatus() {
  const badge = document.getElementById('model-badge');
  const sendBtn = document.getElementById('send-btn');
  const input = document.getElementById('input');
  if (!badge) return;
  fetch('/model-status').then(r => r.json()).then(d => {
    if (d.state === 'ready') {
      badge.textContent = '🟢 ' + d.model + ' (local · free)';
      badge.style.color = '#94a3b8';
      badge.style.borderColor = '#334155';
      if (sendBtn) sendBtn.disabled = false;
      if (input) input.placeholder = 'Ask about the Quran...';
    } else if (d.state === 'error') {
      badge.textContent = '🔴 model error';
      badge.style.color = '#ef4444';
      badge.style.borderColor = '#991b1b';
    } else {
      badge.textContent = '⏳ loading ' + d.model + '...';
      if (sendBtn) sendBtn.disabled = true;
      if (input) input.placeholder = 'Model loading, please wait...';
      setTimeout(pollModelStatus, 1500);
    }
  }).catch(() => setTimeout(pollModelStatus, 2000));
})();
</script>
"""
    html = html.replace("</h1>", f"</h1>{badge_html}", 1)
    html = html.replace("</body>", f"{poll_script}</body>", 1)
    return HTMLResponse(html)


@app.get("/model-status")
async def model_status():
    return _model_status


@app.get("/model-info")
async def model_info():
    return {"model": OLLAMA_MODEL, "backend": "ollama", "cost": "free"}


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


DEEP_DIVE_MODEL = "qwen3:14b"  # escalation model for complex questions


@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        _agent_stream(req.message, req.history,
                      deep_dive=req.deep_dive,
                      full_coverage=req.full_coverage,
                      model_override=req.model_override,
                      local_only=req.local_only),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _extract_priming_keywords(message: str) -> list:
    """Pull 2-3 salient words from the user's message for a fast pre-flight lookup."""
    # Drop common stop words + question shells; return uniques by length (longer = more salient)
    stop = {
        "the","a","an","is","are","was","were","be","been","being","to","of","in","on","at",
        "for","with","by","and","or","but","not","no","do","does","did","have","has","had",
        "what","who","when","where","why","how","which","that","this","these","those","it",
        "its","about","say","says","tell","me","i","you","your","me","can","could","would",
        "should","quran","verse","verses","god","allah",
    }
    import re as _re
    words = _re.findall(r"[a-zA-Z]{4,}", message.lower())
    uniq = []
    seen = set()
    for w in words:
        if w in stop or w in seen:
            continue
        seen.add(w)
        uniq.append(w)
    # prefer longer words (proxy for salience)
    uniq.sort(key=len, reverse=True)
    return uniq[:3]


def _priming_graph_update(session, message: str) -> dict | None:
    """Fast (<200ms) lookup that returns a few seed verses to render before the LLM responds."""
    kws = _extract_priming_keywords(message)
    if not kws:
        return None
    try:
        seed_verses = session.run(
            """
            UNWIND $kws AS kw
            MATCH (k:Keyword) WHERE toLower(k.name) CONTAINS kw
            MATCH (k)<-[:MENTIONS]-(v:Verse)
            WITH v, count(DISTINCT k) AS matchCount
            ORDER BY matchCount DESC, v.reference
            LIMIT 6
            RETURN v.reference AS id, v.text AS text, v.surahName AS surahName,
                   v.arabicText AS arabicText
            """,
            kws=kws,
        ).data()
    except Exception:
        return None
    if not seed_verses:
        return None
    nodes = [
        {
            "id": v["id"],
            "type": "verse",
            "verseId": v["id"],
            "label": f"[{v['id']}]",
            "surahName": v["surahName"] or "",
            "text": (v["text"] or "")[:240],
            "arabicText": v["arabicText"] or "",
        }
        for v in seed_verses
    ]
    return {"nodes": nodes, "links": [], "active": [v["id"] for v in seed_verses[:3]]}


async def _agent_stream(message: str, history: list,
                         deep_dive: bool = False,
                         full_coverage: bool = False,
                         model_override: str | None = None,
                         local_only: bool = False):
    import queue as tqueue
    q: tqueue.SimpleQueue = tqueue.SimpleQueue()

    # Routing decision:
    # - local_only → skip OpenRouter entirely (use when quota hit or offline)
    # - model_override + key → use that OpenRouter model for this request
    # - deep_dive + key → default OpenRouter model
    # - deep_dive no key → local 14B
    # - otherwise → local 8B
    use_openrouter = not local_only and bool(OPENROUTER_API_KEY) and (deep_dive or model_override)
    if use_openrouter:
        active_model = model_override or OPENROUTER_MODEL
        active_backend = "openrouter"
    elif deep_dive or local_only:
        active_model = DEEP_DIVE_MODEL
        active_backend = "ollama"
    else:
        active_model = OLLAMA_MODEL
        active_backend = "ollama"

    # Log the routing decision and also tell the UI
    print(f"  [chat] {active_backend} :: {active_model}")
    try:
        q.put({"t": "tool", "name": "Model", "args": active_backend,
               "summary": active_model})
    except Exception:
        pass

    def run():
        # Allow the fallback branch to rewrite these closures
        nonlocal active_model, active_backend
        # Start a reasoning-memory recording for this query
        try:
            rec = reasoning_memory.start_query(
                text=message,
                backend=f"{active_backend}:{active_model}",
                deep_dive=deep_dive,
            )
        except Exception as _e:
            print(f"  [reasoning_memory] start failed: {_e}")
            rec = None
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

            # ── reasoning memory: find past similar queries + their tool playbooks ──
            try:
                similar = reasoning_memory.find_similar_queries(message, top_k=3, min_sim=0.7)
                # Only surface traces that actually succeeded + had multiple tool calls
                useful = [s for s in similar if s.get("status") == "completed"
                          and (s.get("tool_steps") or [])]
                if useful:
                    playbook_lines = ["\n=== Past reasoning playbooks (similar queries you've answered) ===\n"]
                    for s in useful[:2]:  # cap at 2 to avoid bloat
                        playbook_lines.append(
                            f"\nPrevious question (sim={s['score']:.2f}, "
                            f"{s['turns']} turns, {s['citation_count']} citations):"
                        )
                        playbook_lines.append(f"  Q: {s['text']}")
                        playbook_lines.append(f"  Tool sequence:")
                        for step in (s["tool_steps"] or [])[:8]:
                            playbook_lines.append(
                                f"    - {step['tool_name']}({step['args'][:80]}) "
                                f"-> {step['summary'][:80]}"
                            )
                    playbook_lines.append(
                        "\nConsider this playbook when choosing tools. You may reuse "
                        "the same keywords/queries that worked before, or diverge if "
                        "the new question needs different angles.\n"
                        "=== end playbooks ==="
                    )
                    system_content = system_content + "\n".join(playbook_lines)
                    msgs[0] = {"role": "system", "content": system_content}
                    q.put({"t": "tool", "name": "Reasoning memory",
                           "args": f"{len(useful)} playbook(s)",
                           "summary": f"Found {len(useful)} similar past traces"})
            except Exception as rme:
                print(f"  [reasoning_memory] playbook lookup failed: {rme}")

            msgs.append({"role": "user", "content": message})

            full_text = ""
            turn = 0

            with driver.session(database=NEO4J_DB) as session:
                # Instant "priming" graph update — fast keyword search on user's message.
                # Shows a cluster forming before the LLM even starts responding.
                try:
                    prime = _priming_graph_update(session, message)
                    if prime:
                        q.put({
                            "t": "graph_update",
                            "nodes": prime["nodes"],
                            "links": prime["links"],
                            "active": prime["active"],
                        })
                        q.put({"t": "tool", "name": "Priming search",
                               "args": ", ".join(_extract_priming_keywords(message)),
                               "summary": f"Found {len(prime['nodes'])} candidate verses"})
                except Exception as _e:
                    print(f"  [priming] {_e}")

                # Track tool calls to enforce multi-tool search on topical questions
                total_tool_calls = 0
                tools_used_so_far = set()
                # A question is "topical" if it's longer than a simple ref lookup.
                # Simple lookups like "verse 2:255" don't need exhaustive retrieval.
                _is_simple_lookup = bool(re.search(r"\bverse\s*\d+[:.]\d+", message.lower())) \
                                    or len(message.split()) < 4

                def _needs_more_tools() -> bool:
                    """Enforce at least 2 tool calls with keyword + semantic on topical questions."""
                    if _is_simple_lookup:
                        return False
                    if total_tool_calls < 2:
                        return True
                    # Require both a keyword-family and a semantic search
                    has_kw = bool(tools_used_so_far & {"search_keyword", "traverse_topic"})
                    has_sem = "semantic_search" in tools_used_so_far
                    return not (has_kw and has_sem)

                while turn < MAX_TOOL_TURNS:
                    turn += 1

                    print(f"  [turn {turn}] backend={active_backend} model={active_model} msgs={len(msgs)}")
                    try:
                        if active_backend == "openrouter":
                            resp = openrouter_chat(
                                model=active_model,
                                messages=msgs,
                                tools=OLLAMA_TOOLS,
                                temperature=0.3,
                                num_predict=8192 if full_coverage else MAX_TOKENS,
                            )
                        else:
                            resp = ollama_chat(
                                model=active_model,
                                messages=msgs,
                                tools=OLLAMA_TOOLS,
                                temperature=0.3,
                                num_ctx=40960 if full_coverage else 24576,
                                num_predict=8192 if full_coverage else MAX_TOKENS,
                                think=full_coverage,
                            )
                    except Exception as e:
                        err_body = ""
                        if hasattr(e, 'response') and e.response is not None:
                            try: err_body = e.response.text[:500]
                            except Exception: pass
                        print(f"  [api error] {active_backend}: {e}  body={err_body!r}")
                        # Graceful fallback: OpenRouter rate-limited or errored → local 14B
                        if active_backend == "openrouter":
                            print(f"  [fallback] OpenRouter -> local 14B")
                            active_backend = "ollama"
                            active_model = DEEP_DIVE_MODEL
                            q.put({"t": "tool", "name": "Fallback",
                                   "args": "OpenRouter unavailable",
                                   "summary": "Switching to local 14B"})
                            # Strip OpenAI-style tool_calls from assistant messages + add tool_call_id to
                            # tool messages to avoid format mismatch when retrying on Ollama
                            for m in msgs:
                                if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
                                    for tc in m["tool_calls"]:
                                        args = tc.get("function", {}).get("arguments")
                                        if isinstance(args, str):
                                            try: tc["function"]["arguments"] = json.loads(args)
                                            except Exception: pass
                            resp = ollama_chat(
                                model=active_model,
                                messages=msgs,
                                tools=OLLAMA_TOOLS,
                                temperature=0.3,
                                num_ctx=40960 if full_coverage else 24576,
                                num_predict=8192 if full_coverage else MAX_TOKENS,
                                think=full_coverage,
                            )
                        else:
                            raise

                    msg = resp.get("message", {})
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])

                    # If the model tries to respond without enough retrieval, nudge it
                    if not tool_calls and _needs_more_tools():
                        missing = []
                        if "search_keyword" not in tools_used_so_far and "traverse_topic" not in tools_used_so_far:
                            missing.append("search_keyword or traverse_topic")
                        if "semantic_search" not in tools_used_so_far:
                            missing.append("semantic_search")
                        nudge = (
                            "You have not done enough retrieval yet. "
                            f"Before writing any answer, you MUST call: {', '.join(missing)}. "
                            "Call them now with appropriate arguments for the user's question."
                        )
                        msgs.append({"role": "user", "content": nudge})
                        # Don't emit this to UI — it's internal discipline
                        continue

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
                        # OpenAI/OpenRouter provides `id` — we need to echo it back
                        tool_call_id = tc.get("id", "")

                        total_tool_calls += 1
                        tools_used_so_far.add(tool_name)

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

                        # Execute the tool — also time it for reasoning memory
                        import time as _t
                        _t0 = _t.time()
                        result_str = dispatch_tool(session, tool_name, tool_args, user_query=message)
                        tool_duration_ms = int((_t.time() - _t0) * 1000)

                        # Summary for UI
                        tool_ok = True
                        result_cite_count = 0
                        try:
                            res = json.loads(result_str)
                            if "error" in res:
                                summary = f"Error: {res['error']}"
                                tool_ok = False
                            elif "total_verses" in res:
                                summary = f"Found {res['total_verses']} verses"
                                result_cite_count = int(res.get("total_verses") or 0)
                            elif "verse_id" in res:
                                summary = f"[{res['verse_id']}]"
                                result_cite_count = 1
                            elif "hops" in res:
                                summary = f"Path in {res['hops']} hops"
                            elif "verse_count" in res:
                                summary = f"{res.get('surah_name','')} — {res['verse_count']} verses"
                                result_cite_count = int(res.get("verse_count") or 0)
                            else:
                                summary = "Done"
                        except Exception:
                            summary = "Done"
                            tool_ok = False

                        # Record this tool call in reasoning memory
                        if rec is not None:
                            try:
                                rec.log_tool_call(
                                    turn=turn, order=total_tool_calls,
                                    tool_name=tool_name, args=tool_args,
                                    summary=summary, ok=tool_ok,
                                    duration_ms=tool_duration_ms,
                                    result_citation_count=result_cite_count,
                                )
                            except Exception as _e:
                                print(f"  [reasoning_memory] log_tool_call failed: {_e}")

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

                        # Send compressed tool result back.
                        # full_coverage=True disables truncation — the model sees every retrieved verse in full.
                        # tool_call_id is REQUIRED for OpenRouter (OpenAI format); Ollama ignores it.
                        compressed = compress_tool_result(tool_name, result_str, full_coverage=full_coverage)
                        tool_msg = {"role": "tool", "content": compressed}
                        if tool_call_id:
                            tool_msg["tool_call_id"] = tool_call_id
                            tool_msg["name"] = tool_name  # OpenAI convention
                        msgs.append(tool_msg)

                # Fetch verse texts for tooltips
                refs = _extract_verse_refs(full_text)

                # ── Citation density check + one-shot retry ──
                # For topical questions, expect at least 5 unique citations.
                # If below, re-prompt for expansion with the same tool context.
                MIN_CITATIONS = 5
                if not _is_simple_lookup and len(refs) < MIN_CITATIONS and turn < MAX_TOOL_TURNS:
                    q.put({"t": "tool", "name": "Citation check",
                           "args": f"{len(refs)} citations",
                           "summary": f"Below threshold ({MIN_CITATIONS}) — expanding"})
                    expand = (
                        f"Your answer so far has only {len(refs)} verse citations. "
                        f"A thorough answer needs at least {MIN_CITATIONS}-10 citations. "
                        "Review the tool results you have, add more thematic sections with additional "
                        "verse references that you may have missed, and rewrite the answer. "
                        "Every paragraph must contain at least one [surah:verse] citation."
                    )
                    msgs.append({"role": "user", "content": expand})
                    # Tell the client to replace the answer
                    if active_backend == "openrouter":
                        retry_resp = openrouter_chat(
                            model=active_model,
                            messages=msgs,
                            tools=OLLAMA_TOOLS,
                            temperature=0.3,
                            num_predict=8192 if full_coverage else MAX_TOKENS,
                        )
                    else:
                        retry_resp = ollama_chat(
                            model=active_model,
                            messages=msgs,
                            tools=OLLAMA_TOOLS,
                            temperature=0.3,
                            num_ctx=40960 if full_coverage else 24576,
                            num_predict=8192 if full_coverage else MAX_TOKENS,
                            think=full_coverage,
                        )
                    retry_msg = retry_resp.get("message", {})
                    retry_content = retry_msg.get("content", "")
                    if retry_content and retry_content.strip():
                        full_text = retry_content  # replace the original
                        q.put({"t": "retry", "d": retry_content})
                    refs = _extract_verse_refs(full_text)

                verses = _fetch_verses(session, refs)

                # Save to answer cache
                try:
                    save_answer(message, full_text, verses)
                except Exception as ce:
                    print(f"  [cache] save error: {ce}")

                # Finalize the reasoning-memory trace
                if rec is not None:
                    try:
                        rec.finish(
                            answer_text=full_text,
                            citation_count=len(refs),
                            status="retry_used" if turn > MAX_TOOL_TURNS else "completed",
                        )
                    except Exception as _e:
                        print(f"  [reasoning_memory] finish failed: {_e}")

                q.put({"t": "done", "verses": verses})

        except requests.ConnectionError:
            if rec is not None:
                try: rec.mark_failed("ollama connection")
                except Exception: pass
            q.put({"t": "error", "d": "Cannot connect to Ollama. Make sure it's running: ollama serve"})
        except requests.Timeout:
            if rec is not None:
                try: rec.mark_failed("ollama timeout")
                except Exception: pass
            q.put({"t": "error", "d": "Ollama request timed out (600s). Try a smaller model or shorter question."})
        except Exception as e:
            if rec is not None:
                try: rec.mark_failed(str(e)[:300])
                except Exception: pass
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


def _preload_model():
    """Background thread: verify Ollama + pre-load model into VRAM."""
    global OLLAMA_MODEL
    try:
        _model_status["model"] = OLLAMA_MODEL

        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if OLLAMA_MODEL not in models:
            matches = [m for m in models if OLLAMA_MODEL.split(":")[0] in m]
            if matches:
                print(f"  Model '{OLLAMA_MODEL}' not found, using '{matches[0]}'")
                OLLAMA_MODEL = matches[0]
                _model_status["model"] = OLLAMA_MODEL
            else:
                print(f"  WARNING: Model '{OLLAMA_MODEL}' not found. Available: {models}")
                _model_status["state"] = "error"
                return

        # Pre-load model into VRAM so the first question is fast
        print(f"  Pre-loading {OLLAMA_MODEL} into GPU...")
        requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "think": False,
            "options": {"num_predict": 1},
        }, timeout=120)
        _model_status["state"] = "ready"
        print(f"  Model ready in VRAM")
    except Exception as e:
        print(f"  WARNING: Ollama issue ({e})")
        _model_status["state"] = "error"


if __name__ == "__main__":
    _model_status["model"] = OLLAMA_MODEL

    print(f"\n[FREE] Model: {OLLAMA_MODEL}")
    print(f"[FREE] Cost: $0.00 (local)")
    print(f"[FREE] Quran Graph: http://localhost:{OLLAMA_PORT}\n")

    # Start model pre-load in background so the UI is available immediately
    threading.Thread(target=_preload_model, daemon=True).start()

    import uvicorn
    webbrowser.open(f"http://localhost:{OLLAMA_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=OLLAMA_PORT, log_level="info")
