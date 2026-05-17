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
    {"type": "function", "function": {
        "name": "get_code19_features",
        "description": "Khalifa Code-19 arithmetic features (verse counts, mysterious-letter frequencies, divisibility-by-19). Use for any claim about the mathematical miracle of 19. Cannot be hallucinated — pure arithmetic over Arabic text.",
        "parameters": {"type": "object",
                       "properties": {"scope": {"type": "string", "enum": ["global", "sura", "verse"]},
                                      "target": {"type": "string", "description": "sura number or verseId — required unless scope=global"}},
                       "required": ["scope"]}}},
    {"type": "function", "function": {
        "name": "concept_search",
        "description": "Search by canonical CONCEPT, auto-expanding across surface variants (forgiveness finds verses using 'forgive', 'forgiver', 'forgiveness'). Use as default for thematic English keywords.",
        "parameters": {"type": "object",
                       "properties": {"concept": {"type": "string"},
                                      "top_k": {"type": "integer", "default": 30}},
                       "required": ["concept"]}}},
    {"type": "function", "function": {
        "name": "hybrid_search",
        "description": "Hybrid BM25 + BGE-M3 vector search with RRF fusion + graph enrichment. Better than semantic_search for queries with rare/specific words, names, or Arabic terms. lang='en' or 'ar'.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "top_k": {"type": "integer", "default": 20},
                                      "lang": {"type": "string", "enum": ["en", "ar"], "default": "en"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "recall_similar_query",
        "description": "Find past similar queries from the reasoning memory; returns the tools they used and the answer they produced. Use as a playbook hint, not a final answer.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "top_k": {"type": "integer", "default": 3},
                                      "min_sim": {"type": "number", "default": 0.65}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "run_cypher",
        "description": "Execute a READ-ONLY Cypher query for long-tail graph questions. Schema: Verse(verseId, text, arabicText), Sura, Keyword, ArabicRoot, Lemma; edges MENTIONS, RELATED_TO, MENTIONS_ROOT, SIMILAR_PHRASE, NEXT_VERSE, CONTAINS, SUPPORTS/ELABORATES/QUALIFIES/CONTRASTS/REPEATS. Forbidden: CREATE/MERGE/DELETE/SET/REMOVE/DETACH.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "MATCH/RETURN query"},
                                      "row_limit": {"type": "integer", "description": "default 100, max 500"}},
                       "required": ["query"]}}},
]

# Parse --model early so it's available at module level
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model", default=DEFAULT_MODEL)
_parser.add_argument("--port", type=int, default=8085)
_parser.add_argument(
    "--openrouter",
    action="store_true",
    help="Default to the OpenRouter free model for every chat request "
         "(requires OPENROUTER_API_KEY in .env). Equivalent to PREFER_OPENROUTER=1.",
)
_args, _ = _parser.parse_known_args()
OLLAMA_MODEL = _args.model
OLLAMA_PORT = _args.port
# Default backend toggle. The flag wins; env var is a fallback so desktop
# shortcuts that can't easily pass flags can still set it via .env.
PREFER_OPENROUTER = _args.openrouter or os.getenv("PREFER_OPENROUTER", "").strip().lower() in ("1", "true", "yes")

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

# ── shared agent loop (moved to shared_agent.py in Phase 3a) ──────────────────
# HTTP clients (_openrouter_chat, _ollama_chat), pure helpers
# (_extract_verse_refs, _fetch_verses, _extract_priming_keywords,
# _priming_graph_update, _graph_for_tool) and TOOL_LABELS now live in
# shared_agent.py. The agent loop body itself (was _agent_stream) is
# shared_agent.agent_stream — see the thin wrapper below.

from shared_agent import (
    AgentCollaborators,
    AgentConfig,
    FallbackBackend,
    agent_stream as _shared_agent_stream,
)


# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI()


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
    # Reflect the default routing decision so the UI banner matches what
    # chat requests will actually use. Mirrors the logic at the top of
    # _agent_stream(): PREFER_OPENROUTER + key → OpenRouter by default.
    if PREFER_OPENROUTER and OPENROUTER_API_KEY:
        return {"model": OPENROUTER_MODEL, "backend": "openrouter", "cost": "free"}
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


# ── Sefaria-inspired ref resolver / linker API ────────────────────────────────

class ResolveRefsRequest(BaseModel):
    text: str

@app.post("/api/resolve_refs")
async def api_resolve_refs(req: ResolveRefsRequest):
    """Find Quranic citations in the given text, return resolved verseIds."""
    from ref_resolver import resolve_refs
    matches = resolve_refs(req.text)
    return {
        "input_length": len(req.text),
        "match_count": len(matches),
        "matches": [
            {"start": m.start, "end": m.end, "raw": m.raw,
             "verse_id": m.canonical, "kind": m.kind, "confidence": m.confidence}
            for m in matches
        ],
    }

@app.get("/api/verse/{verse_id}")
async def api_get_verse(verse_id: str):
    """Return one verse by reference (e.g. '2:255')."""
    if ":" not in verse_id:
        return {"error": "verseId must be in format 'surah:verseNum'"}
    with driver.session(database=NEO4J_DB) as s:
        row = s.run("""
            MATCH (v:Verse)
            WHERE v.reference = $vid OR v.verseId = $vid
            RETURN v.reference AS id, v.text AS text,
                   v.arabicText AS arabic, v.surahName AS surahName,
                   v.surah AS surah, v.verseNum AS verseNum
            LIMIT 1
        """, vid=verse_id).single()
    if not row or not row.get("id"):
        return {"error": f"verse {verse_id} not found"}
    return {
        "verse_id": row["id"],
        "surah": row["surah"],
        "surah_name": row["surahName"],
        "verse_num": row["verseNum"],
        "text": row["text"],
        "arabic": row["arabic"],
    }


@app.get("/quran_linker.js")
async def quran_linker_js():
    """Single-file JS widget — drops into any page to auto-link Quranic refs."""
    from fastapi.responses import Response
    from pathlib import Path as _P
    js_path = _P(__file__).parent / "static" / "quran_linker.js"
    if not js_path.exists():
        return Response(content="// quran_linker.js not built yet", media_type="application/javascript")
    return Response(content=js_path.read_text(encoding="utf-8"),
                    media_type="application/javascript")


DEEP_DIVE_MODEL = "qwen3:14b"  # escalation model for complex questions


# ── AgentConfig — bound to app_free's globals at import time ──────────────────
# Built once and passed into every shared_agent.agent_stream() call.
AGENT_CONFIG = AgentConfig(
    backend="ollama",
    default_model=OLLAMA_MODEL,
    tools=OLLAMA_TOOLS,
    system_prompt=SYSTEM_PROMPT,
    max_tool_turns=MAX_TOOL_TURNS,
    max_tokens=MAX_TOKENS,
    enable_priming_graph_update=True,
    enable_reasoning_memory_playbook=True,
    enable_query_classification=True,
    enable_tool_result_compression=True,
    enable_citation_density_retry=True,
    min_citations_for_retry=5,
    enable_citation_verifier=True,  # env-gated by ENABLE_CITATION_VERIFY=1
    openrouter_model=OPENROUTER_MODEL,
    deep_dive_model=DEEP_DIVE_MODEL,
    prefer_openrouter=PREFER_OPENROUTER,
    ollama_url=OLLAMA_URL,
    openrouter_url=OPENROUTER_URL,
    required_tool_classes={
        "keyword retrieval": ["search_keyword", "traverse_topic"],
        "semantic retrieval": ["semantic_search"],
    },
    # Preserves the pre-3a-2 inline fallback: OpenRouter → local deep-dive
    # (qwen3:14b) when OpenRouter raises.
    fallback_chain=(FallbackBackend(backend="ollama", model=DEEP_DIVE_MODEL),),
)


# Per-process collaborators — built once from app_free's module globals and
# threaded through every shared_agent.agent_stream() call. Replaces the
# lazy-import seam that Phase 3a-1 left for cleanup.
AGENT_COLLABORATORS = AgentCollaborators(
    driver=driver,
    reasoning_memory=reasoning_memory,
    db_name=NEO4J_DB,
    openrouter_api_key=OPENROUTER_API_KEY,
)


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


async def _agent_stream(message: str, history: list,
                         deep_dive: bool = False,
                         full_coverage: bool = False,
                         model_override: str | None = None,
                         local_only: bool = False):
    """Back-compat shim — the loop body now lives in shared_agent.agent_stream.

    Kept so scripts/capture_baseline_trajectory.py (and any external callers)
    can keep using app_free._agent_stream. New code should call
    shared_agent.agent_stream(..., AGENT_CONFIG, ...) directly.
    """
    async for frame in _shared_agent_stream(
        message, history, AGENT_CONFIG, AGENT_COLLABORATORS,
        deep_dive=deep_dive,
        full_coverage=full_coverage,
        model_override=model_override,
        local_only=local_only,
    ):
        yield frame


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

    from startup_banner import format_startup_banner
    print(format_startup_banner(
        prefer_openrouter=PREFER_OPENROUTER,
        openrouter_api_key=OPENROUTER_API_KEY,
        openrouter_model=OPENROUTER_MODEL,
        ollama_model=OLLAMA_MODEL,
        port=OLLAMA_PORT,
    ))

    # Start model pre-load in background so the UI is available immediately
    threading.Thread(target=_preload_model, daemon=True).start()

    import uvicorn
    webbrowser.open(f"http://localhost:{OLLAMA_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=OLLAMA_PORT, log_level="info")
