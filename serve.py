#!/usr/bin/env python3
"""
Standalone server for the Quran Knowledge Graph deductions UI.
Does NOT require Neo4j or Anthropic API — serves all static pages
and deduction/comparator/explorer APIs.

Usage:
    python serve.py
    # Opens at http://localhost:8081
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Quran Knowledge Graph")

ROOT = Path(__file__).parent

# Import API routers
from deductions_api import deductions_router
app.include_router(deductions_router)

from comparator_api import comparator_router
app.include_router(comparator_router)


# Serve HTML pages
@app.get("/")
async def index():
    # Redirect to deductions since chat needs Neo4j
    return HTMLResponse("""<!DOCTYPE html><html><head>
    <meta http-equiv="refresh" content="0;url=/deductions">
    </head><body>Redirecting to <a href="/deductions">Deductions</a>...</body></html>""")

@app.get("/stats")
async def stats():
    try:
        return HTMLResponse((ROOT / "stats.html").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return HTMLResponse("<h1>stats.html not found</h1>")

@app.get("/presentation")
async def presentation():
    try:
        return HTMLResponse((ROOT / "presentation.html").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return HTMLResponse("<h1>presentation.html not found</h1>")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Quran Knowledge Graph — Standalone Server")
    print("=" * 50)
    print(f"\n  http://localhost:8081\n")
    print("  Pages:")
    print("    /deductions      — 3D meta-knowledge graph")
    print("    /visualizations  — Chart.js dashboard")
    print("    /comparator      — Parallel passage viewer")
    print("    /explorer        — Tension/contradiction browser")
    print("    /presentation    — Slide deck")
    print("    /stats           — Statistical dashboard")
    print(f"\n{'=' * 50}\n")
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
