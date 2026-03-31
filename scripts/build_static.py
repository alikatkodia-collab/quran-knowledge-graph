#!/usr/bin/env python3
"""
Build static HTML files with embedded JSON data for GitHub Pages deployment.

Takes the HTML pages that normally call API endpoints and inlines the data
directly, so they work without a backend server.

Usage:
    python scripts/build_static.py [--output dist]
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
AUTORESEARCH = ROOT / "autoresearch"
DATA = ROOT / "data"


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


def build_graph_data():
    """Build the data that /api/deductions/graph would return."""
    CATEGORY_COLORS = {
        "monotheism_and_gods_nature": "#10b981",
        "prophecy_and_revelation": "#3b82f6",
        "moral_law_and_ethics": "#f59e0b",
        "worship_and_ritual": "#8b5cf6",
        "afterlife_and_judgment": "#ef4444",
        "creation_and_cosmology": "#06b6d4",
        "prophetic_narratives": "#f97316",
        "social_law": "#ec4899",
        "covenant_and_obedience": "#14b8a6",
        "divine_mercy_and_forgiveness": "#a3e635",
        "warfare_and_struggle": "#dc2626",
        "knowledge_and_wisdom": "#818cf8",
        "mathematical_miracle": "#fbbf24",
        "uncategorized": "#475569",
    }
    data = load_json(AUTORESEARCH / "meta_knowledge_graph.json", {"nodes": [], "edges": []})
    nodes = []
    for n in data.get("nodes", []):
        cat_id = n.get("id", "uncategorized")
        nodes.append({
            "id": cat_id,
            "label": n.get("label", cat_id),
            "description": n.get("description", ""),
            "verse_count": n.get("verse_count", 0),
            "color": CATEGORY_COLORS.get(cat_id, "#475569"),
        })
    edges = [{"source": e["source"], "target": e["target"],
              "weight": e["weight"], "avg_quality": e.get("avg_quality", 0)}
             for e in data.get("edges", [])]
    return {"nodes": nodes, "edges": edges}


def build_insights_data():
    """Build the data that /api/deductions/insights would return."""
    raw = load_json(AUTORESEARCH / "synthesized_insights.json", [])
    verses = load_json(DATA / "verses.json", [])
    lookup = {v["verse_id"]: v["text"] for v in verses}

    raw.sort(key=lambda x: x.get("avg_quality", 0), reverse=True)
    insights = []
    for item in raw[:100]:
        vp = item.get("verse_pair", [])
        insights.append({
            "category": item.get("category", "uncategorized"),
            "verse_pair": vp,
            "bridge_keywords": item.get("bridge_keywords", []),
            "conclusion": item.get("best_conclusion", ""),
            "quality": item.get("avg_quality", 0),
            "verse_texts": {v: lookup.get(v, "") for v in vp},
        })
    return {"insights": insights}


def build_status_data():
    """Build static status data."""
    total = 0
    jsonl = AUTORESEARCH / "all_deductions.jsonl"
    if jsonl.exists():
        with open(jsonl) as f:
            total = sum(1 for _ in f)
    return {
        "total_deductions": total,
        "categories_found": 13,
        "last_round_time": "static build",
        "is_running": False,
    }


def build_stats_data():
    """Build comprehensive stats for visualizations."""
    from collections import Counter
    jsonl = AUTORESEARCH / "all_deductions.jsonl"
    rule_counts = Counter()
    bridge_freq = Counter()
    total = 0

    if jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line.strip())
                total += 1
                rule_counts[d.get("rule", "unknown")] += 1
                for kw in d.get("bridge_keywords", []):
                    bridge_freq[kw] += 1

    return {
        "total_deductions": total,
        "rule_counts": dict(rule_counts),
        "top_bridge_keywords": [{"keyword": k, "count": v}
                                for k, v in bridge_freq.most_common(50)],
        "optimization_results": {
            "graph_before": 75.47, "graph_after": 81.74,
            "retrieval_before": 58.99, "retrieval_after": 68.44,
            "cluster_coherence_before": 35.53, "cluster_coherence_after": 59.87,
        },
    }


def inject_static_data(html, endpoint_data):
    """Replace fetch() calls with inline data in HTML."""
    # Add a script block with all the data
    data_script = "<script>\nwindow.__STATIC_DATA__ = " + json.dumps(endpoint_data, ensure_ascii=False) + ";\n"
    data_script += """
// Override fetch for API calls to use static data
const _originalFetch = window.fetch;
window.fetch = function(url, opts) {
    if (typeof url === 'string') {
        for (const [endpoint, data] of Object.entries(window.__STATIC_DATA__)) {
            if (url.includes(endpoint)) {
                return Promise.resolve(new Response(JSON.stringify(data), {
                    status: 200, headers: {'Content-Type': 'application/json'}
                }));
            }
        }
    }
    return _originalFetch.apply(this, arguments);
};
"""
    data_script += "</script>\n"

    # Inject before closing </head>
    html = html.replace("</head>", data_script + "</head>")
    return html


def build(output_dir):
    """Build all static files."""
    os.makedirs(output_dir, exist_ok=True)

    print("Building static data...")
    graph_data = build_graph_data()
    insights_data = build_insights_data()
    status_data = build_status_data()
    stats_data = build_stats_data()

    endpoint_data = {
        "/api/deductions/graph": graph_data,
        "/api/deductions/insights": insights_data,
        "/api/deductions/status": status_data,
        "/api/deductions/stats": stats_data,
    }

    print(f"  Total deductions: {status_data['total_deductions']:,}")

    # Build each HTML page
    pages = {
        "deductions.html": endpoint_data,
        "visualizations.html": endpoint_data,
        "presentation.html": {},  # No API calls
        "stats.html": {},
    }

    for page, data in pages.items():
        src = ROOT / page
        if not src.exists():
            print(f"  Skipping {page} (not found)")
            continue

        html = src.read_text(encoding="utf-8")
        if data:
            html = inject_static_data(html, data)

        # Fix relative links for static hosting
        html = html.replace('href="/"', 'href="index.html"')
        html = html.replace('href="/stats"', 'href="stats.html"')
        html = html.replace('href="/deductions"', 'href="deductions.html"')
        html = html.replace('href="/visualizations"', 'href="visualizations.html"')
        html = html.replace('href="/presentation"', 'href="presentation.html"')

        dst = os.path.join(output_dir, page)
        with open(dst, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Built {page} ({len(html):,} bytes)")

    # Create index.html that redirects to deductions
    index = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta http-equiv="refresh" content="0;url=deductions.html">
<title>Quran Knowledge Graph</title></head>
<body><p>Redirecting to <a href="deductions.html">Deductions</a>...</p></body></html>"""
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(index)

    print(f"\nStatic site built to {output_dir}/")
    print(f"Preview: python3 -m http.server 8000 --directory {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist", help="Output directory")
    args = parser.parse_args()
    build(args.output)
