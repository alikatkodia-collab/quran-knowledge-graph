# Implementation Prompt: Knowledge Graph Chat + 3D Visualization

Copy everything below this line and paste it into a new Claude Code session to implement the same GUI, chat, and 3D visualization for your own knowledge graph.

---

## PROMPT START

I have a reference implementation of an interactive knowledge graph explorer that I want you to replicate for my project. The reference is at:
**https://github.com/alikatkodia-collab/quran-knowledge-graph**

Clone it for reference:
```
git clone https://github.com/alikatkodia-collab/quran-knowledge-graph.git /tmp/reference
```

Read these 3 files thoroughly before starting — they contain the full architecture:
- `/tmp/reference/app.py` — Backend streaming, graph extraction, FastAPI
- `/tmp/reference/chat.py` — Tool schemas, system prompt, tool dispatch
- `/tmp/reference/index.html` — Frontend: SSE parsing, tooltips, 3D graph

## What This System Does

A web app with:
- **Left panel**: Chat interface where an AI agent answers questions using graph database tools
- **Right panel**: 3D force-directed graph that lights up in real-time as the AI explores
- **Hoverable citations**: Every claim in the AI's response cites a node reference like `[2:255]` — hovering shows the full text

## Architecture Overview

```
User Question
    ↓
POST /chat (FastAPI endpoint)
    ↓
Claude API (agentic tool-use loop)
    ↓ calls tools ↓
Graph Database Queries (Neo4j/your DB)
    ↓ returns results ↓
SSE Event Stream → Frontend
    ↓
- Text deltas stream into chat bubble (markdown rendered)
- Tool calls shown as expandable <details> blocks
- Graph updates inject nodes/links into 3D visualization
- On completion, all cited references get hoverable tooltips
```

## The 5 Components You Need to Build

### 1. Tool Definitions (`chat.py`)

Define your graph tools as an array of Anthropic tool schemas:

```python
TOOLS = [
    {
        "name": "your_tool_name",
        "description": "What this tool does — be specific so Claude knows when to use it",
        "input_schema": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "..."}
            },
            "required": ["param"]
        }
    },
    # ... more tools
]
```

Each tool function queries your database and returns a JSON-serializable dict. Example pattern:

```python
def tool_search(session, keyword: str) -> dict:
    rows = list(session.run("YOUR QUERY HERE", kw=keyword))
    return {"keyword": keyword, "total": len(rows), "results": [...]}

def dispatch_tool(session, tool_name: str, tool_input: dict) -> str:
    if tool_name == "search": result = tool_search(session, **tool_input)
    elif tool_name == "...": ...
    return json.dumps(result)
```

### 2. System Prompt (`chat.py`)

Write a system prompt that tells Claude:
- What domain it's exploring
- What tools are available and when to use each
- **CITATION RULES** — mandatory, every claim must cite a reference inline like `[node_id]`
- How to synthesize answers from tool results
- To stay grounded in data, not general knowledge

Reference pattern from `chat.py` SYSTEM_PROMPT — adapt the citation rules and search mandate for your domain.

### 3. Backend Streaming (`app.py`)

The `/chat` endpoint runs Claude's agentic loop and streams **Server-Sent Events**. There are 5 event types:

```python
# Text delta (Claude's response text, streamed incrementally)
yield f"data: {json.dumps({'t': 'text', 'd': text_chunk})}\n\n"

# Tool execution (shown as expandable block in UI)
yield f"data: {json.dumps({'t': 'tool', 'name': name, 'args': args_str, 'summary': summary})}\n\n"

# Graph update (nodes + links to add to 3D visualization)
yield f"data: {json.dumps({'t': 'graph_update', 'nodes': [...], 'links': [...], 'active': [...]})}\n\n"

# Done (includes all referenced node texts for tooltips)
yield f"data: {json.dumps({'t': 'done', 'verses': {'node_id': 'full text', ...}})}\n\n"

# Error
yield f"data: {json.dumps({'t': 'error', 'd': error_message})}\n\n"
```

**Critical pattern**: The agentic loop runs in a **daemon thread** pushing events onto a `queue.SimpleQueue()`. The async generator polls the queue with `asyncio.sleep(0.05)`. A `None` sentinel signals completion.

**Graph extraction** — after each tool call, extract nodes and links from the result:

```python
def _graph_for_tool(tool_name, tool_input, result_dict):
    nodes, links, active = [], [], []
    # Parse result_dict to find entities and relationships
    # Node format: {"id": "v:123", "type": "entity", "label": "...", "text": "..."}
    # Link format: {"source": "v:123", "target": "v:456", "type": "related"}
    # Active: list of node IDs to highlight
    return {"nodes": nodes, "links": links, "active": active}
```

**Reference text fetching** — after the loop, extract all `[X:Y]` references from the full response and batch-fetch their texts from the database for tooltips.

### 4. Frontend Chat + SSE Parsing (`index.html`)

The frontend consumes the SSE stream using `fetch()` + `ReadableStream`:

```javascript
const resp = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history })
});

const reader = resp.body.getReader();
const dec = new TextDecoder();
let buf = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop(); // Keep incomplete line

    for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const ev = JSON.parse(line.slice(6));

        if (ev.t === 'text')         addText(ev.d);           // Append to markdown bubble
        else if (ev.t === 'tool')    addTool(ev.name, ev.args, ev.summary);  // Expandable details
        else if (ev.t === 'graph_update') applyGraphUpdate(ev); // 3D graph
        else if (ev.t === 'done')    finalize(ev.verses);      // Wrap references as tooltips
        else if (ev.t === 'error')   setError(ev.d);
    }
}
```

**Tooltip system** — after stream completes, walk all text nodes in the response bubble, find patterns matching `[X:Y]`, and wrap them in `<span class="vref" data-vid="X:Y" data-v="full text">`. On mouseover, position a tooltip above the reference showing the full text.

**Markdown rendering** — use `marked.js` to render Claude's response as HTML in real-time as text deltas arrive.

### 5. 3D Graph Visualization (`index.html`)

Uses **Three.js** + **3d-force-graph** library.

```html
<script src="https://cdn.jsdelivr.net/npm/three@0.160/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/3d-force-graph@1/dist/3d-force-graph.min.js"></script>
```

**Initial state**: All nodes pre-positioned on a Fibonacci sphere (one cluster per category/group). Rendered as a single `THREE.Points` buffer geometry for performance (thousands of nodes, single draw call).

**When graph_update arrives**:
1. Deactivate previously active nodes
2. Pin new verse/entity nodes to their pre-computed positions (`node.fx/fy/fz`)
3. Pin keyword/concept nodes to the centroid of their connected nodes
4. Add to force graph data
5. Mark active nodes (glow effect, larger sphere, visible label)

**Node rendering** (custom `nodeThreeObject`):
- Idle nodes: tiny dim spheres
- Conversation nodes: full spheres with labels
- Active nodes: bright spheres with aura glow effect (larger transparent BackSide sphere)

**Color scheme** (dark theme):
- Background: `#060a14`
- Cards/panels: `#0f1a2e`
- Primary accent: `#10b981` (emerald green)
- Secondary accent: `#f59e0b` (gold)
- Text: `#cbd5e1`
- Muted: `#64748b`
- Borders: `#1e293b`

**Camera controls**: Orbit (drag), zoom (scroll), WASD fly-through with Q/E for up/down.

**Node click**: Shows info card at bottom of graph panel with node details.

## Layout

```
┌──────────────────────────────────────────────────┐
│ Header: Title | [Tab1] [Tab2] | Description      │
├────────────────────┬─────────────────────────────┤
│                    │                             │
│   Chat Panel       │      3D Graph Panel         │
│   (40% width)      │      (60% width)            │
│                    │                             │
│   ┌──────────┐    │   [Clear connections btn]    │
│   │ Messages │    │                             │
│   │ (scroll) │    │      ● ●  ●                 │
│   │          │    │     ● ──── ●                │
│   │          │    │      ●  ●                   │
│   └──────────┘    │                             │
│   [Examples]       │   [Node info card]          │
│   [Input] [Send]   │   [WASD hint]              │
└────────────────────┴─────────────────────────────┘
```

## Adapting for Your Domain

Replace these domain-specific pieces:

1. **Tool functions** — Write queries for YOUR graph database
2. **System prompt** — Describe YOUR domain, YOUR tools, YOUR citation format
3. **`_graph_for_tool()`** — Extract nodes/links from YOUR tool results
4. **Reference regex** — Change `\[(\d+:\d+)\]` to match YOUR node ID format
5. **Galaxy positions** — Change the clustering logic to match YOUR node categories
6. **Node colors** — Assign colors per YOUR node types
7. **Example questions** — Update the preset buttons

Everything else (SSE streaming, tooltip system, 3D rendering, chat UI, markdown parsing) is domain-agnostic and works as-is.

## Dependencies

```bash
# Python
pip install fastapi uvicorn anthropic python-dotenv neo4j

# Optional (for semantic search)
pip install sentence-transformers

# Frontend (loaded via CDN, no install needed)
# three.js, 3d-force-graph, marked.js
```

## File Structure to Create

```
your-project/
  chat.py        — Tools array, system prompt, dispatch_tool, tool functions
  app.py         — FastAPI server, /chat streaming endpoint, _graph_for_tool
  index.html     — Full frontend (chat + 3D graph + tooltips)
  .env           — API keys and database credentials
```

Start by reading the reference implementation files, then adapt each component for your domain. The reference repo is at:
**https://github.com/alikatkodia-collab/quran-knowledge-graph**
