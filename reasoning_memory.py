"""
Reasoning memory — records each /chat request as a graph of (:Query), (:ReasoningTrace),
(:ToolCall) nodes so the agent can learn from past reasoning patterns.

Inspired by Neo4j Labs' agent-memory pattern. Scoped down to what this project needs.

Schema:

  (:Query {
     queryId,             // UUID
     text,                // user's question (normalized)
     textEmbedding,       // 384-dim all-MiniLM-L6-v2 vector
     timestamp,           // ISO 8601
     backend,             // "ollama:qwen3:8b" | "openrouter:..." | etc.
     deep_dive,           // bool
   })

  (:ReasoningTrace {
     traceId,             // UUID
     total_duration_ms,
     turn_count,
     tool_call_count,
     citation_count,      // unique [X:Y] citations in final answer
     status,              // "completed" | "retry_used" | "failed"
   })

  (:ToolCall {
     callId,              // UUID
     turn,                // 1, 2, 3...
     order_in_turn,       // 0, 1, 2 (for parallel tool calls)
     tool_name,
     args_json,           // compact JSON string of args
     summary,             // short human-readable summary
     ok,                  // bool
     duration_ms,
     result_citation_count,  // count of verse refs returned
   })

  (:Answer {
     answerId,
     text,                // final answer prose
     text_hash,           // sha1 for dedup
     cited_verses,        // list of "X:Y" strings
     char_count,
   })

Relationships:

  (:Query)-[:TRIGGERED]->(:ReasoningTrace)
  (:ReasoningTrace)-[:HAS_STEP {order}]->(:ToolCall)
  (:Query)-[:PRODUCED]->(:Answer)

Indexes:

  CREATE INDEX query_id IF NOT EXISTS FOR (q:Query) ON (q.queryId);
  CREATE INDEX trace_id IF NOT EXISTS FOR (t:ReasoningTrace) ON (t.traceId);
  CREATE VECTOR INDEX query_embedding IF NOT EXISTS
    FOR (q:Query) ON (q.textEmbedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

Usage:
  rm = ReasoningMemory(driver)
  rm.ensure_schema()
  recorder = rm.start_query("What does the Quran say about patience?", backend="ollama:qwen3:8b")
  recorder.log_tool_call(turn=1, order=0, tool_name="search_keyword", args={"keyword":"patience"},
                          summary="Found 23 verses", ok=True, duration_ms=1200, result_citation_count=23)
  # ... more tool calls ...
  recorder.finish(answer_text="...", turns=3, total_ms=45000)

Querying similar past traces:
  similar = rm.find_similar_queries("How should I endure hardship?", top_k=3, min_sim=0.7)
"""
import hashlib
import json
import time
import uuid
from typing import Optional


def _now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


class ReasoningMemory:
    """Top-level accessor. Owns the Neo4j driver connection and schema setup."""

    def __init__(self, driver, db: str = "quran"):
        self.driver = driver
        self.db = db
        self._embed_model = None

    def _embed(self, text: str):
        if self._embed_model is None:
            # Reuse the same model we use elsewhere
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vec = self._embed_model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return vec.tolist()

    def ensure_schema(self):
        """Create indexes. Safe to call every startup — all use IF NOT EXISTS."""
        with self.driver.session(database=self.db) as s:
            s.run("CREATE INDEX query_id IF NOT EXISTS FOR (q:Query) ON (q.queryId)")
            s.run("CREATE INDEX query_ts IF NOT EXISTS FOR (q:Query) ON (q.timestamp)")
            s.run("CREATE INDEX trace_id IF NOT EXISTS FOR (t:ReasoningTrace) ON (t.traceId)")
            s.run("CREATE INDEX toolcall_id IF NOT EXISTS FOR (tc:ToolCall) ON (tc.callId)")
            s.run("CREATE INDEX toolcall_name IF NOT EXISTS FOR (tc:ToolCall) ON (tc.tool_name)")
            # Vector index — fails silently on already-existing ones
            try:
                s.run("""
                    CREATE VECTOR INDEX query_embedding IF NOT EXISTS
                    FOR (q:Query) ON (q.textEmbedding)
                    OPTIONS {indexConfig: {
                      `vector.dimensions`: 384,
                      `vector.similarity_function`: 'cosine'
                    }}
                """)
            except Exception as e:
                print(f"  [reasoning_memory] vector index setup: {e}")

    def start_query(self, text: str, backend: str, deep_dive: bool = False):
        """Start recording a new query. Returns a QueryRecorder."""
        query_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        embedding = self._embed(text)
        ts = _now_iso()
        with self.driver.session(database=self.db) as s:
            s.run("""
                CREATE (q:Query {
                    queryId: $qid, text: $text, textEmbedding: $emb,
                    timestamp: datetime($ts), backend: $backend, deep_dive: $deep
                })
                CREATE (t:ReasoningTrace {
                    traceId: $tid, total_duration_ms: 0, turn_count: 0,
                    tool_call_count: 0, citation_count: 0, status: 'in_progress'
                })
                CREATE (q)-[:TRIGGERED]->(t)
            """, qid=query_id, text=text, emb=embedding, ts=ts,
                 backend=backend, deep=deep_dive, tid=trace_id)
        return QueryRecorder(self, query_id, trace_id, start_time=time.time())

    def find_similar_queries(self, text: str, top_k: int = 3, min_sim: float = 0.7):
        """Return past queries similar to `text`, ordered by similarity desc."""
        vec = self._embed(text)
        with self.driver.session(database=self.db) as s:
            rows = s.run("""
                CALL db.index.vector.queryNodes('query_embedding', $k, $vec)
                YIELD node, score WHERE score >= $min_sim
                MATCH (node)-[:TRIGGERED]->(t:ReasoningTrace)
                OPTIONAL MATCH (node)-[:PRODUCED]->(a:Answer)
                OPTIONAL MATCH (t)-[hs:HAS_STEP]->(tc:ToolCall)
                WITH node, score, t, a,
                     collect({
                         order: hs.order,
                         turn: tc.turn,
                         tool_name: tc.tool_name,
                         args: tc.args_json,
                         summary: tc.summary,
                         ok: tc.ok
                     }) AS steps
                RETURN node.queryId AS queryId, node.text AS text,
                       node.timestamp AS timestamp,
                       score, t.citation_count AS citation_count,
                       t.status AS status, t.turn_count AS turns,
                       a.text AS answer,
                       [s IN steps WHERE s.tool_name IS NOT NULL] AS tool_steps
                ORDER BY score DESC
            """, vec=vec, k=top_k, min_sim=min_sim).data()
        return rows


class QueryRecorder:
    """A handle for writing tool calls + answer during a single query lifecycle."""

    def __init__(self, memory: ReasoningMemory, query_id: str, trace_id: str, start_time: float):
        self.memory = memory
        self.query_id = query_id
        self.trace_id = trace_id
        self.start_time = start_time
        self.turn_count = 0
        self.tool_call_count = 0

    def log_tool_call(self, turn: int, order: int, tool_name: str, args: dict,
                       summary: str, ok: bool, duration_ms: int,
                       result_citation_count: int = 0):
        call_id = str(uuid.uuid4())
        args_json = json.dumps(args, ensure_ascii=False)[:500]
        self.turn_count = max(self.turn_count, turn)
        self.tool_call_count += 1
        step_order = self.tool_call_count  # global ordering across turns
        with self.memory.driver.session(database=self.memory.db) as s:
            s.run("""
                MATCH (t:ReasoningTrace {traceId: $tid})
                CREATE (tc:ToolCall {
                    callId: $cid, turn: $turn, order_in_turn: $order,
                    tool_name: $name, args_json: $args, summary: $sum,
                    ok: $ok, duration_ms: $dur,
                    result_citation_count: $cites
                })
                CREATE (t)-[:HAS_STEP {order: $step_order}]->(tc)
            """, tid=self.trace_id, cid=call_id, turn=turn, order=order,
                 name=tool_name, args=args_json, sum=summary[:300], ok=ok,
                 dur=duration_ms, cites=result_citation_count,
                 step_order=step_order)

    def finish(self, answer_text: str, citation_count: int, status: str = "completed"):
        total_ms = int((time.time() - self.start_time) * 1000)
        text_hash = hashlib.sha1(answer_text.encode("utf-8")).hexdigest()
        import re
        cited = sorted({f"{m.group(1)}:{m.group(2)}"
                        for m in re.finditer(r"\[(\d+):(\d+)\]", answer_text)})
        ans_id = str(uuid.uuid4())
        with self.memory.driver.session(database=self.memory.db) as s:
            s.run("""
                MATCH (q:Query {queryId: $qid})
                MATCH (t:ReasoningTrace {traceId: $tid})
                CREATE (a:Answer {
                    answerId: $aid, text: $text, text_hash: $hash,
                    cited_verses: $cites, char_count: $cc
                })
                CREATE (q)-[:PRODUCED]->(a)
                SET t.total_duration_ms = $total_ms,
                    t.turn_count = $turns,
                    t.tool_call_count = $tools,
                    t.citation_count = $cite_n,
                    t.status = $status
            """, qid=self.query_id, tid=self.trace_id, aid=ans_id,
                 text=answer_text, hash=text_hash, cites=cited,
                 cc=len(answer_text), total_ms=total_ms, turns=self.turn_count,
                 tools=self.tool_call_count, cite_n=citation_count, status=status)

    def mark_failed(self, error: str):
        with self.memory.driver.session(database=self.memory.db) as s:
            s.run("""
                MATCH (t:ReasoningTrace {traceId: $tid})
                SET t.status = 'failed', t.error_summary = $err
            """, tid=self.trace_id, err=error[:500])
