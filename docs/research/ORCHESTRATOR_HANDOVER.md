# Orchestrator handover: AIE Miami + Europe research, 2026-05-15

**Paste this into your project orchestrator's context, or have the
orchestrator read it at session start.** Self-contained — the
orchestrator doesn't need to read any other research doc until it
needs to act on a specific finding.

---

## Who you are reading this

You are the main project orchestrator for the QKG (Quran Knowledge
Graph) project. On 2026-05-15, a separate research session analysed
four AI Engineer conference livestreams (AIE Miami Day 2 + Miami
Keynote + AIE Europe Day 1 + AIE Europe Day 2 — ~34,000 transcript
lines total) for findings relevant to QKG. The outputs of that
session are committed on the `claude/research-aie-miami-2026` branch.

Your job, when consulting this research, is to:
1. Use the findings to inform decisions about *what* QKG should build.
2. Use the method-of-development findings to inform decisions about
   *how* you collaborate with the operator.
3. Not silently re-derive conclusions the research already reached.
4. Not treat the research as authoritative — it's auto-transcribed
   conference content from 4 livestreams. Sample size 4. Use as
   directional, not dispositive.

---

## What was researched

Four AIE 2026 livestreams. Titles:

1. `DeM_u2Ik0sk` — AIE Miami Day 2 (Cerebras / OpenCode / Cursor / Arize)
2. `6IxSbMhT7v4` — AIE Miami Keynote (OpenCode / DeepMind / OpenAI)
3. `_zdroS0Hc74` — AIE Europe Day 1 (Pi / DeepMind / Anthropic / Cursor / Linear) — **highest-value content**
4. `O_IMsEg91g8` — AIE Europe Day 2 (DeepMind / OpenAI / Vercel / @pragmaticengineer)

Note: 3 and 4 are AIE Europe (London), not Miami, despite the file
naming convention.

---

## Where the outputs live

All on `claude/research-aie-miami-2026` (HEAD `6c75702` at time of
writing).

| File | Purpose | When to read |
|---|---|---|
| `data/research/yt_<id>_transcript.txt` × 4 | Raw transcripts | Only if you need to verify a specific quote |
| `docs/research/yt_<id>.md` × 4 | Per-video analysis | When you want detail on one specific talk |
| `docs/research/aie_miami_synthesis_2026-05-15.md` | Cross-video themes + retrofit-phase mappings | **Read first** when deciding what to build |
| `docs/research/method_of_development_takeaways_2026-05-15.md` | Process / harness / workflow patterns | **Read first** when deciding how to work |
| `data/proposed_tasks_from_yt_2026-05-15.yaml` | 10 concrete task proposals with acceptance criteria | When picking the next implementation task |
| `docs/research/ORCHESTRATOR_HANDOVER.md` | This file | First — entry point |

---

## 60-second executive summary

The four-conference consensus rests on three claims:

1. **MCP context-bloat is the #1 agent-loop problem in 2026, and the
   consensus answers are `tool_search` (progressive discovery) + Code
   Mode (script-tool that composes multiple calls in one turn).**
   David Sora Par (creator of MCP) said Anthropic shipped this in
   Claude Code with "massive reduction in tool context usage."
   Cloudflare and Lori Voss (Arize) corroborated with empirical data:
   MCP-instructed agents took 2.4× more turns than CLI-equivalents on
   matched tasks because of verbose-JSON context overflow.

2. **Knowledge-graph "context graphs" with auditable decision traces
   are exactly what QKG already builds via `reasoning_memory.py`.**
   Nia (Neo4j) gave the marquee KG talk at AIE Miami; her pattern is
   independent validation of QKG's existing architecture. Telecom
   benchmark cited: base 37% → fine-tune 54% → KG+RAG 91%.

3. **Evals are tests, not metrics.** Lori (Arize), Liam (OpenClaw),
   and Dexter (OpenCode) said variants of this independently. Matches
   QKG retrofit's Phase 4 plan. LLM-as-judge with one-sentence
   explanations is the lead pattern.

The four conferences also agreed on a 2026 connectivity stack
(skills + MCP + CLI used together, not one or the other), the
shift from "implementation as scarce resource" to "context window +
human attention as scarce resources," and the discipline of using
control flow rather than prompts for branching logic.

---

## Quick-lookup table — when to consult what

| Situation | Source |
|---|---|
| Deciding whether to consolidate the 21-tool list | Synthesis §"Phase 7 (trim sprawl) — alternatives" + tasks `from_yt_anthropic_progressive_discovery_tool_search` and `from_yt_anthropic_code_mode_spike` |
| Designing the Phase 4 eval rubric | Synthesis §"Phase 4 — extensions" + task `from_yt_arize_eval_add_latency_cost_tool_fidelity` |
| Choosing between MCP server exposure vs CLI | yt_6IxSbMhT7v4.md §"Lori" + yt__zdroS0Hc74.md §"David Sora Par" + synthesis §"Surprises — negative" |
| Embedding-strategy decisions | yt_O_IMsEg91g8.md §"DeepMind" + task `from_yt_deepmind_matryoshka_two_tier_retrieval` |
| Decision-trace / reasoning-memory questions | yt_DeM_u2Ik0sk.md §"Nia" + task `from_yt_neo4j_decision_trace_traversal_tool` |
| Memory lifecycle / forgetting policy | yt_DeM_u2Ik0sk.md §"Alvin" + task `from_yt_outrival_reasoning_memory_decay` |
| Run_cypher safety / authority split | yt_6IxSbMhT7v4.md §"Anna" + task `from_yt_pinterest_split_run_cypher_by_authority` |
| Phase 5 worktree-isolation design | yt_O_IMsEg91g8.md §"Liam" + task `from_yt_openclaw_phase5_worktree_isolation_revision` |
| Operator + agent collaboration style | method_of_development_takeaways §1–9 |
| Code review discipline | method_of_development §2 (Honor / Kodto) |
| Documentation / non-functional requirements | method_of_development §3 (Ryan Leopollo / OpenAI) |
| Planning artefacts before implementation | method_of_development §4 (Dexter / CRISPY) |
| Parallel agent sessions | method_of_development §5 (Liam swim lanes) |
| Skills / dotfiles infrastructure | method_of_development §6 |

---

## The 10 proposed tasks at a glance

All formally specified in
`data/proposed_tasks_from_yt_2026-05-15.yaml` with full acceptance
criteria. Compressed here for triage:

| ID | Pri | One-liner | Status / blockers |
|---|---|---|---|
| `from_yt_arize_eval_add_latency_cost_tool_fidelity` | 85 | Add latency, cost, tool-fidelity to v2 eval rubric | **Blocked** — Phase 4a in flight on parallel branch |
| `from_yt_anthropic_progressive_discovery_tool_search` | 75 | `tool_search` meta-tool to defer tool-description loading | **Blocked** — needs v2 eval to measure |
| `from_yt_deepmind_matryoshka_two_tier_retrieval` | 70 | 256-dim first-pass + 1024-dim re-rank with BGE-M3 MRL | **Half-blocked** — can prep, can't switch default |
| `from_yt_pinterest_split_run_cypher_by_authority` | 70 | Replace regex denylist with parser-level read-only enforcement | Sequenced behind Phase 7 item 30 |
| `from_yt_anthropic_code_mode_spike` | 60 | Python-sandbox tool to compose multiple Neo4j calls per turn | **Blocked** — large spike; v2 eval first |
| `from_yt_opencode_split_system_prompt_by_bucket` | 60 | Split monolithic prompt by classify_query bucket | **Blocked** — Phase 7 #29 + v2 eval |
| `from_yt_outrival_reasoning_memory_decay` | 55 | Forgetting policy for old `:ReasoningTrace` nodes | **Unblocked** |
| `from_yt_neo4j_decision_trace_traversal_tool` | 55 | `recall_traces_for_verse(verseId)` query tool | **Unblocked** — pair with decay above |
| `from_yt_openclaw_phase5_worktree_isolation_revision` | 50 | Document Liam's git-worktree warning; cap concurrent worktrees to 1 | **Unblocked** — docs-only |
| `from_yt_deepmind_gemini_embeddings_2_qrcd_ab` | 45 | A/B Gemini Embeddings 2 vs BGE-M3 on QRCD | **Blocked** — v2 eval first; API cost |

**Unblocked tasks ordered by ascending risk:**

1. Phase 5 worktree doc revision (docs-only, 30 min)
2. Reasoning-memory decay (touches `reasoning_memory.py`, needs regression test, ~2 hrs)
3. Decision-trace traversal tool (touches `chat.py`, depends on decay, ~2 hrs)

---

## The 5 operator-actions ("try this week" punch list)

From `method_of_development_takeaways_2026-05-15.md` §"Five things
worth trying in the next week" — all under 30 min, each a diagnostic
not a permanent change:

1. **5-tool subset diagnostic tick** — Try one tick with shared_agent
   limited to 5 tools + EXHAUSTIVE SEARCH MANDATE stripped. Compare
   trajectory + answer quality to baseline. *Tests whether tool
   bloat hurts QKG in practice.*
2. **Author `docs/QKG_NON_FUNCTIONAL_REQUIREMENTS.md`** — 30 min,
   operator-authored, NOT LLM-generated. Five-to-ten things QKG should
   never do (overclaim, confabulate, hide Khalifa disclosure). Becomes
   reference for v2 eval's framing dimension.
3. **Swim-lane naming in CLAUDE_INDEX.md** — Lane A (loop), Lane B
   (research), Lane C (interactive). 10 min. Pays back when loop resumes.
4. **First `.claude/skills/add-eval-question/`** — Repo-local skill.
   30 min. Stops re-explaining eval schema every session.
5. **Fresh-session second-pass review on largest Phase-3a PR** — Cold
   independent review by a model with no author bias. 20 min. Tests
   whether "second-pass review" surfaces things author-session missed.

---

## Contradictions to current retrofit assumptions

Three findings either revise or complicate items in the existing
`docs/QKG_RETROFIT_PLAN.md`:

1. **Phase 7 item 28 ("consolidate 21 tools to 8-10") has two new
   alternatives, both better-supported by 2026 conference data.**
   - Progressive discovery via `tool_search` (Anthropic-shipped in
     Claude Code).
   - Code Mode (Cloudflare, Kent C. Dodds, David Sora Par all
     advocated).
   The audit item should be revised from "consolidate" to "pick an
   exposure strategy" with three candidates: consolidation, progressive
   discovery, code mode.

2. **Phase 5 item 19 (worktree-per-tick) has a hard-won warning from
   OpenClaw at scale.** Liam Hapton ran 70-80 active worktrees with a
   heavy test harness, called it "hell," and moved to clone-N-times.
   QKG's single-tick concurrency makes this much less risky but the
   design doc should still flag the hazard. Captured in task
   `from_yt_openclaw_phase5_worktree_isolation_revision`.

3. **The audit's wish-list "Expose QKG as an MCP server" carries a
   measured hidden cost.** Lori Voss's 500-test eval showed
   MCP-instructed agents on mature tooling (GitHub MCP) took 2.4×
   more turns and burned far more tokens than CLI-equivalents.
   Before investing in an MCP server, sanity-check whether QKG's
   tool surface would have the same fixed-API-multi-call problem.

---

## Scope constraints

When using this research, respect these constraints:

- **Research branch `claude/research-aie-miami-2026` is research-only.**
  No code changes have landed on it. Implementation tasks open new
  branches (typically `claude/from_yt_<task_id>` or similar). The
  research branch is read-only after merge to main.

- **Parallel session on `phase-4a-eval-infrastructure` owns
  `eval/`, `data/eval/v2/`, `tests/test_eval_v2_*.py`.** Anything
  touching those paths conflicts with the eval-infrastructure session.
  Do not assign tasks that touch them until phase-4a merges.

- **`data/RALPH_STOP` is present.** The Ralph loop is deliberately
  paused per Phase 0 of the retrofit. Do not restart it. Do not assign
  tasks to the loop. Operator decides when to resume (Phase 11
  conditional on Phases 0–10 completing).

- **The audit's "no speculative abstraction without two failing tests"
  rule (Phase 5 item 21) applies to anything derived from this
  research.** Conference content alone is not justification — the
  pattern needs a falsifiable test before the abstraction lands.

---

## Honest caveats this research carries

Take these as priors when consuming the findings:

- **Sample size = 4 livestreams from one conference series.** Findings
  are AIE consensus, not industry consensus. Cross-check against other
  sources (Anthropic / OpenAI engineering blogs, Hacker News
  longform, ICML / NeurIPS retrieval papers) before treating any
  finding as authoritative.
- **Transcripts are auto-generated.** Speaker names ("Nia Mlinaric,"
  "Lori Voss," "Pi," etc.) were inferred from context. Quotes are
  approximate. Timestamps may drift ±5s.
- **Many talks were sampled, not deep-read.** Sections grepped for
  QKG-relevant keywords and read in detail; sections without relevant
  hits were skipped. The OpenRouter / Cerebras / DeepMind-weather /
  Pragmatic-Engineer-fireside content is *light* in the per-video docs
  for this reason.
- **The proposed tasks were drafted without operator validation of
  their acceptance criteria.** The criteria are best-effort. The
  operator should sanity-check before triaging into ralph_backlog.yaml.

---

## How to use this in practice

When a request comes in that could be informed by this research:

1. **First glance**: Is the request in the quick-lookup table above?
   If yes, read that specific section / task only.
2. **Decision questions** ("should QKG do X?"): Read the synthesis
   doc's "Patterns directly applicable" + "Patterns explicitly NOT
   applicable" sections. They explicitly say what does and doesn't
   transfer.
3. **Implementation questions** ("how should I build Y?"): Read the
   specific task in `proposed_tasks_from_yt_2026-05-15.yaml`. It has
   acceptance criteria and blockers.
4. **Process questions** ("how should I work with the operator?"):
   Read the method-of-development doc's relevant section.
5. **Quote-checking**: Only read raw transcripts. Per-video docs cite
   timestamps; cross-reference there.
6. **Surprise findings the operator hasn't heard yet**: Flag the
   three contradictions to retrofit assumptions above explicitly when
   relevant — operator may not have seen them.

When the research is silent on a question:

- Don't invent a finding. Say "the AIE research is silent on this."
- Default to the existing `docs/QKG_AUDIT.md` and
  `docs/QKG_RETROFIT_PLAN.md` for authoritative guidance.
- Default to `CLAUDE.md` and `CLAUDE_INDEX.md` for project context.

---

## Open questions left for the operator

Things this research raised but couldn't resolve without operator
input:

1. **Which Phase 7 #28 path? consolidation vs progressive discovery vs
   code mode.** All three are valid; the right answer depends on
   v2 eval data, which doesn't exist yet.
2. **Does QKG actually want an MCP server?** Lori's data complicates
   the "yes by default" stance. Operator decision.
3. **What audience is QKG for?** Phase 8 item 31 — "submitters /
   academic researchers / agent-tooling devs." The conference content
   subtly pushes toward "agent-tooling devs" (skills + MCP + CLI as
   exposure strategy) but doesn't decide.
4. **When does Phase 4 v2 eval land?** Many of the proposed tasks are
   blocked behind it. The parallel `phase-4a-eval-infrastructure`
   branch will tell.
5. **Should the operator try the 5 "this week" experiments?** Each is
   under 30 min but cumulative. Operator decides ordering / whether
   to skip.

---

End of handover. Branch HEAD at write time: `6c75702`.
