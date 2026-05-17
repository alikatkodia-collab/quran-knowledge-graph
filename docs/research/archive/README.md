# Archived research artefacts

This folder holds research outputs from earlier project states that
have no clean way to merge back into current `main`'s history but are
worth keeping for reference.

## Inventory

### `TOP_100_DISCOVERIES_archived_2026-04-21.md`

Extracted from the now-deleted branch `claude/evaluate-karpathy-research-TCxjg`
(2026-04-21). Format: thematic verse-connection "discoveries" found by
multi-hop graph traversal. Example entries chain verses through shared
keywords across 3+ hops (e.g. `record → hand → right` joining
`[87:18] → [69:19] → [84:7] → [39:67]`).

The branch's other docs (`KARPATHY_AUTORESEARCH_EVALUATION.md`,
`FINDINGS_REPORT.md`) were considered for archive too but their
substantive conclusions have been superseded by the QKG audit + retrofit
plan. Only the discoveries format was kept because the multi-hop
discovery pattern could inspire a future Phase 7 task to regenerate
similar against the current graph.

Note the verse references and graph topology are from the older project
state. Re-running the discovery routine against current Neo4j is the
right way to use this — as a template, not as live data.

## Provenance

- Source branch: `claude/evaluate-karpathy-research-TCxjg` (deleted)
- Author tag: "Nineteen Collab" (per git log)
- Captured: 2026-05-16 during stray-branch cleanup
- Original branch had no merge base with current main (orphaned history)
