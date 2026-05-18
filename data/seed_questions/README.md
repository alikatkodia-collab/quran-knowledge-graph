# Seed-question pools

This directory holds hand-curated and auto-generated question pools that were
historically used to seed `data/answer_cache.json` via overnight runs of
`app_free.py`.

## Provenance

Each YAML file in this directory was extracted from a corresponding
`overnight_seed_phaseN.py` script that lived at the repo root through
2026-04-24. Those scripts imported a now-missing `overnight_seed.py` engine
module (the engine was never tracked in git on `main`) and therefore could
not be run on a fresh checkout. The wrapper scripts were deleted on
2026-05-19 to remove the broken-on-import footgun; the question pools they
contained were extracted here so the curated content stays usable.

## File shape

```yaml
phase: 13
source_file: overnight_seed_phase13.py
provenance: "Phase 13 — final push to cache target 1500+."
count: 150
questions:
  - "How does the Quran structure prophetic narratives for spiritual impact?"
  - "How does the Quran use flashback and chronological shifts in Surah Yusuf?"
  - ...
```

## Counts

| File          | Questions | Original source                |
| ------------- | --------: | ------------------------------ |
| `phase06.yaml`|       160 | `overnight_seed_phase6.py`     |
| `phase07.yaml`|       176 | `overnight_seed_phase7.py`     |
| `phase08.yaml`|       180 | `overnight_seed_phase8.py`     |
| `phase09.yaml`|       170 | `overnight_seed_phase9.py`     |
| `phase10.yaml`|        40 | `overnight_seed_phase10.py`    |
| `phase11.yaml`|       155 | `overnight_seed_phase11.py`    |
| `phase12.yaml`|       155 | `overnight_seed_phase12.py`    |
| `phase13.yaml`|       150 | `overnight_seed_phase13.py`    |
| **Total**     | **1,186** |                                |

## Using them

To re-seed the answer cache from these pools, a future seeder script can:

```python
import yaml
from pathlib import Path

POOL = []
for path in sorted(Path("data/seed_questions").glob("phase*.yaml")):
    POOL.extend(yaml.safe_load(path.read_text(encoding="utf-8"))["questions"])

# POOL is 1186 distinct questions. Feed to /chat with --local_only to seed
# answer_cache.json over a long-running, capped session.
```

The seeder itself is intentionally not in this directory — restoring or
rewriting `overnight_seed.py` is a separate concern. These YAMLs are just
the curated content.

## Phase-13 vs phase-4b eval

There is intentional overlap between these seed questions and the Phase 4b
eval question set in `data/eval/v2/`. The eval questions are a hand-curated
subset of canonical themes (57 main + 15 held-out) with full assertion
shapes. The seed-question YAMLs are a broader pool of variation — useful
for cache warm-up, not for evaluation gating.
