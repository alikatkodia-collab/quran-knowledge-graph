# VerseAnalysis Prompt A/B Report

**Generated:** 2026-04-22T08:46:02.029074Z
**Model:** openai/gpt-oss-120b:free
**Verses tested:** 10

## Summary table

| Metric | v2.0 | v2.1 | Delta |
|---|---|---|---|
| Valid JSON | 10 | 10 | 0 |
| All top-level keys present | 10 | 10 | 0 |
| Avg output chars (lower=leaner) | 5676 | 4905 | -771 |
| Inline `allowed` arrays in output | 0 | 0 | 0 |
| Avg `confidence` on deterministic fields (v2.1 should be 0) | 9.0 | 0.0 | -9.0 |
| Total enum violations | 0 | 0 | 0 |
| Avg wall-clock sec | 59.9 | 65.3 | +5.5 |

## Per-verse detail

| Verse | v2.0 valid | v2.1 valid | v2.0 chars | v2.1 chars | v2.0 enum_viol | v2.1 enum_viol |
|---|---|---|---|---|---|---|
| 1:1 | Y | Y | 5308 | 4266 | 0 | 0 |
| 2:255 | Y | Y | 6196 | 5783 | 0 | 0 |
| 2:173 | Y | Y | 5772 | 4956 | 0 | 0 |
| 12:4 | Y | Y | 5426 | 5261 | 0 | 0 |
| 36:79 | Y | Y | 5711 | 5054 | 0 | 0 |
| 3:159 | Y | Y | 5765 | 4327 | 0 | 0 |
| 74:30 | Y | Y | 5267 | 4863 | 0 | 0 |
| 112:1 | Y | Y | 5589 | 5057 | 0 | 0 |
| 55:13 | Y | Y | 5310 | 4495 | 0 | 0 |
| 4:34 | Y | Y | 6421 | 4992 | 0 | 0 |

## Interpretation

- Lower `output chars` = leaner payload = cheaper per extraction.
- `allowed arrays in output` should be 0 for v2.1 (they were schema docs, not data).
- `confidence on deterministic fields` should be 0 for v2.1; v2.0 may have them.
- `enum violations` = values outside the allowed vocabulary. Lower is better on both.