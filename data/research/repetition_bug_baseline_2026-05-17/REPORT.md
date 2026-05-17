# Repetition-Bug Baseline — qwen3:14b, N=50

Captured 2026-05-18 00:09.
Server: `app_free.py --model qwen3:14b` + `SEMANTIC_SEARCH_INDEX=verse_embedding_m3`, `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`. `/chat` payload: `{local_only: true, history: []}`.

## Headline

- **Verbatim bug rate:** 10/32 = **31.2%** of completed answers
- **Combined (verbatim + near):** 10/32 = **31.2%**

## Run summary

- Total questions attempted: 50
- Completed (CLEAN + BUG_*): 32
- Timed out: 11
- Errored: 7
- Missing (not run): 0
- Median elapsed time per completed question: 182s
- Total wall-time over completed questions: 6029s (100.5 min)

## Verdict counts

| verdict | count |
|---------|------:|
| CLEAN | 22 |
| BUG_NEAR | 0 |
| BUG_VERBATIM | 10 |
| TIMEOUT | 11 |
| ERROR | 7 |
| MISSING | 0 |

## By category

| category | n | CLEAN | BUG_NEAR | BUG_VERBATIM | verbatim % | combined % |
|----------|--:|------:|---------:|-------------:|-----------:|-----------:|
| ABSTRACT | 24 | 15 | 0 | 9 | 37.5% | 37.5% |
| BROAD | 3 | 3 | 0 | 0 | 0.0% | 0.0% |
| COMPLEMENTARY | 5 | 4 | 0 | 1 | 20.0% | 20.0% |

## By answer-length quartile

| quartile | char-range | n | BUG_VERBATIM | verbatim % |
|----------|-----------:|--:|-------------:|-----------:|
| Q1 | 0-6374 | 8 | 1 | 12.5% |
| Q2 | 6375-7244 | 8 | 3 | 37.5% |
| Q3 | 7245-8170 | 8 | 5 | 62.5% |
| Q4 | 8171-17837 | 8 | 1 | 12.5% |

## By tool-call count

| tool-calls | n | BUG_VERBATIM | verbatim % |
|------------|--:|-------------:|-----------:|
| 0-2 | 0 | 0 | 0.0% |
| 3-4 | 0 | 0 | 0.0% |
| 5-6 | 5 | 2 | 40.0% |
| 7+ | 27 | 8 | 29.6% |

## Top 10 duplicated verses across corpus

| verse | duplicate-pair count |
|-------|--------------------:|
| 4:17 | 3 |
| 3:89 | 3 |
| 5:39 | 3 |
| 66:8 | 3 |
| 41:36 | 3 |
| 23:97 | 3 |
| 10:8 | 3 |
| 7:200 | 2 |
| 78:30 | 2 |
| 3:185 | 2 |

## Per-question table

| # | category | verdict | h2 | chars | tools | elapsed (s) | question |
|--:|----------|---------|---:|------:|------:|------------:|----------|
| 1 | ABSTRACT | CLEAN | 7 | 9870 | 7 | 268.5 | What does the Quran say about charity? |
| 2 | ABSTRACT | CLEAN | 1 | 2223 | 7 | 213.2 | What does the Quran teach about forgiveness? |
| 3 | ABSTRACT | BUG_VERBATIM | 6 | 7466 | 6 | 153.7 | What does the Quran say about gratitude? |
| 4 | ABSTRACT | CLEAN | 6 | 4728 | 6 | 151.3 | What does the Quran say about sin? |
| 5 | ABSTRACT | CLEAN | 8 | 11101 | 7 | 223.5 | What does the Quran say about patience? |
| 6 | ABSTRACT | CLEAN | 6 | 6535 | 7 | 98.7 | What does the Quran teach about humility? |
| 7 | ABSTRACT | BUG_VERBATIM | 6 | 6864 | 7 | 114.1 | What does the Quran say about repentance? |
| 8 | ABSTRACT | CLEAN | 6 | 5471 | 8 | 191.3 | How does the Quran describe true belief? |
| 9 | ABSTRACT | BUG_VERBATIM | 6 | 7245 | 6 | 182.2 | What does the Quran say about hypocrisy? |
| 10 | ABSTRACT | BUG_VERBATIM | 6 | 8043 | 7 | 182.3 | What does the Quran teach about justice? |
| 11 | ABSTRACT | CLEAN | 5 | 7653 | 8 | 277.2 | What does the Quran say about the consequences of arrogance? |
| 12 | ABSTRACT | CLEAN | 6 | 7433 | 7 | 161.0 | How does the Quran describe the path of righteousness? |
| 13 | ABSTRACT | CLEAN | 6 | 7955 | 7 | 139.0 | What does the Quran teach about trusting in God? |
| 14 | ABSTRACT | CLEAN | 5 | 4739 | 8 | 121.3 | What does the Quran say about doubt and certainty? |
| 15 | ABSTRACT | CLEAN | 6 | 6746 | 7 | 130.1 | How does the Quran describe sincere worship? |
| 16 | ABSTRACT | CLEAN | 6 | 5222 | 7 | 117.9 | What does the Quran say about Satan and how he misleads peop |
| 17 | ABSTRACT | CLEAN | 6 | 7075 | 7 | 191.7 | What does the Quran teach about being led astray? |
| 18 | ABSTRACT | BUG_VERBATIM | 5 | 4575 | 8 | 189.1 | What does the Quran say about the whisperings of Satan? |
| 19 | ABSTRACT | BUG_VERBATIM | 6 | 7169 | 7 | 165.3 | How does the Quran describe the consequences of following Sa |
| 20 | ABSTRACT | BUG_VERBATIM | 7 | 8399 | 7 | 142.9 | What does the Quran say about wealth and worldly attachment? |
| 21 | ABSTRACT | BUG_VERBATIM | 7 | 7355 | 7 | 202.5 | What does the Quran teach about death and the soul? |
| 22 | ABSTRACT | CLEAN | 5 | 6748 | 6 | 212.2 | What does the Quran say about reflection and contemplation? |
| 23 | ABSTRACT | CLEAN | 5 | 8171 | 7 | 123.8 | How does the Quran describe the mercy of God? |
| 24 | ABSTRACT | ERROR | 0 | 0 | 0 | 325.8 | What does the Quran say about prayer? |
| 25 | ABSTRACT | BUG_VERBATIM | 6 | 7114 | 7 | 172.4 | What does the Quran teach about honoring parents? |
| 26 | BROAD | ERROR | 0 | 0 | 0 | 334.6 | What are the main themes of the Quran? |
| 27 | BROAD | TIMEOUT | 0 | 0 | 7 | 327.7 | How does the Quran describe itself? |
| 28 | BROAD | CLEAN | 8 | 5680 | 7 | 267.3 | What is the Quran's view of revelation? |
| 29 | BROAD | CLEAN | 1 | 15802 | 8 | 223.6 | How does the Quran describe the relationship between God and |
| 30 | BROAD | ERROR | 0 | 0 | 0 | 317.5 | What does the Quran say about its own miraculous nature? |
| 31 | BROAD | CLEAN | 6 | 6375 | 7 | 170.1 | How does the Quran describe the Day of Judgment? |
| 32 | BROAD | TIMEOUT | 0 | 0 | 7 | 396.5 | What is the Quran's view of other scriptures? |
| 33 | BROAD | TIMEOUT | 0 | 0 | 6 | 300.9 | What is the central message of Surah Ar-Rahman? |
| 34 | BROAD | TIMEOUT | 5 | 5962 | 7 | 329.9 | What is the main theme of Surah Yasin? |
| 35 | BROAD | ERROR | 0 | 0 | 0 | 333.0 | What is the message of Surah Al-Kahf? |
| 36 | COMPLEMENTARY | TIMEOUT | 5 | 6500 | 7 | 314.3 | What does the Quran say about fear of God? |
| 37 | COMPLEMENTARY | CLEAN | 10 | 17837 | 7 | 296.3 | What does the Quran teach about kindness? |
| 38 | COMPLEMENTARY | TIMEOUT | 0 | 0 | 6 | 370.2 | How does the Quran describe the believers' relationship to e |
| 39 | COMPLEMENTARY | ERROR | 0 | 0 | 0 | 415.3 | What does the Quran say about the unseen? |
| 40 | COMPLEMENTARY | ERROR | 0 | 0 | 0 | 451.5 | What does the Quran say about envy and jealousy? |
| 41 | COMPLEMENTARY | TIMEOUT | 3 | 15736 | 6 | 367.8 | What does the Quran teach about lying and truthfulness? |
| 42 | COMPLEMENTARY | ERROR | 0 | 0 | 0 | 363.1 | How does the Quran describe paradise? |
| 43 | COMPLEMENTARY | CLEAN | 2 | 9890 | 7 | 294.6 | How does the Quran describe hell? |
| 44 | COMPLEMENTARY | CLEAN | 6 | 10541 | 6 | 182.5 | What does the Quran say about generosity? |
| 45 | COMPLEMENTARY | TIMEOUT | 5 | 6838 | 7 | 385.5 | What does the Quran teach about peace? |
| 46 | COMPLEMENTARY | CLEAN | 5 | 5959 | 9 | 290.4 | What does the Quran say about anger and self-control? |
| 47 | COMPLEMENTARY | TIMEOUT | 6 | 6375 | 7 | 327.1 | What does the Quran say about wisdom? |
| 48 | COMPLEMENTARY | TIMEOUT | 6 | 7779 | 8 | 331.4 | What does the Quran say about freedom and slavery? |
| 49 | COMPLEMENTARY | TIMEOUT | 0 | 0 | 8 | 325.0 | What does the Quran teach about justice in commerce? |
| 50 | COMPLEMENTARY | BUG_VERBATIM | 7 | 8116 | 7 | 178.7 | What does the Quran say about the resurrection? |

## Caveats

- **Server degraded over the run.** All 7 ERROR entries are client-side
  `Read timed out` after >300s waiting for a `/chat` chunk; the 11 TIMEOUT
  entries are also stalls (request started but never produced a final
  frame within the per-question 300s budget). Failures cluster from Q24
  onward — the Ollama / FastAPI process likely accumulated memory or
  KV-cache pressure across the 3.5h run. This was a single uninterrupted
  process; a fresh server per question would likely raise completion
  rate, but is out of scope here.
- **BROAD category is underpowered.** Only 3/10 BROAD questions completed
  cleanly enough to score; treat the 0% verbatim rate there as
  not-yet-measured rather than evidence of cleanliness.
- **No near-verbatim hits.** The 80%-similarity bucket caught zero pairs
  the strict-identity bucket missed. The bug, when it fires, is
  character-for-character identical, not a paraphrase.
- **Per-question elapsed numbers include the 5s inter-request pause that
  began *before* the next question, but exclude server preload (~30s).**
- **All requests routed to `qwen3:14b`** per the first SSE Model frame;
  no OpenRouter fallback fired despite the high error rate (`local_only:
  true` held).

## Plain-English summary

On 32 qwen3:14b completions (out of 50 attempted; 11 timeouts + 7 read-
timeout errors after Q24), the verbatim-explanation-duplicate bug fired
in **31.2%** of answers. All flagged duplicates were character-identical;
no near-verbatim-but-not-identical cases were found, which suggests the
model is literally re-emitting an earlier explanation rather than
paraphrasing.

The bug correlates most strongly with the **ABSTRACT thematic** question
type (9/24 = 37.5%) and with **medium-length answers in the 6400-8200
char range** (Q2-Q3 quartile = 50% vs Q1/Q4 = 12.5%). Tool-call depth
does not discriminate. The top-duplicated verses (4:17, 3:89, 5:39, 66:8
on repentance; 41:36, 23:97 on Satan) are exactly the canonical pick on
each theme — i.e., when the model has a "best verse" for a topic, it
tends to reach for it again later in the same answer instead of citing
a different one.

A 31% base rate on the question types most relevant to user value
(thematic exploration) is substantial enough to justify intervention.
The post-stream dedup approach is well-targeted given that the bug is
strict-identity (a simple seen-set check on the first ~150 chars of each
explanation would catch every flagged case). The outline-first directive
revert is also defensible if the directive was the trigger; this baseline
should be re-run against the alternate prompt to confirm causation.
