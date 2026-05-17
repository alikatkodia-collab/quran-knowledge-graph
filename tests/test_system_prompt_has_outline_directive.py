"""Guard the outline-first directive in the system prompts.

Long thematic answers (e.g. "patience") were repeating the same 3-4 verses
under different H2 headers — non-adjacent section-level repetition that
n-gram decoding penalties can't catch. The fix lives in the system prompt:
an OUTLINE BEFORE WRITING block that forces the agent to enumerate themes
first and forbids citing the same verse under two different H2 headers.

This is a pure-text test — no LLM call. It catches accidental bulk
deletion of the directive in future edits.
"""

from pathlib import Path

import pytest

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

PROMPT_FILES = [
    PROMPTS_DIR / "system_prompt_free.txt",
    PROMPTS_DIR / "system_prompt.txt",
]


@pytest.mark.parametrize("prompt_path", PROMPT_FILES, ids=lambda p: p.name)
def test_prompt_contains_outline_directive(prompt_path: Path) -> None:
    assert prompt_path.exists(), f"Prompt file missing: {prompt_path}"
    text = prompt_path.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "outline before writing" in lowered, (
        f"{prompt_path.name} is missing the 'OUTLINE BEFORE WRITING' "
        "directive header — was it deleted by accident? This block "
        "prevents section-level verse repetition on long answers."
    )

    assert "one theme per h2" in lowered, (
        f"{prompt_path.name} no longer states the one-section-per-theme "
        "rule. Restore the directive."
    )

    # The STOP-check rule is the load-bearing part — it tells the agent
    # to verify before citing, not after writing.
    assert "stop" in lowered and "prior h2" in lowered, (
        f"{prompt_path.name} no longer contains the STOP / prior-H2 check. "
        "Restore the rule that forbids citing a verse under two different "
        "H2 headers."
    )


@pytest.mark.parametrize("prompt_path", PROMPT_FILES, ids=lambda p: p.name)
def test_outline_directive_stays_short(prompt_path: Path) -> None:
    """The directive must stay under 15 lines so it doesn't bloat the prompt."""
    text = prompt_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    start = next(
        (i for i, line in enumerate(lines) if "outline before writing" in line.lower()),
        None,
    )
    assert start is not None, f"{prompt_path.name} has no OUTLINE BEFORE WRITING header"

    # The block ends at the first blank line after the header. Both prompts
    # use a blank line to separate sections.
    block_end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == "":
            block_end = i
            break

    block_len = block_end - start
    assert block_len <= 15, (
        f"{prompt_path.name} OUTLINE block grew to {block_len} lines "
        f"(limit 15). Tighten it — the prompt is already long."
    )
