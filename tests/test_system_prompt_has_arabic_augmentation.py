"""Guard the Arabic-root-augmentation directive in the system prompts.

Future edits to the prompt files must not silently delete the section that
tells the agent to call search_arabic_root and weave the canonical root
anchor into thematic answers. This is a pure-text test — no LLM call.
"""

from pathlib import Path

import pytest

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# At minimum 5 of these canonical anchors must remain in the prompt.
# The full set lives in the prompt itself; this list catches accidental
# bulk deletion without pinning every anchor (so the operator can refine
# the list without breaking the test).
REQUIRED_ANCHORS = ["sabr", "shukr", "tawba", "rahma", "ikhlas"]

PROMPT_FILES = [
    PROMPTS_DIR / "system_prompt_free.txt",
    PROMPTS_DIR / "system_prompt.txt",
]


@pytest.mark.parametrize("prompt_path", PROMPT_FILES, ids=lambda p: p.name)
def test_prompt_contains_arabic_root_augmentation_section(prompt_path: Path) -> None:
    assert prompt_path.exists(), f"Prompt file missing: {prompt_path}"
    text = prompt_path.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "arabic root augmentation" in lowered, (
        f"{prompt_path.name} is missing the 'Arabic root augmentation' "
        "section header — was it deleted by accident?"
    )

    assert "search_arabic_root" in text, (
        f"{prompt_path.name} no longer mentions search_arabic_root in the "
        "augmentation section — the agent will not know to call it."
    )

    present = [a for a in REQUIRED_ANCHORS if a in lowered]
    assert len(present) >= 5, (
        f"{prompt_path.name} only contains {len(present)} of the required "
        f"canonical anchors {REQUIRED_ANCHORS} (found: {present}). Restore "
        "the anchor table."
    )
