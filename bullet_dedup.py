"""Post-stream verbatim-duplicate suppression for verse-explanation bullets.

The local-Ollama agent (qwen3:14b in particular) sometimes re-emits the
same `[X:Y] "<quote>"` bullet inside multiple H2 sections of one answer.
The 2026-05-17 baseline measured a 31% verbatim-bug rate on thematic
ABSTRACT questions; later qwen3:14b spot-checks turned up additional
shapes (non-bold citations, bold citations whose trailing commentary
differs slightly while the verse quote itself is identical). This
module suppresses every observed shape.

Pure function: in goes the buffered streamed answer text, out comes
(cleaned_text, suppressed_verse_ids). Stateless w.r.t. the rest of the
system — one instance per request, no I/O.

Signature is `(verse_id, normalised_first_N_chars_of_rest_of_line)`
where N is ``_SIG_PREFIX_CHARS``. "Rest of line" means everything after
the `[X:Y]` citation marker, stripped of surrounding whitespace. The
N-char window is wide enough to encompass the verse-quote portion of a
typical bullet and narrow enough that "same verse, fresh angle" reuse
(whose rest text diverges within the first ~10 chars) survives.
"""

from __future__ import annotations

import re

_BULLET_LINE = re.compile(
    # Lines that look like a citation bullet. qwen3:14b emits several
    # decoration variants for the same underlying bullet shape:
    #   - **[4:17]** – *"..."*       (bold, original)
    #   - **[4:17]**: "..."          (bold + trailing colon)
    #   - [4:18] "..."               (no bold, no colon)
    #   - [4:18]: "..."              (no bold, with colon)
    #   * **[4:17]** ...             (star bullet)
    #   | **[2:172]** | "..." | ...  (table row)
    #   [2:272] "..."                (no leading bullet marker at all)
    # Both the leading bullet marker (-, *, or |) and the `**` boldface
    # around the `[X:Y]` citation are optional; the citation itself,
    # though, must sit at the start of the line (modulo whitespace) so
    # we never dedup a citation that appears inside flowing prose.
    r"^\s*(?:[-*|]\s+)?(?:\*\*)?\[(?P<verse>\d+:\d+)\](?:\*\*)?(?P<rest>.*)$"
)

# Number of chars from the stripped post-marker text used as the
# duplicate-detection signature. The original 2026-05-17 baseline
# documented duplicates as identical on the first ~150 chars; later
# qwen3:14b spot-checks (q5 charity, verse 2:272) produced duplicates
# whose verse text is identical but where the trailing commentary
# diverges at position ~122 in the rest. 100 chars catches the
# verse-quote portion in all observed variants while still leaving
# plenty of margin against "same verse, fresh angle" reuse (those
# diverge within the first ~10 chars of rest).
_SIG_PREFIX_CHARS = 100


class BulletDedup:
    """Per-request seen-signature set; call ``filter_text`` once at end of stream."""

    def __init__(self) -> None:
        # signature -> first line index (kept for potential future
        # debug logging, not used in the suppression decision today).
        self._seen: dict[tuple[str, str], int] = {}

    def filter_text(self, full_text: str) -> tuple[str, list[str]]:
        """Strip verbatim-duplicate citation bullets from ``full_text``.

        Returns ``(cleaned_text, suppressed_verses)``. A bullet is
        considered a duplicate iff a prior line in the same ``full_text``
        had:

        1. The exact same verse id inside `[X:Y]` (with or without the
           surrounding ``**``), AND
        2. The same first ``_SIG_PREFIX_CHARS`` chars of the post-marker
           text after stripping surrounding whitespace.

        Non-bullet lines and bullets without a recognisable verse
        citation pass through unchanged.
        """
        if not full_text:
            return "", []

        suppressed: list[str] = []
        out_lines: list[str] = []
        for idx, line in enumerate(full_text.split("\n")):
            m = _BULLET_LINE.match(line)
            if m is None:
                out_lines.append(line)
                continue
            verse_id = m.group("verse")
            rest_norm = (m.group("rest") or "").strip()[:_SIG_PREFIX_CHARS]
            key = (verse_id, rest_norm)
            if key in self._seen:
                suppressed.append(verse_id)
                continue
            self._seen[key] = idx
            out_lines.append(line)
        return "\n".join(out_lines), suppressed
