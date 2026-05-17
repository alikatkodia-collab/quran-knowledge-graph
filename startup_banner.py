"""Pure helpers for app_free.py's startup banner.

Lives in its own module so the wording is testable without importing
app_free.py (which connects to Neo4j at import time). The function is
side-effect-free: pass in the resolved config, get the multiline banner
string back.

The wording deliberately says "Configured backend" rather than "Default
backend". The previous phrasing implied the named backend would actually
serve requests, but the agent's fallback chain may swap to local Ollama
mid-request when the configured backend errors (e.g. OpenRouter 404 on a
retired free-tier model). The banner now states intent, not runtime
reality — per-request fallback events are logged separately by
shared_agent.agent_stream.
"""

from __future__ import annotations


def format_startup_banner(
    *,
    prefer_openrouter: bool,
    openrouter_api_key: str,
    openrouter_model: str,
    ollama_model: str,
    port: int,
) -> str:
    lines: list[str] = [""]
    if prefer_openrouter and openrouter_api_key:
        lines.append(
            f"[FREE] Configured backend: OpenRouter ({openrouter_model})"
        )
        lines.append(
            f"[FREE] On error, falls back to local Ollama ({ollama_model}) — "
            f"watch the per-request log for [fallback] lines"
        )
    elif prefer_openrouter and not openrouter_api_key:
        lines.append(
            "[FREE] --openrouter requested but OPENROUTER_API_KEY is not set."
        )
        lines.append(f"[FREE] Falling back to local model: {ollama_model}")
    else:
        lines.append(
            f"[FREE] Configured backend: local Ollama ({ollama_model})"
        )
    lines.append("[FREE] Cost: $0.00 (local)")
    lines.append(f"[FREE] Quran Graph: http://localhost:{port}")
    lines.append("")
    return "\n".join(lines)
