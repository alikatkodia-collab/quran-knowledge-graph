"""Regression tests for the app_free.py startup banner.

The banner is the only operator-visible signal of which backend will serve
requests. The previous wording ("Default backend: X") implied the named
backend would actually be used, but the fallback chain can swap to local
Ollama mid-request without updating the banner. The new wording must say
"Configured backend" and surface the fallback intent.

Concrete repro that motivated this test:
  - Operator sets OPENROUTER_MODEL=deepseek/deepseek-chat-v3:free
  - DeepSeek free-tier returns 404
  - Fallback chain swaps to local qwen3:14b for every request
  - Old banner kept claiming "Default backend: OpenRouter (deepseek...)"
"""

from __future__ import annotations

from startup_banner import format_startup_banner


def test_banner_with_openrouter_names_configured_backend_and_fallback():
    out = format_startup_banner(
        prefer_openrouter=True,
        openrouter_api_key="sk-or-v1-xxx",
        openrouter_model="deepseek/deepseek-chat-v3:free",
        ollama_model="qwen3:14b",
        port=8085,
    )
    assert "Configured backend: OpenRouter (deepseek/deepseek-chat-v3:free)" in out
    assert "falls back to local Ollama (qwen3:14b)" in out
    assert "Quran Graph: http://localhost:8085" in out


def test_banner_never_uses_the_stale_default_backend_phrasing():
    """The word 'Default' is what misled the operator. Don't bring it back."""
    out = format_startup_banner(
        prefer_openrouter=True,
        openrouter_api_key="sk-or-v1-xxx",
        openrouter_model="deepseek/deepseek-chat-v3:free",
        ollama_model="qwen3:14b",
        port=8085,
    )
    assert "Default backend" not in out


def test_banner_without_openrouter_key_warns_and_falls_back():
    out = format_startup_banner(
        prefer_openrouter=True,
        openrouter_api_key="",
        openrouter_model="qwen/qwen3-coder:free",
        ollama_model="qwen3:8b",
        port=8085,
    )
    assert "OPENROUTER_API_KEY is not set" in out
    assert "Falling back to local model: qwen3:8b" in out


def test_banner_local_only_mentions_ollama_as_configured_backend():
    out = format_startup_banner(
        prefer_openrouter=False,
        openrouter_api_key="",
        openrouter_model="qwen/qwen3-coder:free",
        ollama_model="qwen3:8b",
        port=8085,
    )
    assert "Configured backend: local Ollama (qwen3:8b)" in out
    assert "OpenRouter" not in out
