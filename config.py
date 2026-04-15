"""
Quran Knowledge Graph — Centralised configuration.

Loads pipeline_config.yaml once, exposes typed accessors.
The autoresearch optimizer modifies pipeline_config.yaml and re-imports this module.
"""

import os
import yaml
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent
_CONFIG_PATH = _PROJECT_ROOT / "pipeline_config.yaml"

# ── load config ──────────────────────────────────────────────────────────────

def _load() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)

_cfg = _load()

def reload():
    """Re-read config from disk (call after optimizer writes new values)."""
    global _cfg
    _cfg = _load()

def raw() -> dict:
    """Return the full config dict (for serialisation / diffing)."""
    return _cfg

# ── LLM ──────────────────────────────────────────────────────────────────────

def llm_model() -> str:
    return _cfg["llm"]["model"]

def llm_max_tokens() -> int:
    return _cfg["llm"]["max_tokens"]

def llm_temperature() -> float:
    return _cfg["llm"].get("temperature", 0.4)

# ── Embedding ────────────────────────────────────────────────────────────────

def embedding_model() -> str:
    return _cfg["embedding"]["model"]

# ── Retrieval knobs ──────────────────────────────────────────────────────────

def _ret(tool: str, key: str):
    return _cfg["retrieval"][tool][key]

def semantic_default_top_k() -> int:
    return _ret("semantic_search", "default_top_k")

def semantic_max_top_k() -> int:
    return _ret("semantic_search", "max_top_k")

def traverse_seed_limit() -> int:
    return _ret("traverse_topic", "seed_limit")

def traverse_hop1_limit() -> int:
    return _ret("traverse_topic", "hop1_limit")

def traverse_hop2_limit() -> int:
    return _ret("traverse_topic", "hop2_limit")

def traverse_max_hops() -> int:
    return _ret("traverse_topic", "max_hops")

def get_verse_keyword_limit() -> int:
    return _ret("get_verse", "keyword_limit")

def get_verse_neighbour_limit() -> int:
    return _ret("get_verse", "neighbour_limit")

def get_verse_shared_kw_limit() -> int:
    return _ret("get_verse", "shared_kw_limit")

def find_path_max_depth() -> int:
    return _ret("find_path", "max_depth")

def find_path_bridge_kw_limit() -> int:
    return _ret("find_path", "bridge_kw_limit")

def explore_surah_cross_limit() -> int:
    return _ret("explore_surah", "cross_surah_limit")

def search_keyword_fuzzy_prefix() -> int:
    return _ret("search_keyword", "fuzzy_prefix_len")

def search_keyword_fuzzy_limit() -> int:
    return _ret("search_keyword", "fuzzy_limit")

# ── Visualisation ────────────────────────────────────────────────────────────

def vis(key: str) -> int:
    return _cfg["visualisation"][key]

# ── System prompt ────────────────────────────────────────────────────────────

def system_prompt() -> str:
    """Return the system prompt string. If the config value is a .txt path, read it."""
    val = _cfg.get("system_prompt", "")
    if val.endswith(".txt"):
        p = _PROJECT_ROOT / val
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return val

# ── Etymology ───────────────────────────────────────────────────────────────

def _etym(key: str):
    return _cfg.get("etymology", {}).get(key)

def etymology_word_lookup_max() -> int:
    return _etym("word_lookup_max_results") or 50

def etymology_root_family_max() -> int:
    return _etym("root_family_max_lemmas") or 100

def etymology_include_particles() -> bool:
    return _etym("verse_words_include_particles") if _etym("verse_words_include_particles") is not None else True

def etymology_semantic_domain_max() -> int:
    return _etym("semantic_domain_max_roots") or 50

def etymology_wujuh_max() -> int:
    return _etym("wujuh_max_senses") or 20

# ── Scoring thresholds ───────────────────────────────────────────────────────

def scoring(key: str) -> float:
    return _cfg["scoring"][key]

# ── Evaluation ───────────────────────────────────────────────────────────────

def eval_dataset_path() -> Path:
    return _PROJECT_ROOT / _cfg["evaluation"]["test_dataset"]

def eval_metrics() -> list[str]:
    return _cfg["evaluation"]["metrics"]

def eval_weights() -> dict[str, float]:
    return _cfg["evaluation"]["composite_weights"]
