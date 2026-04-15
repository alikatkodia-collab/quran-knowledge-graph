"""
Tool Result Compressor — reduces token cost by slimming down tool results
before they are fed back into the conversation.

Claude only needs verse references + short snippets to decide what to cite.
Full verse text is fetched separately for tooltips via _fetch_verses().

Typical savings: 60-70% fewer tokens per tool result.
"""

import json


def compress_tool_result(tool_name: str, result_str: str) -> str:
    """
    Compress a tool result JSON string for feeding back to Claude.
    Keeps structure + references but trims long text fields.

    The FULL result_str is still used for graph extraction and etymology panels.
    Only the compressed version goes into the conversation history.
    """
    try:
        result = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return result_str

    if "error" in result:
        return result_str  # don't compress errors

    _compress_dict(result)
    return json.dumps(result, ensure_ascii=False)


def _compress_dict(obj):
    """Recursively compress a dict in-place."""
    if isinstance(obj, dict):
        # Trim long text fields
        for key in ("text", "translation"):
            if key in obj and isinstance(obj[key], str) and len(obj[key]) > 100:
                obj[key] = obj[key][:100] + "..."

        # Drop arabic_text entirely (saves ~300 chars per verse)
        # Claude can still reference Arabic roots via the root tools
        obj.pop("arabic_text", None)

        # Drop embeddings if present
        obj.pop("embedding", None)

        # Trim long keyword lists
        if "keywords" in obj and isinstance(obj["keywords"], list) and len(obj["keywords"]) > 8:
            obj["keywords"] = obj["keywords"][:8]

        # Recurse into nested dicts and lists
        for v in obj.values():
            _compress_dict(v)

    elif isinstance(obj, list):
        for item in obj:
            _compress_dict(item)
