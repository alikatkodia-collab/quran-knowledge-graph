"""
Tests for answer_cache's in-memory load cache (the I/O-tax fix).

The 2026-05-19 server-degradation diagnosis flagged answer_cache as
doing ~1.9s of synchronous JSON I/O per /chat request at 1500+ entries
(load + load + save of a multi-MB file). The fix is an in-memory
singleton keyed by file mtime so consecutive reads in the same process
serve from memory.

These tests pin three properties:

  1. Consecutive _load_cache calls in the same process must not
     re-read the file (the headline optimisation).
  2. _load_cache must invalidate and re-read if the file's mtime
     changes (another process modified it — overnight seeder etc.).
  3. _save_cache must update the in-memory view so the next _load
     returns the freshly-saved data without touching disk.

No Neo4j, no LLM, no sentence-transformers — pure I/O behaviour over
a tmp-path fixture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import answer_cache


@pytest.fixture
def tmp_cache(tmp_path: Path, monkeypatch):
    """Point the cache at a tmp file and reset the in-memory singleton."""
    tmp_file = tmp_path / "answer_cache.json"
    monkeypatch.setattr(answer_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(answer_cache, "CACHE_FILE", tmp_file)
    answer_cache._reset_memory_cache_for_tests()
    return tmp_file


def _write_cache(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


def test_consecutive_loads_hit_memory_not_disk(tmp_cache, monkeypatch):
    """Headline optimisation: second _load_cache reads zero bytes from disk."""
    _write_cache(tmp_cache, [{"q": 1}, {"q": 2}])

    open_calls = {"n": 0}
    real_open = open

    def counting_open(path, *args, **kwargs):
        if str(path) == str(tmp_cache):
            open_calls["n"] += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)

    first = answer_cache._load_cache()
    second = answer_cache._load_cache()
    third = answer_cache._load_cache()

    assert first == [{"q": 1}, {"q": 2}]
    assert second == first
    assert third == first
    # Only the first call should have opened the file. The subsequent
    # two should have served from the in-memory singleton.
    assert open_calls["n"] == 1, (
        f"expected exactly 1 open() of the cache file across 3 loads; "
        f"got {open_calls['n']} — the in-memory singleton isn't working "
        f"and the I/O tax is back."
    )


def test_load_invalidates_when_file_mtime_changes(tmp_cache):
    """An external writer changes the file → next load must re-read."""
    _write_cache(tmp_cache, [{"v": "first"}])
    first = answer_cache._load_cache()
    assert first == [{"v": "first"}]

    # Simulate a second process / overnight seeder rewriting the file.
    # Bump the mtime explicitly so the test is robust on file systems
    # with coarse mtime granularity.
    _write_cache(tmp_cache, [{"v": "second"}, {"v": "third"}])
    new_mtime = tmp_cache.stat().st_mtime + 1.0
    os.utime(tmp_cache, (new_mtime, new_mtime))

    second = answer_cache._load_cache()
    assert second == [{"v": "second"}, {"v": "third"}], (
        "Cache did not invalidate when file mtime changed — a stale "
        "in-memory view will hide concurrent writes from other processes."
    )


def test_load_handles_file_deleted_between_calls(tmp_cache):
    """File-not-exists path must return [] cleanly and clear the singleton."""
    _write_cache(tmp_cache, [{"x": 1}])
    first = answer_cache._load_cache()
    assert first == [{"x": 1}]

    tmp_cache.unlink()

    second = answer_cache._load_cache()
    assert second == [], (
        "After the file is deleted, _load_cache must return [] rather "
        "than the stale in-memory view."
    )


def test_save_updates_memory_view(tmp_cache, monkeypatch):
    """After _save_cache, the next _load returns the saved data without disk read."""
    _write_cache(tmp_cache, [{"old": True}])
    answer_cache._load_cache()  # warm the cache

    open_calls = {"n": 0}
    real_open = open

    def counting_open(path, *args, **kwargs):
        if str(path) == str(tmp_cache):
            open_calls["n"] += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)

    new_entries = [{"new": True}, {"more": True}]
    answer_cache._save_cache(new_entries)

    after = answer_cache._load_cache()
    assert after == new_entries

    # The save itself opened the file once (the write). The subsequent
    # load should have served from memory — not opened the file again.
    assert open_calls["n"] == 1, (
        f"expected exactly 1 open() (the save itself); got {open_calls['n']}. "
        f"_save_cache should update the in-memory singleton so the next "
        f"load doesn't re-read from disk."
    )


def test_load_returns_empty_on_corrupt_file(tmp_cache):
    """A corrupted JSON file must not poison the in-memory cache forever."""
    tmp_cache.write_text("{not valid json[[[", encoding="utf-8")

    first = answer_cache._load_cache()
    assert first == []

    # After a corrupt read, the memory cache is cleared. If the file is
    # later repaired, the next load picks up the new content.
    _write_cache(tmp_cache, [{"recovered": True}])
    new_mtime = tmp_cache.stat().st_mtime + 1.0
    os.utime(tmp_cache, (new_mtime, new_mtime))

    second = answer_cache._load_cache()
    assert second == [{"recovered": True}], (
        "After a corrupt read clears the cache, a subsequent valid "
        "write must be picked up on the next load."
    )


def test_repeated_loads_after_write_serve_from_memory(tmp_cache, monkeypatch):
    """End-to-end: 1 write + N reads → exactly 1 disk open total."""
    answer_cache._save_cache([{"first": "save"}])

    open_calls = {"n": 0}
    real_open = open

    def counting_open(path, *args, **kwargs):
        if str(path) == str(tmp_cache):
            open_calls["n"] += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)

    # Five loads after the save should all serve from memory.
    for _ in range(5):
        assert answer_cache._load_cache() == [{"first": "save"}]

    assert open_calls["n"] == 0, (
        f"expected zero opens across 5 loads (all from memory); got {open_calls['n']}."
    )
