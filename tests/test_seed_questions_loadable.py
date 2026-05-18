"""Tests for data/seed_questions/ — the extracted question pools.

These YAMLs are static data, not code, but the project relies on their
shape if/when a future seeder is reintroduced. These tests pin the
contract:

  - All phase YAMLs load as valid YAML
  - Each one has the documented top-level keys
  - The 'questions' field is a list of non-empty strings
  - The 'count' field matches len(questions)
  - The README's manifest table matches reality (counts add up)

Caught early, a malformed YAML or a count drift would otherwise only
surface when something downstream tries to load the pool.
"""

from __future__ import annotations

from pathlib import Path

import yaml


SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed_questions"


def _phase_files() -> list[Path]:
    files = sorted(SEED_DIR.glob("phase*.yaml"))
    assert files, f"no phase*.yaml found in {SEED_DIR}"
    return files


def test_seed_dir_exists():
    assert SEED_DIR.is_dir(), f"{SEED_DIR} should be a directory"
    assert (SEED_DIR / "README.md").is_file(), "README.md should accompany the YAMLs"


def test_every_phase_yaml_loads():
    """Each phase YAML parses as valid YAML and is a top-level dict."""
    for path in _phase_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{path.name}: expected top-level dict"


def test_every_phase_yaml_has_required_keys():
    required = {"phase", "source_file", "count", "questions"}
    for path in _phase_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        missing = required - set(data.keys())
        assert not missing, f"{path.name}: missing keys {missing}"


def test_count_matches_questions_length():
    for path in _phase_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["count"] == len(data["questions"]), (
            f"{path.name}: count={data['count']} but len(questions)="
            f"{len(data['questions'])}"
        )


def test_questions_are_nonempty_strings():
    for path in _phase_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for i, q in enumerate(data["questions"]):
            assert isinstance(q, str), (
                f"{path.name}: questions[{i}] is {type(q).__name__}, not str"
            )
            assert q.strip(), f"{path.name}: questions[{i}] is empty/whitespace"


def test_phase_number_matches_filename():
    for path in _phase_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Filenames are phaseNN.yaml (zero-padded); phase field is int.
        expected = int(path.stem.replace("phase", ""))
        assert data["phase"] == expected, (
            f"{path.name}: phase field = {data['phase']!r}, expected {expected}"
        )


def test_total_question_count_matches_readme():
    """README documents 1186 total. If pool changes, README must too."""
    total = 0
    for path in _phase_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        total += data["count"]
    expected = 1186
    assert total == expected, (
        f"total questions across all phases = {total}; README documents "
        f"{expected}. Either the pool changed (update README.md) or a "
        f"phase file got corrupted."
    )
