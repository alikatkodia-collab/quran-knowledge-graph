"""
Pure-data tests for data/metadata/sura_revelation.py.

No Neo4j, no network — these run in CI without any infrastructure.
"""

from __future__ import annotations

from data.metadata.sura_revelation import SOURCE, SURA_REVELATION


def test_source_label():
    assert SOURCE == "egyptian_standard_2026"


def test_114_entries():
    assert len(SURA_REVELATION) == 114


def test_every_surah_present():
    assert set(SURA_REVELATION.keys()) == set(range(1, 115))


def test_meccan_medinan_counts():
    locations = [entry["location"] for entry in SURA_REVELATION.values()]
    meccan = sum(1 for loc in locations if loc == "Meccan")
    medinan = sum(1 for loc in locations if loc == "Medinan")
    assert meccan == 86, f"expected 86 Meccan, got {meccan}"
    assert medinan == 28, f"expected 28 Medinan, got {medinan}"
    # And nothing else slipped in.
    assert meccan + medinan == 114


def test_location_values_are_strict():
    allowed = {"Meccan", "Medinan"}
    for n, entry in SURA_REVELATION.items():
        assert entry["location"] in allowed, (
            f"surah {n} has invalid location {entry['location']!r}"
        )


def test_order_is_1_to_114_no_duplicates():
    orders = [entry["order"] for entry in SURA_REVELATION.values()]
    assert sorted(orders) == list(range(1, 115)), (
        "revelation_order must be a permutation of 1..114"
    )


def test_order_values_are_ints():
    for n, entry in SURA_REVELATION.items():
        assert isinstance(entry["order"], int), (
            f"surah {n} has non-int order {entry['order']!r}"
        )
        assert 1 <= entry["order"] <= 114


def test_known_egyptian_standard_anchors():
    # First revealed: Al-Alaq (96), order 1
    assert SURA_REVELATION[96]["order"] == 1
    assert SURA_REVELATION[96]["location"] == "Meccan"
    # Last revealed in Egyptian standard: An-Nasr (110), order 114
    assert SURA_REVELATION[110]["order"] == 114
    assert SURA_REVELATION[110]["location"] == "Medinan"
    # First Medinan in chronological order is Al-Baqarah (2), order 87
    assert SURA_REVELATION[2]["order"] == 87
    assert SURA_REVELATION[2]["location"] == "Medinan"
    # At-Tawbah (9) — late Medinan, order 113
    assert SURA_REVELATION[9]["order"] == 113
    assert SURA_REVELATION[9]["location"] == "Medinan"
    # Al-Fatihah (1) — Meccan in the standard classification
    assert SURA_REVELATION[1]["location"] == "Meccan"


def test_chronological_split_at_order_87():
    # In the Egyptian standard, every surah revealed before order 87 is
    # classified as Meccan, and every surah from order 87 onward is Medinan.
    # This is what makes the standard internally consistent — useful sanity
    # check that no entry's location contradicts its order.
    for n, entry in SURA_REVELATION.items():
        if entry["order"] < 87:
            assert entry["location"] == "Meccan", (
                f"surah {n} has order {entry['order']} but location {entry['location']}"
            )
        else:
            assert entry["location"] == "Medinan", (
                f"surah {n} has order {entry['order']} but location {entry['location']}"
            )
