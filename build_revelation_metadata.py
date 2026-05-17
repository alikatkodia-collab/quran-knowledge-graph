"""
Backfill Meccan/Medinan classification and chronological revelation order
onto every (:Sura) node in the Quran knowledge graph.

Source: data/metadata/sura_revelation.py — Egyptian (Cairo / King Fuad)
standard, hand-encoded and bake-in (no network at runtime).

Properties written:
  - revelation_location          "Meccan" | "Medinan"
  - revelation_location_source   "egyptian_standard_2026"
  - revelation_order             int 1..114

Behaviour:
  - ADDITIVE METADATA ONLY. Never touches Verse text, arabicText, or any
    pure-data property.
  - Idempotent: re-running overwrites the three properties to the same
    values but never creates Sura nodes (uses MATCH, not MERGE).
  - Exits non-zero if the graph has fewer than 114 :Sura nodes or if any
    of the 114 surahs is missing.

Usage:
    py build_revelation_metadata.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from data.metadata.sura_revelation import SOURCE, SURA_REVELATION  # noqa: E402

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB       = os.getenv("NEO4J_DATABASE", "quran")


def _count(session, query: str) -> int:
    return session.run(query).single()[0]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print("Sura revelation-metadata backfill")
    print("=" * 60)
    print(f"  Source label : {SOURCE}")
    print(f"  Entries      : {len(SURA_REVELATION)}")
    print(f"  Neo4j URI    : {NEO4J_URI}")
    print(f"  Database     : {NEO4J_DB}")

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session(database=NEO4J_DB) as session:
        sura_count = _count(session, "MATCH (s:Sura) RETURN count(s)")
        print(f"\n  :Sura nodes in graph: {sura_count}")
        if sura_count < 114:
            print(
                f"\nERROR: graph has {sura_count} :Sura nodes; expected 114. "
                "Refusing to backfill an incomplete graph.",
                file=sys.stderr,
            )
            driver.close()
            return 1

        # Check every surah in our table actually has a node before we write.
        present_rows = session.run(
            "MATCH (s:Sura) WHERE s.number IN $nums RETURN s.number AS n",
            nums=list(SURA_REVELATION.keys()),
        ).data()
        present_numbers = {row["n"] for row in present_rows}
        missing = set(SURA_REVELATION.keys()) - present_numbers
        if missing:
            print(
                f"\nERROR: the following surah numbers are missing from the "
                f"graph: {sorted(missing)}. Refusing to silently skip them.",
                file=sys.stderr,
            )
            driver.close()
            return 1

        before_with_location = _count(
            session,
            "MATCH (s:Sura) WHERE s.revelation_location IS NOT NULL RETURN count(s)",
        )
        print(f"  Already have revelation_location: {before_with_location}")

        # Build payload
        rows = [
            {
                "number": n,
                "location": entry["location"],
                "order": entry["order"],
                "source": SOURCE,
            }
            for n, entry in SURA_REVELATION.items()
        ]

        # Idempotent write — MATCH (not MERGE) so we never create a Sura.
        print("\n  Writing properties...")
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (s:Sura {number: row.number})
            SET s.revelation_location = row.location,
                s.revelation_location_source = row.source,
                s.revelation_order = row.order
            RETURN count(s) AS updated
            """,
            rows=rows,
        ).single()
        updated = result["updated"]
        print(f"    Updated {updated} :Sura nodes.")

        # Verification.
        after_with_location = _count(
            session,
            "MATCH (s:Sura) WHERE s.revelation_location IS NOT NULL RETURN count(s)",
        )
        meccan = _count(
            session,
            "MATCH (s:Sura {revelation_location: 'Meccan'}) RETURN count(s)",
        )
        medinan = _count(
            session,
            "MATCH (s:Sura {revelation_location: 'Medinan'}) RETURN count(s)",
        )
        missing_after = _count(
            session,
            "MATCH (s:Sura) WHERE s.revelation_location IS NULL RETURN count(s)",
        )
        order_distinct = _count(
            session,
            "MATCH (s:Sura) RETURN count(DISTINCT s.revelation_order)",
        )

        print("\n  Verification:")
        print(f"    Total with revelation_location : {after_with_location}")
        print(f"    Meccan                         : {meccan}")
        print(f"    Medinan                        : {medinan}")
        print(f"    Missing revelation_location    : {missing_after}")
        print(f"    Distinct revelation_order      : {order_distinct}")

        ok = (
            after_with_location == 114
            and meccan == 86
            and medinan == 28
            and missing_after == 0
            and order_distinct == 114
        )
        if not ok:
            print("\nERROR: verification counts did not match expectations.", file=sys.stderr)
            driver.close()
            return 2

    driver.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
