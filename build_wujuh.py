"""
Build Wujuh wa al-Naza'ir (Quranic polysemy) data.

Reads data/wujuh_nazair.yaml and creates CSV files for Neo4j import.

Usage:
    py build_wujuh.py
"""

import csv
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent
WUJUH_FILE = PROJECT_ROOT / "data" / "wujuh_nazair.yaml"
DATA_DIR = PROJECT_ROOT / "data"


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print("Wujuh wa al-Naza'ir Builder")
    print("=" * 60)

    # Load YAML
    print("\n[1] Loading wujuh data...")
    with open(WUJUH_FILE, encoding='utf-8') as f:
        wujuh = yaml.safe_load(f)
    print(f"  Loaded {len(wujuh)} root entries")

    # Build entries
    wujuh_entries = []
    total_senses = 0

    for root, root_data in wujuh.items():
        lemma = root_data.get('lemma', '')
        for i, sense in enumerate(root_data.get('senses', []), 1):
            sense_id = f"{root}:{i}"
            wujuh_entries.append({
                'root': root,
                'senseId': sense_id,
                'lemma': lemma,
                'meaningEn': sense.get('meaning', ''),
                'meaningAr': sense.get('meaningAr', ''),
                'sampleVerses': json.dumps(sense.get('verses', []), ensure_ascii=False),
            })
            total_senses += 1

    print(f"\n[2] Built {total_senses} wujuh senses across {len(wujuh)} roots")
    avg = total_senses / len(wujuh) if wujuh else 0
    print(f"  Average senses per root: {avg:.1f}")

    # Export CSV
    print(f"\n[3] Exporting CSV...")

    we_csv = DATA_DIR / "wujuh_entries.csv"
    with open(we_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'root', 'senseId', 'lemma', 'meaningEn', 'meaningAr', 'sampleVerses'
        ])
        w.writeheader()
        w.writerows(wujuh_entries)
    print(f"  Wrote {len(wujuh_entries)} wujuh entries to {we_csv.name}")

    print("\nDone.")


if __name__ == '__main__':
    main()
