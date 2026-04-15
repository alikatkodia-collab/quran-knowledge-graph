"""
Build Semantic Domain nodes and root-domain relationships.

Reads data/semantic_domains.yaml and creates CSV files for Neo4j import.

Usage:
    py build_semantic_domains.py
"""

import csv
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent
DOMAINS_FILE = PROJECT_ROOT / "data" / "semantic_domains.yaml"
DATA_DIR = PROJECT_ROOT / "data"


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print("Semantic Domain Builder")
    print("=" * 60)

    # Load YAML
    print("\n[1] Loading semantic domains...")
    with open(DOMAINS_FILE, encoding='utf-8') as f:
        domains = yaml.safe_load(f)
    print(f"  Loaded {len(domains)} domains")

    # Load existing roots to validate
    root_csv = DATA_DIR / "arabic_root_nodes.csv"
    existing_roots = set()
    if root_csv.exists():
        with open(root_csv, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_roots.add(row['root'])
        print(f"  Found {len(existing_roots)} existing roots in graph")

    # Build domain nodes and relationships
    domain_nodes = []
    root_domain_rels = []
    total_roots = 0
    matched_roots = 0

    for domain_id, domain_data in domains.items():
        domain_nodes.append({
            'domainId': domain_id,
            'nameEn': domain_id.replace('_', ' ').title(),
            'nameAr': domain_data.get('arabic', ''),
            'description': domain_data.get('description', ''),
        })

        for root in domain_data.get('roots', []):
            total_roots += 1
            # Strip comment (everything after #)
            root_clean = root.split('#')[0].strip()
            root_domain_rels.append({
                'root': root_clean,
                'domainId': domain_id,
            })
            if root_clean in existing_roots:
                matched_roots += 1

    print(f"\n[2] Built {len(domain_nodes)} domain nodes")
    print(f"  Total root-domain links: {total_roots}")
    if existing_roots:
        print(f"  Roots matching graph: {matched_roots}/{total_roots} ({100*matched_roots/total_roots:.0f}%)")

    # Export CSVs
    print(f"\n[3] Exporting CSVs...")

    dn_csv = DATA_DIR / "semantic_domain_nodes.csv"
    with open(dn_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['domainId', 'nameEn', 'nameAr', 'description'])
        w.writeheader()
        w.writerows(domain_nodes)
    print(f"  Wrote {len(domain_nodes)} domain nodes to {dn_csv.name}")

    rd_csv = DATA_DIR / "root_domain_rels.csv"
    with open(rd_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['root', 'domainId'])
        w.writeheader()
        w.writerows(root_domain_rels)
    print(f"  Wrote {len(root_domain_rels)} root-domain rels to {rd_csv.name}")

    print("\nDone.")


if __name__ == '__main__':
    main()
