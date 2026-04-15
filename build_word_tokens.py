"""
Build word-level token data for the Quranic Etymology layer.

Parses quran-morphology.txt at the segment level and reassembles ~77K word
tokens with full morphological detail: root, lemma, POS, verbal form,
person/gender/number/case/mood/voice, and morphological pattern (wazn).

Also produces Lemma nodes (~1,800 unique lemmas) and MorphPattern nodes (~80).

Usage:
    py build_word_tokens.py                # build all CSVs
    py build_word_tokens.py --stats        # just print statistics
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MORPH_FILE = PROJECT_ROOT / "data" / "quran-morphology.txt"
ARABIC_RAW = PROJECT_ROOT / "data" / "quran-arabic-raw.json"
PATTERNS_FILE = PROJECT_ROOT / "data" / "morph_patterns.json"
DATA_DIR = PROJECT_ROOT / "data"

# Verses excluded from Khalifa's translation
SKIP_VERSES = {(9, 128), (9, 129)}

# Regex patterns for extracting features from the morphology tags
_ROOT_RE = re.compile(r'ROOT:([^|]+)')
_LEM_RE = re.compile(r'LEM:([^|]+)')
_VF_RE = re.compile(r'VF:(\d+)')
_MOOD_RE = re.compile(r'MOOD:(\w+)')

# POS-like feature tags that appear in the features column
_VERB_ASPECTS = {'PERF', 'IMPF', 'IMPV'}
_NOMINAL_TYPES = {'ACT_PCPL', 'PASS_PCPL', 'VN'}
_PARTICLE_TYPES = {'P', 'CONJ', 'DET', 'NEG', 'VOC', 'COND', 'RES',
                   'EMPH', 'EQ', 'ANS', 'INC', 'AVR', 'INTG', 'PREV',
                   'RSLT', 'SUB', 'CERT', 'SUP', 'AMD', 'CIRC', 'COM',
                   'REM', 'FAM', 'RET', 'EXP', 'EXH', 'SURP', 'CAUS',
                   'FUT', 'DIST', 'ADDR'}
_PRONOUN_TYPES = {'PRON'}
_PREFIX_SUFFIX = {'PREF', 'SUFF'}

# Person/gender/number tags
_PGN_TAGS = {'1S', '2MS', '2FS', '3MS', '3FS', '1P', '2MP', '2FP',
             '3MP', '3FP', '2D', '3D'}
_GENDER_TAGS = {'M', 'F', 'MS', 'FS', 'MP', 'FP', 'MD', 'FD'}
_CASE_TAGS = {'NOM', 'ACC', 'GEN'}

# Buckwalter transliteration table (reused from build_arabic_roots.py)
_AR2BW = str.maketrans({
    'ا': 'A', 'ب': 'b', 'ت': 't', 'ث': 'v', 'ج': 'j', 'ح': 'H',
    'خ': 'x', 'د': 'd', 'ذ': '*', 'ر': 'r', 'ز': 'z', 'س': 's',
    'ش': '$', 'ص': 'S', 'ض': 'D', 'ط': 'T', 'ظ': 'Z', 'ع': 'E',
    'غ': 'g', 'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm',
    'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y', 'ء': "'", 'أ': '>',
    'إ': '<', 'آ': '|', 'ؤ': '&', 'ئ': '}', 'ة': 'p', 'ى': 'Y',
    'ٱ': '{',
})


def to_buckwalter(arabic: str) -> str:
    return arabic.translate(_AR2BW)


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel) for clean matching."""
    # Arabic diacritic Unicode range: 0610-061A, 064B-065F, 0670
    return re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670]', '', text)


def parse_features(features_str: str) -> dict:
    """Parse the pipe-separated features string into structured data."""
    tags = features_str.split('|')
    result = {
        'root': None,
        'lemma': None,
        'verb_form': None,
        'aspect': None,      # PERF, IMPF, IMPV
        'mood': None,         # IND, JUS, SUBJ
        'voice': 'ACT',       # ACT or PASS
        'person': None,
        'gender': None,
        'number': None,
        'case': None,
        'state': None,        # INDEF or definite
        'nominal_type': None, # ACT_PCPL, PASS_PCPL, VN
        'is_prefix': False,
        'is_suffix': False,
        'is_proper_noun': False,
        'is_relative': False,
        'is_demonstrative': False,
        'is_pronoun': False,
        'adj': False,
    }

    root_m = _ROOT_RE.search(features_str)
    if root_m:
        result['root'] = root_m.group(1)

    lem_m = _LEM_RE.search(features_str)
    if lem_m:
        result['lemma'] = lem_m.group(1)

    vf_m = _VF_RE.search(features_str)
    if vf_m:
        result['verb_form'] = vf_m.group(1)

    mood_m = _MOOD_RE.search(features_str)
    if mood_m:
        result['mood'] = mood_m.group(1)

    for tag in tags:
        tag = tag.strip()
        if tag in _VERB_ASPECTS:
            result['aspect'] = tag
        elif tag in _NOMINAL_TYPES:
            result['nominal_type'] = tag
        elif tag == 'PASS':
            result['voice'] = 'PASS'
        elif tag in _PGN_TAGS:
            result['person'] = tag[0]  # '1', '2', '3'
            if len(tag) == 2:
                result['gender'] = tag[1]  # M or F
                result['number'] = 'S'     # singular by default
            elif len(tag) == 3:
                result['gender'] = tag[1]
                result['number'] = tag[2]  # S, P, D
        elif tag in _GENDER_TAGS:
            if len(tag) == 1:
                result['gender'] = tag
            else:
                result['gender'] = tag[0]
                result['number'] = tag[1]
        elif tag in _CASE_TAGS:
            result['case'] = tag
        elif tag == 'INDEF':
            result['state'] = 'INDEF'
        elif tag == 'PN':
            result['is_proper_noun'] = True
        elif tag == 'REL':
            result['is_relative'] = True
        elif tag == 'DEM':
            result['is_demonstrative'] = True
        elif tag in _PRONOUN_TYPES:
            result['is_pronoun'] = True
        elif tag == 'ADJ':
            result['adj'] = True
        elif tag in _PREFIX_SUFFIX:
            if tag == 'PREF':
                result['is_prefix'] = True
            else:
                result['is_suffix'] = True
        elif tag == 'INL':
            result['nominal_type'] = 'INL'  # Quranic initials

    return result


def _build_lemma_pattern_map(morph_patterns: dict) -> dict:
    """Build a lookup from example lemmas to their pattern."""
    lemma_map = {}
    for pattern_ar, pdata in morph_patterns.get('common_noun_patterns', {}).items():
        for example in pdata.get('examples', []):
            clean = strip_diacritics(example)
            lemma_map[clean] = pattern_ar
            lemma_map[example] = pattern_ar
    return lemma_map

# Module-level cache
_LEMMA_PATTERN_MAP = None


def determine_wazn(pos: str, features: dict, morph_patterns: dict, lemma: str = '') -> str:
    """Determine the morphological pattern (wazn) for a word."""
    global _LEMMA_PATTERN_MAP
    vf = features.get('verb_form')
    nominal_type = features.get('nominal_type')

    if pos == 'V' and vf:
        vf_data = morph_patterns.get('verbal_forms', {}).get(vf)
        if vf_data:
            return vf_data['pattern']

    if nominal_type in ('ACT_PCPL', 'PASS_PCPL') and vf:
        np_data = morph_patterns.get('nominal_patterns', {}).get(nominal_type, {})
        pf = np_data.get('pattern_forms', {}).get(vf)
        if pf:
            return pf['pattern']

    if nominal_type == 'VN':
        return 'مصدر'  # Verbal noun — patterns vary too much to auto-detect

    # For nouns/adjectives without participle marking, try to match lemma
    # against known common noun patterns by example
    if lemma and pos in ('N', 'P') or features.get('adj'):
        if _LEMMA_PATTERN_MAP is None:
            _LEMMA_PATTERN_MAP = _build_lemma_pattern_map(morph_patterns)
        clean = strip_diacritics(lemma)
        if lemma in _LEMMA_PATTERN_MAP:
            return _LEMMA_PATTERN_MAP[lemma]
        if clean in _LEMMA_PATTERN_MAP:
            return _LEMMA_PATTERN_MAP[clean]

    return ''


def load_arabic_raw() -> dict:
    """Load display-quality Arabic text keyed by verseId."""
    with open(ARABIC_RAW, encoding='utf-8') as f:
        raw = json.load(f)

    verse_arabic = {}
    for surah_num, verses in raw.items():
        for v in verses:
            vid = f"{v['chapter']}:{v['verse']}"
            verse_arabic[vid] = v['text']
    return verse_arabic


def parse_morphology(morph_patterns: dict):
    """Parse quran-morphology.txt into word-level tokens.

    Returns:
        word_tokens: list[dict] — one per word position in the Quran
        lemma_data: dict[str, dict] — lemma → {root, pos, verses}
    """
    # Group segments by word position
    word_segments = defaultdict(list)  # (surah, verse, word) → [segment_dicts]

    with open(MORPH_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 4:
                continue

            pos_str, form, tag, features_str = parts[0], parts[1], parts[2], parts[3]

            loc = pos_str.split(':')
            if len(loc) < 4:
                continue

            surah, verse_num, word_pos, seg_pos = (
                int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3])
            )

            if (surah, verse_num) in SKIP_VERSES:
                continue

            features = parse_features(features_str)
            features['form'] = form
            features['tag'] = tag
            features['seg_pos'] = seg_pos

            word_segments[(surah, verse_num, word_pos)].append(features)

    # Reassemble word tokens from segments
    word_tokens = []
    lemma_data = defaultdict(lambda: {
        'root': '', 'pos_set': set(), 'verse_ids': set(), 'count': 0
    })

    for (surah, verse_num, word_pos), segments in sorted(word_segments.items()):
        vid = f"{surah}:{verse_num}"
        token_id = f"{surah}:{verse_num}:{word_pos}"

        # Concatenate Arabic forms for the full word
        arabic_text = ''.join(seg['form'] for seg in segments)

        # Find the "main" segment (the one with ROOT or the main POS)
        main_seg = None
        for seg in segments:
            if seg['root']:
                main_seg = seg
                break
        if not main_seg:
            # Use the non-prefix/suffix segment
            for seg in segments:
                if not seg['is_prefix'] and not seg['is_suffix']:
                    main_seg = seg
                    break
        if not main_seg:
            main_seg = segments[0]

        root = main_seg['root'] or ''
        lemma = main_seg['lemma'] or ''
        pos = main_seg['tag']
        verb_form = main_seg['verb_form'] or ''
        aspect = main_seg['aspect'] or ''
        mood = main_seg['mood'] or ''
        voice = main_seg['voice'] or ''
        person = main_seg['person'] or ''
        gender = main_seg['gender'] or ''
        number = main_seg['number'] or ''
        case = main_seg['case'] or ''
        state = main_seg['state'] or ''
        nominal_type = main_seg['nominal_type'] or ''
        is_adj = main_seg['adj']

        # Determine morphological pattern
        wazn = determine_wazn(pos, main_seg, morph_patterns, lemma)

        # Determine detailed POS
        if pos == 'V':
            detailed_pos = f"V.{aspect}" if aspect else 'V'
        elif pos == 'N':
            if main_seg['is_pronoun']:
                detailed_pos = 'PRON'
            elif main_seg['is_proper_noun']:
                detailed_pos = 'PN'
            elif main_seg['is_relative']:
                detailed_pos = 'REL'
            elif main_seg['is_demonstrative']:
                detailed_pos = 'DEM'
            elif nominal_type == 'INL':
                detailed_pos = 'INL'
            elif nominal_type == 'ACT_PCPL':
                detailed_pos = 'ACT_PCPL'
            elif nominal_type == 'PASS_PCPL':
                detailed_pos = 'PASS_PCPL'
            elif nominal_type == 'VN':
                detailed_pos = 'VN'
            elif is_adj:
                detailed_pos = 'ADJ'
            else:
                detailed_pos = 'N'
        elif pos == 'P':
            # Particles — check if it's a prefix-only segment
            has_content = any(not s['is_prefix'] and not s['is_suffix'] for s in segments)
            if not has_content:
                detailed_pos = 'PART'
            else:
                detailed_pos = 'PART'
        else:
            detailed_pos = pos

        # Build morph features JSON
        morph_features = {}
        if verb_form:
            morph_features['vf'] = verb_form
        if aspect:
            morph_features['asp'] = aspect
        if mood:
            morph_features['mood'] = mood
        if voice and voice != 'ACT':
            morph_features['voice'] = voice
        if person:
            morph_features['per'] = person
        if gender:
            morph_features['gen'] = gender
        if number:
            morph_features['num'] = number
        if case:
            morph_features['case'] = case
        if state:
            morph_features['state'] = state
        if nominal_type and nominal_type not in ('INL',):
            morph_features['ntype'] = nominal_type

        # Check for prefixes/suffixes
        prefixes = [seg['form'] for seg in segments if seg['is_prefix']]
        suffixes = [seg['form'] for seg in segments if seg['is_suffix']]
        if prefixes:
            morph_features['prefixes'] = prefixes
        if suffixes:
            morph_features['suffixes'] = suffixes

        arabic_clean = strip_diacritics(arabic_text)
        translit_bw = to_buckwalter(arabic_clean)

        token = {
            'tokenId': token_id,
            'verseId': vid,
            'wordPos': word_pos,
            'arabicText': arabic_text,
            'arabicClean': arabic_clean,
            'translitBW': translit_bw,
            'root': root,
            'lemma': lemma,
            'pos': detailed_pos,
            'morphFeatures': morph_features,
            'wazn': wazn,
        }
        word_tokens.append(token)

        # Track lemma data
        if lemma:
            ld = lemma_data[lemma]
            if root:
                ld['root'] = root
            ld['pos_set'].add(detailed_pos)
            ld['verse_ids'].add(vid)
            ld['count'] += 1

    return word_tokens, dict(lemma_data)


def build_lemma_nodes(lemma_data: dict) -> list:
    """Build Lemma node records from aggregated lemma data."""
    from build_arabic_roots import ROOT_GLOSSES

    lemma_nodes = []
    for lemma, data in sorted(lemma_data.items()):
        root = data['root']
        # Get gloss from root glosses
        gloss = ROOT_GLOSSES.get(root, '')
        # Pick the most common POS
        pos_list = sorted(data['pos_set'])
        primary_pos = pos_list[0] if pos_list else ''

        lemma_nodes.append({
            'lemma': lemma,
            'lemmaBW': to_buckwalter(strip_diacritics(lemma)),
            'root': root,
            'glossEn': gloss,
            'pos': primary_pos,
            'verseCount': len(data['verse_ids']),
        })

    return lemma_nodes


def build_pattern_nodes(morph_patterns: dict) -> list:
    """Build MorphPattern node records from the patterns definition file."""
    pattern_nodes = []

    # Verbal form patterns
    for vf_id, vf_data in morph_patterns.get('verbal_forms', {}).items():
        pattern_nodes.append({
            'pattern': vf_data['pattern'],
            'patternBW': vf_data['patternBW'],
            'label': vf_data['label'],
            'meaningTendency': vf_data['meaningTendency'],
        })

    # Nominal patterns (active/passive participle templates)
    for ntype, ntype_data in morph_patterns.get('nominal_patterns', {}).items():
        if 'pattern_forms' in ntype_data:
            for vf_id, pf_data in ntype_data['pattern_forms'].items():
                label = f"{ntype_data['label']} ({morph_patterns['verbal_forms'].get(vf_id, {}).get('label', f'Form {vf_id}')})"
                pattern_nodes.append({
                    'pattern': pf_data['pattern'],
                    'patternBW': pf_data['patternBW'],
                    'label': label,
                    'meaningTendency': ntype_data['meaningTendency'],
                })

    # Common noun patterns
    for pattern_ar, pdata in morph_patterns.get('common_noun_patterns', {}).items():
        pattern_nodes.append({
            'pattern': pattern_ar,
            'patternBW': pdata['patternBW'],
            'label': pdata['label'],
            'meaningTendency': pdata['meaningTendency'],
        })

    return pattern_nodes


def export_csvs(word_tokens, lemma_nodes, pattern_nodes):
    """Export all data as CSVs for Neo4j import."""
    # Word token nodes
    wt_csv = DATA_DIR / "word_token_nodes.csv"
    with open(wt_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'tokenId', 'verseId', 'wordPos', 'arabicText', 'arabicClean',
            'translitBW', 'root', 'lemma', 'pos', 'morphFeatures', 'wazn'
        ])
        w.writeheader()
        for t in word_tokens:
            row = dict(t)
            row['morphFeatures'] = json.dumps(t['morphFeatures'], ensure_ascii=False)
            w.writerow(row)
    print(f"  Wrote {len(word_tokens)} word tokens to {wt_csv.name}")

    # Lemma nodes
    lem_csv = DATA_DIR / "lemma_nodes.csv"
    with open(lem_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'lemma', 'lemmaBW', 'root', 'glossEn', 'pos', 'verseCount'
        ])
        w.writeheader()
        w.writerows(lemma_nodes)
    print(f"  Wrote {len(lemma_nodes)} lemmas to {lem_csv.name}")

    # MorphPattern nodes
    mp_csv = DATA_DIR / "morph_pattern_nodes.csv"
    with open(mp_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'pattern', 'patternBW', 'label', 'meaningTendency'
        ])
        w.writeheader()
        w.writerows(pattern_nodes)
    print(f"  Wrote {len(pattern_nodes)} morph patterns to {mp_csv.name}")

    # Relationship CSVs
    # word → verse
    wv_csv = DATA_DIR / "word_verse_rels.csv"
    with open(wv_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tokenId', 'verseId'])
        for t in word_tokens:
            w.writerow([t['tokenId'], t['verseId']])
    print(f"  Wrote {len(word_tokens)} word-verse rels to {wv_csv.name}")

    # word → lemma
    wl_csv = DATA_DIR / "word_lemma_rels.csv"
    with open(wl_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tokenId', 'lemma'])
        count = 0
        for t in word_tokens:
            if t['lemma']:
                w.writerow([t['tokenId'], t['lemma']])
                count += 1
    print(f"  Wrote {count} word-lemma rels to {wl_csv.name}")

    # lemma → root
    lr_csv = DATA_DIR / "lemma_root_rels.csv"
    with open(lr_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lemma', 'root'])
        count = 0
        for ln in lemma_nodes:
            if ln['root']:
                w.writerow([ln['lemma'], ln['root']])
                count += 1
    print(f"  Wrote {count} lemma-root rels to {lr_csv.name}")

    # word → pattern
    wp_csv = DATA_DIR / "word_pattern_rels.csv"
    valid_patterns = {pn['pattern'] for pn in pattern_nodes}
    with open(wp_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tokenId', 'pattern'])
        count = 0
        for t in word_tokens:
            if t['wazn'] and t['wazn'] in valid_patterns:
                w.writerow([t['tokenId'], t['wazn']])
                count += 1
    print(f"  Wrote {count} word-pattern rels to {wp_csv.name}")


def print_stats(word_tokens, lemma_nodes, pattern_nodes):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("WORD TOKEN STATISTICS")
    print(f"{'='*60}")
    print(f"  Total word tokens:  {len(word_tokens):,}")
    print(f"  Unique lemmas:      {len(lemma_nodes):,}")
    print(f"  Morph patterns:     {len(pattern_nodes):,}")

    # POS distribution
    pos_counts = Counter(t['pos'] for t in word_tokens)
    print(f"\n  POS distribution:")
    for pos, count in pos_counts.most_common():
        print(f"    {pos:12s} {count:6,}")

    # Root coverage
    with_root = sum(1 for t in word_tokens if t['root'])
    print(f"\n  Tokens with root:   {with_root:,} ({100*with_root/len(word_tokens):.1f}%)")

    # Wazn coverage
    with_wazn = sum(1 for t in word_tokens if t['wazn'])
    print(f"  Tokens with wazn:   {with_wazn:,} ({100*with_wazn/len(word_tokens):.1f}%)")

    # Verse coverage
    verse_ids = set(t['verseId'] for t in word_tokens)
    print(f"  Verses covered:     {len(verse_ids):,}")

    # Unique roots
    roots = set(t['root'] for t in word_tokens if t['root'])
    print(f"  Unique roots:       {len(roots):,}")

    # Sample output
    print(f"\n  Sample tokens (1:1):")
    for t in word_tokens[:10]:
        if t['verseId'] == '1:1':
            print(f"    [{t['tokenId']}] {t['arabicText']:>15}  root={t['root']:<6}  "
                  f"lemma={t['lemma']:<12}  pos={t['pos']:<10}  wazn={t['wazn']}")


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description='Build word token data for Quranic Etymology')
    parser.add_argument('--stats', action='store_true', help='Print statistics only')
    args = parser.parse_args()

    print("Word Token Extraction Pipeline")
    print("=" * 60)

    # Load morphological patterns
    print("\n[1] Loading morphological patterns...")
    with open(PATTERNS_FILE, encoding='utf-8') as f:
        morph_patterns = json.load(f)
    print(f"  Loaded {len(morph_patterns.get('verbal_forms', {}))} verbal forms")
    print(f"  Loaded {len(morph_patterns.get('common_noun_patterns', {}))} common noun patterns")

    # Parse morphology
    print("\n[2] Parsing morphology data...")
    word_tokens, lemma_data = parse_morphology(morph_patterns)
    print(f"  Parsed {len(word_tokens):,} word tokens")
    print(f"  Found {len(lemma_data):,} unique lemmas")

    # Build lemma nodes
    print("\n[3] Building lemma nodes...")
    lemma_nodes = build_lemma_nodes(lemma_data)
    print(f"  Built {len(lemma_nodes):,} lemma nodes")

    # Build pattern nodes
    print("\n[4] Building morphological pattern nodes...")
    pattern_nodes = build_pattern_nodes(morph_patterns)
    print(f"  Built {len(pattern_nodes):,} pattern nodes")

    # Print stats
    print_stats(word_tokens, lemma_nodes, pattern_nodes)

    if not args.stats:
        # Export CSVs
        print(f"\n[5] Exporting CSVs...")
        export_csvs(word_tokens, lemma_nodes, pattern_nodes)

    print("\nDone.")


if __name__ == '__main__':
    main()
