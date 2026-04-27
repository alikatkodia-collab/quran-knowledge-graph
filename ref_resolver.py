"""
ref_resolver.py — Sefaria-inspired Quranic citation resolver.

Given a chunk of English or Arabic text, find every Quranic verse citation
and resolve it to a canonical verseId of the form "S:V" where S is the
surah number (1-114) and V is the verse number.

API:
    >>> from ref_resolver import resolve_refs
    >>> matches = resolve_refs("See Quran 2:255 and Surah Al-Fatihah verse 1.")
    >>> [m.canonical for m in matches]
    ['2:255', '1:1']

Supported patterns (high-precision regex):
    [2:255]                              bracket form (most common in our docs)
    (2:255)                              parenthetic form
    Quran 2:255  /  Qur'an 2:255         explicit "Quran"
    Surah X verse Y  /  Sūrah X v. Y     spelled out
    Surah Al-Baqarah verse 255           spelled-out name + verse num
    Surah 2 ayah 255                     "ayah" alternative
    Q 2:255  /  Q. 2:255                 abbreviation
    سورة البقرة آية ٢٥٥                  Arabic spelled-out
    آل عمران 3                            short Arabic ref
    [2:255-258]                          range (returns multiple matches)
    [2:255, 2:286]                       list (multiple matches)

Famous-verse aliases (also recognized):
    Ayat al-Kursi      -> 2:255
    Al-Fatiha / Fatihah -> 1
    Surah Yasin / Ya-Sin -> 36
    etc. (loaded from SURAH_NAME_TO_NUM)

Returns Match dataclass instances with start/end offsets, raw matched
text, the resolved verseId, and a confidence score.
"""

from dataclasses import dataclass
import re
from typing import Iterable

# ── Surah name → number mapping ─────────────────────────────────────────────
# Multiple transliterations per surah. Stored lowercase; we lowercase input
# during matching. Only includes names a typical writer would use.
SURAH_NAMES: list[tuple[int, list[str]]] = [
    (1,  ["al-fatihah", "al-fatiha", "fatihah", "fatiha", "al-faatihah", "the opening", "the opener"]),
    (2,  ["al-baqarah", "al-baqara", "baqarah", "baqara", "the cow"]),
    (3,  ["aal imran", "aal-imran", "ali imran", "ali-imran", "al imran", "imran", "the family of imran"]),
    (4,  ["an-nisa", "an-nisaa", "an-nisaa'", "nisa", "the women"]),
    (5,  ["al-maidah", "al-ma'idah", "al-maaidah", "maidah", "the table", "the feast"]),
    (6,  ["al-an'am", "al-anam", "an'am", "the cattle"]),
    (7,  ["al-a'raf", "al-araf", "a'raf", "araf", "the heights"]),
    (8,  ["al-anfal", "anfal", "the spoils"]),
    (9,  ["at-tawbah", "at-tauba", "at-taubah", "tawbah", "tauba", "bara'ah", "the repentance"]),
    (10, ["yunus", "jonah"]),
    (11, ["hud"]),
    (12, ["yusuf", "joseph"]),
    (13, ["ar-ra'd", "ar-rad", "the thunder"]),
    (14, ["ibrahim", "abraham"]),
    (15, ["al-hijr", "the rocky tract"]),
    (16, ["an-nahl", "the bee"]),
    (17, ["al-isra", "al-israa", "the night journey", "the children of israel", "bani israeel"]),
    (18, ["al-kahf", "kahf", "the cave"]),
    (19, ["maryam", "mary"]),
    (20, ["ta-ha", "taha", "ta ha"]),
    (21, ["al-anbiya", "al-anbiyaa", "the prophets"]),
    (22, ["al-hajj", "hajj", "the pilgrimage"]),
    (23, ["al-mu'minun", "al-muminun", "the believers"]),
    (24, ["an-nur", "nur", "the light"]),
    (25, ["al-furqan", "furqan", "the criterion"]),
    (26, ["ash-shu'ara", "ash-shuara", "the poets"]),
    (27, ["an-naml", "naml", "the ant"]),
    (28, ["al-qasas", "qasas", "the narration", "the stories"]),
    (29, ["al-ankabut", "ankabut", "the spider"]),
    (30, ["ar-rum", "rum", "the romans", "the byzantines"]),
    (31, ["luqman"]),
    (32, ["as-sajdah", "sajdah", "the prostration"]),
    (33, ["al-ahzab", "ahzab", "the parties", "the confederates"]),
    (34, ["saba", "saba'", "sheba"]),
    (35, ["fatir", "the originator"]),
    (36, ["ya-sin", "yasin", "ya sin", "ya-seen", "yaseen"]),
    (37, ["as-saffat", "saffat", "those who set the ranks"]),
    (38, ["sad", "saad"]),
    (39, ["az-zumar", "zumar", "the throngs", "the groups"]),
    (40, ["al-ghafir", "ghafir", "al-mu'min", "the forgiver"]),
    (41, ["fussilat", "ha mim sajdah", "explained in detail"]),
    (42, ["ash-shura", "shura", "the consultation"]),
    (43, ["az-zukhruf", "zukhruf", "the gold ornaments"]),
    (44, ["ad-dukhan", "dukhan", "the smoke"]),
    (45, ["al-jathiyah", "jathiyah", "the kneeling"]),
    (46, ["al-ahqaf", "ahqaf", "the sand dunes"]),
    (47, ["muhammad"]),
    (48, ["al-fath", "fath", "the victory"]),
    (49, ["al-hujurat", "hujurat", "the chambers"]),
    (50, ["qaf"]),
    (51, ["adh-dhariyat", "dhariyat", "the scattering winds"]),
    (52, ["at-tur", "tur", "the mount"]),
    (53, ["an-najm", "najm", "the star"]),
    (54, ["al-qamar", "qamar", "the moon"]),
    (55, ["ar-rahman", "rahman", "the most gracious"]),
    (56, ["al-waqi'ah", "al-waqiah", "waqi'ah", "the inevitable", "the event"]),
    (57, ["al-hadid", "hadid", "the iron"]),
    (58, ["al-mujadilah", "al-mujadalah", "the dispute"]),
    (59, ["al-hashr", "hashr", "the gathering"]),
    (60, ["al-mumtahanah", "mumtahanah", "the examined one"]),
    (61, ["as-saff", "saff", "the ranks"]),
    (62, ["al-jumu'ah", "al-jumuah", "jumu'ah", "friday"]),
    (63, ["al-munafiqun", "munafiqun", "the hypocrites"]),
    (64, ["at-taghabun", "taghabun", "mutual disillusion"]),
    (65, ["at-talaq", "talaq", "the divorce"]),
    (66, ["at-tahrim", "tahrim", "the prohibition"]),
    (67, ["al-mulk", "mulk", "the dominion", "the kingdom"]),
    (68, ["al-qalam", "qalam", "the pen", "noon"]),
    (69, ["al-haqqah", "haqqah", "the inevitable reality"]),
    (70, ["al-ma'arij", "al-maarij", "ma'arij", "the ascending stairways"]),
    (71, ["nuh", "noah"]),
    (72, ["al-jinn", "jinn"]),
    (73, ["al-muzzammil", "muzzammil", "the enshrouded one"]),
    (74, ["al-muddaththir", "al-muddaththir", "muddaththir", "the cloaked one"]),
    (75, ["al-qiyamah", "qiyamah", "the resurrection"]),
    (76, ["al-insan", "insan", "the man", "ad-dahr"]),
    (77, ["al-mursalat", "mursalat", "those sent forth"]),
    (78, ["an-naba", "naba", "the great news"]),
    (79, ["an-nazi'at", "an-naziat", "those who pull out"]),
    (80, ["abasa", "he frowned"]),
    (81, ["at-takwir", "takwir", "the rolling up"]),
    (82, ["al-infitar", "infitar", "the cleaving"]),
    (83, ["al-mutaffifin", "mutaffifin", "the defrauders"]),
    (84, ["al-inshiqaq", "inshiqaq", "the splitting open"]),
    (85, ["al-buruj", "buruj", "the constellations"]),
    (86, ["at-tariq", "tariq", "the night-comer"]),
    (87, ["al-a'la", "al-ala", "a'la", "the most high"]),
    (88, ["al-ghashiyah", "ghashiyah", "the overwhelming"]),
    (89, ["al-fajr", "fajr", "the dawn"]),
    (90, ["al-balad", "balad", "the city"]),
    (91, ["ash-shams", "shams", "the sun"]),
    (92, ["al-layl", "al-lail", "layl", "the night"]),
    (93, ["ad-duha", "duha", "the morning hours"]),
    (94, ["ash-sharh", "al-inshirah", "inshirah", "sharh", "the relief"]),
    (95, ["at-tin", "tin", "the fig"]),
    (96, ["al-alaq", "al-'alaq", "alaq", "the clinging substance", "the clot"]),
    (97, ["al-qadr", "qadr", "the night of decree", "the decree"]),
    (98, ["al-bayyinah", "bayyinah", "the clear evidence"]),
    (99, ["az-zalzalah", "zalzalah", "az-zilzal", "zilzal", "the earthquake"]),
    (100,["al-adiyat", "adiyat", "the courser"]),
    (101,["al-qari'ah", "al-qariah", "qari'ah", "the calamity"]),
    (102,["at-takathur", "takathur", "the rivalry"]),
    (103,["al-asr", "asr", "the time", "the declining day"]),
    (104,["al-humazah", "humazah", "the slanderer"]),
    (105,["al-fil", "fil", "the elephant"]),
    (106,["quraysh", "quraish"]),
    (107,["al-ma'un", "al-maun", "ma'un", "small kindnesses"]),
    (108,["al-kawthar", "kawthar", "abundance"]),
    (109,["al-kafirun", "kafirun", "the disbelievers"]),
    (110,["an-nasr", "nasr", "the help", "the divine aid"]),
    (111,["al-masad", "masad", "al-lahab", "lahab", "the palm fiber"]),
    (112,["al-ikhlas", "ikhlas", "the sincerity", "the purity"]),
    (113,["al-falaq", "falaq", "the daybreak"]),
    (114,["an-nas", "nas", "mankind"]),
]

# Build lookup table
NAME_TO_NUM: dict[str, int] = {}
for num, names in SURAH_NAMES:
    for name in names:
        NAME_TO_NUM[name.lower()] = num

# Famous named verses
NAMED_VERSES: dict[str, str] = {
    "ayat al-kursi":   "2:255",
    "ayat ul-kursi":   "2:255",
    "ayatul kursi":    "2:255",
    "throne verse":    "2:255",
    "the throne verse": "2:255",
    "verse of light":  "24:35",
    "the light verse": "24:35",
    "verse of debt":   "2:282",
    "ayat ad-dayn":    "2:282",
    "the longest verse": "2:282",
    "ayat al-mubahalah": "3:61",
    "verse of mubahalah": "3:61",
    "ayat at-takwur":  "81:1",
}


@dataclass
class Match:
    """A resolved Quran citation."""
    start: int        # char offset in the source
    end: int          # exclusive end offset
    raw: str          # the original matched substring
    canonical: str    # e.g. "2:255" — single verse only; ranges expanded
    confidence: float # 0.0 - 1.0
    kind: str         # "bracket" | "explicit" | "named-surah" | "named-verse" | "arabic" | "range"


# ── regex patterns ──────────────────────────────────────────────────────────

# [2:255] [2:255-258] [2:255,2:286]  — most common form in our project
_RE_BRACKET_LIST = re.compile(
    r"\[\s*(\d{1,3}\s*:\s*\d{1,3}(?:\s*[-–]\s*\d{1,3})?(?:\s*,\s*\d{1,3}\s*:\s*\d{1,3}(?:\s*[-–]\s*\d{1,3})?)*)\s*\]"
)

# (2:255) — same idea, parens
_RE_PAREN = re.compile(r"\(\s*(\d{1,3}\s*:\s*\d{1,3}(?:\s*[-–]\s*\d{1,3})?)\s*\)")

# Quran 2:255  /  Qur'an 2:255  /  Q 2:255  /  Q. 2:255
_RE_EXPLICIT = re.compile(
    r"(?:Qur['ʼ’]?an|Quran|Q\.?)\s+(\d{1,3})\s*:\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?",
    re.IGNORECASE,
)

# Surah X verse Y  /  Surah X ayah Y  /  Sura X v. Y  /  Sūrah X:Y
_RE_SPELLED_NUM = re.compile(
    r"(?:Surah|Sūrah|Sura|Sūra|Chapter)\s+(\d{1,3})\s*(?:[,:]?\s*(?:verse|ayah|aya|v\.?)\s+|[\s:])\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?",
    re.IGNORECASE,
)

# Surah Al-Baqarah verse 255  /  Surat ar-Rahman 13
# Anchored on the "Surah/Sura/Surat/Chapter" prefix so we don't accidentally
# slurp surrounding sentence words like "In " or "From ".
_RE_NAMED_SURAH = re.compile(
    r"(?:Surah|Sūrah|Sura|Sūra|Surat|Chapter)\s+([A-Za-z][A-Za-z'\u2018\u2019\u00E0-\u00FF\- ]{2,40}?)"
    r"\s*(?:[,:]?\s*(?:verse|ayah|aya|v\.?)\s+|[\s:])\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?",
    re.IGNORECASE,
)

# Bare named-surah form: "Al-Fatihah 1", "Ar-Rahman 13" — no prefix.
# Restricted to names that START with Al-/Ad-/An-/At-/As-/Ash-/Az- to avoid
# false positives on random prose.
_RE_BARE_NAMED_SURAH = re.compile(
    r"\b((?:Al|Ad|An|Ar|As|Ash|At|Az)[-\u2010\u2011]"
    r"[A-Za-z'\u2018\u2019\u00E0-\u00FF\- ]{2,30}?)"
    r"\s+(?:verse|ayah|aya|v\.?\s+)?(\d{1,3})(?:\s*[-\u2013]\s*(\d{1,3}))?",
    re.IGNORECASE,
)

# Bare X:Y when surrounded by whitespace or punctuation — risk of false positives
# so we only fire this in contexts where Quran is hinted at (called from a flag)
_RE_BARE_PAIR = re.compile(r"(?<![:\d])(\d{1,3})\s*:\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?(?![:\d])")

# Arabic "سورة <name> آية <n>" — basic
_RE_AR_SURAH_AYAH = re.compile(
    r"سورة\s+([\u0621-\u064A\s]+?)\s+(?:آية|الآية)\s+(\d+|[٠-٩]+)"
)


def _expand_range(s: int, v_start: int, v_end: int | None) -> list[str]:
    if v_end is None or v_end < v_start:
        return [f"{s}:{v_start}"]
    return [f"{s}:{v}" for v in range(v_start, v_end + 1)]


def _arabic_to_western_digits(s: str) -> str:
    table = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return s.translate(table)


def _resolve_named_surah(raw_name: str) -> int | None:
    """Try to match a surah name (handles 'Al-' prefix, hyphenation, etc.)"""
    name = raw_name.strip().lower()
    name = re.sub(r"\s{2,}", " ", name)
    if name in NAME_TO_NUM:
        return NAME_TO_NUM[name]
    # Try without leading "al-" / "ad-" / "an-" / etc.
    no_prefix = re.sub(r"^(?:al|ad|an|ar|as|ash|at|az|aṣ)[ -]", "", name)
    if no_prefix in NAME_TO_NUM:
        return NAME_TO_NUM[no_prefix]
    # Try collapsed form (no hyphens / spaces)
    collapsed = name.replace(" ", "").replace("-", "").replace("'", "").replace("’", "")
    for n, num in NAME_TO_NUM.items():
        if n.replace(" ", "").replace("-", "").replace("'", "").replace("’", "") == collapsed:
            return num
    return None


def _valid_surah(s: int) -> bool:
    return 1 <= s <= 114

def _valid_verse(s: int, v: int) -> bool:
    # The longest surahs have ~286 verses (2:286). 286 is a generous upper bound.
    return _valid_surah(s) and 1 <= v <= 286


def resolve_refs(text: str) -> list[Match]:
    """
    Find every Quranic citation in `text` and return a list of Match objects.
    Ranges are expanded into one Match per contained verse; spans share the
    same `start`/`end` of the surrounding raw match.
    """
    matches: list[Match] = []
    seen_offsets: set[tuple[int, int]] = set()

    def _emit(start: int, end: int, raw: str, refs: Iterable[str], conf: float, kind: str):
        if (start, end) in seen_offsets:
            return
        seen_offsets.add((start, end))
        for ref in refs:
            matches.append(Match(start=start, end=end, raw=raw, canonical=ref,
                                 confidence=conf, kind=kind))

    # 1. Named verses (highest precision)
    lower = text.lower()
    for name, ref in NAMED_VERSES.items():
        idx = 0
        while True:
            i = lower.find(name, idx)
            if i < 0:
                break
            _emit(i, i + len(name), text[i:i + len(name)], [ref], 0.99, "named-verse")
            idx = i + len(name)

    # 2. Bracket form (and ranges/lists inside)
    for m in _RE_BRACKET_LIST.finditer(text):
        chunk = m.group(1)
        refs = []
        for piece in chunk.split(","):
            piece = piece.strip()
            mm = re.match(r"(\d{1,3})\s*:\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?", piece)
            if mm:
                s = int(mm.group(1)); v1 = int(mm.group(2))
                v2 = int(mm.group(3)) if mm.group(3) else None
                if _valid_surah(s):
                    refs.extend(_expand_range(s, v1, v2))
        if refs:
            _emit(m.start(), m.end(), m.group(0), refs, 0.99, "bracket")

    # 3. Parenthetic
    for m in _RE_PAREN.finditer(text):
        mm = re.match(r"(\d{1,3})\s*:\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?", m.group(1))
        if mm:
            s = int(mm.group(1)); v1 = int(mm.group(2))
            v2 = int(mm.group(3)) if mm.group(3) else None
            if _valid_surah(s):
                _emit(m.start(), m.end(), m.group(0), _expand_range(s, v1, v2), 0.95, "bracket")

    # 4. Quran 2:255 / Qur'an X:Y / Q. X:Y
    for m in _RE_EXPLICIT.finditer(text):
        s = int(m.group(1)); v1 = int(m.group(2))
        v2 = int(m.group(3)) if m.group(3) else None
        if _valid_surah(s) and _valid_verse(s, v1):
            _emit(m.start(), m.end(), m.group(0), _expand_range(s, v1, v2), 0.99, "explicit")

    # 5. Surah <num> verse <num>
    for m in _RE_SPELLED_NUM.finditer(text):
        s = int(m.group(1)); v1 = int(m.group(2))
        v2 = int(m.group(3)) if m.group(3) else None
        if _valid_surah(s) and _valid_verse(s, v1):
            _emit(m.start(), m.end(), m.group(0), _expand_range(s, v1, v2), 0.95, "explicit")

    # 6. Surah Al-Baqarah verse 255 (with "Surah" prefix)
    for m in _RE_NAMED_SURAH.finditer(text):
        if any(start <= m.start() < end for start, end in seen_offsets):
            continue
        name = m.group(1)
        s = _resolve_named_surah(name)
        if s is None:
            continue
        v1 = int(m.group(2))
        v2 = int(m.group(3)) if m.group(3) else None
        if _valid_verse(s, v1):
            _emit(m.start(), m.end(), m.group(0), _expand_range(s, v1, v2), 0.92, "named-surah")

    # 7. Bare named-surah ("Al-Fatihah 1") — only with Al-/Ad-/An- prefix
    for m in _RE_BARE_NAMED_SURAH.finditer(text):
        if any(start <= m.start() < end for start, end in seen_offsets):
            continue
        name = m.group(1)
        s = _resolve_named_surah(name)
        if s is None:
            continue
        v1 = int(m.group(2))
        v2 = int(m.group(3)) if m.group(3) else None
        if _valid_verse(s, v1):
            _emit(m.start(), m.end(), m.group(0), _expand_range(s, v1, v2), 0.78, "named-surah")

    # 7. Arabic "سورة <name> آية <n>"
    for m in _RE_AR_SURAH_AYAH.finditer(text):
        ar_name = m.group(1).strip()
        # Try common Arabic name → English transliteration mapping
        # (very minimal — extend later)
        ar_to_num = {
            "الفاتحة": 1, "البقرة": 2, "آل عمران": 3, "النساء": 4, "المائدة": 5,
            "الأنعام": 6, "الأعراف": 7, "الأنفال": 8, "التوبة": 9, "يونس": 10,
            "هود": 11, "يوسف": 12, "الرعد": 13, "إبراهيم": 14, "الحجر": 15,
            "النحل": 16, "الإسراء": 17, "الكهف": 18, "مريم": 19, "طه": 20,
            "الأنبياء": 21, "الحج": 22, "المؤمنون": 23, "النور": 24, "الفرقان": 25,
            "يس": 36, "الرحمن": 55, "الواقعة": 56, "الملك": 67, "نوح": 71,
            "الجن": 72, "القلم": 68, "الإخلاص": 112, "الفلق": 113, "الناس": 114,
        }
        s = ar_to_num.get(ar_name)
        if s is None:
            continue
        v_str = _arabic_to_western_digits(m.group(2))
        try:
            v1 = int(v_str)
        except ValueError:
            continue
        if _valid_verse(s, v1):
            _emit(m.start(), m.end(), m.group(0), [f"{s}:{v1}"], 0.92, "arabic")

    # Sort by source offset
    matches.sort(key=lambda x: (x.start, x.canonical))
    return matches


def link_html(text: str, base_url: str = "/verse",
              css_class: str = "quran-ref") -> str:
    """
    Return `text` with detected refs wrapped in <a> tags.
    Use this for the JS-widget fallback or server-side rendering.
    """
    matches = resolve_refs(text)
    # Group by span
    by_span = {}
    for m in matches:
        by_span.setdefault((m.start, m.end), []).append(m)

    out = []
    cursor = 0
    for (start, end), ms in sorted(by_span.items()):
        if start < cursor:
            continue  # overlapping; skip
        out.append(text[cursor:start])
        # Use first canonical ref
        ref = ms[0].canonical
        raw = text[start:end]
        out.append(f'<a class="{css_class}" data-verse="{ref}" href="{base_url}/{ref}">{raw}</a>')
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    samples = [
        "See [2:255] for the throne verse.",
        "The Quran says (1:1) is the opener.",
        "In Surah Al-Baqarah verse 286, God says...",
        "Quran 24:35 is known as the Light verse.",
        "Q. 36:1 begins Ya-Sin.",
        "Ayat al-Kursi (2:255) is widely recited.",
        "From Surah Yasin verse 1 onward...",
        "Refs: [2:255-258], [3:1, 3:2], plus Q 17:23.",
        "From Al-Fatihah 1 we begin.",
        "Surah Ar-Rahman verse 13 repeats the refrain.",
        "في سورة البقرة آية 255 يقول الله",
    ]
    for s in samples:
        ms = resolve_refs(s)
        print(f"INPUT: {s}")
        if not ms:
            print("  (no matches)")
        for m in ms:
            print(f"  -> [{m.start}:{m.end}] {m.canonical}  ({m.kind}, conf={m.confidence})  raw={m.raw}")
        print()
