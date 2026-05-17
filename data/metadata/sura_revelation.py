"""
Meccan / Medinan classification and chronological revelation order for all
114 surahs, sourced from the standard Egyptian (Cairo / King Fuad, 1924)
edition of the Mushaf — the same reference that Khalifa's translation and
Tanzil.net mirror.

Totals:
  - 86 Meccan
  - 28 Medinan
  - revelation_order spans 1..114 with no duplicates

A handful of surahs (13 Ar-Ra'd, 22 Al-Hajj, 55 Ar-Rahman, 76 Al-Insan,
99 Az-Zalzalah) are disputed in the classical literature — different
classifications place them on opposite sides of the hijrah. This file
follows the Egyptian standard exclusively; alternatives are noted in
CLAUDE.md.

DATA IS BAKED IN INTENTIONALLY — do not fetch over the network. The table
is auditable, stable, and small enough to review by hand.
"""

from __future__ import annotations

SOURCE = "egyptian_standard_2026"

# surah_number -> {"location": "Meccan" | "Medinan", "order": int}
#
# `order` is the chronological position per the Egyptian standard
# (1 = first revealed; 114 = last revealed). Orders 1..86 are all Meccan;
# orders 87..114 are all Medinan in this classification.
SURA_REVELATION: dict[int, dict[str, object]] = {
    1:   {"location": "Meccan",  "order": 5},    # Al-Fatihah
    2:   {"location": "Medinan", "order": 87},   # Al-Baqarah
    3:   {"location": "Medinan", "order": 89},   # Aal-i-Imran
    4:   {"location": "Medinan", "order": 92},   # An-Nisa
    5:   {"location": "Medinan", "order": 112},  # Al-Ma'idah
    6:   {"location": "Meccan",  "order": 55},   # Al-An'am
    7:   {"location": "Meccan",  "order": 39},   # Al-A'raf
    8:   {"location": "Medinan", "order": 88},   # Al-Anfal
    9:   {"location": "Medinan", "order": 113},  # At-Tawbah
    10:  {"location": "Meccan",  "order": 51},   # Yunus
    11:  {"location": "Meccan",  "order": 52},   # Hud
    12:  {"location": "Meccan",  "order": 53},   # Yusuf
    13:  {"location": "Medinan", "order": 96},   # Ar-Ra'd (disputed)
    14:  {"location": "Meccan",  "order": 72},   # Ibrahim
    15:  {"location": "Meccan",  "order": 54},   # Al-Hijr
    16:  {"location": "Meccan",  "order": 70},   # An-Nahl
    17:  {"location": "Meccan",  "order": 50},   # Al-Isra
    18:  {"location": "Meccan",  "order": 69},   # Al-Kahf
    19:  {"location": "Meccan",  "order": 44},   # Maryam
    20:  {"location": "Meccan",  "order": 45},   # Ta-Ha
    21:  {"location": "Meccan",  "order": 73},   # Al-Anbiya
    22:  {"location": "Medinan", "order": 103},  # Al-Hajj (disputed)
    23:  {"location": "Meccan",  "order": 74},   # Al-Mu'minun
    24:  {"location": "Medinan", "order": 102},  # An-Nur
    25:  {"location": "Meccan",  "order": 42},   # Al-Furqan
    26:  {"location": "Meccan",  "order": 47},   # Ash-Shu'ara
    27:  {"location": "Meccan",  "order": 48},   # An-Naml
    28:  {"location": "Meccan",  "order": 49},   # Al-Qasas
    29:  {"location": "Meccan",  "order": 85},   # Al-Ankabut
    30:  {"location": "Meccan",  "order": 84},   # Ar-Rum
    31:  {"location": "Meccan",  "order": 57},   # Luqman
    32:  {"location": "Meccan",  "order": 75},   # As-Sajdah
    33:  {"location": "Medinan", "order": 90},   # Al-Ahzab
    34:  {"location": "Meccan",  "order": 58},   # Saba
    35:  {"location": "Meccan",  "order": 43},   # Fatir
    36:  {"location": "Meccan",  "order": 41},   # Ya-Sin
    37:  {"location": "Meccan",  "order": 56},   # As-Saffat
    38:  {"location": "Meccan",  "order": 38},   # Sad
    39:  {"location": "Meccan",  "order": 59},   # Az-Zumar
    40:  {"location": "Meccan",  "order": 60},   # Ghafir
    41:  {"location": "Meccan",  "order": 61},   # Fussilat
    42:  {"location": "Meccan",  "order": 62},   # Ash-Shura
    43:  {"location": "Meccan",  "order": 63},   # Az-Zukhruf
    44:  {"location": "Meccan",  "order": 64},   # Ad-Dukhan
    45:  {"location": "Meccan",  "order": 65},   # Al-Jathiyah
    46:  {"location": "Meccan",  "order": 66},   # Al-Ahqaf
    47:  {"location": "Medinan", "order": 95},   # Muhammad
    48:  {"location": "Medinan", "order": 111},  # Al-Fath
    49:  {"location": "Medinan", "order": 106},  # Al-Hujurat
    50:  {"location": "Meccan",  "order": 34},   # Qaf
    51:  {"location": "Meccan",  "order": 67},   # Adh-Dhariyat
    52:  {"location": "Meccan",  "order": 76},   # At-Tur
    53:  {"location": "Meccan",  "order": 23},   # An-Najm
    54:  {"location": "Meccan",  "order": 37},   # Al-Qamar
    55:  {"location": "Medinan", "order": 97},   # Ar-Rahman (disputed)
    56:  {"location": "Meccan",  "order": 46},   # Al-Waqi'ah
    57:  {"location": "Medinan", "order": 94},   # Al-Hadid
    58:  {"location": "Medinan", "order": 105},  # Al-Mujadila
    59:  {"location": "Medinan", "order": 101},  # Al-Hashr
    60:  {"location": "Medinan", "order": 91},   # Al-Mumtahanah
    61:  {"location": "Medinan", "order": 109},  # As-Saff
    62:  {"location": "Medinan", "order": 110},  # Al-Jumu'ah
    63:  {"location": "Medinan", "order": 104},  # Al-Munafiqun
    64:  {"location": "Medinan", "order": 108},  # At-Taghabun
    65:  {"location": "Medinan", "order": 99},   # At-Talaq
    66:  {"location": "Medinan", "order": 107},  # At-Tahrim
    67:  {"location": "Meccan",  "order": 77},   # Al-Mulk
    68:  {"location": "Meccan",  "order": 2},    # Al-Qalam
    69:  {"location": "Meccan",  "order": 78},   # Al-Haqqah
    70:  {"location": "Meccan",  "order": 79},   # Al-Ma'arij
    71:  {"location": "Meccan",  "order": 71},   # Nuh
    72:  {"location": "Meccan",  "order": 40},   # Al-Jinn
    73:  {"location": "Meccan",  "order": 3},    # Al-Muzzammil
    74:  {"location": "Meccan",  "order": 4},    # Al-Muddathir
    75:  {"location": "Meccan",  "order": 31},   # Al-Qiyamah
    76:  {"location": "Medinan", "order": 98},   # Al-Insan (disputed)
    77:  {"location": "Meccan",  "order": 33},   # Al-Mursalat
    78:  {"location": "Meccan",  "order": 80},   # An-Naba
    79:  {"location": "Meccan",  "order": 81},   # An-Nazi'at
    80:  {"location": "Meccan",  "order": 24},   # Abasa
    81:  {"location": "Meccan",  "order": 7},    # At-Takwir
    82:  {"location": "Meccan",  "order": 82},   # Al-Infitar
    83:  {"location": "Meccan",  "order": 86},   # Al-Mutaffifin
    84:  {"location": "Meccan",  "order": 83},   # Al-Inshiqaq
    85:  {"location": "Meccan",  "order": 27},   # Al-Buruj
    86:  {"location": "Meccan",  "order": 36},   # At-Tariq
    87:  {"location": "Meccan",  "order": 8},    # Al-A'la
    88:  {"location": "Meccan",  "order": 68},   # Al-Ghashiyah
    89:  {"location": "Meccan",  "order": 10},   # Al-Fajr
    90:  {"location": "Meccan",  "order": 35},   # Al-Balad
    91:  {"location": "Meccan",  "order": 26},   # Ash-Shams
    92:  {"location": "Meccan",  "order": 9},    # Al-Layl
    93:  {"location": "Meccan",  "order": 11},   # Ad-Duha
    94:  {"location": "Meccan",  "order": 12},   # Ash-Sharh
    95:  {"location": "Meccan",  "order": 28},   # At-Tin
    96:  {"location": "Meccan",  "order": 1},    # Al-Alaq
    97:  {"location": "Meccan",  "order": 25},   # Al-Qadr
    98:  {"location": "Medinan", "order": 100},  # Al-Bayyinah
    99:  {"location": "Medinan", "order": 93},   # Az-Zalzalah (disputed)
    100: {"location": "Meccan",  "order": 14},   # Al-Adiyat
    101: {"location": "Meccan",  "order": 30},   # Al-Qari'ah
    102: {"location": "Meccan",  "order": 16},   # At-Takathur
    103: {"location": "Meccan",  "order": 13},   # Al-Asr
    104: {"location": "Meccan",  "order": 32},   # Al-Humazah
    105: {"location": "Meccan",  "order": 19},   # Al-Fil
    106: {"location": "Meccan",  "order": 29},   # Quraysh
    107: {"location": "Meccan",  "order": 17},   # Al-Ma'un
    108: {"location": "Meccan",  "order": 15},   # Al-Kawthar
    109: {"location": "Meccan",  "order": 18},   # Al-Kafirun
    110: {"location": "Medinan", "order": 114},  # An-Nasr
    111: {"location": "Meccan",  "order": 6},    # Al-Masad
    112: {"location": "Meccan",  "order": 22},   # Al-Ikhlas
    113: {"location": "Meccan",  "order": 20},   # Al-Falaq
    114: {"location": "Meccan",  "order": 21},   # An-Nas
}
