"""
Phase 9 — fourth wave of gap-filling questions.

Targets areas still thin:
  - Surah openings (Huroof Muqatta'at / mysterious letters) — each distinct (~20)
  - Character studies of specific prophets in detail (~30)
  - Verse-pairs that echo or contrast (~25)
  - Specific parables and their interpretation (~20)
  - Divine oaths and what they swear by (~15)
  - Women mentioned by description in the Quran (~15)
  - Geography of the Quran (~15)
  - Numbers with Quranic significance (~15)
  - Acts of worship in specific situations (~15)
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
from pathlib import Path
import overnight_seed as engine


PHASE9_RAW = [
    # ── Surah openings / mysterious letters (~20)
    "What does the Quran say about the Huroof Muqatta'at (mysterious opening letters)?",
    "What is the significance of 'Alif Lam Meem' opening several surahs?",
    "What is the significance of 'Alif Lam Ra' in its surah openings?",
    "What is the significance of 'Alif Lam Meem Ra' (Surah 13)?",
    "What is the significance of 'Kaf Ha Ya Ain Sad' (Surah 19)?",
    "What is the significance of 'Ta Ha' at the opening of Surah 20?",
    "What is the significance of 'Ta Sin Meem' openings (Surahs 26 and 28)?",
    "What is the significance of 'Ta Sin' at the opening of Surah 27?",
    "What is the significance of 'Ya Sin' at the opening of Surah 36?",
    "What is the significance of 'Sad' at the opening of Surah 38?",
    "What is the significance of 'Ha Meem' openings (Surahs 40-46)?",
    "What is the significance of 'Ain Sin Qaf' in Surah 42?",
    "What is the significance of 'Qaf' at the opening of Surah 50?",
    "What is the significance of 'Noon' at the opening of Surah 68?",
    "What does Rashad Khalifa's work reveal about the mathematical roles of these letters?",
    "How do the Huroof Muqatta'at connect the mathematical miracle of 19?",
    "What surahs start with praise (alhamdulillah)?",
    "What surahs start with oaths (e.g., By the...)?",
    "What surahs start with direct command ('O prophet', 'Say', 'Read')?",
    "What surahs start with 'When' (idha) describing apocalyptic events?",

    # ── Character studies in depth (~30)
    "What does the Quran teach in detail about Prophet Noah's preaching and ark?",
    "What does the Quran teach in detail about Prophet Abraham's early life and migration?",
    "What does the Quran teach in detail about Prophet Abraham's sacrifice of his son?",
    "What does the Quran teach in detail about Prophet Moses's upbringing in Pharaoh's palace?",
    "What does the Quran teach in detail about Prophet Moses and the magicians?",
    "What does the Quran teach in detail about Prophet Moses's journey with Al-Khidr?",
    "What does the Quran teach in detail about Prophet Joseph's dream and his brothers?",
    "What does the Quran teach in detail about Prophet Joseph's trials in Egypt?",
    "What does the Quran teach in detail about Prophet Joseph's rise to power?",
    "What does the Quran teach in detail about Prophet Solomon's kingdom?",
    "What does the Quran teach in detail about Prophet David's psalms and wisdom?",
    "What does the Quran teach in detail about Prophet John (Yahya) and his early wisdom?",
    "What does the Quran teach in detail about Prophet Jesus's miraculous birth?",
    "What does the Quran teach in detail about Prophet Jesus's miracles?",
    "What does the Quran teach in detail about Prophet Jesus's raising to God?",
    "What does the Quran teach in detail about Prophet Jonah and the whale?",
    "What does the Quran teach in detail about Prophet Jonah's repentance?",
    "What does the Quran teach in detail about Prophet Lot and his people?",
    "What does the Quran teach in detail about Prophet Lot's wife's fate?",
    "What does the Quran teach in detail about Prophet Hud and the Ad people?",
    "What does the Quran teach in detail about Prophet Saleh and the she-camel?",
    "What does the Quran teach in detail about Prophet Shu'ayb and just weights?",
    "What does the Quran teach in detail about Prophet Zakaria's prayer for offspring?",
    "What does the Quran teach in detail about Prophet Job's patience in illness?",
    "What does the Quran teach in detail about Prophet Muhammad's early hardships?",
    "What does the Quran teach in detail about Prophet Muhammad's migration (Hijrah)?",
    "What does the Quran teach in detail about Prophet Muhammad's conquests?",
    "What does the Quran teach in detail about Prophet Muhammad's family life?",
    "What does the Quran teach in detail about Prophet Muhammad's farewell message?",
    "What does the Quran teach in detail about Rashad Khalifa as Messenger of the Covenant?",

    # ── Verse-pairs that echo or contrast (~25)
    "How do 1:6-7 and 2:6-7 contrast the guided and the rejected?",
    "How does 2:62 compare to 5:69 in describing reward for the righteous?",
    "How do 2:30 and 2:34 frame Adam's creation and the angels' prostration?",
    "How do 2:255 and 3:2 converge on God's attributes of Life and Sustenance?",
    "How do 3:190 and 39:9 parallel 'those who reflect' and 'those with understanding'?",
    "How do 4:36 and 17:23 both command rights to parents and others?",
    "How do 6:38 and 10:61 speak of God knowing all things?",
    "How do 7:172 and 30:30 both appeal to the primordial human nature?",
    "How do 9:36 and 2:189 discuss the calendar and sacred months?",
    "How do 11:114 and 29:45 connect prayer to the eradication of evil?",
    "How do 13:11 and 8:53 phrase the principle 'God changes people when they change themselves'?",
    "How do 14:24-26 and 14:32-33 contrast good word/tree and evil word/tree?",
    "How do 16:90 and 49:11 establish ethical commands about speech and action?",
    "How do 17:82 and 41:44 describe the Quran as healing and mercy?",
    "How do 17:84 and 7:99 discuss each person's path?",
    "How do 20:55 and 71:17-18 describe creation and return to earth?",
    "How do 21:30 and 41:11 both describe the origin of heavens and earth?",
    "How do 23:1-11 and 25:63-76 list qualities of successful/righteous believers?",
    "How do 25:63 and 3:134 both define the servants of the Gracious?",
    "How do 29:64 and 47:36 contrast this life with the Hereafter?",
    "How do 33:40 and 61:6 address prophethood's seal and succession?",
    "How do 39:53 and 12:87 offer hope against despair of God's mercy?",
    "How do 40:60 and 2:186 both promise God answers when called?",
    "How do 50:16 and 2:115 describe God's closeness?",
    "How do 103:1-3 and 23:1-11 list conditions for success?",

    # ── Parables and interpretation (~20)
    "Explain the parable of the garden in flames (68:17-33) in detail",
    "Explain the parable of the mosquito (2:26)",
    "Explain the parable of the spider's house (29:41)",
    "Explain the parable of the donkey carrying books (62:5)",
    "Explain the parable of the dog panting (7:176)",
    "Explain the parable of the hypocrite's shelter (16:92)",
    "Explain the parable of the two men with gardens (18:32-44)",
    "Explain the parable of the good and bad tree (14:24-26)",
    "Explain the parable of the city devastated and revived (2:259)",
    "Explain the parable of God's light in niches (24:35)",
    "Explain the parable of two seas in a man (35:12)",
    "Explain the parable of a slave owned by contending masters (39:29)",
    "Explain the parable of the rain and the crop (57:20)",
    "Explain the parable of the mountain and the Quran (59:21)",
    "Explain the parable of barren land revived (50:9-11)",
    "Explain the parable of the one lost in the desert (6:71)",
    "Explain the parable of God's words as ink of the sea (31:27)",
    "Explain the parable of believers and seed-sprouts (48:29)",
    "Explain the parable of grown son and elderly parents (46:15-17)",
    "Explain the parable of those who give charity sincerely (2:261)",

    # ── Divine oaths (~15)
    "What does the Quran swear by 'the dawn' (Surah 89)?",
    "What does the Quran swear by 'the fig and the olive' (Surah 95)?",
    "What does the Quran swear by 'the pen and what they write' (Surah 68:1)?",
    "What does the Quran swear by 'the morning' (Surah 93)?",
    "What does the Quran swear by 'time' (Surah 103)?",
    "What does the Quran swear by 'the sun and its brightness' (Surah 91)?",
    "What does the Quran swear by 'the night' (Surahs 92 and 89)?",
    "What does the Quran swear by 'the star' (Surah 53)?",
    "What does the Quran swear by 'those who scatter' (Surah 51)?",
    "What does the Quran swear by 'those arranged in ranks' (Surah 37)?",
    "What does the Quran swear by 'those who snatch' (Surah 79)?",
    "What does the Quran swear by 'those flung afar' (Surah 77)?",
    "What does the Quran swear by 'the summit' (Surah 90)?",
    "What does the Quran swear by 'the resurrection soul' (Surah 75)?",
    "What does the Quran swear by 'the heavens and the zodiacal constellations' (Surah 85)?",

    # ── Women in the Quran (~15)
    "What does the Quran say about Eve and her creation?",
    "What does the Quran say about Sara, wife of Abraham?",
    "What does the Quran say about Hagar and the zamzam origin narrative?",
    "What does the Quran say about the mother of Moses?",
    "What does the Quran say about the sister of Moses?",
    "What does the Quran say about the wife of Pharaoh (Asiya)?",
    "What does the Quran say about the daughters of the old man of Madyan (Shu'ayb)?",
    "What does the Quran say about the Queen of Sheba (Bilqis)?",
    "What does the Quran say about the wife of Imran (mother of Mary)?",
    "What does the Quran say about Mary, mother of Jesus, in detail?",
    "What does the Quran say about the wives of Noah and Lot?",
    "What does the Quran say about the wife of Abu Lahab (Surah 111)?",
    "What does the Quran say about the women addressed as 'O women of the prophet'?",
    "What does the Quran say about the believing woman who pledged allegiance?",
    "What does the Quran say about the female companions in paradise?",

    # ── Geography of the Quran (~15)
    "What does the Quran say about the Kaaba and the sacred mosque?",
    "What does the Quran say about Mount Sinai (at-Tur)?",
    "What does the Quran say about the sacred valley of Tuwa?",
    "What does the Quran say about Mount Judi (where Noah's ark rested)?",
    "What does the Quran say about the dwellings of Thamud (Hijr)?",
    "What does the Quran say about the city of Madyan?",
    "What does the Quran say about the garden of Saba and its destruction?",
    "What does the Quran say about Egypt in its narratives?",
    "What does the Quran say about the Blessed Land around the Aqsa mosque?",
    "What does the Quran say about Babylon in reference to Harut and Marut?",
    "What does the Quran say about the farthest mosque (Masjid al-Aqsa)?",
    "What does the Quran say about the two seas that meet?",
    "What does the Quran say about the cities of Lot's people?",
    "What does the Quran say about Iram of the pillars?",
    "What does the Quran say about the cave where the sleepers rested?",

    # ── Numbers with Quranic significance (~15)
    "What is the significance of the number 19 in the Quran?",
    "What is the significance of the number 7 in the Quran?",
    "What is the significance of the number 12 in the Quran?",
    "What is the significance of the number 40 in the Quran?",
    "What is the significance of the number 3 in the Quran?",
    "What is the significance of the number 5 in the Quran?",
    "What is the significance of the number 9 in the Quran?",
    "What is the significance of the number 70 in the Quran?",
    "What is the significance of the number 2 (pairs) in the Quran?",
    "What is the significance of the number 30 (nights for Moses) in the Quran?",
    "What is the significance of the number 50 (thousand years) in the Quran?",
    "What is the significance of the number 300 in Quran (People of the Cave)?",
    "What is the significance of the number 1000 in the Quran?",
    "What is the significance of the number 100 in the Quran?",
    "What is the significance of the number 80 in the Quran?",

    # ── Acts of worship in specific situations (~15)
    "How does the Quran direct worship in times of fear and war?",
    "How does the Quran direct worship while traveling long distances?",
    "How does the Quran direct worship when ill or bed-ridden?",
    "How does the Quran direct worship for women during menstruation?",
    "How does the Quran direct worship during pregnancy and postpartum?",
    "How does the Quran direct worship when boarding ships or transport?",
    "How does the Quran direct worship after successful work or harvest?",
    "How does the Quran direct worship during a drought or flood?",
    "How does the Quran direct worship before an important decision?",
    "How does the Quran direct worship upon witnessing God's signs in nature?",
    "How does the Quran direct worship when facing death?",
    "How does the Quran direct worship when one receives good news?",
    "How does the Quran direct worship during the last ten nights of Ramadan?",
    "How does the Quran direct worship in response to calamity or misfortune?",
    "How does the Quran direct worship after committing a sin?",
]


def filter_new(questions, cache_path="data/answer_cache.json"):
    try:
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        seen = {e.get("question", "").strip().lower() for e in cache}
    except Exception:
        seen = set()
    new = [q for q in questions if q.strip().lower() not in seen]
    return new


if __name__ == "__main__":
    fresh = filter_new(PHASE9_RAW)
    print(f"[phase9] total={len(PHASE9_RAW)}, after dedup={len(fresh)}")
    sf = Path("overnight_seed.state.json")
    if sf.exists():
        try:
            st = json.loads(sf.read_text(encoding="utf-8"))
            st["done"] = [q for q in st.get("done", []) if q in fresh]
            st["failed"] = []
            sf.write_text(json.dumps(st, indent=2), encoding="utf-8")
            print(f"[phase9] state pruned to {len(st['done'])} matching done entries")
        except Exception as e:
            print(f"[phase9] state prune failed: {e}")
    engine.QUESTIONS = fresh
    engine.main()
