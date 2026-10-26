"""Rule-based Brazilian Portuguese G2P in espeak-ng pt-br conventions, as misaki hands them to Kokoro.

The prototype of KokoroEngine/PortuguesePhonemes.swift: the Swift port follows it step by step.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

VOWELS = set("aeiouáéíóúâêôãõàüy")
FRONT = set("eiéêíy")
VOICED = set("bdɡlmnɾrvzʒʤjw")
PHONEME_VOWELS = set("aeiouɐæɛɔʊyAIOWYũõ")
PUNCTUATION = set(';:,.!?—…"()“”')

UNSTRESSED = {
    "o",
    "os",
    "a",
    "as",
    "ao",
    "aos",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "no",
    "na",
    "nos",
    "nas",
    "num",
    "numa",
    "e",
    "é",
    "que",
    "se",
    "me",
    "te",
    "lhes",
    "vos",
    "com",
    "por",
    "mas",
    "ou",
    "seu",
    "seus",
    "meu",
    "meus",
    "teu",
    "teus",
    "um",
    "uns",
    "há",
    "foi",
    "tem",
    "são",
    "ser",
    "ter",
    "sou",
    "nós",
    "lo",
    "la",
    "los",
    "las",
    "eu",
    "pra",
    "vai",
    "faz",
}
SECONDARY = {
    "uma",
    "umas",
    "para",
    "não",
    "sua",
    "suas",
    "minha",
    "minhas",
    "você",
    "vocês",
    "como",
    "era",
    "eram",
    "está",
    "estava",
    "será",
    "seria",
    "pode",
    "podem",
    "podia",
    "tinha",
    "até",
    "ele",
    "ela",
    "eles",
    "elas",
    "este",
    "esta",
    "estes",
    "estas",
    "esse",
    "esses",
    "neste",
    "nesta",
    "nesse",
    "nesses",
    "onde",
    "aquela",
    "fazer",
    "teve",
    "sempre",
    "nunca",
    "tenho",
    "tinham",
    "foram",
    "estão",
    "deles",
    "nossa",
    "nosso",
    "posso",
    "desde",
}
CLITICS = {
    "me",
    "te",
    "se",
    "lhe",
    "lhes",
    "o",
    "a",
    "os",
    "as",
    "lo",
    "la",
    "los",
    "las",
    "nos",
    "vos",
    "no",
    "na",
}
WORDS = {
    "e": "i",
    "é": "ɛ",
    "à": "ˌaː",
    "às": "aːs",
    "ao": "W",
    "aos": "Ws",
    "o": "ʊ",
    "os": "ʊs",
    "lhe": "ʎˈy",
    "muito": "mwˈiŋtʊ",
    "muita": "mwˈiŋtæ",
    "muitos": "mwˈiŋtʊs",
    "muitas": "mwˈiŋtæs",
    "melhor": "meljˈɔr",
    "pior": "piˈɔr",
    "maior": "mIˈɔr",
    "menor": "menˈɔr",
    "olha": "ˈɔljæ",
    "olhe": "ˈɔljy",
}
# Letter names, for words without a vowel (PDF, HTML), which are spelled out.
LETTERS = {
    "b": "bê",
    "c": "cê",
    "d": "dê",
    "f": "éfe",
    "g": "gê",
    "h": "agá",
    "j": "jóta",
    "k": "cá",
    "l": "éle",
    "m": "éme",
    "n": "éne",
    "p": "pê",
    "q": "quê",
    "r": "érre",
    "s": "ésse",
    "t": "tê",
    "v": "vê",
    "w": "dáblio",
    "x": "xis",
    "z": "zê",
    "ç": "cê",
}
# Unstressed words whose final vowel still glides before a vowel.
GLIDING = {"ʤy", "sy", "my", "ʧy"}
# Stems where x reads ks or s instead of ʃ.
X_KS = ("fix", "táx", "tax", "tóx", "tox", "sex", "nex", "plex", "flex", "flux", "léx", "lex", "doxo", "oxig")
X_S = ("máx", "próx", "aux", "troux", "sintax")
DIPHTHONGS = {
    "ai": "I",
    "au": "W",
    "ei": "A",
    "éi": "ɛɪ",
    "eu": "eʊ",
    "éu": "ɛʊ",
    "oi": "oɪ",
    "ói": "Y",
    "ou": "ow",
    "iu": "iʊ",
    "ui": "uɪ",
    "ão": "ɐ̃ʊ̃",
    "ãe": "ɐ̃y",
    "õe": "õj",
    "ãi": "ɐ̃j",
}
ACCENTED = {
    "á": "a",
    "à": "a",
    "â": "ɐ̃",
    "ã": "ɐ̃",
    "é": "ɛ",
    "ê": "e",
    "í": "i",
    "ó": "ɔ",
    "ô": "o",
    "õ": "õ",
    "ú": "u",
    "y": "i",
}
NASAL = {"a": "ɐ̃", "e": "A", "i": "i", "o": "o", "u": "ũ"}
PLAIN = {
    "á": "a",
    "à": "a",
    "â": "a",
    "ã": "a",
    "é": "e",
    "ê": "e",
    "í": "i",
    "ó": "o",
    "ô": "o",
    "õ": "o",
    "ú": "u",
    "y": "i",
}


@dataclass
class Unit:
    """One vowel letter, or a consonant spelled with one or two letters."""

    vowel: bool
    text: str


def units(word: str) -> list[Unit]:
    out: list[Unit] = []
    i = 0
    while i < len(word):
        c, n1, n2 = word[i], word[i + 1 : i + 2], word[i + 2 : i + 3]
        if c in "qg" and n1 in ("u", "ü") and n2 and n2 in VOWELS:
            out.append(Unit(False, c + ("u" if n1 == "u" and n2 in FRONT else "w")))
            i += 2
        elif c + n1 in ("ch", "lh", "nh", "rr", "ss", "sh", "ph", "th", "ck", "ll", "sç", "mm"):
            out.append(Unit(False, c + n1))
            i += 2
        elif c + n1 in ("sc", "xc") and n2 in FRONT:
            out.append(Unit(False, "ss"))
            i += 2
        else:
            out.append(Unit(c in VOWELS, c))
            i += 1
    return out


def nuclei(us: list[Unit]) -> list[list[int]]:
    """The vowel units of each syllable nucleus, by index."""
    groups: list[list[int]] = []
    for i, u in enumerate(us):
        if u.vowel:
            if groups and groups[-1][-1] == i - 1 and len(groups[-1]) == 1 and glides(us, groups[-1][0], i):
                groups[-1].append(i)
            else:
                groups.append([i])
    return groups


def is_coda(us: list[Unit], i: int) -> bool:
    """Whether consonant unit `i` closes its syllable."""
    return i + 1 >= len(us) or not us[i + 1].vowel


def glides(us: list[Unit], i: int, j: int) -> bool:
    """Whether vowel `j` joins vowel `i` as the second half of a falling diphthong."""
    first, second = us[i].text, us[j].text
    if first in "ãõâ" and second in "oei":
        return True
    if second not in ("i", "u") or first == second:
        return False
    if j + 1 < len(us) and not us[j + 1].vowel:
        nxt = us[j + 1].text
        if nxt == "nh" or (nxt in ("m", "n", "l", "r", "z") and is_coda(us, j + 1)):
            return False
    return True


def stressed(word: str, us: list[Unit], groups: list[list[int]]) -> int:
    for marks in ("áéíóúâêô", "ãõ"):
        for k, g in enumerate(groups):
            if any(us[i].text in marks for i in g):
                return k
    if len(groups) > 1 and re.search(r"(?:[aeo]s?|am|em|ens)$", word):
        return len(groups) - 2
    return len(groups) - 1


def open_mid(letter: str, tail: str) -> bool:
    """Whether a stressed unaccented e or o is open (ɛ, ɔ), from the spelling after it."""
    if letter == "e":
        return bool(
            re.fullmatch(
                r"las?|stas?|ssas?|rr[ae]s?|rn[aoe]s?|gas?|cem?|ves?|lhas?|jas?|tos?|tas?|ras?|ram|ro|rt[ao]s?|rvas?",
                tail,
            )
        )
    return bool(
        re.fullmatch(
            r"l[ao]s?|ras?|rt[aeo]s?|rm[ae]s?|rn[aeo]s?|ss[ao]s?|sas?|stas?|sos|vas?|vos|dem?|gos?|tas?|pas?|chas?|jas?|lhos",
            tail,
        )
    )


@dataclass
class Syllable:
    mark: str
    phonemes: str
    consumed: int  # coda units the vowel absorbed


def nucleus(us: list[Unit], g: list[int], k: int, p: int, level: str, word: str) -> Syllable:
    text = "".join(us[i].text for i in g)
    end = g[-1] + 1
    after = us[end].text if end < len(us) else ""
    nasal_onset = after in ("m", "n", "nh") and not is_coda(us, end)
    tail = "".join(u.text for u in us[end:])
    tonic = k == p and level != "none"
    if len(g) == 2:
        v = DIPHTHONGS.get(text) or ACCENTED.get(text[0], text[0]) + {"i": "ɪ", "u": "ʊ"}.get(
            text[1], text[1]
        )
    elif tonic:
        if text in ("a", "á", "â"):
            v = "ɐ̃" if nasal_onset or (text == "â" and after in ("m", "n")) else "a"
        elif text in ("e", "o"):
            v = text if nasal_onset or not open_mid(text, tail) else {"e": "ɛ", "o": "ɔ"}[text]
        else:
            v = ACCENTED.get(text, text)
    elif tail in ("", "s") and (text in ("e", "o") or (text == "a" and k > p)):
        v = {"a": "æ", "e": "y", "o": "ʊ"}[text]
    elif text in ("a", "à") and nasal_onset:
        v = "ɐ̃" if after == "nh" else "æ"
    else:
        v = ACCENTED.get(text, text)
    if text == "i" and k > p and end < len(us) and us[end].vowel:
        v = "j"
    consumed = 0
    if end < len(us) and is_coda(us, end):
        rest = us[end + 1].text if end + 1 < len(us) else ""
        base = PLAIN.get(text, text)
        if after in ("m", "n"):
            consumed = 1
            if not rest and k > p and text == "a" and after == "m":
                v = "ɐ̃ʊ̃"
            else:
                v = NASAL.get(base, v) + ("m" if rest in ("p", "b") else "ŋ")
        elif after == "l":
            consumed = 1
            if base == "a":
                v = "W"
            elif base == "e":
                v = ("ɛ" if tonic and not rest and text != "ê" else "e") + "ʊ"
            elif base == "i":
                v = "iʊ"
            elif base == "o" and tonic:
                v = "ɔl"
            else:
                v = v + "w"
    mark = ""
    if k == p:
        mark = {"primary": "ˈ", "secondary": "ˌ", "none": ""}[level]
    return Syllable(mark, v, consumed)


def phonemize_word(word: str, level: str) -> str:
    """`level`: "primary", "secondary" or "none", the stress of the stressed syllable."""
    if word in WORDS:
        return WORDS[word]
    us = units(word)
    groups = nuclei(us)
    if not groups:
        names = [LETTERS[c] for c in word if c in LETTERS]
        return "".join(
            phonemize_word(n, "primary" if j == len(names) - 1 else "secondary") for j, n in enumerate(names)
        )
    p = stressed(word, us, groups)
    last = len(groups) - 1
    secondary: set[int] = set()
    if level == "primary":
        if p >= 2 or (p == 1 and p < last):
            secondary.add(0)
        secondary.update(range(2, p - 1, 2))
    segments: list[tuple[int, Syllable | None]] = []
    starts = {g[0]: k for k, g in enumerate(groups)}
    i = 0
    while i < len(us):
        if i in starts:
            k = starts[i]
            syllable = nucleus(us, groups[k], k, p, level, word)
            if k in secondary:
                syllable.mark = "ˌ"
            segments.append((i, syllable))
            i = groups[k][-1] + 1 + syllable.consumed
        else:
            segments.append((i, None))
            i += 1
    rendered: list[str] = []
    for i, syllable in reversed(segments):
        if syllable is not None:
            rendered.append(syllable.mark + syllable.phonemes)
        else:
            rendered.append(consonant(us, i, word, "".join(reversed(rendered)).lstrip("ˈˌ")))
    return "".join(reversed(rendered))


def consonant(us: list[Unit], i: int, word: str, following: str) -> str:
    t = us[i].text
    prev = us[i - 1] if i > 0 else None
    nxt = us[i + 1] if i + 1 < len(us) else None
    first = following[:1]
    palatal = first in ("i", "y", "j")
    match t:
        case "t":
            return "ʧ" if palatal or nxt is None else "t"
        case "d":
            return "ʤ" if palatal or nxt is None else "d"
        case "c":
            return "s" if nxt is not None and nxt.text in FRONT else "k"
        case "ç" | "ss" | "sç":
            return "s"
        case "g":
            return "ʒ" if nxt is not None and nxt.text in FRONT else "ɡ"
        case "gw":
            return "ɡw"
        case "qu":
            return "k"
        case "gu":
            return "ɡ"
        case "qw":
            return "kw"
        case "ch" | "sh":
            return "ʃ"
        case "lh":
            return "ʎ" if i == 0 else "lj"
        case "nh":
            return "ɲ"
        case "rr":
            return "x"
        case "r":
            if i == 0 or (prev is not None and prev.text in ("n", "l", "s")):
                return "x"
            if prev is not None and not prev.vowel:
                return "r"
            if nxt is None:
                return "r"
            return "ɾ" if nxt.vowel else "ɾə"
        case "s" | "z":
            if t == "s" and i == 0:
                return "s"
            if nxt is None:
                return "s"
            if not nxt.vowel:
                return "z" if first in VOICED else "s"
            return "z" if t == "z" or (prev is not None and prev.vowel) else "s"
        case "x":
            stem = "".join(u.text for u in us[: i + 1])
            if i == 0:
                return "ʃ"
            if nxt is None:
                return "ks"
            if not nxt.vowel:
                return "s"
            if stem.endswith(X_KS):
                return "ks"
            if stem.endswith(X_S):
                return "s"
            if stem in ("ex", "êx", "hex", "hêx"):
                return "z"
            return "ʃ"
        case "h":
            return ""
        case "j":
            return "ʒ"
        case "ph":
            return "f"
        case "th":
            return "t"
        case "ck":
            return "k"
        case "ll":
            return "l"
        case _:
            return t


TOKEN = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*|[^\s\w]")


def phonemize(text: str, word: Callable[[str, str], str] | None = None) -> str:
    """`word` phonemizes one word at a stress level, `phonemize_word` by default."""
    word = word or phonemize_word
    text = unicodedata.normalize("NFC", text).replace("«", "“").replace("»", "”").replace("–", "—")
    items: list[tuple[bool, str]] = []  # (is word, phonemes or punctuation)
    for token in TOKEN.findall(text):
        if not token[0].isalpha():
            if token in PUNCTUATION:
                items.append((False, token))
            continue
        lower = token.lower()
        parts = [part for part in lower.replace("'", "").replace("’", "").split("-") if part]
        rendered = []
        for j, part in enumerate(parts):
            if part in UNSTRESSED or (j > 0 and part in CLITICS):
                level = "none"
            elif part in SECONDARY:
                level = "secondary"
            else:
                level = "primary"
            rendered.append(word(part, level))
        items.append((True, "".join(rendered)))
    out = ""
    for n, (is_word, phonemes) in enumerate(items):
        if not is_word:
            if phonemes in ("(", "“", "—") and out and not out.endswith(" "):
                out += " "
            out += phonemes + (" " if phonemes == "—" else "")
            continue
        if n + 1 < len(items) and items[n + 1][0]:
            nxt = items[n + 1][1].lstrip("ˈˌ")
            vowel_next = nxt[:1] in PHONEME_VOWELS
            if phonemes.endswith("s") and (vowel_next or nxt[:1] in VOICED):
                phonemes = phonemes[:-1] + "z"
            elif vowel_next and phonemes == "ʊ":
                phonemes = "u"
            elif (
                vowel_next
                and phonemes.endswith(("y", "ʊ"))
                and phonemes[-2:-1] not in PHONEME_VOWELS
                and ("ˈ" in phonemes or phonemes in GLIDING)
            ):
                phonemes = phonemes[:-1] + {"y": "j", "ʊ": "w"}[phonemes[-1]]
            elif vowel_next and phonemes.endswith("r"):
                phonemes = phonemes[:-1] + "ɾ"
        if out and not out.endswith((" ", "(", "“")):
            out += " "
        out += phonemes
    return unicodedata.normalize("NFD", out.strip())
