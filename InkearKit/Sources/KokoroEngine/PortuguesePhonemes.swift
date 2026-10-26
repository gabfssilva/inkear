import Foundation

/// Brazilian Portuguese text as the phonemes Kokoro's Portuguese voices were trained on.
///
/// Kokoro learned Portuguese from espeak-ng `pt-br` output as misaki's `EspeakG2P` rewrites it:
/// stress marks before the stressed vowel, `A`, `I`, `W` for [eɪ], [aɪ], [aʊ], `ʧ` and `ʤ` for the
/// affricates, and nasal vowels with a combining tilde. These rules derive that output from the
/// spelling, including espeak-ng's unstressed function words and its sandhi between words. They
/// miss mostly on whether an unaccented stressed e or o is open, which the spelling does not show:
/// on 600 sentences of the Portuguese corpus they differ from espeak-ng in 1.6% of the phonemes.
/// `tools/kokoro-pt` holds the Python prototype and the comparison.
enum PortuguesePhonemes {
  /// `text` as Kokoro phonemes: words separated by spaces, with the punctuation Kokoro reads.
  static func phonemes(for text: String) -> String {
    var items: [Item] = []
    for token in tokens(
      in: text.precomposedStringWithCanonicalMapping
        .replacing("«", with: "“").replacing("»", with: "”").replacing("–", with: "—"))
    {
      switch token {
      case .mark(let mark):
        items.append(.mark(mark))
      case .word(let word):
        let parts = word.lowercased().filter { $0 != "'" && $0 != "’" }.split(separator: "-")
        let rendered = parts.enumerated().map { index, part in
          let part = String(part)
          let stress: Stress =
            if unstressed.contains(part) || (index > 0 && clitics.contains(part)) {
              .unstressed
            } else if secondary.contains(part) {
              .secondary
            } else {
              .primary
            }
          return Self.phonemes(ofWord: part, stress: stress)
        }
        items.append(.word(rendered.joined()))
      }
    }
    var out = ""
    for (index, item) in items.enumerated() {
      switch item {
      case .mark(let mark):
        if "(“—".contains(mark), !out.isEmpty, !out.hasSuffix(" ") { out += " " }
        out.append(mark)
        if mark == "—" { out += " " }
      case .word(var phonemes):
        if index + 1 < items.count, case .word(let next) = items[index + 1] {
          phonemes = joined(phonemes, before: next)
        }
        if !out.isEmpty, !out.hasSuffix(" "), !out.hasSuffix("("), !out.hasSuffix("“") {
          out += " "
        }
        out += phonemes
      }
    }
    return out.trimmingCharacters(in: .whitespaces).decomposedStringWithCanonicalMapping
  }

  private enum Item {
    case word(String)
    case mark(Character)
  }

  /// The stress of a word's stressed syllable.
  private enum Stress { case primary, secondary, unstressed }

  /// Words made of letters, joined inside by apostrophes and hyphens, and the punctuation Kokoro
  /// reads. Digits and other symbols are dropped.
  private static func tokens(in text: String) -> [Item] {
    var tokens: [Item] = []
    var word = ""
    var joiner: Character?
    for character in text {
      if character.isLetter {
        if let pending = joiner { word.append(pending) }
        joiner = nil
        word.append(character)
      } else if "'’-".contains(character), !word.isEmpty, joiner == nil {
        joiner = character
      } else {
        if !word.isEmpty { tokens.append(.word(word)) }
        word = ""
        joiner = nil
        if punctuation.contains(character) { tokens.append(.mark(character)) }
      }
    }
    if !word.isEmpty { tokens.append(.word(word)) }
    return tokens
  }

  /// `phonemes` as espeak-ng says them before the word `next`: a final s voices before a vowel or
  /// a voiced consonant, a final r taps before a vowel, and an unstressed final e or o glides.
  private static func joined(_ phonemes: String, before next: String) -> String {
    guard let first = next.unicodeScalars.first(where: { $0 != "ˈ" && $0 != "ˌ" }),
      let last = phonemes.unicodeScalars.last
    else { return phonemes }
    let vowelNext = phonemeVowels.contains(first)
    let start = String(phonemes.unicodeScalars.dropLast())
    if last == "s", vowelNext || voiced.contains(first) {
      return start + "z"
    }
    guard vowelNext else { return phonemes }
    if phonemes == "ʊ" { return "u" }
    if last == "y" || last == "ʊ",
      phonemes.unicodeScalars.dropLast().last.map(phonemeVowels.contains) != true,
      phonemes.contains("ˈ") || gliding.contains(phonemes)
    {
      return start + (last == "y" ? "j" : "w")
    }
    return last == "r" ? start + "ɾ" : phonemes
  }

  private static func phonemes(ofWord word: String, stress: Stress) -> String {
    if let fixed = words[word] { return fixed }
    let units = Self.units(of: Array(word))
    let nuclei = Self.nuclei(of: units)
    guard !nuclei.isEmpty else {
      let names = word.compactMap { letterNames[$0] }
      return names.enumerated().map { index, name in
        phonemes(ofWord: name, stress: index == names.count - 1 ? .primary : .secondary)
      }.joined()
    }
    let stressed = stressedNucleus(of: word, units, nuclei)
    var secondaries: Set<Int> = []
    if stress == .primary {
      if stressed >= 2 || (stressed == 1 && stressed < nuclei.count - 1) { secondaries.insert(0) }
      secondaries.formUnion(stride(from: 2, to: stressed - 1, by: 2))
    }
    var segments: [Segment] = []
    var index = 0
    while index < units.count {
      if let k = nuclei.firstIndex(where: { $0.lowerBound == index }) {
        var syllable = nucleus(
          units, nuclei[k], stressed: k == stressed, afterStress: k > stressed, stress: stress)
        if secondaries.contains(k) { syllable.mark = "ˌ" }
        segments.append(.vowel(syllable))
        index = nuclei[k].upperBound + 1 + syllable.consumed
      } else {
        segments.append(.consonant(index))
        index += 1
      }
    }
    var right = ""
    for segment in segments.reversed() {
      switch segment {
      case .vowel(let syllable):
        right = syllable.mark + syllable.phonemes + right
      case .consonant(let index):
        let following = right.unicodeScalars.first { $0 != "ˈ" && $0 != "ˌ" }
        right = consonant(units, index, following: following) + right
      }
    }
    return right
  }

  private enum Segment {
    case vowel(Syllable)
    case consonant(Int)
  }

  /// One vowel letter, or a consonant spelled with one or two letters.
  private struct Unit {
    let isVowel: Bool
    let text: String
  }

  private static func units(of letters: [Character]) -> [Unit] {
    var units: [Unit] = []
    var i = 0
    while i < letters.count {
      let c = letters[i]
      let next = i + 1 < letters.count ? letters[i + 1] : nil
      let after = i + 2 < letters.count ? letters[i + 2] : nil
      if c == "q" || c == "g", next == "u" || next == "ü", let after, vowels.contains(after) {
        let silent = next == "u" && front.contains(String(after))
        units.append(Unit(isVowel: false, text: String(c) + (silent ? "u" : "w")))
        i += 2
      } else if let next, digraphs.contains(String([c, next])) {
        units.append(Unit(isVowel: false, text: String([c, next])))
        i += 2
      } else if c == "s" || c == "x", next == "c", let after, front.contains(String(after)) {
        units.append(Unit(isVowel: false, text: "ss"))
        i += 2
      } else {
        units.append(Unit(isVowel: vowels.contains(c), text: String(c)))
        i += 1
      }
    }
    return units
  }

  /// The vowel units of each syllable nucleus.
  private static func nuclei(of units: [Unit]) -> [ClosedRange<Int>] {
    var nuclei: [ClosedRange<Int>] = []
    for (index, unit) in units.enumerated() where unit.isVowel {
      if let last = nuclei.last, last.upperBound == index - 1, last.count == 1,
        glides(units, last.lowerBound, index)
      {
        nuclei[nuclei.count - 1] = last.lowerBound...index
      } else {
        nuclei.append(index...index)
      }
    }
    return nuclei
  }

  /// Whether consonant unit `index` closes its syllable.
  private static func isCoda(_ units: [Unit], _ index: Int) -> Bool {
    index + 1 >= units.count || !units[index + 1].isVowel
  }

  /// Whether vowel `second` joins vowel `first` as the end of a falling diphthong.
  private static func glides(_ units: [Unit], _ first: Int, _ second: Int) -> Bool {
    let (a, b) = (units[first].text, units[second].text)
    if ["ã", "õ", "â"].contains(a), ["o", "e", "i"].contains(b) { return true }
    guard b == "i" || b == "u", a != b else { return false }
    if second + 1 < units.count, !units[second + 1].isVowel {
      let next = units[second + 1].text
      // Hiatus before a syllable-closing nasal, l, r or z, and before nh: a-in-da, sa-ir, ra-i-nha.
      if next == "nh" || (["m", "n", "l", "r", "z"].contains(next) && isCoda(units, second + 1)) {
        return false
      }
    }
    return true
  }

  private static func stressedNucleus(
    of word: String, _ units: [Unit], _ nuclei: [ClosedRange<Int>]
  ) -> Int {
    for marks: Set<String> in [acuteOrCircumflex, ["ã", "õ"]] {
      let marked = nuclei.firstIndex { nucleus in
        nucleus.contains { marks.contains(units[$0].text) }
      }
      if let marked { return marked }
    }
    let penultimate = ["a", "e", "o", "as", "es", "os", "am", "em", "ens"]
    return nuclei.count > 1 && penultimate.contains(where: word.hasSuffix)
      ? nuclei.count - 2 : nuclei.count - 1
  }

  /// Whether a stressed unaccented e or o is open, from the spelling after it.
  private static func isOpen(_ letter: String, before tail: String) -> Bool {
    (letter == "e" ? openE : openO).contains(tail)
  }

  private struct Syllable {
    var mark: String
    let phonemes: String
    /// Coda consonants the vowel absorbed: a nasal m or n, or a vocalized l.
    let consumed: Int
  }

  private static func nucleus(
    _ units: [Unit], _ range: ClosedRange<Int>, stressed: Bool, afterStress: Bool, stress: Stress
  ) -> Syllable {
    let text = range.map { units[$0].text }.joined()
    let end = range.upperBound + 1
    let after = end < units.count ? units[end].text : ""
    let nasalOnset = ["m", "n", "nh"].contains(after) && !isCoda(units, end)
    let tail = units[end...].map(\.text).joined()
    let tonic = stressed && stress != .unstressed
    var v: String
    if range.count == 2 {
      let (first, second) = (units[range.lowerBound].text, units[range.upperBound].text)
      v =
        diphthongs[text]
        ?? (accented[first] ?? first) + (second == "i" ? "ɪ" : second == "u" ? "ʊ" : second)
    } else if tonic {
      switch text {
      case "a", "á", "â":
        v = nasalOnset || (text == "â" && (after == "m" || after == "n")) ? "ɐ\u{303}" : "a"
      case "e", "o":
        v = nasalOnset || !isOpen(text, before: tail) ? text : (text == "e" ? "ɛ" : "ɔ")
      default:
        v = accented[text] ?? text
      }
    } else if tail == "" || tail == "s", text == "e" || text == "o" || (text == "a" && afterStress) {
      v = text == "a" ? "æ" : (text == "e" ? "y" : "ʊ")
    } else if text == "a" || text == "à", nasalOnset {
      v = after == "nh" ? "ɐ\u{303}" : "æ"
    } else {
      v = accented[text] ?? text
    }
    if text == "i", afterStress, end < units.count, units[end].isVowel { v = "j" }
    var consumed = 0
    if end < units.count, isCoda(units, end) {
      let rest = end + 1 < units.count ? units[end + 1].text : ""
      let base = plain[text] ?? text
      if after == "m" || after == "n" {
        consumed = 1
        if rest.isEmpty, afterStress, text == "a", after == "m" {
          v = "ɐ\u{303}ʊ\u{303}"
        } else {
          v = (nasal[base] ?? v) + (rest == "p" || rest == "b" ? "m" : "ŋ")
        }
      } else if after == "l" {
        consumed = 1
        switch base {
        case "a": v = "W"
        case "e": v = (tonic && rest.isEmpty && text != "ê" ? "ɛ" : "e") + "ʊ"
        case "i": v = "iʊ"
        case "o" where tonic: v = "ɔl"
        default: v += "w"
        }
      }
    }
    let mark =
      switch (stressed, stress) {
      case (true, .primary): "ˈ"
      case (true, .secondary): "ˌ"
      default: ""
      }
    return Syllable(mark: mark, phonemes: v, consumed: consumed)
  }

  /// The consonant unit at `index`, before the phoneme `following` it in the word.
  private static func consonant(
    _ units: [Unit], _ index: Int, following: Unicode.Scalar?
  ) -> String {
    let text = units[index].text
    let previous = index > 0 ? units[index - 1] : nil
    let next = index + 1 < units.count ? units[index + 1] : nil
    let palatal = following == "i" || following == "y" || following == "j"
    let frontVowelNext = next.map { front.contains($0.text) } ?? false
    switch text {
    case "t": return palatal || next == nil ? "ʧ" : "t"
    case "d": return palatal || next == nil ? "ʤ" : "d"
    case "c": return frontVowelNext ? "s" : "k"
    case "g": return frontVowelNext ? "ʒ" : "ɡ"
    case "ç", "ss", "sç": return "s"
    case "gw": return "ɡw"
    case "qw": return "kw"
    case "gu": return "ɡ"
    case "qu": return "k"
    case "ch", "sh": return "ʃ"
    case "lh": return index == 0 ? "ʎ" : "lj"
    case "nh": return "ɲ"
    case "rr": return "x"
    case "r":
      if index == 0 || previous.map({ ["n", "l", "s"].contains($0.text) }) == true { return "x" }
      if let previous, !previous.isVowel { return "r" }
      guard let next else { return "r" }
      return next.isVowel ? "ɾ" : "ɾə"
    case "s", "z":
      if text == "s", index == 0 { return "s" }
      guard let next else { return "s" }
      if !next.isVowel { return following.map(voiced.contains) == true ? "z" : "s" }
      return text == "z" || previous?.isVowel == true ? "z" : "s"
    case "x":
      let stem = units[...index].map(\.text).joined()
      guard index > 0 else { return "ʃ" }
      guard let next else { return "ks" }
      if !next.isVowel { return "s" }
      if xAsKs.contains(where: stem.hasSuffix) { return "ks" }
      if xAsS.contains(where: stem.hasSuffix) { return "s" }
      return ["ex", "êx", "hex", "hêx"].contains(stem) ? "z" : "ʃ"
    case "h": return ""
    case "j": return "ʒ"
    case "ph": return "f"
    case "th": return "t"
    case "ck": return "k"
    case "ll": return "l"
    default: return text
    }
  }

  private static let vowels: Set<Character> = [
    "a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú", "â", "ê", "ô", "ã", "õ", "à", "ü", "y",
  ]
  private static let front: Set<String> = ["e", "i", "é", "ê", "í", "y"]
  private static let acuteOrCircumflex: Set<String> = ["á", "é", "í", "ó", "ú", "â", "ê", "ô"]
  private static let digraphs: Set<String> = [
    "ch", "lh", "nh", "rr", "ss", "sh", "ph", "th", "ck", "ll", "sç", "mm",
  ]
  private static let punctuation: Set<Character> = [
    ";", ":", ",", ".", "!", "?", "—", "…", "\"", "(", ")", "“", "”",
  ]
  private static let voiced = Set("bdɡlmnɾrvzʒʤjw".unicodeScalars)
  private static let phonemeVowels = Set("aeiouɐæɛɔʊyAIOWY".unicodeScalars)

  private static let diphthongs: [String: String] = [
    "ai": "I", "au": "W", "ei": "A", "éi": "ɛɪ", "eu": "eʊ", "éu": "ɛʊ", "oi": "oɪ", "ói": "Y",
    "ou": "ow", "iu": "iʊ", "ui": "uɪ", "ão": "ɐ\u{303}ʊ\u{303}", "ãe": "ɐ\u{303}y",
    "õe": "o\u{303}j", "ãi": "ɐ\u{303}j",
  ]
  private static let accented: [String: String] = [
    "á": "a", "à": "a", "â": "ɐ\u{303}", "ã": "ɐ\u{303}", "é": "ɛ", "ê": "e", "í": "i", "ó": "ɔ",
    "ô": "o", "õ": "o\u{303}", "ú": "u", "y": "i",
  ]
  private static let nasal: [String: String] = [
    "a": "ɐ\u{303}", "e": "A", "i": "i", "o": "o", "u": "u\u{303}",
  ]
  private static let plain: [String: String] = [
    "á": "a", "à": "a", "â": "a", "ã": "a", "é": "e", "ê": "e", "í": "i", "ó": "o", "ô": "o",
    "õ": "o", "ú": "u", "y": "i",
  ]

  /// Spellings after a stressed e that make it open (ɛ).
  private static let openE: Set<String> = [
    "la", "las", "sta", "stas", "ssa", "ssas", "rra", "rras", "rre", "rres", "rna", "rnas", "rno",
    "rnos", "rne", "rnes", "ga", "gas", "ce", "cem", "ve", "ves", "lha", "lhas", "ja", "jas", "to",
    "tos", "ta", "tas", "ra", "ras", "ram", "ro", "rta", "rtas", "rto", "rtos", "rva", "rvas",
  ]
  /// Spellings after a stressed o that make it open (ɔ).
  private static let openO: Set<String> = [
    "la", "las", "lo", "los", "ra", "ras", "rta", "rtas", "rte", "rtes", "rto", "rtos", "rma",
    "rmas", "rme", "rmes", "rna", "rnas", "rne", "rnes", "rno", "rnos", "ssa", "ssas", "sso", "ssos",
    "sa", "sas", "sta", "stas", "sos", "va", "vas", "vos", "de", "dem", "go", "gos", "ta", "tas",
    "pa", "pas", "cha", "chas", "ja", "jas", "lhos",
  ]
  /// Words espeak-ng leaves unstressed.
  private static let unstressed: Set<String> = [
    "o", "os", "a", "as", "ao", "aos", "de", "do", "da", "dos", "das", "no", "na", "nos", "nas",
    "num", "numa", "e", "é", "que", "se", "me", "te", "lhes", "vos", "com", "por", "mas", "ou",
    "seu", "seus", "meu", "meus", "teu", "teus", "um", "uns", "há", "foi", "tem", "são", "ser",
    "ter", "sou", "nós", "lo", "la", "los", "las", "eu", "pra", "vai", "faz",
  ]
  /// Words espeak-ng gives only a secondary stress.
  private static let secondary: Set<String> = [
    "uma", "umas", "para", "não", "sua", "suas", "minha", "minhas", "você", "vocês", "como", "era",
    "eram", "está", "estava", "será", "seria", "pode", "podem", "podia", "tinha", "até", "ele",
    "ela", "eles", "elas", "este", "esta", "estes", "estas", "esse", "esses", "neste", "nesta",
    "nesse", "nesses", "onde", "aquela", "fazer", "teve", "sempre", "nunca", "tenho", "tinham",
    "foram", "estão", "deles", "nossa", "nosso", "posso", "desde",
  ]
  /// Pronouns after a hyphen, unstressed: disse-me, fazê-lo.
  private static let clitics: Set<String> = [
    "me", "te", "se", "lhe", "lhes", "o", "a", "os", "as", "lo", "la", "los", "las", "nos", "vos",
    "no", "na",
  ]
  private static let words: [String: String] = [
    "e": "i", "é": "ɛ", "à": "ˌaː", "às": "aːs", "ao": "W", "aos": "Ws", "o": "ʊ", "os": "ʊs",
    "lhe": "ʎˈy", "muito": "mwˈiŋtʊ", "muita": "mwˈiŋtæ", "muitos": "mwˈiŋtʊs",
    "muitas": "mwˈiŋtæs", "melhor": "meljˈɔr", "pior": "piˈɔr", "maior": "mIˈɔr",
    "menor": "menˈɔr", "olha": "ˈɔljæ", "olhe": "ˈɔljy",
  ]
  /// Unstressed words whose final vowel still glides before a vowel: de, se, me, te.
  private static let gliding: Set<String> = ["ʤy", "sy", "my", "ʧy"]
  /// Spellings of letters, for words without a vowel (PDF, HTML), which are spelled out with the
  /// stress on the last letter.
  private static let letterNames: [Character: String] = [
    "b": "bê", "c": "cê", "d": "dê", "f": "éfe", "g": "gê", "h": "agá", "j": "jóta", "k": "cá",
    "l": "éle", "m": "éme", "n": "éne", "p": "pê", "q": "quê", "r": "érre", "s": "ésse", "t": "tê",
    "v": "vê", "w": "dáblio", "x": "xis", "z": "zê", "ç": "cê",
  ]
  /// Stems where x reads ks or s rather than ʃ.
  private static let xAsKs = [
    "fix", "táx", "tax", "tóx", "tox", "sex", "nex", "plex", "flex", "flux", "léx", "lex", "doxo",
    "oxig",
  ]
  private static let xAsS = ["máx", "próx", "aux", "troux", "sintax"]
}
