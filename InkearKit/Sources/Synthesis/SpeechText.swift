import Foundation

/// Rewrites written text the way it is spoken, before synthesis.
public enum SpeechText {
  /// `text` in `language` ("pt" or "en"; any other language gets only the language-independent
  /// rules) with numbers, abbreviations, symbols, citations and URLs spelled out.
  public static func normalized(_ text: String, language: String) -> String {
    let dot =
      switch language {
      case "pt": " ponto "
      case "en": " dot "
      default: "."
      }
    let independent = independentRules(text, dot: dot)
    let spoken =
      switch language {
      case "pt": portuguese(independent)
      case "en": english(independent)
      default: independent
      }
    return spoken.replacing(pattern: #" {2,}"#, with: " ")
  }

  private static func independentRules(_ text: String, dot: String) -> String {
    text
      .replacing(pattern: #" *\[[0-9]+(?:\s*[,–-]\s*[0-9]+)*\]"#, with: "")
      .replacing(pattern: #"(?<=\p{L})[⁰¹²³⁴⁵⁶⁷⁸⁹]+"#, with: "")
      .replacing(pattern: #"\b(?:https?:\/\/|www\.)\S+"#) { spokenURL($0.whole, dot: dot) }
  }

  private static func portuguese(_ text: String) -> String {
    let spelling = Spelling("pt_BR")
    let symbols =
      text
      .replacing(pattern: #" *°C"#, with: " graus Celsius")
      .replacing(pattern: #" *& *"#, with: " e ")
      .replacing(pattern: #"§ *"#, with: "parágrafo ")
      .replacing(pattern: #"(?<=[0-9]) *\+ *(?=[0-9])"#, with: " mais ")
      .replacing(pattern: #"(?<=[0-9]) *= *(?=[0-9])"#, with: " igual a ")
      .replacing(pattern: #"(?<=[0-9]) *× *(?=[0-9])"#, with: " vezes ")
    let contextual =
      symbols
      .replacing(pattern: #"(?<!\p{L})(?:pág|p)\.(?=\s*[0-9])"#, with: "página")
      .replacing(pattern: #"(?<!\p{L})(?:págs|pp)\.(?=\s*[0-9])"#, with: "páginas")
      .replacing(
        pattern: #"(?<!\p{L})(?:Cap|cap)\.(?=\s*(?:[0-9]|[IVXLCDM]+(?!\p{L})))"#, with: "capítulo"
      )
      .replacing(pattern: #"(?<!\p{L})(?:n\.º|N\.º|nº|Nº|n°)(?=\s*[0-9])"#, with: "número")
      .replacing(pattern: #"(?<!\p{L})D\.(?=\s+\p{Lu})"#, with: "Dom")
    let abbreviated = expandAbbreviations(
      contextual, portugueseAbbreviations, terminal: ["etc", "a.C", "d.C"])
    return portugueseNumbers(abbreviated, spelling)
  }

  private static func portugueseNumbers(_ text: String, _ spelling: Spelling) -> String {
    text
      .replacing(
        pattern:
          #"(?<!\p{L})(?<kw>(?i:capítulo|parte|livro|volume|tomo|ato|cena|século))\s+(?<num>[IVXLCDM]+)(?!\p{L})"#
      ) {
        romanSpoken($0["kw"], $0["num"], spelling)
      }
      .replacing(pattern: #"(?<![0-9.,])(?<n>[0-9]+)\.?(?<s>[ºª])(?!\p{L})"#) {
        portugueseOrdinal($0["n"], feminine: $0["s"] == "ª") ?? String($0.whole)
      }
      .replacing(pattern: #"(?<![0-9/])(?<d>[0-9]{1,2})/(?<m>[0-9]{1,2})/(?<y>[0-9]{4})(?![0-9])"#)
    {
      portugueseDate($0["d"], $0["m"], $0["y"], spelling) ?? String($0.whole)
    }
      .replacing(pattern: #"(?<![0-9])(?<h>[0-9]{1,2})h(?<m>[0-9]{2})?(?![0-9\p{L}])"#) {
        portugueseTime($0["h"], minutes: $0.optional("m"), spelling) ?? String($0.whole)
      }
      .replacing(pattern: #"(?<![0-9])(?<h>[0-9]{1,2}):(?<m>[0-9]{2})(?![0-9])"#) {
        portugueseTime($0["h"], minutes: $0.optional("m"), spelling) ?? String($0.whole)
      }
      .replacing(pattern: #"(?<![0-9])(?<a>[0-9]{4}) ?[–-] ?(?<b>[0-9]{4})(?![0-9])"#) {
        numberWords($0["a"], style: .portuguese, spelling: spelling)
          + " a "
          + numberWords($0["b"], style: .portuguese, spelling: spelling)
      }
      .replacing(
        pattern:
          #"(?<!\p{L})(?<c>R|US)\$ ?(?<whole>[0-9]+(?:\.[0-9]{3})*)(?:,(?<cents>[0-9]{2}))?(?![0-9])"#
      ) {
        portugueseMoney($0["c"], whole: $0["whole"], cents: $0.optional("cents"), spelling)
          ?? String($0.whole)
      }
      .replacing(pattern: #"(?<![0-9.,])(?<n>[0-9]+(?:\.[0-9]{3})*(?:,[0-9]+)?) ?%"#) {
        numberWords($0["n"], style: .portuguese, spelling: spelling) + " por cento"
      }
      .replacing(
        pattern:
          #"(?<![\p{L}0-9.,])(?<n>[0-9]+(?:\.[0-9]{3})*(?:,[0-9]+)?)(?![0-9])(?<gap>\s+(?<next>\p{L}+))?"#
      ) {
        numberWords(
          $0["n"], style: .portuguese, spelling: spelling,
          feminine: $0.optional("next").map { isFeminineNoun($0) } ?? false)
          + String($0.optional("gap") ?? "")
      }
  }

  private static func english(_ text: String) -> String {
    let spelling = Spelling("en_US")
    let symbols =
      text
      .replacing(pattern: #" *°C"#, with: " degrees Celsius")
      .replacing(pattern: #" *& *"#, with: " and ")
      .replacing(pattern: #"§ *"#, with: "section ")
      .replacing(pattern: #"(?<=[0-9]) *\+ *(?=[0-9])"#, with: " plus ")
      .replacing(pattern: #"(?<=[0-9]) *= *(?=[0-9])"#, with: " equals ")
      .replacing(pattern: #"(?<=[0-9]) *× *(?=[0-9])"#, with: " times ")
      .replacing(pattern: #"(?<!\p{L})pp\.(?=\s*[0-9])"#, with: "pages")
      .replacing(pattern: #"(?<!\p{L})p\.(?=\s*[0-9])"#, with: "page")
      .replacing(pattern: #"(?<!\p{L})Ch\.(?=\s*[0-9])"#, with: "Chapter")
      .replacing(pattern: #"(?<!\p{L})No\.(?=\s*[0-9])"#, with: "number")
    let abbreviated = expandAbbreviations(symbols, englishAbbreviations, terminal: ["etc"])
    return englishNumbers(abbreviated, spelling)
  }

  private static func englishNumbers(_ text: String, _ spelling: Spelling) -> String {
    text
      .replacing(
        pattern:
          #"(?<!\p{L})(?<kw>(?i:chapter|part|book|volume|act|scene))\s+(?<num>[IVXLCDM]+)(?!\p{L})"#
      ) {
        romanSpoken($0["kw"], $0["num"], spelling)
      }
      .replacing(pattern: #"(?<![0-9.,])(?<n>[0-9]+)(?:st|nd|rd|th)(?!\p{L})"#) {
        englishOrdinal($0["n"], spelling) ?? String($0.whole)
      }
      .replacing(
        pattern:
          #"(?<!\p{L})(?:US)?\$ ?(?<whole>[0-9]+(?:,[0-9]{3})*)(?:\.(?<cents>[0-9]{2}))?(?![0-9])"#
      ) {
        englishMoney($0["whole"], cents: $0.optional("cents"), spelling) ?? String($0.whole)
      }
      .replacing(pattern: #"(?<![0-9.,])(?<n>[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?) ?%"#) {
        numberWords($0["n"], style: .english, spelling: spelling) + " percent"
      }
      .replacing(pattern: #"(?<![0-9])(?<a>[0-9]{4}) ?[–-] ?(?<b>[0-9]{4})(?![0-9])"#) {
        englishNumber($0["a"], spelling) + " to " + englishNumber($0["b"], spelling)
      }
      .replacing(pattern: #"(?<![\p{L}0-9.,])(?<n>[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)(?![0-9])"#) {
        englishNumber($0["n"], spelling)
      }
  }

  /// Expands dotted abbreviations. The period is kept at the end of the text, and after
  /// `terminal` abbreviations followed by a capital letter (they may end a sentence).
  private static func expandAbbreviations(
    _ text: String, _ table: [String: String], terminal: Set<String>
  ) -> String {
    text
      .replacing(pattern: #"(?<!\p{L})(?<a>\p{L}+(?:\.\p{L}+)*)\.(?=\s*$)"#) {
        expand($0["a"], table, keepPeriod: true)
      }
      .replacing(pattern: #"(?<!\p{L})(?<a>\p{L}+(?:\.\p{L}+)*)\.(?=\s+\p{Lu})"#) {
        expand($0["a"], table, keepPeriod: terminal.contains(String($0["a"])))
      }
      .replacing(pattern: #"(?<!\p{L})(?<a>\p{L}+(?:\.\p{L}+)*)\."#) {
        expand($0["a"], table, keepPeriod: false)
      }
  }

  private static func expand(
    _ word: Substring, _ table: [String: String], keepPeriod: Bool
  ) -> String {
    guard let expansion = table[String(word)] else { return "\(word)." }
    return keepPeriod ? expansion + "." : expansion
  }

  private static func spokenURL(_ url: Substring, dot: String) -> String {
    var core = String(url)
    var trailing = ""
    while let last = core.last, ".,;:!?)\"'".contains(last) {
      trailing.insert(last, at: trailing.startIndex)
      core.removeLast()
    }
    let host =
      core
      .replacing(pattern: #"^(?:https?:\/\/)?(?:www\.)?"#, with: "")
      .prefix { !"/?#".contains($0) }
    return host.split(separator: ".").joined(separator: dot) + trailing
  }

  private static func romanSpoken(_ keyword: Substring, _ numeral: Substring, _ spelling: Spelling)
    -> String
  {
    guard let value = romanValue(numeral), (1...4000).contains(value) else {
      return "\(keyword) \(numeral)"
    }
    return "\(keyword) \(spelling.words(value))"
  }

  private static func romanValue(_ numeral: Substring) -> Int? {
    let digits: [Character: Int] = [
      "I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000,
    ]
    var total = 0
    var previous = 0
    for character in numeral.reversed() {
      guard let value = digits[character] else { return nil }
      total += value < previous ? -value : value
      previous = value
    }
    return total
  }

  /// Spells a number in `style`. Decimals read the fraction digit by digit. Returns the numeral
  /// unchanged when it does not fit an `Int`.
  private static func numberWords(
    _ numeral: Substring, style: NumberStyle, spelling: Spelling, feminine: Bool = false
  ) -> String {
    let parts = numeral.split(
      separator: style.decimal, maxSplits: 1, omittingEmptySubsequences: false)
    guard let whole = parts.first, let integer = Int(whole.filter({ $0 != style.thousands })) else {
      return String(numeral)
    }
    guard parts.count == 2 else {
      let words = spelling.words(integer)
      return feminine ? feminineForm(words) : words
    }
    let digits = parts[1].compactMap { $0.wholeNumberValue }.map { spelling.words($0) }
    return ([spelling.words(integer), style.point] + digits).joined(separator: " ")
  }

  /// The feminine form of a spelled cardinal: "um" → "uma", "dois" → "duas", hundreds in "-tos" → "-tas".
  private static func feminineForm(_ words: String) -> String {
    var words = words
    if words.hasPrefix("dois mil") {
      words = "duas mil" + words.dropFirst(8)
    }
    for (masculine, feminine) in [("um", "uma"), ("dois", "duas"), ("tos", "tas")]
    where words.hasSuffix(masculine) {
      return String(words.dropLast(masculine.count)) + feminine
    }
    return words
  }

  /// Whether the word after a number makes the number feminine: a noun ending in "a"/"as" that is
  /// not in the masculine list.
  private static func isFeminineNoun(_ word: Substring) -> Bool {
    let lowercased = word.lowercased()
    guard lowercased.count >= 2, lowercased.hasSuffix("a") || lowercased.hasSuffix("as") else {
      return false
    }
    return !masculineNouns.contains(lowercased)
  }

  private static func portugueseOrdinal(_ digits: Substring, feminine: Bool) -> String? {
    guard let value = Int(digits), (1...999).contains(value) else { return nil }
    let hundred = value / 100
    let ten = value % 100 / 10
    let unit = value % 10
    var parts: [String] = []
    if hundred > 0 { parts.append(ordinalHundreds[hundred]) }
    if ten == 1 {
      parts.append("décimo")
    } else if ten > 1 {
      parts.append(ordinalTens[ten])
    }
    if unit > 0 { parts.append(ordinalUnits[unit]) }
    let words = parts.joined(separator: " ")
    return feminine ? feminineOrdinal(words) : words
  }

  private static func feminineOrdinal(_ words: String) -> String {
    words.split(separator: " ")
      .map { $0.hasSuffix("o") ? String($0.dropLast()) + "a" : String($0) }
      .joined(separator: " ")
  }

  private static func portugueseDate(
    _ day: Substring, _ month: Substring, _ year: Substring, _ spelling: Spelling
  ) -> String? {
    guard let d = Int(day), let m = Int(month), let y = Int(year),
      (1...31).contains(d), (1...12).contains(m)
    else { return nil }
    let dayWords = d == 1 ? "primeiro" : spelling.words(d)
    return "\(dayWords) de \(portugueseMonths[m - 1]) de \(spelling.words(y))"
  }

  private static func portugueseTime(
    _ hour: Substring, minutes: Substring?, _ spelling: Spelling
  ) -> String? {
    guard let h = Int(hour) else { return nil }
    let hours = feminineForm(spelling.words(h))
    let m = minutes.flatMap { Int($0) } ?? 0
    if m > 0 { return "\(hours) e \(spelling.words(m))" }
    return h == 1 ? "uma hora" : "\(hours) horas"
  }

  private static func portugueseMoney(
    _ code: Substring, whole: Substring, cents: Substring?, _ spelling: Spelling
  ) -> String? {
    guard let value = Int(whole.filter({ $0 != "." })) else { return nil }
    let unit: String
    if code == "R" {
      unit = value == 1 ? "real" : "reais"
    } else {
      unit = value == 1 ? "dólar" : "dólares"
    }
    var phrase = "\(spelling.words(value)) \(unit)"
    if let cents, let value = Int(cents), value > 0 {
      phrase += " e \(spelling.words(value)) \(value == 1 ? "centavo" : "centavos")"
    }
    return phrase
  }

  private static func englishNumber(_ numeral: Substring, _ spelling: Spelling) -> String {
    if numeral.count == 4, let year = Int(numeral), let reading = englishYear(year, spelling) {
      return reading
    }
    return numberWords(numeral, style: .english, spelling: spelling)
  }

  /// Four-digit years in pairs: "nineteen eighty-four", "nineteen oh five", "twenty twenty-four".
  private static func englishYear(_ year: Int, _ spelling: Spelling) -> String? {
    let rest = year % 100
    switch year {
    case 1100...1999:
      let pair =
        rest == 0 ? "hundred" : rest < 10 ? "oh \(spelling.words(rest))" : spelling.words(rest)
      return "\(spelling.words(year / 100)) \(pair)"
    case 2000...2009:
      return rest == 0 ? "two thousand" : "two thousand \(spelling.words(rest))"
    case 2010...2099:
      return "twenty \(spelling.words(rest))"
    default:
      return nil
    }
  }

  private static func englishOrdinal(_ digits: Substring, _ spelling: Spelling) -> String? {
    guard let value = Int(digits) else { return nil }
    let cardinal = spelling.words(value)
    guard let cut = cardinal.lastIndex(where: { $0 == " " || $0 == "-" }) else {
      return ordinalWord(cardinal)
    }
    return String(cardinal[...cut]) + ordinalWord(String(cardinal[cardinal.index(after: cut)...]))
  }

  private static func ordinalWord(_ word: String) -> String {
    return switch word {
    case "one": "first"
    case "two": "second"
    case "three": "third"
    case "five": "fifth"
    case "eight": "eighth"
    case "nine": "ninth"
    case "twelve": "twelfth"
    default: word.hasSuffix("y") ? String(word.dropLast()) + "ieth" : word + "th"
    }
  }

  private static func englishMoney(_ whole: Substring, cents: Substring?, _ spelling: Spelling)
    -> String?
  {
    guard let dollars = Int(whole.filter({ $0 != "," })) else { return nil }
    var phrase = "\(spelling.words(dollars)) \(dollars == 1 ? "dollar" : "dollars")"
    if let cents, let value = Int(cents), value > 0 {
      phrase += " and \(spelling.words(value)) \(value == 1 ? "cent" : "cents")"
    }
    return phrase
  }

  private static let portugueseAbbreviations: [String: String] = [
    "Sr": "Senhor", "Sra": "Senhora", "Srta": "Senhorita", "Dr": "Doutor", "Dra": "Doutora",
    "Prof": "Professor", "Profa": "Professora", "Exmo": "Excelentíssimo", "Exma": "Excelentíssima",
    "Sto": "Santo", "Sta": "Santa", "Av": "Avenida", "vol": "volume", "ed": "edição",
    "etc": "etcétera", "fig": "figura", "Fig": "figura", "tab": "tabela", "Tab": "tabela",
    "séc": "século", "a.C": "antes de Cristo", "d.C": "depois de Cristo", "obs": "observação",
    "ex": "exemplo",
  ]

  private static let englishAbbreviations: [String: String] = [
    "Mr": "Mister", "Mrs": "Missus", "Ms": "Miz", "Dr": "Doctor", "Prof": "Professor",
    "vs": "versus", "etc": "et cetera", "Fig": "Figure", "Eq": "Equation", "Sec": "Section",
    "Vol": "Volume", "approx": "approximately",
  ]

  private static let portugueseMonths = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
  ]

  private static let ordinalUnits = [
    "", "primeiro", "segundo", "terceiro", "quarto", "quinto", "sexto", "sétimo", "oitavo", "nono",
  ]

  private static let ordinalTens = [
    "", "décimo", "vigésimo", "trigésimo", "quadragésimo", "quinquagésimo", "sexagésimo",
    "septuagésimo", "octogésimo", "nonagésimo",
  ]

  private static let ordinalHundreds = [
    "", "centésimo", "ducentésimo", "trecentésimo", "quadringentésimo", "quingentésimo",
    "sexcentésimo", "septingentésimo", "octingentésimo", "nongentésimo",
  ]

  private static let masculineNouns: Set<String> = [
    "dia", "dias", "mapa", "mapas", "problema", "problemas", "sistema", "sistemas", "programa",
    "programas", "tema", "temas", "planeta", "planetas", "clima", "idioma", "idiomas", "poema",
    "poemas", "drama", "dramas", "telefonema", "cinema",
  ]
}

/// A match of an ICU regular expression, with its named groups.
private struct Match {
  let whole: Substring
  private let text: String
  private let result: NSTextCheckingResult

  init(_ result: NSTextCheckingResult, in text: String) {
    self.text = text
    self.result = result
    whole = Range(result.range, in: text).map { text[$0] } ?? ""
  }

  /// The text of group `name`, or nil when the group did not take part in the match.
  func optional(_ name: String) -> Substring? {
    Range(result.range(withName: name), in: text).map { text[$0] }
  }

  subscript(_ name: String) -> Substring { optional(name) ?? "" }
}

extension String {
  /// Replaces the matches of an ICU `pattern`; unlike Swift Regex, ICU supports lookbehind.
  fileprivate func replacing(pattern: String, _ transform: (Match) -> String) -> String {
    let regex: NSRegularExpression
    do {
      regex = try NSRegularExpression(pattern: pattern)
    } catch {
      preconditionFailure("invalid pattern \(pattern): \(error)")
    }
    var result = ""
    var last = startIndex
    for match in regex.matches(in: self, range: NSRange(startIndex..., in: self)) {
      guard let range = Range(match.range, in: self) else { continue }
      result += self[last..<range.lowerBound]
      result += transform(Match(match, in: self))
      last = range.upperBound
    }
    return result + self[last...]
  }

  fileprivate func replacing(pattern: String, with replacement: String) -> String {
    replacing(pattern: pattern) { _ in replacement }
  }
}

/// Spells numbers with a `NumberFormatter`. One instance is made per `SpeechText.normalized` call.
private struct Spelling {
  private let formatter = NumberFormatter()

  init(_ identifier: String) {
    formatter.locale = Locale(identifier: identifier)
    formatter.numberStyle = .spellOut
  }

  func words(_ number: Int) -> String {
    formatter.string(from: NSNumber(value: number)) ?? String(number)
  }
}

/// How a language writes numerals: separators and the word for the decimal point.
private struct NumberStyle {
  let thousands: Character
  let decimal: Character
  let point: String

  static let portuguese = NumberStyle(thousands: ".", decimal: ",", point: "vírgula")
  static let english = NumberStyle(thousands: ",", decimal: ".", point: "point")
}
