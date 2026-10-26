import Foundation

/// The text front end of one language: text to phoneme symbols, then to the model's token ids.
///
/// Words come from the lexicon, or else are spelled out grapheme by grapheme. Symbols the model
/// has no token for are dropped.
struct MagpieFrontend {
  private let language: String
  private let ids: [String: Int]
  private let offset: Int
  private let eos: Int
  private let space: String
  private let padWithSpace: Bool
  private let prefix: String
  private let pattern: NSRegularExpression
  private let lowercaseWords: Bool
  private let lexicon: [String: String]
  private let heteronyms: Set<String>

  /// The fields of `tokenizer_<language>.json` the front end reads.
  private struct Config: Decodable {
    let offset: Int
    let eos: Int
    let tokens: [String]
    let space: String
    let padWithSpace: Bool
    let graphemePrefix: String
    let wordPattern: String
    let lowercaseWords: Bool
  }

  init(directory: URL, language: String) throws {
    let config = try JSONDecoder.snakeCase.decode(
      Config.self, from: Data(contentsOf: directory.appending(path: "tokenizer_\(language).json")))
    let lines = try String(
      contentsOf: directory.appending(path: "lexicon_\(language).tsv"), encoding: .utf8)
    let heteronymFile = directory.appending(path: "heteronyms_\(language).txt")
    let heteronymList = try FileManager.default.fileExists(atPath: heteronymFile.path(percentEncoded: false))
      ? String(contentsOf: heteronymFile, encoding: .utf8) : ""

    self.language = language
    offset = config.offset
    eos = config.eos
    ids = Dictionary(
      config.tokens.enumerated().map { ($1, $0) }, uniquingKeysWith: { _, last in last })
    space = config.space
    padWithSpace = config.padWithSpace
    prefix = config.graphemePrefix
    pattern = try NSRegularExpression(pattern: config.wordPattern)
    lowercaseWords = config.lowercaseWords
    lexicon = Dictionary(
      lines.split(whereSeparator: \.isNewline).compactMap { line -> (String, String)? in
        let parts = line.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2 else { return nil }
        return (String(parts[0]), String(parts[1]))
      }, uniquingKeysWith: { _, last in last })
    heteronyms = Set(heteronymList.split(whereSeparator: \.isWhitespace).map(String.init))
  }

  /// The token ids of `text`, ending with the end-of-text id.
  func encode(_ text: String) -> [Int] {
    let text = normalized(text)
    var pieces: [String] = []
    for match in pattern.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
      let word = Self.group(match, 1, in: text)
      let unchanged = Self.group(match, 2, in: text)
      if !unchanged.isEmpty {
        pieces += unchanged.dropFirst().dropLast()
          .split(separator: " ", omittingEmptySubsequences: false)
          .map { prefix + String($0) }
      } else {
        let run =
          word.isEmpty
          ? Self.group(match, 3, in: text) : (lowercaseWords ? word.lowercased() : word)
        pieces += symbols(for: run)
      }
    }
    var kept = pieces.filter { ids[$0] != nil }
    while kept.last == space { kept.removeLast() }
    if padWithSpace { kept = [space] + kept + [space] }
    return kept.compactMap { ids[$0] }.map { $0 + offset } + [eos]
  }

  private func normalized(_ text: String) -> String {
    guard language == "en" else {
      return text.precomposedStringWithCanonicalMapping.replacingOccurrences(of: "’", with: "'")
    }
    let unaccented = String(
      String.UnicodeScalarView(
        text.decomposedStringWithCanonicalMapping.unicodeScalars.filter {
          $0.properties.generalCategory != .nonspacingMark
        }))
    return unaccented
      .replacingOccurrences(of: "’", with: "'")
      .replacingOccurrences(of: "“", with: "\"")
      .replacingOccurrences(of: "”", with: "\"")
      .precomposedStringWithCanonicalMapping
  }

  /// The symbols of one run of text, uppercased: its characters when it has no letter or digit.
  private func symbols(for run: String) -> [String] {
    let word = run.uppercased()
    if !word.contains(where: { $0.isLetter || $0.isNumber }) {
      return Self.scalars(word)
    }
    if heteronyms.contains(word) { return graphemes(word) }
    if let found = lookup(word) { return found }
    let parts = word.split(separator: "-", omittingEmptySubsequences: false)
    if parts.count == 1 { return graphemes(word) }
    var joined: [String] = []
    for part in parts {
      joined += symbols(for: String(part)) + ["-"]
    }
    return Array(joined.dropLast())
  }

  /// The lexicon pronunciation of `word`. English also takes the possessive and plural of a word
  /// whose stem is in the lexicon, with NeMo's en-US endings.
  private func lookup(_ word: String) -> [String]? {
    if let pronunciation = lexicon[word] { return Self.scalars(pronunciation) }
    guard language == "en" else { return nil }
    for suffix in ["'S", "S"] where word.count > suffix.count && word.hasSuffix(suffix) {
      let stem = String(word.dropLast(suffix.count))
      guard let pronunciation = lexicon[stem] else { continue }
      let ending: [String]
      if stem.hasSuffix("T") {
        ending = ["s"]
      } else if suffix == "'S" && stem.hasSuffix("S") {
        ending = ["ɪ", "z"]
      } else {
        ending = ["z"]
      }
      return Self.scalars(pronunciation) + ending
    }
    return nil
  }

  private func graphemes(_ word: String) -> [String] {
    Self.scalars(word).map { prefix + $0 }
  }

  private static func scalars(_ text: String) -> [String] {
    text.unicodeScalars.map { String($0) }
  }

  private static func group(
    _ match: NSTextCheckingResult, _ index: Int, in text: String
  ) -> String {
    Range(match.range(at: index), in: text).map { String(text[$0]) } ?? ""
  }
}

extension JSONDecoder {
  /// A decoder that reads the snake_case keys of the model's JSON files as camelCase properties.
  static var snakeCase: JSONDecoder {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
  }
}
