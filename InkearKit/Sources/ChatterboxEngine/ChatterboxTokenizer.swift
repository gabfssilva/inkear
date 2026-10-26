// Adapted from FluidAudio's Chatterbox tokenizer, Apache-2.0.
#if canImport(CoreAI)
  import Foundation

  /// The BPE tokenizer of `tokenizer.json` as the `tokenizers` library runs it for this pack: added
  /// tokens are matched first, and the rest is split into words and merged by rank. The pack has no
  /// normalizer, so the text is not lowercased or decomposed.
  struct ChatterboxTokenizer: Sendable {
    private struct File: Decodable {
      struct Model: Decodable {
        let vocab: [String: Int]
        let merges: [String]
        let unkToken: String
      }

      let model: Model
      let addedTokens: [AddedToken]
    }

    private struct AddedToken: Decodable, Sendable {
      let content: String
      let id: Int
    }

    private let vocab: [String: Int]
    /// Longest first, so that a longer token wins where two start at the same place.
    private let addedTokens: [AddedToken]
    /// Ranks of the "left right" merges, lowest first.
    private let mergeRanks: [String: Int]
    private let unknown: Int

    init(url: URL) throws {
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      let file = try decoder.decode(File.self, from: Data(contentsOf: url))
      guard let unknown = file.model.vocab[file.model.unkToken] else {
        throw ChatterboxError.invalidConfiguration("tokenizer unknown token")
      }
      vocab = file.model.vocab
      addedTokens = file.addedTokens.sorted { $0.content.count > $1.content.count }
      mergeRanks = Dictionary(
        file.model.merges.enumerated().map { ($0.element, $0.offset) },
        uniquingKeysWith: { first, _ in first })
      self.unknown = unknown
    }

    /// The pack's `punc_norm`: first letter capitalized, whitespace collapsed, punctuation mapped
    /// to the model's set, and a final stop unless the text ends with one.
    static func normalize(_ text: String) -> String {
      if text.isEmpty { return "You need to add some text for me to talk." }
      var text = text
      if let first = text.first, first.isLowercase {
        text = first.uppercased() + text.dropFirst()
      }
      text = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
      let replacements = [
        ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "), ("—", "-"),
        ("–", "-"), (" ,", ","), ("\u{201C}", "\""), ("\u{201D}", "\""), ("\u{2018}", "'"),
        ("\u{2019}", "'"),
      ]
      for (old, new) in replacements {
        text = text.replacingOccurrences(of: old, with: new)
      }
      while text.hasSuffix(" ") { text.removeLast() }
      let enders: Set<Character> = [".", "!", "?", "-", ","]
      if !(text.last.map { enders.contains($0) } ?? false) { text += "." }
      return text
    }

    /// The ids of `text`, with each added token as one id and the rest as merged words.
    func encode(_ text: String) -> [Int] {
      var ids: [Int] = []
      var plain = ""
      var index = text.startIndex
      while index < text.endIndex {
        let rest = text[index...]
        if let token = addedTokens.first(where: { rest.hasPrefix($0.content) }) {
          ids += Self.words(plain).flatMap { bpe($0) }
          plain = ""
          ids.append(token.id)
          index = text.index(index, offsetBy: token.content.count)
        } else {
          plain.append(text[index])
          index = text.index(after: index)
        }
      }
      return ids + Self.words(plain).flatMap { bpe($0) }
    }

    /// The `\w+|[^\w\s]+` pieces of `text`, as the Whitespace pre-tokenizer splits them.
    private static func words(_ text: String) -> [String] {
      var words: [String] = []
      var word = ""
      var kind: Bool?
      for scalar in text.unicodeScalars {
        let next: Bool? = scalar.properties.isWhitespace ? nil : isWordScalar(scalar)
        if next == nil || next != kind {
          if !word.isEmpty { words.append(word) }
          word = ""
        }
        if next != nil { word.unicodeScalars.append(scalar) }
        kind = next
      }
      if !word.isEmpty { words.append(word) }
      return words
    }

    /// Unicode's `\w`: alphabetic, marks, decimal digits, connector punctuation and joiners.
    private static func isWordScalar(_ scalar: Unicode.Scalar) -> Bool {
      let properties = scalar.properties
      if properties.isAlphabetic { return true }
      switch properties.generalCategory {
      case .nonspacingMark, .spacingMark, .enclosingMark, .decimalNumber, .connectorPunctuation:
        return true
      default:
        return scalar == "\u{200C}" || scalar == "\u{200D}"
      }
    }

    /// Character-level BPE over the scalars of `word`; the vocabulary is per scalar, not per
    /// grapheme cluster.
    private func bpe(_ word: String) -> [Int] {
      var parts = word.unicodeScalars.map { String($0) }
      while parts.count > 1 {
        var best: (rank: Int, index: Int)?
        for index in 0..<(parts.count - 1) {
          guard let rank = mergeRanks[parts[index] + " " + parts[index + 1]] else { continue }
          if rank < (best?.rank ?? Int.max) { best = (rank, index) }
        }
        guard let merge = best else { break }
        parts[merge.index] += parts[merge.index + 1]
        parts.remove(at: merge.index + 1)
      }
      return parts.map { vocab[$0] ?? unknown }
    }
  }
#endif
