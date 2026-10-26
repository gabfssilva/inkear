import Foundation

/// The Qwen2 byte-level BPE of `tokenizer.json`, encoding only. Special tokens are not recognized;
/// the caller adds their ids.
struct OmniVoiceTokenizer: Sendable {
  private struct Pair: Hashable, Decodable {
    let left: String
    let right: String

    init(left: String, right: String) {
      self.left = left
      self.right = right
    }

    init(from decoder: Decoder) throws {
      var values = try decoder.unkeyedContainer()
      left = try values.decode(String.self)
      right = try values.decode(String.self)
    }
  }

  private struct File: Decodable {
    struct Model: Decodable {
      let vocab: [String: Int]
      let merges: [Pair]
    }

    let model: Model
  }

  private let ids: [String: Int]
  private let ranks: [Pair: Int]

  init(url: URL) throws {
    let file = try JSONDecoder().decode(File.self, from: Data(contentsOf: url))
    ids = file.model.vocab
    var ranks: [Pair: Int] = [:]
    for (rank, pair) in file.model.merges.enumerated() {
      ranks[pair] = rank
    }
    self.ranks = ranks
  }

  /// The token ids of `text` in Unicode NFC, split with the Qwen2 pre-tokenizer pattern before BPE.
  func encode(_ text: String) -> [Int32] {
    let composed = text.precomposedStringWithCanonicalMapping
    let pieces = composed.matches(of: /(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/)
    return pieces.flatMap { match in
      bpe(String(match.output).utf8.map { Self.byteSymbols[Int($0)] }).compactMap { ids[$0] }
    }.map { Int32($0) }
  }

  private func bpe(_ symbols: [String]) -> [String] {
    var word = symbols
    while true {
      let merges = word.indices.dropLast().compactMap { index in
        ranks[Pair(left: word[index], right: word[index + 1])].map { (index, $0) }
      }
      guard let best = merges.min(by: { $0.1 < $1.1 }) else { return word }
      let index = best.0
      let merged = word[index] + word[index + 1]
      word.replaceSubrange(index...(index + 1), with: [merged])
    }
  }

  /// The GPT-2 byte-to-symbol table: printable bytes stand for themselves, the others for code
  /// points from U+0100 up.
  private static let byteSymbols: [String] = {
    let printable = Set(Array(33...126) + Array(161...172) + Array(174...255))
    var next: UInt32 = 256
    return (0..<256).map { byte -> String in
      if printable.contains(byte) { return String(Character(Unicode.Scalar(UInt8(byte)))) }
      defer { next += 1 }
      return Unicode.Scalar(next).map { String(Character($0)) } ?? ""
    }
  }()
}
