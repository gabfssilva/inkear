// Adapted from speech-swift/Sources/AudioCommon/Tokenizer.swift, Apache-2.0.
import Foundation

/// The Qwen2 byte-level BPE, encoding only: `vocab.json` maps symbols to ids and `merges.txt`
/// ranks the merges.
struct Qwen3Tokenizer {
  private let ids: [String: Int]
  private let ranks: [String: Int]

  init(vocabulary: URL, merges: URL) throws {
    ids = try JSONDecoder().decode([String: Int].self, from: Data(contentsOf: vocabulary))
    var ranks: [String: Int] = [:]
    for line in try String(contentsOf: merges, encoding: .utf8).split(separator: "\n")
    where !line.hasPrefix("#") {
      ranks[String(line)] = ranks.count
    }
    self.ranks = ranks
  }

  /// The token ids of `text` in Unicode NFC, split with the Qwen2 pre-tokenizer pattern before BPE.
  func encode(_ text: String) -> [Int] {
    let composed = text.precomposedStringWithCanonicalMapping
    // The Qwen2 pre-tokenizer pattern: contractions, letter runs, digits, punctuation, spaces.
    let pieces = composed.matches(of: /(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/)
    return pieces.flatMap { match in
      bpe(String(match.output).utf8.map { Self.byteSymbols[Int($0)] }).compactMap { ids[$0] }
    }
  }

  private func bpe(_ symbols: [String]) -> [String] {
    var word = symbols
    while true {
      let merges = word.indices.dropLast().compactMap { index in
        ranks["\(word[index]) \(word[index + 1])"].map { (index, $0) }
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
