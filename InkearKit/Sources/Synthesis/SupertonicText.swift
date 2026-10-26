import Foundation

/// Text preprocessing for Supertonic, ported from the upstream reference (`py/helper.py`).
///
/// Everything works on code points, as Python's `re` does upstream: after NFKD normalization,
/// accents are code points of their own, and Swift would otherwise group them into characters.
enum SupertonicText {
  static let languages: Set<String> = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
    "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
  ]

  /// The model's character ids for `text`, dropping characters outside the vocabulary.
  static func ids(for text: String, language: String, indexer: [Int32]) -> [Int32] {
    preprocess(text, language: language).unicodeScalars.compactMap { scalar in
      let index = Int(scalar.value)
      return index < indexer.count && indexer[index] >= 0 ? indexer[index] : nil
    }
  }

  /// Normalizes `text` the way the model was trained on, and wraps it in language tags.
  static func preprocess(_ text: String, language: String) -> String {
    let substituted = text.decomposedStringWithCompatibilityMapping.unicodeScalars
      .filter { !isEmoji($0) }
      .map { substitutions[$0] ?? String($0) }
      .joined()
      .replacing("e.g.,", with: "for example, ")
      .replacing("i.e.,", with: "that is, ")
    var text =
      substituted
      .replacing(/\ ([,.!?;:'])/.matchingSemantics(.unicodeScalar)) { String($0.output.1) }
      .replacing(/"{2,}/.matchingSemantics(.unicodeScalar), with: "\"")
      .replacing(/'{2,}/.matchingSemantics(.unicodeScalar), with: "'")
      .replacing(/\s+/.matchingSemantics(.unicodeScalar), with: " ")
      .trimmingCharacters(in: .whitespaces)
    if text.firstMatch(of: /[.!?;:,'"')\]}…。」』】〉》›»]$/.matchingSemantics(.unicodeScalar)) == nil {
      text += "."
    }
    return "<\(language)>\(text)</\(language)>"
  }

  private static let substitutions: [Unicode.Scalar: String] = Dictionary(
    uniqueKeysWithValues: [
      ("–‑—", "-"), ("\u{201C}\u{201D}", "\""), ("\u{2018}\u{2019}´`", "'"), ("_[]|/#→←", " "),
      ("♥☆♡©\\", ""), ("@", " at "),
    ].flatMap { scalars, replacement in scalars.unicodeScalars.map { ($0, replacement) } }
  )

  private static func isEmoji(_ scalar: Unicode.Scalar) -> Bool {
    emoji.contains { $0.contains(scalar.value) }
  }

  private static let emoji: [ClosedRange<UInt32>] = [
    0x1F600...0x1F64F, 0x1F300...0x1F5FF, 0x1F680...0x1F6FF, 0x1F700...0x1F77F, 0x1F780...0x1F7FF,
    0x1F800...0x1F8FF, 0x1F900...0x1F9FF, 0x1FA00...0x1FA6F, 0x1FA70...0x1FAFF, 0x2600...0x26FF,
    0x2700...0x27BF, 0x1F1E6...0x1F1FF,
  ]
}
