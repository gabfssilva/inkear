import NaturalLanguage

extension Block {
  /// The ranges of the block's sentences in `text`, in order, trimmed of surrounding whitespace.
  /// Uses `NLTokenizer(unit: .sentence)`. A text without sentence punctuation is one sentence.
  /// Empty or whitespace-only text has no sentences.
  public var sentences: [Range<String.Index>] {
    let tokenizer = NLTokenizer(unit: .sentence)
    tokenizer.string = text
    return tokenizer.tokens(for: text.startIndex..<text.endIndex).compactMap {
      trimmed($0, in: text)
    }
  }
}

extension DocumentStructure {
  /// ISO 639-1 code ("pt", "en", …) of the dominant language of the paragraphs, from up to
  /// ~4000 characters of paragraph text starting at `mainStart` (or the first block), via
  /// `NLLanguageRecognizer`; nil when undetermined.
  public var language: String? {
    var sample = ""
    for block in blocks.dropFirst(mainStart ?? 0) where block.kind == .paragraph {
      sample += block.text + "\n"
      if sample.count >= 4000 { break }
    }
    let recognizer = NLLanguageRecognizer()
    recognizer.processString(String(sample.prefix(4000)))
    return recognizer.dominantLanguage.flatMap { $0 == .undetermined ? nil : $0.rawValue }
  }
}

/// `range` without leading and trailing whitespace, or nil when it has none left.
private func trimmed(_ range: Range<String.Index>, in text: String) -> Range<String.Index>? {
  let content = text[range]
  guard let first = content.firstIndex(where: { !$0.isWhitespace }),
    let last = content.lastIndex(where: { !$0.isWhitespace })
  else { return nil }
  return first..<text.index(after: last)
}
