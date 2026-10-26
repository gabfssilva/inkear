import NaturalLanguage
import PDFStructure

/// A document as the sequence of sentences read aloud, with navigation.
public struct Script: Sendable {
  public let structure: DocumentStructure
  /// The text each block is narrated as, one entry per block.
  private let narrated: [String]
  /// Sentence ranges per block, in each block's narrated text.
  private let ranges: [[Range<String.Index>]]
  /// Every sentence in reading order.
  private let positions: [ReadingPosition]
  /// Per block, the index in `positions` of its first sentence; count is the block count plus one.
  private let blockStarts: [Int]
  /// Characters before each sentence, and in total at the end; count is `positions.count + 1`.
  private let charactersBefore: [Int]
  /// Start blocks of the sections, ascending.
  private let sectionStarts: [Int]
  /// Indices into `structure.sections`, in the order of `sectionStarts`.
  private let sectionOrder: [Int]

  /// Speaking rate at speed 1, used to estimate durations.
  public static let charactersPerSecond: Double = 14.5

  public init(_ structure: DocumentStructure) {
    self.structure = structure
    let narrated = structure.blocks.map(Self.narrate)
    var ranges: [[Range<String.Index>]] = []
    var positions: [ReadingPosition] = []
    var blockStarts = [0]
    var charactersBefore = [0]
    for (index, block) in structure.blocks.enumerated() {
      let text = narrated[index]
      let sentences = Self.sentences(of: text)
      ranges.append(sentences)
      for (sentence, range) in sentences.enumerated() {
        positions.append(ReadingPosition(block: index, sentence: sentence))
        charactersBefore.append(
          charactersBefore[charactersBefore.count - 1] + text[range].count)
      }
      blockStarts.append(positions.count)
    }
    self.narrated = narrated
    let sectionOrder = structure.sections.indices.sorted { a, b in
      (structure.sections[a].block, a) < (structure.sections[b].block, b)
    }
    self.ranges = ranges
    self.positions = positions
    self.blockStarts = blockStarts
    self.charactersBefore = charactersBefore
    self.sectionOrder = sectionOrder
    self.sectionStarts = sectionOrder.map { structure.sections[$0].block }
  }

  /// The first sentence at or after `mainStart`, else the first sentence.
  public var start: ReadingPosition {
    let mainIndex = structure.mainStart.map { sentenceIndex(atOrAfterBlock: $0) } ?? 0
    return positions.indices.contains(mainIndex)
      ? positions[mainIndex] : (positions.first ?? .start)
  }

  public var isEmpty: Bool { positions.isEmpty }

  public func sentences(inBlock block: Int) -> [Range<String.Index>] {
    ranges.indices.contains(block) ? ranges[block] : []
  }

  public func text(at position: ReadingPosition) -> String? {
    guard ranges.indices.contains(position.block),
      ranges[position.block].indices.contains(position.sentence)
    else { return nil }
    return String(narrated[position.block][ranges[position.block][position.sentence]])
  }

  /// Whether `block` is a heading, which the narrator reads with a longer pause around it.
  public func isHeading(_ block: Int) -> Bool {
    guard structure.blocks.indices.contains(block) else { return false }
    if case .heading = structure.blocks[block].kind { return true }
    return false
  }

  /// The text `block` is narrated as.
  ///
  /// A heading reads as separate sentences, so "Capítulo 1 — O menino que sentia o chão"
  /// becomes "Capítulo 1. O menino que sentia o chão." and the narrator pauses between them.
  static func narrate(_ block: Block) -> String {
    guard case .heading = block.kind else { return block.text }
    let text = block.text
      .replacingOccurrences(of: " — ", with: ". ")
      .replacingOccurrences(of: " – ", with: ". ")
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard let last = text.last, !".,!?……".contains(last) else { return text }
    return text + "."
  }

  /// The first sentence after `position`, skipping blocks without sentences.
  public func next(after position: ReadingPosition) -> ReadingPosition? {
    let index = sentenceIndex(after: position)
    return positions.indices.contains(index) ? positions[index] : nil
  }

  /// The sentences from `position` on, in its block, whose characters add up to at most
  /// `characters`; always at least the sentence at `position`.
  public func run(from position: ReadingPosition, characters: Int) -> [ReadingPosition] {
    var index = sentenceIndex(atOrAfter: position)
    guard positions.indices.contains(index) else { return [] }
    let first = index
    while index + 1 < positions.count, positions[index + 1].block == positions[first].block,
      charactersBefore[index + 2] - charactersBefore[first] <= characters
    {
      index += 1
    }
    return Array(positions[first...index])
  }

  /// The last sentence before `position`, skipping blocks without sentences.
  public func previous(before position: ReadingPosition) -> ReadingPosition? {
    let index = sentenceIndex(atOrAfter: position)
    return index > 0 ? positions[index - 1] : nil
  }

  /// The sentence at `position`, or else the nearest sentence before it, or the first one.
  public func clamped(_ position: ReadingPosition) -> ReadingPosition {
    guard let first = positions.first else { return .start }
    let index = sentenceIndex(atOrAfter: position)
    if index < positions.count, positions[index] == position { return position }
    return index == 0 ? first : positions[index - 1]
  }

  /// The index of the last section starting at or before `position`'s block.
  public func section(containing position: ReadingPosition) -> Int? {
    let count = sectionCount(atOrBefore: position.block)
    return count > 0 ? sectionOrder[count - 1] : nil
  }

  /// The first sentence at or after the section's first block; the last sentence when the
  /// section has no text after it.
  public func start(ofSection index: Int) -> ReadingPosition {
    guard structure.sections.indices.contains(index),
      let position = sentence(atOrAfterBlock: structure.sections[index].block) ?? positions.last
    else { return start }
    return position
  }

  public func nextSection(after position: ReadingPosition) -> ReadingPosition? {
    let count = sectionCount(atOrBefore: position.block)
    guard count < sectionStarts.count else { return nil }
    return sentence(atOrAfterBlock: sectionStarts[count])
  }

  /// The start of the current section when `position` is past it, else the previous section's
  /// start.
  public func previousSection(before position: ReadingPosition) -> ReadingPosition? {
    let count = sectionCount(atOrBefore: position.block)
    guard count > 0 else { return nil }
    let currentStart = sentenceIndex(atOrAfterBlock: sectionStarts[count - 1])
    if currentStart < sentenceIndex(atOrAfter: position) {
      return positions[currentStart]
    }
    guard count > 1 else { return nil }
    return sentence(atOrAfterBlock: sectionStarts[count - 2])
  }

  /// The first sentence of the first block on or after `page`.
  public func position(onPage page: Int) -> ReadingPosition? {
    structure.firstBlock(onOrAfterPage: page).flatMap { sentence(atOrAfterBlock: $0) }
  }

  public func page(of position: ReadingPosition) -> Int {
    structure.blocks.indices.contains(position.block)
      ? structure.blocks[position.block].pages.lowerBound : 0
  }

  public func fraction(of position: ReadingPosition) -> Double {
    guard let total = charactersBefore.last, total > 0 else { return 0 }
    return Double(charactersBefore[sentenceIndex(atOrAfter: position)]) / Double(total)
  }

  /// Characters from the start of `position`'s section to `position`, and in the whole section.
  public func sectionCharacters(at position: ReadingPosition) -> (before: Int, total: Int) {
    let span = sentenceSpan(ofSectionContaining: position)
    let origin = charactersBefore[span.lowerBound]
    return (
      before: charactersBefore[sentenceIndex(atOrAfter: position)] - origin,
      total: charactersBefore[span.upperBound] - origin
    )
  }
}

extension Script {
  /// The ranges of `text`'s sentences, in order, trimmed of surrounding whitespace. A text
  /// without sentence punctuation is one sentence. Empty or whitespace-only text has none.
  private static func sentences(of text: String) -> [Range<String.Index>] {
    let tokenizer = NLTokenizer(unit: .sentence)
    tokenizer.string = text
    return tokenizer.tokens(for: text.startIndex..<text.endIndex).compactMap {
      trimmed($0, in: text)
    }
  }

  /// `range` without leading and trailing whitespace, or nil when it has none left.
  private static func trimmed(_ range: Range<String.Index>, in text: String) -> Range<String.Index>? {
    let content = text[range]
    guard let first = content.firstIndex(where: { !$0.isWhitespace }),
      let last = content.lastIndex(where: { !$0.isWhitespace })
    else { return nil }
    return first..<text.index(after: last)
  }

  /// Number of sentences strictly before `position`; the index of `position` when it is a sentence.
  private func sentenceIndex(atOrAfter position: ReadingPosition) -> Int {
    Self.partitionPoint(positions.count) { positions[$0] >= position }
  }

  /// Number of sentences at or before `position`.
  private func sentenceIndex(after position: ReadingPosition) -> Int {
    let index = sentenceIndex(atOrAfter: position)
    return index < positions.count && positions[index] == position ? index + 1 : index
  }

  /// Index of the first sentence in `block` or after it.
  private func sentenceIndex(atOrAfterBlock block: Int) -> Int {
    blockStarts[min(max(block, 0), structure.blocks.count)]
  }

  private func sentence(atOrAfterBlock block: Int) -> ReadingPosition? {
    let index = sentenceIndex(atOrAfterBlock: block)
    return positions.indices.contains(index) ? positions[index] : nil
  }

  /// Number of section starts at or before `block`.
  private func sectionCount(atOrBefore block: Int) -> Int {
    Self.partitionPoint(sectionStarts.count) { sectionStarts[$0] > block }
  }

  /// Sentence indices of the section containing `position`. Text before the first section is
  /// a section of its own, and a document without sections is one section.
  private func sentenceSpan(ofSectionContaining position: ReadingPosition) -> Range<Int> {
    let count = sectionCount(atOrBefore: position.block)
    let lower = count > 0 ? sectionStarts[count - 1] : 0
    let upper = count < sectionStarts.count ? sectionStarts[count] : structure.blocks.count
    return sentenceIndex(atOrAfterBlock: lower)..<sentenceIndex(atOrAfterBlock: upper)
  }

  /// The first index in `0..<count` for which `isAfter` holds, for a predicate that is false
  /// up to that index and true from it on.
  private static func partitionPoint(_ count: Int, _ isAfter: (Int) -> Bool) -> Int {
    var low = 0
    var high = count
    while low < high {
      let middle = (low + high) / 2
      if isAfter(middle) {
        high = middle
      } else {
        low = middle + 1
      }
    }
    return low
  }
}
