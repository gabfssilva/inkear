import CoreGraphics
import Foundation

/// A unit of reading: a heading or a paragraph.
public struct Block: Sendable, Equatable, Codable {
  public enum Kind: Sendable, Equatable, Codable {
    /// A heading; level 1 is the most prominent.
    case heading(level: Int)
    case paragraph
  }

  public let kind: Kind
  public let text: String
  /// Indices of the pages the block spans.
  public let pages: ClosedRange<Int>

  public init(kind: Kind, text: String, pages: ClosedRange<Int>) {
    self.kind = kind
    self.text = text
    self.pages = pages
  }
}

extension Block {
  /// Groups the lines of `pages`, already in reading order and free of furniture, into blocks.
  ///
  /// Each line on a page in `linePerBlock`, such as a table of contents, is a block of its own.
  static func assemble(_ pages: [PageText], linePerBlock: Set<Int> = []) -> [Block] {
    let layout = Layout(pages: pages)
    var drafts: [[Placed]] = []
    for current in layout.placed(pages) {
      if let previous = drafts.last?.last, !linePerBlock.contains(current.page),
        !layout.startsBlock(current, after: previous)
      {
        drafts[drafts.count - 1].append(current)
      } else {
        drafts.append([current])
      }
    }
    let headings = drafts.map(layout.isHeading)
    // Levels rank the heading sizes used more than once; one-off sizes, as on title pages,
    // take the level of the next recurring size below them.
    let sizes = zip(drafts, headings).filter(\.1).map { Layout.size(of: $0.0) }
    let recurring = Dictionary(grouping: sizes, by: \.self).filter { $0.value.count > 1 }.keys
    return zip(drafts, headings).map { draft, heading in
      let size = Layout.size(of: draft)
      return Block(
        kind: heading ? .heading(level: 1 + recurring.count { $0 > size }) : .paragraph,
        text: joined(draft.map(\.line.text)),
        pages: draft[0].page...draft[draft.count - 1].page
      )
    }
  }

  /// Joins line texts with spaces, undoing hyphenation where a word was split across lines.
  static func joined(_ lines: [String]) -> String {
    lines.map { $0.replacing("\t", with: " ") }.reduce("") { text, line in
      guard let last = text.last else { return line }
      let hyphenated =
        (last == "-" || last == "\u{00AD}") && text.dropLast().last?.isLetter == true
        && line.first?.isLowercase == true
      return hyphenated ? text.dropLast() + line : text + " " + line
    }
  }
}

/// A line with its page and the horizontal extent of its column.
private struct Placed {
  let page: Int
  let line: Line
  let column: ClosedRange<CGFloat>
}

/// Document-wide layout metrics and the rules that use them.
private struct Layout {
  let body: Double
  /// The usual distance between consecutive baselines of body text.
  let spacing: Double
  let bodyIsBold: Bool
  // Kept here rather than written inline, since a regex literal is parsed again each time it is
  // evaluated, and `Regex` is not `Sendable`, so it cannot be a static constant.
  private let sentenceEnd = /[.!?:…]["'”’»)\]]*$/
  private let chapterName =
    /(?i)(?:cap[ií]tulo|chapter|parte|part|livro|book|ato|act)\b.{0,80}|[IVXLC]{1,7}[.)]?/

  init(pages: [PageText]) {
    let body = Typography(pages: pages).bodySize
    let bodyLines = pages.map { page in page.lines.filter { abs($0.fontSize - body) < 0.5 } }
    let gaps = bodyLines.flatMap { lines -> [Double] in
      let deltas = zip(lines, lines.dropFirst()).map { Double($0.bounds.minY - $1.bounds.minY) }
      return deltas.filter { $0 > 0 && $0 < body * 2.5 }
    }.sorted()
    let characters = bodyLines.joined().reduce(0) { $0 + $1.text.count }
    let boldCharacters = bodyLines.joined().filter(\.isBold).reduce(0) { $0 + $1.text.count }
    self.body = body
    spacing = gaps.isEmpty ? body * 1.2 : gaps[gaps.count / 2]
    bodyIsBold = boldCharacters * 2 > characters
  }

  func placed(_ pages: [PageText]) -> [Placed] {
    pages.flatMap { page in
      let body = page.lines.filter { abs($0.fontSize - self.body) < 0.5 }.map(\.bounds)
      return page.lines.map { line in
        // A line's column is the extent of the body lines it overlaps horizontally.
        let column = body.filter { $0.minX < line.bounds.maxX && $0.maxX > line.bounds.minX }
        return Placed(
          page: page.index,
          line: line,
          column: (column.map(\.minX).min() ?? line.bounds.minX)...(column.map(\.maxX).max()
            ?? line.bounds.maxX)
        )
      }
    }
  }

  func startsBlock(_ current: Placed, after previous: Placed) -> Bool {
    let (a, b) = (previous.line, current.line)
    if abs(a.fontSize - b.fontSize) > 0.75 || a.isBold != b.isBold { return true }
    // A further indent past a line that ends mid-sentence, before one that starts in lowercase,
    // is a hanging indent continuing the same item, not a new paragraph.
    let continues = !endsSentence(a.text) && b.text.first?.isLowercase == true
    let indented = b.bounds.minX > a.bounds.minX + b.fontSize * 0.6 && !continues
    if current.page != previous.page || b.bounds.minY > a.bounds.minY {
      return endsSentence(a.text) || indented
    }
    let gap = Double(a.bounds.minY - b.bounds.minY)
    if gap > spacing * 1.45 * max(1, max(a.fontSize, b.fontSize) / body) || indented { return true }
    let short = a.bounds.maxX < previous.column.upperBound - a.fontSize * 2
    return short && endsSentence(a.text)
  }

  func isHeading(_ draft: [Placed]) -> Bool {
    let text = Block.joined(draft.map(\.line.text))
    guard draft.count <= 3, text.count <= 160 else { return false }
    let larger = Self.size(of: draft) >= body + 1
    let bold = draft.allSatisfy(\.line.isBold) && !bodyIsBold
    let named =
      draft.count == 1 && text.wholeMatch(of: chapterName) != nil
    return larger || bold || named
  }

  static func size(of draft: [Placed]) -> Double {
    ((draft.map(\.line.fontSize).max() ?? 0) * 2).rounded() / 2
  }

  func endsSentence(_ text: String) -> Bool {
    text.contains(sentenceEnd)
  }
}
