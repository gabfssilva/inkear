import CoreGraphics
import Foundation

/// Running headers, running footers and page numbers: text in the page margins that belongs
/// to the page rather than to the document's content.
///
/// A margin line is furniture when it reads as a page number, or when, with digits ignored, it
/// repeats on at least three pages. Lines larger than body text are never furniture, so chapter
/// titles opening pages are kept.
struct Furniture {
  // Built once rather than written inline, as Blocks.swift explains.
  private let pageNumber =
    /(?i)[\s\-–—\[\(]*(?:p(?:ágina|age|\.)?\s*)?(?:\d{1,4}|[ivxlcdm]{1,7})(?:\s*(?:de|of|\/)\s*\d{1,4})?[\s\-–—\]\)]*/
  private let leadingNumber = /(\d{1,4})\s.*/
  private let trailingNumber = /.*\s(\d{1,4})/
  private let digits = /\d+/
  private let spaces = /\s+/

  /// Removes furniture from `pages`, whose lines must be in reading order.
  func removed(from pages: [PageText]) -> [PageText] {
    let body = Typography(pages: pages).bodySize
    let candidates = pages.map { page in
      (page.index, Self.marginLines(of: page).filter { $0.fontSize <= body + 0.5 })
    }
    let repeated = Dictionary(
      grouping: candidates.flatMap { Set($0.1.map(normalized)) }, by: \.self
    ).filter { $0.value.count >= 3 }.keys
    // Running headers that carry the page number: the number at either end of the line keeps
    // the same offset from the page index, whatever the rest of the line says.
    let offsets = Dictionary(
      grouping: candidates.flatMap { index, lines in
        lines.compactMap { edgeNumber($0.text).map { $0 - index } }
      },
      by: \.self
    )
    let pageOffset = offsets.filter { $0.value.count >= 3 }.max { $0.value.count < $1.value.count }?
      .key
    return zip(pages, candidates).map { page, candidates in
      let furniture = candidates.1.filter { line in
        isPageNumber(line.text) || repeated.contains(normalized(line))
          || (pageOffset != nil && edgeNumber(line.text).map { $0 - page.index } == pageOffset)
      }
      return page.with(lines: page.lines.filter { !furniture.contains($0) })
    }
  }

  func isPageNumber(_ text: String) -> Bool {
    text.wholeMatch(of: pageNumber) != nil
  }

  /// The two topmost and two bottommost rows, when they lie near the page edges.
  private static func marginLines(of page: PageText) -> [Line] {
    let byHeight = page.lines.sorted { $0.bounds.maxY > $1.bounds.maxY }
    let margin = page.bounds.height * 0.12
    let top = byHeight.prefix(2).filter { $0.bounds.minY > page.bounds.maxY - margin }
    let bottom = byHeight.suffix(2).filter { $0.bounds.maxY < page.bounds.minY + margin }
    return Array(top) + bottom.filter { !top.contains($0) }
  }

  /// The number a line starts or ends with, as running headers place page numbers.
  private func edgeNumber(_ text: String) -> Int? {
    let match = text.wholeMatch(of: leadingNumber) ?? text.wholeMatch(of: trailingNumber)
    return match.flatMap { Int($0.1) }
  }

  private func normalized(_ line: Line) -> String {
    line.text.lowercased()
      .replacing(digits, with: "#")
      .replacing(spaces, with: " ")
  }
}
