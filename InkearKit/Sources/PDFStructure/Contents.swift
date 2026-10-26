import CoreGraphics
import Foundation

/// A table of contents printed on the pages of the document.
struct Contents: Equatable {
  struct Entry: Equatable {
    enum Target: Equatable {
      /// A page number as printed, such as "12" or "vii".
      case printed(String)
      /// A page index, from a link on the entry.
      case page(Int)
    }

    let title: String
    let level: Int
    let target: Target
  }

  /// Indices of the pages the table occupies.
  let pages: [Int]
  let entries: [Entry]

  /// The first run of pages that read as a table of contents, if any.
  ///
  /// A page qualifies when it is titled like a table of contents and lists entries, or when
  /// most of its lines are entries: lines linking to other pages, or lines ending in page
  /// numbers, after leader dots or a wide gap, that mostly increase. Entry levels follow
  /// indentation.
  static func find(in pages: [PageText]) -> Contents? {
    let listings = pages.map(Listing.init)
    guard let first = listings.firstIndex(where: \.qualifies) else { return nil }
    let run = listings[first...].prefix(while: \.qualifies)
    let items = run.flatMap(\.items)
    let indents = Set(items.map(\.indentStep)).sorted()
    return Contents(
      pages: run.map(\.page),
      entries: items.map { item in
        Entry(
          title: item.title, level: (indents.firstIndex(of: item.indentStep) ?? 0) + 1,
          target: item.target)
      }
    )
  }

  static func isTitle(_ text: String) -> Bool {
    text.count <= 24
      && ["sumario", "indice", "conteudo", "contents", "table of contents"].contains(
        FrontMatter.fold(text))
  }

  /// Where each entry points: its link's page, or the PDF page index of its printed number.
  ///
  /// Printed numbers resolve through page labels when the PDF defines them. Otherwise, the
  /// offset between printed and actual page numbers is the one most entries agree on, judged
  /// by where blocks matching their titles start; without agreement, page "1" is the first page.
  func resolved(pages: [PageText], blocks: [Block]) -> [ContentsEntry] {
    let labels = Dictionary(
      pages.compactMap { page in page.label.map { ($0.lowercased(), page.index) } }
    ) { a, _ in a }
    let printed: (String) -> Int? =
      if labels.isEmpty {
        { [offset = likelyOffset(blocks: blocks) ?? -1] number in Int(number).map { $0 + offset } }
      } else {
        { number in labels[number.lowercased()] }
      }
    return entries.compactMap { entry in
      let index =
        switch entry.target {
        case .page(let index): index
        case .printed(let number): printed(number)
        }
      guard let index, pages.indices.contains(index) else { return nil }
      return ContentsEntry(title: entry.title, level: entry.level, page: index)
    }
  }

  private func likelyOffset(blocks: [Block]) -> Int? {
    let table = Set(pages)
    let candidates = blocks.filter { !table.contains($0.pages.lowerBound) && $0.text.count <= 200 }
      .map { (page: $0.pages.lowerBound, text: FrontMatter.fold($0.text)) }
    let votes = entries.flatMap { entry -> [Int] in
      guard case .printed(let printed) = entry.target, let number = Int(printed) else { return [] }
      let title = FrontMatter.fold(entry.title)
      return candidates.filter { ContentsEntry.matches(folded: title, $0.text) }.map {
        $0.page - number
      }
    }
    return Dictionary(votes.map { ($0, 1) }, uniquingKeysWith: +).max { $0.value < $1.value }?.key
  }
}

/// The entries a page lists, if it reads as part of a table of contents.
private struct Listing {
  struct Item {
    let title: String
    let target: Contents.Entry.Target
    let indentStep: Double
  }

  let page: Int
  let items: [Item]
  let qualifies: Bool

  init(_ page: PageText) {
    let left = page.lines.map(\.bounds.minX).min() ?? 0
    let lines = page.lines.filter { !Contents.isTitle($0.text) }
    let linked = lines.compactMap { line -> Item? in
      // An entry's link spans most of its line, unlike a citation's. Link rectangles of
      // adjacent entries touch, so a line takes the one it overlaps most.
      let overlaps = page.links.filter { $0.page != page.index }.map { link in
        (link, link.bounds.intersection(line.bounds))
      }
      return overlaps.filter { $0.1.width * 2 >= line.bounds.width }.max {
        $0.1.height < $1.1.height
      }
      .map { Item(title: line.text, target: .page($0.0.page), indentStep: Self.step(line, left)) }
    }
    let printed = Self.printed(lines, left: left)
    let titled = page.lines.prefix(3).contains { Contents.isTitle($0.text) }
    let byLinks = Self.listsInOrder(linked, among: lines.count)
    let byNumbers = Self.listsInOrder(printed, among: lines.count)
    self.page = page.index
    items = linked.count >= printed.count ? linked : printed
    qualifies = (titled && items.count >= 2) || byLinks || byNumbers
  }

  /// Whether `items` make most of a page's `lines`, pointing to pages that mostly increase.
  private static func listsInOrder(_ items: [Item], among lines: Int) -> Bool {
    let pages = items.compactMap { item -> Int? in
      switch item.target {
      case .page(let index): index
      case .printed(let number): Int(number)
      }
    }
    let increasing = zip(pages, pages.dropFirst()).count { $0 <= $1 }
    return items.count >= 4 && items.count * 2 >= lines
      && increasing * 5 >= max(1, pages.count - 1) * 4
  }

  /// Entries ending in a page number, joining titles that wrap onto a second line.
  private static func printed(_ lines: [Line], left: CGFloat) -> [Item] {
    var items: [Item] = []
    var pending: Line?
    for line in lines {
      guard
        ["\t", "..", ". .", "…", "·", "__"].contains(where: line.text.contains),
        endsInPageNumber(line.text),
        let match = line.text.wholeMatch(
          of:
            /(?<title>.*?\p{L}.*?)(?:(?:\s*[.·…_]){2,}\s*|\s*\t\s*)(?<page>\d{1,4}|[ivxlcdm]{1,6})/)
      else {
        pending = line
        continue
      }
      let title = [pending?.text, String(match.title)].compactMap { $0 }.joined(separator: " ")
      items.append(
        Item(
          title: title, target: .printed(String(match.page)),
          indentStep: step(pending ?? line, left)))
      pending = nil
    }
    return items
  }

  /// Whether `text` ends in a page number after a space or a leader, as every entry does. Most
  /// lines fail this at once, sparing them the entry pattern, which backtracks at length over
  /// the lines with leaders that it rejects.
  private static func endsInPageNumber(_ text: String) -> Bool {
    func ends(in isPart: (Character) -> Bool, upTo length: Int) -> Bool {
      let number = text.reversed().prefix(while: isPart)
      guard (1...length).contains(number.count), let before = text.dropLast(number.count).last
      else { return false }
      return before.isWhitespace || ".·…_".contains(before)
    }
    return ends(in: \.isNumber, upTo: 4) || ends(in: { "ivxlcdm".contains($0) }, upTo: 6)
  }

  private static func step(_ line: Line, _ left: CGFloat) -> Double {
    (Double(line.bounds.minX - left) / 6).rounded()
  }
}
