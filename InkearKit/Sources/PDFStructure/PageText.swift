import CoreGraphics
import PDFKit

/// A line of text as laid out on a page.
public struct Line: Sendable, Equatable {
  public let text: String
  /// Bounds in page space: origin at the bottom-left, y grows upwards.
  public let bounds: CGRect
  public let fontSize: Double
  public let isBold: Bool

  public init(text: String, bounds: CGRect, fontSize: Double, isBold: Bool) {
    self.text = text
    self.bounds = bounds
    self.fontSize = fontSize
    self.isBold = isBold
  }
}

/// A link on a page to another page of the same document.
public struct Link: Sendable, Equatable {
  public let bounds: CGRect
  public let page: Int

  public init(bounds: CGRect, page: Int) {
    self.bounds = bounds
    self.page = page
  }
}

/// The text lines of one page, in the order PDFKit reports them, which is not reading order.
public struct PageText: Sendable, Equatable {
  public let index: Int
  /// The page label the PDF defines (`/PageLabels`), such as "iv" or "12"; `nil` when it defines none.
  public let label: String?
  public let bounds: CGRect
  public let lines: [Line]
  public let links: [Link]

  public init(index: Int, label: String?, bounds: CGRect, lines: [Line], links: [Link] = []) {
    self.index = index
    self.label = label
    self.bounds = bounds
    self.lines = lines
    self.links = links
  }

  /// The share of letters among the visible characters, or `nil` with too few to judge.
  ///
  /// Text from fonts without a Unicode mapping extracts as symbols, far below the share of
  /// letters in any language.
  public var letterRatio: Double? {
    let characters = lines.flatMap(\.text.unicodeScalars).filter {
      !CharacterSet.whitespaces.contains($0)
    }
    guard characters.count >= 20 else { return nil }
    return Double(characters.count(where: CharacterSet.letters.contains)) / Double(characters.count)
  }

  func with(lines: [Line]) -> PageText {
    PageText(index: index, label: label, bounds: bounds, lines: lines, links: links)
  }
}

extension PageText {
  /// Extracts the text lines of every page of `document`, reporting to `progress` how many
  /// pages are done.
  ///
  /// Batches of pages are extracted in parallel, each from a copy of the document of its own,
  /// as PDFKit objects are not safe to share across threads.
  public static func extract(
    from document: PDFDocument, progress: @Sendable (_ done: Int) -> Void = { _ in }
  ) async throws -> [PageText] {
    guard
      let data = try document.documentURL.map({ try Data(contentsOf: $0, options: .mappedIfSafe) })
        ?? document.dataRepresentation()
    else { throw CocoaError(.fileReadCorruptFile) }
    let hasLabels =
      document.documentRef?.catalog.flatMap { catalog in
        var labels: CGPDFDictionaryRef?
        return CGPDFDictionaryGetDictionary(catalog, "PageLabels", &labels)
      } ?? false
    let workers = ProcessInfo.processInfo.activeProcessorCount
    let size = max(4, document.pageCount / (workers * 4))
    var batches = stride(from: 0, to: document.pageCount, by: size).map {
      $0..<min($0 + size, document.pageCount)
    }.makeIterator()
    return try await withThrowingTaskGroup(of: [PageText].self) { group in
      func addNext() {
        guard let batch = batches.next() else { return }
        group.addTask { try extract(batch, from: data, hasLabels: hasLabels) }
      }
      for _ in 0..<workers { addNext() }
      var pages: [PageText] = []
      for try await batch in group {
        pages += batch
        progress(pages.count)
        addNext()
      }
      return pages.sorted { $0.index < $1.index }
    }
  }

  private static func extract(_ indices: Range<Int>, from data: Data, hasLabels: Bool) throws
    -> [PageText]
  {
    guard let document = PDFDocument(data: data) else { throw CocoaError(.fileReadCorruptFile) }
    return try indices.compactMap { index in
      try Task.checkCancellation()
      return document.page(at: index).map { page in
        let bounds = page.bounds(for: .cropBox)
        return PageText(
          index: index,
          label: hasLabels ? page.label : nil,
          bounds: bounds,
          lines: page.selection(for: bounds)?.selectionsByLine().compactMap {
            line(from: $0, on: page)
          } ?? [],
          links: page.annotations.compactMap { annotation in
            let destination =
              annotation.destination ?? (annotation.action as? PDFActionGoTo)?.destination
            return destination?.page.map {
              Link(bounds: annotation.bounds, page: document.index(for: $0))
            }
          }
        )
      }
    }
  }

  private static func line(from selection: PDFSelection, on page: PDFPage) -> Line? {
    guard
      let text = selection.string?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty,
      let attributed = selection.attributedString
    else { return nil }
    let fonts = fontRuns(in: attributed)
    let dominant = fonts.max { $0.length < $1.length }
    return Line(
      text: text,
      bounds: selection.bounds(for: page),
      fontSize: dominant.map { Double($0.font.pointSize) } ?? 0,
      isBold: dominant.map { isBold($0.font.fontName) } ?? false
    )
  }

  private static func fontRuns(in text: NSAttributedString) -> [(font: PlatformFont, length: Int)] {
    var runs: [(font: PlatformFont, length: Int)] = []
    text.enumerateAttribute(.font, in: NSRange(location: 0, length: text.length)) {
      value, range, _ in
      if let font = value as? PlatformFont { runs.append((font, range.length)) }
    }
    return runs
  }

  private static func isBold(_ fontName: String) -> Bool {
    let name = fontName.lowercased()
    return ["bold", "black", "heavy", "semibold", "demi"].contains { name.contains($0) }
  }
}

#if canImport(AppKit)
  private typealias PlatformFont = NSFont
#else
  private typealias PlatformFont = UIFont
#endif
