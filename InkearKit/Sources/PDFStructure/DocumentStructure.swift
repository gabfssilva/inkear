import Foundation
import PDFKit

/// The readable content of a PDF: its blocks in reading order, and the sections that index them.
public struct DocumentStructure: Sendable, Equatable, Codable {
  public let pageCount: Int
  public let blocks: [Block]
  public let sections: [Section]
  /// The first block of the printed table of contents, if the document has one.
  public let contentsStart: Int?
  /// The block where the main text starts, past covers, tables of contents, prefaces and the like.
  public let mainStart: Int?
  /// Whether the text of some pages was recognized from their image.
  public let isRecognized: Bool

  public init(
    pageCount: Int, blocks: [Block], sections: [Section], contentsStart: Int?, mainStart: Int?,
    isRecognized: Bool = false
  ) {
    self.pageCount = pageCount
    self.blocks = blocks
    self.sections = sections
    self.contentsStart = contentsStart
    self.mainStart = mainStart
    self.isRecognized = isRecognized
  }

  /// The first block that starts on page `index` or later.
  public func firstBlock(onOrAfterPage index: Int) -> Int? {
    blocks.firstIndex { $0.pages.lowerBound >= index }
  }
}

/// How far ``DocumentStructure/analyze(_:progress:)`` has got.
public enum AnalysisProgress: Sendable, Equatable {
  /// Extracting the text layer: `page` pages done out of `of`.
  case reading(page: Int, of: Int)
  /// Recognizing the text of pages without a usable text layer: `page` done out of `of`.
  case recognizing(page: Int, of: Int)
  /// Building blocks and sections, after recognizing the text of `recognizedPages` pages.
  case structuring(recognizedPages: Int)

  /// The pages the current step has done and has to do; `nil` while structuring.
  public var pages: (done: Int, of: Int)? {
    switch self {
    case .reading(let page, let of), .recognizing(let page, let of): (page, of)
    case .structuring: nil
    }
  }
}

/// A sentence in a document: its block, and its index in that block's ``Block/sentences``.
public struct ReadingPosition: Sendable, Hashable, Codable, Comparable {
  public let block: Int
  public let sentence: Int

  public init(block: Int, sentence: Int) {
    self.block = block
    self.sentence = sentence
  }

  public static let start = ReadingPosition(block: 0, sentence: 0)

  public static func < (a: ReadingPosition, b: ReadingPosition) -> Bool {
    (a.block, a.sentence) < (b.block, b.sentence)
  }
}

/// A navigable division of the document, such as a chapter.
public struct Section: Sendable, Equatable, Codable {
  public let title: String
  /// Nesting depth; 1 is the outermost.
  public let level: Int
  /// The index of the section's first block.
  public let block: Int

  public init(title: String, level: Int, block: Int) {
    self.title = title
    self.level = level
    self.block = block
  }
}

extension DocumentStructure {
  /// Analyzes `document`, recognizing the text of pages without a usable text layer.
  ///
  /// A text layer is not usable when it is missing from a page; when most pages have fewer
  /// letters than symbols, as fonts without a Unicode mapping extract; or when its sizes were
  /// measured rather than set, the mark of an earlier OCR pass, which Vision usually improves on.
  /// Pages of code, tables or figures can have few letters too, so those calls are made for the
  /// whole document, not per page.
  public static func analyze(
    _ document: PDFDocument, progress: @Sendable (AnalysisProgress) -> Void = { _ in }
  ) async throws -> DocumentStructure {
    let count = document.pageCount
    progress(.reading(page: 0, of: count))
    let extracted = try await PageText.extract(from: document) { done in
      progress(.reading(page: done, of: count))
    }
    let ratios = extracted.compactMap(\.letterRatio)
    let unreadable = ratios.count { $0 < 0.5 } * 2 > ratios.count
    let earlierOCR = Typography(pages: extracted).isMeasured
    let recognized = Set(
      extracted.filter {
        $0.lines.isEmpty || earlierOCR || (unreadable && ($0.letterRatio ?? 0) < 0.5)
      }.map(\.index))
    var pages: [PageText] = []
    var done = 0
    for page in extracted {
      try Task.checkCancellation()
      if recognized.contains(page.index), let pdfPage = document.page(at: page.index) {
        progress(.recognizing(page: done, of: recognized.count))
        pages.append(try await PageText.recognize(pdfPage, replacing: page))
        done += 1
      } else {
        pages.append(page)
      }
    }
    progress(.structuring(recognizedPages: recognized.count))
    return DocumentStructure(
      pages: pages, outline: ContentsEntry.outline(of: document), isRecognized: !recognized.isEmpty)
  }

  /// Builds the structure of extracted `pages`.
  ///
  /// Sections come from `outline` when the PDF has one, else from a printed table of contents,
  /// else from the headings found in the layout.
  init(pages extracted: [PageText], outline: [ContentsEntry], isRecognized: Bool = false) {
    let pages = Furniture().removed(
      from: Typography(pages: extracted).steadied(extracted).map {
        $0.with(lines: ReadingOrder.lines(of: $0))
      })
    let contents = Contents.find(in: pages)
    let blocks = Block.assemble(pages, linePerBlock: Set(contents?.pages ?? []))
    let entries = outline.isEmpty ? contents?.resolved(pages: pages, blocks: blocks) ?? [] : outline
    let sections =
      entries.isEmpty ? Self.headingSections(blocks) : entries.compactMap { $0.section(in: blocks) }
    let contentsStart = contents.flatMap { contents in
      blocks.firstIndex { contents.pages.contains($0.pages.lowerBound) }
    }
    self.init(
      pageCount: pages.count,
      blocks: blocks,
      sections: sections,
      contentsStart: contentsStart,
      mainStart: Self.mainStart(
        pages: pages, blocks: blocks, sections: sections, contentsStart: contentsStart),
      isRecognized: isRecognized
    )
  }

  /// Sections from the headings in the layout. Consecutive headings, such as a chapter number
  /// above the chapter's title, make one section.
  static func headingSections(_ blocks: [Block]) -> [Section] {
    var sections: [Section] = []
    var previousIsHeading = false
    for (index, block) in blocks.enumerated() {
      guard case .heading(let level) = block.kind else {
        previousIsHeading = false
        continue
      }
      if previousIsHeading, let last = sections.popLast() {
        sections.append(
          Section(
            title: last.title + " — " + block.text, level: min(last.level, level), block: last.block
          ))
      } else {
        sections.append(Section(title: block.text, level: level, block: index))
      }
      previousIsHeading = true
    }
    return sections
  }

  /// Where the main text starts.
  ///
  /// When page labels switch from roman to arabic numbers, it is the page labelled "1", or the
  /// first section after it that is not front matter if front matter continues there. Otherwise,
  /// it is the first section past the table of contents that reads like a chapter, or else the
  /// first that is not front matter.
  static func mainStart(
    pages: [PageText], blocks: [Block], sections: [Section], contentsStart: Int?
  ) -> Int? {
    let labels = pages.map { $0.label?.lowercased() ?? "" }
    if let one = labels.firstIndex(of: "1"),
      labels[..<one].contains(where: { $0.wholeMatch(of: /[ivxlcdm]+/) != nil }),
      let numbered = blocks.firstIndex(where: { $0.pages.contains(one) })
    {
      let after = sections.filter { $0.block >= numbered }
      guard let first = after.first, FrontMatter.contains(first.title) else { return numbered }
      return after.first { !FrontMatter.contains($0.title) }?.block ?? numbered
    }
    let pastContents = sections.filter { $0.block > (contentsStart ?? -1) }
    let body = (pastContents.isEmpty ? sections : pastContents).filter {
      !FrontMatter.contains($0.title)
    }
    return (body.first { FrontMatter.isChapter($0.title) } ?? body.first)?.block
  }
}

/// Titles that front matter goes by, and titles that chapters go by.
enum FrontMatter {
  static func contains(_ title: String) -> Bool {
    let folded = fold(title)
    return Contents.isTitle(title)
      || titles.contains { title in
        [title, title + "s"].contains { folded == $0 || folded.hasPrefix($0 + " ") }
      }
  }

  static func isChapter(_ title: String) -> Bool {
    title.firstMatch(
      of:
        /^(?i:cap[ií]tulo|chapter|parte|part|livro|book)\s+\S|^[IVXLC]{1,7}[.)]?(?:\s|$)|^\d{1,3}[.)]?\s/
    ) != nil
  }

  static func fold(_ text: String) -> String {
    text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
      .replacing(/[^\p{L}\p{N} ]/, with: "")
      .replacing(/\s+/, with: " ")
      .trimmingCharacters(in: .whitespaces)
  }

  private static let titles = [
    "capa", "cover", "folha de rosto", "title page", "half title", "ficha catalografica",
    "copyright", "licenca", "license", "creditos", "credits", "dedicatoria", "dedication",
    "epigrafe", "epigraph", "agradecimento", "acknowledgment", "acknowledgement", "prefacio",
    "preface", "foreword", "apresentacao", "advertencia", "ao leitor", "nota do editor",
    "nota da edicao", "nota editorial", "editors note", "sobre o autor", "about the author",
    "colaboradores", "contributors", "lista de figuras",
    "list of figures", "lista de tabelas", "list of tables", "lista de abreviaturas",
    "abreviatura", "abbreviation", "resumo", "abstract",
  ]
}

/// A titled pointer to a page, from the PDF outline or a printed table of contents.
struct ContentsEntry: Equatable {
  let title: String
  let level: Int
  let page: Int

  /// The PDF outline (bookmarks), flattened with each entry's depth.
  static func outline(of document: PDFDocument) -> [ContentsEntry] {
    func entries(of node: PDFOutline, level: Int) -> [ContentsEntry] {
      (0..<node.numberOfChildren).compactMap { node.child(at: $0) }.flatMap { child in
        let own = child.destination?.page.map {
          ContentsEntry(title: child.label ?? "", level: level, page: document.index(for: $0))
        }
        return (own.map { [$0] } ?? []) + entries(of: child, level: level + 1)
      }
    }
    return document.outlineRoot.map { entries(of: $0, level: 1) } ?? []
  }

  /// Whether a block reading `text` can be the heading titled `title`: one contains the other,
  /// and a text inside the title is at least half of it.
  static func matches(_ title: String, _ text: String) -> Bool {
    matches(folded: FrontMatter.fold(title), FrontMatter.fold(text))
  }

  /// ``matches(_:_:)`` for texts already passed through ``FrontMatter/fold(_:)``.
  static func matches(folded title: String, _ text: String) -> Bool {
    guard !title.isEmpty, text.count >= 3, text.count <= max(100, title.count * 2) else {
      return false
    }
    return text.contains(title) || (title.contains(text) && text.count * 2 >= title.count)
  }

  /// The section this entry points to: the block on its page that matches the title, or else
  /// the first block starting on that page.
  func section(in blocks: [Block]) -> Section? {
    let onPage = blocks.indices.filter { blocks[$0].pages.contains(page) }
    let heading = onPage.first { Self.matches(title, blocks[$0].text) }
    return (heading ?? onPage.first { blocks[$0].pages.lowerBound == page } ?? onPage.first).map {
      Section(title: title, level: level, block: $0)
    }
  }
}
