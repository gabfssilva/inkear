import CoreGraphics
import Testing

@testable import PDFStructure

@Suite struct ContentsTests {
  @Test func readsAPrintedTableOfContents() throws {
    let pages = [
      page(
        0,
        [
          line("Sumário", x: 160, y: 560, width: 80, size: 16),
          line("Prefácio .................... 7", y: 520),
          line("Capítulo I ................. 9", y: 506),
          line("O começo ................. 11", x: 60, y: 492, width: 300),
          line("Capítulo II, que tem um título longo que", y: 478),
          line("continua na linha de baixo ............ 15", y: 464),
        ])
    ]

    let contents = try #require(Contents.find(in: pages))

    #expect(contents.pages == [0])
    #expect(
      contents.entries == [
        .init(title: "Prefácio", level: 1, target: .printed("7")),
        .init(title: "Capítulo I", level: 1, target: .printed("9")),
        .init(title: "O começo", level: 2, target: .printed("11")),
        .init(
          title: "Capítulo II, que tem um título longo que continua na linha de baixo", level: 1,
          target: .printed("15")),
      ])
  }

  @Test func readsATableOfContentsMadeOfLinks() throws {
    let lines = (1...5).map { line("Capítulo \($0)", y: 520 - CGFloat($0) * 12, width: 80) }
    // As in real tables of contents, each link reaches a little into the entry below it.
    let links = lines.enumerated().map { index, line in
      Link(
        bounds: CGRect(
          x: line.bounds.minX, y: line.bounds.minY - 2, width: line.bounds.width, height: 13.5),
        page: 3 + index * 4)
    }
    let pages = [
      PageText(
        index: 0, label: nil, bounds: CGRect(x: 0, y: 0, width: 400, height: 600), lines: lines,
        links: links)
    ]

    let contents = try #require(Contents.find(in: pages))

    #expect(
      contents.entries.map(\.title) == [
        "Capítulo 1", "Capítulo 2", "Capítulo 3", "Capítulo 4", "Capítulo 5",
      ])
    #expect(
      contents.entries.map(\.target) == [.page(3), .page(7), .page(11), .page(15), .page(19)])
  }

  @Test func doesNotTakeNumberedTitlesForPageNumbers() {
    let pages = [
      page(0, (1...6).map { line("Capítulo \($0)", y: 520 - CGFloat($0) * 14, width: 80) })
    ]

    #expect(Contents.find(in: pages) == nil)
  }

  @Test func ignoresProseWhoseLinesEndInNumbers() {
    let pages = [
      page(
        0,
        [
          line("A guerra começou em 1914", y: 520),
          line("e acabou em 1918, mas os efeitos duraram até 1939", y: 506),
          line("e muitos outros anos depois de 1945", y: 492),
          line("o que se estudou até 1960", y: 478),
          line("e se estuda ainda em 1900", y: 464),
        ])
    ]

    #expect(Contents.find(in: pages) == nil)
  }

  @Test func resolvesPrintedPagesThroughTheOffsetTitlesAgreeOn() {
    let contents = Contents(
      pages: [0],
      entries: [
        .init(title: "Capítulo I", level: 1, target: .printed("1")),
        .init(title: "Capítulo II", level: 1, target: .printed("3")),
      ]
    )
    let blocks = [
      Block(kind: .paragraph, text: "Capítulo I .... 1", pages: 0...0),
      Block(kind: .heading(level: 1), text: "CAPÍTULO I", pages: 2...2),
      Block(kind: .heading(level: 1), text: "CAPÍTULO II", pages: 4...4),
    ]

    let resolved = contents.resolved(pages: (0..<6).map { page($0, []) }, blocks: blocks)

    #expect(resolved.map(\.page) == [2, 4])
  }

  @Test func resolvesPrintedPagesThroughPageLabels() {
    let contents = Contents(
      pages: [1], entries: [.init(title: "Prefácio", level: 1, target: .printed("iii"))])
    let pages = ["i", "ii", "iii"].enumerated().map { page($0, [], label: $1) }

    #expect(contents.resolved(pages: pages, blocks: []).map(\.page) == [2])
  }

  @Test func indexesTheDocumentByItsPrintedContents() {
    let chapter = { (index: Int, title: String) in
      self.page(
        index,
        [
          line(title, x: 150, y: 540, width: 100, size: 18),
          line("Texto do capítulo, longo o bastante para ser corpo.", y: 500),
          line("Mais uma linha do corpo do texto.", y: 486, width: 200),
        ])
    }
    let pages = [
      page(
        0,
        [
          line("Sumário", x: 160, y: 560, width: 80, size: 16), line("Prefácio ..... 2", y: 520),
          line("Capítulo Um ..... 3", y: 506),
        ]),
      chapter(1, "Prefácio"),
      chapter(2, "Capítulo Um"),
    ]

    let structure = DocumentStructure(pages: pages, outline: [])

    #expect(structure.sections.map(\.title) == ["Prefácio", "Capítulo Um"])
    #expect(structure.contentsStart == 0)
    #expect(structure.mainStart.map { structure.blocks[$0].text } == "Capítulo Um")
  }

  private func page(_ index: Int, _ lines: [Line], label: String? = nil) -> PageText {
    PageText(
      index: index, label: label, bounds: CGRect(x: 0, y: 0, width: 400, height: 600), lines: lines)
  }
}
