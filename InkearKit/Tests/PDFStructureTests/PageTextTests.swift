import PDFKit
import Synchronization
import Testing

@testable import PDFStructure

@Suite struct PageTextTests {
  @Test func extractsLinesWithFontAndPosition() async throws {
    let document = try #require(
      SyntheticPDF().document(pages: [
        [
          .init(string: "Capítulo I", x: 140, y: 500, font: "Times-Bold", size: 20),
          .init(string: "Primeira linha.", x: 50, y: 450),
        ]
      ]))

    let pages = try await PageText.extract(from: document)
    let lines = pages[0].lines.sorted { $0.bounds.maxY > $1.bounds.maxY }

    #expect(pages.count == 1)
    #expect(pages[0].label == nil)
    #expect(lines.map(\.text) == ["Capítulo I", "Primeira linha."])
    #expect(lines.map(\.fontSize) == [20, 11])
    #expect(lines.map(\.isBold) == [true, false])
    #expect(lines[1].bounds.minX == 50)
  }

  @Test func extractsPagesInOrderReportingProgress() async throws {
    let texts = (1...30).map { "Página \($0)." }
    let document = try #require(
      SyntheticPDF().document(pages: texts.map { [.init(string: $0, x: 50, y: 450)] }))
    let reported = Mutex<[Int]>([])

    let pages = try await PageText.extract(from: document) { done in
      reported.withLock { $0.append(done) }
    }

    #expect(pages.map(\.index) == Array(0..<30))
    #expect(pages.map { $0.lines.map(\.text) } == texts.map { [$0] })
    #expect(reported.withLock { $0 } == reported.withLock { $0 }.sorted())
    #expect(reported.withLock { $0.last } == 30)
  }
}

@Suite struct RecognitionTests {
  @Test func recognizesPagesWithoutTextLayer() async throws {
    let source = try #require(
      SyntheticPDF().document(pages: [
        [
          .init(string: "Capítulo Primeiro", x: 100, y: 500, font: "Times-Bold", size: 22),
          .init(string: "Era uma vez uma página escaneada.", x: 50, y: 440, size: 14),
        ]
      ]))
    let scanned = try #require(Self.imageOnly(source))

    let structure = try await DocumentStructure.analyze(scanned)

    #expect(
      structure.blocks.map(\.text) == ["Capítulo Primeiro", "Era uma vez uma página escaneada."])
    #expect(structure.blocks.map(\.kind) == [.heading(level: 1), .paragraph])
  }

  /// A copy of `document` whose pages are images of the original pages.
  private static func imageOnly(_ document: PDFDocument) -> PDFDocument? {
    guard let page = document.page(at: 0) else { return nil }
    let image = page.thumbnail(of: CGSize(width: 1200, height: 1800), for: .cropBox)
    let copy = PDFDocument()
    guard let imagePage = PDFPage(image: image) else { return nil }
    copy.insert(imagePage, at: 0)
    return copy
  }
}
