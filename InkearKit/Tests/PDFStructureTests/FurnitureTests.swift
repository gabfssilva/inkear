import CoreGraphics
import Testing

@testable import PDFStructure

@Suite struct FurnitureTests {
  @Test(arguments: ["12", "xiv", "- 7 -", "Página 3", "Page 3 of 10", "[42]"])
  func recognizesPageNumbers(text: String) {
    #expect(Furniture().isPageNumber(text))
  }

  @Test(arguments: ["Capítulo 1", "1984 foi um ano.", "Vida e obra"])
  func doesNotTakeTextForPageNumbers(text: String) {
    #expect(!Furniture().isPageNumber(text))
  }

  @Test func removesPageNumbersAndRunningHeaders() {
    let pages = (0..<4).map { index in
      page(
        index,
        [
          line("\(10 + index) DOM CASMURRO", y: 570, size: 9),
          line("Texto do corpo da página \(index).", y: 500),
          line("Mais texto do corpo.", y: 486),
          line("\(10 + index)", y: 30, size: 9),
        ])
    }

    let kept = Furniture().removed(from: pages).map { $0.lines.map(\.text) }

    #expect(kept.allSatisfy { $0.count == 2 && $0[0].hasPrefix("Texto do corpo") })
  }

  @Test func keepsChapterTitlesThatOpenPages() {
    let pages = (0..<4).map { index in
      page(
        index,
        [
          line("CAPÍTULO \(index + 1)", y: 560, size: 18),
          line("Primeiro parágrafo do capítulo.", y: 500),
          line("Continua o parágrafo.", y: 486),
        ])
    }

    let kept = Furniture().removed(from: pages)

    #expect(kept.map { $0.lines.count } == [3, 3, 3, 3])
  }

  @Test func keepsMarginTextThatDoesNotRepeat() {
    let pages = (0..<4).map { index in
      page(
        index,
        [
          line("Linha \(["um", "dois", "três", "quatro"][index]) no topo.", y: 570),
          line("Corpo.", y: 500),
        ])
    }

    #expect(Furniture().removed(from: pages).map { $0.lines.count } == [2, 2, 2, 2])
  }

  private func page(_ index: Int, _ lines: [Line]) -> PageText {
    PageText(
      index: index, label: nil, bounds: CGRect(x: 0, y: 0, width: 400, height: 600), lines: lines)
  }
}
