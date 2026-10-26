import CoreGraphics
import Testing

@testable import PDFStructure

@Suite struct ReadingOrderTests {
  @Test func readsSingleColumnTopToBottom() {
    let lines = [line("terceira", y: 400), line("primeira", y: 500), line("segunda", y: 450)]

    #expect(ordered(lines) == ["primeira", "segunda", "terceira"])
  }

  @Test func joinsFragmentsOfTheSameRowMarkingWideGapsWithTabs() {
    let lines = [
      line("12", x: 340, y: 500, width: 10), line("Capítulo 1", x: 50, y: 500, width: 70),
      line("continua", x: 124, y: 480, width: 40), line("Texto que", x: 50, y: 480, width: 70),
    ]

    #expect(ordered(lines) == ["Capítulo 1\t12", "Texto que continua"])
  }

  @Test func readsTheLeftColumnBeforeTheRightOne() {
    let lines = column("e", x: 40, from: 500) + column("d", x: 210, from: 500)

    #expect(ordered(lines.shuffled()) == ["e0", "e1", "e2", "e3", "d0", "d1", "d2", "d3"])
  }

  @Test func splitsColumnsIntoBandsAroundFullWidthLines() {
    let title = line("Título", x: 40, y: 560, width: 320)
    let caption = line("Figura", x: 40, y: 400, width: 320)
    let lines =
      [title, caption] + column("a", x: 40, from: 540) + column("b", x: 210, from: 540)
      + column("c", x: 40, from: 380) + column("d", x: 210, from: 380)

    #expect(
      ordered(lines.shuffled()) == [
        "Título", "a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3",
        "Figura", "c0", "c1", "c2", "c3", "d0", "d1", "d2", "d3",
      ])
  }

  @Test func findsTheGutterUnderCenteredLinesThatCrossIt() {
    let authors = line("Autora Um, Autor Dois", x: 140, y: 560, width: 120)
    let lines =
      [authors] + column("a", x: 40, from: 520) + column("b", x: 210, from: 520)
      + column("c", x: 40, from: 460) + column("d", x: 210, from: 460)

    #expect(
      ordered(lines.shuffled()) == [
        "Autora Um, Autor Dois", "a0", "a1", "a2", "a3", "c0", "c1", "c2", "c3", "b0", "b1", "b2",
        "b3",
        "d0", "d1", "d2", "d3",
      ])
  }

  @Test func keepsShortLeftAlignedLinesInOneColumn() {
    let verse = (0..<8).map { line("verso \($0)", x: 40, y: 500 - CGFloat($0) * 14, width: 110) }

    #expect(ReadingOrder.gutter(of: verse) == nil)
  }

  private func ordered(_ lines: [Line]) -> [String] {
    ReadingOrder.lines(
      of: PageText(
        index: 0, label: nil, bounds: .init(x: 0, y: 0, width: 400, height: 600), lines: lines)
    )
    .map(\.text)
  }

  private func column(_ prefix: String, x: CGFloat, from top: CGFloat) -> [Line] {
    (0..<4).map { line("\(prefix)\($0)", x: x, y: top - CGFloat($0) * 14, width: 150) }
  }
}

func line(
  _ text: String, x: CGFloat = 40, y: CGFloat, width: CGFloat = 320, size: Double = 11,
  bold: Bool = false
)
  -> Line
{
  Line(
    text: text, bounds: CGRect(x: x, y: y, width: width, height: size), fontSize: size, isBold: bold
  )
}
