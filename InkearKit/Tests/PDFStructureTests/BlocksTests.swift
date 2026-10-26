import CoreGraphics
import Testing

@testable import PDFStructure

@Suite struct BlocksTests {
  @Test func splitsParagraphsAtIndentedFirstLines() {
    let blocks = assemble([
      [
        line("Era uma vez um texto que ocupava a linha inteira", x: 52, y: 500),
        line("e continuava na linha de baixo até terminar.", y: 486, width: 250),
        line("Outro parágrafo começava recuado e ia até o fim", x: 52, y: 472),
        line("da linha seguinte.", y: 458, width: 90),
      ]
    ])

    #expect(
      blocks.map(\.text) == [
        "Era uma vez um texto que ocupava a linha inteira e continuava na linha de baixo até terminar.",
        "Outro parágrafo começava recuado e ia até o fim da linha seguinte.",
      ])
  }

  @Test func splitsParagraphsAtVerticalGaps() {
    let blocks = assemble([
      [
        line("Primeiro parágrafo em estilo de bloco, sem recuo,", y: 500),
        line("que termina aqui", y: 486),
        line("Segundo parágrafo depois de um espaço maior.", y: 460),
        line("Mais uma linha dele.", y: 446),
      ]
    ])

    #expect(blocks.count == 2)
  }

  @Test func keepsAnIndentedQuotationTogether() {
    let blocks = assemble([
      [
        line("Texto antes da citação, que termina com dois pontos:", y: 500, width: 250),
        line("Uma citação longa recuada dos dois lados, que", x: 70, y: 486, width: 260),
        line("ocupa várias linhas sem recuo de primeira linha", x: 70, y: 472, width: 260),
        line("e termina aqui.", x: 70, y: 458, width: 80),
      ]
    ])

    #expect(blocks.count == 2)
    #expect(blocks[1].text.hasPrefix("Uma citação longa"))
  }

  @Test func keepsHangingIndentsInTheirItem() {
    let blocks = assemble([
      [
        line("• Primeiro item da lista, que não cabe numa", y: 500),
        line("linha só e continua recuado.", x: 52, y: 486, width: 200),
        line("• Segundo item.", y: 472, width: 100),
      ]
    ])

    #expect(
      blocks.map(\.text) == [
        "• Primeiro item da lista, que não cabe numa linha só e continua recuado.",
        "• Segundo item.",
      ])
  }

  @Test func continuesUnfinishedSentencesOnTheNextPage() {
    let blocks = assemble([
      [line("Uma frase que começa numa página", y: 100)],
      [line("e acaba na outra.", y: 560, width: 90)],
    ])

    #expect(blocks.map(\.text) == ["Uma frase que começa numa página e acaba na outra."])
    #expect(blocks[0].pages == 0...1)
  }

  @Test func undoesHyphenation() {
    #expect(Block.joined(["uma palavra com-", "prida aqui"]) == "uma palavra comprida aqui")
    #expect(Block.joined(["de 1990-", "2000"]) == "de 1990- 2000")
  }

  @Test func ranksHeadingLevelsByTheSizesTheyRecurIn() {
    let chapter = { (y: CGFloat) in
      [
        line("Capítulo", x: 150, y: y, width: 100, size: 18),
        line("Seção", x: 40, y: y - 40, width: 100, size: 14),
        line("Texto do corpo, longo o bastante para ser o corpo do", y: y - 70),
        line("documento, com várias linhas de tamanho normal.", y: y - 84, width: 250),
        line("Subseção em negrito", y: y - 110, width: 120, bold: true),
        line("Mais texto do corpo, que segue normalmente pela", y: y - 130),
        line("página, até o fim.", y: y - 144, width: 100),
      ]
    }
    let blocks = assemble([
      [line("Título do Livro", x: 100, y: 560, width: 200, size: 24)] + chapter(500), chapter(560),
    ])

    #expect(
      blocks.map(\.kind) == [.heading(level: 1)]
        + Array(
          repeating: [
            .heading(level: 1), .heading(level: 2), .paragraph, .heading(level: 3), .paragraph,
          ], count: 2
        )
        .flatMap { $0 })
  }

  private func assemble(_ pages: [[Line]]) -> [Block] {
    Block.assemble(
      pages.enumerated().map { index, lines in
        PageText(
          index: index, label: nil, bounds: CGRect(x: 0, y: 0, width: 400, height: 600),
          lines: lines)
      })
  }

  @Test func makesOneSectionOfConsecutiveHeadings() {
    let blocks = [
      Block(kind: .heading(level: 2), text: "CAPÍTULO II", pages: 1...1),
      Block(kind: .heading(level: 1), text: "DO LIVRO", pages: 1...1),
      Block(kind: .paragraph, text: "Agora que expliquei o título…", pages: 1...1),
      Block(kind: .heading(level: 2), text: "CAPÍTULO III", pages: 2...2),
    ]

    #expect(
      DocumentStructure.headingSections(blocks) == [
        Section(title: "CAPÍTULO II — DO LIVRO", level: 1, block: 0),
        Section(title: "CAPÍTULO III", level: 2, block: 3),
      ])
  }
}

@Suite struct TypographyTests {
  @Test func steadiesMeasuredSizesAroundTheBodySize() {
    let sizes = (0..<60).map { 12 + Double($0 % 50 - 25) * 0.08 }
    let lines =
      sizes.enumerated().map { index, size in
        line("Linha número \(index) do texto", y: 560 - CGFloat(index) * 9, size: size)
      } + [line("Título", y: 20, size: 20)]
    let page = PageText(
      index: 0, label: nil, bounds: CGRect(x: 0, y: 0, width: 400, height: 600), lines: lines)

    let typography = Typography(pages: [page])
    let steadied = typography.steadied([page])[0].lines.map(\.fontSize)

    #expect(typography.isMeasured)
    #expect(Set(steadied) == [typography.bodySize, 20])
  }

  @Test func leavesFontSizesAlone() {
    let lines = (0..<60).map {
      line("Linha \($0)", y: 560 - CGFloat($0) * 9, size: $0 % 10 == 0 ? 11.5 : 10)
    }
    let page = PageText(
      index: 0, label: nil, bounds: CGRect(x: 0, y: 0, width: 400, height: 600), lines: lines)

    let typography = Typography(pages: [page])

    #expect(!typography.isMeasured)
    #expect(typography.steadied([page]) == [page])
  }
}
