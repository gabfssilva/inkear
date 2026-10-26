import Testing

@testable import PDFStructure

@Suite struct SentencesTests {
  @Test func splitsAtTerminalPunctuation() {
    let block = paragraph("Ele chegou cedo. Ela ainda dormia? Não! Acordaram juntos.")

    #expect(
      block.sentences.map { String(block.text[$0]) } == [
        "Ele chegou cedo.", "Ela ainda dormia?", "Não!", "Acordaram juntos.",
      ])
  }

  @Test func keepsClosingQuotesWithTheirSentence() {
    let block = paragraph("Ela disse: \"Vamos embora.\" Ninguém respondeu.")

    #expect(
      block.sentences.map { String(block.text[$0]) } == [
        "Ela disse: \"Vamos embora.\"", "Ninguém respondeu.",
      ])
  }

  @Test func textWithoutPunctuationIsOneSentence() {
    let text = "uma linha sem pontuação alguma"
    let block = paragraph(text)

    #expect(block.sentences.map { String(block.text[$0]) } == [text])
  }

  @Test(arguments: ["", " \n\t "])
  func blankTextHasNoSentences(text: String) {
    #expect(paragraph(text).sentences.isEmpty)
  }

  @Test func rangesAreTrimmedAndInOrder() {
    let block = paragraph("  Primeira frase.   \n\nSegunda frase aqui!  Terceira? ")
    let ranges = block.sentences

    #expect(ranges.count >= 2)
    #expect(
      ranges.allSatisfy { range in
        let content = block.text[range]
        return content.first?.isWhitespace == false && content.last?.isWhitespace == false
      })
    #expect(zip(ranges, ranges.dropFirst()).allSatisfy { $0.upperBound <= $1.lowerBound })
  }

  @Test func detectsPortuguese() {
    let structure = document([paragraph(portuguese)])

    #expect(structure.language == "pt")
  }

  @Test func detectsEnglish() {
    let structure = document([paragraph(english)])

    #expect(structure.language == "en")
  }

  @Test func languageComesFromTheMainTextOnly() {
    let structure = document(
      [heading("Sumário"), paragraph(portuguese), paragraph(english), paragraph(english)],
      mainStart: 2)

    #expect(structure.language == "en")
  }

  @Test func languageIsNilWithoutParagraphs() {
    #expect(document([]).language == nil)
    #expect(document([heading("Chapter One")]).language == nil)
  }

  private let portuguese =
    "A cidade acordou cedo naquela manhã de inverno. As crianças corriam pelas ruas enquanto "
    + "os pais preparavam o café da manhã. Ninguém imaginava o que estava por vir naquele dia."

  private let english =
    "The city woke up early that winter morning. The children ran through the streets while "
    + "their parents prepared breakfast. Nobody imagined what was about to happen that day."

  private func paragraph(_ text: String) -> Block {
    Block(kind: .paragraph, text: text, pages: 0...0)
  }

  private func heading(_ text: String) -> Block {
    Block(kind: .heading(level: 1), text: text, pages: 0...0)
  }

  private func document(_ blocks: [Block], mainStart: Int? = nil) -> DocumentStructure {
    DocumentStructure(
      pageCount: 1, blocks: blocks, sections: [], contentsStart: nil, mainStart: mainStart,
      isRecognized: false)
  }
}
