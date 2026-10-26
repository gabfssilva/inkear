import Foundation
import PDFKit
import Testing

@testable import PDFStructure

/// Regression checks against the open PDFs in `corpus/` (see its `manifest.json`).
///
/// Documents that need OCR take about a minute each; skip the suite with
/// `swift test --skip CorpusTests` for a quick run.
@Suite struct CorpusTests {
  struct Expectation: Sendable, CustomTestStringConvertible {
    let file: String
    /// How the block where the main text starts begins.
    var main: String
    /// How the first block of the table of contents begins, when the document has one.
    var contents: String? = nil
    /// Section titles the document must have.
    var sections: [String] = []
    /// Sentences some block must contain, intact.
    var sentences: [String] = []

    var testDescription: String { file }
  }

  static let corpus = URL(filePath: #filePath).deletingLastPathComponent()
    .appending(path: "../../../corpus").standardized

  static let expectations: [Expectation] = [
    .init(
      file: "en-book-think-python", main: "Chapter 1", contents: "Contents",
      sections: ["The way of the program", "What is a program?"]),
    .init(file: "en-book-ocr-alice", main: "CHAPTER I"),
    .init(file: "en-scan-alice", main: "CHAPTER I"),
    .init(
      file: "en-paper-2col-bert", main: "1 Introduction",
      sentences: ["We introduce a new language representation model called BERT"]),
    .init(file: "en-paper-t5", main: "1. Introduction"),
    .init(
      file: "pt-book-bras-cubas", main: "CAPÍTULO 1",
      sentences: ["Algum tempo hesitei se devia abrir estas memórias pelo princípio ou pelo fim"]),
    .init(
      file: "pt-book-dom-casmurro", main: "CAPÍTULO PRIMEIRO",
      sections: ["CAPÍTULO II — DO LIVRO"],
      sentences: ["Uma noite destas, vindo da cidade para o Engenho Novo"]),
    .init(
      file: "pt-book-o-mulato", main: "O Mulato", contents: "Índice",
      sections: ["Capítulo 1", "Capítulo 2"]),
    .init(
      file: "pt-book-pro-git", main: "Introdução", contents: "Table of Contents",
      sections: ["Primeiros Passos", "Fundamentos do Git"]),
    .init(file: "pt-paper-rbef", main: "1. Introdução"),
    .init(
      file: "pt-scan-esau-e-jacob", main: "CAPITULO PRIMEIRO",
      sentences: ["Era a primeira vez que as duas iam ao morro do Castello"]),
  ]

  @Test(arguments: expectations)
  func findsTheStructureOf(_ expected: Expectation) async throws {
    let document = try #require(
      PDFDocument(url: Self.corpus.appending(path: "\(expected.file).pdf")))

    let structure = try await DocumentStructure.analyze(document)

    let main = try #require(structure.mainStart.map { structure.blocks[$0].text })
    #expect(main.hasPrefix(expected.main))
    #expect(structure.contentsStart.map { structure.blocks[$0].text } == expected.contents)
    let titles = Set(structure.sections.map(\.title))
    for section in expected.sections {
      #expect(titles.contains(section), "missing section \(section)")
    }
    for sentence in expected.sentences {
      #expect(structure.blocks.contains { $0.text.contains(sentence) }, "missing \(sentence)")
    }
  }
}
