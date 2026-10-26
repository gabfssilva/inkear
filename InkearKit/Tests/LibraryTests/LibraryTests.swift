import CoreGraphics
import Foundation
import PDFStructure
import Synthesis
import Testing

@testable import Library

private let sampleDate = Date(timeIntervalSince1970: 1_700_000_000)

@Suite struct LibraryTests {
  @Test func addCopiesThePDFAndStartsPreparing() async throws {
    try await withSandbox { root in
      let source = root.appending(path: "source.pdf")
      try makePDF(at: source, title: "  Dom Casmurro  ", author: " Machado de Assis ")
      let library = try Library(
        directory: root.appending(path: "library", directoryHint: .isDirectory))

      let record = try await library.add(pdfAt: source, at: sampleDate)

      #expect(record.title == "Dom Casmurro")
      #expect(record.author == "Machado de Assis")
      #expect(record.pageCount == 1)
      #expect(record.addedAt == sampleDate)
      #expect(record.status == .preparing)
      #expect(record.bookmark == nil)
      #expect(!record.isFinished)
      let copied = try Data(contentsOf: library.pdfURL(of: record.id))
      let original = try Data(contentsOf: source)
      #expect(copied == original)
    }
  }

  @Test(
    arguments: [
      nil, "   ", #"file://C:\cdpub\brascubas\brascubas.html"#, #"C:\docs\relatorio.doc"#,
      "relatorio-final.docx",
    ] as [String?])
  func titleFallsBackToTheFileNameWithoutExtension(title: String?) async throws {
    try await withSandbox { root in
      let source = root.appending(path: "Relatório anual.pdf")
      try makePDF(at: source, title: title, author: "fe")
      let library = try Library(
        directory: root.appending(path: "library", directoryHint: .isDirectory))

      let record = try await library.add(pdfAt: source, at: sampleDate)

      #expect(record.title == "Relatório anual")
      #expect(record.author == nil)
    }
  }

  @Test func recordsPersistAcrossInstances() async throws {
    try await withSandbox { root in
      let (library, record) = try await sampleLibrary(in: root)

      let reopened = try Library(directory: library.directory)
      let records = await reopened.records
      let found = await reopened.record(record.id)

      #expect(records == [record])
      #expect(found == record)
    }
  }

  @Test func storeMarksTheDocumentReadyWithItsOverview() async throws {
    try await withSandbox { root in
      let (library, record) = try await sampleLibrary(in: root)
      let structure = DocumentStructure(
        pageCount: 1,
        blocks: [
          Block(kind: .heading(level: 1), text: "Capítulo 1", pages: 0...0),
          Block(kind: .paragraph, text: "Era uma vez.", pages: 0...0),
        ],
        sections: [Section(title: "Capítulo 1", level: 1, block: 0)],
        contentsStart: nil,
        mainStart: 0,
        isRecognized: true)

      let stored = try await library.store(structure, for: record.id)

      let overview = DocumentRecord.Overview(structure)
      #expect(stored.status == .ready(overview))
      #expect(overview.sectionCount == 1)
      #expect(overview.isRecognized)
      #expect(overview.mainStart == ReadingPosition(block: 0, sentence: 0))
      #expect(overview.mainStartTitle == "Capítulo 1")

      let reopened = try Library(directory: library.directory)
      let read = try await reopened.structure(of: record.id)
      let found = await reopened.record(record.id)
      #expect(read == structure)
      #expect(found?.status == .ready(overview))
    }
  }

  @Test func bookmarkAndSettingsArePersisted() async throws {
    try await withSandbox { root in
      let (library, record) = try await sampleLibrary(in: root)
      let bookmark = Bookmark(
        position: ReadingPosition(block: 4, sentence: 2), section: 1, fraction: 0.25,
        date: sampleDate)
      let settings = ListeningSettings.standard
        .with(voice: VoiceID(engine: .kokoro, name: "pf_dora")).with(speed: 1.25)
        .with(language: "en").with(quality: .high)

      try await library.save(bookmark, for: record.id)
      try await library.save(settings, for: record.id)

      let reopened = try Library(directory: library.directory)
      let found = await reopened.record(record.id)
      #expect(found?.bookmark == bookmark)
      #expect(found?.settings == settings)
    }
  }

  @Test func settingsSavedBeforeEnginesAndQualityReadAsSupertonicAtNormal() throws {
    let saved = Data(#"{"voice":"F4","speed":1}"#.utf8)

    let settings = try JSONDecoder().decode(ListeningSettings.self, from: saved)

    #expect(
      settings
        == ListeningSettings(
          voice: VoiceID(engine: .supertonic, name: "F4"), speed: 1, language: nil,
          quality: .normal))
  }

  @Test func finishIsPersisted() async throws {
    try await withSandbox { root in
      let (library, record) = try await sampleLibrary(in: root)

      try await library.finish(record.id)

      let reopened = try Library(directory: library.directory)
      let found = await reopened.record(record.id)
      #expect(found?.isFinished == true)
    }
  }

  @Test func removeDeletesTheDocumentDirectory() async throws {
    try await withSandbox { root in
      let (library, record) = try await sampleLibrary(in: root)
      let folder = library.pdfURL(of: record.id).deletingLastPathComponent()
      #expect(FileManager.default.fileExists(atPath: folder.path))

      try await library.remove(record.id)

      #expect(!FileManager.default.fileExists(atPath: folder.path))
      let found = await library.record(record.id)
      let records = await library.records
      #expect(found == nil)
      #expect(records.isEmpty)
    }
  }

  @Test func updatesYieldCurrentRecordsThenEveryChange() async throws {
    try await withSandbox { root in
      let (library, first) = try await sampleLibrary(in: root)
      let stream = await library.updates()
      var iterator = stream.makeAsyncIterator()

      let initial = await iterator.next()
      #expect(initial == [first])

      let second = try await library.add(
        pdfAt: library.pdfURL(of: first.id), at: sampleDate.addingTimeInterval(60))
      let changed = await iterator.next()
      #expect(changed == [second, first])
    }
  }

  @Test func addRejectsFilesThatArePDFs() async throws {
    try await withSandbox { root in
      let source = root.appending(path: "notes.pdf")
      try Data("not a pdf".utf8).write(to: source)
      let library = try Library(
        directory: root.appending(path: "library", directoryHint: .isDirectory))

      await #expect(throws: LibraryError.unreadablePDF) {
        try await library.add(pdfAt: source, at: sampleDate)
      }
      let records = await library.records
      #expect(records.isEmpty)
    }
  }
}

/// Runs `body` with a fresh directory that is removed afterwards.
private func withSandbox<T>(_ body: (URL) async throws -> T) async throws -> T {
  let root = URL.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
  try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: root) }
  return try await body(root)
}

/// A library under `root` holding one PDF titled "Sample", added with `sampleDate`.
private func sampleLibrary(in root: URL) async throws -> (Library, DocumentRecord) {
  let source = root.appending(path: "source.pdf")
  try makePDF(at: source, title: "Sample")
  let library = try Library(directory: root.appending(path: "library", directoryHint: .isDirectory))
  let record = try await library.add(pdfAt: source, at: sampleDate)
  return (library, record)
}

/// Writes a one-page PDF with a filled rectangle and the given document info.
private func makePDF(at url: URL, title: String? = nil, author: String? = nil) throws {
  var info: [String: String] = [:]
  if let title { info[kCGPDFContextTitle as String] = title }
  if let author { info[kCGPDFContextAuthor as String] = author }
  let context = try #require(
    CGContext(url as CFURL, mediaBox: nil, info.isEmpty ? nil : info as CFDictionary))
  context.beginPDFPage(nil)
  context.setFillColor(gray: 0.5, alpha: 1)
  context.fill(CGRect(x: 72, y: 72, width: 200, height: 200))
  context.endPDFPage()
  context.closePDF()
}
