import Foundation
import PDFKit
import PDFStructure

public enum LibraryError: Error, Sendable, Equatable {
  case unreadablePDF
  case unknownDocument(DocumentID)
}

/// Owns the documents on disk: `<directory>/index.json` (the records) and, per document,
/// `<directory>/<uuid>/document.pdf` and `<directory>/<uuid>/structure.json`.
/// No method suspends internally.
public actor Library {
  private struct State {
    var records: [DocumentID: DocumentRecord]
    var subscribers: [UUID: AsyncStream<[DocumentRecord]>.Continuation] = [:]
  }

  public nonisolated let directory: URL
  private var state: State

  public init(directory: URL) throws {
    self.directory = directory
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let indexURL = directory.appending(path: "index.json")
    var records: [DocumentRecord] = []
    if FileManager.default.fileExists(atPath: indexURL.path) {
      records = try JSONFile.decode([DocumentRecord].self, from: Data(contentsOf: indexURL))
    }
    state = State(records: Dictionary(uniqueKeysWithValues: records.map { ($0.id, $0) }))
  }

  /// Most recently listened first (by bookmark date), then most recently added.
  public var records: [DocumentRecord] { sortedRecords }

  public func record(_ id: DocumentID) -> DocumentRecord? {
    state.records[id]
  }

  /// Yields the sorted records now and after every change. Supports many subscribers.
  public func updates() -> AsyncStream<[DocumentRecord]> {
    let (stream, continuation) = AsyncStream.makeStream(of: [DocumentRecord].self)
    let key = UUID()
    continuation.onTermination = { [weak self] _ in
      Task { await self?.unsubscribe(key) }
    }
    state.subscribers[key] = continuation
    continuation.yield(sortedRecords)
    return stream
  }

  /// Copies the PDF at `source` into the library with status `.preparing`. Throws `unreadablePDF`.
  public func add(pdfAt source: URL, at date: Date) throws -> DocumentRecord {
    guard let document = PDFDocument(url: source), document.pageCount > 0 else {
      throw LibraryError.unreadablePDF
    }
    let attributes = document.documentAttributes
    let id = DocumentID()
    let record = DocumentRecord(
      id: id,
      title: plausibleTitle(attributes?[PDFDocumentAttribute.titleAttribute] as? String)
        ?? source.deletingPathExtension().lastPathComponent,
      author: nonBlank(attributes?[PDFDocumentAttribute.authorAttribute] as? String)
        .flatMap { $0.count(where: \.isLetter) >= 3 ? $0 : nil },
      pageCount: document.pageCount,
      addedAt: date,
      status: .preparing,
      bookmark: nil,
      isFinished: false,
      settings: .standard)
    try FileManager.default.createDirectory(at: folder(of: id), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: source, to: pdfURL(of: id))
    var records = state.records
    records[id] = record
    try commit(records)
    return record
  }

  public nonisolated func pdfURL(of id: DocumentID) -> URL {
    file("document.pdf", of: id)
  }

  public func structure(of id: DocumentID) throws -> DocumentStructure {
    _ = try existing(id)
    return try JSONFile.decode(
      DocumentStructure.self, from: Data(contentsOf: file("structure.json", of: id)))
  }

  /// Writes structure.json and sets status `.ready(Overview(structure))`.
  public func store(_ structure: DocumentStructure, for id: DocumentID) throws -> DocumentRecord {
    _ = try existing(id)
    try JSONFile.encode(structure).write(to: file("structure.json", of: id), options: .atomic)
    return try update(id) { $0.with(status: .ready(DocumentRecord.Overview(structure))) }
  }

  public func fail(_ id: DocumentID, reason: String) throws {
    try update(id) { $0.with(status: .failed(reason: reason)) }
  }

  public func save(_ bookmark: Bookmark, for id: DocumentID) throws {
    try update(id) { $0.with(bookmark: bookmark) }
  }

  public func save(_ settings: ListeningSettings, for id: DocumentID) throws {
    try update(id) { $0.with(settings: settings) }
  }

  public func finish(_ id: DocumentID) throws {
    try update(id) { $0.with(isFinished: true) }
  }

  /// Deletes the document's directory too.
  public func remove(_ id: DocumentID) throws {
    _ = try existing(id)
    try FileManager.default.removeItem(at: folder(of: id))
    var records = state.records
    records[id] = nil
    try commit(records)
  }

  private var sortedRecords: [DocumentRecord] {
    state.records.values.sorted { lhs, rhs in
      switch (lhs.bookmark?.date, rhs.bookmark?.date) {
      case (let left?, let right?): left > right
      case (.some, nil): true
      case (nil, .some): false
      case (nil, nil): lhs.addedAt > rhs.addedAt
      }
    }
  }

  private func existing(_ id: DocumentID) throws -> DocumentRecord {
    guard let record = state.records[id] else { throw LibraryError.unknownDocument(id) }
    return record
  }

  @discardableResult
  private func update(
    _ id: DocumentID, _ change: (DocumentRecord) -> DocumentRecord
  ) throws -> DocumentRecord {
    let current = try existing(id)
    let updated = change(current)
    var records = state.records
    records[id] = updated
    try commit(records)
    return updated
  }

  /// Writes the index first, so a failed write leaves the in-memory state unchanged.
  private func commit(_ records: [DocumentID: DocumentRecord]) throws {
    try JSONFile.encode(Array(records.values))
      .write(to: directory.appending(path: "index.json"), options: .atomic)
    state.records = records
    let sorted = sortedRecords
    for continuation in state.subscribers.values {
      continuation.yield(sorted)
    }
  }

  private func unsubscribe(_ key: UUID) {
    state.subscribers[key] = nil
  }

  private nonisolated func folder(of id: DocumentID) -> URL {
    directory.appending(path: id.raw.uuidString, directoryHint: .isDirectory)
  }

  private nonisolated func file(_ name: String, of id: DocumentID) -> URL {
    folder(of: id).appending(path: name)
  }
}

/// Analyzes the document's PDF and stores the result. On an error other than cancellation, marks
/// the document failed and rethrows. On cancellation, rethrows without changing the record.
public func prepare(
  _ id: DocumentID, in library: Library, progress: @Sendable (AnalysisProgress) -> Void
) async throws -> DocumentRecord {
  do {
    guard let document = PDFDocument(url: library.pdfURL(of: id)) else {
      throw LibraryError.unreadablePDF
    }
    let structure = try await DocumentStructure.analyze(document, progress: progress)
    return try await library.store(structure, for: id)
  } catch {
    if error is CancellationError { throw error }
    try await library.fail(id, reason: error.localizedDescription)
    throw error
  }
}

private enum JSONFile {
  static func encode<T: Encodable>(_ value: T) throws -> Data {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    return try encoder.encode(value)
  }

  static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    return try decoder.decode(type, from: data)
  }
}

/// The title from the PDF's metadata, unless it is the path or file name of the document it was
/// made from, as converters often leave there.
private func plausibleTitle(_ text: String?) -> String? {
  nonBlank(text).flatMap { title in
    title.contains("://") || title.contains("\\")
      || title.firstMatch(of: /\.[A-Za-z0-9]{2,4}$/) != nil ? nil : title
  }
}

private func nonBlank(_ text: String?) -> String? {
  guard let trimmed = text?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty
  else { return nil }
  return trimmed
}
