import Foundation

/// Downloads files from public Hugging Face repos.
public enum HuggingFace {
  /// The files of a public repo at a revision whose paths start with one of `prefixes` (all when
  /// empty).
  public struct Source: Sendable {
    public let repo: String
    public let revision: String
    public let prefixes: [String]

    public init(repo: String, revision: String = "main", prefixes: [String] = []) {
      self.repo = repo
      self.revision = revision
      self.prefixes = prefixes
    }
  }

  /// Downloads the files of `repo` at `revision` whose paths start with one of `prefixes` (all
  /// when empty) into `directory`, keeping their relative paths.
  ///
  /// Files already in `directory` with the expected size are skipped. `progress` receives the
  /// fraction of the total bytes done across all files.
  public static func download(
    repo: String, revision: String = "main", prefixes: [String] = [], into directory: URL,
    progress: @escaping @Sendable (Double) -> Void
  ) async throws {
    try await download(
      [Source(repo: repo, revision: revision, prefixes: prefixes)], into: directory,
      progress: progress)
  }

  /// Downloads the files of every source into `directory`, as one download of all their bytes.
  public static func download(
    _ sources: [Source], into directory: URL, progress: @escaping @Sendable (Double) -> Void
  ) async throws {
    var files: [(source: Source, entry: Entry)] = []
    for source in sources {
      files += try await listFiles(repo: source.repo, revision: source.revision)
        .filter { file in
          source.prefixes.isEmpty || source.prefixes.contains { file.path.hasPrefix($0) }
        }
        .map { (source, $0) }
    }
    let total = max(Double(files.reduce(0) { $0 + $1.entry.size }), 1)
    var completed: Int64 = 0
    for (source, file) in files {
      try Task.checkCancellation()
      let destination = directory.appending(path: file.path)
      if fileSize(at: destination) != file.size {
        let base = completed
        let reporter = ProgressReporter { written in
          progress(Double(base + written) / total)
        }
        let address = try url(
          "https://huggingface.co/\(encode(source.repo))/resolve/\(encode(source.revision))/\(encode(file.path))"
        )
        let (temporary, response) = try await URLSession.shared.download(
          for: URLRequest(url: address), delegate: reporter)
        try validate(response)
        try FileManager.default.createDirectory(
          at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: temporary, to: destination)
      }
      completed += file.size
      progress(Double(completed) / total)
    }
  }

  private struct Entry: Decodable {
    let type: String
    let path: String
    let size: Int64
  }

  /// The files of a repo tree, following the pages of the listing.
  private static func listFiles(repo: String, revision: String) async throws -> [Entry] {
    var page: URL? = try url(
      "https://huggingface.co/api/models/\(encode(repo))/tree/\(encode(revision))?recursive=true")
    var entries: [Entry] = []
    while let current = page {
      let (data, response) = try await URLSession.shared.data(from: current)
      let http = try validate(response)
      entries += try JSONDecoder().decode([Entry].self, from: data).filter { $0.type == "file" }
      page = nextPage(http)
    }
    return entries
  }

  /// The target of the `rel="next"` entry of a `Link` header, if any.
  private static func nextPage(_ response: HTTPURLResponse) -> URL? {
    guard
      let link = response.value(forHTTPHeaderField: "Link"),
      let part = link.components(separatedBy: ",").first(where: { $0.contains("rel=\"next\"") }),
      let open = part.firstIndex(of: "<"),
      let close = part.firstIndex(of: ">")
    else { return nil }
    return URL(string: String(part[part.index(after: open)..<close]))
  }

  @discardableResult
  private static func validate(_ response: URLResponse) throws -> HTTPURLResponse {
    guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
      throw URLError(.badServerResponse)
    }
    return http
  }

  private static func url(_ string: String) throws -> URL {
    guard let url = URL(string: string) else { throw URLError(.badURL) }
    return url
  }

  /// Percent-encodes each segment of `path`, keeping the slashes.
  private static func encode(_ path: String) -> String {
    path.components(separatedBy: "/")
      .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? $0 }
      .joined(separator: "/")
  }

  /// The size of the file at `url`, or nil when there is none.
  private static func fileSize(at url: URL) -> Int64? {
    (try? FileManager.default.attributesOfItem(atPath: url.path(percentEncoded: false)))?[.size] as? Int64
  }
}

/// Reports the bytes a download has written so far.
private final class ProgressReporter: NSObject, URLSessionDownloadDelegate, Sendable {
  private let onWrite: @Sendable (Int64) -> Void

  init(onWrite: @escaping @Sendable (Int64) -> Void) {
    self.onWrite = onWrite
    super.init()
  }

  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64
  ) {
    onWrite(totalBytesWritten)
  }

  /// Required by `URLSessionDownloadDelegate`; the file is moved into place by the caller.
  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) {}
}
