import Foundation

/// The model families the app can read with.
public enum EngineID: String, Sendable, Codable, CaseIterable, Hashable {
  case supertonic, kokoro, pocket, qwen3, magpie, moss, chatterbox, omnivoice, phonon
}

/// A voice of one engine.
public struct VoiceID: Hashable, Sendable, Codable {
  public let engine: EngineID
  public let name: String

  public init(engine: EngineID, name: String) {
    self.engine = engine
    self.name = name
  }
}

/// A speech engine whose models are downloaded on demand into a directory of their own.
///
/// Every model runs on the Neural Engine or the CPU, never the GPU: iOS denies the GPU to apps in
/// the background, where the app keeps reading with the screen locked.
public protocol InstallableEngine: SpeechEngine {
  static var id: EngineID { get }
  /// The model's name as its authors write it, such as "Kokoro".
  static var displayName: String { get }
  /// Bytes `install` downloads, shown before downloading.
  static var downloadSize: Int64 { get }
  /// The languages the engine speaks, as ISO 639-1 codes, known before installing.
  static var supportedLanguages: Set<String> { get }
  /// The voice to start with, one of `voices` once loaded.
  static var defaultVoice: String { get }
  /// The version of the files `install` downloads, such as a Hugging Face commit. Files of another
  /// revision count as not installed.
  static var revision: String { get }

  /// Whether `directory` holds everything `load(from:)` needs.
  static func isInstalled(in directory: URL) -> Bool

  /// Downloads the models into `directory`, reporting the fraction done when it is known, and
  /// ends with ``markInstalled(in:)``.
  static func install(into directory: URL, progress: @escaping @Sendable (Double?) -> Void)
    async throws

  /// Deletes `directory` and whatever the system keeps for its models elsewhere.
  static func uninstall(from directory: URL) throws

  /// Loads every model the engine runs, so that no synthesis waits for one, reporting the
  /// functions loaded when the engine knows them.
  static func load(from directory: URL, progress: @escaping @Sendable (LoadProgress) -> Void)
    async throws -> Self
}

extension InstallableEngine {
  /// Whether `install` finished in `directory` for ``revision``, which it records with
  /// ``markInstalled(in:)``.
  public static func isInstalled(in directory: URL) -> Bool {
    (try? String(contentsOf: marker(in: directory), encoding: .utf8)) == revision
  }

  public static func uninstall(from directory: URL) throws {
    try FileManager.default.removeItem(at: directory)
  }

  /// Records that every file of ``revision`` is in `directory`.
  public static func markInstalled(in directory: URL) throws {
    try Data(revision.utf8).write(to: marker(in: directory))
  }

  private static func marker(in directory: URL) -> URL {
    directory.appending(path: "installed")
  }
}
