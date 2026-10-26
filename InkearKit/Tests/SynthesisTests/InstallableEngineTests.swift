import Foundation
import Testing

@testable import Synthesis

@Suite struct InstallableEngineTests {
  @Test func anEngineInstalledUnderAPathWithSpacesStaysInstalled() throws {
    let directory = FileManager.default.temporaryDirectory
      .appending(path: "Application Support \(UUID())").appending(path: "Models/silent")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }

    #expect(!Silent.isInstalled(in: directory))
    try Silent.markInstalled(in: directory)
    #expect(Silent.isInstalled(in: directory))
  }

  @Test func anInstallOfAnotherRevisionIsNotInstalled() throws {
    let directory = FileManager.default.temporaryDirectory.appending(path: "silent-\(UUID())")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    try Data("an older revision".utf8).write(to: directory.appending(path: "installed"))

    #expect(!Silent.isInstalled(in: directory))
  }
}

/// An engine with nothing to download.
private struct Silent: InstallableEngine {
  static let id = EngineID.kokoro
  static let displayName = "Silent"
  static let downloadSize: Int64 = 0
  static let supportedLanguages: Set<String> = ["pt"]
  static let usesQuality = false
  static let defaultVoice = "silent"
  static let revision = "1"
  let voices = [defaultVoice]
  let languages = supportedLanguages

  static func install(into directory: URL, progress: @escaping @Sendable (Double?) -> Void) throws
  {
    try markInstalled(in: directory)
  }

  static func load(from directory: URL, progress: @escaping @Sendable (LoadProgress) -> Void)
    -> Silent
  {
    Silent()
  }

  func synthesize(
    _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality
  ) -> SpeechStream {
    SpeechStream { emit in emit(Speech(samples: [], sampleRate: 24_000)) }
  }
}
