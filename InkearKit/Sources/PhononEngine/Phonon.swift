import Foundation
import PhononTTS
import Synthesis

/// Phonon 40M multilingual, from Gradium, run with its own Core ML core.
///
/// The bundle is exported by Gradium's `export_coreml` example and hosted in the public
/// repository `itsmechico/phonon-40m-multilingual-coreml`. The flow LM runs on the Neural
/// Engine (fp16); the Mimi decoder runs on the CPU (f32). Audio streams frame by frame, 80 ms
/// at a time, so the first audio arrives almost immediately.
///
/// The runtime normalizes text per language, so the engine passes the text as received, sets
/// the language on the runtime, and does not run `SpeechText` again.
public actor PhononEngine: InstallableEngine {
  public static let id = EngineID.phonon
  public static let displayName = "Phonon"
  /// The sum of the sizes of the files in `itsmechico/phonon-40m-multilingual-coreml`.
  public static let downloadSize: Int64 = 251_005_229
  public static let supportedLanguages: Set<String> = ["en", "fr", "de", "es", "pt"]
  public static let usesQuality = false
  /// A Brazilian Portuguese voice from the voice catalog, the language most readings start with.
  /// The voices are named "locale · Name" so the language of each shows in the voice grid.
  public static let defaultVoice = "pt-BR · Manuela-Lu"
  /// The commit of `itsmechico/phonon-40m-multilingual-coreml` the app downloads.
  public static let revision = "fc86f3e1d144977d033650bb75e634b8e68d0161"

  private static let repository = "itsmechico/phonon-40m-multilingual-coreml"
  private static let sampleRate = 24_000.0
  /// The locales the voice catalog names, by the ISO 639-1 code of the language they speak.
  private static let localesByLanguage: [String: Set<String>] = [
    "pt": ["pt-BR"],
    "en": ["en-US", "en-GB", "en"],
    "fr": ["fr"],
    "de": ["de"],
    "es": ["es-ES", "es-MX"],
  ]
  /// The language the app maps an ISO 639-1 code to, as the runtime names them.
  private static func language(_ code: String) -> Language {
    Language(rawValue: code) ?? .none
  }
  /// `voices` whose locale speaks `language`; all of them when the engine has no voice for it.
  public static func filtered(_ voices: [String], for language: String) -> [String] {
    guard let locales = localesByLanguage[language] else { return voices }
    let matching = voices.filter { voice in
      guard let locale = voice.split(separator: " · ", maxSplits: 1).first else { return false }
      return locales.contains(String(locale))
    }
    return matching.isEmpty ? voices : matching
  }

  private let tts: Phonon
  /// The voice `tts` speaks with, one of `voices`.
  private var voice: String
  /// The language `tts` normalizes text as.
  private var language: Language
  public nonisolated let voices: [String]
  public nonisolated let languages = supportedLanguages

  /// The voices whose locale speaks `language`, from the `"locale · Name"` names.
  public nonisolated func voices(for language: String) -> [String] {
    Self.filtered(voices, for: language)
  }

  private init(tts: Phonon) {
    self.tts = tts
    voices = tts.voices
    voice = tts.voices.first ?? Self.defaultVoice
    language = .none
  }

  public static func install(
    into directory: URL, progress: @escaping @Sendable (Double?) -> Void
  ) async throws {
    try await HuggingFace.download(
      repo: repository, revision: revision, prefixes: [], into: directory) {
      progress($0)
    }
    try markInstalled(in: directory)
  }

  public static func uninstall(from directory: URL) throws {
    try FileManager.default.removeItem(at: directory)
  }

  public static func load(
    from directory: URL, progress: @escaping @Sendable (LoadProgress) -> Void
  ) async throws -> PhononEngine {
    progress(LoadProgress(done: 0, total: 1, started: .now))
    let tts = try await Phonon.load(models: directory, language: .none)
    progress(LoadProgress(done: 1, total: 1, started: .now))
    return PhononEngine(tts: tts)
  }

  public nonisolated func synthesize(
    _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality
  ) -> SpeechStream {
    SpeechStream { emit in
      try await self.synthesize(
        text, voice: voice, language: language, speed: speed, quality: quality, emit: emit)
    }
  }

  /// Emits the audio the runtime streams, 80 ms per part, stretched to `speed`.
  private func synthesize(
    _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality,
    emit: @escaping @Sendable (Speech) -> Void
  ) async throws {
    guard Self.supportedLanguages.contains(language) else {
      throw PhononEngineError.unsupportedLanguage(language)
    }
    guard voices.contains(voice) else { throw PhononEngineError.unknownVoice(voice) }
    if voice != self.voice {
      try await tts.setVoice(voice)
      self.voice = voice
    }
    let normalized = Self.language(language)
    if normalized != self.language {
      try await tts.setLanguage(normalized)
      self.language = normalized
    }
    let output = try SpeechEmitter(sampleRate: Self.sampleRate, speed: speed, emit: emit)
    for try await chunk in tts.stream(text) {
      try output.append(chunk)
    }
    try output.finish()
  }
}

public enum PhononEngineError: Error, Equatable {
  case unknownVoice(String)
  case unsupportedLanguage(String)
}

extension PhononEngineError: LocalizedError {
  public var errorDescription: String? { "Phonon: \(self)" }
}
