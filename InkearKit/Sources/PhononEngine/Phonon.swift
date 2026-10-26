import Foundation
import PhononTTS
import Synthesis

/// Phonon 40M, from Gradium, run with its own Core ML core: one model per language.
///
/// Gradium trains a 40M checkpoint for each language, and each speaks its language better than
/// the multilingual one. Their Core ML bundles are exported by Gradium's `export_coreml` example,
/// each with the voices of its language, and hosted in the public repositories
/// `itsmechico/phonon-40m-<language>-coreml`. Installing downloads all of them, so the engine
/// reads every language it lists; one is loaded at a time, the one of the text being read.
///
/// The flow LM runs on the Neural Engine (fp16); the Mimi decoder runs on the CPU (f32). Audio
/// streams frame by frame, 80 ms at a time, so the first audio arrives almost immediately.
///
/// The runtime normalizes text per language, so the engine passes the text as received, sets
/// the language on the runtime, and does not run `SpeechText` again.
public actor PhononEngine: InstallableEngine {
  public static let id = EngineID.phonon
  public static let displayName = "Phonon"
  public static let downloadSize: Int64 = models.reduce(0) { $0 + $1.size }
  public static let supportedLanguages = Set(models.map(\.language))
  public static let usesQuality = true
  /// A Brazilian Portuguese voice from the voice catalog, the language most readings start with.
  /// The voices are named "locale · Name" so the language of each shows in the voice grid.
  public static let defaultVoice = "pt-BR · Manuela-Lu"
  /// Every model's commit, so that a change to any of them counts as a new version.
  public static let revision = models.map(\.revision).joined(separator: "+")

  /// One language's bundle: the repository, its commit, and the sum of its files' sizes.
  struct Model: Sendable {
    let language: String
    let revision: String
    let size: Int64

    var repository: String { "itsmechico/phonon-40m-\(language)-coreml" }
  }

  /// The models install downloads, Portuguese first: the language most readings start with.
  static let models = [
    Model(language: "pt", revision: "3e9329f9ee9dad26c0f12120cd5688b9f421c424", size: 222_167_497),
    Model(language: "en", revision: "e6a425fa66b0e0e0a175ebaa2355562effc688c2", size: 228_573_424),
    Model(language: "fr", revision: "fbfb25fa4fea7f1da3ad40f983dcedebe88e9198", size: 222_934_946),
    Model(language: "de", revision: "a14591e958ff2b9687d95a8e4ad34321bda36c68", size: 222_422_736),
    Model(language: "es", revision: "fd4737407661ec62bfd40532523769c26b1208d1", size: 225_498_816),
  ]

  private static let sampleRate = 24_000.0
  /// The locales the voice catalog names, by the ISO 639-1 code of the language they speak.
  private static let localesByLanguage: [String: Set<String>] = [
    "pt": ["pt-BR"],
    "en": ["en-US", "en-GB", "en"],
    "fr": ["fr"],
    "de": ["de"],
    "es": ["es-ES", "es-MX"],
  ]
  /// Euler steps of the flow per frame. The checkpoints were distilled for one; more give
  /// cleaner audio at the cost of one more run of the small flow net each.
  static func steps(for quality: SpeechQuality) -> Int {
    switch quality {
    case .low: 1
    case .normal: 3
    case .high: 6
    case .highest: 12
    }
  }

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

  /// Where install puts `language`'s model.
  private static func directory(of language: String, in directory: URL) -> URL {
    directory.appending(path: language)
  }

  /// The voices of a bundle, as its `bundle.json` lists them, read without loading the model.
  private static func voices(inBundle directory: URL) throws -> [String] {
    struct Bundle: Decodable { let voices: [String] }
    let data = try Data(contentsOf: directory.appending(path: "bundle.json"))
    return try JSONDecoder().decode(Bundle.self, from: data).voices
  }

  /// The model loaded now, the one of the language last read.
  private struct Loaded {
    let language: String
    let tts: Phonon
    /// The voice `tts` speaks with, one of its `voices`.
    var voice: String
    /// The flow steps per frame `tts` takes.
    var steps: Int
  }

  private let directory: URL
  /// Each language's voices, by ISO 639-1 code.
  private let voicesByLanguage: [String: [String]]
  private var loaded: Loaded
  public nonisolated let voices: [String]
  public nonisolated let languages = supportedLanguages

  /// The voices whose locale speaks `language`, from the `"locale · Name"` names.
  public nonisolated func voices(for language: String) -> [String] {
    Self.filtered(voices, for: language)
  }

  private init(directory: URL, voicesByLanguage: [String: [String]], loaded: Loaded) {
    self.directory = directory
    self.voicesByLanguage = voicesByLanguage
    self.loaded = loaded
    voices = Self.models.flatMap { voicesByLanguage[$0.language] ?? [] }
  }

  public static func install(
    into directory: URL, progress: @escaping @Sendable (Double?) -> Void
  ) async throws {
    // One download of all the models' bytes, each model into its language's directory.
    let total = Double(downloadSize)
    var done: Int64 = 0
    for model in models {
      let (base, share) = (Double(done), Double(model.size))
      try await HuggingFace.download(
        repo: model.repository, revision: model.revision, prefixes: [],
        into: Self.directory(of: model.language, in: directory)
      ) { progress((base + $0 * share) / total) }
      done += model.size
    }
    try markInstalled(in: directory)
  }

  public static func uninstall(from directory: URL) throws {
    try FileManager.default.removeItem(at: directory)
  }

  /// Compiles every model for this device, the slow part of a first launch, so that switching
  /// language later only loads one, and keeps Portuguese's loaded.
  public static func load(
    from directory: URL, progress: @escaping @Sendable (LoadProgress) -> Void
  ) async throws -> PhononEngine {
    var voicesByLanguage: [String: [String]] = [:]
    for (index, model) in models.enumerated() {
      progress(LoadProgress(done: index, total: models.count, started: .now))
      let bundle = Self.directory(of: model.language, in: directory)
      voicesByLanguage[model.language] = try voices(inBundle: bundle)
      if Phonon.needsCompiling(bundle) {
        _ = try await Phonon.load(models: bundle, language: language(model.language))
      }
    }
    progress(LoadProgress(done: models.count, total: models.count, started: .now))
    let loaded = try await load(models[0].language, from: directory)
    return PhononEngine(directory: directory, voicesByLanguage: voicesByLanguage, loaded: loaded)
  }

  private static func load(_ language: String, from directory: URL) async throws -> Loaded {
    let tts = try await Phonon.load(
      models: Self.directory(of: language, in: directory), language: Self.language(language))
    return Loaded(language: language, tts: tts, voice: tts.voice, steps: tts.steps)
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
    guard let languageVoices = voicesByLanguage[language] else {
      throw PhononEngineError.unsupportedLanguage(language)
    }
    guard voices.contains(voice) else { throw PhononEngineError.unknownVoice(voice) }
    if language != loaded.language {
      loaded = try await Self.load(language, from: directory)
    }
    // Each model has only its language's voices: a voice of another language reads with the
    // first of this one's, since the model is what speaks the language.
    let speaking = languageVoices.contains(voice) ? voice : languageVoices.first ?? loaded.voice
    if speaking != loaded.voice {
      try await loaded.tts.setVoice(speaking)
      loaded.voice = speaking
    }
    let steps = Self.steps(for: quality)
    if steps != loaded.steps {
      try await loaded.tts.setSteps(steps)
      loaded.steps = steps
    }
    let output = try SpeechEmitter(sampleRate: Self.sampleRate, speed: speed, emit: emit)
    for try await chunk in loaded.tts.stream(text) {
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
