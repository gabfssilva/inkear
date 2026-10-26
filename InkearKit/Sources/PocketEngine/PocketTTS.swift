import FluidAudio
import Foundation
import Synthesis

/// Kyutai's PocketTTS, run through FluidAudio's Core ML pipeline.
///
/// The engine holds one manager per language pack, so `pt` and `en` each load their own models.
/// Every stage runs on the CPU, which Pocket is designed for: FluidAudio's Neural Engine variants
/// of the Portuguese pack (the `.ane` placement) produce a fraction of a second of noise, and the
/// standard rank-5 models run several times slower on the Neural Engine than on the CPU.
/// `quality` has no effect: FluidAudio exposes no step count for PocketTTS. Pocket has no speed
/// control, so `speed` time-stretches the output.
public actor PocketTTS: InstallableEngine {
  public static let id = EngineID.pocket
  public static let displayName = "Pocket TTS"
  public static let defaultVoice = "alba"
  /// The models FluidAudio downloads, fixed by its version.
  public static let revision = "FluidAudio 0.17.7"
  /// The bytes of both packs' standard models and constants.
  public static let downloadSize: Int64 = 1_059_211_132
  public static let supportedLanguages: Set<String> = Set(packs.keys)
  public static let usesQuality = false

  /// The voices every language pack ships, as FluidAudio's documentation lists them.
  static let voiceNames = [
    "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "cosette", "eponine",
    "estelle", "eve", "fantine", "george", "giovanni", "jane", "javert", "jean", "juergen", "lola",
    "marius", "mary", "michael", "paul", "peter_yearsley", "rafael", "stuart_bell", "vera",
  ]

  private static let packs: [String: PocketTtsLanguage] = ["en": .english, "pt": .portuguese]

  private static let placement = PocketTtsModelPlacement.gpu
  private static let computeUnits = PocketTtsComputeUnits.uniform(.cpuOnly)

  public nonisolated let voices = PocketTTS.voiceNames
  public nonisolated let languages = PocketTTS.supportedLanguages

  private let managers: [String: PocketTtsManager]

  private init(managers: [String: PocketTtsManager]) {
    self.managers = managers
  }

  public static func install(
    into directory: URL, progress: @escaping @Sendable (Double?) -> Void
  ) async throws {
    progress(nil)
    for language in packs.values {
      try await makeManager(language, in: directory).initialize()
    }
    try markInstalled(in: directory)
  }

  /// FluidAudio loads the models without telling which, so `progress` is never called.
  public static func load(
    from directory: URL, progress: @escaping @Sendable (LoadProgress) -> Void
  ) async throws -> PocketTTS {
    var managers: [String: PocketTtsManager] = [:]
    for (code, language) in packs {
      let manager = makeManager(language, in: directory)
      try await manager.initialize()
      managers[code] = manager
    }
    return PocketTTS(managers: managers)
  }

  public nonisolated func synthesize(
    _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality
  ) -> SpeechStream {
    SpeechStream { emit in
      try await self.synthesize(
        text, voice: voice, language: language, speed: speed, quality: quality, emit: emit)
    }
  }

  /// Emits a single part: the runtime returns the whole speech.
  private func synthesize(
    _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality,
    emit: @escaping @Sendable (Speech) -> Void
  ) async throws {
    guard let manager = managers[language] else {
      throw PocketEngineError.unsupportedLanguage(language)
    }
    let result = try await manager.synthesizeDetailed(
      text: SpeechText.normalized(text, language: language), voice: voice)
    let output = try SpeechEmitter(
      sampleRate: Double(PocketTtsConstants.audioSampleRate), speed: speed, emit: emit)
    try output.append(result.samples)
    try output.finish()
  }

  private static func makeManager(_ language: PocketTtsLanguage, in directory: URL)
    -> PocketTtsManager
  {
    PocketTtsManager(
      language: language, directory: directory, placement: placement,
      computeUnits: computeUnits)
  }
}

public enum PocketEngineError: Error, Equatable {
  case unsupportedLanguage(String)
}
