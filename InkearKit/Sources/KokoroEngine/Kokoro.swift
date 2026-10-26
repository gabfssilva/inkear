import FluidAudio
import Foundation
import Synthesis

/// Kokoro 82M on FluidAudio's seven-stage Core ML runtime, in Brazilian Portuguese and English.
///
/// English goes through FluidAudio's own front end. Portuguese, which FluidAudio does not read,
/// goes through ``PortuguesePhonemes``; both drive the same models. `speed` is Kokoro's own
/// duration scale. `quality` has no effect: Kokoro synthesizes in one pass, with no setting that
/// trades time for quality.
public actor Kokoro: InstallableEngine {
  public static let id = EngineID.kokoro
  public static let displayName = "Kokoro"
  public static let defaultVoice = "pf_dora"
  /// The models FluidAudio downloads, fixed by its version.
  public static let revision = "FluidAudio 0.17.7"
  /// The seven stages with their vocabulary and default voice, the English G2P model with the JSON
  /// files FluidAudio fetches beside it, and the other six voices as JSON.
  public static let downloadSize: Int64 = 141_018_130
  public static let supportedLanguages: Set<String> = ["pt", "en"]
  public static let usesQuality = false

  /// Kokoro's Brazilian Portuguese voices, then four American English ones.
  static let voiceNames = [
    "pf_dora", "pm_alex", "pm_santa", "af_heart", "af_bella", "am_michael", "am_adam",
  ]

  /// FluidAudio's default runs the noise and tail stages on the GPU before OS 27.
  private static let computeUnits = KokoroAneComputeUnits.aneTailCpu
  private static let sampleRate = Double(KokoroAneConstants.sampleRate)
  private static let pauses = Set(";:,.!?—…".unicodeScalars)

  public nonisolated let voices = Kokoro.voiceNames
  public nonisolated let languages = Kokoro.supportedLanguages

  /// The Brazilian Portuguese voices (`pf_`/`pm_`), then the American English ones.
  public nonisolated func voices(for language: String) -> [String] {
    switch language {
    case "pt": ["pf_dora", "pm_alex", "pm_santa"]
    case "en": ["af_heart", "af_bella", "am_michael", "am_adam"]
    default: voices
    }
  }

  private let manager: KokoroAneManager

  private init(manager: KokoroAneManager) {
    self.manager = manager
  }

  /// Downloads the models and every voice. Besides `directory`, FluidAudio keeps the English G2P
  /// model and lexicon in its shared cache, `fluidaudio/Models/kokoro` in Application Support.
  public static func install(
    into directory: URL, progress: @escaping @Sendable (Double?) -> Void
  ) async throws {
    progress(nil)
    try await makeManager(directory).initialize(preloadVoices: Set(voiceNames))
    try markInstalled(in: directory)
  }

  /// FluidAudio loads the models without telling which, so `progress` is never called.
  public static func load(
    from directory: URL, progress: @escaping @Sendable (LoadProgress) -> Void
  ) async throws -> Kokoro {
    let manager = makeManager(directory)
    try await manager.initialize()
    return Kokoro(manager: manager)
  }

  public nonisolated func synthesize(
    _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality
  ) -> SpeechStream {
    SpeechStream { emit in
      try await self.synthesize(
        text, voice: voice, language: language, speed: speed, quality: quality, emit: emit)
    }
  }

  /// Emits one part per piece. The model applies the speed itself.
  private func synthesize(
    _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality,
    emit: @escaping @Sendable (Speech) -> Void
  ) async throws {
    guard voices.contains(voice) else { throw KokoroError.unknownVoice(voice) }
    let phonemes = try await phonemize(
      SpeechText.normalized(text, language: language), language: language)
    let silence = [Float](repeating: 0, count: Int(0.3 * Self.sampleRate / speed))
    let output = try SpeechEmitter(sampleRate: Self.sampleRate, speed: 1, emit: emit)
    for (index, piece) in Self.pieces(of: phonemes).enumerated() {
      try output.append(
        (index > 0 ? silence : []) + (try await speak(piece, voice: voice, speed: Float(speed))))
    }
    try output.finish()
  }

  private func phonemize(_ text: String, language: String) async throws -> String {
    switch language {
    case "pt": return PortuguesePhonemes.phonemes(for: text)
    case "en": return try await manager.phonemes(for: text)
    default: throw KokoroError.unsupportedLanguage(language)
    }
  }

  /// Synthesizes `phonemes`, halving them while their audio runs past the runtime's frame cap,
  /// which long pieces reach at slow speeds.
  private func speak(_ phonemes: String, voice: String, speed: Float) async throws -> [Float] {
    do {
      return try await manager.synthesizeFromPhonemesDetailed(phonemes, voice: voice, speed: speed)
        .samples
    } catch KokoroAneError.acousticFramesExceedCap(let have, let cap) {
      guard let (head, tail) = Self.halves(of: phonemes) else {
        throw KokoroAneError.acousticFramesExceedCap(have: have, cap: cap)
      }
      return try await speak(head, voice: voice, speed: speed)
        + speak(tail, voice: voice, speed: speed)
    }
  }

  /// `phonemes` in pieces of at most the runtime's 510 phonemes, each cut after the last pause
  /// mark in the second half of what fits, or else at the last space. Pieces without a letter are
  /// dropped.
  private static func pieces(of phonemes: String) -> [String] {
    let limit = KokoroAneConstants.maxPhonemeLength
    var rest = Array(phonemes.unicodeScalars)[...]
    var pieces: [String] = []
    while !rest.isEmpty {
      var end = rest.endIndex
      if rest.count > limit {
        let window = rest.prefix(limit)
        let pause = window.lastIndex { pauses.contains($0) }.map { $0 + 1 }
        let cut =
          pause.flatMap { $0 - rest.startIndex > limit / 2 ? $0 : nil }
          ?? window.lastIndex(of: " ")
        end = cut.flatMap { $0 > rest.startIndex ? $0 : nil } ?? window.endIndex
      }
      let piece = string(rest[..<end]).trimmingCharacters(in: .whitespaces)
      if piece.contains(where: \.isLetter) { pieces.append(piece) }
      rest = rest[end...]
    }
    return pieces
  }

  /// `phonemes` cut in two at the clause boundary nearest its middle, or else the nearest space.
  private static func halves(of phonemes: String) -> (String, String)? {
    let scalars = Array(phonemes.unicodeScalars)
    let middle = scalars.count / 2
    let spaces = scalars.indices.filter { scalars[$0] == " " }
    let nearest = { (cuts: [Int]) in cuts.min { abs($0 - middle) < abs($1 - middle) } }
    guard
      let cut = nearest(spaces.filter { $0 > 0 && pauses.contains(scalars[$0 - 1]) })
        ?? nearest(spaces)
    else { return nil }
    return (string(scalars[..<cut]), string(scalars[(cut + 1)...]))
  }

  private static func string(_ scalars: ArraySlice<Unicode.Scalar>) -> String {
    var string = ""
    string.unicodeScalars.append(contentsOf: scalars)
    return string
  }

  private static func makeManager(_ directory: URL) -> KokoroAneManager {
    KokoroAneManager(directory: directory, computeUnits: computeUnits, version: .legacy)
  }
}

public enum KokoroError: Error, Equatable {
  case unknownVoice(String)
  case unsupportedLanguage(String)
}
