#if canImport(CoreAI)
  import CoreAI
  import Foundation
  import Synthesis

  /// OmniVoice (k2-fsa), a masked-generation TTS.
  ///
  /// The step runs with Core AI on the Neural Engine, from a model compiled ahead of time for the
  /// device's architecture; the decoder runs with Core ML.
  ///
  /// A voice is a reference clip the model continues, so the voice sets the pace and the loudness.
  /// The text is generated in pieces of whole sentences, joined while they fit in `maxFrames`.
  public actor OmniVoice: InstallableEngine {
    public static let id = EngineID.omnivoice
    public static let displayName = "OmniVoice"
    public static let downloadSize: Int64 = 1_324_169_088
    public static let supportedLanguages: Set<String> = ["pt", "en"]
    public static let usesQuality = true
    public static let defaultVoice = "f1"
    public static let revision = "\(coreMLRevision) \(coreAIRevision)"
    public nonisolated let voices = voiceNames
    public nonisolated let languages = supportedLanguages

    private static let coreMLRepository = "itsmechico/omnivoice-coreml-ane"
    private static let coreMLRevision = "b679cd13783a616bbe9782a43992df2159d95411"
    private static let coreAIRepository = "itsmechico/omnivoice-coreai"
    private static let coreAIRevision = "2ff4b4bbaccd74963ce657490f415bd746174be8"
    /// The architectures the step is compiled for.
    private static let architectures: Set<String> = [
      "h13g", "h14g", "h15g", "h16g", "h16p", "h17g", "h17p", "h18p", "h19p",
    ]
    private static let voiceNames = ["f1", "f2", "m1", "m2", "rogerio"]
    private static let prefixes = [
      "HiggsAudio2Decoder.mlmodelc/", "omnivoice.json", "tokenizer.json", "tokenizer_config.json",
      "voices/",
    ]
    /// Pieces with more frames than this are split in two before generation.
    private static let maxFrames = 300
    /// The one bucket target, in frames, the engine runs: about 13 s, which holds `maxFrames`.
    /// Shorter text is padded to it: every bucket holds its own 1.2 GB once loaded, and switching
    /// between loaded ones takes seconds.
    private static let targets: Set<Int> = [320]
    private let bundle: OmniVoiceBundle

    private init(bundle: OmniVoiceBundle) {
      self.bundle = bundle
    }

    public static func install(
      into directory: URL, progress: @escaping @Sendable (Double?) -> Void
    ) async throws {
      let architecture = AIModel.deviceArchitectureName
      guard architectures.contains(architecture) else {
        throw OmniVoiceError.unsupportedDevice(architecture)
      }
      // Another revision's step is about to be overwritten.
      try deleteCompiledStep(in: directory)
      try await HuggingFace.download(
        [
          .init(repo: coreMLRepository, revision: coreMLRevision, prefixes: prefixes),
          .init(
            repo: coreAIRepository, revision: coreAIRevision,
            prefixes: [OmniVoiceBundle.stepPackage + "/"]),
        ], into: directory
      ) { progress($0) }
      try markInstalled(in: directory)
    }

    public static func uninstall(from directory: URL) throws {
      try deleteCompiledStep(in: directory)
      try FileManager.default.removeItem(at: directory)
    }

    /// Deletes the step Core AI compiled from `directory`, which its persistent policy keeps.
    /// Core AI finds the entries by the package's hash file, so a partial download has none.
    private static func deleteCompiledStep(in directory: URL) throws {
      let step = directory.appending(path: OmniVoiceBundle.stepPackage)
      let hash = step.appending(path: "main.hash").path(percentEncoded: false)
      guard FileManager.default.fileExists(atPath: hash) else { return }
      try AIModelCache.default.deleteEntries(for: step)
    }

    public static func load(
      from directory: URL, progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws -> OmniVoice {
      OmniVoice(
        bundle: try await OmniVoiceBundle(
          directory: directory, voiceNames: voiceNames, targets: targets, progress: progress))
    }

    public nonisolated func synthesize(
      _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality
    ) -> SpeechStream {
      SpeechStream { emit in
        try await self.synthesize(
          text, voice: voice, language: language, speed: speed, quality: quality, emit: emit)
      }
    }

    /// Emits one part per utterance.
    private func synthesize(
      _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality,
      emit: @escaping @Sendable (Speech) -> Void
    ) async throws {
      guard Self.supportedLanguages.contains(language) else {
        throw OmniVoiceError.unsupportedLanguage(language)
      }
      guard let reference = bundle.voices[voice] else { throw OmniVoiceError.unknownVoice(voice) }
      guard let steps = bundle.qualitySteps[quality.rawValue] else {
        throw OmniVoiceError.invalidConfiguration("quality \(quality.rawValue)")
      }
      let normalized = SpeechText.normalized(text, language: language)
      let output = try SpeechEmitter(
        sampleRate: OmniVoiceBundle.sampleRate, speed: speed, emit: emit)
      for utterance in utterances(in: normalized, language: language, voice: reference) {
        try output.append(
          try await speak(utterance, language: language, voice: reference, steps: steps))
      }
      try output.finish()
    }

    /// The sentences of `text`, each ending in punctuation, joined while the result still fits.
    private func utterances(in text: String, language: String, voice: OmniVoiceVoice) -> [String] {
      var utterances: [String] = []
      for sentence in Self.sentences(of: text).map(Self.terminated) {
        if let last = utterances.last,
          fits(last + " " + sentence, language: language, voice: voice)
        {
          utterances[utterances.count - 1] = last + " " + sentence
        } else {
          utterances.append(sentence)
        }
      }
      return utterances
    }

    private func fits(_ text: String, language: String, voice: OmniVoiceVoice) -> Bool {
      let conditioning = bundle.frontEnd.conditioning(text, language: language, voice: voice)
      return conditioning.target <= Self.maxFrames
        && bundle.frontEnd.bucket(for: conditioning) != nil
    }

    /// The mono samples of `text`, split at its middle clause or space when it does not fit.
    private func speak(
      _ text: String, language: String, voice: OmniVoiceVoice, steps: Int
    ) async throws -> [Float] {
      if fits(text, language: language, voice: voice) {
        return try await bundle.generate(text, language: language, voice: voice, steps: steps)
      }
      guard let parts = Self.halves(of: text) else { throw OmniVoiceError.tooLong }
      let head = try await speak(parts.0, language: language, voice: voice, steps: steps)
      let tail = try await speak(parts.1, language: language, voice: voice, steps: steps)
      return head + tail
    }

    /// The sentences of `text`, each without the whitespace that followed it.
    private static func sentences(of text: String) -> [String] {
      let ends = text.ranges(of: /[.!?…]["'”’»)]*\s+/).map(\.upperBound)
      return zip([text.startIndex] + ends, ends + [text.endIndex])
        .map { text[$0..<$1].trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    }

    /// `sentence` with a period appended when it has no terminal punctuation.
    private static func terminated(_ sentence: String) -> String {
      sentence.contains(/[.!?…]["'”’»)]*$/) ? sentence : sentence + "."
    }

    /// `text` cut in two at the clause boundary nearest its middle, or else the nearest space.
    private static func halves(of text: String) -> (String, String)? {
      let middle = text.count / 2
      let nearest = { (cuts: [String.Index]) in
        cuts.min {
          abs(text.distance(from: text.startIndex, to: $0) - middle)
            < abs(text.distance(from: text.startIndex, to: $1) - middle)
        }
      }
      guard
        let cut = nearest(text.ranges(of: /[,;:]\s/).map(\.upperBound))
          ?? nearest(text.ranges(of: /\s/).map(\.upperBound)),
        cut > text.startIndex, cut < text.endIndex
      else { return nil }
      return (
        text[..<cut].trimmingCharacters(in: .whitespacesAndNewlines),
        text[cut...].trimmingCharacters(in: .whitespacesAndNewlines)
      )
    }
  }

  public enum OmniVoiceError: Error, Equatable {
    case unknownVoice(String)
    case unsupportedLanguage(String)
    /// The text does not fit the largest bucket, or cannot be split further.
    case tooLong
    case missingOutput(String)
    case invalidConfiguration(String)
    /// An output holds values the model cannot produce.
    case invalidOutput(String)
    /// The device's architecture has no compiled step.
    case unsupportedDevice(String)
  }

  extension OmniVoiceError: LocalizedError {
    public var errorDescription: String? { "OmniVoice: \(self)" }
  }
#endif
