#if canImport(CoreAI)
  import Foundation
  import Synthesis

  /// Chatterbox Multilingual pt-BR, run with Core AI on the Neural Engine, from packages compiled
  /// ahead of time for the device's architecture.
  ///
  /// A voice is a reference recording the model continues. Text is spoken in pieces of whole
  /// sentences, joined while they fit the prefill; a piece that T3 does not finish, or whose speech
  /// does not fit the flow, is split in half and spoken again.
  public actor Chatterbox: InstallableEngine {
    public static let id = EngineID.chatterbox
    public static let displayName = "Chatterbox"
    public static let downloadSize: Int64 = 607_157_310
    public static let supportedLanguages: Set<String> = ["pt", "en"]
    public static let usesQuality = true
    public static let defaultVoice = "rachel"
    public static let revision = "92771cba88a72c27632d3403383e4044f0788731"
    public nonisolated let voices = voiceNames
    public nonisolated let languages = supportedLanguages

    private static let repository = "itsmechico/chatterbox-multilingual-pt-br-coreai"
    private static let voiceNames = ["rachel", "thiago"]
    private static let prefixes =
      ChatterboxFunctions.packages.map { NeuralEnginePackages.name($0) + "/" }
      + ["tables/", "voices/", "flow_noise.f32", "tokenizer.json", "config.json"]
    private static let sampleRate = 24_000.0
    /// Pieces are joined while they stay at or under this many characters.
    private static let maxCharacters = 300
    /// The prompt's tokens and the speech tokens together must stay within this many.
    private static let maxFlowTokens = 518
    /// The T3 decode windows loaded. Each one holds its own copy of T3's weights, and a step in a
    /// wider one is slower: 5.3, 7.5 and 8.7 ms on an M5 Max.
    private static let decodeWindows: Set<Int> = [384, 768, 1024]
    private let bundle: ChatterboxBundle
    private let functions: ChatterboxFunctions

    private init(bundle: ChatterboxBundle, functions: ChatterboxFunctions) {
      self.bundle = bundle
      self.functions = functions
    }

    public static func install(
      into directory: URL, progress: @escaping @Sendable (Double?) -> Void
    ) async throws {
      try NeuralEnginePackages.requireSupportedDevice()
      // Another revision's packages are about to be overwritten.
      try NeuralEnginePackages.deleteCompiled(ChatterboxFunctions.packages, in: directory)
      try await HuggingFace.download(
        repo: repository, revision: revision, prefixes: prefixes, into: directory) {
        progress($0)
      }
      try markInstalled(in: directory)
    }

    public static func uninstall(from directory: URL) throws {
      try NeuralEnginePackages.deleteCompiled(ChatterboxFunctions.packages, in: directory)
      try FileManager.default.removeItem(at: directory)
    }

    public static func load(
      from directory: URL, progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws -> Chatterbox {
      let bundle = try ChatterboxBundle(
        directory: directory, voiceNames: voiceNames, decodeWindows: decodeWindows,
        flowTokens: maxFlowTokens)
      return Chatterbox(
        bundle: bundle,
        functions: try await ChatterboxFunctions(
          directory: directory, config: bundle.config, progress: progress))
    }

    public nonisolated func synthesize(
      _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality
    ) -> SpeechStream {
      SpeechStream { emit in
        try await self.synthesize(
          text, voice: voice, language: language, speed: speed, quality: quality, emit: emit)
      }
    }

    /// Emits one part per piece.
    private func synthesize(
      _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality,
      emit: @escaping @Sendable (Speech) -> Void
    ) async throws {
      guard Self.supportedLanguages.contains(language) else {
        throw ChatterboxError.unsupportedLanguage(language)
      }
      guard let reference = bundle.voices[voice] else {
        throw ChatterboxError.unknownVoice(voice)
      }
      guard let steps = bundle.config.flow.steps[quality.rawValue] else {
        throw ChatterboxError.invalidConfiguration("quality \(quality.rawValue)")
      }
      var rng = SystemRandomNumberGenerator()
      let silence = [Float](repeating: 0, count: Int(0.3 * Self.sampleRate))
      let normalized = SpeechText.normalized(text, language: language)
      let output = try SpeechEmitter(sampleRate: Self.sampleRate, speed: speed, emit: emit)
      for (index, piece) in pieces(in: normalized, voice: reference, language: language).enumerated()
      {
        let audio = try await speak(
          piece, language: language, voice: reference, steps: steps, rng: &rng)
        try output.append((index > 0 ? silence : []) + audio)
      }
      try output.finish()
    }

    /// The sentences of `text`, each ending in punctuation, joined while they fit one prefill and
    /// `maxCharacters`: the model reads a sentence better with its neighbors.
    private func pieces(in text: String, voice: ChatterboxVoice, language: String) -> [String] {
      var pieces: [String] = []
      for sentence in Self.sentences(of: text).map(Self.terminated) {
        if let last = pieces.last {
          let joined = last + " " + sentence
          if joined.count <= Self.maxCharacters,
            bundle.prefillFits(bundle.textIDs(joined, language: language), voice: voice)
          {
            pieces[pieces.count - 1] = joined
            continue
          }
        }
        pieces.append(sentence)
      }
      return pieces
    }

    /// The mono audio of `text`, split in half and spoken again when T3 does not finish it or its
    /// speech exceeds the flow.
    private func speak(
      _ text: String, language: String, voice: ChatterboxVoice, steps: Int,
      rng: inout SystemRandomNumberGenerator
    ) async throws -> [Float] {
      let ids = bundle.textIDs(text, language: language)
      if bundle.prefillFits(ids, voice: voice) {
        let generation = try await bundle.generate(
          ids, voice: voice, functions: functions, rng: &rng)
        if generation.ended,
          generation.tokens.count + voice.promptTokens.count <= Self.maxFlowTokens
        {
          return try await bundle.audio(
            generation.tokens, voice: voice, steps: steps, functions: functions, rng: &rng)
        }
      }
      guard let (head, tail) = Self.halves(of: text) else { throw ChatterboxError.tooLong }
      let first = try await speak(head, language: language, voice: voice, steps: steps, rng: &rng)
      let second = try await speak(tail, language: language, voice: voice, steps: steps, rng: &rng)
      return first + second
    }

    /// `text` split into its sentences, each ending with its punctuation.
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

  public enum ChatterboxError: Error, Equatable {
    case unknownVoice(String)
    case unsupportedLanguage(String)
    /// The text has no split that fits the models.
    case tooLong
    case missingOutput(String)
    case invalidConfiguration(String)
  }
#endif
