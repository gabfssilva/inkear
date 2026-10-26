#if canImport(CoreAI)
  import Foundation
  import Synthesis

  /// NVIDIA Magpie TTS Multilingual 357M, run with Core AI on the Neural Engine, from packages
  /// compiled ahead of time for the device's architecture, and its codec with Core ML.
  ///
  /// Each of the five speakers reads either language. `quality` has no effect: the model samples
  /// with the same settings at every quality.
  public actor MagpieTTS: InstallableEngine {
    public static let id = EngineID.magpie
    public static let displayName = "Magpie TTS"
    public static let downloadSize: Int64 = 641_108_538
    public static let supportedLanguages: Set<String> = ["pt", "en"]
    public static let usesQuality = false
    public static let defaultVoice = "Sofia"
    public static let revision = "fc02b729f98a68129a173ca780bd0bbd699a8c32"
    public nonisolated let voices = voiceNames
    public nonisolated let languages = supportedLanguages

    private static let repository = "itsmechico/magpie-tts-multilingual-357m-coreai"
    private static let voiceNames = ["Sofia", "Jason", "John", "Aria", "Leo"]
    private static let prefixes =
      MagpieBundle.packages.map { NeuralEnginePackages.name($0) + "/" } + [
        "nanocodec_decoder.mlmodelc/", "text_embeddings.f16", "speakers.f16",
        "decoder_positions.f16", "magpie.json", "tokenizer_pt.json", "tokenizer_en.json",
        "lexicon_pt.tsv", "lexicon_en.tsv", "heteronyms_en.txt", "audio_bos_embedding.f32",
      ]
    /// Silence between the pieces of a text, in seconds.
    private static let pause = 0.3
    private let bundle: MagpieBundle

    private init(bundle: MagpieBundle) {
      self.bundle = bundle
    }

    public static func install(
      into directory: URL, progress: @escaping @Sendable (Double?) -> Void
    ) async throws {
      try NeuralEnginePackages.requireSupportedDevice()
      // Another revision's packages are about to be overwritten.
      try NeuralEnginePackages.deleteCompiled(MagpieBundle.packages, in: directory)
      try await HuggingFace.download(
        repo: repository, revision: revision, prefixes: prefixes, into: directory) {
        progress($0)
      }
      try markInstalled(in: directory)
    }

    public static func uninstall(from directory: URL) throws {
      try NeuralEnginePackages.deleteCompiled(MagpieBundle.packages, in: directory)
      try FileManager.default.removeItem(at: directory)
    }

    public static func load(
      from directory: URL, progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws -> MagpieTTS {
      MagpieTTS(
        bundle: try await MagpieBundle(
          directory: directory, languages: supportedLanguages, progress: progress))
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
      guard let speaker = bundle.config.speakers[voice] else {
        throw MagpieEngineError.unknownVoice(voice)
      }
      guard let frontend = bundle.frontends[language] else {
        throw MagpieEngineError.unsupportedLanguage(language)
      }
      let sampleRate = bundle.config.sampleRate
      let silence = [Float](repeating: 0, count: Int(Self.pause * sampleRate))
      let output = try SpeechEmitter(sampleRate: sampleRate, speed: speed, emit: emit)
      let pieces = utterances(of: SpeechText.normalized(text, language: language), frontend)
      for (index, piece) in pieces.enumerated() {
        let speech = try await speak(piece, speaker: speaker, frontend)
        try output.append((index > 0 ? silence : []) + speech)
      }
      try output.finish()
    }

    /// The sentences of `text`, each ending in punctuation, joined while they fit in the text
    /// bucket: the model reads a sentence better with its neighbors.
    private func utterances(of text: String, _ frontend: MagpieFrontend) -> [String] {
      let limit = bundle.config.textTokens
      var utterances: [String] = []
      for sentence in Self.sentences(of: text).map(Self.terminated) {
        if let last = utterances.last, frontend.encode(last + " " + sentence).count <= limit {
          utterances[utterances.count - 1] = last + " " + sentence
        } else {
          utterances.append(sentence)
        }
      }
      return utterances
    }

    /// The mono audio of `text`, split in two and generated again when it has too many tokens.
    private func speak(_ text: String, speaker: Int, _ frontend: MagpieFrontend) async throws
      -> [Float]
    {
      let ids = frontend.encode(text)
      if ids.count > bundle.config.textTokens, let (head, tail) = Self.halves(of: text) {
        return try await speak(head, speaker: speaker, frontend)
          + speak(tail, speaker: speaker, frontend)
      }
      return try await bundle.speech(ids, speaker: speaker)
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

  public enum MagpieEngineError: Error, Equatable {
    case unknownVoice(String)
    case unsupportedLanguage(String)
    /// The text has more tokens than the encoder holds, and cannot be split further.
    case tooLong
    case missingOutput(String)
  }
#endif
