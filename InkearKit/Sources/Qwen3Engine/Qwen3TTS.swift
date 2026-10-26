// Adapted from speech-swift/Sources/Qwen3TTSCoreML/Qwen3TTSCoreML.swift, Apache-2.0.
#if canImport(CoreAI)
  import Foundation
  import Synthesis

  /// Qwen3-TTS 0.6B with its base voices, run with Core AI on the Neural Engine, from packages
  /// compiled ahead of time for the device's architecture.
  ///
  /// The model has no denoising steps, so `quality` has no effect: every call samples with the
  /// model's defaults (temperature 0.9, top-k 50, repetition penalty 1.05).
  ///
  /// Speech is produced one sentence at a time. A prompt holds at most 117 text tokens and the
  /// speech decoder at most 10 s of frames, so a sentence longer than `maxCharacters` is split at
  /// clause or word boundaries first, and one that still does not fit, or whose speech hits that
  /// cap, is split and synthesized again.
  public actor Qwen3TTS: InstallableEngine {
    public static let id = EngineID.qwen3
    public static let displayName = "Qwen3-TTS"
    public static let downloadSize: Int64 = 1_195_301_034
    /// The codec language identifier of each supported language, by ISO 639-1 code.
    private static let codecLanguages = ["pt": 2071, "en": 2050]
    public static let supportedLanguages = Set(codecLanguages.keys)
    public static let usesQuality = false
    public static let defaultVoice = "thiago"
    public static let revision = "eba19a5b27b554ace756d243520061810e055fe6"
    private static let voiceNames = ["thiago", "peter"]
    /// The gain of each voice, which brings the loudest of a dozen Portuguese sentences to a peak
    /// of 0.95: the model speaks thiago at about half the level of peter. A stream cannot scale by
    /// the peak of speech it has not synthesized yet.
    private static let gains: [String: Float] = ["thiago": 3.65, "peter": 1.86]
    public nonisolated let voices = voiceNames
    public nonisolated let languages = supportedLanguages

    /// thiago speaks Portuguese; peter, English.
    public nonisolated func voices(for language: String) -> [String] {
      language == "pt" ? ["thiago"] : ["peter"]
    }

    private static let repository = "itsmechico/qwen3-tts-0.6b-base-coreai"
    private static let prefixes =
      Qwen3Bundle.packages.map { NeuralEnginePackages.name($0) + "/" } + [
        "text_embeddings.f16", "code_embeddings.f16", "speech_codebooks.f16", "config.json",
        "vocab.json", "merges.txt", "tts_pad_embed.npy", "tts_bos_embed.npy", "tts_eos_embed.npy",
        "voices/",
      ]
    private static let sampleRate = 24_000.0
    /// About 9 s of speech. A longer sentence would often run into the 10 s cap, which costs a
    /// full generation that is then discarded.
    static let maxCharacters = 120
    /// Frames whose audio a call emits before the rest of its first piece, about 2 s.
    static let earlyFrames = 24
    /// Frames the speech decoder reads ahead: the audio of a frame changes with the next two. Decoded
    /// with this many more, the early frames are the start of the whole piece's audio.
    static let decoderLookahead = 4
    private let bundle: Qwen3Bundle

    private init(bundle: Qwen3Bundle) {
      self.bundle = bundle
    }

    public static func install(
      into directory: URL, progress: @escaping @Sendable (Double?) -> Void
    ) async throws {
      try NeuralEnginePackages.requireSupportedDevice()
      // Another revision's packages are about to be overwritten.
      try NeuralEnginePackages.deleteCompiled(Qwen3Bundle.packages, in: directory)
      try await HuggingFace.download(
        repo: repository, revision: revision, prefixes: prefixes, into: directory) {
        progress($0)
      }
      try markInstalled(in: directory)
    }

    public static func uninstall(from directory: URL) throws {
      try NeuralEnginePackages.deleteCompiled(Qwen3Bundle.packages, in: directory)
      try FileManager.default.removeItem(at: directory)
    }

    public static func load(
      from directory: URL, progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws -> Qwen3TTS {
      Qwen3TTS(
        bundle: try await Qwen3Bundle(
          directory: directory, voiceNames: voiceNames, progress: progress))
    }

    public nonisolated func synthesize(
      _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality
    ) -> SpeechStream {
      SpeechStream { emit in
        try await self.synthesize(
          text, voice: voice, language: language, speed: speed, quality: quality, emit: emit)
      }
    }

    /// Emits the audio of the first ``earlyFrames`` of the first piece as soon as the decoder can
    /// tell it, then each piece.
    private func synthesize(
      _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality,
      emit: @escaping @Sendable (Speech) -> Void
    ) async throws {
      guard let speaker = bundle.voices[voice], let gain = Self.gains[voice] else {
        throw Qwen3Error.unknownVoice(voice)
      }
      guard let codecLanguage = Self.codecLanguages[language] else {
        throw Qwen3Error.unsupportedLanguage(language)
      }
      // Silence is 0.3 s before stretching, so it is 0.3 / speed s in the output.
      let silence = [Float](repeating: 0, count: Int(0.3 * Self.sampleRate))
      let output = try SpeechEmitter(sampleRate: Self.sampleRate, speed: speed, emit: emit)
      let voice = Voice(speaker: speaker, gain: gain, codecLanguage: codecLanguage)
      var first = true
      let sentences = Self.sentences(of: SpeechText.normalized(text, language: language))
      for (index, sentence) in sentences.enumerated() {
        if index > 0 { try output.append(silence) }
        for piece in Self.pieces(of: sentence) {
          try await speak(piece, voice: voice, early: first, into: output)
          first = false
        }
      }
      try output.finish()
    }

    /// Emits the audio of `text`, split in two and synthesized again when its prompt is too long or
    /// the model does not end the speech before the frame cap. A single word that hits the cap is
    /// kept as it is cut.
    ///
    /// With `early`, the start of the audio is emitted once its frames exist; should the take then
    /// hit the cap and be split, that start is heard again before the halves.
    private func speak(_ text: String, voice: Voice, early: Bool, into output: SpeechEmitter)
      async throws
    {
      var emitted = 0
      let codes = try await bundle.codes(
        for: text, codecLanguage: voice.codecLanguage, speaker: voice.speaker,
        prefixFrames: early ? Self.earlyFrames + Self.decoderLookahead : nil
      ) { prefix in
        let start = Array(
          try await bundle.speech(codebooks: prefix)
            .prefix(Self.earlyFrames * Qwen3Bundle.samplesPerFrame))
        try output.append(voice.amplified(start))
        emitted = start.count
      }
      let parts = Self.halves(of: text)
      if let codes, codes.finished || parts == nil {
        let audio = try await bundle.speech(codebooks: codes.codebooks)
        try output.append(voice.amplified(Array(audio.dropFirst(emitted))))
        return
      }
      guard let (head, tail) = parts else { throw Qwen3Error.tooLong }
      try await speak(head, voice: voice, early: false, into: output)
      try await speak(tail, voice: voice, early: false, into: output)
    }

    /// What a request speaks with.
    private struct Voice {
      let speaker: [Float]
      let gain: Float
      let codecLanguage: Int

      /// `samples` times the voice's gain, clipped to ±1.
      func amplified(_ samples: [Float]) -> [Float] {
        samples.map { min(max($0 * gain, -1), 1) }
      }
    }

    /// `text` split into its sentences, each ending with its punctuation.
    private static func sentences(of text: String) -> [String] {
      let ends = text.ranges(of: /[.!?…]["'”’»)]*\s+/).map(\.upperBound)
      return zip([text.startIndex] + ends, ends + [text.endIndex])
        .map { text[$0..<$1].trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    }

    /// `sentence` cut in halves until every piece has at most `maxCharacters` or cannot be cut.
    static func pieces(of sentence: String) -> [String] {
      guard sentence.count > maxCharacters, let (head, tail) = halves(of: sentence) else {
        return [sentence]
      }
      return pieces(of: head) + pieces(of: tail)
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

  public enum Qwen3Error: Error, Equatable {
    case unknownVoice(String)
    case unsupportedLanguage(String)
    /// The text does not fit the talker's positions, and it cannot be split further.
    case tooLong
    case missingOutput(String)
    case invalidEmbedding(String)
  }
#endif
