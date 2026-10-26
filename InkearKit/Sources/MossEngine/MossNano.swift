import Foundation
import Synthesis

/// MOSS-TTS-Nano 100M, run with Core ML.
///
/// A voice is a reference clip the model continues, so any voice speaks either language.
/// Each generation reads as many whole sentences as fit in 75 tokens. A take that does not end,
/// or whose length is implausible for its text, is generated again up to the quality's retries.
public actor MossNano: InstallableEngine {
  public static let id = EngineID.moss
  public static let displayName = "MOSS-TTS-Nano"
  public static let downloadSize: Int64 = 313_115_936
  public static let supportedLanguages: Set<String> = ["pt", "en"]
  public static let usesQuality = true
  public static let defaultVoice = "rachel"
  public static let revision = "abd6964f01c4f024460a93ea9a095d30f557696f"
  public nonisolated let voices = voiceNames
  public nonisolated let languages = supportedLanguages

  /// The LibriVox readers of each language: rachel and thiago for Portuguese, ava and adam for
  /// English.
  public nonisolated func voices(for language: String) -> [String] {
    switch language {
    case "pt": ["rachel", "thiago"]
    case "en": ["ava", "adam"]
    default: voices
    }
  }

  private static let repository = "itsmechico/moss-tts-nano-coreml-ane"
  private static let voiceNames = ["rachel", "thiago", "ava", "adam"]
  private static let prefixes = [
    "GlobalDecoder.mlmodelc/", "LocalDecoder.mlmodelc/", "CodecDecoder.mlmodelc/", "config.json",
    "tokenizer.json", "voices/rachel.json", "voices/thiago.json", "voices/ava.json",
    "voices/adam.json",
  ]
  private static let sampleRate = 48_000.0
  /// Sentences with more tokens than this are split in two before generation.
  private static let maxTokens = 75
  private let bundle: MossNanoBundle

  private init(bundle: MossNanoBundle) {
    self.bundle = bundle
  }

  public static func install(
    into directory: URL, progress: @escaping @Sendable (Double?) -> Void
  ) async throws {
    try await HuggingFace.download(
      repo: repository, revision: revision, prefixes: prefixes, into: directory) {
      progress($0)
    }
    try markInstalled(in: directory)
  }

  public static func load(
    from directory: URL, progress: @escaping @Sendable (LoadProgress) -> Void
  ) async throws -> MossNano {
    MossNano(
      bundle: try MossNanoBundle(directory: directory, voiceNames: voiceNames, progress: progress))
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
  ) throws {
    guard Self.supportedLanguages.contains(language) else {
      throw MossEngineError.unsupportedLanguage(language)
    }
    guard let reference = bundle.voices[voice] else { throw MossEngineError.unknownVoice(voice) }
    guard let policy = bundle.config.quality[quality.rawValue] else {
      throw MossEngineError.invalidConfiguration("quality \(quality.rawValue)")
    }
    let silence = [Float](repeating: 0, count: Int(0.3 * Self.sampleRate))
    let output = try SpeechEmitter(sampleRate: Self.sampleRate, speed: speed, emit: emit)
    let utterances = try utterances(in: SpeechText.normalized(text, language: language))
    for (index, utterance) in utterances.enumerated() {
      try output.append(
        (index > 0 ? silence : []) + speak(utterance, voice: reference, policy: policy))
    }
    try output.finish()
  }

  /// The sentences of `text`, each ending in punctuation, joined while they fit in `maxTokens`:
  /// the model reads a sentence better with its neighbors.
  private func utterances(in text: String) throws -> [String] {
    var utterances: [String] = []
    for sentence in Self.sentences(of: text).map(Self.terminated) {
      if let last = utterances.last,
        try bundle.tokenizer.encode(last + " " + sentence).count <= Self.maxTokens
      {
        utterances[utterances.count - 1] = last + " " + sentence
      } else {
        utterances.append(sentence)
      }
    }
    return utterances
  }

  /// The mono audio of `text`, split in two and generated again when it has too many tokens.
  private func speak(
    _ text: String, voice: MossNanoVoice, policy: MossNanoConfig.Quality
  ) throws -> [Float] {
    let ids = try bundle.tokenizer.encode(text)
    if ids.count > Self.maxTokens, let (head, tail) = Self.halves(of: text) {
      return try speak(head, voice: voice, policy: policy)
        + speak(tail, voice: voice, policy: policy)
    }
    let take = try generate(ids, voice: voice, policy: policy)
    return Speech(averaging: take.stereo, sampleRate: Self.sampleRate).samples
  }

  /// One take of `ids`, generated again while it is not plausible, up to the quality's retries.
  /// The last take is returned, plausible or not.
  private func generate(
    _ ids: [Int32], voice: MossNanoVoice, policy: MossNanoConfig.Quality
  ) throws -> MossNanoBundle.Take {
    let ratio = bundle.config.plausibleFramesPerToken
    let tokens = Double(ids.count)
    let maxFrames = min(bundle.config.maxFrames, Int(ratio.upper * tokens) + 1)
    let plausible = (ratio.lower * tokens)...(ratio.upper * tokens)
    func accepted(_ take: MossNanoBundle.Take) -> Bool {
      take.ended && plausible.contains(Double(take.frames))
    }

    var take = try bundle.generate(ids, voice: voice, depth: policy.codebooks, maxFrames: maxFrames)
    var retries = 0
    while retries < policy.retries && !accepted(take) {
      take = try bundle.generate(ids, voice: voice, depth: policy.codebooks, maxFrames: maxFrames)
      retries += 1
    }
    return take
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

public enum MossEngineError: Error, Equatable {
  case unknownVoice(String)
  case unsupportedLanguage(String)
  /// The prompt does not fit the prefill buckets, or the text cannot be split further.
  case tooLong
  case missingOutput(String)
  case invalidConfiguration(String)
  case invalidInput(String)
}
