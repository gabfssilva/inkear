/// A text-to-speech model the app can read with.
///
/// Each model family the app supports conforms to this protocol. Supertonic ships in the app;
/// the others are ``InstallableEngine``s, downloaded when first chosen.
public protocol SpeechEngine: Sendable {
  /// Identifiers of the voices the engine offers.
  var voices: [String] { get }

  /// The languages the engine speaks, as ISO 639-1 codes.
  var languages: Set<String> { get }

  /// The voices that speak `language`, when the engine knows them; all of them otherwise.
  func voices(for language: String) -> [String]

  /// Whether `quality` changes the speech. Engines with no setting that trades time for quality
  /// ignore it.
  static var usesQuality: Bool { get }

  /// Synthesizes `text` in `language`, given as an ISO 639-1 code such as "pt" or "en", yielding
  /// the speech in parts as they are ready. Engines that cannot produce part of a sentence yield
  /// one part per piece they synthesize, or a single part.
  ///
  /// - Parameters:
  ///   - speed: 1 is the voice's natural pace; 2 is twice as fast.
  ///   - quality: How much work goes into the audio; higher qualities take longer.
  func synthesize(
    _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality
  ) -> SpeechStream
}

extension SpeechEngine {
  /// Engines without a mapping return every voice for every language.
  public func voices(for language: String) -> [String] {
    voices
  }
}

/// How much work an engine puts into speech, from the fastest to the cleanest.
public enum SpeechQuality: String, Sendable, Codable, CaseIterable {
  case low, normal, high, highest
}

/// Mono audio as floating-point samples.
public struct Speech: Sendable, Equatable {
  public let samples: [Float]
  public let sampleRate: Double

  public init(samples: [Float], sampleRate: Double) {
    self.samples = samples
    self.sampleRate = sampleRate
  }

  public var duration: Double { sampleRate > 0 ? Double(samples.count) / sampleRate : 0 }
}
