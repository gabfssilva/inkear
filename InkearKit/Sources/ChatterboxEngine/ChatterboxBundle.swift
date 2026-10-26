#if canImport(CoreAI)
  import Foundation
  import Synthesis

  /// The constants of `config.json`.
  struct ChatterboxConfig: Decodable, Sendable {
    struct TextIDs: Decodable, Sendable {
      let start: Int
      let stop: Int
    }

    struct SpeechIDs: Decodable, Sendable {
      let start: Int
      let stop: Int
      let flowVocab: Int
    }

    struct T3: Decodable, Sendable {
      let context: Int
      let masked: Float
      let ropeInverseFrequencies: [Double]
      let prefillLengths: [Int]
      let decodeWindows: [Int]
    }

    struct Sampling: Decodable, Sendable {
      let cfgWeight: Double
      let temperature: Double
      let repetitionPenalty: Double
      let minP: Double
      let topP: Double
      let maxNewTokens: Int
    }

    struct Flow: Decodable, Sendable {
      let tokens: [Int]
      let steps: [String: Int]
    }

    struct Vocoder: Decodable, Sendable {
      let frames: Int
      let context: Int
      let samplesPerFrame: Int
      let sourceWeight: [Double]
      let sourceBias: Double
    }

    let text: TextIDs
    let speech: SpeechIDs
    let t3: T3
    let sampling: Sampling
    let flow: Flow
    let vocoder: Vocoder
    let samplesPerToken: Int
  }

  extension ChatterboxConfig {
    /// The configuration with the largest prefill only, the `decodeWindows` of its own, and the
    /// smallest flow size that holds `flowTokens`: shorter inputs are padded to them.
    func sizes(decodeWindows: Set<Int>, flowTokens: Int) -> ChatterboxConfig {
      let flowSize = flow.tokens.first { $0 >= flowTokens } ?? flow.tokens.max() ?? flowTokens
      return ChatterboxConfig(
        text: text, speech: speech,
        t3: T3(
          context: t3.context, masked: t3.masked, ropeInverseFrequencies: t3.ropeInverseFrequencies,
          prefillLengths: t3.prefillLengths.max().map { [$0] } ?? [],
          decodeWindows: t3.decodeWindows.filter(decodeWindows.contains)),
        sampling: sampling, flow: Flow(tokens: [flowSize], steps: flow.steps), vocoder: vocoder,
        samplesPerToken: samplesPerToken)
    }
  }

  /// A voice: the conditioning T3 and the flow read, taken from a reference recording.
  struct ChatterboxVoice: Sendable {
    /// T3's conditioning rows, (condRows, 1024), row-major.
    let t3Cond: [Float]
    let promptTokens: [Int32]
    /// The prompt's mel, frame-major: (2 · promptTokens.count, 80).
    let promptMel: [Float]
    let spks: [Float]

    init(directory: URL) throws {
      t3Cond = try readArray(Float.self, from: directory.appending(path: "t3_cond.f32"))
      promptTokens = try readArray(Int32.self, from: directory.appending(path: "prompt_tokens.i32"))
      promptMel = try readArray(Float.self, from: directory.appending(path: "prompt_mel.f32"))
      spks = try readArray(Float.self, from: directory.appending(path: "spks.f32"))
    }

    var condRows: Int { t3Cond.count / ChatterboxBundle.hidden }
  }

  /// The pack's tensors and constants, read from its directory. Its Core AI functions are in
  /// `ChatterboxFunctions`.
  struct ChatterboxBundle: Sendable {
    static let hidden = 1024
    static let melChannels = 80
    static let headDimension = 64

    let config: ChatterboxConfig
    let tokenizer: ChatterboxTokenizer
    let textEmbeddings: [Float16]
    let textPositions: [Float16]
    let speechEmbeddings: [Float16]
    let speechPositions: [Float16]
    /// The flow encoder's token embeddings, (flowVocab, 512) row-major.
    let flowEmbeddings: [Float16]
    let voices: [String: ChatterboxVoice]
    /// The flow's initial noise, (80, 1536) channel-major.
    let noise: [Float]
    let directory: URL

    /// Reads the pack in `directory`, keeping the sizes of
    /// ``ChatterboxConfig/sizes(decodeWindows:flowTokens:)``.
    /// Each size is a function that holds its own copy of the weights once loaded, and the iPhone
    /// ends the app above about 3.6 GB.
    init(directory: URL, voiceNames: [String], decodeWindows: Set<Int>, flowTokens: Int) throws {
      self.directory = directory
      config = try Self.loadConfig(from: directory)
        .sizes(decodeWindows: decodeWindows, flowTokens: flowTokens)
      tokenizer = try ChatterboxTokenizer(url: directory.appending(path: "tokenizer.json"))
      func table(_ name: String) throws -> [Float16] {
        try readArray(Float16.self, from: directory.appending(path: "tables/\(name).f16"))
      }
      textEmbeddings = try table("text_emb")
      textPositions = try table("text_pos")
      speechEmbeddings = try table("speech_emb")
      speechPositions = try table("speech_pos")
      flowEmbeddings = try table("flow_emb")
      noise = try readArray(Float.self, from: directory.appending(path: "flow_noise.f32"))
      voices = try Dictionary(
        uniqueKeysWithValues: voiceNames.map { name in
          try (name, ChatterboxVoice(directory: directory.appending(path: "voices/\(name)")))
        })
    }

    private static func loadConfig(from directory: URL) throws -> ChatterboxConfig {
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      return try decoder.decode(
        ChatterboxConfig.self, from: Data(contentsOf: directory.appending(path: "config.json")))
    }

    /// The ids of `text` as the front end builds them: [START] + tokens of "[lang]" + the
    /// normalized text with spaces as [SPACE] + [STOP].
    func textIDs(_ text: String, language: String) -> [Int] {
      let spaced = ChatterboxTokenizer.normalize(text)
        .replacingOccurrences(of: " ", with: "[SPACE]")
      return [config.text.start] + tokenizer.encode("[\(language)]" + spaced) + [config.text.stop]
    }

    /// Whether T3's prefill takes the conditioning, the text and the two start-of-speech rows.
    func prefillFits(_ ids: [Int], voice: ChatterboxVoice) -> Bool {
      voice.condRows + ids.count + 2 <= (config.t3.prefillLengths.max() ?? 0)
    }

    /// One draw from the logits of T3's two rows (conditional, unconditional): guidance, repetition
    /// penalty, temperature, min-p, then top-p.
    static func sample(
      _ logits: [Float16], generated: [Int], sampling: ChatterboxConfig.Sampling,
      rng: inout SystemRandomNumberGenerator
    ) -> Int {
      let vocab = logits.count / 2
      var scores = (0..<vocab).map { index -> Double in
        let conditional = Double(logits[index])
        return conditional + sampling.cfgWeight * (conditional - Double(logits[vocab + index]))
      }
      for index in Set(generated) {
        scores[index] =
          scores[index] < 0
          ? scores[index] * sampling.repetitionPenalty : scores[index] / sampling.repetitionPenalty
      }
      scores = scores.map { $0 / sampling.temperature }

      let peak = scores.max() ?? 0
      let exponentials = scores.map { exp($0 - peak) }
      let total = exponentials.reduce(0, +)
      let probabilities = exponentials.map { $0 / total }
      let threshold = sampling.minP * (probabilities.max() ?? 0)
      for index in 0..<vocab where probabilities[index] < threshold {
        scores[index] = -.infinity
      }

      let order = (0..<vocab).sorted { scores[$0] < scores[$1] }
      let top = scores.max() ?? 0
      let ascending = order.map { exp(scores[$0] - top) }
      let mass = ascending.reduce(0, +)
      var cumulative = 0.0
      for (rank, index) in order.enumerated() {
        cumulative += ascending[rank] / mass
        if cumulative <= 1 - sampling.topP && rank < vocab - 1 {
          scores[index] = -.infinity
        }
      }

      let weights = scores.map { exp($0 - top) }
      let weightTotal = weights.reduce(0, +)
      var target = Double.random(in: 0..<weightTotal, using: &rng)
      var chosen = 0
      for (index, weight) in weights.enumerated() where weight > 0 {
        if target < weight { return index }
        target -= weight
        chosen = index
      }
      return chosen
    }
  }

  /// The speech tokens T3 produced for one text, and whether it stopped on its stop token rather
  /// than at a cap.
  struct ChatterboxGeneration: Sendable {
    let tokens: [Int]
    let ended: Bool
  }

  /// The elements of a raw little-endian file.
  func readArray<Element>(_ type: Element.Type, from url: URL) throws -> [Element] {
    try Data(contentsOf: url).withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
      Array(buffer.bindMemory(to: Element.self))
    }
  }
#endif
