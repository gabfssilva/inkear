import CoreML
import Foundation
import os

/// Supertonic 3, run with Core ML.
///
/// Reads a directory as `supertonic-convert` writes it: one package per stage (`.mlpackage`, or
/// `.mlmodelc` once compiled) holding a function per input bucket named `t<text>_l<latent>`, plus
/// `tts.json`, `unicode_indexer.json` and `voice_styles/`.
///
/// The actor owns the models of every stage and bucket, all loaded by `load`. Its methods never
/// suspend internally: predictions run synchronously on the actor.
public actor Supertonic: SpeechEngine {
  public static let usesQuality = true
  public nonisolated let voices: [String]
  public nonisolated let languages = SupertonicText.languages
  private let assets: Assets
  private let models: [String: MLModel]

  /// Loads the assets and the models in `directory`, compiling the stage packages if needed, and
  /// reports each model function loaded.
  public static func load(
    from directory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
    progress: @escaping @Sendable (LoadProgress) -> Void = { _ in }
  ) async throws -> Supertonic {
    var packages: [Stage: URL] = [:]
    for stage in Stage.allCases {
      let compiled = directory.appending(path: "\(stage.rawValue).mlmodelc")
      packages[stage] =
        FileManager.default.fileExists(atPath: compiled.path(percentEncoded: false))
        ? compiled
        : try await MLModel.compileModel(
          at: directory.appending(path: "\(stage.rawValue).mlpackage"))
    }
    let functions = try await MLModelAsset(url: packages[.vectorEstimator] ?? directory)
      .functionNames
    let assets = try Assets(
      directory: directory, packages: packages, buckets: functions.compactMap(Bucket.init))
    var loader = ModelLoader(
      directory: directory, total: Stage.allCases.count * assets.buckets.count,
      computeUnits: computeUnits, progress: progress)
    var models: [String: MLModel] = [:]
    for stage in Stage.allCases {
      for bucket in assets.buckets {
        models[key(stage, bucket)] = try loader.load(
          contentsOf: packages[stage] ?? directory, function: bucket.name)
      }
    }
    return Supertonic(assets: assets, models: models)
  }

  private init(assets: Assets, models: [String: MLModel]) {
    self.assets = assets
    self.models = models
    self.voices = assets.styles.keys.sorted()
  }

  public nonisolated func synthesize(
    _ text: String, voice: String, language: String, speed: Double = 1,
    quality: SpeechQuality = .normal
  ) -> SpeechStream {
    SpeechStream { emit in
      try await self.synthesize(
        text, voice: voice, language: language, speed: speed, quality: quality, emit: emit)
    }
  }

  /// Emits one part per utterance. The model applies the speed itself.
  private func synthesize(
    _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality,
    emit: @escaping @Sendable (Speech) -> Void
  ) throws {
    guard let style = assets.styles[voice] else { throw SupertonicError.unknownVoice(voice) }
    guard SupertonicText.languages.contains(language) else {
      throw SupertonicError.unsupportedLanguage(language)
    }
    let silence = [Float](repeating: 0, count: Int(0.3 * Double(assets.sampleRate) / speed))
    let spoken = SpeechText.normalized(text, language: language)
    let output = try SpeechEmitter(sampleRate: Double(assets.sampleRate), speed: 1, emit: emit)
    for (index, utterance) in utterances(in: spoken, language: language).enumerated() {
      try output.append(
        (index > 0 ? silence : [])
          + speak(
            utterance, language: language, style: style, speed: Self.pace * speed,
            steps: Self.steps(for: quality)))
    }
    try output.finish()
  }

  /// The model's speed factor for a natural pace, as upstream uses it.
  static let pace = 1.05

  /// Denoising steps of the vector estimator, whose run is most of the synthesis time.
  static func steps(for quality: SpeechQuality) -> Int {
    switch quality {
    case .low: 4
    case .normal: 8
    case .high: 12
    case .highest: 16
    }
  }

  /// One pass through the model, with its intermediate results.
  struct Run {
    let bucket: Bucket
    let seconds: Double
    let frames: Int
    let textEmbedding: [Float]
    let latent: [Float]
    let samples: [Float]
  }

  /// Synthesizes `ids` in the smallest bucket that pads both them and their latent frames.
  ///
  /// - Parameter noise: The initial latent for a number of frames, as 144 rows of that many values.
  func run(
    ids: [Int32], style: Style, steps: Int = Supertonic.steps(for: .normal),
    speed: Double = Supertonic.pace,
    noise: @Sendable (Int) -> [Float] = Supertonic.gaussian
  ) throws -> Run {
    let clock = ContinuousClock()
    let started = clock.now
    guard let textBucket = assets.bucket(ids: ids.count) else { throw SupertonicError.tooLong }
    let textInputs = { (bucket: Bucket) -> [String: MLMultiArray] in
      [
        "text_ids": Self.array(
          ids + Array(repeating: 0, count: bucket.text - ids.count), [1, bucket.text]),
        "text_mask": Self.mask(ids.count, of: bucket.text),
      ]
    }
    let duration = try predict(
      .durationPredictor, textBucket,
      textInputs(textBucket).merging(["style_dp": Self.array(style.dp, [1, 8, 16])]) { a, _ in a })
    let seconds = Double(duration.scalars[0]) / speed
    let samples = Int(seconds * Double(assets.sampleRate))
    let frames = (samples + assets.samplesPerFrame - 1) / assets.samplesPerFrame
    guard let bucket = assets.bucket(ids: ids.count, frames: frames) else {
      throw SupertonicError.tooLong
    }

    let styleTTL = Self.array(style.ttl, [1, 50, 256])
    let text = textInputs(bucket)
    let embedding = try predict(
      .textEncoder, bucket, text.merging(["style_ttl": styleTTL]) { a, _ in a })
    let encoded = clock.now
    let initial = noise(frames)
    var latent = MLShapedArray<Float>(
      scalars: (0..<assets.latentChannels).flatMap { row in
        initial[(row * frames)..<((row + 1) * frames)]
          + Array(repeating: 0, count: bucket.latent - frames)
      },
      shape: [1, assets.latentChannels, bucket.latent]
    )
    let fixed: [String: MLMultiArray] = [
      "text_emb": MLMultiArray(embedding),
      "style_ttl": styleTTL,
      "latent_mask": Self.mask(frames, of: bucket.latent),
      "text_mask": Self.mask(ids.count, of: bucket.text),
      "total_step": Self.array([Float(steps)], [1]),
    ]
    for step in 0..<steps {
      latent = try predict(
        .vectorEstimator, bucket,
        fixed.merging([
          "noisy_latent": MLMultiArray(latent), "current_step": Self.array([Float(step)], [1]),
        ]) { a, _ in a }
      )
    }
    let estimated = clock.now
    let wav = try predict(.vocoder, bucket, ["latent": MLMultiArray(latent)])
    let finished = clock.now
    let milliseconds = { (duration: Duration) in duration / .milliseconds(1) }
    Self.timings.info(
      "\(bucket.name, privacy: .public): encode \(milliseconds(encoded - started), privacy: .public) ms, \(steps, privacy: .public) steps \(milliseconds(estimated - encoded), privacy: .public) ms, vocode \(milliseconds(finished - estimated), privacy: .public) ms, \(Double(samples) / Double(self.assets.sampleRate), privacy: .public) s audio"
    )
    return Run(
      bucket: bucket, seconds: seconds, frames: frames, textEmbedding: embedding.scalars,
      latent: latent.scalars,
      samples: Array(wav.scalars.prefix(samples))
    )
  }

  /// Per-stage timings of each pass, for checking where the models run on a device.
  private static let timings = Logger(subsystem: "inkear", category: "supertonic")

  /// Standard normal values, by the Box–Muller transform.
  static func gaussian(frames: Int) -> [Float] {
    (0..<(144 * frames)).map { _ in
      let (u, v) = (Float.random(in: Float.ulpOfOne...1), Float.random(in: 0...1))
      return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
    }
  }

  /// Splits `text` into utterances of whole sentences that each fit the largest bucket.
  private func utterances(in text: String, language: String) -> [String] {
    let ends = text.ranges(of: /[.!?…]["'”’»)]*\s+/).map(\.upperBound)
    let sentences = zip([text.startIndex] + ends, ends + [text.endIndex])
      .map { text[$0..<$1].trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty }
    var utterances: [String] = []
    for sentence in sentences {
      if let last = utterances.last,
        assets.bucket(ids: ids(of: last + " " + sentence, language).count) != nil
      {
        utterances[utterances.count - 1] = last + " " + sentence
      } else {
        utterances.append(sentence)
      }
    }
    return utterances
  }

  /// Synthesizes one utterance, halving it at a clause or word boundary while it does not fit.
  private func speak(
    _ utterance: String, language: String, style: Style, speed: Double, steps: Int
  ) throws -> [Float] {
    do {
      return try run(ids: ids(of: utterance, language), style: style, steps: steps, speed: speed)
        .samples
    } catch SupertonicError.tooLong {
      guard let (head, tail) = Self.halves(of: utterance) else { throw SupertonicError.tooLong }
      return try speak(head, language: language, style: style, speed: speed, steps: steps)
        + speak(tail, language: language, style: style, speed: speed, steps: steps)
    }
  }

  func style(_ voice: String) -> Style? { assets.styles[voice] }

  func ids(of text: String, _ language: String) -> [Int32] {
    SupertonicText.ids(for: text, language: language, indexer: assets.indexer)
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
    return (text[..<cut].trimmingCharacters(in: .whitespaces), String(text[cut...]))
  }

  private func predict(_ stage: Stage, _ bucket: Bucket, _ inputs: [String: MLMultiArray]) throws
    -> MLShapedArray<Float>
  {
    let model = try model(stage, bucket)
    let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
    guard let name = model.modelDescription.outputDescriptionsByName.keys.first,
      let array = output.featureValue(for: name)?.multiArrayValue
    else { throw SupertonicError.missingOutput(stage.rawValue) }
    return MLShapedArray<Float>(converting: array)
  }

  private func model(_ stage: Stage, _ bucket: Bucket) throws -> MLModel {
    guard let model = models[Self.key(stage, bucket)] else {
      throw SupertonicError.missingModel(Self.key(stage, bucket))
    }
    return model
  }

  private static func key(_ stage: Stage, _ bucket: Bucket) -> String {
    "\(stage.rawValue)/\(bucket.name)"
  }

  private static func array(_ scalars: [Float], _ shape: [Int]) -> MLMultiArray {
    MLMultiArray(MLShapedArray<Float>(scalars: scalars, shape: shape))
  }

  private static func array(_ scalars: [Int32], _ shape: [Int]) -> MLMultiArray {
    MLMultiArray(MLShapedArray<Int32>(scalars: scalars, shape: shape))
  }

  private static func mask(_ valid: Int, of length: Int) -> MLMultiArray {
    array((0..<length).map { $0 < valid ? Float(1) : 0 }, [1, 1, length])
  }
}

public enum SupertonicError: Error, Equatable {
  case unknownVoice(String)
  case unsupportedLanguage(String)
  /// The text could not be split into pieces that fit the largest bucket.
  case tooLong
  case missingOutput(String)
  case missingModel(String)
}

enum Stage: String, CaseIterable, Sendable {
  case durationPredictor = "duration_predictor"
  case textEncoder = "text_encoder"
  case vectorEstimator = "vector_estimator"
  case vocoder
}

/// Input lengths a model function is specialized for.
struct Bucket: Hashable, Sendable {
  let text: Int
  let latent: Int

  var name: String { "t\(text)_l\(latent)" }

  init(text: Int, latent: Int) {
    self.text = text
    self.latent = latent
  }

  init?(_ name: String) {
    guard let match = name.wholeMatch(of: /t(\d+)_l(\d+)/), let text = Int(match.1),
      let latent = Int(match.2)
    else {
      return nil
    }
    self.init(text: text, latent: latent)
  }
}

/// A voice: the style embeddings the model conditions on.
struct Style: Sendable {
  let ttl: [Float]
  let dp: [Float]
}

/// Everything the engine reads from its directory, besides the models themselves.
private struct Assets: Sendable {
  let directory: URL
  let packages: [Stage: URL]
  /// Ascending by size.
  let buckets: [Bucket]
  let sampleRate: Int
  let samplesPerFrame: Int
  let latentChannels: Int
  let indexer: [Int32]
  let styles: [String: Style]

  init(directory: URL, packages: [Stage: URL], buckets: [Bucket]) throws {
    struct Config: Decodable {
      struct Autoencoder: Decodable {
        let sample_rate: Int
        let base_chunk_size: Int
      }
      struct TextToLatent: Decodable {
        let chunk_compress_factor: Int
        let latent_dim: Int
      }
      let ae: Autoencoder
      let ttl: TextToLatent
    }
    struct StyleFile: Decodable {
      struct Tensor: Decodable { let data: [[[Float]]] }
      let style_ttl: Tensor
      let style_dp: Tensor
    }
    let decoder = JSONDecoder()
    let config = try decoder.decode(
      Config.self, from: Data(contentsOf: directory.appending(path: "tts.json")))
    let voices = directory.appending(path: "voice_styles")
    self.directory = directory
    self.packages = packages
    self.buckets = buckets.sorted { ($0.text, $0.latent) < ($1.text, $1.latent) }
    sampleRate = config.ae.sample_rate
    samplesPerFrame = config.ae.base_chunk_size * config.ttl.chunk_compress_factor
    latentChannels = config.ttl.latent_dim * config.ttl.chunk_compress_factor
    indexer = try decoder.decode(
      [Int32].self, from: Data(contentsOf: directory.appending(path: "unicode_indexer.json")))
    styles = try Dictionary(
      uniqueKeysWithValues: FileManager.default.contentsOfDirectory(
        at: voices, includingPropertiesForKeys: nil
      )
      .filter { $0.pathExtension == "json" }
      .map { url in
        let file = try decoder.decode(StyleFile.self, from: Data(contentsOf: url))
        return (
          url.deletingPathExtension().lastPathComponent,
          Style(
            ttl: file.style_ttl.data.flatMap { $0.flatMap { $0 } },
            dp: file.style_dp.data.flatMap { $0.flatMap { $0 } })
        )
      })
  }

  /// The smallest bucket that pads `ids` text ids and, when given, `frames` latent frames.
  ///
  /// A bucket must leave at least one position of padding: the packages are converted for
  /// padded inputs, which give different outputs than unpadded ones.
  func bucket(ids: Int, frames: Int? = nil) -> Bucket? {
    buckets.first { bucket in bucket.text > ids && (frames.map { $0 < bucket.latent } ?? true) }
  }
}
