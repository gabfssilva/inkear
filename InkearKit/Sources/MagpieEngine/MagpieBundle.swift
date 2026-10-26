#if canImport(CoreAI)
  import CoreAI
  import CoreML
  import Foundation
  import Synthesis

  /// The models of a Magpie TTS bundle, and the generation loop and codec that run them.
  ///
  /// The text encoder, the decoder and the local transformer run with Core AI on the Neural
  /// Engine, from packages compiled ahead of time for the device's architecture. The codec runs
  /// with Core ML, on the CPU and the Neural Engine: the ahead-of-time compiler gets it wrong. The
  /// text embeddings, the speakers' context and the decoder's position embeddings are tables the
  /// bundle reads itself: MPSGraph runs a lookup by integer index on the GPU, which iOS denies to
  /// apps in the background.
  struct MagpieBundle {
    /// The packages, which `coreai-aot` compiled for each iPhone architecture.
    static let packages = ["text_encoder", "decoder", "local_transformer"]

    let config: MagpieConfig
    let frontends: [String: MagpieFrontend]
    private let bos: [Float]
    /// (text vocabulary, 768).
    private let textEmbeddings: Float16Table
    /// The context of each speaker, (5 · 217, 768).
    private let speakers: Float16Table
    /// (positions, 768).
    private let positions: Float16Table
    private let encoder: InferenceFunction
    private let prefill: InferenceFunction
    private let step: InferenceFunction
    private let sampler: InferenceFunction
    private let codec: MLModel

    private static let hidden = 768
    /// Positions of the prefill: the speaker's context, padded to a width the Neural Engine
    /// compiler takes.
    private static let prefillPositions = 224
    /// The value of an additive mask at a hidden position.
    private static let hiddenPosition: Float16 = -1e4
    /// The codec's streaming histories, by input name, with their shapes.
    private static let histories: [(name: String, shape: [Int])] = [
      ("history_32", [1, 32, 64]), ("history_864", [1, 864, 64]),
      ("history_432", [1, 432, 1216]), ("history_216", [1, 216, 1216]),
      ("history_108", [1, 108, 1216]), ("history_54", [1, 54, 1216]),
      ("history_27", [1, 27, 1216]),
    ]

    init(
      directory: URL, languages: Set<String>, progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws {
      config = try JSONDecoder.snakeCase.decode(
        MagpieConfig.self, from: Data(contentsOf: directory.appending(path: "magpie.json")))
      frontends = try Dictionary(
        uniqueKeysWithValues: languages.map {
          ($0, try MagpieFrontend(directory: directory, language: $0))
        })
      let embedding = try Data(contentsOf: directory.appending(path: "audio_bos_embedding.f32"))
      bos = embedding.withUnsafeBytes { bytes in
        stride(from: 0, to: bytes.count, by: 4).map {
          Float(
            bitPattern: UInt32(
              littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
        }
      }
      let table = { (name: String) in
        try Float16Table(directory.appending(path: "\(name).f16"), width: Self.hidden)
      }
      textEmbeddings = try table("text_embeddings")
      speakers = try table("speakers")
      positions = try table("decoder_positions")

      var loader = ModelLoader(directory: directory, total: 5, progress: progress)
      let package = { (name: String) in
        try await NeuralEnginePackages.specialize(
          directory.appending(path: NeuralEnginePackages.name(name)))
      }
      encoder = try await loader.load("main", of: try await package("text_encoder"))
      let decoder = try await package("decoder")
      prefill = try await loader.load("prefill", of: decoder)
      step = try await loader.load("step", of: decoder)
      sampler = try await loader.load("main", of: try await package("local_transformer"))
      codec = try loader.load("nanocodec_decoder")
    }

    /// The mono audio of the token ids `ids` spoken by `speaker`, at the model's sample rate.
    func speech(
      _ ids: [Int], speaker: Int, isolation: isolated (any Actor)? = #isolation
    ) async throws -> [Float] {
      try decode(try await frames(ids, speaker: speaker))
    }

    /// The codec frames of `ids`, each frame's 8 codes, generated until the model ends the speech
    /// or the step limit is reached.
    private func frames(
      _ ids: [Int], speaker: Int, isolation: isolated (any Actor)? = #isolation
    ) async throws -> [[Int]] {
      let textTokens = config.textTokens
      guard ids.count <= textTokens else { throw MagpieEngineError.tooLong }
      let codebooks = config.codebooks
      let framesPerStep = config.framesPerStep
      let codesPerStep = framesPerStep * codebooks
      let vocabulary = config.vocabPerCodebook
      let context = config.contextFrames
      let mask = NDArray(
        scalars: (0..<textTokens).map { $0 < ids.count ? Float16(1) : 0 }, shape: [1, textTokens])
      let padded = ids + Array(repeating: 0, count: textTokens - ids.count)
      let encoded = try await encoder.outputs([
        "embeds": Self.array(padded.map(textEmbeddings.row), shape: [1, textTokens, Self.hidden]),
        "mask": mask,
      ])
      let speakerRows = (0..<Self.prefillPositions).map { position in
        position < context
          ? speakers.row(speaker * context + position) : [Float](repeating: 0, count: Self.hidden)
      }
      let prefilled = try await prefill.outputs([
        "context": Self.array(speakerRows, shape: [1, Self.prefillPositions, Self.hidden]),
        "encoder_output": try Self.output(encoded, "encoder_output"),
        "encoder_mask": mask,
      ])
      var cache = DecoderCache(positions: config.cachePositions)
      cache.write(
        try Self.output(prefilled, "keys"), try Self.output(prefilled, "values"), count: context,
        at: 0)
      let crossKeys = try Self.output(prefilled, "cross_keys")
      let crossValues = try Self.output(prefilled, "cross_values")
      let scalar = { (value: Float) in NDArray(scalars: [Float16(value)], shape: [1]) }

      var alignment = Alignment(textLength: ids.count, settings: config.attentionPrior)
      var prior = [Float](repeating: 1, count: textTokens)
      var audioEmb = bos
      var generator = SystemRandomNumberGenerator()
      var generated: [[Int]] = []
      for index in 0..<config.maxSteps {
        try Task.checkCancellation()
        let forbidEOS = index * framesPerStep < config.minFrames
        let position = context + index
        let stepped = try await step.outputs([
          "audio_emb": Self.array(
            [zip(audioEmb, positions.row(position)).map(+)], shape: [1, 1, Self.hidden]),
          "mask": NDArray(
            scalars: (0..<config.cachePositions).map { $0 < position ? 0 : Self.hiddenPosition },
            shape: [1, config.cachePositions]),
          "encoder_mask": mask,
          "attn_prior": NDArray(scalars: prior.map(Float16.init), shape: [1, textTokens]),
          "cfg_scale": scalar(config.quality.cfgScale),
          "key_cache": cache.keys,
          "value_cache": cache.values,
          "cross_keys": crossKeys,
          "cross_values": crossValues,
        ])
        cache.write(
          try Self.output(stepped, "key"), try Self.output(stepped, "value"), count: 1,
          at: position)
        let logits = try Self.values(stepped, "logits")
        prior = alignment.update(try Self.values(stepped, "alignment"))

        let noise = (0..<(codesPerStep * vocabulary)).map { _ in Float16(Self.gumbel(&generator)) }
        let sampled = try await sampler.outputs([
          "h_last": try Self.output(stepped, "h_last"),
          "gumbel": NDArray(scalars: noise, shape: [codesPerStep, vocabulary]),
          "temperature": scalar(config.quality.temperature),
          "cfg_scale": scalar(config.quality.cfgScale),
          "forbid_eos": scalar(forbidEOS ? 1 : 0),
        ])
        let drawn = try Self.values(sampled, "codes").map { Int($0.rounded()) }
        audioEmb = try Self.values(sampled, "audio_emb")

        let picked = greedy(logits, forbidEOS: forbidEOS)
        let ended = (0..<framesPerStep).filter { frame in
          (0..<codebooks).contains { codebook in
            let index = frame * codebooks + codebook
            return drawn[index] == config.audioEos || picked[index] == config.audioEos
          }
        }
        let kept = ended.first ?? framesPerStep
        generated += (0..<kept).map { frame in
          Array(drawn[(frame * codebooks)..<((frame + 1) * codebooks)])
        }
        if !ended.isEmpty { return generated }
      }
      return generated
    }

    /// The argmax of each codebook's logits, leaving out the forbidden codes and, while
    /// `forbidEOS`, the end code.
    private func greedy(_ logits: [Float], forbidEOS: Bool) -> [Int] {
      let vocabulary = config.vocabPerCodebook
      let excluded = Set(config.forbiddenCodes + (forbidEOS ? [config.audioEos] : []))
      return (0..<(config.framesPerStep * config.codebooks)).map { row in
        (0..<vocabulary).reduce(nil as Int?) { current, code in
          guard !excluded.contains(code) else { return current }
          guard let current else { return code }
          let better = logits[row * vocabulary + code] > logits[row * vocabulary + current]
          return better ? code : current
        } ?? 0
      }
    }

    /// The mono audio of `frames`, 8 codes each, at the model's sample rate.
    ///
    /// The codec runs on calls of `codecFramesPerCall` frames, with zero latents after the last
    /// frame, and carries its histories from one call to the next.
    private func decode(_ frames: [[Int]]) throws -> [Float] {
      let chunk = config.codecFramesPerCall
      let levels = config.fsqLevels
      let channels = config.codebooks * levels.count
      let padded = frames.count + (chunk - frames.count % chunk) % chunk

      // Finite scalar quantization: each code is four digits, each mapped to (digit - half) / half.
      var latents = [Float](repeating: 0, count: channels * padded)
      for (index, frame) in frames.enumerated() {
        for (codebook, code) in frame.enumerated() {
          var base = 1
          for (digit, level) in levels.enumerated() {
            let half = level / 2
            let value = Float((code / base) % level - half) / Float(half)
            latents[(codebook * levels.count + digit) * padded + index] = value
            base *= level
          }
        }
      }

      var histories: [String: MLMultiArray] = [:]
      for history in Self.histories {
        histories[history.name] = try Self.multiArray(
          [Float16](repeating: 0, count: history.shape.reduce(1, *)), history.shape, .float16)
      }
      var audio: [Float] = []
      for start in stride(from: 0, to: padded, by: chunk) {
        var block: [Float] = []
        block.reserveCapacity(channels * chunk)
        for channel in 0..<channels {
          let row = channel * padded + start
          block += latents[row..<(row + chunk)]
        }
        var inputs = histories
        inputs["latents"] = try Self.multiArray(block, [1, channels, chunk], .float32)
        let output = try codec.prediction(
          from: MLDictionaryFeatureProvider(
            dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) }))
        let samples = try output.multiArray("audio")
        audio += Self.values(samples, as: Float.self)
        for history in Self.histories {
          histories[history.name] = try output.multiArray("\(history.name)_out")
        }
      }
      return Array(audio.prefix(frames.count * config.samplesPerFrame))
    }

    private static func gumbel(_ generator: inout SystemRandomNumberGenerator) -> Float {
      let u = (Double(generator.next() >> 11) + 0.5) / 9_007_199_254_740_992
      return Float(-log(-log(u)))
    }

    /// A Float16 array of `rows`, row-major, with the given shape.
    private static func array(_ rows: [[Float]], shape: [Int]) -> NDArray {
      NDArray(scalars: rows.flatMap { $0.map(Float16.init) }, shape: shape)
    }

    private static func output(_ outputs: [String: NDArray], _ name: String) throws -> NDArray {
      guard let array = outputs[name] else { throw MagpieEngineError.missingOutput(name) }
      return array
    }

    /// The Float16 output `name` as floats, in row-major order.
    private static func values(_ outputs: [String: NDArray], _ name: String) throws -> [Float] {
      try output(outputs, name).scalars(as: Float16.self).map(Float.init)
    }

    /// A row-major array of `values`, with the given shape and data type.
    private static func multiArray<T>(
      _ values: [T], _ shape: [Int], _ type: MLMultiArrayDataType
    ) throws -> MLMultiArray {
      let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: type)
      array.withUnsafeMutableBytes { buffer, _ in
        let pointer = buffer.assumingMemoryBound(to: T.self)
        for (index, value) in values.enumerated() { pointer[index] = value }
      }
      return array
    }

    /// The values of `array` in row-major order. Core ML can pad the rows of an output, so each
    /// row is read at the strides of its dimensions.
    private static func values<T>(_ array: MLMultiArray, as _: T.Type) -> [T] {
      let shape = array.shape.map(\.intValue)
      let strides = array.strides.map(\.intValue)
      return array.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) -> [T] in
        let source = buffer.assumingMemoryBound(to: T.self)
        let width = shape[shape.count - 1]
        let step = strides[strides.count - 1]
        var flat: [T] = []
        flat.reserveCapacity(array.count)
        var index = [Int](repeating: 0, count: shape.count - 1)
        for _ in 0..<(array.count / width) {
          let offset = zip(index, strides).reduce(0) { $0 + $1.0 * $1.1 }
          for i in 0..<width { flat.append(source[offset + i * step]) }
          var axis = index.count - 1
          while axis >= 0 {
            index[axis] += 1
            if index[axis] < shape[axis] { break }
            index[axis] = 0
            axis -= 1
          }
        }
        return flat
      }
    }
  }

  /// The decoder's self-attention caches: keys (24, 12, 64, positions), transposed for the
  /// attention product, and values (24, 12, positions, 64). Rows 2i and 2i + 1 are the
  /// conditional and unconditional rows of layer i.
  ///
  /// The step takes the caches as inputs and returns its key and value, which the caches write at
  /// the step's slot: the Neural Engine compiler of Core AI crashes on a model that writes into a
  /// state at a position known only at run time.
  private struct DecoderCache {
    private(set) var keys: NDArray
    private(set) var values: NDArray

    init(positions: Int) {
      keys = NDArray(shape: [24, 12, 64, positions], scalarType: .float16)
      values = NDArray(shape: [24, 12, positions, 64], scalarType: .float16)
    }

    /// Writes the first `count` positions of `key` and `value`, the prefill's or a step's, from
    /// `slot` on.
    mutating func write(_ key: NDArray, _ value: NDArray, count: Int, at slot: Int) {
      Self.copy(key, count: count, along: 3, into: &keys, at: slot)
      Self.copy(value, count: count, along: 2, into: &values, at: slot)
    }

    /// Copies the first `count` positions of `source` along `axis` into `target` from `slot`.
    private static func copy(
      _ source: NDArray, count: Int, along axis: Int, into target: inout NDArray, at slot: Int
    ) {
      source.view(as: Float16.self).withUnsafePointer { from, shape, fromStrides in
        target.mutableView(as: Float16.self).withUnsafeMutablePointer { to, _, toStrides in
          var extent = (0..<4).map { shape[$0] }
          extent[axis] = count
          let start = slot * toStrides[axis]
          for a in 0..<extent[0] {
            for b in 0..<extent[1] {
              for c in 0..<extent[2] {
                for d in 0..<extent[3] {
                  let offset = a * toStrides[0] + b * toStrides[1] + c * toStrides[2]
                  to[start + offset + d * toStrides[3]] =
                    from[
                      a * fromStrides[0] + b * fromStrides[1] + c * fromStrides[2] + d
                        * fromStrides[3]]
                }
              }
            }
          }
        }
      }
    }
  }

  /// The attended text position of the decoder, and the cross-attention prior built from it.
  ///
  /// The decoder's attention is pushed forward through the text one token at a time, and a
  /// position that has been attended `sinkThreshold` times is no longer attended to.
  private struct Alignment {
    let textLength: Int
    let settings: MagpieConfig.AttentionPrior
    private var last = 1
    private var counts: [Int: Int] = [:]

    init(textLength: Int, settings: MagpieConfig.AttentionPrior) {
      self.textLength = textLength
      self.settings = settings
    }

    /// The prior of the next step, given the alignment `scores` of the current one.
    mutating func update(_ scores: [Float]) -> [Float] {
      let start = counts[last, default: 0] >= settings.sinkThreshold ? last + 1 : last
      let end = min(start + settings.lookahead, textLength - 3)
      var attended = textLength - 1
      if start < end {
        var best = start
        for position in start..<end where scores[position] > scores[best] {
          best = position
        }
        attended = best
      }
      counts[attended, default: 0] += 1
      last = attended

      var prior = [Float](repeating: settings.epsilon, count: scores.count)
      if textLength <= 5 {
        prior = [Float](repeating: 1, count: scores.count)
      } else {
        prior[max(1, attended - 1)] = 1
        prior[attended] = 1
        for offset in 1...settings.lookahead {
          prior[min(attended + offset, textLength - 1)] = 1
        }
      }
      for (position, count) in counts where count >= settings.sinkThreshold {
        for index in 0...position { prior[index] = settings.epsilon }
      }
      return prior
    }
  }

  /// The constants of `magpie.json`.
  struct MagpieConfig: Decodable {
    struct AttentionPrior: Decodable {
      let epsilon: Float
      let lookahead: Int
      let sinkThreshold: Int
    }

    struct Quality: Decodable {
      let cfgScale: Float
      let temperature: Float
    }

    let sampleRate: Double
    let samplesPerFrame: Int
    let codebooks: Int
    let framesPerStep: Int
    let vocabPerCodebook: Int
    let audioEos: Int
    let forbiddenCodes: [Int]
    let textTokens: Int
    let contextFrames: Int
    let cachePositions: Int
    let maxSteps: Int
    let minFrames: Int
    let fsqLevels: [Int]
    let codecFramesPerCall: Int
    let speakers: [String: Int]
    let attentionPrior: AttentionPrior
    let quality: Quality
  }

  extension MLFeatureProvider {
    fileprivate func multiArray(_ name: String) throws -> MLMultiArray {
      guard let array = featureValue(for: name)?.multiArrayValue else {
        throw MagpieEngineError.missingOutput(name)
      }
      return array
    }
  }
#endif
