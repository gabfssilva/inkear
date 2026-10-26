import CoreML
import Foundation
import Synthesis

/// The Core ML functions of a MOSS-TTS-Nano bundle, and the generation loop and codec that run them.
///
/// Each function is an `MLModel` of its `.mlmodelc`, loaded by function name on the Neural Engine.
/// Tensors are Float16 except the token ids, which are Int32.
struct MossNanoBundle {
  /// The stereo audio of one generation and how it ended.
  struct Take {
    /// Planar samples at 48 kHz: the left channel, then the right.
    let stereo: [[Float]]
    let ended: Bool
    let frames: Int
  }

  static let hidden = 768
  static let layers = 12
  static let heads = 12
  static let headDim = 64
  static let codebooks = 16
  static let codebookSize = 1024
  static let candidates = 32
  static let ropeBase = 10_000.0
  static let samplesPerFrame = 3840
  static let padding: Int32 = 1024
  static let pad: Int32 = 3
  static let voiceSlot: Int32 = 8
  static let assistantSlot: Int32 = 9
  static let audioEnd: Int32 = 7
  static let settings: [Float16] = [1, 1 / 0.8, 25, 0.95, 1.2, 1 / 1.2]
  static let head: [Int32] = [4, 600, 289, 10356, 13, 10356, 10425, 1860, 4546, 12907, 10363, 13325, 11492, 6]
  static let after: [Int32] = [
    10356, 10425, 3965, 7738, 11492, 505, 587, 10356, 10425, 352, 500, 856, 11492, 505, 587,
    10356, 10425, 2128, 1247, 594, 11492, 505, 587, 10356, 10425, 348, 909, 561, 3648, 11492,
    505, 587, 10356, 10425, 2818, 1305, 355, 348, 909, 11492, 505, 587, 10356, 10425, 484, 339,
    10367, 783, 11492, 505, 587, 10356, 10425, 2427, 980, 11492,
  ]
  static let tail: [Int32] = [10356, 14, 5, 4, 8165, 430, 6]

  let config: MossNanoConfig
  let tokenizer: MossNanoTokenizer
  let voices: [String: MossNanoVoice]
  private let prefills: [Int: MLModel]
  private let steps: [Int: MLModel]
  private let frameModels: [Int: MLModel]
  private let codec: MLModel

  /// A layer's key or value cache, `[layers, heads, slots, headDim]`.
  private struct KVCache {
    let key: MLMultiArray
    let value: MLMultiArray
  }

  init(
    directory: URL, voiceNames: [String], progress: @escaping @Sendable (LoadProgress) -> Void
  ) throws {
    let config = try JSONDecoder().decode(
      MossNanoConfig.self, from: Data(contentsOf: directory.appending(path: "config.json")))
    let decoder = JSONDecoder()
    let voices = try Dictionary(
      uniqueKeysWithValues: voiceNames.map { name in
        try (
          name,
          decoder.decode(
            MossNanoVoice.self,
            from: Data(contentsOf: directory.appending(path: "voices/\(name).json")))
        )
      })
    var loader = ModelLoader(
      directory: directory,
      total: config.prefillRows.count + config.contexts.count + config.codebookDepths.count + 1,
      progress: progress)
    var prefills: [Int: MLModel] = [:]
    for rows in config.prefillRows {
      prefills[rows] = try loader.load("GlobalDecoder", function: "prefill_\(rows)")
    }
    var steps: [Int: MLModel] = [:]
    for context in config.contexts {
      steps[context] = try loader.load("GlobalDecoder", function: "step_\(context)")
    }
    var frameModels: [Int: MLModel] = [:]
    for depth in config.codebookDepths {
      frameModels[depth] = try loader.load("LocalDecoder", function: "frame_\(depth)")
    }
    self.config = config
    self.tokenizer = try MossNanoTokenizer(
      data: Data(contentsOf: directory.appending(path: "tokenizer.json")))
    self.voices = voices
    self.prefills = prefills
    self.steps = steps
    self.frameModels = frameModels
    self.codec = try loader.load("CodecDecoder", function: "chunk_\(config.chunkFrames)")
  }

  /// Generates `ids` spoken in `voice` with `depth` codebooks per frame, at most `maxFrames` frames.
  ///
  /// The noise is drawn afresh on every call, so repeated calls give different takes.
  func generate(_ ids: [Int32], voice: MossNanoVoice, depth: Int, maxFrames: Int) throws -> Take {
    guard let frameModel = frameModels[depth] else {
      throw MossEngineError.invalidConfiguration("codebook depth \(depth)")
    }
    let rows = prompt(ids, voice: voice)
    guard rows.allSatisfy({ $0.count == Self.codebooks + 1 }) else {
      throw MossEngineError.invalidConfiguration("prompt row width")
    }
    let length = rows.count
    guard let size = config.prefillRows.first(where: { $0 >= length }), let prefill = prefills[size]
    else { throw MossEngineError.tooLong }
    guard let largest = config.contexts.max() else {
      throw MossEngineError.invalidConfiguration("no contexts")
    }

    // Causal attention keeps the padding rows from touching the real ones.
    let filler = [Self.pad] + Array(repeating: Self.padding, count: Self.codebooks)
    let padded = (0..<size).flatMap { $0 < length ? rows[$0] : filler }
    let prefilled = try Self.predict(
      prefill, ["rows": Self.int32(padded, [size, Self.codebooks + 1])])
    let keyRows = try Self.values(Self.output(prefilled, "keys"))
    let valueRows = try Self.values(Self.output(prefilled, "values"))
    let prefillHidden = try Self.values(Self.output(prefilled, "hidden"))
    var hiddenValues = Array(
      prefillHidden[((length - 1) * Self.hidden)..<(length * Self.hidden)])

    var caches: [Int: KVCache] = [:]
    for context in config.contexts {
      let shape = [Self.layers, Self.heads, context, Self.headDim]
      let cache = KVCache(key: try Self.zeros(shape), value: try Self.zeros(shape))
      let copied = min(length, context)
      Self.copySlots(
        keyRows, sourceSlots: size, into: cache.key, targetSlots: context, targetSlot: 0,
        count: copied)
      Self.copySlots(
        valueRows, sourceSlots: size, into: cache.value, targetSlots: context, targetSlot: 0,
        count: copied)
      caches[context] = cache
    }

    let settings = try Self.float16(Self.settings, [1, 6])
    var seen = [Float16](repeating: 0, count: depth * Self.codebookSize)
    var codes: [[Int32]] = []
    var pending: [[Int32]] = []
    var decoded = 0
    var stereo: [[Float]] = [[], []]
    var codecCaches = try config.codecCaches.mapValues { try Self.zeros($0) }
    var ended = false
    var generator = SystemRandomNumberGenerator()

    func decode() throws {
      var codeValues = [Float16](
        repeating: Float16(Self.padding), count: config.chunkFrames * Self.codebooks)
      for (index, frameIDs) in pending.enumerated() {
        for c in 0..<Self.codebooks {
          codeValues[index * Self.codebooks + c] = Float16(frameIDs[c])
        }
      }
      var inputs = codecCaches
      inputs["codes"] = try Self.float16(codeValues, [config.chunkFrames, Self.codebooks])
      inputs["past_frames"] = try Self.float16(
        [Float16(min(decoded, config.codecWindowFrames))], [1, 1])
      let out = try Self.predict(codec, inputs)
      for name in codecCaches.keys {
        codecCaches[name] = try Self.output(out, "new_\(name)")
      }
      let audio = try Self.values(Self.output(out, "audio"))
      let samples = audio.count / 2
      let kept = pending.count * Self.samplesPerFrame
      for channel in 0..<2 {
        stereo[channel] += audio[(channel * samples)..<(channel * samples + kept)].map { Float($0) }
      }
      decoded += pending.count
      pending.removeAll()
    }

    while codes.count < maxFrames && length + codes.count < largest {
      let noise = try Self.float16(
        (0..<(depth * Self.candidates)).map { _ in Self.gumbel(&generator) },
        [depth, Self.candidates])
      let textNoise = try Self.float16(
        [Self.gumbel(&generator), Self.gumbel(&generator)], [1, 2])
      let hiddenInput = try Self.float16(hiddenValues, [1, Self.hidden])
      let seenInput = try Self.float16(seen, [depth, Self.codebookSize])
      let sampled = try Self.predict(
        frameModel,
        [
          "hidden": hiddenInput, "seen": seenInput, "gumbel": noise, "text_gumbel": textNoise,
          "settings": settings,
        ])
      let endFlag = try Self.values(Self.output(sampled, "end"))
      if endFlag[0] > 0.5 {
        ended = true
        break
      }
      let drawn = try Self.values(Self.output(sampled, "codes"))
      var frameIDs = [Int32](repeating: Self.padding, count: Self.codebooks)
      for c in 0..<depth {
        frameIDs[c] = Int32(drawn[c].rounded())
        seen[c * Self.codebookSize + Int(frameIDs[c])] = 1
      }
      codes.append(frameIDs)
      pending.append(frameIDs)
      if pending.count == config.chunkFrames { try decode() }

      // The step feeds the frame at `position`; the caches hold rows 0..<position.
      let position = length + codes.count - 1
      guard let context = config.contexts.first(where: { $0 >= position }),
        let step = steps[context], let cache = caches[context]
      else { throw MossEngineError.tooLong }
      let row = Self.int32([Self.assistantSlot] + frameIDs, [1, Self.codebooks + 1])
      let mask = try Self.float16(
        (0..<context).map { $0 < position ? 0 : Float16(config.masked) }, [1, context])
      var cosine = [Float16](repeating: 0, count: Self.headDim)
      var sine = cosine
      for i in 0..<(Self.headDim / 2) {
        let angle = Double(position) * pow(Self.ropeBase, -Double(2 * i) / Double(Self.headDim))
        cosine[i] = Float16(cos(angle))
        cosine[i + Self.headDim / 2] = cosine[i]
        sine[i] = Float16(sin(angle))
        sine[i + Self.headDim / 2] = sine[i]
      }
      let cosInput = try Self.float16(cosine, [1, Self.headDim])
      let sinInput = try Self.float16(sine, [1, Self.headDim])
      let stepped = try Self.predict(
        step,
        [
          "row": row, "key_cache": cache.key, "value_cache": cache.value, "mask": mask,
          "cos": cosInput, "sin": sinInput,
        ])
      let newKey = try Self.values(Self.output(stepped, "key"))
      let newValue = try Self.values(Self.output(stepped, "value"))
      for (slots, target) in caches where position < slots {
        Self.copySlots(
          newKey, sourceSlots: 1, into: target.key, targetSlots: slots, targetSlot: position,
          count: 1)
        Self.copySlots(
          newValue, sourceSlots: 1, into: target.value, targetSlots: slots, targetSlot: position,
          count: 1)
      }
      hiddenValues = try Self.values(Self.output(stepped, "hidden"))
    }
    if !pending.isEmpty { try decode() }
    return Take(stereo: stereo, ended: ended, frames: codes.count)
  }

  /// The prompt rows of `ids` spoken in `voice`: the head, one row per voice frame, then the tail.
  private func prompt(_ ids: [Int32], voice: MossNanoVoice) -> [[Int32]] {
    let voiceRows = voice.codes.map { [Self.voiceSlot] + $0 }
    let tailIDs = [Self.audioEnd] + Self.after + ids + Self.tail
    return Self.textRows(Self.head) + voiceRows + Self.textRows(tailIDs)
  }

  private static func textRows(_ ids: [Int32]) -> [[Int32]] {
    ids.map { [$0] + Array(repeating: padding, count: codebooks) }
  }

  private static func gumbel(_ generator: inout SystemRandomNumberGenerator) -> Float16 {
    let u = (Double(generator.next() >> 11) + 0.5) / 9_007_199_254_740_992
    return Float16(-log(-log(u)))
  }

  private static func zeros(_ shape: [Int]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    array.withUnsafeMutableBytes { buffer, _ in
      _ = buffer.initializeMemory(as: UInt8.self, repeating: 0)
    }
    return array
  }

  private static func float16(_ values: [Float16], _ shape: [Int]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    array.withUnsafeMutableBytes { buffer, _ in
      let pointer = buffer.assumingMemoryBound(to: Float16.self)
      for (index, value) in values.enumerated() { pointer[index] = value }
    }
    return array
  }

  private static func int32(_ values: [Int32], _ shape: [Int]) -> MLMultiArray {
    MLMultiArray(MLShapedArray<Int32>(scalars: values, shape: shape))
  }

  private static func predict(_ model: MLModel, _ inputs: [String: MLMultiArray]) throws
    -> MLFeatureProvider
  {
    try model.prediction(
      from: MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) }))
  }

  private static func output(_ result: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
    guard let array = result.featureValue(for: name)?.multiArrayValue else {
      throw MossEngineError.missingOutput(name)
    }
    return array
  }

  /// The values of `array` in row-major order. Core ML can pad the rows of an output, so each row
  /// is read at the strides of its dimensions.
  private static func values(_ array: MLMultiArray) -> [Float16] {
    let shape = array.shape.map(\.intValue)
    let strides = array.strides.map(\.intValue)
    return array.withUnsafeBytes { buffer in
      let source = buffer.assumingMemoryBound(to: Float16.self)
      let width = shape[shape.count - 1], step = strides[strides.count - 1]
      var flat: [Float16] = []
      flat.reserveCapacity(array.count)
      var index = [Int](repeating: 0, count: shape.count - 1)
      for _ in 0..<(array.count / width) {
        let offset = zip(index, strides).reduce(0) { $0 + $1.0 * $1.1 }
        for i in 0..<width { flat.append(source[offset + i * step]) }
        var d = index.count - 1
        while d >= 0 {
          index[d] += 1
          if index[d] < shape[d] { break }
          index[d] = 0
          d -= 1
        }
      }
      return flat
    }
  }

  /// Copies `count` slots of each attention head from `source`, which holds `sourceSlots` slots per
  /// head, into `target`, which holds `targetSlots` slots per head, starting at `targetSlot`.
  private static func copySlots(
    _ source: [Float16], sourceSlots: Int, into target: MLMultiArray, targetSlots: Int,
    targetSlot: Int, count: Int
  ) {
    target.withUnsafeMutableBytes { buffer, _ in
      let pointer = buffer.assumingMemoryBound(to: Float16.self)
      for head in 0..<(layers * heads) {
        for i in 0..<(count * headDim) {
          pointer[(head * targetSlots + targetSlot) * headDim + i] =
            source[head * sourceSlots * headDim + i]
        }
      }
    }
  }
}

/// The model's configuration, from `config.json`.
struct MossNanoConfig: Decodable {
  struct Quality: Decodable {
    let codebooks: Int
    let retries: Int
  }

  /// The bounds of frames per token that a plausible take falls in, from a two-element JSON array.
  struct FramesPerToken: Decodable {
    let lower: Double
    let upper: Double

    init(from decoder: Decoder) throws {
      var values = try decoder.unkeyedContainer()
      lower = try values.decode(Double.self)
      upper = try values.decode(Double.self)
    }
  }

  let prefillRows: [Int]
  let contexts: [Int]
  let codebookDepths: [Int]
  let chunkFrames: Int
  let codecWindowFrames: Int
  let codecCaches: [String: [Int]]
  let masked: Float
  let maxFrames: Int
  let plausibleFramesPerToken: FramesPerToken
  let quality: [String: Quality]

  // Explicit keys: a key decoding strategy would also rewrite the cache names in `codecCaches`.
  private enum CodingKeys: String, CodingKey {
    case prefillRows = "prefill_rows"
    case contexts
    case codebookDepths = "codebook_depths"
    case chunkFrames = "chunk_frames"
    case codecWindowFrames = "codec_window_frames"
    case codecCaches = "codec_caches"
    case masked
    case maxFrames = "max_frames"
    case plausibleFramesPerToken = "plausible_frames_per_token"
    case quality
  }
}

/// A reference voice: the codebook frames of a clip the model continues.
struct MossNanoVoice: Decodable {
  let codes: [[Int32]]
}
