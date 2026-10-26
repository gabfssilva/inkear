#if canImport(CoreAI)
  import CoreAI
#endif
import CoreML
import Foundation
import Synthesis

/// The parts of omnivoice.json the engine reads.
struct OmniVoiceConfig: Decodable, Sendable {
  let qualitySteps: [String: Int]
  let stepBuckets: [OmniVoiceBucket]
  let decoderFrames: [Int]
  let duration: DurationEstimator

  private enum CodingKeys: String, CodingKey {
    case qualitySteps = "quality_steps"
    case stepBuckets = "step_buckets"
    case decoderFrames = "decoder_frames"
    case duration
  }
}

/// A reference voice: the codes of a recording per codebook, its transcript and its loudness.
struct OmniVoiceVoice: Decodable, Sendable {
  let text: String
  let rms: Double
  let tokens: [[Int32]]
}

#if canImport(CoreAI)
  /// The functions of an OmniVoice bundle, and the unmasking loop that runs them.
  ///
  /// Only the step functions and decoders of `targets` load, all up front. The step functions
  /// run with Core AI on the Neural Engine from ``stepPackage``, compiled ahead of time for the
  /// device's architecture; each holds its own copy of the 1.2 GB of weights once loaded. The
  /// decoders run with Core ML. Token ids are Int32 and the step returns them as Float16,
  /// computed without an integer argmax, whose output Core AI mislabels on the Neural Engine;
  /// masks and scores are Float16, audio Float32.
  actor OmniVoiceBundle {
    /// The step compiled ahead of time, for the Neural Engine, for this device's architecture.
    static let stepPackage = "step-ane.\(AIModel.deviceArchitectureName).aimodelc"
    static let sampleRate = 24_000.0
    private static let samplesPerFrame = 960
    private static let tShift = 0.1
    private static let layerPenalty: Float = 5
    private static let positionTemperature: Float = 5

    nonisolated let voices: [String: OmniVoiceVoice]
    nonisolated let qualitySteps: [String: Int]
    nonisolated let frontEnd: OmniVoiceFrontEnd

    private let decoderFrames: [Int]
    private let stepFunctions: [String: InferenceFunction]
    private let decoderModels: [Int: MLModel]

    init(
      directory: URL, voiceNames: [String], targets: Set<Int>,
      progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws {
      let config = try JSONDecoder().decode(
        OmniVoiceConfig.self, from: Data(contentsOf: directory.appending(path: "omnivoice.json")))
      let decoder = JSONDecoder()
      voices = try Dictionary(
        uniqueKeysWithValues: voiceNames.map { name in
          try (
            name,
            decoder.decode(
              OmniVoiceVoice.self, from: Data(contentsOf: directory.appending(path: "voices/\(name).json")))
          )
        })
      let buckets = config.stepBuckets.filter { targets.contains($0.target) }
      let decoderFrames = config.decoderFrames.filter(targets.contains)
      frontEnd = OmniVoiceFrontEnd(
        tokenizer: try OmniVoiceTokenizer(url: directory.appending(path: "tokenizer.json")),
        duration: config.duration, buckets: buckets)
      qualitySteps = config.qualitySteps
      self.decoderFrames = decoderFrames

      var loader = ModelLoader(
        directory: directory, total: buckets.count + decoderFrames.count, progress: progress)
      let stepModel = directory.appending(path: Self.stepPackage)
      var stepFunctions: [String: InferenceFunction] = [:]
      for bucket in buckets {
        stepFunctions[bucket.function] = try await loader.load {
          // The default policy lets the system purge the compiled step, and compiling it again
          // takes minutes: the Neural Engine compiler crashes on the ahead-of-time binary, and
          // MPSGraph then compiles the step from its graph. OmniVoice.uninstall deletes the entry.
          let model = try await AIModel.specialize(
            contentsOf: stepModel, options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine),
            cachePolicy: .persistent)
          guard let function = try model.loadFunction(named: bucket.function) else {
            throw OmniVoiceError.invalidConfiguration("no step function \(bucket.function)")
          }
          return function
        }
      }
      var decoderModels: [Int: MLModel] = [:]
      for frames in decoderFrames {
        decoderModels[frames] = try loader.load("HiggsAudio2Decoder", function: "f\(frames)")
      }
      self.stepFunctions = stepFunctions
      self.decoderModels = decoderModels
    }

    /// The mono samples of `text`, which must fit a bucket, with 0.1 s of silence on each side.
    func generate(
      _ text: String, language: String, voice: OmniVoiceVoice, steps: Int
    ) async throws -> [Float] {
      let conditioning = frontEnd.conditioning(text, language: language, voice: voice)
      guard let bucket = frontEnd.bucket(for: conditioning) else { throw OmniVoiceError.tooLong }
      let tokens = try await unmask(conditioning, bucket: bucket, steps: steps)
      let audio = try decodeAudio(tokens, frames: conditioning.target)
      return OmniVoiceAudio.postprocess(audio, voice: voice)
    }

    /// Iterative unmasking. Each step reveals the codes the model is most sure of, with Gumbel
    /// noise on the ranking so that the order varies between takes.
    private func unmask(
      _ conditioning: OmniVoiceConditioning, bucket: OmniVoiceBucket, steps: Int
    ) async throws -> [[Int32]] {
      let codebooks = OmniVoiceFrontEnd.codebooks
      let mask = OmniVoiceFrontEnd.mask
      let frames = conditioning.target
      let window = bucket.target
      let cond = bucket.cond
      let end = bucket.sequence
      let length = conditioning.length
      let function = try stepFunction(bucket)

      var ids = [[Int32]](repeating: [Int32](repeating: mask, count: end), count: codebooks)
      var audioMask = [Float16](repeating: 0, count: end)
      var valid = audioMask
      for codebook in 0..<codebooks {
        ids[codebook].replaceSubrange((cond - length)..<cond, with: conditioning.codes[codebook])
      }
      audioMask.replaceSubrange((cond - length)..<cond, with: conditioning.audioMask)
      audioMask.replaceSubrange(
        (end - frames)..<end, with: repeatElement(Float16(1), count: frames))
      valid.replaceSubrange((cond - length)..<cond, with: repeatElement(Float16(1), count: length))
      valid.replaceSubrange((end - frames)..<end, with: repeatElement(Float16(1), count: frames))

      var chosen = [[Int32]](repeating: [Int32](repeating: mask, count: frames), count: codebooks)
      let penalties = (0..<codebooks).map { Self.layerPenalty * Float($0) }
      var generator = SystemRandomNumberGenerator()
      for count in Self.schedule(frames: frames, steps: steps) where count > 0 {
        var result = try await function.run(inputs: [
          "input_ids": NDArray(scalars: ids.flatMap { $0 }, shape: [1, codebooks, end]),
          "audio_mask": NDArray(scalars: audioMask, shape: [1, end]),
          "valid": NDArray(scalars: valid, shape: [1, end]),
        ])
        guard let predictedOutput = result.remove("tokens")?.ndArray else {
          throw OmniVoiceError.missingOutput("tokens")
        }
        guard let confidenceOutput = result.remove("scores")?.ndArray else {
          throw OmniVoiceError.missingOutput("scores")
        }
        let predicted = try Self.codeIDs(predictedOutput)
        let confidence = Self.values(confidenceOutput, as: Float16.self)

        var candidates: [(codebook: Int, frame: Int, score: Float)] = []
        for codebook in 0..<codebooks {
          for frame in 0..<frames where chosen[codebook][frame] == mask {
            let source = codebook * window + window - frames + frame
            let certainty = Float(confidence[source]) - penalties[codebook]
            let noisy = certainty / Self.positionTemperature + Self.gumbel(&generator)
            candidates.append((codebook, frame, noisy))
          }
        }
        for candidate in candidates.sorted(by: { $0.score > $1.score }).prefix(count) {
          let source = candidate.codebook * window + window - frames + candidate.frame
          chosen[candidate.codebook][candidate.frame] = predicted[source]
        }
        for codebook in 0..<codebooks {
          ids[codebook].replaceSubrange((cond - frames)..<cond, with: chosen[codebook])
          ids[codebook].replaceSubrange((end - frames)..<end, with: chosen[codebook])
        }
      }
      return chosen
    }

    private func decodeAudio(_ tokens: [[Int32]], frames: Int) throws -> [Float] {
      guard let size = decoderFrames.first(where: { $0 >= frames }) else {
        throw OmniVoiceError.tooLong
      }
      let codebooks = OmniVoiceFrontEnd.codebooks
      let model = try decoderModel(size)
      var codes = [Int32](repeating: 0, count: codebooks * size)
      for codebook in 0..<codebooks {
        for frame in 0..<frames {
          codes[codebook * size + frame] = tokens[codebook][frame]
        }
      }
      let frameMask = (0..<size).map { $0 < frames ? Float16(1) : Float16(0) }
      let codesArray = try Self.array(codes, shape: [1, codebooks, size], dataType: .int32)
      let maskArray = try Self.array(frameMask, shape: [1, size], dataType: .float16)
      let result = try Self.predict(model, ["codes": codesArray, "frame_mask": maskArray])
      let audioOutput = try Self.output(result, "audio")
      let audio = Self.values(audioOutput, as: Float.self)
      return Array(audio.prefix(frames * Self.samplesPerFrame))
    }

    private func stepFunction(_ bucket: OmniVoiceBucket) throws -> InferenceFunction {
      guard let function = stepFunctions[bucket.function] else {
        throw OmniVoiceError.invalidConfiguration("no step function \(bucket.function)")
      }
      return function
    }

    private func decoderModel(_ frames: Int) throws -> MLModel {
      guard let model = decoderModels[frames] else {
        throw OmniVoiceError.invalidConfiguration("no decoder for \(frames) frames")
      }
      return model
    }

    /// How many of the masked codes each step reveals, on the time-shifted schedule of upstream.
    static func schedule(frames: Int, steps: Int) -> [Int] {
      let increment = 1 / Double(steps)
      let times = (0...steps).map { index -> Double in
        let t = index == steps ? 1 : Double(index) * increment
        return Self.tShift * t / (1 + (Self.tShift - 1) * t)
      }
      let total = frames * OmniVoiceFrontEnd.codebooks
      var remaining = total
      return (0..<steps).map { step -> Int in
        let count =
          step == steps - 1
          ? remaining
          : min(Int(ceil(Double(total) * (times[step + 1] - times[step]))), remaining)
        remaining -= count
        return count
      }
    }

    private static func gumbel(_ generator: inout SystemRandomNumberGenerator) -> Float {
      let u = Float.random(in: 0..<1, using: &generator)
      return -log(-log(u + 1e-10) + 1e-10)
    }

    private static func predict(_ model: MLModel, _ inputs: [String: MLMultiArray]) throws
      -> MLFeatureProvider
    {
      try model.prediction(
        from: MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) }))
    }

    private static func output(_ result: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
      guard let array = result.featureValue(for: name)?.multiArrayValue else {
        throw OmniVoiceError.missingOutput(name)
      }
      return array
    }

    /// A new array of `dataType` holding `elements` in row-major order.
    private static func array<T>(
      _ elements: [T], shape: [Int], dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
      let result = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: dataType)
      result.withUnsafeMutableBytes { buffer, strides in
        let pointer = buffer.assumingMemoryBound(to: T.self)
        let steps = shape.indices.map { strides[$0] }
        for (element, offset) in zip(elements, rowMajorOffsets(shape: shape, strides: steps)) {
          pointer[offset] = element
        }
      }
      return result
    }

    /// The elements of `array` in row-major order. Core ML may pad the rows of an output, so each
    /// element is read at the strides the array reports.
    private static func values<T>(_ array: MLMultiArray, as _: T.Type) -> [T] {
      let shape = array.shape.map(\.intValue)
      let strides = array.strides.map(\.intValue)
      let offsets = rowMajorOffsets(shape: shape, strides: strides)
      return array.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) -> [T] in
        let pointer = buffer.assumingMemoryBound(to: T.self)
        return offsets.map { pointer[$0] }
      }
    }

    /// The code ids of the step's `tokens`, each below the mask id.
    private static func codeIDs(_ array: NDArray) throws -> [Int32] {
      let ids: [Int32?] =
        switch array.scalarType {
        case .float16: values(array, as: Float16.self).map { Int32(exactly: $0) }
        case .int16: values(array, as: Int16.self).map { Int32($0) }
        case .int32: values(array, as: Int32.self)
        default: [nil]
        }
      let mask = OmniVoiceFrontEnd.mask
      guard let invalid = ids.firstIndex(where: { $0.map { $0 < 0 || $0 >= mask } ?? true }) else {
        return ids.compactMap { $0 }
      }
      throw OmniVoiceError.invalidOutput(
        "tokens \(array.scalarType) \(array.shape) strides \(array.strides): "
          + "\(ids[invalid].map(String.init) ?? "no id") at \(invalid)")
    }

    /// The elements of `array` in row-major order, read at the strides the array reports.
    private static func values<T: BitwiseCopyable>(_ array: NDArray, as _: T.Type) -> [T] {
      array.view(as: T.self).withUnsafePointer { pointer, shape, strides in
        rowMajorOffsets(
          shape: (0..<shape.count).map { shape[$0] },
          strides: (0..<strides.count).map { strides[$0] }
        ).map { pointer[$0] }
      }
    }

    /// The offsets of the elements of `shape` in row-major order, with `strides` between elements.
    private static func rowMajorOffsets(shape: [Int], strides: [Int]) -> [Int] {
      shape.indices.reduce([0]) { offsets, dimension in
        offsets.flatMap { offset in
          (0..<shape[dimension]).map { offset + $0 * strides[dimension] }
        }
      }
    }
  }
#endif
