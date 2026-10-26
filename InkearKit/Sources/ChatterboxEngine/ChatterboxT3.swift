#if canImport(CoreAI)
  import CoreAI
  import Foundation

  extension ChatterboxBundle {
    /// T3: one prefill of the text, then one decode per token over a cache the engine keeps.
    func generate(
      _ ids: [Int], voice: ChatterboxVoice, functions: ChatterboxFunctions,
      rng: inout SystemRandomNumberGenerator
    ) async throws -> ChatterboxGeneration {
      let hidden = Self.hidden
      let rows = prefillRows(ids, voice: voice)
      let length = rows[0].count / hidden
      guard let bucket = config.t3.prefillLengths.first(where: { $0 >= length }) else {
        throw ChatterboxError.tooLong
      }
      let rowSize = bucket * hidden
      var embeds = [Float16](repeating: 0, count: 2 * rowSize)
      for (index, values) in rows.enumerated() {
        embeds.replaceSubrange(index * rowSize..<index * rowSize + values.count, with: values)
      }
      var last = [Float16](repeating: 0, count: bucket)
      last[length - 1] = 1
      let (prefillCosines, prefillSines) = rope(Array(0..<bucket))
      let prefill = try await functions.run(
        "t3", "prefill_\(bucket)",
        [
          "embeds": NDArray(scalars: embeds, shape: [2, bucket, hidden]),
          "cos": NDArray(scalars: prefillCosines, shape: [bucket, Self.headDimension]),
          "sin": NDArray(scalars: prefillSines, shape: [bucket, Self.headDimension]),
          "last": NDArray(scalars: last, shape: [1, bucket]),
        ])
      guard let keys = prefill["keys"], let values = prefill["values"] else {
        throw ChatterboxError.missingOutput("keys")
      }
      var cache = T3Cache(keys: keys, values: values)
      var logits = try Self.logits(prefill)

      let sampling = config.sampling
      var generated = [config.speech.start]
      var ended = false
      for step in 0..<sampling.maxNewTokens {
        let token = Self.sample(logits, generated: generated, sampling: sampling, rng: &rng)
        generated.append(token)
        let position = length + step
        if token == config.speech.stop {
          ended = true
          break
        }
        if position >= config.t3.context { break }
        guard let window = config.t3.decodeWindows.first(where: { $0 > position }) else {
          throw ChatterboxError.tooLong
        }
        cache.grow(to: window)
        let embed = (0..<hidden).map { index -> Float16 in
          Float16(
            Float(speechEmbeddings[token * hidden + index])
              + Float(speechPositions[(step + 1) * hidden + index]))
        }
        let (cosines, sines) = rope([position])
        let mask = (0..<window).map { Float16($0 < position ? 0 : config.t3.masked) }
        let decoded = try await functions.run(
          "t3", "decode_\(window)",
          [
            "embed": NDArray(scalars: embed, shape: [1, hidden]),
            "cos": NDArray(scalars: cosines, shape: [1, Self.headDimension]),
            "sin": NDArray(scalars: sines, shape: [1, Self.headDimension]),
            "mask": NDArray(scalars: mask, shape: [1, window]),
            "key_cache": cache.keys,
            "value_cache": cache.values,
          ])
        guard let key = decoded["key"], let value = decoded["value"] else {
          throw ChatterboxError.missingOutput("key")
        }
        cache.write(key: key, value: value, at: position)
        logits = try Self.logits(decoded)
      }
      let tokens = generated.dropFirst().filter { $0 < config.speech.flowVocab }
      return ChatterboxGeneration(tokens: tokens, ended: ended)
    }

    private static func logits(_ outputs: [String: NDArray]) throws -> [Float16] {
      guard let logits = outputs["logits"] else { throw ChatterboxError.missingOutput("logits") }
      return logits.scalars(as: Float16.self)
    }

    /// The two prefill rows, each (conditioning + text + start of speech, 1024): the conditional
    /// row carries text embeddings plus positions; the unconditional row keeps only the positions.
    private func prefillRows(_ ids: [Int], voice: ChatterboxVoice) -> [[Float16]] {
      let hidden = Self.hidden
      let start = config.speech.start
      let startRow = (0..<hidden).map {
        Float(speechEmbeddings[start * hidden + $0]) + Float(speechPositions[$0])
      }
      let tail = startRow + startRow
      let text: [Float] = ids.enumerated().flatMap { index, id -> [Float] in
        (0..<hidden).map {
          Float(textEmbeddings[id * hidden + $0]) + Float(textPositions[index * hidden + $0])
        }
      }
      let positions: [Float] = ids.indices.flatMap { index -> [Float] in
        (0..<hidden).map { Float(textPositions[index * hidden + $0]) }
      }
      return [text, positions].map { part in
        (voice.t3Cond + part + tail).map { Float16($0) }
      }
    }

    /// cos and sin of `positions` under T3's RoPE, each (count, 64) row-major.
    private func rope(_ positions: [Int]) -> (cosines: [Float16], sines: [Float16]) {
      let inverse = config.t3.ropeInverseFrequencies
      var cosines: [Float16] = []
      var sines: [Float16] = []
      for position in positions {
        for index in 0..<Self.headDimension {
          let angle = Double(position) * inverse[index % inverse.count]
          cosines.append(Float16(cos(angle)))
          sines.append(Float16(sin(angle)))
        }
      }
      return (cosines, sines)
    }
  }

  /// T3's keys and values for one text, sized to the decode window in use: keys (layers · 2, 16,
  /// 64, slots), transposed for the attention product, and values (layers · 2, 16, slots, 64).
  ///
  /// The decode functions take the cache as input and return the token's key and value, which the
  /// cache writes at the token's position: the Neural Engine compiler of Core AI crashes on a
  /// model that writes into a state at a position known only at run time.
  struct T3Cache {
    private(set) var keys: NDArray
    private(set) var values: NDArray

    /// The cache the prefill returned.
    init(keys: NDArray, values: NDArray) {
      self.keys = keys
      self.values = values
    }

    private var slots: Int { keys.shape[3] }

    /// Moves the cache into arrays of `slots` slots, when it has fewer.
    mutating func grow(to slots: Int) {
      guard slots > self.slots else { return }
      let shape = keys.shape
      var grownKeys = NDArray(shape: [shape[0], shape[1], shape[2], slots], scalarType: .float16)
      var grownValues = NDArray(shape: [shape[0], shape[1], slots, shape[2]], scalarType: .float16)
      Self.copy(keys, into: &grownKeys)
      Self.copy(values, into: &grownValues)
      keys = grownKeys
      values = grownValues
    }

    /// Writes `key` (layers · 2, 16, 64, 1) and `value` (layers · 2, 16, 1, 64) at `slot`.
    mutating func write(key: NDArray, value: NDArray, at slot: Int) {
      Self.copy(key, into: &keys, offset: [0, 0, 0, slot])
      Self.copy(value, into: &values, offset: [0, 0, slot, 0])
    }

    /// Copies every element of the rank-4 `source` into `target`, shifted by `offset`.
    private static func copy(_ source: NDArray, into target: inout NDArray, offset: [Int] = [0, 0, 0, 0])
    {
      source.view(as: Float16.self).withUnsafePointer { from, shape, fromStrides in
        target.mutableView(as: Float16.self).withUnsafeMutablePointer { to, _, toStrides in
          for a in 0..<shape[0] {
            for b in 0..<shape[1] {
              for c in 0..<shape[2] {
                let fromRow = a * fromStrides[0] + b * fromStrides[1] + c * fromStrides[2]
                let toRow =
                  (a + offset[0]) * toStrides[0] + (b + offset[1]) * toStrides[1]
                  + (c + offset[2]) * toStrides[2] + offset[3] * toStrides[3]
                for d in 0..<shape[3] {
                  to[toRow + d * toStrides[3]] = from[fromRow + d * fromStrides[3]]
                }
              }
            }
          }
        }
      }
    }
  }
#endif
