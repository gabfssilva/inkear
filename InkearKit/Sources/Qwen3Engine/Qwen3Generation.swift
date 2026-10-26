// Adapted from speech-swift/Sources/Qwen3TTSCoreML/{PromptBuilder,Sampler,MultiCodeDecoderCoreML}.swift, Apache-2.0.
#if canImport(CoreAI)
  import CoreAI
  import Foundation
  import Synthesis

  /// The codec frames of one utterance: `codebooks[k][t]` is codebook k of frame t.
  struct Codes {
    let codebooks: [[Int32]]
    /// Whether the model ended the speech itself, rather than the frame cap.
    let finished: Bool
  }

  extension Qwen3Bundle {
    private static let codecEOS = 2150
    private static let codecVocabulary = 3072
    private static let temperature = 0.9
    private static let topK = 50
    private static let repetitionPenalty = 1.05

    /// The frames of `text` spoken with `speaker`, or nil when the text has more tokens than the
    /// prefill holds. Once `prefixFrames` frames exist and more may follow, `onPrefix` receives
    /// them.
    func codes(
      for text: String, codecLanguage: Int, speaker: [Float], prefixFrames: Int? = nil,
      onPrefix: ([[Int32]]) async throws -> Void = { _ in },
      isolation: isolated (any Actor)? = #isolation
    ) async throws -> Codes? {
      let tokens = tokenizer.encode(text)
      guard tokens.count <= Self.maxTextTokens else { return nil }
      let rows = prompt(tokens, codecLanguage: codecLanguage, speaker: speaker)
      let length = rows.count

      let prefilled = try await prefill.outputs([
        "input_embeds": NDArray(
          scalars: Self.columns(of: rows).map(Float16.init),
          shape: [1, Self.hidden, 1, Self.prefillColumns]),
        "key_padding_mask": NDArray(
          scalars: (0..<Self.prefillColumns).flatMap { row in
            (0..<Self.prefillColumns).map { column -> Float16 in column <= row ? 0 : -1e4 }
          }, shape: [Self.prefillColumns, Self.prefillColumns]),
      ])
      var cache = TalkerCache()
      try cache.write(Self.output(prefilled, "key_cache"), Self.output(prefilled, "value_cache"), count: length)
      var logits = Array(
        try Self.values(prefilled, "logits")
          .dropFirst((length - 1) * Self.codecVocabulary).prefix(Self.codecVocabulary))
      let prompted = try Self.values(prefilled, "hidden_states")
      var hidden = (0..<Self.hidden).map { prompted[$0 * Self.prefillColumns + length - 1] }

      var history: [Int] = []
      var codebooks = Array(repeating: [Int32](), count: Self.codebooks)
      var finished = false
      let limit = min(Self.frames, Self.positions - length)
      for index in 0..<limit {
        try Task.checkCancellation()
        let token = Self.codecToken(logits, history: history)
        if token == Self.codecEOS {
          finished = true
          break
        }
        history.append(token)
        let first = codeEmbedding(token)
        let residual = try await predictResidual(hidden: hidden, first: first)
        codebooks[0].append(Int32(token))
        for (codebook, value) in residual.codes.enumerated() {
          codebooks[codebook + 1].append(value)
        }
        guard index + 1 < limit else { break }
        if index + 1 == prefixFrames { try await onPrefix(codebooks) }

        let position = length + index
        let next = try await decode.outputs([
          "input_embeds": NDArray(
            scalars: Self.sum(Self.sum(first, residual.embedding), textPad).map(Float16.init),
            shape: [1, Self.hidden, 1, 1]),
          "key_padding_mask": NDArray(
            scalars: (0..<Self.positions).map { $0 < position ? Float16(0) : -1e4 },
            shape: [1, Self.positions]),
          "position": NDArray(
            scalars: (0..<Self.positions).map { $0 == position ? Float16(1) : 0 },
            shape: [1, Self.positions]),
          "key_cache": cache.keys,
          "value_cache": cache.values,
        ])
        try cache.write(Self.output(next, "key"), Self.output(next, "value"), at: position)
        logits = try Self.values(next, "logits")
        hidden = try Self.values(next, "hidden_states")
      }
      return Codes(codebooks: codebooks, finished: finished)
    }

    /// The prompt rows: role, control and speaker, then the text with the codec padding, then the
    /// closing control embeddings.
    private func prompt(_ tokens: [Int], codecLanguage: Int, speaker: [Float]) -> [[Float]] {
      var rows: [[Float]] = []
      for token in [151644, 77091, 198] {  // <|im_start|> assistant newline
        rows.append(textEmbedding(token))
      }
      for token in [2154, 2156, codecLanguage, 2157] {
        rows.append(Self.sum(textPad, codeEmbedding(token)))
      }
      rows.append(Self.sum(textPad, speaker))
      let codecPad = codeEmbedding(2148)
      rows.append(Self.sum(textBos, codecPad))
      for token in tokens {
        rows.append(Self.sum(textEmbedding(token), codecPad))
      }
      rows.append(Self.sum(textEos, codecPad))
      rows.append(Self.sum(textPad, codeEmbedding(2149)))
      return rows
    }

    /// The 15 residual codes of a frame and the sum of their embeddings, predicted from the
    /// talker's hidden state and the frame's first code, with Gumbel noise for sampling.
    private func predictResidual(hidden: [Float], first: [Float]) async throws -> (
      codes: [Int32], embedding: [Float]
    ) {
      let noise = (0..<(Self.codebooks - 1) * Self.residualVocabulary).map { _ in
        Float16(-log(-log(Float.random(in: Float.ulpOfOne..<1))))
      }
      let output = try await multiCodeDecoder.outputs([
        "hidden_states": NDArray(scalars: hidden.map(Float16.init), shape: [1, Self.hidden, 1, 1]),
        "code_embeds": NDArray(scalars: first.map(Float16.init), shape: [1, Self.hidden, 1, 1]),
        "noise": NDArray(scalars: noise, shape: [Self.codebooks - 1, Self.residualVocabulary]),
      ])
      return (
        try Self.values(output, "codes").map { Int32($0.rounded()) },
        try Self.values(output, "codes_embeds")
      )
    }

    private static func output(_ outputs: [String: NDArray], _ name: String) throws -> NDArray {
      guard let array = outputs[name] else { throw Qwen3Error.missingOutput(name) }
      return array
    }

    private static func values(_ outputs: [String: NDArray], _ name: String) throws -> [Float] {
      try output(outputs, name).scalars(as: Float16.self).map(Float.init)
    }

    /// The next first-codebook token. Tokens already sampled are penalized, control tokens are
    /// masked, and the end token is allowed from the third frame on, so the speech is never empty.
    private static func codecToken(_ logits: [Float], history: [Int]) -> Int {
      var logits = logits.map(Double.init)
      for token in Set(history) {
        logits[token] =
          logits[token] > 0 ? logits[token] / repetitionPenalty : logits[token] * repetitionPenalty
      }
      for token in 2048..<codecVocabulary where history.count < 2 || token != codecEOS {
        logits[token] = -.infinity
      }
      return sample(logits.map { $0 / temperature })
    }

    /// Draws an index from the softmax of the `topK` largest logits.
    private static func sample(_ logits: [Double]) -> Int {
      let top = logits.indices.sorted { logits[$0] > logits[$1] }.prefix(topK)
      let peak = logits[top.first ?? 0]
      let weights = top.map { exp(logits[$0] - peak) }
      var draw = Double.random(in: 0..<weights.reduce(0, +))
      for (index, weight) in zip(top, weights) {
        draw -= weight
        if draw < 0 { return index }
      }
      return top.last ?? 0
    }

    /// The row-major `[hidden, prefillColumns]` values of the prompt rows, one row per column.
    private static func columns(of rows: [[Float]]) -> [Float] {
      var values = [Float](repeating: 0, count: hidden * prefillColumns)
      for (column, row) in rows.enumerated() {
        for (channel, value) in row.enumerated() {
          values[channel * prefillColumns + column] = value
        }
      }
      return values
    }

    private static func sum(_ a: [Float], _ b: [Float]) -> [Float] {
      zip(a, b).map { $0 + $1 }
    }
  }

  /// The talker's keys and values, `[1, 28672, 1, positions]` each, channel
  /// `layer * 1024 + head * 128 + dim`.
  ///
  /// The decode function takes the cache as input and returns the step's key and value, which the
  /// cache writes at the step's slot: the Neural Engine compiler of Core AI crashes on a model that
  /// writes into a state at a position known only at run time.
  private struct TalkerCache {
    private(set) var keys = NDArray(
      shape: [1, Qwen3Bundle.talkerKVWidth, 1, Qwen3Bundle.positions], scalarType: .float16)
    private(set) var values = NDArray(
      shape: [1, Qwen3Bundle.talkerKVWidth, 1, Qwen3Bundle.positions], scalarType: .float16)

    /// Writes the first `count` positions of the prefill's `key` and `value` at slot 0 on.
    mutating func write(_ key: NDArray, _ value: NDArray, count: Int) {
      Self.copy(key, count: count, into: &keys, at: 0)
      Self.copy(value, count: count, into: &values, at: 0)
    }

    /// Writes one step's `key` and `value`, `[1, 28672, 1, 1]`, at `slot`.
    mutating func write(_ key: NDArray, _ value: NDArray, at slot: Int) {
      Self.copy(key, count: 1, into: &keys, at: slot)
      Self.copy(value, count: 1, into: &values, at: slot)
    }

    /// Copies the first `count` positions of every channel of `source` into `target` from `slot`.
    private static func copy(_ source: NDArray, count: Int, into target: inout NDArray, at slot: Int) {
      source.view(as: Float16.self).withUnsafePointer { from, _, fromStrides in
        target.mutableView(as: Float16.self).withUnsafeMutablePointer { to, _, toStrides in
          for channel in 0..<Qwen3Bundle.talkerKVWidth {
            for position in 0..<count {
              to[channel * toStrides[1] + (slot + position) * toStrides[3]] =
                from[channel * fromStrides[1] + position * fromStrides[3]]
            }
          }
        }
      }
    }
  }
#endif
