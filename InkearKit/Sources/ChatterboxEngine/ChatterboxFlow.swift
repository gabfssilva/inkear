#if canImport(CoreAI)
  import CoreAI
  import Foundation

  extension ChatterboxBundle {
    private static let timeEmbeddingDimension = 320
    private static let harmonics = 9
    private static let sampleRate = 24_000
    /// Mel frames the flow makes of each speech token.
    static let melFramesPerToken = 2
    private static let flowEmbedding = 512

    /// The flow's mel for `tokens` after the voice's prompt, channel-major: (80, 2 · tokens.count).
    func flowMel(
      _ tokens: [Int], voice: ChatterboxVoice, steps: Int, functions: ChatterboxFunctions
    ) async throws -> [Float16] {
      let channels = Self.melChannels
      let prompt = voice.promptTokens.count
      let total = prompt + tokens.count
      guard let n = config.flow.tokens.first(where: { $0 >= total }) else {
        throw ChatterboxError.tooLong
      }
      let frames = Self.melFramesPerToken * n
      let ids = voice.promptTokens.map(Int.init) + tokens
      let tokenMask = (0..<n).map { Float16($0 < total ? 1 : 0) }
      let mu = try await functions.run(
        "flow_encoder", "n\(n)",
        [
          "embeds": NDArray(
            scalars: flowEmbeds(ids, padded: n), shape: [1, Self.flowEmbedding, 1, n]),
          "mask": NDArray(scalars: tokenMask, shape: [1, n]),
        ], output: "mu")

      let melFrames = voice.promptMel.count / channels
      var cond = [Float16](repeating: 0, count: channels * frames)
      for frame in 0..<melFrames {
        for channel in 0..<channels {
          cond[channel * frames + frame] = Float16(voice.promptMel[frame * channels + channel])
        }
      }
      let mask = (0..<frames).map { Float16($0 < Self.melFramesPerToken * total ? 1 : 0) }
      let spks = voice.spks.map { Float16($0) }
      let noiseFrames = noise.count / channels
      var x = (0..<channels).flatMap { channel in
        (0..<frames).map { Float16(noise[channel * noiseFrames + $0]) }
      }

      let schedule = Self.cosineSchedule(steps)
      for step in 0..<steps {
        x = try await functions.run(
          "flow_step", "m\(frames)",
          [
            "x": NDArray(scalars: x, shape: [1, channels, frames]),
            "mu": NDArray(scalars: mu, shape: [1, channels, frames]),
            "cond": NDArray(scalars: cond, shape: [1, channels, frames]),
            "spks": NDArray(scalars: spks, shape: [1, channels]),
            "mask": NDArray(scalars: mask, shape: [1, 1, frames]),
            "t_emb": NDArray(
              scalars: Self.timeEmbedding(schedule[step]), shape: [1, Self.timeEmbeddingDimension]),
            "dt": NDArray(scalars: [Float16(schedule[step + 1] - schedule[step])], shape: [1]),
          ], output: "x_next")
      }
      let generated = 2 * tokens.count
      return (0..<channels).flatMap { channel in
        (0..<generated).map { x[channel * frames + 2 * prompt + $0] }
      }
    }

    /// The flow encoder's embeddings of `ids`, channel-major (512, `padded`), zero past the ids:
    /// the app looks them up, since MPSGraph runs a lookup by integer index on the GPU.
    private func flowEmbeds(_ ids: [Int], padded: Int) -> [Float16] {
      let width = Self.flowEmbedding
      var embeds = [Float16](repeating: 0, count: width * padded)
      for (position, id) in ids.enumerated() {
        for channel in 0..<width {
          embeds[channel * padded + position] = flowEmbeddings[id * width + channel]
        }
      }
      return embeds
    }

    /// The waveform of `tokens` at 24 kHz, without the last token's audio.
    func audio(
      _ tokens: [Int], voice: ChatterboxVoice, steps: Int, functions: ChatterboxFunctions,
      rng: inout SystemRandomNumberGenerator
    ) async throws -> [Float] {
      let mel = try await flowMel(tokens, voice: voice, steps: steps, functions: functions)
      let frames = 2 * tokens.count
      // The function leaves out the predictor's final absolute value.
      let f0 = try await windows(
        functions, package: "f0", output: "f0", mel: mel, frames: frames, rate: 1, source: nil
      ).map { abs($0) }
      let source = excitation(f0: f0, rng: &rng)
      let wave = try await windows(
        functions, package: "hift", output: "audio", mel: mel, frames: frames,
        rate: config.vocoder.samplesPerFrame, source: source)
      return Self.finish(
        wave.map { Float($0) }, tokens: tokens.count, samplesPerToken: config.samplesPerToken)
    }

    /// Runs a vocoder package over fixed windows of the channel-major `mel` (80, frames), keeping
    /// each window's trusted outputs. `rate` is the output count per frame; `source` is HiFT's
    /// per-sample excitation.
    private func windows(
      _ functions: ChatterboxFunctions, package: String, output: String, mel: [Float16], frames: Int,
      rate: Int, source: [Float16]?
    ) async throws -> [Float16] {
      let channels = Self.melChannels
      let size = config.vocoder.frames
      let hop = config.vocoder.samplesPerFrame
      let padded = frames + size
      var paddedMel = [Float16](repeating: 0, count: channels * padded)
      for channel in 0..<channels {
        paddedMel.replaceSubrange(
          channel * padded..<channel * padded + frames,
          with: mel[channel * frames..<(channel + 1) * frames])
      }
      let paddedSource = (source ?? []) + [Float16](repeating: 0, count: size * hop)

      var out = [Float16](repeating: 0, count: frames * rate)
      for (start, keep, stop) in Self.chunks(
        frames: frames, size: size, context: config.vocoder.context)
      {
        var inputs: [String: NDArray] = [:]
        let melWindow = (0..<channels).flatMap { channel in
          paddedMel[channel * padded + start..<channel * padded + start + size]
        }
        inputs["mel"] = NDArray(scalars: melWindow, shape: [1, channels, size])
        if source != nil {
          inputs["source"] = NDArray(
            scalars: paddedSource[start * hop..<(start + size) * hop], shape: [1, 1, size * hop])
        }
        let values = try await functions.run(package, "t\(size)", inputs, output: output)
        out.replaceSubrange(
          keep * rate..<stop * rate,
          with: values[(keep - start) * rate..<(stop - start) * rate])
      }
      return out
    }

    /// (input start, keep start, keep stop) windows covering `frames`. Kept frames have `context`
    /// frames of real input on each side, or the utterance edge.
    static func chunks(frames: Int, size: Int, context: Int) -> [(start: Int, keep: Int, stop: Int)] {
      var spans: [(start: Int, keep: Int, stop: Int)] = []
      var out = 0
      while out < frames {
        let start = min(max(out - context, 0), max(frames - size, 0))
        let stop = start + size >= frames ? frames : start + size - context
        spans.append((start, out, stop))
        out = stop
      }
      return spans
    }

    /// HiFT's harmonic source for the per-frame `f0`: 480 samples per frame. Phases accumulate in
    /// float64, since in float16 they would lose their precision within a second.
    private func excitation(f0: [Float16], rng: inout SystemRandomNumberGenerator) -> [Float16] {
      let vocoder = config.vocoder
      let offsets = (0..<Self.harmonics).map { harmonic -> Double in
        harmonic == 0 ? 0 : Double.random(in: -Double.pi...Double.pi, using: &rng)
      }
      var phases = [Double](repeating: 0, count: Self.harmonics)
      var samples: [Float16] = []
      samples.reserveCapacity(f0.count * vocoder.samplesPerFrame)
      for frame in f0 {
        let pitch = Double(frame)
        let voiced = pitch > 10 ? 1.0 : 0.0
        let noiseScale = voiced * 0.003 + (1 - voiced) * 0.1 / 3
        for _ in 0..<vocoder.samplesPerFrame {
          var sum = 0.0
          for harmonic in 0..<Self.harmonics {
            phases[harmonic] += pitch * Double(harmonic + 1) / Double(Self.sampleRate)
            let phase = phases[harmonic] - phases[harmonic].rounded(.down)
            let sine = 0.1 * sin(2 * Double.pi * phase + offsets[harmonic])
            let noise = noiseScale * Self.gaussian(&rng)
            sum += vocoder.sourceWeight[harmonic] * (sine * voiced + noise)
          }
          samples.append(Float16(tanh(sum + vocoder.sourceBias)))
        }
      }
      return samples
    }

    /// The fade in and the trim the reference applies: the first 20 ms are silent, then a 20 ms
    /// raised-cosine ramp, and the last token's audio is dropped.
    static func finish(_ audio: [Float], tokens: Int, samplesPerToken: Int) -> [Float] {
      let trim = sampleRate / 50
      var output = audio
      for index in 0..<min(2 * trim, output.count) {
        let fade = index < trim ? 0 : (cos(Double.pi * (1 - Double(index - trim) / Double(trim - 1))) + 1) / 2
        output[index] *= Float(fade)
      }
      return Array(output.prefix(max(1, tokens - 1) * samplesPerToken))
    }

    /// The cosine schedule's timesteps t_0 = 0 … t_steps = 1.
    static func cosineSchedule(_ steps: Int) -> [Double] {
      (0...steps).map { 1 - cos(Double($0) / Double(steps) * 0.5 * Double.pi) }
    }

    /// The estimator's sinusoidal embedding of `t` (scale 1000), in float16.
    static func timeEmbedding(_ t: Double) -> [Float16] {
      let half = timeEmbeddingDimension / 2
      let frequencies = (0..<half).map { exp(Double($0) * -(log(10_000.0) / Double(half - 1))) }
      let angles = frequencies.map { 1000 * t * $0 }
      return angles.map { Float16(sin($0)) } + angles.map { Float16(cos($0)) }
    }

    private static func gaussian(_ rng: inout SystemRandomNumberGenerator) -> Double {
      let u = 1 - Double.random(in: 0..<1, using: &rng)
      let v = Double.random(in: 0..<1, using: &rng)
      return sqrt(-2 * log(u)) * cos(2 * Double.pi * v)
    }
  }
#endif
