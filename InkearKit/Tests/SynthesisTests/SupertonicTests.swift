import Accelerate
import Foundation
import Testing

@testable import Synthesis

/// Checks the Swift pipeline against reference runs of the same Core ML packages from Python,
/// written by `supertonic-convert` and `supertonic-fixtures` under `tools/coreml/build`.
///
/// The references come from the Mac's CPU. Other backends, such as the Neural Engine or the iOS
/// simulator's CPU, compute the fp16 models differently, and the diffusion steps amplify the
/// difference into waveforms that drift apart while sounding alike; there, the audio only has to
/// sound the same, by the criterion of `tools/coreml/tests/test_coreml.py`.
@Suite struct SupertonicTests {
  static let build = URL(filePath: #filePath).deletingLastPathComponent()
    .appending(path: "../../../tools/coreml/build").standardized
  static let engine = Task {
    try await Supertonic.load(from: build.appending(path: "supertonic-3"), computeUnits: .cpuOnly)
  }

  @Test(arguments: ["short-pt", "caminho-pt", "long-en"])
  func matchesThePythonPipeline(fixture name: String) async throws {
    let fixture = try Fixture(name: name, in: Self.build.appending(path: "fixtures"))
    let engine = try await Self.engine.value
    let style = try #require(await engine.style(fixture.voice))
    let noise = try fixture.array("noise")

    let ids = await engine.ids(of: fixture.text, fixture.lang)
    let run = try await engine.run(
      ids: ids, style: style, steps: fixture.steps, speed: fixture.speed, noise: { _ in noise })

    #expect(ids == fixture.ids)
    #expect(
      run.bucket == Bucket(text: fixture.bucket.text_length, latent: fixture.bucket.latent_length))
    #expect(abs(run.seconds - fixture.seconds) / fixture.seconds < 1e-4)
    #expect(run.frames == fixture.frames)
    let wav = try fixture.array("wav")
    #if os(macOS)
      let textEmbeddingError = relativeError(run.textEmbedding, try fixture.array("text_emb"))
      let latentError = relativeError(run.latent, try fixture.array("latent"))
      let samplesError = relativeError(run.samples, wav)
      #expect(textEmbeddingError < 1e-3)
      #expect(latentError < 1e-2)
      #expect(samplesError < 1e-2)
    #endif
    let otherNoise = try await engine.run(
      ids: ids, style: style, steps: fixture.steps, speed: fixture.speed)
    let distance = logSpectralDistance(run.samples, wav)
    let natural = logSpectralDistance(otherNoise.samples, wav)
    #expect(distance < natural / 3)
  }

  @Test func splitsTextLongerThanTheLargestBucket() async throws {
    let engine = try await Self.engine.value
    let sentence = "Esta frase se repete muitas vezes, sem nenhum ponto final no meio dela"
    let text = Array(repeating: sentence, count: 12).joined(separator: ", ") + "."

    let speech = try await Speech(collecting: engine.synthesize(text, voice: "F1", language: "pt"))

    #expect(speech.duration > 30)
  }

  @Test func rejectsUnknownVoices() async throws {
    let engine = try await Self.engine.value

    await #expect(throws: SupertonicError.unknownVoice("Z9")) {
      try await Speech(collecting: engine.synthesize("Olá.", voice: "Z9", language: "pt"))
    }
  }

  private func relativeError(_ actual: [Float], _ expected: [Float]) -> Float {
    let difference = zip(actual, expected).reduce(Float(0)) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }
    let norm = expected.reduce(Float(0)) { $0 + $1 * $1 }
    return actual.count == expected.count ? (difference / norm).squareRoot() : .infinity
  }
}

/// Mean log-spectral distance in dB, ignoring bins `floor` dB below the loudest one, as
/// `supertonic_coreml.metrics.log_spectral_distance` computes it.
private func logSpectralDistance(_ actual: [Float], _ expected: [Float], floor: Float = 80) -> Float
{
  let length = min(actual.count, expected.count)
  let (a, e) = (spectrum(actual.prefix(length)), spectrum(expected.prefix(length)))
  let cutoff = (e.lazy.compactMap { $0.max() }.max() ?? 0) - floor
  let frames = zip(a, e).map { frame in
    let squares = zip(frame.0, frame.1).map { max($0, cutoff) - max($1, cutoff) }.map { $0 * $0 }
    return (squares.reduce(0, +) / Float(squares.count)).squareRoot()
  }
  return frames.reduce(0, +) / Float(max(frames.count, 1))
}

/// Magnitudes in dB of 2048-sample Hann-windowed frames every 512 samples.
private func spectrum(_ wav: ArraySlice<Float>) -> [[Float]] {
  let size = 2048
  let window = (0..<size).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(size - 1)) }
  guard
    let dft = try? vDSP.DiscreteFourierTransform(
      count: size, direction: .forward, transformType: .complexComplex, ofType: Float.self)
  else { return [] }
  let zeros = [Float](repeating: 0, count: size)
  return stride(from: wav.startIndex, through: wav.endIndex - size, by: 512).map { start in
    let frame = vDSP.multiply(wav[start..<(start + size)], window)
    let (real, imaginary) = dft.transform(real: frame, imaginary: zeros)
    return (0...(size / 2)).map { bin in
      20 * log10((real[bin] * real[bin] + imaginary[bin] * imaginary[bin]).squareRoot() + 1e-9)
    }
  }
}

private struct Fixture: Decodable {
  struct Bucket: Decodable {
    let text_length: Int
    let latent_length: Int
  }
  struct Array: Decodable {
    let file: String
    let shape: [Int]
  }

  let text: String
  let lang: String
  let voice: String
  let steps: Int
  let speed: Double
  let bucket: Bucket
  let ids: [Int32]
  let seconds: Double
  let frames: Int
  let arrays: [String: Array]
  private(set) var directory = URL(filePath: "/")

  enum CodingKeys: CodingKey {
    case text, lang, voice, steps, speed, bucket, ids, seconds, frames, arrays
  }

  init(name: String, in directory: URL) throws {
    self = try JSONDecoder().decode(
      Fixture.self, from: Data(contentsOf: directory.appending(path: "\(name).json")))
    self.directory = directory
  }

  func array(_ name: String) throws -> [Float] {
    let file = try #require(arrays[name]).file
    let data = try Data(contentsOf: directory.appending(path: file))
    return data.withUnsafeBytes { Swift.Array($0.bindMemory(to: Float.self)) }
  }
}
