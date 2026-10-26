import AVFoundation
import Synchronization

extension Speech {
  /// Averages equal-length `channels` into one mono signal, sample by sample.
  public init(averaging channels: [[Float]], sampleRate: Double) {
    let scale = 1 / Float(channels.count)
    let count = channels.first?.count ?? 0
    self.init(
      samples: (0..<count).map { index in channels.reduce(0) { $0 + $1[index] } * scale },
      sampleRate: sampleRate)
  }

  /// The speech `rate` times faster, with its pitch unchanged.
  public func stretched(by rate: Double) throws -> Speech {
    let stretcher = try SpeechStretcher(rate: rate, sampleRate: sampleRate)
    return Speech(
      samples: try stretcher.stretch(samples) + stretcher.finish(), sampleRate: sampleRate)
  }

  /// Seconds of silence before the first sound: 10 ms frames whose RMS stays under 0.01.
  public var leadingSilence: Double {
    Double(Self.silentFrames(in: frameEnergies)) / 100
  }

  /// Seconds of silence after the last sound.
  public var trailingSilence: Double {
    Double(Self.silentFrames(in: frameEnergies.reversed())) / 100
  }

  /// The speech with `seconds` of silence added before it, or, when negative, with that much of
  /// its leading silence cut, never the last 20 ms of it, so that soft onsets stay whole.
  public func padded(by seconds: Double) -> Speech {
    let count = Int((abs(seconds) * sampleRate).rounded())
    guard seconds < 0 else {
      return Speech(samples: [Float](repeating: 0, count: count) + samples, sampleRate: sampleRate)
    }
    let removable = max(Int((leadingSilence - Self.margin) * sampleRate), 0)
    return Speech(samples: Array(samples.dropFirst(min(count, removable))), sampleRate: sampleRate)
  }

  /// The speech over pink noise at -60 dBFS, about the background most voices carry.
  ///
  /// Engines and the narrator pause with zeros, and some voices have no background of their own.
  /// With noise-cancelling headphones, a drop to digital silence is felt as a change in pressure.
  public func overRoomTone() -> Speech {
    guard !samples.isEmpty else { return self }
    let noise = Self.pinkNoise(count: samples.count)
    let scale = Self.roomToneLevel / (noise.reduce(0) { $0 + $1 * $1 } / Float(noise.count)).squareRoot()
    return Speech(samples: zip(samples, noise).map { $0 + $1 * scale }, sampleRate: sampleRate)
  }

  /// Silence kept next to speech when silence is cut.
  private static let margin = 0.02

  /// RMS of the room tone: -60 dBFS.
  private static let roomToneLevel: Float = 0.001

  /// Paul Kellet's economy pink filter over white noise, after the filter has settled.
  private static func pinkNoise(count: Int) -> [Float] {
    let settling = 4_096
    var (b0, b1, b2): (Float, Float, Float) = (0, 0, 0)
    var noise: [Float] = []
    noise.reserveCapacity(count)
    for index in 0..<(settling + count) {
      let white = Float.random(in: -1...1)
      b0 = 0.99765 * b0 + white * 0.0990460
      b1 = 0.96300 * b1 + white * 0.2965164
      b2 = 0.57000 * b2 + white * 1.0526913
      if index >= settling { noise.append(b0 + b1 + b2 + white * 0.1848) }
    }
    return noise
  }

  /// Mean square of each 10 ms frame, the last one possibly shorter.
  private var frameEnergies: [Float] {
    let frame = max(Int(sampleRate / 100), 1)
    return stride(from: 0, to: samples.count, by: frame).map { start in
      let slice = samples[start..<min(start + frame, samples.count)]
      return slice.reduce(0) { $0 + $1 * $1 } / Float(slice.count)
    }
  }

  private static func silentFrames(in energies: some Sequence<Float>) -> Int {
    energies.prefix { $0 < 0.0001 }.count
  }

  /// The speech cut into consecutive parts, one per weight, the k-th lasting about its share of
  /// the weights.
  ///
  /// Each cut moves from its estimate to the quietest point within half the shorter of its two
  /// parts, so sentences read together come apart at the pauses between them.
  public func split(proportionalTo weights: [Double]) -> [Speech] {
    let frame = max(Int(sampleRate / 100), 1)
    let frames = samples.count / frame
    guard weights.count > 1, frames > 0 else {
      return [self] + Array(repeating: Speech(samples: [], sampleRate: sampleRate), count: max(weights.count - 1, 0))
    }
    let energy = (0..<frames).map { index in
      samples[(index * frame)..<((index + 1) * frame)].reduce(0) { $0 + $1 * $1 }
    }
    // Summed over 150 ms, so that a pause is quieter than a stop consonant.
    let quiet = energy.indices.map { index in
      energy[max(index - 7, 0)...min(index + 7, frames - 1)].reduce(0, +)
    }
    let total = weights.reduce(0, +)
    var cuts: [Int] = []
    var before = 0.0
    for (weight, next) in zip(weights, weights.dropFirst()) {
      before += weight
      let estimate = Int(Double(frames) * before / total)
      let reach = Int(Double(frames) * min(weight, next) / total / 2)
      let lower = max(cuts.last ?? 0, estimate - reach)
      let upper = max(lower, min(frames, estimate + reach))
      let score = { (cut: Int) in (quiet[min(cut, frames - 1)], abs(cut - estimate)) }
      cuts.append((lower...upper).min { score($0) < score($1) } ?? lower)
    }
    let bounds = [0] + cuts.map { $0 * frame } + [samples.count]
    return zip(bounds, bounds.dropFirst()).map {
      Speech(samples: Array(samples[$0..<$1]), sampleRate: sampleRate)
    }
  }

}


/// Makes speech that arrives in parts `rate` times faster, with its pitch unchanged: the parts
/// it returns, joined, are the whole speech stretched at once.
///
/// Renders through `AVAudioUnitTimePitch` in offline manual rendering mode, fed by a source node
/// that reads the samples given so far, in slices of a fixed size: the time-pitch output depends
/// on the slice sizes, and a player node fed buffer by buffer does not reproduce it. A slice is
/// rendered only while the input holds more than the time-pitch reads for it. The calls block
/// for the time the rendering takes.
public final class SpeechStretcher {
  private let rate: Double
  private let renderer: Renderer?
  private var rendered = 0

  public init(rate: Double, sampleRate: Double) throws {
    self.rate = rate
    renderer = rate == 1 ? nil : try Renderer(rate: rate, sampleRate: sampleRate)
  }

  /// The stretched audio that `samples` completes, which lags the input by a few slices.
  public func stretch(_ samples: [Float]) throws -> [Float] {
    guard let renderer else { return samples }
    renderer.feed.append(samples)
    var ready: [Float] = []
    while renderer.feed.unread >= renderer.margin {
      ready += try renderer.slice()
      if renderer.feed.starved { throw StretchError.render }
    }
    rendered += ready.count
    return ready
  }

  /// The rest of the stretched audio, once every part was given.
  public func finish() throws -> [Float] {
    guard let renderer else { return [] }
    let total = Int((Double(renderer.feed.received) / rate).rounded())
    var rest: [Float] = []
    while rendered + rest.count < total {
      rest += try renderer.slice()
    }
    let kept = Array(rest.prefix(max(total - rendered, 0)))
    rendered += kept.count
    return kept
  }

  private final class Renderer {
    let feed = Feed()
    /// Unread input that guarantees the next slice never reads past it.
    let margin: Int
    private let engine = AVAudioEngine()
    private let output: AVAudioPCMBuffer

    init(rate: Double, sampleRate: Double) throws {
      guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)
      else { throw StretchError.format }
      margin = 2 * Int((Double(Self.chunk) * rate).rounded(.up)) + Int(Self.chunk)
      let feed = feed
      let source = AVAudioSourceNode(format: format) { _, _, frames, list in
        guard let data = UnsafeMutableAudioBufferListPointer(list).first?.mData else {
          return kAudio_ParamError
        }
        feed.read(into: data.assumingMemoryBound(to: Float.self), count: Int(frames))
        return noErr
      }
      let timePitch = AVAudioUnitTimePitch()
      timePitch.rate = Float(rate)
      engine.attach(source)
      engine.attach(timePitch)
      try engine.connectNode(source, to: timePitch, format: format)
      try engine.connectNode(timePitch, to: engine.mainMixerNode, format: format)
      try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: Self.chunk)
      guard
        let output = AVAudioPCMBuffer(
          pcmFormat: engine.manualRenderingFormat, frameCapacity: Self.chunk)
      else { throw StretchError.render }
      self.output = output
      try engine.start()
    }

    deinit { engine.stop() }

    func slice() throws -> [Float] {
      let status = try engine.renderOffline(Self.chunk, to: output)
      guard status == .success, let channel = output.floatChannelData?[0] else {
        throw StretchError.render
      }
      return Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
    }

    private static let chunk: AVAudioFrameCount = 4096
  }

  /// The input given so far, read by the render thread; past its end it reads silence.
  private final class Feed: Sendable {
    private struct State {
      var samples: [Float] = []
      var read = 0
      var starved = false
    }
    private let state = Mutex(State())

    var received: Int { state.withLock { $0.samples.count } }
    var unread: Int { state.withLock { $0.samples.count - $0.read } }
    /// Whether a read went past the input given so far.
    var starved: Bool { state.withLock { $0.starved } }

    func append(_ samples: [Float]) {
      state.withLock { $0.samples += samples }
    }

    func read(into output: UnsafeMutablePointer<Float>, count: Int) {
      state.withLock { state in
        for index in 0..<count {
          let position = state.read + index
          output[index] = position < state.samples.count ? state.samples[position] : 0
        }
        if state.read + count > state.samples.count { state.starved = true }
        state.read += count
      }
    }
  }
}

private enum StretchError: Error {
  case format, render
}
