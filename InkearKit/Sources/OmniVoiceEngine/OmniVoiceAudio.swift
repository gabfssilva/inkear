import Foundation

/// The output stage of a piece: silence trimming, loudness, fades and padding, as the reference
/// pipeline does them, at 24 kHz.
enum OmniVoiceAudio {
  /// 10 ms.
  private static let chunk = 240
  /// 0.1 s: the fade at each edge and the silence around each piece.
  private static let tenthSecond = 2_400
  private static let silenceDBFS = -50.0

  static func postprocess(_ audio: [Float], voice: OmniVoiceVoice) -> [Float] {
    var trimmed = trimSilence(audio)
    if voice.rms < 0.1 {
      let gain = Float(voice.rms / 0.1)
      trimmed = trimmed.map { $0 * gain }
    }
    let fade = min(tenthSecond, trimmed.count / 2)
    for (index, gain) in ramp(from: 0, to: 1, count: fade).enumerated() {
      trimmed[index] *= gain
    }
    let fadeStart = trimmed.count - fade
    for (index, gain) in ramp(from: 1, to: 0, count: fade).enumerated() {
      trimmed[fadeStart + index] *= gain
    }
    let padding = [Float](repeating: 0, count: tenthSecond)
    return padding + trimmed + padding
  }

  /// Inner silences longer than 1 s shrink to 0.5 s on each side, and the edges keep 0.1 s. A
  /// 10 ms chunk is silent when it is below -50 dBFS.
  static func trimSilence(_ audio: [Float]) -> [Float] {
    let chunks = audio.count / chunk
    guard chunks > 0 else { return audio }
    let loud = (0..<chunks).map { index -> Bool in
      let start = index * chunk
      let squares = audio[start..<(start + chunk)].reduce(0.0) { $0 + Double($1) * Double($1) }
      let rms = (squares / Double(chunk)).squareRoot()
      return 20 * log10(max(rms, 1e-10)) >= silenceDBFS
    }
    guard let first = loud.firstIndex(of: true), let last = loud.lastIndex(of: true) else {
      return audio
    }
    var keep = [Bool](repeating: true, count: chunks)
    for index in keep.indices where index < first - 10 || index > last + 10 {
      keep[index] = false
    }
    var runStart: Int?
    for index in first...last {
      if !loud[index] {
        runStart = runStart ?? index
      } else if let start = runStart {
        if index - start > 100 {
          for silent in (start + 50)..<(index - 50) {
            keep[silent] = false
          }
        }
        runStart = nil
      }
    }
    var trimmed: [Float] = []
    for index in keep.indices where keep[index] {
      trimmed += audio[(index * chunk)..<((index + 1) * chunk)]
    }
    if keep[chunks - 1] {
      trimmed += audio[(chunks * chunk)...]
    }
    return trimmed
  }

  /// `count` values from `start` to `end`, both included, as `numpy.linspace` gives them.
  private static func ramp(from start: Float, to end: Float, count: Int) -> [Float] {
    guard count > 1 else { return count == 1 ? [start] : [] }
    return (0..<count).map { start + (end - start) * Float($0) / Float(count - 1) }
  }
}
