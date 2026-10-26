import Foundation
import Testing

@testable import Synthesis

@Suite struct SpeechProcessingTests {
  /// One second of a 440 Hz tone at 24 kHz.
  private let tone = Speech(
    samples: (0..<24_000).map { Float(sin(2 * .pi * 440 * Double($0) / 24_000)) * 0.5 },
    sampleRate: 24_000)

  @Test(arguments: [0.8, 1.25, 2])
  func stretchingChangesDurationButNotPitch(rate: Double) throws {
    let stretched = try tone.stretched(by: rate)

    #expect(stretched.sampleRate == tone.sampleRate)
    #expect(stretched.samples.count == Int((Double(tone.samples.count) / rate).rounded()))
    // Away from the edges, a 440 Hz tone crosses zero 880 times a second.
    let middle = stretched.samples.dropFirst(2_400).dropLast(2_400)
    let crossings = zip(middle, middle.dropFirst()).count { ($0 < 0) != ($1 < 0) }
    let perSecond = Double(crossings) / (Double(middle.count) / stretched.sampleRate)
    #expect(abs(perSecond - 880) < 880 * 0.05)
  }

  @Test func stretchingByOneKeepsTheSpeech() throws {
    #expect(try tone.stretched(by: 1) == tone)
  }

  @Test(arguments: [0.8, 1.25, 2])
  func stretchingInPartsSoundsAsStretchingTheWhole(rate: Double) throws {
    // Three seconds of a gliding, pulsing tone, given in parts of uneven sizes.
    let samples = (0..<72_000).map { index -> Float in
      let time = Double(index) / 24_000
      return Float(sin(2 * .pi * (200 + 60 * time) * time) * (0.6 + 0.4 * sin(2 * .pi * 3 * time)))
    }
    let whole = try Speech(samples: samples, sampleRate: 24_000).stretched(by: rate)
    let stretcher = try SpeechStretcher(rate: rate, sampleRate: 24_000)
    var parts: [Float] = []
    var start = 0
    for size in [700, 13_000, 1, 24_000, 5_299, 29_000] {
      parts += try stretcher.stretch(Array(samples[start..<(start + size)]))
      start += size
    }
    parts += try stretcher.finish()

    #expect(parts == whole.samples)
  }

  @Test func splittingCutsAtThePausesNearTheEstimates() throws {
    // Three "sentences" of 1, 2 and 1.5 s apart by 0.3 s pauses; the second has an 80 ms gap,
    // like a stop consonant, near where the first cut is estimated.
    let rate = 24_000.0
    let sound = { (seconds: Double) in
      (0..<Int(seconds * rate)).map { Float(sin(2 * .pi * 440 * Double($0) / rate)) * 0.5 }
    }
    let pause = { (seconds: Double) in [Float](repeating: 0, count: Int(seconds * rate)) }
    let samples =
      sound(1) + pause(0.3) + sound(0.45) + pause(0.08) + sound(1.47) + pause(0.3) + sound(1)
      + sound(0.5)
    let speech = Speech(samples: samples, sampleRate: rate)

    // Character counts rarely match the durations; these are off by up to a third.
    let parts = speech.split(proportionalTo: [1.4, 1.8, 1.6])

    #expect(parts.count == 3)
    #expect(parts.flatMap(\.samples) == samples)
    let first = parts[0].duration
    let second = first + parts[1].duration
    #expect((1.0...1.3).contains(first), "first cut at \(first) s")
    #expect((3.3...3.6).contains(second), "second cut at \(second) s")
  }

  @Test func splittingIntoOnePartKeepsTheSpeech() {
    #expect(tone.split(proportionalTo: [3]) == [tone])
  }

  @Test func averagingDownmixesChannels() {
    let mono = Speech(averaging: [[1, 0, -1], [0, 0, 1]], sampleRate: 48_000)

    #expect(mono == Speech(samples: [0.5, 0, 0], sampleRate: 48_000))
  }
}

@Suite struct SilenceTests {
  /// 0.25 s of silence, 0.5 s of a tone, 0.15 s of silence, at 8 kHz.
  private let speech = Speech(
    samples: [Float](repeating: 0, count: 2_000)
      + (0..<4_000).map { Float(sin(2 * .pi * 220 * Double($0) / 8_000)) * 0.3 }
      + [Float](repeating: 0, count: 1_200),
    sampleRate: 8_000)

  @Test func silenceIsMeasuredAtBothEnds() {
    #expect(abs(speech.leadingSilence - 0.25) < 0.011)
    #expect(abs(speech.trailingSilence - 0.15) < 0.011)
  }

  @Test func paddingAddsSilenceBefore() {
    let padded = speech.padded(by: 0.5)

    #expect(padded.samples.count == speech.samples.count + 4_000)
    #expect(abs(padded.leadingSilence - 0.75) < 0.011)
  }

  @Test func negativePaddingCutsLeadingSilenceButKeepsAMargin() {
    #expect(abs(speech.padded(by: -0.1).leadingSilence - 0.15) < 0.011)
    #expect(abs(speech.padded(by: -5).leadingSilence - 0.02) < 0.011)
  }
}

@Suite struct RoomToneTests {
  /// 0.5 s of digital silence, 0.5 s of a tone and 0.5 s of digital silence, at 24 kHz.
  private let speech = Speech(
    samples: [Float](repeating: 0, count: 12_000)
      + (0..<12_000).map { Float(sin(2 * .pi * 220 * Double($0) / 24_000)) * 0.3 }
      + [Float](repeating: 0, count: 12_000),
    sampleRate: 24_000)

  @Test func digitalSilenceBecomesNoiseAtMinus60dBFS() {
    let toned = speech.overRoomTone()
    let frames = stride(from: 0, to: toned.samples.count, by: 240).map { start in
      toned.samples[start..<(start + 240)].reduce(0) { $0 + $1 * $1 } / 240
    }
    let silent = frames[0..<50] + frames[100...]

    #expect(silent.allSatisfy { $0 > 1e-7 })
    #expect((0.5e-6...2e-6) ~= silent.reduce(0, +) / Float(silent.count))
  }

  @Test func speechKeepsItsLengthAndLevel() {
    let toned = speech.overRoomTone()

    #expect(toned.samples.count == speech.samples.count)
    #expect(zip(toned.samples, speech.samples).allSatisfy { abs($0 - $1) < 0.01 })
  }
}
