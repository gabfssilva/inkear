import AVFoundation
import Engines
import Foundation
import Synthesis
import Testing

/// Installs every downloadable engine and reads a Portuguese and an English sentence with its
/// default voice. The engines download several GB, so this runs only when `INKEAR_MODELS` names
/// the directory to install them into; `INKEAR_SAMPLES`, when set, receives the audio as WAV.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["INKEAR_MODELS"] != nil))
struct EnginesTests {
  static let sentences = [
    (
      "pt",
      "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem da Central um rapaz aqui do bairro, que eu conheço de vista e de chapéu."
    ),
    (
      "en",
      "The quick brown fox jumps over the lazy dog, and then it runs back into the forest before anyone notices."
    ),
  ]

  @Test(arguments: installableEngines.map { $0.id })
  func installsAndReads(_ id: EngineID) async throws {
    let environment = ProcessInfo.processInfo.environment
    let type = try #require(installableEngines.first { $0.id == id })
    let directory = URL(filePath: try #require(environment["INKEAR_MODELS"]))
      .appending(path: id.rawValue)
    if !type.isInstalled(in: directory) {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      try await type.install(into: directory) { fraction in
        print("\(id): installing \(fraction.map { "\(Int($0 * 100))%" } ?? "…")")
      }
    }
    let engine = try await type.load(from: directory) { _ in }

    for (language, text) in Self.sentences {
      let started = ContinuousClock.now
      let speech = try await Speech(
        collecting: engine.synthesize(
          text, voice: type.defaultVoice, language: language, speed: 1, quality: .normal))
      let seconds = started.duration(to: .now) / .seconds(1)
      print("\(id) \(language): \(speech.duration) s of audio in \(seconds) s, RTF \(seconds / speech.duration)")
      #expect((3...20).contains(speech.duration))
      if let samples = environment["INKEAR_SAMPLES"] {
        try write(speech, to: URL(filePath: samples).appending(path: "\(id.rawValue)-\(language).wav"))
      }
    }
  }

  private func write(_ speech: Speech, to url: URL) throws {
    let format = try #require(
      AVAudioFormat(standardFormatWithSampleRate: speech.sampleRate, channels: 1))
    let buffer = try #require(
      AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(speech.samples.count)))
    buffer.frameLength = buffer.frameCapacity
    let channel = try #require(buffer.floatChannelData)
    for (index, sample) in speech.samples.enumerated() { channel[0][index] = sample }
    try AVAudioFile(forWriting: url, settings: format.settings).write(from: buffer)
  }
}
