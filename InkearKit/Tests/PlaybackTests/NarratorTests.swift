import Foundation
import PDFStructure
import Synthesis
import Testing

@testable import Playback

@Suite struct NarratorTests {
  @Test func runsHoldWhatIsSynthesizedBeforeTheQueuedAudioEnds() {
    #expect(Narrator.runCharacters(queued: 0, realTimeFactor: 0.5, speed: 1) == 0)
    #expect(Narrator.runCharacters(queued: 5, realTimeFactor: nil, speed: 1) == 0)
    // 5 s queued at RTF 0.5 leave 7.5 s of synthesis, which reads 7.5 s of speech.
    #expect(Narrator.runCharacters(queued: 5, realTimeFactor: 0.5, speed: 1) == 108)
    #expect(Narrator.runCharacters(queued: 5, realTimeFactor: 0.5, speed: 2) == 217)
    #expect(Narrator.runCharacters(queued: 30, realTimeFactor: 0.5, speed: 1) == 300)
  }
}

@Suite struct PauseTests {
  @Test(arguments: [
    ("Ela abriu a janela.", 0.45), ("Quem era?", 0.45), ("Que susto!", 0.45),
    ("E então…", 0.6), ("E então...", 0.6), ("Disse o seguinte:", 0.35), ("Primeiro; ", 0.35),
    ("— Vamos embora.”", 0.45), ("um título sem ponto", 0.25),
  ])
  func punctuationSetsThePauseBetweenSentences(text: String, seconds: Double) {
    #expect(Narrator.pause(after: text, before: .sentence) == seconds)
  }

  @Test func newParagraphsAndSectionsPauseLonger() {
    #expect(Narrator.pause(after: "Fim.", before: .paragraph) == 0.8)
    #expect(Narrator.pause(after: "Fim.", before: .section) == 1.2)
  }
}

/// Plays through the Mac's audio output.
@MainActor @Suite struct NarratorTimeTests {
  /// Speaks any text as `seconds` of a constant level, inaudible but above the narrator's
  /// silence threshold.
  private struct Steady: SpeechEngine {
    let seconds: Double
    var voices: [String] { ["steady"] }
    var languages: Set<String> { ["en"] }
    static var usesQuality: Bool { false }

    func synthesize(
      _ text: String, voice: String, language: String, speed: Double, quality: SpeechQuality
    ) -> SpeechStream {
      let speech = Speech(
        samples: [Float](repeating: 0.02, count: Int(seconds * 24_000)), sampleRate: 24_000)
      return SpeechStream { emit in emit(speech) }
    }
  }

  @Test func elapsedTimeFollowsTheAudioWithinASentence() async throws {
    // 58 characters, which the narrator estimates at 4 s.
    let text = String(repeating: "a", count: 57) + "."
    let script = Script(
      DocumentStructure(
        pageCount: 1, blocks: [Block(kind: .paragraph, text: text, pages: 0...0)], sections: [],
        contentsStart: nil, mainStart: nil))
    let narrator = Narrator(
      script: script, title: "", artwork: nil, engine: Steady(seconds: 4),
      output: AudioOutput(), nowPlaying: nil,
      voice: VoiceID(engine: .supertonic, name: "steady"), speed: 1, language: "en",
      quality: .normal, position: script.start)
    defer { narrator.stop() }

    narrator.play()
    let deadline = ContinuousClock.now + .seconds(5)
    while narrator.state != .playing, ContinuousClock.now < deadline {
      try await Task.sleep(for: .milliseconds(10))
    }
    #expect(narrator.state == .playing)
    try await Task.sleep(for: .seconds(1.5))
    let playing = narrator.sectionTime.elapsed
    #expect((1...2).contains(playing))

    narrator.pause()
    let paused = narrator.sectionTime.elapsed
    try await Task.sleep(for: .seconds(0.5))
    #expect(narrator.sectionTime.elapsed == paused)
    #expect(narrator.position == script.start)
  }
}
