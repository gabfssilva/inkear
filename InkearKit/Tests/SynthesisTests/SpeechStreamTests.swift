import Synchronization
import Testing

@testable import Synthesis

@Suite struct SpeechStreamTests {
  @Test func collectingJoinsThePartsInOrder() async throws {
    let stream = SpeechStream { emit in
      emit(Speech(samples: [1, 2], sampleRate: 8_000))
      emit(Speech(samples: [], sampleRate: 8_000))
      emit(Speech(samples: [3], sampleRate: 8_000))
    }

    #expect(try await Speech(collecting: stream) == Speech(samples: [1, 2, 3], sampleRate: 8_000))
  }

  @Test func errorsReachTheConsumer() async {
    let stream = SpeechStream { _ in throw Failure() }

    await #expect(throws: Failure.self) { try await Speech(collecting: stream) }
  }

  @Test func stoppingTheIterationCancelsTheSynthesis() async throws {
    let emitted = Mutex(0)
    let stream = SpeechStream { emit in
      while true {
        try Task.checkCancellation()
        emitted.withLock { $0 += 1 }
        emit(Speech(samples: [0], sampleRate: 8_000))
        try await Task.sleep(for: .milliseconds(5))
      }
    }
    for try await _ in stream { break }

    await stream.cancel()
    let stopped = emitted.withLock { $0 }
    try await Task.sleep(for: .milliseconds(50))

    #expect(emitted.withLock { $0 } == stopped)
  }

  private struct Failure: Error {}
}
