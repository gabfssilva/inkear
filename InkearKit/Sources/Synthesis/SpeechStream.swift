/// The speech of one request in parts that play back to back, each yielded as soon as it is
/// synthesized. It yields at least one part; parts may be empty.
///
/// The synthesis runs in a task of its own, which stops when the consumer stops iterating.
public struct SpeechStream: AsyncSequence, Sendable {
  public typealias Element = Speech

  private let parts: AsyncThrowingStream<Speech, any Error>
  private let producer: Task<Void, Never>

  /// A stream of the parts `produce` emits.
  public init(
    _ produce: @escaping @Sendable (_ emit: @escaping @Sendable (Speech) -> Void) async throws
      -> Void
  ) {
    let (parts, continuation) = AsyncThrowingStream<Speech, any Error>.makeStream()
    let producer = Task {
      do {
        try await produce { continuation.yield($0) }
        continuation.finish()
      } catch {
        continuation.finish(throwing: error)
      }
    }
    continuation.onTermination = { _ in producer.cancel() }
    self.parts = parts
    self.producer = producer
  }

  public func makeAsyncIterator() -> AsyncThrowingStream<Speech, any Error>.Iterator {
    parts.makeAsyncIterator()
  }

  /// Stops the synthesis and waits until the engine has let go of it.
  public func cancel() async {
    producer.cancel()
    await producer.value
  }
}

extension Speech {
  /// The parts of `stream` joined into one speech.
  public init(collecting stream: SpeechStream) async throws {
    var samples: [Float] = []
    var sampleRate = 0.0
    for try await part in stream {
      samples += part.samples
      sampleRate = part.sampleRate
    }
    self.init(samples: samples, sampleRate: sampleRate)
  }
}

/// Hands the audio an engine synthesizes piece by piece to a ``SpeechStream`` at the requested
/// speed: each piece is stretched as it comes, so the parts sound as the whole speech stretched
/// at once.
public final class SpeechEmitter {
  private let sampleRate: Double
  private let stretcher: SpeechStretcher
  private let emit: @Sendable (Speech) -> Void

  /// - Parameter speed: The rate the audio is stretched by; 1 for engines that apply the speed
  ///   themselves.
  public init(sampleRate: Double, speed: Double, emit: @escaping @Sendable (Speech) -> Void) throws
  {
    self.sampleRate = sampleRate
    stretcher = try SpeechStretcher(rate: speed, sampleRate: sampleRate)
    self.emit = emit
  }

  /// Emits the stretched audio of `samples` that is ready; throws once the synthesis was
  /// cancelled.
  public func append(_ samples: [Float]) throws {
    try Task.checkCancellation()
    let ready = try stretcher.stretch(samples)
    if !ready.isEmpty { emit(Speech(samples: ready, sampleRate: sampleRate)) }
  }

  /// Emits the rest of the audio, which may be empty; call it once, after the last piece.
  public func finish() throws {
    emit(Speech(samples: try stretcher.finish(), sampleRate: sampleRate))
  }
}
