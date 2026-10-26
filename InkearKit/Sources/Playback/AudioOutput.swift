import AVFoundation
import Foundation
import Synchronization
import Synthesis

/// Plays mono speech buffers back to back.
@MainActor public final class AudioOutput {
  public enum Event: Sendable, Equatable {
    case interruptionBegan
    case interruptionEnded(shouldResume: Bool)
    case outputLost  // route change with reason .oldDeviceUnavailable (headphones unplugged)
    case mediaServicesReset
  }

  private let engine = AVAudioEngine()
  private let player = AVAudioPlayerNode()
  private var format: AVAudioFormat?
  private let generation = Generation()
  /// The player's timeline when it last stopped running.
  private var stoppedSeconds = 0.0
  private let eventStream: AsyncStream<Event>
  private let continuation: AsyncStream<Event>.Continuation
  #if os(iOS)
    private let observers: [Task<Void, Never>]
  #endif

  public var events: AsyncStream<Event> { eventStream }

  public var isRunning: Bool { player.isPlaying }

  /// Seconds on the player's timeline since the last reset. It stands still while paused and runs
  /// on while playing with nothing scheduled.
  public var playedSeconds: Double {
    guard let now = player.lastRenderTime, let time = player.playerTime(forNodeTime: now) else {
      return stoppedSeconds
    }
    return Double(time.sampleTime) / time.sampleRate
  }

  public init() {
    let (stream, streamContinuation) = AsyncStream.makeStream(of: Event.self)
    eventStream = stream
    continuation = streamContinuation
    engine.attach(player)
    #if os(iOS)
      observers = Self.observe(continuation)
    #endif
  }

  isolated deinit {
    continuation.finish()
    #if os(iOS)
      for observer in observers { observer.cancel() }
    #endif
  }

  /// iOS: AVAudioSession category .playback, mode .spokenAudio, policy .longFormAudio; activates it.
  /// macOS: does nothing.
  public func activate() throws {
    #if os(iOS)
      let session = AVAudioSession.sharedInstance()
      try session.setCategory(.playback, mode: .spokenAudio, policy: .longFormAudio)
      try session.setActive(true)
    #endif
  }

  /// Starts the engine if it is not running, then the player node.
  public func play() throws {
    // The engine cannot start without a path to the output, so before the first buffer the
    // player is connected at the rate Supertonic speaks; `schedule` reconnects for other rates.
    if format == nil {
      _ = try connectedFormat(sampleRate: 44_100)
    }
    if !engine.isRunning {
      try engine.start()
    }
    try player.playAudio()
  }

  public func pause() {
    stoppedSeconds = playedSeconds
    player.pause()
  }

  /// Drops every scheduled buffer. Completion handlers of dropped buffers are never delivered.
  public func reset() {
    generation.advance()
    player.stop()
    stoppedSeconds = 0
  }

  /// Schedules `speech` after the buffers already scheduled; `onPlayed` runs on the main actor
  /// once its last sample has been played back (`.dataPlayedBack`).
  ///
  /// Empty speech is not played; `onPlayed` still runs. A change of sample rate resets the player,
  /// dropping the buffers scheduled before it, and keeps it playing if it was.
  public func schedule(_ speech: Speech, onPlayed: @escaping @MainActor () -> Void) throws {
    let format = try speech.samples.isEmpty ? nil : connectedFormat(sampleRate: speech.sampleRate)
    let playback = generation
    let expected = playback.current
    let finish: @Sendable () -> Void = {
      Task { @MainActor in
        if playback.current == expected {
          onPlayed()
        }
      }
    }
    guard let format, let buffer = makeBuffer(speech, format: format) else {
      finish()
      return
    }
    player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in finish() }
  }

  /// Returns the player's format for `sampleRate`, reconnecting the player when the rate changes.
  private func connectedFormat(sampleRate: Double) throws -> AVAudioFormat? {
    if let format, format.sampleRate == sampleRate {
      return format
    }
    guard
      let candidate = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false
      )
    else {
      return nil
    }
    let wasPlaying = player.isPlaying
    if format != nil {
      reset()
      engine.disconnectNodeOutput(player)
      format = nil
    }
    try engine.connectNode(player, to: engine.mainMixerNode, format: candidate)
    format = candidate
    if wasPlaying { try player.playAudio() }
    return candidate
  }

  private func makeBuffer(_ speech: Speech, format: AVAudioFormat) -> AVAudioPCMBuffer? {
    let count = AVAudioFrameCount(speech.samples.count)
    guard
      let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: count),
      let channel = buffer.floatChannelData?[0]
    else {
      return nil
    }
    for (index, sample) in speech.samples.enumerated() {
      channel[index] = sample
    }
    buffer.frameLength = count
    return buffer
  }
}

/// Playback generation shared with completion handlers, which run on the audio thread.
private final class Generation: Sendable {
  private let value = Mutex(0)

  var current: Int { value.withLock { $0 } }

  func advance() {
    value.withLock { $0 += 1 }
  }
}

#if os(iOS)
  extension AudioOutput {
    /// One task per notification, each mapping its notifications to events off the main actor.
    nonisolated private static func observe(_ continuation: AsyncStream<Event>.Continuation)
      -> [Task<Void, Never>]
    {
      let names = [
        AVAudioSession.didBecomeInactiveNotification,
        AVAudioSession.resumptionRecommendationNotification,
        AVAudioSession.routeChangeNotification,
        AVAudioSession.mediaServicesWereResetNotification,
      ]
      return names.map { name in
        Task {
          for await notification in NotificationCenter.default.notifications(named: name) {
            if let event = event(from: notification) {
              continuation.yield(event)
            }
          }
        }
      }
    }

    nonisolated private static func event(from notification: Notification) -> Event? {
      switch notification.name {
      case AVAudioSession.didBecomeInactiveNotification:
        let context =
          notification.userInfo?[AVAudioSession.deactivationContextKey]
          as? AVAudioSession.DeactivationContext
        return context?.source == .system ? .interruptionBegan : nil
      case AVAudioSession.resumptionRecommendationNotification:
        guard
          let context = notification.userInfo?[AVAudioSession.resumptionContextKey]
            as? AVAudioSession.ResumptionContext
        else {
          return nil
        }
        return .interruptionEnded(shouldResume: context.recommendation == .shouldResume)
      case AVAudioSession.routeChangeNotification:
        let reason = (notification.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt)
          .flatMap(AVAudioSession.RouteChangeReason.init(rawValue:))
        return reason == .oldDeviceUnavailable ? .outputLost : nil
      case AVAudioSession.mediaServicesWereResetNotification:
        return .mediaServicesReset
      default:
        return nil
      }
    }
  }
#endif
