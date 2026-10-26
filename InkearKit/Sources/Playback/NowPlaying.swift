import MediaPlayer

#if canImport(UIKit)
  import UIKit
#elseif canImport(AppKit)
  import AppKit
#endif

/// Lock screen / Control Center metadata and remote commands.
@MainActor
public final class NowPlaying {
  public enum Command: Sendable, Equatable {
    case play, pause, toggle, nextSection, previousSection
  }

  private let commandStream: AsyncStream<Command>
  private let continuation: AsyncStream<Command>.Continuation
  private var targets: [(command: MPRemoteCommand, token: Any)] = []

  public init() {
    let (stream, continuation) = AsyncStream.makeStream(of: Command.self)
    commandStream = stream
    self.continuation = continuation
  }

  /// Remote commands from the lock screen and headphones; registers the handlers on first access.
  public var commands: AsyncStream<Command> {
    registerTargets()
    return commandStream
  }

  /// `data` is PNG or JPEG. `elapsed` and `duration` are seconds at the reading speed, so the
  /// system advances `elapsed` at rate 1 while playing. Registers the remote command handlers if
  /// `clear()` removed them.
  public func update(
    title: String, section: String?, artwork: Data?,
    elapsed: TimeInterval, duration: TimeInterval, isPlaying: Bool
  ) {
    registerTargets()
    var info: [String: Any] = [
      MPNowPlayingInfoPropertyMediaType: MPNowPlayingInfoMediaType.audio.rawValue,
      MPMediaItemPropertyTitle: section ?? title,
      MPMediaItemPropertyAlbumTitle: title,
      MPNowPlayingInfoPropertyElapsedPlaybackTime: elapsed,
      MPMediaItemPropertyPlaybackDuration: duration,
      MPNowPlayingInfoPropertyPlaybackRate: isPlaying ? 1 : 0,
    ]
    if let artwork {
      info[MPMediaItemPropertyArtwork] = Self.artwork(from: artwork)
    }
    let center = MPNowPlayingInfoCenter.default()
    center.nowPlayingInfo = info
    #if os(macOS)
      center.playbackState = isPlaying ? .playing : .paused
    #endif
  }

  public func clear() {
    let center = MPNowPlayingInfoCenter.default()
    center.nowPlayingInfo = nil
    #if os(macOS)
      center.playbackState = .stopped
    #endif
    for target in targets {
      target.command.removeTarget(target.token)
    }
    targets.removeAll()
  }

  private func registerTargets() {
    guard targets.isEmpty else { return }
    let center = MPRemoteCommandCenter.shared()
    center.skipForwardCommand.isEnabled = false
    center.skipBackwardCommand.isEnabled = false
    center.changePlaybackPositionCommand.isEnabled = false
    register(center.playCommand, .play)
    register(center.pauseCommand, .pause)
    register(center.togglePlayPauseCommand, .toggle)
    register(center.nextTrackCommand, .nextSection)
    register(center.previousTrackCommand, .previousSection)
  }

  private func register(_ command: MPRemoteCommand, _ value: Command) {
    // The handler runs on the main thread but is not main-actor isolated, so it captures only the continuation.
    let stream = continuation
    command.isEnabled = true
    let token = command.addTarget { @Sendable _ in
      stream.yield(value)
      return .success
    }
    targets.append((command: command, token: token))
  }

  /// The image is decoded inside the handler because the system calls it off the main thread.
  private nonisolated static func artwork(from data: Data) -> MPMediaItemArtwork {
    MPMediaItemArtwork(boundsSize: CGSize(width: 512, height: 768)) { @Sendable _ in
      #if canImport(UIKit)
        UIImage(data: data) ?? UIImage()
      #elseif canImport(AppKit)
        NSImage(data: data) ?? NSImage()
      #endif
    }
  }
}
