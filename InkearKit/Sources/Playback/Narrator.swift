import Foundation
import Observation
import PDFStructure
import Synthesis
import os

public enum SleepTimer: Sendable, Equatable, Hashable {
  case off
  case endOfSection
  case minutes(Int)
}

/// Reads a script aloud: synthesizes sentences ahead of playback, follows the sentence being
/// heard, and answers transport commands. UI-facing state, so it lives on the main actor.
@MainActor @Observable public final class Narrator {
  public enum State: Sendable, Equatable { case paused, loading, playing, finished }

  public let script: Script
  public let title: String
  public private(set) var state: State = .paused
  /// The sentence being heard, or the next one to be heard.
  public private(set) var position: ReadingPosition
  public private(set) var voice: VoiceID
  public private(set) var speed: Double
  public private(set) var language: String
  public private(set) var quality: SpeechQuality
  public var sleepTimer: SleepTimer = .off { didSet { armSleepTimer() } }
  /// Called on every position change and on finishing.
  public var onProgress: (@MainActor (ReadingPosition, _ finished: Bool) -> Void)?
  public private(set) var lastError: String?

  private let artwork: Data?
  /// Nil between ``releaseEngine()`` and the next ``set(engine:voice:)``.
  private var engine: (any SpeechEngine)?
  private let output: AudioOutput
  private let nowPlaying: NowPlaying?
  /// Audio scheduled on `output`, in playback order; the head is being heard.
  @ObservationIgnored private var queue: [Scheduled] = []
  /// Numbers the audio scheduled on `output`.
  @ObservationIgnored private var scheduledCount = 0
  /// The synthesis whose speech the loop is reading, until it ends.
  @ObservationIgnored private var synthesis: SpeechStream?
  /// Where the audio of the sentence at `position` started on the output's timeline; nil until it
  /// starts.
  @ObservationIgnored private var heardFrom: Double?
  /// The sentence of the audio scheduled last, and the seconds of silence that end that audio.
  @ObservationIgnored private var tail: (sentence: ReadingPosition, silence: Double)?
  /// The next sentence to synthesize, or nil once the text is exhausted.
  @ObservationIgnored private var cursor: ReadingPosition?
  /// Bumped whenever scheduled audio is discarded; results from older generations are ignored.
  @ObservationIgnored private var generation = 0
  /// Seconds of synthesis per second of audio, measured on the last synthesis.
  @ObservationIgnored private var realTimeFactor: Double?
  @ObservationIgnored private var loopTask: Task<Void, Never>?
  @ObservationIgnored private var sleepTask: Task<Void, Never>?
  @ObservationIgnored private var eventsTask: Task<Void, Never>?
  @ObservationIgnored private var commandsTask: Task<Void, Never>?

  /// Without `engine`, playing waits for ``set(engine:voice:)``.
  public init(
    script: Script, title: String, artwork: Data?, engine: (any SpeechEngine)?,
    output: AudioOutput, nowPlaying: NowPlaying?,
    voice: VoiceID, speed: Double, language: String, quality: SpeechQuality,
    position: ReadingPosition
  ) {
    self.script = script
    self.title = title
    self.artwork = artwork
    self.engine = engine
    self.output = output
    self.nowPlaying = nowPlaying
    self.voice = voice
    self.speed = speed
    self.language = language
    self.quality = quality
    self.position = script.clamped(position)
    self.cursor = self.position

    let events = output.events
    eventsTask = Task { [weak self] in
      for await event in events {
        guard let self else { return }
        self.handle(event)
      }
    }
    if let nowPlaying {
      let commands = nowPlaying.commands
      commandsTask = Task { [weak self] in
        for await command in commands {
          guard let self else { return }
          self.handle(command)
        }
      }
    }
  }

  deinit {
    eventsTask?.cancel()
    commandsTask?.cancel()
  }

  public func play() {
    guard !isPlayingIntent else { return }
    lastError = nil
    if state == .finished { restart(at: script.start) }
    do {
      try output.activate()
      try output.play()
    } catch {
      report(error)
      return
    }
    state = queue.isEmpty ? .loading : .playing
    refill()
    updateNowPlaying()
  }

  public func pause() {
    guard isPlayingIntent else { return }
    output.pause()
    state = .paused
    updateNowPlaying()
  }

  public func toggle() {
    if isPlayingIntent {
      pause()
    } else {
      play()
    }
  }

  /// Moves to `position`, keeping playback going if it was playing.
  public func jump(to position: ReadingPosition) {
    restart(at: script.clamped(position))
  }

  public func nextSection() {
    if let target = script.nextSection(after: position) { jump(to: target) }
  }

  public func previousSection() {
    if let target = script.previousSection(before: position) { jump(to: target) }
  }

  public func nextSentence() {
    if let target = script.next(after: position) { jump(to: target) }
  }

  public func previousSentence() {
    if let target = script.previous(before: position) { jump(to: target) }
  }

  /// Reads on with `voice` of `engine`, from the current position.
  public func set(engine: any SpeechEngine, voice: VoiceID) {
    guard voice != self.voice || self.engine == nil else { return }
    self.engine = engine
    self.voice = voice
    restart(at: position)
  }

  /// Stops synthesizing and lets go of the engine once the synthesis in flight ends, so that the
  /// next engine never shares the memory with this one. Reading goes on after
  /// ``set(engine:voice:)``.
  public func releaseEngine() async {
    let running = loopTask
    let pending = synthesis
    discardPending()
    engine = nil
    if isPlayingIntent { state = .loading }
    await running?.value
    await pending?.cancel()
  }

  public func set(speed: Double) {
    guard speed != self.speed else { return }
    self.speed = speed
    restart(at: position)
  }

  public func set(language: String) {
    guard language != self.language else { return }
    self.language = language
    restart(at: position)
  }

  public func set(quality: SpeechQuality) {
    guard quality != self.quality else { return }
    self.quality = quality
    restart(at: position)
  }

  public func stop() {
    pause()
    discardPending()
    cursor = position
    nowPlaying?.clear()
  }

  /// Estimated seconds into the current section and the section's total, at the current speed.
  /// Within the sentence being heard, it counts the seconds its audio has played, up to the
  /// sentence's estimate, so that it never runs back.
  public var sectionTime: (elapsed: TimeInterval, total: TimeInterval) {
    let characters = script.sectionCharacters(at: position)
    let rate = Script.charactersPerSecond * speed
    let sentence = Double(script.text(at: position)?.count ?? 0) / rate
    let heard = heardFrom.map { min(max(0, output.playedSeconds - $0), sentence) } ?? 0
    return (Double(characters.before) / rate + heard, Double(characters.total) / rate)
  }

  // MARK: - Synthesis loop

  /// The most characters of a paragraph synthesized in one call, about 20 s of speech.
  nonisolated private static let maxRunCharacters = 300

  /// The characters of the next run whose synthesis, at `realTimeFactor`, ends before three
  /// quarters of the `queued` seconds have played, up to ``maxRunCharacters``. With nothing queued
  /// or nothing measured, it is 0, which reads one sentence.
  /// Where one sentence ends and the next begins.
  enum Boundary: Sendable { case sentence, paragraph, section }

  /// Seconds of silence after `text`, at natural pace, before a sentence across `boundary`. A new
  /// section or paragraph gets the longest pauses; otherwise the punctuation ending `text`
  /// decides.
  nonisolated static func pause(after text: String, before boundary: Boundary) -> Double {
    switch boundary {
    case .section: return 1.2
    case .paragraph: return 0.8
    case .sentence: break
    }
    let ending = text.trimmingCharacters(
      in: .whitespacesAndNewlines.union(CharacterSet(charactersIn: "\"'”’»)]")))
    if ending.hasSuffix("…") || ending.hasSuffix("...") { return 0.6 }
    switch ending.last {
    case ".", "!", "?": return 0.45
    case ":", ";": return 0.35
    default: return 0.25
    }
  }

  nonisolated static func runCharacters(queued: Double, realTimeFactor: Double?, speed: Double)
    -> Int
  {
    guard let realTimeFactor, realTimeFactor > 0 else { return 0 }
    let seconds = queued * 0.75 / realTimeFactor
    return min(maxRunCharacters, Int(seconds * Script.charactersPerSecond * speed))
  }

  private var isPlayingIntent: Bool { state == .playing || state == .loading }

  private var queuedSeconds: Double { queue.reduce(0) { $0 + $1.seconds } }

  private var queuedSentences: Int {
    zip(queue, queue.dropFirst()).count { $0.position != $1.position } + (queue.isEmpty ? 0 : 1)
  }

  /// Starts the synthesis loop unless one is running.
  private func refill() {
    guard loopTask == nil else { return }
    let current = generation
    loopTask = Task { [weak self] in
      await self?.fill(current)
    }
  }

  private func fill(_ current: Int) async {
    defer {
      if current == generation { loopTask = nil }
    }
    while current == generation, queuedSentences < 3 || queuedSeconds < 20, let start = cursor,
      let engine
    {
      // Sentences of a paragraph are read together, which gives the model their context, as many
      // as can be synthesized before the queued audio ends.
      let run = script.run(
        from: start,
        characters: Self.runCharacters(
          queued: queuedSeconds, realTimeFactor: realTimeFactor, speed: speed))
      let texts = run.compactMap { script.text(at: $0) }
      guard let last = run.last, texts.count == run.count else {
        cursor = script.next(after: start)
        continue
      }
      let text = texts.joined(separator: " ")
      let interval = synthesisSignposter.beginInterval("synthesize")
      defer { synthesisSignposter.endInterval("synthesize", interval) }
      let started = Date.now
      let stream = engine.synthesize(
        text, voice: voice.name, language: language, speed: speed, quality: quality)
      synthesis = stream
      defer { if current == generation { synthesis = nil } }
      let seconds: Double
      do {
        seconds = try await read(stream, of: run, texts: texts, generation: current)
      } catch {
        guard current == generation else { return }
        report(error)
        return
      }
      guard current == generation else { return }
      cursor = script.next(after: last)
      guard seconds > 0 else { continue }

      let wall = Date.now.timeIntervalSince(started)
      let factor = wall / seconds
      realTimeFactor = factor
      synthesisLog.info(
        "\(run.count, privacy: .public) sentences, \(text.count, privacy: .public) characters, \(seconds, privacy: .public) s audio, \(wall, privacy: .public) s wall, RTF \(factor, privacy: .public)"
      )
    }
    if queue.isEmpty, cursor == nil, current == generation {
      finish()
    }
  }

  /// Schedules the speech of the sentences of `run`, returning its seconds. A single sentence
  /// plays part by part as the parts arrive, so its start is heard while the rest is synthesized.
  /// Sentences read together are cut apart once their whole speech is known; by then the audio
  /// queued ahead still plays.
  private func read(
    _ stream: SpeechStream, of run: [ReadingPosition], texts: [String], generation current: Int
  ) async throws -> Double {
    guard run.count > 1 else {
      var seconds = 0.0
      for try await part in stream where !part.samples.isEmpty {
        guard current == generation else { break }
        try schedule(part, of: run[0], pausing: true, generation: current)
        seconds += part.duration
      }
      return seconds
    }
    let speech = try await Speech(collecting: stream)
    guard current == generation, !speech.samples.isEmpty else { return 0 }
    let parts = speech.split(proportionalTo: texts.map { Double($0.count) })
    // An empty part would report itself played before the parts ahead of it. The cuts are
    // estimated from the audio and some fall between words, so the parts play as the model
    // spoke them, with its own pauses: only the run starts after a pause of the narrator.
    var pausing = true
    for (sentence, part) in zip(run, parts) where !part.samples.isEmpty {
      try schedule(part, of: sentence, pausing: pausing, generation: current)
      pausing = false
    }
    return speech.duration
  }

  /// Schedules `part` of `sentence`. When `pausing`, the first part of a sentence starts after the
  /// pause from the sentence before it, counting the silence that ends the audio queued ahead;
  /// with nothing queued, it starts at once.
  private func schedule(
    _ part: Speech, of sentence: ReadingPosition, pausing: Bool, generation current: Int
  ) throws {
    var speech = part
    if pausing, tail?.sentence != sentence {
      var pause = 0.0
      if let tail, !queue.isEmpty, let text = script.text(at: tail.sentence) {
        pause = Self.pause(after: text, before: boundary(from: tail.sentence, to: sentence)) / speed
          - tail.silence
      }
      speech = speech.padded(by: pause - speech.leadingSilence)
    }
    let silence = speech.trailingSilence
    let wholeSilence = silence >= speech.duration && tail?.sentence == sentence
    tail = (sentence, wholeSilence ? (tail?.silence ?? 0) + silence : silence)
    speech = speech.overRoomTone()

    scheduledCount += 1
    let number = scheduledCount
    try output.schedule(speech) { [weak self] in
      self?.played(number, generation: current)
    }
    if queue.isEmpty {
      move(to: sentence)
      heardFrom = heardFrom ?? output.playedSeconds
    }
    queue.append(Scheduled(number: number, position: sentence, seconds: speech.duration))
    if state == .loading { state = .playing }
  }

  private func boundary(from previous: ReadingPosition, to next: ReadingPosition) -> Boundary {
    if script.section(containing: previous) != script.section(containing: next) { return .section }
    return previous.block == next.block ? .sentence : .paragraph
  }

  /// Called when the output has played the audio numbered `number` to its end.
  private func played(_ number: Int, generation current: Int) {
    guard current == generation, let head = queue.first, head.number == number else { return }
    let sentence = head.position
    queue.removeFirst()
    if let head = queue.first {
      move(to: head.position)
      heardFrom = heardFrom ?? output.playedSeconds
    } else if let next = cursor {
      if state == .playing { state = .loading }
      move(to: next)
    } else {
      finish()
      return
    }
    if sleepTimer == .endOfSection,
      script.section(containing: sentence) != script.section(containing: position)
    {
      sleepTimer = .off
      pause()
    }
    refill()
    updateNowPlaying()
  }

  private func finish() {
    guard state != .finished else { return }
    output.pause()
    state = .finished
    updateNowPlaying()
    onProgress?(position, true)
  }

  private func move(to target: ReadingPosition) {
    guard target != position else { return }
    position = target
    heardFrom = nil
    onProgress?(target, false)
  }

  /// Drops scheduled audio and synthesis in flight, and invalidates their results.
  private func discardPending() {
    generation += 1
    loopTask?.cancel()
    loopTask = nil
    output.reset()
    queue.removeAll()
    heardFrom = nil
    tail = nil
  }

  private func restart(at target: ReadingPosition) {
    let wasPlaying = isPlayingIntent
    discardPending()
    cursor = target
    move(to: target)
    state = wasPlaying ? .loading : .paused
    if wasPlaying {
      do {
        try output.play()
      } catch {
        report(error)
      }
    }
    refill()
    updateNowPlaying()
  }

  private func report(_ error: any Error) {
    lastError = error.localizedDescription
    pause()
  }

  // MARK: - Sleep timer

  private func armSleepTimer() {
    sleepTask?.cancel()
    sleepTask = nil
    guard case .minutes(let minutes) = sleepTimer else { return }
    sleepTask = Task { [weak self] in
      do {
        try await Task.sleep(for: .seconds(minutes * 60))
      } catch {
        return
      }
      self?.pause()
      self?.sleepTimer = .off
    }
  }

  // MARK: - Now Playing and system events

  private var sectionTitle: String? {
    guard let index = script.section(containing: position) else { return nil }
    return script.structure.sections.indices.contains(index)
      ? script.structure.sections[index].title : nil
  }

  private func updateNowPlaying() {
    let time = sectionTime
    nowPlaying?.update(
      title: title, section: sectionTitle, artwork: artwork,
      elapsed: time.elapsed, duration: time.total, isPlaying: state == .playing)
  }

  private func handle(_ event: AudioOutput.Event) {
    switch event {
    case .interruptionBegan, .outputLost:
      pause()
    case .interruptionEnded(let shouldResume):
      if shouldResume { play() }
    case .mediaServicesReset:
      jump(to: position)
    }
  }

  private func handle(_ command: NowPlaying.Command) {
    switch command {
    case .play: play()
    case .pause: pause()
    case .toggle: toggle()
    case .nextSection: nextSection()
    case .previousSection: previousSection()
    }
  }
}

/// Audio of a sentence scheduled on the output: the whole sentence, or a part of it.
private struct Scheduled: Sendable {
  let number: Int
  let position: ReadingPosition
  let seconds: Double
}

private let synthesisLog = Logger(subsystem: "inkear", category: "synthesis")
private let synthesisSignposter = OSSignposter(subsystem: "inkear", category: "synthesis")
