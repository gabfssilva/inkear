import Engines
import Foundation
import Library
import PDFKit
import PDFStructure
import Playback
import SwiftUI
import Synthesis
import UIKit

/// The library, the listening session and the background preparations of the app.
@MainActor @Observable final class AppModel {
  enum Route: Hashable {
    case preparing(DocumentID)
    case ready(DocumentID)
    case player
    case models
  }

  /// `preparing` is the first load after a download, which compiles the models for the Neural
  /// Engine; later loads reuse that compilation until the app is updated. Both report the model
  /// functions loaded when the engine knows them.
  enum EngineState: Equatable {
    case notInstalled, downloading(Double?), preparing(LoadProgress?), installed
    case loading(LoadProgress?), ready, failed(String)

    var isLoading: Bool {
      if case .loading = self { true } else { false }
    }
  }

  static let speeds: [Double] = [0.8, 1, 1.25, 1.5, 2]
  static let languages: [String] = ["pt", "en"]

  var path: [Route] = []
  var isImporting = false
  var searchText = ""
  private(set) var records: [DocumentRecord] = []
  private(set) var progress: [DocumentID: AnalysisProgress] = [:]
  private(set) var narrator: Narrator?
  private(set) var listening: DocumentID?
  private(set) var engineStates: [EngineID: EngineState] = [:]
  /// The engines on the device: Supertonic and the downloaded ones.
  private(set) var installedEngines: Set<EngineID> = [.supertonic]
  private(set) var voices: [EngineID: [String]] = [:]
  private(set) var alert: String?
  /// The engine the narrator waits for: the book's while it opens, or the one chosen for it.
  private(set) var awaitedEngine: EngineID?
  /// Seconds each model function of an engine took to compile the last time, in load order, by
  /// which the loading bar advances while a function compiles.
  private(set) var loadTimes: [EngineID: [Double]] = AppModel.savedLoadTimes()

  private let library: Library
  private let output = AudioOutput()
  private let nowPlaying = NowPlaying()
  @ObservationIgnored private var loadedEngines: [EngineID: any SpeechEngine] = [:]
  @ObservationIgnored private var engineTasks: [EngineID: Task<any SpeechEngine, any Error>] = [:]
  /// Downloads and deletions in flight, at most one per engine.
  @ObservationIgnored private var fileTasks: [EngineID: Task<Void, Never>] = [:]
  @ObservationIgnored private var scripts: [DocumentID: Script] = [:]
  private var covers: [DocumentID: UIImage] = [:]
  @ObservationIgnored private var coverTasks: [DocumentID: Task<Void, Never>] = [:]
  @ObservationIgnored private var lastWrite: Task<Void, Never>?
  @ObservationIgnored var preparations: [DocumentID: Task<Void, Never>] = [:]
  @ObservationIgnored var registeredPreparations: Set<DocumentID> = []

  init() {
    CompiledModelCache.clearIfReinstalled()
    let directory = URL.applicationSupportDirectory.appending(path: "Library")
    do {
      library = try Library(directory: directory)
    } catch {
      fatalError("Cannot open the library at \(directory.path()): \(error)")
    }
    engineStates[.supertonic] = .installed
    for installable in installableEngines {
      let installed = installable.isInstalled(in: Self.directory(of: installable.id))
      engineStates[installable.id] = installed ? .installed : .notInstalled
      if installed { installedEngines.insert(installable.id) }
    }
    Task {
      var resumed = false
      for await updated in await self.library.updates() {
        self.records = updated
        // Analyses interrupted by the app quitting start over.
        if !resumed {
          resumed = true
          for record in updated where record.status == .preparing {
            self.startPreparation(record.id, title: record.title)
          }
        }
      }
    }
  }

  func record(_ id: DocumentID) -> DocumentRecord? {
    records.first { $0.id == id }
  }

  var engineIDs: [EngineID] { [.supertonic] + installableEngines.map { $0.id } }

  func displayName(of id: EngineID) -> String {
    id == .supertonic ? "Supertonic" : installableType(id)?.displayName ?? id.rawValue
  }

  func downloadSize(of id: EngineID) -> Int64? {
    installableType(id)?.downloadSize
  }

  func usesQuality(_ id: EngineID) -> Bool {
    id == .supertonic ? Supertonic.usesQuality : installableType(id)?.usesQuality ?? false
  }

  /// The loaded engine `id`, loading it first if needed. It must be installed.
  func engine(_ id: EngineID) async throws -> any SpeechEngine {
    if let loaded = loadedEngines[id] { return loaded }
    let task = engineTasks[id] ?? startEngine(id)
    return try await task.value
  }

  /// Downloads `id`, then loads it once so that choosing it later does not wait for the first
  /// compilation.
  func install(_ id: EngineID) {
    guard let installable = installableType(id), fileTasks[id] == nil else { return }
    fileTasks[id] = Task {
      defer { self.fileTasks[id] = nil }
      let directory = Self.directory(of: id)
      do {
        self.engineStates[id] = .downloading(nil)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try await installable.install(into: directory) { fraction in
          Task { @MainActor in
            if case .downloading = self.engineStates[id] {
              self.engineStates[id] = .downloading(fraction)
            }
          }
        }
        self.engineStates[id] = .preparing(nil)
        _ = try await installable.load(from: directory) { progress in
          Task { @MainActor in
            if case .preparing(let previous)? = self.engineStates[id] {
              self.learnLoadTimes(of: id, from: previous, to: progress)
              self.engineStates[id] = .preparing(progress)
            }
          }
        }
        self.installedEngines.insert(id)
        self.engineStates[id] = .installed
      } catch {
        self.engineStates[id] = .failed(error.localizedDescription)
      }
    }
  }

  /// Removes the files of `id`. When the book being read uses it, the book reads on with
  /// Supertonic first.
  func delete(_ id: EngineID) {
    guard let installable = installableType(id), fileTasks[id] == nil, engineTasks[id] == nil
    else { return }
    fileTasks[id] = Task {
      defer { self.fileTasks[id] = nil }
      if self.narrator?.voice.engine == id {
        guard await self.switchEngine(to: .supertonic) else { return }
      }
      self.loadedEngines[id] = nil
      self.installedEngines.remove(id)
      self.engineStates[id] = .notInstalled
      let directory = Self.directory(of: id)
      do {
        try await Task.detached { try installable.uninstall(from: directory) }.value
      } catch {
        self.alert = error.localizedDescription
      }
    }
  }

  func showModels() {
    path.append(.models)
  }

  func selectEngine(_ id: EngineID) {
    Task { await self.switchEngine(to: id) }
  }

  /// Reads on with the default voice of `id` after the narrator lets go of its engine:
  /// two engines in memory at once can pass the app's memory limit. False when `id` fails to load.
  @discardableResult
  private func switchEngine(to id: EngineID) async -> Bool {
    if let narrator, narrator.voice.engine != id { await narrator.releaseEngine() }
    unloadEngines(except: id)
    do {
      _ = try await awaitEngine(id)
    } catch {
      alert = error.localizedDescription
      return false
    }
    setVoice(VoiceID(engine: id, name: defaultVoice(of: id)))
    return true
  }

  var filteredRecords: [DocumentRecord] {
    let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !query.isEmpty else { return records }
    return records.filter { document in
      document.title.localizedStandardContains(query)
        || (document.author?.localizedStandardContains(query) ?? false)
    }
  }

  func importDocuments(_ result: Result<[URL], any Error>) {
    switch result {
    case .success(let urls):
      for url in urls { add(url, opening: urls.count == 1) }
    case .failure(let error):
      alert = error.localizedDescription
    }
  }

  func importDocument(at url: URL) {
    add(url, opening: true)
  }

  func open(_ id: DocumentID) {
    guard let document = record(id) else { return }
    switch document.status {
    case .preparing:
      path.append(.preparing(id))
      startPreparation(id, title: document.title)
    case .failed:
      path.append(.preparing(id))
    case .ready:
      if document.bookmark == nil {
        path.append(.ready(id))
      } else {
        Task { await self.listen(id, from: nil) }
      }
    }
  }

  func cancelPreparation(_ id: DocumentID) {
    remove(id)
  }

  /// The script of a ready document, built once and cached.
  func script(for id: DocumentID) async -> Script? {
    if let cached = scripts[id] { return cached }
    let structure: DocumentStructure
    do {
      structure = try await library.structure(of: id)
    } catch {
      alert = error.localizedDescription
      return nil
    }
    let built = await Task.detached { Script(structure) }.value
    scripts[id] = built
    return built
  }

  func listen(_ id: DocumentID, from position: ReadingPosition?) async {
    guard let document = record(id), case .ready = document.status else { return }
    guard let loaded = await script(for: id) else { return }
    var settings = document.settings
    // The book's model was deleted, or replaced by a new revision not downloaded yet.
    if !installedEngines.contains(settings.voice.engine) {
      settings = settings.with(
        voice: VoiceID(engine: .supertonic, name: defaultVoice(of: .supertonic)))
      let changed = settings
      persist { library in try await library.save(changed, for: id) }
    }
    if let narrator, narrator.voice.engine != settings.voice.engine {
      narrator.stop()
      await narrator.releaseEngine()
    }
    unloadEngines(except: settings.voice.engine)
    let artwork = await thumbnail(of: id)?.jpegData(compressionQuality: 0.8)
    let fresh = Narrator(
      script: loaded, title: document.title, artwork: artwork,
      engine: loadedEngines[settings.voice.engine], output: output, nowPlaying: nowPlaying,
      voice: settings.voice, speed: settings.speed,
      language: effectiveLanguage(of: document), quality: settings.quality,
      position: loaded.clamped(position ?? document.bookmark?.position ?? loaded.start))
    fresh.onProgress = { [weak self] reached, finished in
      self?.saveProgress(reached, finished: finished, of: id, in: loaded)
    }
    narrator?.stop()
    narrator = fresh
    listening = id
    path.removeAll { $0 == .player }
    path.append(.player)
    fresh.play()
    do {
      let speaker = try await awaitEngine(settings.voice.engine)
      if narrator === fresh { fresh.set(engine: speaker, voice: settings.voice) }
    } catch {
      alert = error.localizedDescription
      if narrator === fresh { stopListening() }
    }
  }

  func showPlayer() {
    guard narrator != nil else { return }
    path.removeAll { $0 == .player }
    path.append(.player)
  }

  func cover(for id: DocumentID) -> Image? {
    if let image = covers[id] { return Image(uiImage: image) }
    if coverTasks[id] == nil {
      coverTasks[id] = Task { _ = await self.thumbnail(of: id) }
    }
    return nil
  }

  func setVoice(_ voice: VoiceID) {
    if let speaker = loadedEngines[voice.engine] {
      narrator?.set(engine: speaker, voice: voice)
      unloadEngines(except: voice.engine)
    }
    updateSettings { $0.with(voice: voice) }
  }

  /// The voices of `id` that speak `language`, when its engine knows them; all of them otherwise.
  func voices(of id: EngineID, language: String) -> [String] {
    if let loaded = loadedEngines[id] { return loaded.voices(for: language) }
    return voices[id] ?? []
  }

  func setSpeed(_ speed: Double) {
    narrator?.set(speed: speed)
    updateSettings { $0.with(speed: speed) }
  }

  func setQuality(_ quality: SpeechQuality) {
    narrator?.set(quality: quality)
    updateSettings { $0.with(quality: quality) }
  }

  /// `nil` restores the detected language. When the narrator's voice does not speak the new
  /// language, it switches to the first voice of that language.
  func setLanguage(_ language: String?) {
    updateSettings { $0.with(language: language) }
    if let id = listening, let document = record(id) {
      let resolved = language ?? effectiveLanguage(of: document)
      narrator?.set(language: resolved)
      if let narrator, let speaker = loadedEngines[narrator.voice.engine],
        !speaker.voices(for: resolved).contains(narrator.voice.name),
        let first = speaker.voices(for: resolved).first
      {
        setVoice(VoiceID(engine: narrator.voice.engine, name: first))
      }
    }
  }

  func remove(_ id: DocumentID) {
    if listening == id { stopListening() }
    stopPreparation(id)
    preparations[id] = nil
    coverTasks[id]?.cancel()
    coverTasks[id] = nil
    covers[id] = nil
    scripts[id] = nil
    path.removeAll { $0 == .preparing(id) || $0 == .ready(id) }
    persist { library in try await library.remove(id) }
  }

  func dismissAlert() {
    alert = nil
  }

  // MARK: Engines

  private func installableType(_ id: EngineID) -> (any InstallableEngine.Type)? {
    installableEngines.first { $0.id == id }
  }

  private static let loadTimesKey = "engineLoadTimes"

  private static func savedLoadTimes() -> [EngineID: [Double]] {
    guard let data = UserDefaults.standard.data(forKey: loadTimesKey),
      let saved = try? JSONDecoder().decode([String: [Double]].self, from: data)
    else { return [:] }
    return Dictionary(
      uniqueKeysWithValues: saved.compactMap { key, times in EngineID(rawValue: key).map { ($0, times) } })
  }

  private func learnLoadTimes(of id: EngineID, from previous: LoadProgress?, to next: LoadProgress) {
    guard let previous else { return }
    let times = previous.times(loadTimes[id] ?? [], until: next)
    guard times != loadTimes[id] else { return }
    loadTimes[id] = times
    let saved = Dictionary(uniqueKeysWithValues: loadTimes.map { ($0.key.rawValue, $0.value) })
    if let data = try? JSONEncoder().encode(saved) {
      UserDefaults.standard.set(data, forKey: Self.loadTimesKey)
    }
  }

  private func defaultVoice(of id: EngineID) -> String {
    installableType(id)?.defaultVoice ?? "F3"
  }

  /// Only the engine in use stays in memory: the models take up to 1.7 GB each.
  private func unloadEngines(except kept: EngineID) {
    for id in loadedEngines.keys where id != kept {
      loadedEngines[id] = nil
      engineStates[id] = .installed
    }
  }

  private static func directory(of id: EngineID) -> URL {
    URL.applicationSupportDirectory.appending(path: "Models").appending(path: id.rawValue)
  }

  /// `id` loaded, shown by the player in place of the play button while it loads.
  private func awaitEngine(_ id: EngineID) async throws -> any SpeechEngine {
    if let loaded = loadedEngines[id] { return loaded }
    awaitedEngine = id
    defer { if awaitedEngine == id { awaitedEngine = nil } }
    return try await engine(id)
  }

  private func startEngine(_ id: EngineID) -> Task<any SpeechEngine, any Error> {
    let task = Task { try await self.loadEngine(id) }
    engineTasks[id] = task
    return task
  }

  /// Loads `id` and records the outcome; a failed load is retried by the next request.
  private func loadEngine(_ id: EngineID) async throws -> any SpeechEngine {
    defer { engineTasks[id] = nil }
    engineStates[id] = .loading(nil)
    do {
      let loaded = try await loadModels(of: id) { progress in
        Task { @MainActor in
          if case .loading(let previous)? = self.engineStates[id] {
            self.learnLoadTimes(of: id, from: previous, to: progress)
            self.engineStates[id] = .loading(progress)
          }
        }
      }
      loadedEngines[id] = loaded
      voices[id] = loaded.voices
      engineStates[id] = .ready
      return loaded
    } catch {
      engineStates[id] = .failed(error.localizedDescription)
      throw error
    }
  }

  private func loadModels(
    of id: EngineID, progress: @escaping @Sendable (LoadProgress) -> Void
  ) async throws -> any SpeechEngine {
    if id == .supertonic {
      guard let models = Bundle.main.url(forResource: "supertonic-3", withExtension: nil) else {
        throw URLError(.fileDoesNotExist)
      }
      return try await Supertonic.load(from: models, progress: progress)
    }
    guard let installable = installableType(id), installedEngines.contains(id) else {
      throw URLError(.fileDoesNotExist)
    }
    return try await installable.load(from: Self.directory(of: id), progress: progress)
  }

  // MARK: Preparation

  /// Runs the analysis of `id` on the main actor, reporting into `progress` and into the
  /// background task's progress when one is given.
  func begin(_ id: DocumentID, reporting background: Progress?) {
    // Started from the background task's launch handler, the analysis would inherit its
    // background priority and run on the efficiency cores only, many times slower.
    preparations[id] = Task(priority: .userInitiated) {
      await self.run(id, reporting: background)
      self.preparations[id] = nil
    }
  }

  /// Waits for the analysis of `id`, starting it first when the system launched the background
  /// task before the analysis was started.
  func attachPreparation(_ id: DocumentID, reporting background: Progress) async {
    if preparations[id] == nil { begin(id, reporting: background) }
    await preparations[id]?.value
  }

  func stopPreparation(_ id: DocumentID) {
    preparations[id]?.cancel()
  }

  private func run(_ id: DocumentID, reporting background: Progress?) async {
    let started = ContinuousClock.now
    preparationLog.info(
      "\(id, privacy: .public): analysis started, in a background task: \(background != nil, privacy: .public)"
    )
    defer {
      preparationLog.info(
        "\(id, privacy: .public): analysis ended after \(started.duration(to: .now), privacy: .public)"
      )
    }
    progress[id] = .reading(page: 0, of: record(id)?.pageCount ?? 0)
    let report: @Sendable (AnalysisProgress) -> Void = { step in
      background?.completedUnitCount = AppModel.backgroundUnits(for: step)
      Task { @MainActor in self.progress[id] = step }
    }
    do {
      let document = try await prepare(id, in: library, progress: report)
      if case .ready = document.status, path.last == .preparing(id) {
        path[path.count - 1] = .ready(id)
      }
    } catch {
      // The record keeps the failure reason; a cancelled run leaves the record as it was.
    }
    progress[id] = nil
  }

  // MARK: Import

  private func add(_ url: URL, opening: Bool) {
    Task {
      let added: DocumentRecord
      do {
        added = try await self.addToLibrary(url)
      } catch {
        self.alert = error.localizedDescription
        return
      }
      if opening { self.path.append(.preparing(added.id)) }
      self.startPreparation(added.id, title: added.title)
    }
  }

  private func addToLibrary(_ url: URL) async throws -> DocumentRecord {
    let scoped = url.startAccessingSecurityScopedResource()
    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
    return try await library.add(pdfAt: url, at: .now)
  }

  // MARK: Listening

  private func saveProgress(
    _ reached: ReadingPosition, finished: Bool, of id: DocumentID, in script: Script
  ) {
    let bookmark = Bookmark(
      position: reached, section: script.section(containing: reached),
      fraction: script.fraction(of: reached), date: .now)
    persist { library in
      try await library.save(bookmark, for: id)
      if finished { try await library.finish(id) }
    }
  }

  private func updateSettings(_ change: (ListeningSettings) -> ListeningSettings) {
    guard let id = listening, let document = record(id) else { return }
    let settings = change(document.settings)
    persist { library in try await library.save(settings, for: id) }
  }

  private func stopListening() {
    narrator?.stop()
    narrator = nil
    listening = nil
    path.removeAll { $0 == .player }
  }

  private func effectiveLanguage(of document: DocumentRecord) -> String {
    if let language = document.settings.language { return language }
    if case .ready(let overview) = document.status,
      let detected = overview.language, Self.languages.contains(detected)
    {
      return detected
    }
    return "pt"
  }

  /// Runs library writes one after another, so the last write requested is the last one stored.
  private func persist(_ write: @escaping @Sendable (Library) async throws -> Void) {
    let previous = lastWrite
    let library = self.library
    lastWrite = Task {
      await previous?.value
      do {
        try await write(library)
      } catch {
        self.alert = error.localizedDescription
      }
    }
  }

  // MARK: Covers

  private func thumbnail(of id: DocumentID) async -> UIImage? {
    if let image = covers[id] { return image }
    let url = library.pdfURL(of: id)
    let rendered = await Task.detached { AppModel.renderCover(at: url) }.value
    guard let image = rendered.flatMap(UIImage.init(data:)) else { return nil }
    covers[id] = image
    return image
  }

  /// The first page as JPEG data, rendered off the main actor.
  nonisolated private static func renderCover(at url: URL) -> Data? {
    guard let document = PDFDocument(url: url), let page = document.page(at: 0) else {
      return nil
    }
    return page.thumbnail(of: CGSize(width: 240, height: 360), for: .mediaBox)
      .jpegData(compressionQuality: 0.8)
  }
}
