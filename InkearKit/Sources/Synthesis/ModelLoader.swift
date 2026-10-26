import CoreML
import Foundation

/// How far loading an engine has come, in Core ML functions loaded.
public struct LoadProgress: Sendable, Equatable {
  public let done: Int
  public let total: Int
  /// When the function after the `done` ones began loading.
  public let started: Date

  public init(done: Int, total: Int, started: Date) {
    self.done = done
    self.total = total
    self.started = started
  }

  public var fraction: Double { total > 0 ? Double(done) / Double(total) : 1 }

  /// The share of the load done at `now`, given the seconds each function took to compile the
  /// last time, in load order. The function loading advances with the time it has taken: up to
  /// 90% of its share at its time from last time, and only slowly towards 99% when it runs over.
  /// Without a time for every function, the share of the functions loaded.
  public func fraction(at now: Date, expecting times: [Double]) -> Double {
    guard isTimed(by: times), done < total else { return fraction }
    let expected = times[done]
    let elapsed = max(now.timeIntervalSince(started), 0)
    let share =
      elapsed < expected
      ? 0.9 * elapsed / expected : 0.9 + 0.09 * (1 - exp(-(elapsed - expected) / expected))
    return (times.prefix(done).reduce(0, +) + share * expected) / times.reduce(0, +)
  }

  /// Whether `times` has a time for every function.
  public func isTimed(by times: [Double]) -> Bool {
    times.count == total && times.allSatisfy { $0 > 0 }
  }

  /// `times` with the seconds the function loaded between `self` and `next` took. A time found
  /// in the system's cache, far shorter than a compilation, does not replace one.
  public func times(_ times: [Double], until next: LoadProgress) -> [Double] {
    guard next.done == done + 1, next.total == total, done < total else { return times }
    var updated = times.count == total ? times : [Double](repeating: 0, count: total)
    let seconds = next.started.timeIntervalSince(started)
    if updated[done] == 0 || seconds > updated[done] / 4 {
      updated[done] = seconds
    }
    return updated
  }
}

/// Loads the functions of an engine's directory one after another, reporting each one done.
///
/// The first load of a function after the app is installed or updated compiles it for the
/// Neural Engine, which can take minutes; later loads read the system's cache and take a moment.
public struct ModelLoader {
  private let directory: URL
  private let computeUnits: MLComputeUnits
  private let progress: (LoadProgress) -> Void
  private var current: LoadProgress

  /// Reports that none of the `total` functions is loaded yet.
  public init(
    directory: URL, total: Int, computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
    progress: @escaping (LoadProgress) -> Void
  ) {
    self.directory = directory
    self.computeUnits = computeUnits
    self.progress = progress
    current = LoadProgress(done: 0, total: total, started: .now)
    progress(current)
  }

  /// Loads `function` of `package`.mlmodelc in the directory, or its default function.
  public mutating func load(_ package: String, function: String? = nil) throws -> MLModel {
    try load(contentsOf: directory.appending(path: "\(package).mlmodelc"), function: function)
  }

  /// Loads `function` of the compiled model at `url`, or its default function.
  public mutating func load(contentsOf url: URL, function: String? = nil) throws -> MLModel {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    if let function { configuration.functionName = function }
    let model = try MLModel(contentsOf: url, configuration: configuration)
    advance()
    return model
  }

  /// Runs `load`, a function loaded by other means than Core ML, and reports it done.
  public mutating func load<Model>(_ load: () async throws -> Model) async rethrows -> Model {
    let model = try await load()
    advance()
    return model
  }

  private mutating func advance() {
    current = LoadProgress(done: current.done + 1, total: current.total, started: .now)
    progress(current)
  }
}

