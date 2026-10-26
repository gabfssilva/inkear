import Foundation
import Testing

@testable import Synthesis

@Suite struct LoadProgressTests {
  private let start = Date(timeIntervalSinceReferenceDate: 0)

  @Test func theFunctionLoadingAdvancesWithTheTimeItTookLastTime() {
    let times = [200.0, 4, 6]
    let first = LoadProgress(done: 0, total: 3, started: start)
    let second = LoadProgress(done: 1, total: 3, started: start + 210)

    #expect(first.fraction(at: start, expecting: times) == 0)
    #expect(abs(first.fraction(at: start + 100, expecting: times) - 0.45 * 200 / 210) < 1e-9)
    #expect(abs(first.fraction(at: start + 200, expecting: times) - 0.9 * 200 / 210) < 1e-9)
    #expect(first.fraction(at: start + 10_000, expecting: times) < 200.0 / 210)
    #expect(abs(second.fraction(at: start + 210, expecting: times) - 200.0 / 210) < 1e-9)
  }

  @Test func withoutTimesTheShareIsTheFunctionsLoaded() {
    let progress = LoadProgress(done: 1, total: 2, started: start)

    #expect(progress.fraction(at: start + 60, expecting: []) == 0.5)
    #expect(progress.fraction(at: start + 60, expecting: [100, 0]) == 0.5)
  }

  @Test func compilationTimesAreLearnedAndOutlastCachedLoads() {
    let compiled = [
      LoadProgress(done: 0, total: 2, started: start),
      LoadProgress(done: 1, total: 2, started: start + 220),
      LoadProgress(done: 2, total: 2, started: start + 223),
    ]
    let learned = zip(compiled, compiled.dropFirst()).reduce([Double]()) { $1.0.times($0, until: $1.1) }
    let cached = [
      LoadProgress(done: 0, total: 2, started: start),
      LoadProgress(done: 1, total: 2, started: start + 1.5),
      LoadProgress(done: 2, total: 2, started: start + 1.6),
    ]
    let kept = zip(cached, cached.dropFirst()).reduce(learned) { $1.0.times($0, until: $1.1) }

    #expect(learned == [220, 3])
    #expect(kept == [220, 3])
  }
}
