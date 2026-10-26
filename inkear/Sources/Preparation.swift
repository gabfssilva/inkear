import BackgroundTasks
import Foundation
import Library
import PDFStructure
import os

let preparationLog = Logger(subsystem: "inkear", category: "preparation")

extension AppModel {
  /// Starts the analysis of `id` unless it is already running. The analysis continues as a
  /// background task when the system accepts the request, and runs in the foreground otherwise.
  func startPreparation(_ id: DocumentID, title: String) {
    guard preparations[id] == nil else { return }
    Task {
      if await continueInBackground(id, title: title) {
        preparationLog.info("\(id, privacy: .public): background task requested")
      } else if preparations[id] == nil {
        begin(id, reporting: nil)
      }
    }
  }

  /// Background task identifier of the analysis of `id`; matches `dev.gabfssilva.inkear.prepare.*`
  /// in `BGTaskSchedulerPermittedIdentifiers`.
  nonisolated static func preparationIdentifier(for id: DocumentID) -> String {
    "dev.gabfssilva.inkear.prepare.\(id)"
  }

  /// Share of a background task's progress for an analysis step, out of 100 units: reading
  /// fills the first half, recognition most of the second, and the structure the rest.
  nonisolated static func backgroundUnits(for step: AnalysisProgress) -> Int64 {
    switch step {
    case .reading(let page, let of):
      return of > 0 ? Int64(50 * page / of) : 0
    case .recognizing(let page, let of):
      return 50 + (of > 0 ? Int64(45 * page / of) : 0)
    case .structuring:
      return 95
    }
  }

  private func continueInBackground(_ id: DocumentID, title: String) async -> Bool {
    if !registeredPreparations.contains(id) {
      guard registerBackgroundHandler(for: id) else { return false }
      registeredPreparations.insert(id)
    }
    do {
      try await Self.submitBackgroundRequest(for: id, title: title)
      return true
    } catch {
      return false
    }
  }

  /// `submitTaskRequest(_:)` must not be called from the main thread.
  @concurrent nonisolated private static func submitBackgroundRequest(
    for id: DocumentID, title: String
  ) async throws {
    let request = BGContinuedProcessingTaskRequest(
      identifier: preparationIdentifier(for: id), title: title,
      subtitle: String(localized: "Preparing for listening"))
    request.strategy = .fail
    try await BGTaskScheduler.shared.submitTaskRequest(request)
  }

  /// Installs the launch handler of `id`. The system calls it off the main actor. `BGTask` is
  /// not `Sendable`, so it stays on the handler's queue, which waits for the analysis to end.
  nonisolated private func registerBackgroundHandler(for id: DocumentID) -> Bool {
    BGTaskScheduler.shared.register(
      forTaskWithIdentifier: Self.preparationIdentifier(for: id), using: nil
    ) { bgTask in
      guard let task = bgTask as? BGContinuedProcessingTask else {
        bgTask.setTaskCompleted(success: false)
        return
      }
      let progress = task.progress
      progress.totalUnitCount = 100
      task.expirationHandler = {
        Task { @MainActor in self.stopPreparation(id) }
      }
      let finished = DispatchSemaphore(value: 0)
      Task { @MainActor in
        await self.attachPreparation(id, reporting: progress)
        finished.signal()
      }
      finished.wait()
      task.setTaskCompleted(success: true)
    }
  }
}
