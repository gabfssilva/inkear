import Synthesis
import SwiftUI

/// The load of a speech engine: a title, a bar, and why it can take minutes. With the compile
/// `times` of every function from last time, the bar advances with the time the load has taken
/// and the percentage shows; without them, it fills by functions loaded and the share of the one
/// loading sweeps. Without progress, the whole bar sweeps.
struct EngineLoadingView: View {
  let title: Text
  let progress: LoadProgress?
  let times: [Double]

  @Environment(\.theme) private var theme

  var body: some View {
    TimelineView(.animation(minimumInterval: 0.25)) { context in
      let fraction = progress?.timedFraction(at: context.date, expecting: times)
      VStack(alignment: .leading, spacing: 6) {
        HStack(alignment: .firstTextBaseline) {
          title
          Spacer(minLength: 8)
          if let fraction {
            Text(verbatim: percent(fraction))
              .font(Typeface.mono(11))
          }
        }
        if let fraction {
          ProgressLine(value: fraction, tint: theme.action)
        } else {
          ProgressLine(
            value: progress?.fraction, tint: theme.action,
            sweepingTo: progress.map { Double($0.done + 1) / Double(max($0.total, 1)) })
        }
        OptimizingNote()
          .font(Typeface.ui(12))
      }
    }
    .font(Typeface.ui(13))
    .foregroundStyle(theme.ink2)
  }
}

/// The play button while the narrator waits for its engine: a ring filled as the bar of
/// ``EngineLoadingView`` around the percentage, or a turning arc without the times of last load.
struct LoadingRing: View {
  let progress: LoadProgress?
  let times: [Double]
  let size: CGFloat

  @Environment(\.theme) private var theme

  var body: some View {
    let timed = progress?.isTimed(by: times) ?? false
    TimelineView(.animation(minimumInterval: timed ? 0.25 : nil)) { context in
      let fraction = progress?.timedFraction(at: context.date, expecting: times)
      let turn = context.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 1.4)
      ZStack {
        Circle()
          .stroke(theme.line, lineWidth: 3)
        Circle()
          .trim(from: 0, to: fraction ?? 0.25)
          .stroke(theme.action, style: StrokeStyle(lineWidth: 3, lineCap: .round))
          .rotationEffect(.degrees(fraction == nil ? turn / 1.4 * 360 - 90 : -90))
        Circle()
          .fill(theme.control)
          .padding(8)
        if let fraction {
          Text(verbatim: percent(fraction))
            .font(Typeface.mono(15, .medium))
            .foregroundStyle(theme.ink)
        }
      }
      .frame(width: size, height: size)
      .accessibilityElement()
      .accessibilityLabel("Preparing the voice")
      .accessibilityValue(fraction.map(percent) ?? "")
    }
  }
}

/// Why loading a voice can take minutes.
struct OptimizingNote: View {
  var body: some View {
    Label("The voice model is being optimized to run on your device.", systemImage: "info.circle")
  }
}

extension LoadProgress {
  /// The share loaded by the time the load has taken, when the times of last load tell it.
  fileprivate func timedFraction(at now: Date, expecting times: [Double]) -> Double? {
    isTimed(by: times) ? fraction(at: now, expecting: times) : nil
  }
}

/// Rounded down, so that it reads 100% only once loaded.
private func percent(_ fraction: Double) -> String {
  "\(Int(fraction * 100))%"
}
