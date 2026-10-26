import Library
import PDFStructure
import SwiftUI

/// Shown while a document is analyzed: the steps, the share of pages read or recognized, and
/// cancel.
/// A failed document shows its reason instead, and can be removed.
struct PreparingView: View {
  let model: AppModel
  let id: DocumentID

  @Environment(\.theme) private var theme
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    if let record = model.record(id) {
      let reason = failure(of: record)
      VStack(alignment: .leading, spacing: 0) {
        Button {
          dismiss()
        } label: {
          Image(systemName: "chevron.left")
            .font(.system(size: 20, weight: .medium))
            .frame(width: 48, height: 48)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(theme.ink)
        .accessibilityLabel("Library")
        .padding(.leading, 8)
        .padding(.top, 8)

        VStack(alignment: .leading, spacing: 0) {
          MonoLabel(reason == nil ? "Preparing" : "Failed")
          Text(verbatim: record.title)
            .font(Typeface.text(36))
            .tracking(-0.54)
            .padding(.top, 10)
          if let author = record.author {
            Text(verbatim: author)
              .font(Typeface.ui(16))
              .foregroundStyle(theme.ink2)
              .padding(.top, 8)
          }

          if let reason {
            Text(verbatim: reason)
              .font(Typeface.ui(15))
              .foregroundStyle(theme.ink2)
              .padding(.top, 56)
          } else {
            progressBlock
            stepsBlock(pageCount: record.pageCount)
          }
        }
        .padding(.horizontal, 24)
        .padding(.top, 12)

        Spacer(minLength: 24)

        VStack(spacing: 16) {
          if reason == nil {
            Text(
              "Nothing leaves your device. You can go back to the library while the analysis continues."
            )
            .font(Typeface.ui(13.5))
            .lineSpacing(2)
            .foregroundStyle(theme.ink2)
            .frame(maxWidth: .infinity, alignment: .leading)
            outlineButton("Cancel import") {
              model.cancelPreparation(id)
              dismiss()
            }
          } else {
            outlineButton("Remove") {
              model.remove(id)
              dismiss()
            }
          }
        }
        .padding(.horizontal, 24)
        .padding(.bottom, 40)
      }
    }
  }

  /// The recognition step is `nil`, and hidden, once no page turned out to need it.
  private var stepStates: (open: StepState, recognize: StepState?, structure: StepState) {
    switch model.progress[id] {
    case nil, .reading?: (.active, .pending, .pending)
    case .recognizing?: (.done, .active, .pending)
    case .structuring(let recognizedPages)?: (.done, recognizedPages > 0 ? .done : nil, .active)
    }
  }

  /// Fraction of the pages the current step has done; nil when the step has no page count.
  private var fraction: Double? {
    guard let pages = model.progress[id]?.pages, pages.of > 0 else { return nil }
    return Double(pages.done) / Double(pages.of)
  }

  private var progressBlock: some View {
    VStack(alignment: .leading, spacing: 10) {
      if let fraction {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
          Text(verbatim: "\(Int((fraction * 100).rounded()))")
            .font(Typeface.mono(80))
            .tracking(-4.8)
          Text(verbatim: "%")
            .font(Typeface.mono(22))
            .foregroundStyle(theme.ink2)
        }
      }
      ProgressLine(value: fraction, tint: theme.preparing)
        .padding(.top, fraction == nil ? 0 : 8)
      if let pages = model.progress[id]?.pages {
        Text("page \(pages.done) of \(pages.of)")
          .font(Typeface.mono(12))
          .foregroundStyle(theme.ink2)
      }
    }
    .padding(.top, 56)
  }

  private func stepsBlock(pageCount: Int) -> some View {
    let states = stepStates
    return VStack(alignment: .leading, spacing: 22) {
      StepRow(state: states.open, title: "Open the PDF", detail: "\(pageCount) pages")
      if let recognize = states.recognize {
        StepRow(
          state: recognize, title: "Recognize the text", detail: "On this device, page by page")
      }
      StepRow(
        state: states.structure, title: "Find the structure",
        detail: "Chapters, contents and where the text starts")
    }
    .padding(.top, 44)
  }

  private func outlineButton(_ title: LocalizedStringKey, action: @escaping () -> Void) -> some View
  {
    Button(action: action) {
      Text(title)
        .font(Typeface.ui(15, .medium))
        .frame(maxWidth: .infinity)
        .frame(height: 50)
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(theme.chipOutline, lineWidth: 1))
    }
    .buttonStyle(.plain)
    .foregroundStyle(theme.ink)
  }

  private func failure(of record: DocumentRecord) -> String? {
    switch record.status {
    case .failed(let reason): reason
    case .preparing, .ready: nil
    }
  }
}

private enum StepState {
  case done, active, pending
}

private struct StepRow: View {
  let state: StepState
  let title: LocalizedStringKey
  let detail: LocalizedStringKey

  @Environment(\.theme) private var theme

  var body: some View {
    HStack(alignment: .top, spacing: 14) {
      indicator
        .frame(width: 22, height: 22)
      VStack(alignment: .leading, spacing: 3) {
        Text(title)
          .font(Typeface.ui(15.5, .medium))
          .foregroundStyle(state == .pending ? theme.ink2 : theme.ink)
        Text(detail)
          .font(Typeface.ui(13))
          .foregroundStyle(theme.ink2)
      }
    }
  }

  @ViewBuilder
  private var indicator: some View {
    switch state {
    case .done:
      ZStack {
        Circle().fill(theme.action)
        Image(systemName: "checkmark")
          .font(.system(size: 10, weight: .bold))
          .foregroundStyle(theme.actionInk)
      }
    case .active:
      ZStack {
        Circle().stroke(theme.line, lineWidth: 2)
        Circle()
          .trim(from: 0, to: 0.25)
          .stroke(theme.preparing, style: StrokeStyle(lineWidth: 2, lineCap: .round))
          .rotationEffect(.degrees(-90))
      }
    case .pending:
      Circle().stroke(theme.thumbnailBar, lineWidth: 1.75)
    }
  }
}
