import Synthesis
import SwiftUI

/// The speech models: the one included with the app and the ones to download.
struct ModelsView: View {
  let model: AppModel

  @Environment(\.theme) private var theme
  @Environment(\.dismiss) private var dismiss
  @State private var deleting: EngineID?

  var body: some View {
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
      .accessibilityLabel("Back")
      .padding(.leading, 8)

      Text("Models")
        .font(Typeface.ui(34, .semibold))
        .tracking(-1.2)
        .foregroundStyle(theme.ink)
        .padding(.horizontal, 20)
        .padding(.top, 4)
      Text("Downloaded models appear under Voice while listening.")
        .font(Typeface.ui(15))
        .foregroundStyle(theme.ink2)
        .padding(.horizontal, 20)
        .padding(.top, 6)

      ScrollView {
        VStack(spacing: 0) {
          ForEach(model.engineIDs, id: \.self) { id in
            row(id)
              .overlay(alignment: .bottom) {
                Rectangle().fill(theme.fineLine).frame(height: 1)
              }
          }
        }
        .padding(.horizontal, 20)
        .padding(.top, 12)
      }
    }
    .foregroundStyle(theme.ink)
    .background(theme.ground)
    .confirmationDialog(
      Text("Delete \(deleting.map(model.displayName(of:)) ?? "")?"),
      isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } }),
      titleVisibility: .visible, presenting: deleting
    ) { id in
      Button("Delete", role: .destructive) { model.delete(id) }
    } message: { _ in
      Text("Download it again to read with it.")
    }
  }

  private func row(_ id: EngineID) -> some View {
    HStack(spacing: 12) {
      VStack(alignment: .leading, spacing: 4) {
        Text(verbatim: model.displayName(of: id))
          .font(Typeface.text(19, .medium))
        detail(id)
          .font(Typeface.ui(13))
          .foregroundStyle(theme.ink2)
      }
      .frame(maxWidth: .infinity, alignment: .leading)
      action(id)
    }
    .padding(.vertical, 14)
  }

  @ViewBuilder
  private func detail(_ id: EngineID) -> some View {
    switch model.engineStates[id] {
    case .downloading(let fraction)?:
      HStack(spacing: 10) {
        ProgressLine(value: fraction, tint: theme.action)
        if let fraction {
          Text(fraction, format: .percent.precision(.fractionLength(0)))
            .font(Typeface.mono(11))
        }
      }
    case .preparing(let progress)?:
      EngineLoadingView(
        title: Text("Preparing"), progress: progress, times: model.loadTimes[id] ?? [])
    case .failed(let message)? where !model.installedEngines.contains(id):
      Text(verbatim: message)
        .lineLimit(2)
    default:
      if id == .supertonic {
        Text("Included with the app")
      } else if model.installedEngines.contains(id) {
        Text("Downloaded · \(size(of: id))")
          .accessibilityIdentifier("downloaded-\(id.rawValue)")
      } else {
        Text(verbatim: size(of: id))
      }
    }
  }

  @ViewBuilder
  private func action(_ id: EngineID) -> some View {
    if id == .supertonic {
      EmptyView()
    } else if model.installedEngines.contains(id) {
      Button {
        deleting = id
      } label: {
        Image(systemName: "trash")
          .font(.system(size: 16, weight: .medium))
          .foregroundStyle(theme.ink2)
          .frame(width: 44, height: 44)
          .contentShape(Rectangle())
      }
      .buttonStyle(.plain)
      .accessibilityLabel("Delete \(model.displayName(of: id))")
      .accessibilityIdentifier("delete-\(id.rawValue)")
    } else {
      switch model.engineStates[id] {
      case .downloading?, .preparing?:
        EmptyView()
      case .failed?:
        downloadButton("Retry", id)
      default:
        downloadButton("Download", id)
      }
    }
  }

  private func downloadButton(_ title: LocalizedStringKey, _ id: EngineID) -> some View {
    Button(title) {
      model.install(id)
    }
    .font(Typeface.ui(14, .semibold))
    .padding(.horizontal, 14)
    .frame(height: 34)
    .background(theme.control, in: Capsule())
    .buttonStyle(.plain)
    .accessibilityIdentifier("download-\(id.rawValue)")
  }

  private func size(of id: EngineID) -> String {
    ByteCountFormatter.string(fromByteCount: model.downloadSize(of: id) ?? 0, countStyle: .file)
  }
}
