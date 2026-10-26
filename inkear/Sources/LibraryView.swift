import Library
import PDFStructure
import Playback
import SwiftUI

struct LibraryView: View {
  @Bindable var model: AppModel
  @Environment(\.theme) private var theme
  @State private var isSearching = false
  @FocusState private var searchFocused: Bool

  var body: some View {
    VStack(spacing: 0) {
      header
      if isSearching {
        searchField
      }
      content
    }
    .background(theme.ground)
    .safeAreaInset(edge: .bottom) {
      if model.narrator != nil {
        MiniPlayer(model: model)
      }
    }
  }

  private var header: some View {
    HStack(alignment: .bottom) {
      VStack(alignment: .leading, spacing: 8) {
        MonoLabel("\(model.records.count) PDFs")
        Text("Library")
          .font(Typeface.ui(34, .semibold))
          .tracking(-1.2)
          .foregroundStyle(theme.ink)
      }
      Spacer(minLength: 12)
      HStack(spacing: 6) {
        RoundButton(systemImage: "waveform", action: model.showModels)
          .accessibilityLabel("Models")
        RoundButton(systemImage: "magnifyingglass", action: toggleSearch)
        RoundButton(systemImage: "plus", filled: true) {
          model.isImporting = true
        }
      }
    }
    .padding(.horizontal, 20)
    .padding(.top, 8)
    .padding(.bottom, 4)
  }

  private var searchField: some View {
    HStack(spacing: 8) {
      Image(systemName: "magnifyingglass")
        .foregroundStyle(theme.ink2)
      TextField("Search", text: $model.searchText)
        .font(Typeface.ui(16))
        .foregroundStyle(theme.ink)
        .focused($searchFocused)
    }
    .padding(.horizontal, 12)
    .frame(height: 40)
    .background(theme.control, in: Capsule())
    .padding(.horizontal, 20)
    .padding(.top, 8)
    .onAppear { searchFocused = true }
  }

  @ViewBuilder private var content: some View {
    if model.records.isEmpty {
      emptyState
    } else if model.filteredRecords.isEmpty {
      Text("No matches")
        .font(Typeface.ui(15))
        .foregroundStyle(theme.ink2)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    } else {
      list
    }
  }

  private var list: some View {
    List {
      ForEach(model.filteredRecords) { record in
        Button {
          model.open(record.id)
        } label: {
          DocumentRow(
            record: record,
            cover: model.cover(for: record.id),
            progress: model.progress[record.id],
            isListening: model.listening == record.id,
            fraction: fraction(of: record))
        }
        .buttonStyle(.plain)
        .listRowBackground(theme.ground)
        .listRowInsets(EdgeInsets(top: 0, leading: 20, bottom: 0, trailing: 20))
        .listRowSeparatorTint(theme.fineLine)
        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
          Button(role: .destructive) {
            model.remove(record.id)
          } label: {
            Label("Delete", systemImage: "trash")
          }
        }
      }
    }
    .listStyle(.plain)
    .scrollContentBackground(.hidden)
  }

  private var emptyState: some View {
    VStack(spacing: 14) {
      Spacer()
      Text("No PDFs yet")
        .font(Typeface.text(22, .medium))
        .foregroundStyle(theme.ink)
      Text("Import a PDF to listen to it.")
        .font(Typeface.ui(15))
        .foregroundStyle(theme.ink2)
      Button("Import PDF") {
        model.isImporting = true
      }
      .buttonStyle(PrimaryButtonStyle())
      .padding(.horizontal, 20)
      .padding(.top, 6)
      Spacer()
    }
    .multilineTextAlignment(.center)
  }

  private func toggleSearch() {
    isSearching.toggle()
    if !isSearching {
      model.searchText = ""
    }
  }

  /// Listening position while the narrator plays the document, the saved bookmark otherwise.
  private func fraction(of record: DocumentRecord) -> Double? {
    if model.listening == record.id, let narrator = model.narrator {
      return narrator.script.fraction(of: narrator.position)
    }
    return record.bookmark?.fraction
  }
}

private struct DocumentRow: View {
  @Environment(\.theme) private var theme
  let record: DocumentRecord
  let cover: Image?
  let progress: AnalysisProgress?
  let isListening: Bool
  let fraction: Double?

  var body: some View {
    HStack(spacing: 14) {
      CoverView(image: cover, title: record.title)
        .frame(width: 36)
      VStack(alignment: .leading, spacing: 0) {
        Text(verbatim: record.title)
          .font(Typeface.text(19, .medium))
          .foregroundStyle(theme.ink)
          .lineLimit(1)
        Text(verbatim: subtitle)
          .font(Typeface.ui(13))
          .foregroundStyle(theme.ink2)
          .lineLimit(1)
          .padding(.top, 1)
        HStack(spacing: 10) {
          badge
          meter
        }
        .padding(.top, 9)
      }
      .frame(maxWidth: .infinity, alignment: .leading)
    }
    .padding(.vertical, 13)
    .contentShape(Rectangle())
  }

  private var subtitle: String {
    let detail: String
    if case .ready(let overview) = record.status, overview.sectionCount > 0 {
      detail = String(localized: "\(overview.sectionCount) chapters")
    } else {
      detail = String(localized: "\(record.pageCount) pages")
    }
    return [record.author, detail].compactMap { $0 }.joined(separator: " · ")
  }

  @ViewBuilder private var badge: some View {
    switch record.status {
    case .preparing:
      if case .recognizing? = progress {
        Chip("Recognizing", style: .preparing)
      } else {
        Chip("Preparing", style: .preparing)
      }
    case .failed:
      Chip("Failed", style: .outlined)
    case .ready:
      if isListening {
        Chip("Listening", style: .voice)
      } else if record.isFinished {
        Chip("Done", style: .outlined)
      } else if let bookmark = record.bookmark {
        Chip("Ch. \((bookmark.section ?? 0) + 1)", style: .outlined)
      } else {
        Chip("New", style: .outlined)
      }
    }
  }

  @ViewBuilder private var meter: some View {
    switch record.status {
    case .preparing:
      ProgressLine(value: preparedFraction, tint: theme.preparing)
        .frame(maxWidth: .infinity)
      if let value = preparedFraction {
        Text(value, format: .percent.precision(.fractionLength(0)))
          .font(Typeface.mono(11))
          .foregroundStyle(theme.ink2)
      }
    case .failed:
      EmptyView()
    case .ready:
      if let fraction, !record.isFinished {
        ProgressLine(value: fraction, tint: theme.action)
          .frame(maxWidth: .infinity)
        Text(fraction, format: .percent.precision(.fractionLength(0)))
          .font(Typeface.mono(11))
          .foregroundStyle(theme.ink2)
      }
    }
  }

  /// Share of the pages read or recognized so far; nil while structuring (indeterminate bar).
  private var preparedFraction: Double? {
    guard let pages = progress?.pages, pages.of > 0 else { return nil }
    return Double(pages.done) / Double(pages.of)
  }
}

struct MiniPlayer: View {
  @Environment(\.theme) private var theme
  let model: AppModel

  var body: some View {
    if let narrator = model.narrator {
      HStack(spacing: 2) {
        Button {
          model.showPlayer()
        } label: {
          HStack(spacing: 12) {
            CoverView(
              image: model.listening.flatMap { model.cover(for: $0) }, title: narrator.title
            )
            .frame(width: 30)
            VStack(alignment: .leading, spacing: 1) {
              Text(verbatim: narrator.title)
                .font(Typeface.ui(14.5, .medium))
                .foregroundStyle(theme.ink)
                .lineLimit(1)
              if let section = sectionTitle(of: narrator) {
                Text(verbatim: section)
                  .font(Typeface.ui(12.5))
                  .foregroundStyle(theme.ink2)
                  .lineLimit(1)
              }
            }
            Spacer(minLength: 0)
          }
          .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        RoundButton(systemImage: narrator.state == .playing ? "pause.fill" : "play.fill") {
          narrator.toggle()
        }
      }
      .padding(.leading, 10)
      .padding(.trailing, 6)
      .frame(height: 64)
      .clipShape(RoundedRectangle(cornerRadius: 18))
      .overlay(alignment: .bottom) {
        ProgressLine(value: narrator.script.fraction(of: narrator.position), tint: theme.action)
      }
      .background {
        RoundedRectangle(cornerRadius: 18)
          .fill(theme.surface)
          .shadow(color: .black.opacity(0.1), radius: 16, y: 12)
      }
      .overlay {
        RoundedRectangle(cornerRadius: 18).strokeBorder(theme.cardRing, lineWidth: 1)
      }
      .padding(.horizontal, 12)
      .padding(.bottom, 8)
    }
  }

  private func sectionTitle(of narrator: Narrator) -> String? {
    narrator.script.section(containing: narrator.position).map {
      narrator.script.structure.sections[$0].title
    }
  }
}
