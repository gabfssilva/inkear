import Library
import PDFStructure
import Playback
import SwiftUI

/// The document after analysis: the start of its text, the front matter it skips, and the ways to listen.
struct ReadyView: View {
  let model: AppModel
  let id: DocumentID

  @Environment(\.theme) private var theme
  @Environment(\.dismiss) private var dismiss
  @State private var script: Script?
  @State private var showsContents = false

  var body: some View {
    if let record = model.record(id), case .ready(let overview) = record.status {
      VStack(alignment: .leading, spacing: 0) {
        header(for: record)

        VStack(alignment: .leading, spacing: 0) {
          MonoLabel("Ready to listen")
          Text(verbatim: record.title)
            .font(Typeface.text(34))
            .tracking(-0.51)
            .padding(.top, 10)
          if let author = record.author {
            Text(verbatim: author)
              .font(Typeface.ui(16))
              .foregroundStyle(theme.ink2)
              .padding(.top, 8)
          }
          pills(for: overview)
          startCard(overview: overview)
            .padding(.top, 28)
          rows
            .padding(.top, 14)
        }
        .padding(.horizontal, 24)

        Spacer(minLength: 24)
      }
      .task { script = await model.script(for: id) }
      .sheet(isPresented: $showsContents) {
        if let script {
          ContentsSheet(
            script: script, current: model.record(id)?.bookmark?.position ?? script.start,
            onSelect: { position in Task { await model.listen(id, from: position) } }
          )
          .presentationDetents([.large])
        }
      }
    }
  }

  private func header(for record: DocumentRecord) -> some View {
    HStack {
      Button {
        dismiss()
      } label: {
        Image(systemName: "chevron.left")
          .font(.system(size: 20, weight: .medium))
          .frame(width: 48, height: 48)
          .contentShape(Rectangle())
      }
      .buttonStyle(.plain)
      .accessibilityLabel("Library")
      Spacer()
      Menu {
        Button("Remove", role: .destructive) {
          model.remove(id)
          dismiss()
        }
      } label: {
        Image(systemName: "ellipsis")
          .font(.system(size: 20, weight: .medium))
          .frame(width: 48, height: 48)
          .contentShape(Rectangle())
      }
      .accessibilityLabel("More options")
    }
    .foregroundStyle(theme.ink)
    .padding(.leading, 8)
    .padding(.trailing, 8)
    .padding(.top, 8)
  }

  private func pills(for overview: DocumentRecord.Overview) -> some View {
    HStack(spacing: 6) {
      if overview.sectionCount > 0 {
        Pill(text: Text("\(overview.sectionCount) chapters"))
      }
      if let language = overview.language.flatMap({
        Locale.current.localizedString(forLanguageCode: $0)
      }) {
        Pill(text: Text(verbatim: language))
      }
      if overview.isRecognized {
        Pill(text: Text("Text via OCR"))
      }
    }
    .padding(.top, 16)
  }

  private func startCard(overview: DocumentRecord.Overview) -> some View {
    VStack(alignment: .leading, spacing: 0) {
      Chip("Start of text", style: .voice)
      if let title = overview.mainStartTitle {
        Text(verbatim: title)
          .font(Typeface.text(26))
          .padding(.top, 16)
      }
      if let script, let excerpt = excerptText(of: script) {
        Text(verbatim: excerpt)
          .font(Typeface.text(16, italic: true))
          .lineSpacing(3)
          .foregroundStyle(theme.ink2)
          .padding(.top, 10)
      }
      if let script, let skipped = skipped(in: script) {
        HStack(spacing: 8) {
          Image(systemName: "forward.end")
            .font(.system(size: 14, weight: .regular))
          skipped
            .font(Typeface.ui(13))
            .lineLimit(2)
        }
        .foregroundStyle(theme.ink2)
        .padding(.top, 16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(alignment: .top) {
          Rectangle().fill(theme.cardRing).frame(height: 1)
        }
      }
      Button {
        Task { await model.listen(id, from: nil) }
      } label: {
        HStack(spacing: 10) {
          Image(systemName: "play.fill")
          Text("Listen")
        }
      }
      .buttonStyle(PrimaryButtonStyle())
      .padding(.top, 18)
    }
    .padding(20)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(theme.surface, in: RoundedRectangle(cornerRadius: 20))
    .overlay(RoundedRectangle(cornerRadius: 20).stroke(theme.cardRing, lineWidth: 1))
  }

  private var rows: some View {
    VStack(spacing: 0) {
      choice("From the start of the PDF", "Includes everything before the text") {
        Task { await model.listen(id, from: .start) }
      }
      Rectangle().fill(theme.fineLine).frame(height: 1)
      choice("Choose from contents", "Chapters and pages") {
        showsContents = true
      }
    }
  }

  private func choice(
    _ title: LocalizedStringKey, _ detail: LocalizedStringKey, action: @escaping () -> Void
  ) -> some View {
    Button(action: action) {
      HStack(spacing: 12) {
        VStack(alignment: .leading, spacing: 2) {
          Text(title)
            .font(Typeface.ui(15.5, .medium))
          Text(detail)
            .font(Typeface.ui(13))
            .foregroundStyle(theme.ink2)
        }
        Spacer(minLength: 0)
        Image(systemName: "chevron.right")
          .font(.system(size: 14, weight: .medium))
          .foregroundStyle(theme.ink2)
      }
      .frame(minHeight: 58)
      .contentShape(Rectangle())
    }
    .buttonStyle(.plain)
    .foregroundStyle(theme.ink)
  }

  /// The first two sentences of prose from the start of the text; the card shows the heading.
  private func excerptText(of script: Script) -> String? {
    let blocks = script.structure.blocks
    let prose = sequence(first: script.start) { script.next(after: $0) }
      .lazy.filter { blocks[$0.block].kind == .paragraph }
      .prefix(2)
    let text = prose.compactMap { script.text(at: $0) }.joined(separator: " ")
    return text.isEmpty ? nil : text
  }

  /// What starting at the main text skips: the titles of the sections before it, joined in the
  /// list format of the current locale, or their count when there are too many to read at a glance.
  private func skipped(in script: Script) -> Text? {
    guard let mainStart = script.structure.mainStart else { return nil }
    let titles = script.structure.sections.filter { $0.block < mainStart }.map(\.title)
    switch titles.count {
    case 0: return nil
    case 1...3: return Text("Skips \(ListFormatter.localizedString(byJoining: titles))")
    default: return Text("Skips \(titles.count) sections")
    }
  }
}

private struct Pill: View {
  let text: Text

  @Environment(\.theme) private var theme

  var body: some View {
    text
      .font(Typeface.mono(10.5))
      .tracking(0.84)
      .foregroundStyle(theme.ink2)
      .textCase(.uppercase)
      .padding(.horizontal, 9)
      .padding(.vertical, 4)
      .overlay(Capsule().stroke(theme.chipOutline, lineWidth: 1))
  }
}
