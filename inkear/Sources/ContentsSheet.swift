import PDFStructure
import Playback
import SwiftUI

/// The document's contents: chapters, a page grid, and a "Go to page" field.
///
/// Selecting a position calls `onSelect` and dismisses the sheet.
struct ContentsSheet: View {
  let script: Script
  let current: ReadingPosition?
  let onSelect: (ReadingPosition) -> Void

  @Environment(\.theme) private var theme
  @Environment(\.dismiss) private var dismiss
  @State private var tab = Tab.chapters
  @State private var pageInput = ""

  private enum Tab: Hashable { case chapters, pages }

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      header
      tabs
      switch tab {
      case .chapters: chapters
      case .pages: pages
      }
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    .presentationBackground(theme.ground)
  }

  private var header: some View {
    HStack {
      Text("Contents")
        .font(Typeface.ui(22, .semibold))
        .tracking(-0.55)
        .foregroundStyle(theme.ink)
      Spacer()
      RoundButton(systemImage: "xmark") { dismiss() }
    }
    .padding(.leading, 24)
    .padding(.trailing, 12)
    .padding(.top, 10)
  }

  private var tabs: some View {
    HStack(spacing: 3) {
      tabButton("Chapters", value: .chapters)
      tabButton("Pages", value: .pages)
    }
    .padding(3)
    .background(theme.control, in: RoundedRectangle(cornerRadius: 12))
    .padding(.horizontal, 24)
    .padding(.top, 18)
  }

  private func tabButton(_ title: LocalizedStringKey, value: Tab) -> some View {
    let selected = tab == value
    return Button {
      tab = value
    } label: {
      Text(title)
        .font(Typeface.ui(14, .medium))
        .frame(maxWidth: .infinity)
        .frame(height: 34)
        .background(selected ? theme.surface : Color.clear, in: RoundedRectangle(cornerRadius: 9))
        .foregroundStyle(selected ? theme.ink : theme.ink2)
    }
    .buttonStyle(.plain)
    .accessibilityAddTraits(selected ? AccessibilityTraits.isSelected : [])
  }

  // MARK: Chapters

  private var chapters: some View {
    ScrollViewReader { proxy in
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 0) {
          if hasFrontMatter {
            beforeText
          }
          ForEach(script.structure.sections.indices, id: \.self) { index in
            sectionRow(index)
              .id(index)
          }
        }
        .padding(.horizontal, 24)
        .padding(.top, 14)
        .padding(.bottom, 32)
      }
      .onAppear {
        if let index = currentSection {
          proxy.scrollTo(index, anchor: .center)
        }
      }
    }
  }

  private var hasFrontMatter: Bool {
    (script.structure.mainStart ?? 0) > 0
  }

  private var beforeText: some View {
    Button {
      select(script.clamped(ReadingPosition.start))
    } label: {
      HStack(spacing: 12) {
        Text("Before the text")
          .font(Typeface.ui(14.5))
          .frame(maxWidth: .infinity, alignment: .leading)
        Text("\(script.page(of: script.start)) pages")
          .font(Typeface.mono(12))
        Image(systemName: "chevron.down")
          .font(.system(size: 16, weight: .medium))
      }
      .foregroundStyle(theme.ink2)
      .frame(minHeight: 48)
      .overlay(alignment: .bottom) {
        Rectangle().fill(theme.fineLine).frame(height: 1)
      }
    }
    .buttonStyle(.plain)
  }

  private func sectionRow(_ index: Int) -> some View {
    let section = script.structure.sections[index]
    let position = script.start(ofSection: index)
    let isCurrent = index == currentSection
    let indent = CGFloat(max(section.level - 1, 0)) * 16
    return Button {
      select(position)
    } label: {
      HStack(spacing: 12) {
        marker(index, isCurrent: isCurrent)
        VStack(alignment: .leading, spacing: 2) {
          Text(verbatim: section.title)
            .font(Typeface.text(18))
            .lineLimit(1)
          if isCurrent {
            MonoLabel("Listening")
          }
        }
        .padding(.leading, indent)
        .frame(maxWidth: .infinity, alignment: .leading)
        Text(verbatim: "p. \(script.page(of: position) + 1)")
          .font(Typeface.mono(12))
          .foregroundStyle(theme.ink2)
      }
      .foregroundStyle(theme.ink)
      .frame(minHeight: 50)
      .padding(isCurrent ? 12 : 0)
      .background(isCurrent ? theme.surface : Color.clear, in: RoundedRectangle(cornerRadius: 14))
      .overlay(
        RoundedRectangle(cornerRadius: 14).stroke(
          isCurrent ? theme.cardRing : Color.clear, lineWidth: 1)
      )
      .overlay(alignment: .bottom) {
        if !isCurrent {
          Rectangle().fill(theme.fineLine).frame(height: 1)
        }
      }
      .padding(.vertical, isCurrent ? 6 : 0)
    }
    .buttonStyle(.plain)
  }

  @ViewBuilder
  private func marker(_ index: Int, isCurrent: Bool) -> some View {
    if isCurrent {
      Image(systemName: "waveform")
        .font(.system(size: 13, weight: .bold))
        .foregroundStyle(theme.voiceInk)
        .frame(width: 28, height: 28)
        .background(theme.voice, in: Circle())
    } else {
      Text(verbatim: "\(index + 1)")
        .font(Typeface.mono(12.5))
        .foregroundStyle(theme.ink2)
        .frame(width: 28)
    }
  }

  private var currentSection: Int? {
    current.flatMap { script.section(containing: $0) }
  }

  // MARK: Pages

  private var pages: some View {
    ScrollView {
      VStack(spacing: 18) {
        goToPage
        LazyVGrid(
          columns: Array(repeating: GridItem(.flexible(), spacing: 14), count: 5), spacing: 18
        ) {
          ForEach(0..<script.structure.pageCount, id: \.self) { index in
            pageCell(index)
          }
        }
      }
      .padding(.horizontal, 24)
      .padding(.top, 16)
      .padding(.bottom, 32)
    }
  }

  private var goToPage: some View {
    HStack {
      Text("Go to page")
        .font(Typeface.ui(14.5))
        .foregroundStyle(theme.ink2)
      Spacer()
      TextField("", text: $pageInput, prompt: Text(verbatim: currentPageNumber))
        .keyboardType(.numberPad)
        .submitLabel(.go)
        .onSubmit { goToTypedPage() }
        .font(Typeface.mono(15))
        .multilineTextAlignment(.center)
        .foregroundStyle(theme.ink)
        .frame(width: 64, height: 36)
        .background(theme.ground, in: RoundedRectangle(cornerRadius: 8))
    }
    .padding(.leading, 14)
    .padding(.trailing, 7)
    .frame(height: 50)
    .background(theme.surface, in: RoundedRectangle(cornerRadius: 12))
    .overlay(RoundedRectangle(cornerRadius: 12).stroke(theme.fineLine, lineWidth: 1))
  }

  private func pageCell(_ index: Int) -> some View {
    let isCurrent = index == currentPage
    return Button {
      if let position = script.position(onPage: index) {
        select(position)
      }
    } label: {
      VStack(spacing: 6) {
        PageThumbnail(isCover: index == 0)
          .aspectRatio(3.0 / 4.0, contentMode: .fit)
          .frame(maxWidth: .infinity)
          .background(theme.surface, in: RoundedRectangle(cornerRadius: 4))
          .overlay(
            RoundedRectangle(cornerRadius: 4)
              .stroke(isCurrent ? theme.action : theme.line, lineWidth: isCurrent ? 2 : 1)
          )
        Text(verbatim: "\(index + 1)")
          .font(Typeface.mono(11.5))
          .padding(.horizontal, 6)
          .padding(.vertical, 1)
          .background(isCurrent ? theme.voice : Color.clear, in: RoundedRectangle(cornerRadius: 4))
          .foregroundStyle(isCurrent ? theme.voiceInk : theme.ink2)
      }
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Page \(index + 1)")
  }

  private var currentPage: Int? {
    current.map { script.page(of: $0) }
  }

  private var currentPageNumber: String {
    currentPage.map { String($0 + 1) } ?? ""
  }

  private func goToTypedPage() {
    guard let number = Int(pageInput),
      number >= 1, number <= script.structure.pageCount,
      let position = script.position(onPage: number - 1)
    else { return }
    select(position)
  }

  private func select(_ position: ReadingPosition) {
    onSelect(position)
    dismiss()
  }
}

/// A schematic page: a cover with two title bars, or lines of text.
private struct PageThumbnail: View {
  let isCover: Bool
  @Environment(\.theme) private var theme

  private static let textFractions: [CGFloat] = [1, 1, 0.8, 1, 1, 0.65, 1, 1, 0.9]

  var body: some View {
    GeometryReader { geometry in
      VStack(alignment: .leading, spacing: 3) {
        if isCover {
          Spacer(minLength: 0)
          line(fraction: 0.7, height: 4, color: theme.thumbnailBar, width: geometry.size.width)
          line(fraction: 0.45, height: 3, color: theme.line, width: geometry.size.width)
          Spacer(minLength: 0)
        } else {
          ForEach(Self.textFractions.indices, id: \.self) { index in
            line(
              fraction: Self.textFractions[index], height: 2, color: theme.line,
              width: geometry.size.width)
          }
          Spacer(minLength: 0)
        }
      }
      .padding(8)
    }
  }

  private func line(fraction: CGFloat, height: CGFloat, color: Color, width: CGFloat) -> some View {
    Rectangle()
      .fill(color)
      .frame(width: (width - 16) * fraction, height: height)
      .frame(maxWidth: .infinity, alignment: isCover ? .center : .leading)
  }
}
