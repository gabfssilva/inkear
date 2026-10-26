import PDFStructure
import Playback
import SwiftUI
import Synthesis

/// How the listening screen shows the text.
enum PlayerMode: String {
  case text, focus
}

/// The listening screen: the document in full screen, with the sentence being read set larger on
/// the voice color, or in focus mode only the sentences around it. A tap shows the controls over
/// the text and another hides them; with VoiceOver they stay. A long press jumps to a paragraph
/// or a sentence, since a tap only toggles the controls.
struct PlayerView: View {
  let model: AppModel

  @Environment(\.theme) private var theme
  @Environment(\.accessibilityVoiceOverEnabled) private var voiceOver
  @AppStorage("playerMode") private var mode = PlayerMode.text
  @State private var tappedControls = false
  @State private var showsContents = false
  @State private var showsReading = false
  /// The top safe area, which the text reaches into, fading out, above the block being read.
  @State private var topInset: CGFloat = 0

  private static let textSize: CGFloat = 19
  /// The size of the sentence being read, relative to the text around it.
  private static let currentScale: CGFloat = 1.5
  /// Between the top safe area and the block being read.
  private static let topMargin: CGFloat = 20
  /// Text fades out over this height above the footer.
  private static let bottomFade: CGFloat = 36
  /// The strip at the bottom of the screen, over the home indicator's margin, that holds the
  /// footer.
  private static let footerHeight: CGFloat = 40

  var body: some View {
    if let narrator = model.narrator {
      let showsControls = tappedControls || voiceOver
      // The load of the voice shows without a tap: it is all the bottom bar holds meanwhile.
      let showsBottom = showsControls || model.awaitedEngine != nil
      reading(narrator)
        .overlay(alignment: .bottom) {
          if !showsBottom {
            footer(narrator)
              .transition(.opacity)
          }
        }
        .overlay(alignment: .top) {
          if showsControls {
            chrome(header(narrator))
              .transition(.opacity.combined(with: .offset(y: -14)))
          }
        }
        .overlay(alignment: .bottom) {
          if showsBottom {
            chrome(controls(narrator))
              .transition(.opacity.combined(with: .offset(y: 14)))
          }
        }
        .animation(.easeOut(duration: 0.25), value: showsControls)
        .animation(.easeOut(duration: 0.25), value: showsBottom)
        .statusBarHidden(!showsControls)
        .persistentSystemOverlays(showsControls ? .automatic : .hidden)
        .onChange(of: narrator.lastError) { _, error in
          if error != nil {
            tappedControls = true
          }
        }
        .sheet(isPresented: $showsContents) {
          ContentsSheet(script: narrator.script, current: narrator.position) {
            narrator.jump(to: $0)
          }
          .presentationDetents([.large])
        }
        .sheet(isPresented: $showsReading) {
          VoiceSheet(model: model)
            .presentationDetents([.medium, .large])
        }
    } else {
      emptyState
    }
  }

  /// The controls' bars: over the text, on a translucent ground that reaches the screen's edge.
  private func chrome(_ bar: some View) -> some View {
    bar
      .frame(maxWidth: .infinity)
      .background {
        ZStack {
          Rectangle().fill(.ultraThinMaterial)
          theme.ground.opacity(0.88)
        }
        .ignoresSafeArea()
      }
  }

  // MARK: Header and empty state

  private func header(_ narrator: Narrator) -> some View {
    HStack(spacing: 0) {
      backButton
        .frame(width: 96, alignment: .leading)
      VStack(spacing: 1) {
        Text(verbatim: narrator.title)
          .font(Typeface.ui(14, .medium))
          .foregroundStyle(theme.ink)
          .lineLimit(1)
        if let author = model.listening.flatMap({ model.record($0)?.author }) {
          Text(verbatim: author)
            .font(Typeface.ui(12.5))
            .foregroundStyle(theme.ink2)
            .lineLimit(1)
        }
      }
      .frame(maxWidth: .infinity)
      HStack(spacing: 0) {
        iconButton("list.bullet", label: "Contents") { showsContents = true }
        iconButton("ellipsis", label: "Reading") { showsReading = true }
      }
      .frame(width: 96, alignment: .trailing)
    }
    .padding(.horizontal, 8)
    .padding(.vertical, 8)
  }

  private var backButton: some View {
    iconButton("chevron.down", label: "Library", action: close)
  }

  private func iconButton(
    _ systemImage: String, label: LocalizedStringKey, action: @escaping () -> Void
  ) -> some View {
    Button(action: action) {
      Image(systemName: systemImage)
        .font(.system(size: 20, weight: .medium))
        .frame(width: 48, height: 48)
        .contentShape(Rectangle())
    }
    .buttonStyle(.plain)
    .foregroundStyle(theme.ink)
    .accessibilityLabel(label)
  }

  private var emptyState: some View {
    VStack(spacing: 0) {
      HStack {
        backButton
        Spacer()
      }
      .padding(.horizontal, 8)
      .padding(.top, 8)
      Spacer()
      VStack(spacing: 10) {
        Text("Nothing is playing")
          .font(Typeface.text(24))
          .foregroundStyle(theme.ink)
        Text("Choose a document in the library to start listening.")
          .font(Typeface.ui(14))
          .foregroundStyle(theme.ink2)
          .multilineTextAlignment(.center)
      }
      .padding(.horizontal, 28)
      Spacer()
    }
  }

  private func close() {
    if !model.path.isEmpty {
      model.path.removeLast()
    }
  }

  // MARK: Text and focus modes

  /// A tap on the text shows or hides the controls.
  private func reading(_ narrator: Narrator) -> some View {
    Group {
      switch mode {
      case .text: paragraphs(narrator)
      case .focus: focus(narrator)
      }
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .contentShape(Rectangle())
    .onTapGesture { tappedControls.toggle() }
  }

  /// The current block, the block before it and the next six, scrolled so the current block is
  /// at the top, under the text above it fading out up to the screen's edge.
  private func paragraphs(_ narrator: Narrator) -> some View {
    let blockCount = narrator.script.structure.blocks.count
    let current = narrator.position.block
    let blocks = max(0, current - 1)..<min(blockCount, current + 8)
    return ScrollViewReader { proxy in
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 14) {
          ForEach(blocks, id: \.self) { index in
            paragraph(index, narrator: narrator)
              .id(index)
          }
        }
        .padding(.horizontal, 28)
        .padding(.bottom, 24)
      }
      .scrollIndicators(.hidden)
      // Unclipped, the text past the scroll view's edges fades out under the mask instead of
      // being cut. The insets are outside the scroll view: scrolling the current block to the top
      // would hide a content inset.
      .scrollClipDisabled()
      .padding(.top, topInset + Self.topMargin)
      .padding(.bottom, Self.footerHeight)
      .mask {
        VStack(spacing: 0) {
          LinearGradient(colors: [.clear, .black], startPoint: .top, endPoint: .bottom)
            .frame(height: topInset)
          Color.black
          LinearGradient(colors: [.black, .clear], startPoint: .top, endPoint: .bottom)
            .frame(height: Self.bottomFade)
          Color.clear
            .frame(height: Self.footerHeight)
        }
      }
      .ignoresSafeArea(edges: .vertical)
      .onGeometryChange(for: CGFloat.self) { $0.safeAreaInsets.top } action: { topInset = $0 }
      .onAppear { proxy.scrollTo(current, anchor: .top) }
      .onChange(of: current) { _, block in
        withAnimation { proxy.scrollTo(block, anchor: .top) }
      }
    }
  }

  @ViewBuilder
  private func paragraph(_ index: Int, narrator: Narrator) -> some View {
    if !narrator.script.sentences(inBlock: index).isEmpty {
      blockText(index, narrator: narrator)
        .lineSpacing(9)
        .textRenderer(SentenceMarker(color: theme.voice))
        .jumps(to: ReadingPosition(block: index, sentence: 0), with: narrator)
    }
  }

  private func blockText(_ index: Int, narrator: Narrator) -> Text {
    let script = narrator.script
    let block = script.structure.blocks[index]
    let isHeading = if case .heading = block.kind { true } else { false }
    let pieces = script.sentences(inBlock: index).enumerated().map { sentence, range in
      self.sentence(
        String(block.text[range]), heading: isHeading,
        isCurrent: narrator.position == ReadingPosition(block: index, sentence: sentence))
    }
    return Text(joining: pieces, separator: " ")
  }

  /// A sentence in its block's style, or larger on the voice color when it is the one being read.
  private func sentence(_ text: String, heading: Bool, isCurrent: Bool) -> Text {
    let size = isCurrent ? Self.textSize * Self.currentScale : Self.textSize
    let piece = Text(verbatim: text)
      .font(heading ? Typeface.ui(size, .semibold) : Typeface.text(size))
      .foregroundStyle(isCurrent ? theme.voiceInk : heading ? theme.ink : theme.ink2)
    return isCurrent ? piece.customAttribute(CurrentSentence()) : piece
  }

  /// The previous sentence, the current one and the next two, centered.
  private func focus(_ narrator: Narrator) -> some View {
    let script = narrator.script
    let position = narrator.position
    let previous = script.previous(before: position).map { [$0] } ?? []
    return VStack(spacing: 18) {
      ForEach(previous + [position] + following(position, in: script), id: \.self) { sentence in
        focusSentence(sentence, narrator: narrator)
      }
    }
    .padding(.horizontal, 28)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .padding(.bottom, Self.footerHeight)
  }

  @ViewBuilder
  private func focusSentence(_ position: ReadingPosition, narrator: Narrator) -> some View {
    if let text = narrator.script.text(at: position) {
      sentence(text, heading: false, isCurrent: position == narrator.position)
        .lineSpacing(8)
        .multilineTextAlignment(.center)
        .textRenderer(SentenceMarker(color: theme.voice))
        .frame(maxWidth: .infinity)
        .jumps(to: position, with: narrator)
    }
  }

  private func following(_ position: ReadingPosition, in script: Script) -> [ReadingPosition] {
    var result: [ReadingPosition] = []
    var cursor = position
    while result.count < 2, let next = script.next(after: cursor) {
      result.append(next)
      cursor = next
    }
    return result
  }

  /// Under the text while the controls are hidden: the chapter and the time left in it, or that
  /// the reading is paused.
  private func footer(_ narrator: Narrator) -> some View {
    ticking(narrator) { footerText(narrator) }
      .font(Typeface.mono(11))
      .tracking(1.2)
      .textCase(.uppercase)
      .foregroundStyle(theme.ink2)
      .lineLimit(1)
      .padding(.bottom, 24)
      .frame(height: Self.footerHeight, alignment: .bottom)
      .ignoresSafeArea(edges: .bottom)
      .allowsHitTesting(false)
  }

  private func footerText(_ narrator: Narrator) -> Text {
    let chapter = narrator.script.section(containing: narrator.position).map { $0 + 1 }
    let time = narrator.sectionTime
    let minutes = max(1, Int((max(0, time.total - time.elapsed) / 60).rounded(.up)))
    switch (isPlaying(narrator), chapter) {
    case (true, let chapter?): return Text("Ch. \(chapter) · \(minutes) min to go")
    case (true, nil): return Text("\(minutes) min to go")
    case (false, let chapter?): return Text("Paused · ch. \(chapter)")
    case (false, nil): return Text("Paused")
    }
  }

  private func isPlaying(_ narrator: Narrator) -> Bool {
    narrator.state == .playing || narrator.state == .loading
  }

  // MARK: Progress and controls

  /// While the narrator waits for its engine, only its load: the engine's name, a ring with the
  /// load in place of the transport, and why it takes long. After an error that stopped the
  /// reading, the error shows under the transport until the next play.
  private func controls(_ narrator: Narrator) -> some View {
    VStack(spacing: 16) {
      if let awaited = model.awaitedEngine {
        VStack(spacing: 4) {
          MonoLabel("Preparing the voice")
          Text(verbatim: model.displayName(of: awaited))
            .font(Typeface.ui(17, .medium))
            .foregroundStyle(theme.ink)
        }
        LoadingRing(
          progress: loadProgress(of: awaited), times: model.loadTimes[awaited] ?? [], size: 68)
        OptimizingNote()
          .font(Typeface.ui(12.5))
          .foregroundStyle(theme.ink2)
          .frame(maxWidth: .infinity, alignment: .leading)
      } else {
        progress(narrator)
        transport(narrator)
        if let error = narrator.lastError {
          Label(error, systemImage: "exclamationmark.triangle")
            .font(Typeface.ui(12.5))
            .foregroundStyle(theme.ink2)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityIdentifier("playbackError")
        }
      }
    }
    .padding(.horizontal, 28)
    .padding(.top, 18)
    .padding(.bottom, 14)
    .animation(.default, value: model.awaitedEngine)
  }

  private func progress(_ narrator: Narrator) -> some View {
    ticking(narrator) {
      let time = narrator.sectionTime
      let fraction = time.total > 0 ? min(1, time.elapsed / time.total) : 0
      let remaining = max(0, time.total - time.elapsed)
      let page = narrator.script.page(of: narrator.position) + 1
      VStack(spacing: 9) {
        ProgressLine(value: fraction, tint: theme.action)
        HStack {
          Text(verbatim: clock(time.elapsed))
            .accessibilityIdentifier("elapsed")
          Spacer()
          Text("p. \(page)")
          Spacer()
          Text(verbatim: "−" + clock(remaining))
        }
        .font(Typeface.mono(12))
        .foregroundStyle(theme.ink2)
      }
    }
  }

  /// `content` redrawn every second while the narrator plays, since the time it shows advances
  /// with the audio between changes of the narrator's state.
  private func ticking<Content: View>(
    _ narrator: Narrator, @ViewBuilder content: @escaping () -> Content
  ) -> some View {
    TimelineView(.animation(minimumInterval: 1, paused: narrator.state != .playing)) { _ in
      content()
    }
  }

  private func transport(_ narrator: Narrator) -> some View {
    let isPlaying = isPlaying(narrator)
    return HStack(spacing: 44) {
      RoundButton(systemImage: "backward.end.fill", size: 52) { narrator.previousSection() }
        .accessibilityLabel("Previous chapter")
      RoundButton(systemImage: isPlaying ? "pause.fill" : "play.fill", size: 68, filled: true) {
        narrator.toggle()
      }
      .accessibilityLabel(isPlaying ? "Pause" : "Play")
      RoundButton(systemImage: "forward.end.fill", size: 52) { narrator.nextSection() }
        .accessibilityLabel("Next chapter")
    }
  }

  private func loadProgress(of id: EngineID) -> LoadProgress? {
    if case .loading(let progress)? = model.engineStates[id] { progress } else { nil }
  }

  private func clock(_ seconds: Double) -> String {
    Duration.seconds(max(0, seconds)).formatted(.time(pattern: .minuteSecond))
  }
}

extension View {
  /// Jumps to `position` on a long press, or on VoiceOver's activation.
  fileprivate func jumps(to position: ReadingPosition, with narrator: Narrator) -> some View {
    onLongPressGesture { narrator.jump(to: position) }
      .accessibilityAddTraits(.isButton)
      .accessibilityAction { narrator.jump(to: position) }
  }
}

/// Marks the sentence being read, for ``SentenceMarker`` to draw the voice color behind it.
private struct CurrentSentence: TextAttribute {}

/// Draws a rounded marker in `color` behind each line's part of the sentence being read.
private struct SentenceMarker: TextRenderer {
  let color: Color

  func draw(layout: Text.Layout, in context: inout GraphicsContext) {
    for run in layout.flatMap({ $0 }) where run[CurrentSentence.self] != nil {
      let bounds = run.typographicBounds.rect.insetBy(dx: -4, dy: -1)
      context.fill(Path(roundedRect: bounds, cornerRadius: 7), with: .color(color))
    }
    for line in layout {
      context.draw(line)
    }
  }
}

extension Text {
  /// `pieces` as one text with `separator` between them, each piece keeping its own style.
  ///
  /// Built from an interpolation rather than a literal, so that its format string is neither
  /// localized nor extracted to the string catalog.
  fileprivate init(joining pieces: [Text], separator: String) {
    var interpolation = LocalizedStringKey.StringInterpolation(
      literalCapacity: separator.count * pieces.count, interpolationCount: pieces.count)
    for (index, piece) in pieces.enumerated() {
      if index > 0 {
        interpolation.appendLiteral(separator)
      }
      interpolation.appendInterpolation(piece)
    }
    self.init(LocalizedStringKey(stringInterpolation: interpolation))
  }
}
