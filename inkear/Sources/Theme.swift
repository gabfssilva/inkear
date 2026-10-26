import SwiftUI

/// The color themes the reader picks from, each with a light and a dark palette.
enum ThemeID: String, CaseIterable, Identifiable {
  case sage, ink, sepia, graphite, blue, red, orange

  var id: Self { self }

  /// The theme's name in the picker.
  var title: LocalizedStringKey {
    switch self {
    case .sage: "Sage"
    case .ink: "Ink"
    case .sepia: "Sepia"
    case .graphite: "Graphite"
    case .blue: "Blue"
    case .red: "Red"
    case .orange: "Orange"
    }
  }
}

struct Theme: Sendable {
  let ground, surface, control, line, fineLine, chipOutline, cardRing, thumbnailBar: Color
  let ink, ink2: Color
  /// Primary buttons.
  let action, actionInk: Color
  /// The marker behind the sentence being read, and chips.
  let voice, voiceInk: Color
  /// Graphics only.
  let preparing: Color
}

/// The base colors of a palette. The other `Theme` colors are derived from them.
private struct Palette {
  let ground, ink, ink2, voice, voiceInk, action, actionInk, line: UInt32

  init(
    _ ground: UInt32, _ ink: UInt32, _ ink2: UInt32, _ voice: UInt32,
    _ voiceInk: UInt32, _ action: UInt32, _ actionInk: UInt32, _ line: UInt32
  ) {
    self.ground = ground
    self.ink = ink
    self.ink2 = ink2
    self.voice = voice
    self.voiceInk = voiceInk
    self.action = action
    self.actionInk = actionInk
    self.line = line
  }
}

extension Theme {
  /// The palette of `id` for the light or dark appearance.
  static func of(_ id: ThemeID, dark: Bool) -> Theme {
    let pair = palettes(of: id)
    return Theme(dark ? pair.dark : pair.light, dark: dark)
  }

  private static func palettes(of id: ThemeID) -> (light: Palette, dark: Palette) {
    switch id {
    case .sage:
      (
        Palette(0xF4F1EA, 0x2F342B, 0x5E6356, 0xA3B18A, 0x2F342B, 0x4B5D3A, 0xF4F1EA, 0xD6CCB8),
        Palette(0x2F342B, 0xF4F1EA, 0xA09E8C, 0x4D5843, 0xF4F1EA, 0xA3B18A, 0x2F342B, 0x454B3F)
      )
    case .ink:
      (
        Palette(0xF2F1ED, 0x1C2033, 0x5A5F73, 0xCDD3F1, 0x1C2033, 0x34419C, 0xF2F1ED, 0xD6D5CF),
        Palette(0x13151D, 0xE9E7E1, 0x8E92A3, 0x2C3463, 0xF1F0FA, 0xAEB8F0, 0x13151D, 0x2A2E3D)
      )
    case .sepia:
      (
        Palette(0xF3EAD8, 0x3B2F23, 0x6B5C4B, 0xEDCB80, 0x2E2318, 0x7E5218, 0xF8F1E3, 0xDCCDB0),
        Palette(0x1C1814, 0xEEE3CF, 0xA39581, 0x4D3B1F, 0xF6E9CF, 0xE2B25F, 0x1C1814, 0x3A3128)
      )
    case .graphite:
      (
        Palette(0xEFEFEC, 0x1A1A19, 0x5C5C58, 0xE2F26B, 0x1A1A19, 0x1A1A19, 0xEFEFEC, 0xD3D3CF),
        Palette(0x121212, 0xE6E6E3, 0x8D8D88, 0xC5D84E, 0x121212, 0xE6E6E3, 0x121212, 0x2C2C2C)
      )
    case .blue:
      (
        Palette(0xF6F8FB, 0x13213A, 0x4E5B72, 0xCFE0FF, 0x13213A, 0x1F5FD6, 0xF6F8FB, 0xD8DEE8),
        Palette(0x0D1626, 0xE6ECF5, 0x8D9BB1, 0x1E3D73, 0xEAF1FF, 0x7FAAFF, 0x0D1626, 0x223049)
      )
    case .red:
      (
        Palette(0xFBF6F4, 0x2A1715, 0x6A5552, 0xF7CEC7, 0x2A1715, 0xC2332B, 0xFFF7F5, 0xE9DAD6),
        Palette(0x1A1112, 0xF3E6E4, 0xAA918E, 0x5C2020, 0xFBE9E6, 0xFF7D70, 0x1A1112, 0x3A2627)
      )
    case .orange:
      (
        Palette(0xFBF7F1, 0x2B1D12, 0x6B5A4A, 0xFFD4AA, 0x2B1D12, 0xC4530F, 0xFFF8F0, 0xEADFD2),
        Palette(0x17120E, 0xF2E8DE, 0xA69482, 0x5E3511, 0xFFEBD8, 0xFF9A4D, 0x17120E, 0x33281F)
      )
    }
  }

  private init(_ p: Palette, dark: Bool) {
    func tone(_ weight: Double) -> Color {
      Color(hex: mix(p.ground, p.ink, weight))
    }

    self.init(
      ground: Color(hex: p.ground),
      surface: dark ? tone(0.06) : Color(hex: mix(p.ground, 0xFFFFFF, 0.6)),
      control: tone(dark ? 0.10 : 0.07),
      line: Color(hex: p.line),
      fineLine: tone(dark ? 0.12 : 0.10),
      chipOutline: tone(dark ? 0.28 : 0.22),
      cardRing: tone(dark ? 0.12 : 0.09),
      thumbnailBar: tone(0.40),
      ink: Color(hex: p.ink),
      ink2: Color(hex: p.ink2),
      action: Color(hex: p.action),
      actionInk: Color(hex: p.actionInk),
      voice: Color(hex: p.voice),
      voiceInk: Color(hex: p.voiceInk),
      preparing: tone(0.50)
    )
  }
}

extension EnvironmentValues {
  @Entry var theme: Theme = .of(.sage, dark: false)
}

extension View {
  /// Injects the chosen theme, light or dark, and paints the ground behind the content.
  func themed() -> some View {
    modifier(ThemedModifier())
  }
}

enum Typeface {
  static func ui(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
    Font.custom("Geist", size: size).weight(weight)
  }

  static func text(_ size: CGFloat, _ weight: Font.Weight = .regular, italic: Bool = false) -> Font
  {
    let font = Font.custom("Newsreader", size: size).weight(weight)
    return italic ? font.italic() : font
  }

  static func mono(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
    Font.custom("Geist Mono", size: size).weight(weight)
  }
}

struct Chip: View {
  enum Style { case voice, outlined, preparing }

  @Environment(\.theme) private var theme
  private let title: Text
  private let style: Style

  init(_ title: LocalizedStringKey, style: Style) {
    self.title = Text(title)
    self.style = style
  }

  init(verbatim title: String, style: Style) {
    self.title = Text(verbatim: title)
    self.style = style
  }

  var body: some View {
    title
      .font(Typeface.mono(10.5))
      .tracking(0.84)
      .textCase(.uppercase)
      .lineLimit(1)
      .fixedSize()
      .foregroundStyle(foreground)
      .padding(.horizontal, style == .preparing ? 0 : 6)
      .padding(.vertical, style == .preparing ? 0 : 3)
      .background(background, in: RoundedRectangle(cornerRadius: 4))
      .overlay {
        if style == .outlined {
          RoundedRectangle(cornerRadius: 4).strokeBorder(theme.chipOutline, lineWidth: 1)
        }
      }
  }

  private var foreground: Color {
    switch style {
    case .voice: theme.voiceInk
    case .outlined: theme.ink
    case .preparing: theme.ink2
    }
  }

  private var background: Color {
    switch style {
    case .voice: theme.voice
    case .outlined, .preparing: .clear
    }
  }
}

struct MonoLabel: View {
  @Environment(\.theme) private var theme
  private let title: Text

  init(_ title: LocalizedStringKey) {
    self.title = Text(title)
  }

  var body: some View {
    title
      .font(Typeface.mono(11))
      .tracking(1.32)
      .textCase(.uppercase)
      .foregroundStyle(theme.ink2)
  }
}

/// A 3pt line. With a nil value it sweeps a short segment across the track.
struct ProgressLine: View {
  @Environment(\.theme) private var theme
  @State private var sweeping = false
  private let value: Double?
  private let tint: Color
  private let sweepEnd: Double?

  private static let sweepFraction = 0.38

  /// A bar filled to `value`, or swept whole when it is nil. With `sweepingTo`, the share from
  /// `value` to it sweeps too: the work in progress.
  init(value: Double?, tint: Color, sweepingTo sweepEnd: Double? = nil) {
    self.value = value
    self.tint = tint
    self.sweepEnd = sweepEnd
  }

  var body: some View {
    GeometryReader { proxy in
      let width = proxy.size.width
      ZStack(alignment: .leading) {
        Capsule().fill(theme.line)
        if let value {
          let filled = min(max(value, 0), 1)
          Capsule()
            .fill(tint)
            .frame(width: width * filled)
          if let sweepEnd, sweepEnd > filled {
            let segment = width * (min(sweepEnd, 1) - filled)
            Capsule()
              .fill(tint.opacity(0.5))
              .frame(width: segment * Self.sweepFraction)
              .offset(x: sweeping ? segment : -segment * Self.sweepFraction)
              .frame(width: segment, alignment: .leading)
              .clipped()
              .offset(x: width * filled)
          }
        } else {
          Capsule()
            .fill(tint)
            .frame(width: width * Self.sweepFraction)
            .offset(x: sweeping ? width : -width * Self.sweepFraction)
        }
      }
    }
    .frame(height: 3)
    .clipShape(Capsule())
    .onAppear {
      guard value == nil || sweepEnd != nil else { return }
      withAnimation(.linear(duration: 1.4).repeatForever(autoreverses: false)) {
        sweeping = true
      }
    }
  }
}

struct PrimaryButtonStyle: ButtonStyle {
  func makeBody(configuration: Configuration) -> some View {
    ButtonBody(configuration: configuration, filled: true)
  }
}

struct SecondaryButtonStyle: ButtonStyle {
  func makeBody(configuration: Configuration) -> some View {
    ButtonBody(configuration: configuration, filled: false)
  }
}

private struct ButtonBody: View {
  @Environment(\.theme) private var theme
  let configuration: ButtonStyleConfiguration
  let filled: Bool

  var body: some View {
    configuration.label
      .font(Typeface.ui(16, .semibold))
      .foregroundStyle(filled ? theme.actionInk : theme.ink)
      .frame(maxWidth: filled ? .infinity : nil)
      .frame(height: 54)
      .padding(.horizontal, filled ? 0 : 24)
      .background(filled ? theme.action : theme.control, in: Capsule())
      .opacity(configuration.isPressed ? 0.8 : 1)
  }
}

struct RoundButton: View {
  @Environment(\.theme) private var theme
  private let systemImage: String
  private let size: CGFloat
  private let filled: Bool
  private let action: () -> Void

  init(systemImage: String, size: CGFloat = 44, filled: Bool = false, action: @escaping () -> Void)
  {
    self.systemImage = systemImage
    self.size = size
    self.filled = filled
    self.action = action
  }

  var body: some View {
    Button(action: action) {
      Image(systemName: systemImage)
        .font(.system(size: size * 0.5))
        .frame(width: size, height: size)
        .foregroundStyle(filled ? theme.actionInk : theme.ink)
        .background(filled ? theme.action : .clear, in: Circle())
        .contentShape(Circle())
    }
    .buttonStyle(.plain)
  }
}

/// A PDF cover at 2:3. Without an image it draws a page with two title bars.
struct CoverView: View {
  @Environment(\.theme) private var theme
  private let image: Image?
  private let title: String

  init(image: Image?, title: String) {
    self.image = image
    self.title = title
  }

  var body: some View {
    Color.clear
      .aspectRatio(2.0 / 3.0, contentMode: .fit)
      .overlay {
        if let image {
          image.resizable().scaledToFill()
        } else {
          placeholder
        }
      }
      .clipShape(RoundedRectangle(cornerRadius: 4))
      .overlay {
        RoundedRectangle(cornerRadius: 4).strokeBorder(theme.cardRing, lineWidth: 1)
      }
      .accessibilityLabel(title)
  }

  private var placeholder: some View {
    GeometryReader { proxy in
      VStack(spacing: 4) {
        Rectangle()
          .fill(theme.thumbnailBar)
          .frame(width: proxy.size.width * 0.7, height: 3)
        Rectangle()
          .fill(theme.thumbnailBar)
          .frame(width: proxy.size.width * 0.45, height: 3)
      }
      .frame(maxWidth: .infinity, alignment: .top)
      .padding(.top, proxy.size.height * 0.16)
      .frame(width: proxy.size.width, height: proxy.size.height, alignment: .top)
      .background(theme.surface)
    }
  }
}

private struct ThemedModifier: ViewModifier {
  @Environment(\.colorScheme) private var colorScheme
  @AppStorage("theme") private var themeID = ThemeID.sage

  func body(content: Content) -> some View {
    let theme = Theme.of(themeID, dark: colorScheme == .dark)
    content
      .environment(\.theme, theme)
      .background(theme.ground.ignoresSafeArea())
  }
}

/// Mixes two RGB colors per channel. `weight` 0 gives `a`, 1 gives `b`.
private func mix(_ a: UInt32, _ b: UInt32, _ weight: Double) -> UInt32 {
  func channel(_ shift: UInt32) -> UInt32 {
    let from = Double((a >> shift) & 0xFF)
    let to = Double((b >> shift) & 0xFF)
    return UInt32((from + (to - from) * weight).rounded())
  }
  return channel(16) << 16 | channel(8) << 8 | channel(0)
}

extension Color {
  fileprivate init(hex: UInt32) {
    self.init(
      red: Double((hex >> 16) & 0xFF) / 255,
      green: Double((hex >> 8) & 0xFF) / 255,
      blue: Double(hex & 0xFF) / 255
    )
  }
}
