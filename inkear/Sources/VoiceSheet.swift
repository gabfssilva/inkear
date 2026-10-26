import Playback
import SwiftUI
import Synthesis

/// Groups voices by their locale prefix, e.g. "pt-BR · Manuela-Lu" -> "pt-BR" + "Manuela-Lu".
private func voiceGroups(_ voices: [String]) -> [VoiceGroup] {
  Synthesis.voiceGroups(voices)
}

/// Speed, sleep timer, voice, quality (for engines that use it) and language for the narrator in
/// `model`, then the reading mode and the color theme.
struct VoiceSheet: View {
  let model: AppModel

  @Environment(\.theme) private var theme
  @Environment(\.colorScheme) private var colorScheme
  @Environment(\.dismiss) private var dismiss
  @AppStorage("playerMode") private var mode = PlayerMode.text
  @AppStorage("theme") private var themeID = ThemeID.sage

  private static let sleepOptions: [SleepTimer] = [
    .off, .endOfSection, .minutes(15), .minutes(30), .minutes(60),
  ]

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      header
      if let narrator = model.narrator {
        ScrollView {
          VStack(alignment: .leading, spacing: 24) {
            playbackSection(narrator)
            voiceSection(narrator)
            appearanceSection
          }
          .padding(.horizontal, 24)
          .padding(.top, 14)
          .padding(.bottom, 24)
        }
      }
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    .presentationBackground(theme.ground)
  }

  private var header: some View {
    HStack {
      Text("Reading")
        .font(Typeface.ui(22, .semibold))
        .tracking(-0.55)
        .foregroundStyle(theme.ink)
      Spacer()
      RoundButton(systemImage: "xmark") { dismiss() }
        .accessibilityLabel("Close")
    }
    .padding(.leading, 24)
    .padding(.trailing, 12)
    .padding(.top, 8)
  }

  private func sectionTitle(_ title: LocalizedStringKey) -> some View {
    Text(title)
      .font(Typeface.ui(13, .medium))
      .foregroundStyle(theme.ink2)
  }

  // MARK: Playback

  private func playbackSection(_ narrator: Narrator) -> some View {
    VStack(alignment: .leading, spacing: 10) {
      sectionTitle("Playback")
      HStack(spacing: 6) {
        ForEach(AppModel.speeds, id: \.self) { speed in
          speedButton(speed, selected: speed == narrator.speed)
        }
      }
      .accessibilityElement(children: .contain)
      .accessibilityLabel("Speed")
      sleepMenu(narrator)
    }
  }

  private func speedButton(_ speed: Double, selected: Bool) -> some View {
    let label = speed.formatted(.number.precision(.fractionLength(0...2))) + "×"
    return Button {
      model.setSpeed(speed)
    } label: {
      Text(verbatim: label)
        .font(Typeface.mono(13.5, .medium))
        .frame(maxWidth: .infinity)
        .frame(height: 44)
        .background(selected ? theme.action : theme.control, in: RoundedRectangle(cornerRadius: 12))
        .foregroundStyle(selected ? theme.actionInk : theme.ink)
    }
    .buttonStyle(.plain)
    .accessibilityAddTraits(selected ? AccessibilityTraits.isSelected : [])
  }

  // MARK: Voice

  private func voiceSection(_ narrator: Narrator) -> some View {
    let engine = narrator.voice.engine
    return VStack(alignment: .leading, spacing: 10) {
      sectionTitle("Voice")
      VStack(spacing: 0) {
        modelMenu(narrator)
        engineProgress()
      }
      LazyVStack(alignment: .leading, spacing: 16) {
        ForEach(voiceGroups(model.voices(of: engine, language: narrator.language)), id: \.self) { group in
          VStack(alignment: .leading, spacing: 8) {
            if let title = group.title {
              Text(verbatim: title)
                .font(Typeface.ui(13, .medium))
                .foregroundStyle(theme.ink2)
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 64), spacing: 6)], spacing: 6) {
              ForEach(group.voices, id: \.self) { name in
                let voice = VoiceID(engine: engine, name: group.title.map { "\($0) · \(name)" } ?? name)
                voiceButton(voice, label: name, selected: voice == narrator.voice)
              }
            }
          }
        }
      }
      VStack(spacing: 0) {
        if model.usesQuality(engine) {
          qualityMenu(narrator)
        }
        languageMenu(narrator)
      }
    }
  }

  private func modelMenu(_ narrator: Narrator) -> some View {
    Menu {
      ForEach(model.engineIDs.filter(model.installedEngines.contains), id: \.self) { id in
        Button {
          model.selectEngine(id)
        } label: {
          if id == narrator.voice.engine {
            Label(model.displayName(of: id), systemImage: "checkmark")
          } else if let status = engineStatus(id) {
            VStack(alignment: .leading) {
              Text(verbatim: model.displayName(of: id))
              status
            }
          } else {
            Text(verbatim: model.displayName(of: id))
          }
        }
      }
      Divider()
      Button {
        dismiss()
        model.showModels()
      } label: {
        Label("Download models", systemImage: "arrow.down.circle")
      }
    } label: {
      settingRow("Model", value: Text(verbatim: model.displayName(of: narrator.voice.engine)))
        .overlay(alignment: .bottom) {
          Rectangle().fill(theme.fineLine).frame(height: 1)
        }
    }
    .buttonStyle(.plain)
  }

  /// The state of an installed engine that is not the current one, when it is worth showing.
  private func engineStatus(_ id: EngineID) -> Text? {
    switch model.engineStates[id] {
    case .loading?:
      return Text("Loading")
    case .failed?:
      return Text("Failed")
    case .notInstalled?, .downloading?, .preparing?, .installed?, .ready?, nil:
      return nil
    }
  }

  /// The engine being loaded, shown under the model row.
  @ViewBuilder
  private func engineProgress() -> some View {
    if let id = model.engineIDs.first(where: { model.engineStates[$0]?.isLoading == true }),
      case .loading(let progress)? = model.engineStates[id]
    {
      EngineLoadingView(
        title: Text("Loading \(model.displayName(of: id))"), progress: progress,
        times: model.loadTimes[id] ?? [])
        .padding(.vertical, 8)
    }
  }

  private func voiceButton(_ voice: VoiceID, label: String, selected: Bool) -> some View {
    Button {
      model.setVoice(voice)
    } label: {
      Text(verbatim: label)
        .font(Typeface.mono(13.5, .medium))
        .lineLimit(1)
        .minimumScaleFactor(0.7)
        .frame(maxWidth: .infinity)
        .frame(height: 48)
        .background(selected ? theme.voice : theme.control, in: RoundedRectangle(cornerRadius: 12))
        .overlay {
          if selected {
            RoundedRectangle(cornerRadius: 12).strokeBorder(theme.ink, lineWidth: 1.5)
          }
        }
        .foregroundStyle(selected ? theme.voiceInk : theme.ink)
    }
    .buttonStyle(.plain)
    .accessibilityAddTraits(selected ? AccessibilityTraits.isSelected : [])
  }

  // MARK: Quality and language

  private func qualityMenu(_ narrator: Narrator) -> some View {
    Menu {
      ForEach(SpeechQuality.allCases, id: \.self) { quality in
        Button {
          model.setQuality(quality)
        } label: {
          if quality == narrator.quality {
            Label(qualityTitle(quality), systemImage: "checkmark")
          } else {
            Text(qualityTitle(quality))
          }
        }
      }
    } label: {
      settingRow("Quality", value: Text(qualityTitle(narrator.quality)))
        .overlay(alignment: .bottom) {
          Rectangle().fill(theme.fineLine).frame(height: 1)
        }
    }
    .buttonStyle(.plain)
  }

  private func qualityTitle(_ quality: SpeechQuality) -> LocalizedStringKey {
    switch quality {
    case .low: "Low"
    case .normal: "Normal"
    case .high: "High"
    case .highest: "Highest"
    }
  }

  private func languageMenu(_ narrator: Narrator) -> some View {
    let detected = narrator.script.structure.language
    let isDetected = narrator.language == detected
    return Menu {
      ForEach(AppModel.languages, id: \.self) { code in
        Button {
          model.setLanguage(code)
        } label: {
          if narrator.language == code {
            Label(languageName(code), systemImage: "checkmark")
          } else {
            Text(verbatim: languageName(code))
          }
        }
      }
      Button {
        model.setLanguage(nil)
      } label: {
        if isDetected {
          Label("Detected", systemImage: "checkmark")
        } else {
          Text("Detected")
        }
      }
    } label: {
      settingRow("Language", value: languageValue(narrator.language, isDetected: isDetected))
    }
    .buttonStyle(.plain)
  }

  private func languageValue(_ code: String, isDetected: Bool) -> Text {
    let name = languageName(code)
    return isDetected ? Text("\(name) · detected") : Text(verbatim: name)
  }

  private func languageName(_ code: String) -> String {
    let name = Locale.current.localizedString(forLanguageCode: code) ?? code
    return String(name.prefix(1)).uppercased() + name.dropFirst()
  }

  // MARK: Sleep timer

  private func sleepMenu(_ narrator: Narrator) -> some View {
    Menu {
      ForEach(Self.sleepOptions, id: \.self) { option in
        Button {
          narrator.sleepTimer = option
        } label: {
          if option == narrator.sleepTimer {
            Label(sleepTitle(option), systemImage: "checkmark")
          } else {
            Text(sleepTitle(option))
          }
        }
      }
    } label: {
      settingRow("Stop reading", value: Text(sleepTitle(narrator.sleepTimer)))
    }
    .buttonStyle(.plain)
  }

  private func sleepTitle(_ timer: SleepTimer) -> LocalizedStringKey {
    switch timer {
    case .off: "Off"
    case .endOfSection: "End of chapter"
    case .minutes(let minutes): "\(minutes) min"
    }
  }

  // MARK: Appearance

  private var appearanceSection: some View {
    VStack(alignment: .leading, spacing: 10) {
      sectionTitle("Appearance")
      modeControl
      themeGrid
    }
    .padding(.top, 22)
    .overlay(alignment: .top) {
      Rectangle().fill(theme.fineLine).frame(height: 1)
    }
  }

  private var modeControl: some View {
    HStack(spacing: 3) {
      modeButton("Text", value: .text)
      modeButton("Focus", value: .focus)
    }
    .padding(3)
    .background(theme.control, in: RoundedRectangle(cornerRadius: 12))
    .accessibilityElement(children: .contain)
    .accessibilityLabel("Reading mode")
  }

  private func modeButton(_ title: LocalizedStringKey, value: PlayerMode) -> some View {
    let selected = mode == value
    return Button {
      mode = value
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

  private var themeGrid: some View {
    LazyVGrid(
      columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 4), spacing: 12
    ) {
      ForEach(ThemeID.allCases) { id in
        themeButton(id)
      }
    }
    .padding(.top, 4)
  }

  private func themeButton(_ id: ThemeID) -> some View {
    let palette = Theme.of(id, dark: colorScheme == .dark)
    let selected = id == themeID
    return Button {
      themeID = id
    } label: {
      VStack(spacing: 6) {
        HStack(spacing: 6) {
          Text(verbatim: "Aa")
            .font(Typeface.text(18))
            .foregroundStyle(palette.voiceInk)
            .padding(.horizontal, 5)
            .background(palette.voice, in: RoundedRectangle(cornerRadius: 5))
          Circle().fill(palette.action).frame(width: 8, height: 8)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 56)
        .background(palette.ground, in: RoundedRectangle(cornerRadius: 12))
        .overlay {
          RoundedRectangle(cornerRadius: 12)
            .strokeBorder(selected ? theme.ink : theme.chipOutline, lineWidth: selected ? 2 : 1)
        }
        Text(id.title)
          .font(Typeface.ui(12))
          .foregroundStyle(selected ? theme.ink : theme.ink2)
      }
    }
    .buttonStyle(.plain)
    .accessibilityLabel(Text(id.title))
    .accessibilityAddTraits(selected ? AccessibilityTraits.isSelected : [])
  }

  private func settingRow(_ title: LocalizedStringKey, value: Text) -> some View {
    HStack(spacing: 12) {
      Text(title)
        .frame(maxWidth: .infinity, alignment: .leading)
      value
        .foregroundStyle(theme.ink2)
      Image(systemName: "chevron.right")
        .font(.system(size: 13, weight: .semibold))
        .foregroundStyle(theme.ink2)
    }
    .font(Typeface.ui(15))
    .foregroundStyle(theme.ink)
    .frame(minHeight: 52)
    .contentShape(Rectangle())
  }
}
