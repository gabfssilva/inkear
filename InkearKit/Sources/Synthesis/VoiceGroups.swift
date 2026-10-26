/// A group of voices shown under one heading.
public struct VoiceGroup: Sendable, Equatable, Hashable {
  /// The heading, such as "pt-BR"; nil when the group has no locale.
  public let title: String?
  /// The voices of the group, without the locale prefix.
  public var voices: [String]

  public init(title: String?, voices: [String]) {
    self.title = title
    self.voices = voices
  }
}

/// Groups `voices` by the locale prefix of their names, such as `"pt-BR · Manuela-Lu"`.
/// Voices without a locale prefix form one unheaded group, in their order.
public func voiceGroups(_ voices: [String]) -> [VoiceGroup] {
  var groups: [VoiceGroup] = []
  for voice in voices {
    let parts = voice.split(separator: " · ", maxSplits: 1).map(String.init)
    if parts.count == 2, isLocale(parts[0]) {
      let title = parts[0]
      if let index = groups.lastIndex(where: { $0.title == title }) {
        groups[index].voices.append(parts[1])
      } else {
        groups.append(VoiceGroup(title: title, voices: [parts[1]]))
      }
    } else {
      if let index = groups.lastIndex(where: { $0.title == nil }) {
        groups[index].voices.append(voice)
      } else {
        groups.append(VoiceGroup(title: nil, voices: [voice]))
      }
    }
  }
  return groups
}

/// Whether `token` looks like a locale tag: two lowercase letters, optionally `-` and two
/// uppercase letters, such as `pt`, `pt-BR`, `en-US` or `es-MX`.
private func isLocale(_ token: String) -> Bool {
  let parts = token.split(separator: "-").map(String.init)
  switch parts.count {
  case 1:
    return parts[0].count == 2 && parts[0].allSatisfy { $0.isLetter && $0.isLowercase }
  case 2:
    return parts[0].count == 2 && parts[0].allSatisfy { $0.isLetter && $0.isLowercase }
      && parts[1].count == 2 && parts[1].allSatisfy { $0.isLetter && $0.isUppercase }
  default:
    return false
  }
}
