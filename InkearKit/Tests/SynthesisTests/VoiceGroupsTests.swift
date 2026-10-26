import Synthesis
import Testing

@Suite struct VoiceGroupsTests {
  @Test func groupsByLocalePrefix() {
    #expect(
      voiceGroups(["pt-BR · Manuela-Lu", "pt-BR · Davi", "en-US · Zoey", "fr · Noémie"])
        == [
          VoiceGroup(title: "pt-BR", voices: ["Manuela-Lu", "Davi"]),
          VoiceGroup(title: "en-US", voices: ["Zoey"]),
          VoiceGroup(title: "fr", voices: ["Noémie"]),
        ])
  }

  @Test func unheadedVoicesFormOneGroup() {
    #expect(
      voiceGroups(["pf_dora", "af_heart", "pt-BR · Clarice"])
        == [
          VoiceGroup(title: nil, voices: ["pf_dora", "af_heart"]),
          VoiceGroup(title: "pt-BR", voices: ["Clarice"]),
        ])
  }

  @Test func localeLikePrefixesAreNotSplit() {
    #expect(voiceGroups(["a · b"]) == [VoiceGroup(title: nil, voices: ["a · b"])])
    #expect(voiceGroups(["pt-br · X"]) == [VoiceGroup(title: nil, voices: ["pt-br · X"])])
    #expect(voiceGroups(["pt-BRR · X"]) == [VoiceGroup(title: nil, voices: ["pt-BRR · X"])])
  }
}
