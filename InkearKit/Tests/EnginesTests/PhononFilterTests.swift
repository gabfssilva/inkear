import PhononEngine
import Testing

/// The voice-to-language filter of `PhononEngine`, pure and model-free.
@Suite struct PhononFilterTests {
  static let voices = [
    "pt-BR · Manuela-Lu", "pt-BR · Davi", "en-US · Zoey", "en-GB · Maeve",
    "fr · Noémie", "de · Annika", "es-ES · Noa", "es-MX · Lalo", "en · Sir Bramwell",
  ]

  @Test(arguments: [
    ("pt", ["pt-BR · Manuela-Lu", "pt-BR · Davi"]),
    ("en", ["en-US · Zoey", "en-GB · Maeve", "en · Sir Bramwell"]),
    ("fr", ["fr · Noémie"]),
    ("de", ["de · Annika"]),
    ("es", ["es-ES · Noa", "es-MX · Lalo"]),
    ("it", Self.voices),
  ])
  func filtersByLanguage(_ pair: (String, [String])) {
    #expect(PhononEngine.filtered(Self.voices, for: pair.0) == pair.1)
  }

  @Test func fallsBackToAllWhenNoVoiceMatches() {
    #expect(PhononEngine.filtered(["pf_dora", "af_heart"], for: "pt") == ["pf_dora", "af_heart"])
  }
}
