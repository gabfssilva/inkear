#if canImport(CoreAI)
  import Foundation
  import Testing

  @testable import Qwen3Engine

  @Suite struct Qwen3TTSTests {
    @Test func longSentencesAreCutIntoPiecesThatFitTheDecoder() {
      let sentence =
        "Vivia de pequenos golpes na internet: vendia produtos inexistentes e remédios falsificados "
        + "para velhinhas com câncer, enganava trouxas com serviços fantasmas, invadia computadores e "
        + "realizava transferências on-line de contas desprotegidas."

      let pieces = Qwen3TTS.pieces(of: sentence)

      #expect(pieces.count > 1)
      #expect(pieces.allSatisfy { $0.count <= Qwen3TTS.maxCharacters })
      #expect(pieces.joined(separator: " ") == sentence)
    }

    @Test func sentencesThatFitStayWhole() {
      let sentence = "O pastor Sérgio é mesmo fã da bispa Lygia."

      #expect(Qwen3TTS.pieces(of: sentence) == [sentence])
    }

    /// The audio of the first frames, decoded with the lookahead, is the start of the audio of all
    /// frames. Needs the model, installed under `INKEAR_MODELS`.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["INKEAR_MODELS"] != nil))
    func theEarlyFramesSoundAsTheStartOfTheWholePiece() async throws {
      let models = try #require(ProcessInfo.processInfo.environment["INKEAR_MODELS"])
      let bundle = try await Qwen3Bundle(
        directory: URL(filePath: models).appending(path: "qwen3"), voiceNames: [], progress: { _ in })
      let decoded = Qwen3TTS.earlyFrames + Qwen3TTS.decoderLookahead
      let samples = Qwen3TTS.earlyFrames * Qwen3Bundle.samplesPerFrame
      var generator = SystemRandomNumberGenerator()
      for _ in 0..<3 {
        let codebooks = (0..<Qwen3Bundle.codebooks).map { _ in
          (0..<100).map { _ in Int32.random(in: 0..<2048, using: &generator) }
        }

        let start = try await bundle.speech(codebooks: codebooks.map { Array($0.prefix(decoded)) })
        let whole = try await bundle.speech(codebooks: codebooks)

        #expect(start.prefix(samples) == whole.prefix(samples))
      }
    }
  }
#endif
