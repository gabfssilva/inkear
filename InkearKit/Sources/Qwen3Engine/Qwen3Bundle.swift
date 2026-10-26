// Adapted from speech-swift/Sources/Qwen3TTSCoreML/{EmbedderModels,SpeechDecoderCoreML,TalkerGenerator}.swift, Apache-2.0.
#if canImport(CoreAI)
  import CoreAI
  import Foundation
  import Synthesis

  /// The Core AI functions of a Qwen3-TTS 0.6B bundle, and the tensors read from its directory.
  ///
  /// Every function runs on the Neural Engine, from packages compiled ahead of time for the
  /// device's architecture. The embeddings are tables the bundle reads itself: MPSGraph runs a
  /// lookup by integer index on the GPU, which iOS denies to apps in the background.
  struct Qwen3Bundle: Sendable {
    /// The packages, which `coreai-aot` compiled for each iPhone architecture.
    static let packages = ["CodeDecoder", "MultiCodeDecoder", "SpeechDecoder"]
    static let hidden = 1024
    /// Positions of the talker's KV cache. The prompt and the generated frames share them.
    static let positions = 256
    /// Columns of the prefill. The prompt takes its text tokens plus 11 control rows.
    static let prefillColumns = 128
    static let maxTextTokens = prefillColumns - 11
    static let talkerKVWidth = 28672
    static let codebooks = 16
    static let residualVocabulary = 2048
    static let frames = 125
    static let samplesPerFrame = 1920
    private static let codecDimension = 512

    let tokenizer: Qwen3Tokenizer
    let textPad: [Float]
    let textBos: [Float]
    let textEos: [Float]
    /// The speaker embedding of each voice, by name.
    let voices: [String: [Float]]

    /// The text embedding through the text projection, (text vocabulary, 1024).
    private let textEmbeddings: Float16Table
    /// The codec embedding, (codec vocabulary, 1024).
    private let codeEmbeddings: Float16Table
    /// The speech decoder's codebooks, (16 · 2048, 512), codebook-major.
    private let speechCodebooks: Float16Table

    let prefill: InferenceFunction
    let decode: InferenceFunction
    let multiCodeDecoder: InferenceFunction
    private let speechDecoder: InferenceFunction

    init(
      directory: URL, voiceNames: [String], progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws {
      var loader = ModelLoader(directory: directory, total: 4, progress: progress)
      let package = { (name: String) in
        try await NeuralEnginePackages.specialize(
          directory.appending(path: NeuralEnginePackages.name(name)))
      }
      let codeDecoder = try await package("CodeDecoder")
      prefill = try await loader.load("prefill", of: codeDecoder)
      decode = try await loader.load("decode", of: codeDecoder)
      multiCodeDecoder = try await loader.load("main", of: try await package("MultiCodeDecoder"))
      speechDecoder = try await loader.load("main", of: try await package("SpeechDecoder"))
      tokenizer = try Qwen3Tokenizer(
        vocabulary: directory.appending(path: "vocab.json"),
        merges: directory.appending(path: "merges.txt"))
      textEmbeddings = try Float16Table(directory.appending(path: "text_embeddings.f16"), width: Self.hidden)
      codeEmbeddings = try Float16Table(directory.appending(path: "code_embeddings.f16"), width: Self.hidden)
      speechCodebooks = try Float16Table(
        directory.appending(path: "speech_codebooks.f16"), width: Self.codecDimension)
      textPad = try Self.vector(npy: directory.appending(path: "tts_pad_embed.npy"))
      textBos = try Self.vector(npy: directory.appending(path: "tts_bos_embed.npy"))
      textEos = try Self.vector(npy: directory.appending(path: "tts_eos_embed.npy"))
      let speakers = try voiceNames.map { name in
        (name, try Self.vector(npy: directory.appending(path: "voices/\(name).npy")))
      }
      voices = Dictionary(uniqueKeysWithValues: speakers)
    }

    func textEmbedding(_ token: Int) -> [Float] {
      textEmbeddings.row(token)
    }

    func codeEmbedding(_ token: Int) -> [Float] {
      codeEmbeddings.row(token)
    }

    /// The waveform of up to `frames` frames of 16 codebooks, given codebook by codebook. The
    /// decoder takes the sum of the frames' code embeddings; frames past the codes are code 0.
    func speech(codebooks: [[Int32]]) async throws -> [Float] {
      let count = codebooks.first?.count ?? 0
      let width = Self.codecDimension
      var sums = [Float16](repeating: 0, count: width * Self.frames)
      for frame in 0..<Self.frames {
        var sum = [Float](repeating: 0, count: width)
        for (codebook, codes) in codebooks.enumerated() {
          let code = frame < count ? Int(codes[frame]) : 0
          let row = speechCodebooks.row(codebook * Self.residualVocabulary + code)
          for channel in 0..<width { sum[channel] += row[channel] }
        }
        for channel in 0..<width { sums[channel * Self.frames + frame] = Float16(sum[channel]) }
      }
      let outputs = try await speechDecoder.outputs([
        "codes_sum": NDArray(scalars: sums, shape: [1, width, Self.frames])
      ])
      guard let audio = outputs["audio"] else { throw Qwen3Error.missingOutput("audio") }
      return Array(audio.scalars(as: Float16.self).prefix(count * Self.samplesPerFrame).map(Float.init))
    }

    /// The Float32 vector of an `.npy` file, which is the last `hidden` values of the file.
    static func vector(npy url: URL) throws -> [Float] {
      let data = try Data(contentsOf: url)
      let width = hidden * MemoryLayout<UInt32>.size
      guard data.count >= width else { throw Qwen3Error.invalidEmbedding(url.lastPathComponent) }
      return data.suffix(width).withUnsafeBytes { raw in
        (0..<hidden).map { index in
          Float(
            bitPattern: UInt32(
              littleEndian: raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)))
        }
      }
    }
  }

#endif
