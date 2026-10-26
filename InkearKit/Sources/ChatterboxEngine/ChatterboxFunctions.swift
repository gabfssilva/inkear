#if canImport(CoreAI)
  import CoreAI
  import Foundation
  import Synthesis

  /// The Core AI functions of the bundle, one per size in the configuration, all loaded on the
  /// Neural Engine up front.
  struct ChatterboxFunctions: Sendable {
    /// The packages, which `coreai-aot` compiled for each iPhone architecture.
    static let packages = ["t3", "flow_encoder", "flow_step", "f0", "hift"]

    private let functions: [String: InferenceFunction]

    init(
      directory: URL, config: ChatterboxConfig,
      progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws {
      let frames = config.vocoder.frames
      let functions =
        config.t3.prefillLengths.map { ("t3", "prefill_\($0)") }
        + config.t3.decodeWindows.map { ("t3", "decode_\($0)") }
        + config.flow.tokens.map { ("flow_encoder", "n\($0)") }
        + config.flow.tokens.map {
          ("flow_step", "m\(ChatterboxBundle.melFramesPerToken * $0)")
        }
        + [("f0", "t\(frames)"), ("hift", "t\(frames)")]
      var loader = ModelLoader(directory: directory, total: functions.count, progress: progress)
      var models: [String: AIModel] = [:]
      var loaded: [String: InferenceFunction] = [:]
      for (package, function) in functions {
        let model: AIModel
        if let specialized = models[package] {
          model = specialized
        } else {
          model = try await NeuralEnginePackages.specialize(
            directory.appending(path: NeuralEnginePackages.name(package)))
          models[package] = model
        }
        loaded["\(package)/\(function)"] = try await loader.load(function, of: model)
      }
      self.functions = loaded
    }

    /// Runs `function` of `package` and returns its outputs by name.
    func run(_ package: String, _ function: String, _ inputs: [String: NDArray]) async throws
      -> [String: NDArray]
    {
      guard let loaded = functions["\(package)/\(function)"] else {
        throw ChatterboxError.invalidConfiguration("no function \(function) in \(package)")
      }
      return try await loaded.outputs(inputs)
    }

    /// Runs `function` of `package` and returns its one `output` as Float16, row-major.
    func run(
      _ package: String, _ function: String, _ inputs: [String: NDArray], output: String
    ) async throws -> [Float16] {
      guard let array = try await run(package, function, inputs)[output] else {
        throw ChatterboxError.missingOutput(output)
      }
      return array.scalars(as: Float16.self)
    }
  }
#endif
