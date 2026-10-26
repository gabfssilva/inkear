#if canImport(CoreAI)
  import CoreAI
  import Foundation

  /// Core AI packages that `coreai-aot` compiled ahead of time for the Neural Engine of each
  /// iPhone architecture, named `<package>.<architecture>.aimodelc`.
  ///
  /// A package loads with the compute unit it was compiled for, the Neural Engine; otherwise the
  /// device compiles it again, for minutes. Core AI keys what it compiles by the package's
  /// content, so the compilation outlives app updates, which move the app's container.
  public enum NeuralEnginePackages {
    /// The architectures `coreai-aot` compiles for: the A17 Pro and later.
    public static let architectures: Set<String> = ["h16p", "h17g", "h17p", "h18p", "h19p"]

    /// The directory name of `package` compiled for this device.
    public static func name(_ package: String) -> String {
      "\(package).\(AIModel.deviceArchitectureName).aimodelc"
    }

    /// Throws when no package was compiled for this device's architecture.
    public static func requireSupportedDevice() throws {
      let architecture = AIModel.deviceArchitectureName
      guard architectures.contains(architecture) else {
        throw NeuralEnginePackageError.unsupportedDevice(architecture)
      }
    }

    /// Specializes the model at `url` for the Neural Engine.
    ///
    /// The default cache policy lets the system purge the result, and specializing again can take
    /// minutes: on iOS 27.0.1 the Neural Engine compiler crashes on the ahead-of-time binary, and
    /// MPSGraph then compiles the model from its graph. ``deleteCompiled(_:in:)`` deletes it.
    public static func specialize(_ url: URL) async throws -> AIModel {
      try await AIModel.specialize(
        contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine),
        cachePolicy: .persistent)
    }

    /// Deletes what Core AI compiled from `packages` in `directory`, before the packages are
    /// deleted or overwritten. Core AI finds the entries by a package's hash file, so a package
    /// whose download did not finish has none.
    public static func deleteCompiled(_ packages: [String], in directory: URL) throws {
      for package in packages {
        let url = directory.appending(path: name(package))
        guard FileManager.default.fileExists(atPath: url.appending(path: "main.hash").path()) else {
          continue
        }
        try AIModelCache.default.deleteEntries(for: url)
      }
    }
  }

  public enum NeuralEnginePackageError: Error, Equatable {
    /// No package was compiled for the device's architecture.
    case unsupportedDevice(String)
  }

  extension NeuralEnginePackageError: LocalizedError {
    public var errorDescription: String? { "Neural Engine packages: \(self)" }
  }

  extension ModelLoader {
    /// Loads `function` of `model`, which ``NeuralEnginePackages/specialize(_:)`` returned.
    public mutating func load(_ function: String, of model: AIModel) async throws
      -> InferenceFunction
    {
      try await load {
        guard let loaded = try model.loadFunction(named: function) else {
          throw InferenceError.missingFunction(function)
        }
        return loaded
      }
    }
  }

  extension InferenceFunction {
    /// Runs the function and returns its outputs by name.
    public func outputs(_ inputs: [String: NDArray]) async throws -> [String: NDArray] {
      var outputs = try await run(inputs: inputs)
      var arrays: [String: NDArray] = [:]
      for name in descriptor.outputNames {
        guard let array = outputs.remove(name)?.ndArray else {
          throw InferenceError.missingOutput(name)
        }
        arrays[name] = array
      }
      return arrays
    }
  }

  public enum InferenceError: Error, Equatable {
    case missingFunction(String)
    case missingOutput(String)
  }

  extension InferenceError: LocalizedError {
    public var errorDescription: String? { "Core AI: \(self)" }
  }

  extension NDArray {
    /// The elements in row-major order, read at the strides the array reports: the Neural Engine
    /// may pad the rows of an output.
    public func scalars<T: BitwiseCopyable>(as _: T.Type = T.self) -> [T] {
      view(as: T.self).withUnsafePointer { pointer, shape, strides in
        rowMajorOffsets(
          shape: (0..<shape.count).map { shape[$0] },
          strides: (0..<strides.count).map { strides[$0] }
        ).map { pointer[$0] }
      }
    }
  }

  /// The offsets of the elements of `shape` in row-major order, with `strides` between elements.
  public func rowMajorOffsets(shape: [Int], strides: [Int]) -> [Int] {
    shape.indices.reduce([0]) { offsets, dimension in
      offsets.flatMap { offset in
        (0..<shape[dimension]).map { offset + $0 * strides[dimension] }
      }
    }
  }
#endif
