// swift-tools-version: 6.2
import PackageDescription

let package = Package(
  name: "InkearKit",
  platforms: [.iOS("27.0"), .macOS("27.0")],
  products: [
    .library(name: "PDFStructure", targets: ["PDFStructure"]),
    .library(name: "Synthesis", targets: ["Synthesis"]),
    .library(name: "Engines", targets: ["Engines"]),
    .library(name: "Library", targets: ["Library"]),
    .library(name: "Playback", targets: ["Playback"]),
  ],
  dependencies: [
    .package(url: "https://github.com/FluidInference/FluidAudio", exact: "0.17.7", traits: []),
  ],
  targets: [
    .target(name: "PDFStructure"),
    .testTarget(name: "PDFStructureTests", dependencies: ["PDFStructure"]),
    .target(name: "Synthesis"),
    .testTarget(name: "SynthesisTests", dependencies: ["Synthesis"]),
    .target(
      name: "KokoroEngine",
      dependencies: ["Synthesis", .product(name: "FluidAudio", package: "FluidAudio")]),
    .target(
      name: "PocketEngine",
      dependencies: ["Synthesis", .product(name: "FluidAudio", package: "FluidAudio")]),
    .target(name: "Qwen3Engine", dependencies: ["Synthesis"]),
    .testTarget(name: "Qwen3EngineTests", dependencies: ["Qwen3Engine"]),
    .target(name: "MossEngine", dependencies: ["Synthesis"]),
    .target(name: "MagpieEngine", dependencies: ["Synthesis"]),
    .target(name: "OmniVoiceEngine", dependencies: ["Synthesis"]),
    .target(name: "ChatterboxEngine", dependencies: ["Synthesis"]),
    // Phonon's Rust core, built from xn-ptts 0.4.2 with tools/phonon-coreml/steps.patch by its
    // ios/build-xcframework.sh, and the Swift wrapper around it.
    .binaryTarget(
      name: "PhononCore",
      url:
        "https://huggingface.co/itsmechico/phonon-core-runtime/resolve/d70123932f1de84a824b160a423d136ac9a6b248/PhononCore.xcframework.zip",
      checksum: "61cedbdd9555b08939739ffa607b0465a3baca5aa04ffe0ad8e7bf952264549a"),
    .target(
      name: "PhononTTS",
      dependencies: ["PhononCore"],
      linkerSettings: [
        .linkedFramework("CoreML"), .linkedFramework("CoreVideo"),
        .linkedFramework("Accelerate"), .linkedFramework("AVFoundation"),
      ]),
    .target(name: "PhononEngine", dependencies: ["Synthesis", "PhononTTS"]),
    .target(
      name: "Engines",
      dependencies: [
        "Synthesis", "KokoroEngine", "PocketEngine", "MossEngine", "Qwen3Engine", "MagpieEngine",
        "OmniVoiceEngine", "ChatterboxEngine", "PhononEngine",
      ]),
    .testTarget(
      name: "EnginesTests",
      dependencies: ["Engines", "Synthesis", "PhononEngine"]),
    .target(name: "Library", dependencies: ["PDFStructure", "Synthesis"]),
    .testTarget(name: "LibraryTests", dependencies: ["Library", "PDFStructure", "Synthesis"]),
    .target(name: "Playback", dependencies: ["PDFStructure", "Synthesis"]),
    .testTarget(name: "PlaybackTests", dependencies: ["Playback", "PDFStructure", "Synthesis"]),
  ]
)
