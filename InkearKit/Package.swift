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
    .package(url: "https://github.com/gradium-ai/xn-ptts.git", from: "0.4.2"),
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
    .target(
      name: "PhononEngine",
      dependencies: ["Synthesis", .product(name: "ptts", package: "xn-ptts")]),
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
