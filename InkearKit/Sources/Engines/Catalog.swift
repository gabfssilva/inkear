import ChatterboxEngine
import KokoroEngine
import MagpieEngine
import MossEngine
import OmniVoiceEngine
import PhononEngine
import PocketEngine
import Qwen3Engine
import Synthesis

#if canImport(CoreAI)
  /// The engines the app can download, in the order the app lists them.
  public let installableEngines: [any InstallableEngine.Type] = [
    Kokoro.self, PocketTTS.self, MossNano.self, Qwen3TTS.self, MagpieTTS.self,
    OmniVoice.self, Chatterbox.self, PhononEngine.self,
  ]
#else
  /// The engines the app can download, in the order the app lists them, less those that run
  /// with Core AI, which the iOS Simulator SDK does not have.
  public let installableEngines: [any InstallableEngine.Type] = [
    Kokoro.self, PocketTTS.self, MossNano.self, PhononEngine.self,
  ]
#endif
