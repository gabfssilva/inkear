import Foundation
import os

/// The cache of the models Core ML compiled for the Neural Engine.
///
/// Core ML keys a compiled model by the path of its `.mlmodelc`, and the app's containers move
/// with every install or update, so the entries of an earlier install are never read again.
enum CompiledModelCache {
  /// Deletes the cache when the app runs from a new location; call it before any model loads.
  static func clearIfReinstalled() {
    let location = Bundle.main.bundleURL.path
    let defaults = UserDefaults.standard
    guard defaults.string(forKey: locationKey) != location else { return }
    if let identifier = Bundle.main.bundleIdentifier {
      let cache = URL.cachesDirectory.appending(path: identifier)
        .appending(path: "com.apple.e5rt.e5bundlecache")
      do {
        try FileManager.default.removeItem(at: cache)
        log.info("deleted the compiled models of an earlier install")
      } catch CocoaError.fileNoSuchFile {
      } catch {
        log.error("cannot delete \(cache.path(), privacy: .public): \(error, privacy: .public)")
      }
    }
    defaults.set(location, forKey: locationKey)
  }

  private static let locationKey = "compiledModelCacheLocation"
  private static let log = Logger(subsystem: "inkear", category: "models")
}
