import Foundation

/// A row-major Float16 table, mapped from its file rather than read, so that a large table costs
/// memory only for the rows read.
public struct Float16Table: Sendable {
  private let data: Data
  private let width: Int

  public init(_ url: URL, width: Int) throws {
    data = try Data(contentsOf: url, options: .alwaysMapped)
    self.width = width
  }

  public func row(_ index: Int) -> [Float] {
    data.withUnsafeBytes { raw in
      (0..<width).map { column in
        Float(
          raw.loadUnaligned(
            fromByteOffset: (index * width + column) * MemoryLayout<Float16>.size,
            as: Float16.self))
      }
    }
  }
}
