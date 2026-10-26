import CoreGraphics
import PDFKit
import Vision

extension PageText {
  /// `extracted` with its lines recognized from a rendering of `page`, for pages without a
  /// usable text layer.
  ///
  /// Recognized lines carry no font information: their font size is the height of their bounds,
  /// and they are never bold.
  static func recognize(_ page: PDFPage, replacing extracted: PageText) async throws -> PageText {
    let bounds = page.bounds(for: .cropBox)
    guard let image = render(page, bounds: bounds, scale: 3) else {
      return extracted.with(lines: [])
    }
    let observations = try await RecognizeDocumentsRequest().perform(on: image)
    let lines = observations.flatMap(\.document.text.lines).map { observation in
      let box = observation.boundingBox.toImageCoordinates(bounds.size, origin: .lowerLeft)
        .offsetBy(dx: bounds.minX, dy: bounds.minY)
      return Line(
        text: observation.transcript, bounds: box, fontSize: Double(box.height), isBold: false)
    }
    return extracted.with(lines: lines)
  }

  private static func render(_ page: PDFPage, bounds: CGRect, scale: CGFloat) -> CGImage? {
    let size = CGSize(width: bounds.width * scale, height: bounds.height * scale)
    guard
      let context = CGContext(
        data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8,
        bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { return nil }
    context.setFillColor(CGColor(gray: 1, alpha: 1))
    context.fill(CGRect(origin: .zero, size: size))
    context.scaleBy(x: scale, y: scale)
    page.draw(with: .cropBox, to: context)
    return context.makeImage()
  }
}
