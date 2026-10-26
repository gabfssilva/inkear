import CoreText
import PDFKit

/// Builds PDFs from explicitly placed lines, so tests control layout and fonts exactly.
struct SyntheticPDF {
  struct Text {
    let string: String
    let x: CGFloat
    let y: CGFloat
    var font = "Times-Roman"
    var size: CGFloat = 11
  }

  var pageSize = CGSize(width: 400, height: 600)

  func document(pages: [[Text]]) -> PDFDocument? {
    let data = NSMutableData()
    var box = CGRect(origin: .zero, size: pageSize)
    guard
      let consumer = CGDataConsumer(data: data),
      let context = CGContext(consumer: consumer, mediaBox: &box, nil)
    else { return nil }
    for texts in pages {
      context.beginPDFPage(nil)
      for text in texts {
        let font = CTFontCreateWithName(text.font as CFString, text.size, nil)
        let line = CTLineCreateWithAttributedString(
          NSAttributedString(string: text.string, attributes: [.font: font]))
        context.textPosition = CGPoint(x: text.x, y: text.y)
        CTLineDraw(line, context)
      }
      context.endPDFPage()
    }
    context.closePDF()
    return PDFDocument(data: data as Data)
  }
}
