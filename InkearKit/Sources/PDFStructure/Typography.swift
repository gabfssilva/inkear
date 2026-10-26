/// Document-wide text metrics.
struct Typography {
  /// The size of body text.
  ///
  /// With font sizes, it is the size carrying the most characters, to the nearest half point.
  /// With measured sizes, it is the median size by characters.
  let bodySize: Double
  /// Whether sizes were measured from glyph boxes, as OCR yields them, rather than read from
  /// fonts. Measured sizes differ slightly from line to line, giving over a hundred distinct
  /// sizes to the tenth of a point where set type has a few dozen at most.
  let isMeasured: Bool

  init(pages: [PageText]) {
    let lines = pages.flatMap(\.lines)
    isMeasured = Set(lines.map { ($0.fontSize * 10).rounded() }).count > 40
    if isMeasured {
      let half = lines.reduce(0) { $0 + $1.text.count } / 2
      var seen = 0
      var median = 0.0
      for line in lines.sorted(by: { $0.fontSize < $1.fontSize }) {
        seen += line.text.count
        median = line.fontSize
        if seen > half { break }
      }
      bodySize = median
    } else {
      let characters = Dictionary(grouping: lines, by: { ($0.fontSize * 2).rounded() / 2 })
        .mapValues { $0.reduce(0) { $0 + $1.text.count } }
      bodySize = characters.max { $0.value < $1.value }?.key ?? 0
    }
  }

  /// `pages` with measured sizes within a quarter of the body size set to the body size, so
  /// that measurement noise does not read as changes of font.
  func steadied(_ pages: [PageText]) -> [PageText] {
    guard isMeasured else { return pages }
    return pages.map { page in
      page.with(
        lines: page.lines.map { line in
          abs(line.fontSize - bodySize) > bodySize * 0.25
            ? line
            : Line(text: line.text, bounds: line.bounds, fontSize: bodySize, isBold: line.isBold)
        })
    }
  }
}
