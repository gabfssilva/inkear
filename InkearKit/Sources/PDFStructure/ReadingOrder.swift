import CoreGraphics

/// Puts a page's lines in reading order.
///
/// Fragments on the same row of the same column are joined into one line. In a two-column
/// layout, lines that cross the gutter (titles, wide figures) split the page into bands, and
/// each band is read left column first.
enum ReadingOrder {
  static func lines(of page: PageText) -> [Line] {
    guard let gutter = gutter(of: page.lines) else { return rows(page.lines) }
    var ordered: [Line] = []
    var left: [Line] = []
    var right: [Line] = []
    func flush() {
      ordered += rows(left) + rows(right)
      left = []
      right = []
    }
    for line in page.lines.sorted(by: { $0.bounds.maxY > $1.bounds.maxY }) {
      if line.bounds.maxX <= gutter.lowerBound {
        left.append(line)
      } else if line.bounds.minX >= gutter.upperBound {
        right.append(line)
      } else {
        flush()
        ordered.append(line)
      }
    }
    flush()
    return ordered
  }

  /// Joins lines sharing a row into one line each, and orders rows top to bottom.
  static func rows(_ lines: [Line]) -> [Line] {
    var rows: [[Line]] = []
    for line in lines.sorted(by: { $0.bounds.maxY > $1.bounds.maxY }) {
      if let last = rows.last?.last, sameRow(last, line) {
        rows[rows.count - 1].append(line)
      } else {
        rows.append([line])
      }
    }
    return rows.map(join)
  }

  /// The vertical strip separating two columns, if the lines form two columns.
  ///
  /// The strip is the widest one in the middle of the text that few lines cross: a title or
  /// an author list centered above the columns may cross it, and reads as a band of its own.
  static func gutter(of lines: [Line]) -> ClosedRange<CGFloat>? {
    guard let left = lines.map(\.bounds.minX).min(), let right = lines.map(\.bounds.maxX).max()
    else {
      return nil
    }
    let width = right - left
    let narrow = lines.filter { $0.bounds.width < width * 0.6 }
    guard narrow.count >= 6 else { return nil }
    let crossing = { (x: Int) in
      narrow.count {
        $0.bounds.minX.rounded(.down) <= CGFloat(x) && $0.bounds.maxX.rounded(.up) >= CGFloat(x)
      }
    }
    let tolerated = narrow.count / 10
    var widest = 0..<0
    var run = 0..<0
    for x in Int(left + width * 0.3)...Int(left + width * 0.7) {
      run = crossing(x) > tolerated ? (x + 1)..<(x + 1) : run.lowerBound..<(x + 1)
      if run.count > widest.count { widest = run }
    }
    guard widest.count >= 6 else { return nil }
    let gutter = CGFloat(widest.lowerBound)...CGFloat(widest.upperBound)
    let sides = (
      left: narrow.count { $0.bounds.maxX <= gutter.lowerBound },
      right: narrow.count { $0.bounds.minX >= gutter.upperBound }
    )
    return min(sides.left, sides.right) >= max(3, narrow.count / 5) ? gutter : nil
  }

  private static func sameRow(_ a: Line, _ b: Line) -> Bool {
    let overlap = min(a.bounds.maxY, b.bounds.maxY) - max(a.bounds.minY, b.bounds.minY)
    return overlap > min(a.bounds.height, b.bounds.height) * 0.5
  }

  private static func join(_ row: [Line]) -> Line {
    guard row.count > 1 else { return row[0] }
    let pieces = row.sorted { $0.bounds.minX < $1.bounds.minX }
    let main = pieces.max { $0.text.count < $1.text.count } ?? pieces[0]
    // A wide gap inside a row, as between a contents entry and its page number, becomes a tab.
    let text = zip(pieces, pieces.dropFirst()).reduce(pieces[0].text) { text, pair in
      text + (pair.1.bounds.minX - pair.0.bounds.maxX > main.fontSize * 2 ? "\t" : " ")
        + pair.1.text
    }
    return Line(
      text: text,
      bounds: pieces.dropFirst().reduce(pieces[0].bounds) { $0.union($1.bounds) },
      fontSize: main.fontSize,
      isBold: main.isBold
    )
  }
}
