import Foundation

/// A step function's shape: `cond` conditional positions and `target` audio frames.
struct OmniVoiceBucket: Decodable, Equatable, Sendable {
  let function: String
  let cond: Int
  let target: Int

  var sequence: Int { cond + target }
}

/// A request's codes: per codebook the text ids, the reference codes and the masked target.
struct OmniVoiceConditioning: Sendable {
  let codes: [[Int32]]
  let audioMask: [Float16]
  let target: Int

  var length: Int { audioMask.count }
}

/// The text side of a request, from the text to the codes the step model reads.
struct OmniVoiceFrontEnd: Sendable {
  static let codebooks = 8
  static let mask: Int32 = 1024

  private static let denoise: Int32 = 151_669
  private static let langStart: Int32 = 151_670
  private static let langEnd: Int32 = 151_671
  private static let instructStart: Int32 = 151_672
  private static let instructEnd: Int32 = 151_673
  private static let textStart: Int32 = 151_674
  private static let textEnd: Int32 = 151_675

  let tokenizer: OmniVoiceTokenizer
  let duration: DurationEstimator
  let buckets: [OmniVoiceBucket]

  /// The conditioning of `text`, already normalized, spoken in `language` after `voice`'s reference.
  func conditioning(_ text: String, language: String, voice: OmniVoiceVoice) -> OmniVoiceConditioning {
    let style: [Int32] =
      [Self.denoise, Self.langStart] + tokenizer.encode(language)
      + [Self.langEnd, Self.instructStart] + tokenizer.encode("None") + [Self.instructEnd]
    let spoken: [Int32] =
      [Self.textStart] + tokenizer.encode(Self.combine(text, reference: voice.text)) + [Self.textEnd]
    let ids = style + spoken
    let target = duration.frames(text, reference: voice.text, referenceFrames: voice.tokens[0].count)
    let codes: [[Int32]] = voice.tokens.map { row in
      ids + row + Array(repeating: Self.mask, count: target)
    }
    let audioMask =
      [Float16](repeating: 0, count: ids.count)
      + [Float16](repeating: 1, count: voice.tokens[0].count + target)
    return OmniVoiceConditioning(codes: codes, audioMask: audioMask, target: target)
  }

  /// The smallest bucket that holds `conditioning`, or nil when none does.
  func bucket(for conditioning: OmniVoiceConditioning) -> OmniVoiceBucket? {
    buckets
      .filter { $0.cond >= conditioning.length && $0.target >= conditioning.target }
      .min { $0.sequence < $1.sequence }
  }

  /// The reference transcript and the text, joined by a space, with the whitespace rules of the
  /// reference pipeline: no line breaks, ASCII parentheses, single spaces, and none around CJK.
  static func combine(_ text: String, reference: String) -> String {
    let edges = CharacterSet.whitespacesAndNewlines
    let joined = "\(reference.trimmingCharacters(in: edges)) \(text.trimmingCharacters(in: edges))"
    let unbroken = String(joined.unicodeScalars.filter { $0 != "\r" && $0 != "\n" })
    let ascii = unbroken
      .replacingOccurrences(of: "（", with: "(")
      .replacingOccurrences(of: "）", with: ")")
    let spaced = ascii.replacing(/[ \t]+/, with: " ")
    return spaced.replacing(/\s+(?=[\u{4E00}-\u{9FFF}])/, with: "")
      .replacing(/([\u{4E00}-\u{9FFF}])\s+/) { String($0.output.1) }
  }
}

/// Target frames from character weights, ported from `omnivoice/utils/duration.py`
/// (Apache-2.0, Xiaomi Corp.); the tables are those of omnivoice.json.
struct DurationEstimator: Decodable, Sendable {
  struct ScriptRange: Decodable, Sendable {
    let end: UInt32
    let script: String

    init(from decoder: Decoder) throws {
      var values = try decoder.unkeyedContainer()
      end = try values.decode(UInt32.self)
      script = try values.decode(String.self)
    }
  }

  let weights: [String: Double]
  /// Sorted by `end`; a code point takes the script of the first range that reaches it.
  let ranges: [ScriptRange]
  let lowThreshold: Double
  let boostStrength: Double

  private enum CodingKeys: String, CodingKey {
    case weights
    case ranges
    case lowThreshold = "low_threshold"
    case boostStrength = "boost_strength"
  }

  /// The frames `text` takes at the pace of `reference`, which takes `referenceFrames` frames.
  func frames(_ text: String, reference: String, referenceFrames: Int) -> Int {
    let referenceWeight = totalWeight(reference)
    guard referenceFrames > 0, referenceWeight > 0 else { return 1 }
    let speed = referenceWeight / Double(referenceFrames)
    var seconds = totalWeight(text) / speed
    if seconds < lowThreshold {
      seconds = lowThreshold * pow(seconds / lowThreshold, 1 / boostStrength)
    }
    return max(1, Int(seconds))
  }

  private func totalWeight(_ text: String) -> Double {
    text.unicodeScalars.reduce(0) { $0 + characterWeight($1) }
  }

  private func characterWeight(_ scalar: Unicode.Scalar) -> Double {
    let code = scalar.value
    if code == 0x0640 { return named("mark") }
    switch scalar.properties.generalCategory {
    case .nonspacingMark, .spacingMark, .enclosingMark:
      return named("mark")
    case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
      .initialPunctuation, .finalPunctuation, .otherPunctuation, .mathSymbol, .currencySymbol,
      .modifierSymbol, .otherSymbol:
      return named("punctuation")
    case .spaceSeparator, .lineSeparator, .paragraphSeparator:
      return named("space")
    case .decimalNumber, .letterNumber, .otherNumber:
      return named("digit")
    default:
      break
    }
    if let range = ranges.first(where: { $0.end >= code }) {
      return named(range.script)
    }
    return code > 0x20000 ? named("cjk") : named("default")
  }

  private func named(_ key: String) -> Double {
    weights[key, default: 1]
  }
}
