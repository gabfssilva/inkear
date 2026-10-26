import Foundation
import PDFStructure
import Synthesis

public struct DocumentID: Hashable, Sendable, Codable, CustomStringConvertible {
  public let raw: UUID

  public init(raw: UUID = UUID()) {
    self.raw = raw
  }

  public var description: String { raw.uuidString }
}

public struct ListeningSettings: Sendable, Codable, Equatable {
  /// The voice, Supertonic's "F3" by default.
  public let voice: VoiceID
  /// 1 is the voice's natural pace.
  public let speed: Double
  /// Overrides the detected language; nil means detected.
  public let language: String?
  public let quality: SpeechQuality

  public static let standard = ListeningSettings(
    voice: VoiceID(engine: .supertonic, name: "F3"), speed: 1, language: nil, quality: .normal)

  public init(voice: VoiceID, speed: Double, language: String?, quality: SpeechQuality) {
    self.voice = voice
    self.speed = speed
    self.language = language
    self.quality = quality
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    do {
      voice = try container.decode(VoiceID.self, forKey: .voice)
    } catch DecodingError.typeMismatch {
      // Libraries saved before there were other engines stored only the Supertonic voice name.
      voice = VoiceID(engine: .supertonic, name: try container.decode(String.self, forKey: .voice))
    }
    speed = try container.decode(Double.self, forKey: .speed)
    language = try container.decodeIfPresent(String.self, forKey: .language)
    // Libraries saved before quality was a setting have none.
    quality = try container.decodeIfPresent(SpeechQuality.self, forKey: .quality) ?? .normal
  }

  public func with(voice: VoiceID) -> ListeningSettings {
    ListeningSettings(voice: voice, speed: speed, language: language, quality: quality)
  }

  public func with(speed: Double) -> ListeningSettings {
    ListeningSettings(voice: voice, speed: speed, language: language, quality: quality)
  }

  public func with(language: String?) -> ListeningSettings {
    ListeningSettings(voice: voice, speed: speed, language: language, quality: quality)
  }

  public func with(quality: SpeechQuality) -> ListeningSettings {
    ListeningSettings(voice: voice, speed: speed, language: language, quality: quality)
  }
}

/// Where the listener stopped.
public struct Bookmark: Sendable, Codable, Equatable {
  public let position: ReadingPosition
  /// Index into `DocumentStructure.sections`, for "CAP. 2".
  public let section: Int?
  /// 0...1 of the document's characters before `position`.
  public let fraction: Double
  public let date: Date

  public init(position: ReadingPosition, section: Int?, fraction: Double, date: Date) {
    self.position = position
    self.section = section
    self.fraction = fraction
    self.date = date
  }
}

public struct DocumentRecord: Sendable, Codable, Equatable, Identifiable {
  public enum Status: Sendable, Codable, Equatable {
    case preparing
    case failed(reason: String)
    case ready(Overview)
  }

  public struct Overview: Sendable, Codable, Equatable {
    public let sectionCount: Int
    /// `DocumentStructure.language`.
    public let language: String?
    public let isRecognized: Bool
    /// `(mainStart, 0)`.
    public let mainStart: ReadingPosition?
    /// Title of the last section whose block is at or before `mainStart`.
    public let mainStartTitle: String?

    public init(_ structure: DocumentStructure) {
      sectionCount = structure.sections.count
      language = structure.language
      isRecognized = structure.isRecognized
      mainStart = structure.mainStart.map { ReadingPosition(block: $0, sentence: 0) }
      mainStartTitle = structure.mainStart.flatMap { mainStart in
        structure.sections.last { $0.block <= mainStart }?.title
      }
    }
  }

  public let id: DocumentID
  /// The PDF title attribute when it is not blank, else the file name without extension.
  public let title: String
  /// The PDF author attribute when it is not blank.
  public let author: String?
  public let pageCount: Int
  public let addedAt: Date
  public let status: Status
  /// nil when the document was never listened to.
  public let bookmark: Bookmark?
  public let isFinished: Bool
  public let settings: ListeningSettings

  public func with(status: Status) -> DocumentRecord {
    DocumentRecord(
      id: id, title: title, author: author, pageCount: pageCount, addedAt: addedAt,
      status: status, bookmark: bookmark, isFinished: isFinished, settings: settings)
  }

  public func with(bookmark: Bookmark?) -> DocumentRecord {
    DocumentRecord(
      id: id, title: title, author: author, pageCount: pageCount, addedAt: addedAt,
      status: status, bookmark: bookmark, isFinished: isFinished, settings: settings)
  }

  public func with(isFinished: Bool) -> DocumentRecord {
    DocumentRecord(
      id: id, title: title, author: author, pageCount: pageCount, addedAt: addedAt,
      status: status, bookmark: bookmark, isFinished: isFinished, settings: settings)
  }

  public func with(settings: ListeningSettings) -> DocumentRecord {
    DocumentRecord(
      id: id, title: title, author: author, pageCount: pageCount, addedAt: addedAt,
      status: status, bookmark: bookmark, isFinished: isFinished, settings: settings)
  }
}
