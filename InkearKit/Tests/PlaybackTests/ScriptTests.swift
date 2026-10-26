import PDFStructure
import Testing

@testable import Playback

@Suite struct ScriptTests {
  @Test func startHonorsMainStart() {
    let blocks = ["Capa. Título.", "Um. Dois.", "", "Três. Quatro."]

    #expect(makeScript(blocks, mainStart: 3).start == position(3, 0))
    #expect(makeScript(blocks, mainStart: nil).start == position(0, 0))
    #expect(makeScript(blocks, mainStart: 2).start == position(3, 0))
  }

  @Test func startIsTheFirstSentenceWhenMainStartHasNoText() {
    #expect(makeScript(["", "Um. Dois."], mainStart: 0).start == position(1, 0))
    #expect(makeScript(["Um. Dois.", ""], mainStart: 9).start == position(0, 0))
  }

  @Test func nextSkipsBlocksWithoutSentences() {
    let script = makeScript(["Um. Dois.", "", "Três."])

    #expect(script.next(after: position(0, 0)) == position(0, 1))
    #expect(script.next(after: position(0, 1)) == position(2, 0))
    #expect(script.next(after: position(2, 0)) == nil)
  }

  @Test func runGathersSentencesOfTheBlockWithinTheCharacters() {
    let script = makeScript(["Um. Dois. Três.", "Quatro."])

    #expect(script.run(from: position(0, 0), characters: 0) == [position(0, 0)])
    #expect(script.run(from: position(0, 0), characters: 8) == [position(0, 0), position(0, 1)])
    #expect(script.run(from: position(0, 0), characters: 12) == [position(0, 0), position(0, 1)])
    #expect(
      script.run(from: position(0, 0), characters: 100)
        == [position(0, 0), position(0, 1), position(0, 2)])
    #expect(script.run(from: position(0, 2), characters: 100) == [position(0, 2)])
    #expect(script.run(from: position(1, 0), characters: 100) == [position(1, 0)])
  }

  @Test func previousSkipsBlocksWithoutSentences() {
    let script = makeScript(["Um. Dois.", "", "Três."])

    #expect(script.previous(before: position(2, 0)) == position(0, 1))
    #expect(script.previous(before: position(0, 1)) == position(0, 0))
    #expect(script.previous(before: position(0, 0)) == nil)
  }

  @Test func previousIsStrictlyBeforeAPositionWithoutSentences() {
    let script = makeScript(["Um. Dois.", "", "Três."])

    #expect(script.previous(before: position(1, 0)) == position(0, 1))
  }

  @Test func navigationWithoutSentencesReturnsNil() {
    let script = makeScript(["", "  "])

    #expect(script.isEmpty)
    #expect(script.start == .start)
    #expect(script.next(after: .start) == nil)
    #expect(script.previous(before: .start) == nil)
    #expect(script.text(at: .start) == nil)
  }

  @Test func sentencesAreAddressedByBlockAndIndex() throws {
    let script = makeScript(["Um. Dois. Três.", ""])

    #expect(script.sentences(inBlock: 0).count == 3)
    #expect(script.sentences(inBlock: 1).isEmpty)
    #expect(script.sentences(inBlock: 7).isEmpty)
    #expect(try #require(script.text(at: position(0, 2))).hasPrefix("Três"))
    #expect(script.text(at: position(0, 3)) == nil)
    #expect(script.text(at: position(1, 0)) == nil)
  }

  @Test func headingsNarrateAsSeparateSentences() throws {
    let script = makeScript(
      ["Capítulo 1 — O menino que sentia o chão", "Havia uma vila."],
      headings: [0])

    #expect(script.sentences(inBlock: 0).count == 2)
    #expect(try #require(script.text(at: position(0, 0))) == "Capítulo 1.")
    #expect(try #require(script.text(at: position(0, 1))) == "O menino que sentia o chão.")
    #expect(script.isHeading(0))
    #expect(!script.isHeading(1))
  }

  @Test func headingsWithoutDashAreOneSentenceEndedByAPeriod() throws {
    let script = makeScript(["Índice", "Um. Dois."], headings: [0])

    #expect(script.sentences(inBlock: 0).count == 1)
    #expect(try #require(script.text(at: position(0, 0))) == "Índice.")
  }

  @Test func clampedMovesToTheSentenceAtOrBeforeThePosition() {
    let script = makeScript(["Um. Dois.", "", "Três."])

    #expect(script.clamped(position(0, 1)) == position(0, 1))
    #expect(script.clamped(position(1, 0)) == position(0, 1))
    #expect(script.clamped(position(9, 0)) == position(2, 0))
    #expect(script.clamped(position(0, 5)) == position(0, 1))
  }

  @Test func sectionContainingIsTheLastSectionStartedBeforeThePosition() {
    let script = makeScript(
      ["Um. Dois.", "Três.", "Quatro. Cinco.", ""],
      sections: [section("A", block: 0), section("B", block: 2)])

    #expect(script.section(containing: position(0, 0)) == 0)
    #expect(script.section(containing: position(1, 0)) == 0)
    #expect(script.section(containing: position(2, 1)) == 1)
    #expect(script.section(containing: position(3, 0)) == 1)
  }

  @Test func frontMatterBeforeTheFirstSectionHasNoSection() {
    let script = makeScript(["Capa.", "Um. Dois."], sections: [section("A", block: 1)])

    #expect(script.section(containing: position(0, 0)) == nil)
    #expect(script.section(containing: position(1, 0)) == 0)
  }

  @Test func sectionStartIsTheFirstSentenceAtOrAfterItsBlock() {
    let script = makeScript(
      ["Um. Dois.", "", "Três.", "Quatro."],
      sections: [section("A", block: 0), section("B", block: 1)])

    #expect(script.start(ofSection: 0) == position(0, 0))
    #expect(script.start(ofSection: 1) == position(2, 0))
  }

  @Test func nextSectionIsTheStartOfTheFollowingSection() {
    let script = makeScript(
      ["Um. Dois.", "Três.", "Quatro. Cinco."],
      sections: [section("A", block: 0), section("B", block: 2)])

    #expect(script.nextSection(after: position(0, 1)) == position(2, 0))
    #expect(script.nextSection(after: position(1, 0)) == position(2, 0))
    #expect(script.nextSection(after: position(2, 1)) == nil)
  }

  @Test func nextSectionFromFrontMatterIsTheFirstSection() {
    let script = makeScript(["Capa.", "Um."], sections: [section("A", block: 1)])

    #expect(script.nextSection(after: position(0, 0)) == position(1, 0))
  }

  @Test func previousSectionRestartsTheCurrentSectionWhenPastItsStart() {
    let script = makeScript(
      ["Um. Dois.", "Três.", "Quatro. Cinco."],
      sections: [section("A", block: 0), section("B", block: 2)])

    #expect(script.previousSection(before: position(2, 1)) == position(2, 0))
  }

  @Test func previousSectionGoesToThePreviousSectionAtItsStart() {
    let script = makeScript(
      ["Um. Dois.", "Três.", "Quatro. Cinco."],
      sections: [section("A", block: 0), section("B", block: 2)])

    #expect(script.previousSection(before: position(2, 0)) == position(0, 0))
    #expect(script.previousSection(before: position(0, 1)) == position(0, 0))
    #expect(script.previousSection(before: position(0, 0)) == nil)
  }

  @Test func previousSectionFromFrontMatterIsNil() {
    let script = makeScript(["Capa.", "Um."], sections: [section("A", block: 1)])

    #expect(script.previousSection(before: position(0, 0)) == nil)
  }

  @Test func positionOnPageSkipsBlocksWithoutSentences() {
    let script = makeScript(
      ["Um. Dois.", "", "Três.", "Quatro."], pages: [0, 1, 2, 2])

    #expect(script.position(onPage: 0) == position(0, 0))
    #expect(script.position(onPage: 1) == position(2, 0))
    #expect(script.position(onPage: 2) == position(2, 0))
    #expect(script.position(onPage: 5) == nil)
  }

  @Test func fractionStartsAtZeroAndIncreases() {
    let script = makeScript(["Um. Dois. Três.", "Quatro.", "", "Cinco. Seis. Sete."])

    var positions = [position(0, 0)]
    while let next = script.next(after: positions[positions.count - 1]) {
      positions.append(next)
    }
    let fractions = positions.map(script.fraction(of:))

    #expect(fractions.first == 0)
    #expect(zip(fractions, fractions.dropFirst()).allSatisfy { $0 < $1 })
    #expect(fractions.allSatisfy { (0..<1).contains($0) })
  }

  @Test func fractionIsCharactersBeforeTheSentenceOverTheTotal() throws {
    let script = makeScript(["Um. Dois.", "Três."])
    let lengths = try [
      #require(script.text(at: position(0, 0))).count,
      #require(script.text(at: position(0, 1))).count,
      #require(script.text(at: position(1, 0))).count,
    ]
    let total = lengths.reduce(0, +)

    #expect(script.fraction(of: position(0, 0)) == 0)
    #expect(script.fraction(of: position(0, 1)) == Double(lengths[0]) / Double(total))
    #expect(script.fraction(of: position(1, 0)) == Double(lengths[0] + lengths[1]) / Double(total))
  }

  @Test func sectionCharactersCountsOnlyTheCurrentSection() throws {
    let script = makeScript(
      ["Um. Dois.", "Três.", "Quatro. Cinco."],
      sections: [section("A", block: 0), section("B", block: 2)])
    let lengths = try [
      #require(script.text(at: position(0, 0))).count,
      #require(script.text(at: position(0, 1))).count,
      #require(script.text(at: position(1, 0))).count,
    ]

    let inside = script.sectionCharacters(at: position(1, 0))
    #expect(inside.before == lengths[0] + lengths[1])
    #expect(inside.total == lengths[0] + lengths[1] + lengths[2])
    #expect(script.sectionCharacters(at: position(0, 0)).before == 0)
    #expect(script.sectionCharacters(at: position(2, 0)).before == 0)
    #expect(
      script.sectionCharacters(at: position(2, 0)).total
        == script.sectionCharacters(at: position(2, 1)).total)
  }

  @Test func documentWithoutSectionsIsOneSection() throws {
    let script = makeScript(["Um. Dois.", "Três."])
    let lengths = try [
      #require(script.text(at: position(0, 0))).count,
      #require(script.text(at: position(0, 1))).count,
      #require(script.text(at: position(1, 0))).count,
    ]

    #expect(script.section(containing: position(1, 0)) == nil)
    #expect(script.nextSection(after: .start) == nil)
    #expect(script.previousSection(before: position(1, 0)) == nil)
    let at = script.sectionCharacters(at: position(1, 0))
    #expect(at.before == lengths[0] + lengths[1])
    #expect(at.total == lengths.reduce(0, +))
  }

  private func makeScript(
    _ texts: [String], sections: [Section] = [], mainStart: Int? = nil, pages: [Int]? = nil,
    headings: [Int] = []
  ) -> Script {
    let pages = pages ?? texts.indices.map { $0 }
    let blocks = zip(texts.indices, zip(texts, pages)).map { index, pair in
      let (text, page) = pair
      return Block(
        kind: headings.contains(index) ? .heading(level: 1) : .paragraph,
        text: text, pages: page...page)
    }
    return Script(
      DocumentStructure(
        pageCount: (pages.max() ?? 0) + 1, blocks: blocks, sections: sections,
        contentsStart: nil, mainStart: mainStart))
  }

  private func section(_ title: String, block: Int) -> Section {
    Section(title: title, level: 1, block: block)
  }

  private func position(_ block: Int, _ sentence: Int) -> ReadingPosition {
    ReadingPosition(block: block, sentence: sentence)
  }
}
