import XCTest

/// Imports a book, waits for its analysis, listens to it and opens the sheets, saving a
/// screenshot of each screen to `INKEAR_SHOTS` when set (pass `TEST_RUNNER_INKEAR_SHOTS` to
/// `xcodebuild`).
final class ListeningFlowTests: XCTestCase {
  private static let corpus = URL(filePath: #filePath).deletingLastPathComponent()
    .appending(path: "../../corpus").standardized

  override func setUp() {
    continueAfterFailure = false
  }

  @MainActor
  func testImportsPreparesAndListensToABook() throws {
    let app = launch(language: "en", importing: "pt-book-dom-casmurro.pdf")
    save(app, "en-1-preparing")

    let listen = app.buttons["Listen"]
    XCTAssertTrue(listen.waitForExistence(timeout: 180), "the analysis did not finish")
    save(app, "en-2-ready")
    listen.tap()
    XCTAssertTrue(listen.waitForNonExistence(timeout: 10))
    save(app, "en-13-loading")

    setControls(app, visible: true)
    XCTAssertTrue(app.buttons["Pause"].waitForExistence(timeout: 120), "playback did not start")
    let elapsed = app.staticTexts["elapsed"]
    let start = elapsed.label
    XCTAssertTrue(
      waitUntil(timeout: 120) { elapsed.label != start }, "the reading position did not advance")
    save(app, "en-3-player")

    let beforeBackground = seconds(elapsed.label)
    XCUIDevice.shared.press(.home)
    pause(for: 20)
    app.activate()
    XCTAssertTrue(elapsed.waitForExistence(timeout: 10))
    XCTAssertGreaterThanOrEqual(
      seconds(elapsed.label) - beforeBackground, 8, "reading did not continue in the background")

    XCUIDevice.shared.appearance = .dark
    app.buttons["Reading"].tap()
    app.buttons["Focus"].tap()
    app.buttons["Close"].tap()
    setControls(app, visible: false)
    save(app, "en-8-dark-focus")
    setControls(app, visible: true)
    app.buttons["Reading"].tap()
    app.buttons["Text"].tap()
    app.buttons["Close"].tap()
    setControls(app, visible: false)
    save(app, "en-9-dark-text")
    setControls(app, visible: true)
    XCUIDevice.shared.appearance = .light

    app.buttons["Contents"].tap()
    XCTAssertTrue(app.buttons["Chapters"].waitForExistence(timeout: 10))
    save(app, "en-4-contents")
    app.buttons["Pages"].tap()
    save(app, "en-5-pages")
    app.swipeDown(velocity: .fast)

    app.buttons["Reading"].tap()
    XCTAssertTrue(app.staticTexts["Voice"].waitForExistence(timeout: 10))
    save(app, "en-6-voice")
    // At half height, the sheet cuts the quality row, and its menu then misses the tap.
    app.scrollViews.containing(.staticText, identifier: "Voice").firstMatch.swipeUp()
    let quality = { (value: String) in
      app.buttons.containing(NSPredicate(format: "label BEGINSWITH 'Quality' AND label ENDSWITH %@", value))
        .firstMatch
    }
    quality("Normal").tap()
    app.buttons["High"].tap()
    XCTAssertTrue(quality("High").waitForExistence(timeout: 10), "the quality did not change")

    // Downloads Kokoro, about 140 MB, unless an earlier run left it installed.
    let modelMenu = app.buttons.containing(NSPredicate(format: "label BEGINSWITH 'Model'")).firstMatch
    modelMenu.tap()
    app.buttons["Download models"].tap()
    let download = app.buttons["download-kokoro"]
    if download.waitForExistence(timeout: 10) { download.tap() }
    XCTAssertTrue(
      app.descendants(matching: .any)["downloaded-kokoro"].waitForExistence(timeout: 600),
      "Kokoro did not download")
    save(app, "en-10-models")
    app.buttons["Back"].tap()
    app.buttons["Reading"].tap()
    modelMenu.tap()
    app.buttons.containing(NSPredicate(format: "label BEGINSWITH 'Kokoro'")).firstMatch.tap()
    XCTAssertTrue(app.buttons["pf_dora"].waitForExistence(timeout: 120), "Kokoro did not load")
    save(app, "en-11-kokoro")
    XCTAssertFalse(
      app.buttons.containing(NSPredicate(format: "label BEGINSWITH 'Quality'")).firstMatch.exists,
      "the quality shows for Kokoro, which ignores it")
    // The first swipe may only raise the sheet to full height.
    let sheet = app.scrollViews.containing(.button, identifier: "pf_dora").firstMatch
    let lastTheme = app.buttons["Orange"]
    for _ in 0..<3 where !lastTheme.isHittable {
      sheet.swipeUp()
    }
    XCTAssertTrue(lastTheme.isHittable, "the end of the sheet is out of reach")
    save(app, "en-12-voice-end")
    app.buttons["Close"].tap()
    let switched = elapsed.label
    XCTAssertTrue(
      waitUntil(timeout: 120) { elapsed.label != switched }, "Kokoro did not read on")

    app.buttons["Pause"].tap()
    XCTAssertTrue(app.buttons["Play"].waitForExistence(timeout: 10))
    app.buttons["Library"].tap()
    app.buttons["Library"].firstMatch.tap()
    XCTAssertTrue(app.buttons["Play"].waitForExistence(timeout: 10), "no mini player")
    save(app, "en-7-library")

    app.terminate()
    let reopened = XCUIApplication()
    reopened.launch()
    reopened.buttons["Models"].tap()
    XCTAssertTrue(
      reopened.descendants(matching: .any)["downloaded-kokoro"].waitForExistence(timeout: 10),
      "Kokoro is not installed after reopening the app")

    reopened.buttons["delete-kokoro"].tap()
    reopened.buttons["Delete"].tap()
    XCTAssertTrue(
      reopened.buttons["download-kokoro"].waitForExistence(timeout: 10), "Kokoro was not deleted")
  }

  @MainActor
  func testRecognizesAScanInPortuguese() throws {
    let app = launch(language: "pt-BR", importing: "pt-scan-esau-e-jacob.pdf")
    XCTAssertTrue(
      app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH 'página '")).firstMatch
        .waitForExistence(timeout: 60), "no recognition progress")
    save(app, "pt-1-recognizing")
    let listen = app.buttons["Ouvir"]
    XCTAssertTrue(listen.waitForExistence(timeout: 300), "the analysis did not finish")
    save(app, "pt-2-ready")
    app.buttons["Biblioteca"].firstMatch.tap()
    save(app, "pt-3-library")
  }

  /// Listens for two minutes with the app open and five from the home screen, for Instruments
  /// to record on a device (Phase 4 of PLAN.md). The PDF has to be in the app's Documents.
  @MainActor
  func testListensInTheForegroundAndInTheBackground() throws {
    #if targetEnvironment(simulator)
      throw XCTSkip("measures a device")
    #else
      let app = launch(language: "en", importing: "pt-book-dom-casmurro.pdf")
      let listen = app.buttons["Listen"]
      XCTAssertTrue(listen.waitForExistence(timeout: 600), "the analysis did not finish")
      listen.tap()
      XCTAssertTrue(listen.waitForNonExistence(timeout: 10))
      setControls(app, visible: true)
      XCTAssertTrue(app.buttons["Pause"].waitForExistence(timeout: 120), "playback did not start")
      let elapsed = app.staticTexts["elapsed"]

      pause(for: 120)
      let beforeBackground = seconds(elapsed.label)
      XCUIDevice.shared.press(.home)
      pause(for: 300)
      app.activate()
      XCTAssertTrue(elapsed.waitForExistence(timeout: 10))
      XCTAssertGreaterThanOrEqual(
        seconds(elapsed.label) - beforeBackground, 290, "reading stalled in the background")
      app.buttons["Pause"].tap()
    #endif
  }

  @MainActor
  private func launch(language: String, importing file: String) -> XCUIApplication {
    let app = XCUIApplication()
    app.launchArguments = [
      "-AppleLanguages", "(\(language))", "-AppleLocale", language.replacing("-", with: "_"),
    ]
    #if targetEnvironment(simulator)
      app.launchEnvironment["INKEAR_IMPORT"] = Self.corpus.appending(path: file).path()
    #else
      app.launchEnvironment["INKEAR_IMPORT"] = file
    #endif
    app.launch()
    return app
  }

  /// Taps the middle of the player, which shows its controls or hides them.
  @MainActor
  private func setControls(_ app: XCUIApplication, visible: Bool) {
    let contents = app.buttons["Contents"]
    guard contents.exists != visible else { return }
    app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
    let changed =
      visible ? contents.waitForExistence(timeout: 10) : contents.waitForNonExistence(timeout: 10)
    XCTAssertTrue(changed, "the controls did not \(visible ? "show" : "hide")")
  }

  @MainActor
  private func waitUntil(timeout: TimeInterval, _ condition: @escaping () -> Bool) -> Bool {
    let predicate = NSPredicate { _, _ in condition() }
    let expectation = XCTNSPredicateExpectation(predicate: predicate, object: nil)
    return XCTWaiter.wait(for: [expectation], timeout: timeout) == .completed
  }

  /// Seconds in a clock label such as "1:05".
  private func seconds(_ clock: String) -> Int {
    clock.split(separator: ":").compactMap { Int($0) }.reduce(0) { $0 * 60 + $1 }
  }

  private func pause(for seconds: TimeInterval) {
    _ = XCTWaiter.wait(for: [XCTestExpectation(description: "pause")], timeout: seconds)
  }

  @MainActor
  private func save(_ app: XCUIApplication, _ name: String) {
    let shot = app.screenshot()
    let attachment = XCTAttachment(screenshot: shot)
    attachment.name = name
    attachment.lifetime = .keepAlways
    add(attachment)
    if let directory = ProcessInfo.processInfo.environment["INKEAR_SHOTS"] {
      try? shot.pngRepresentation.write(to: URL(filePath: directory).appending(path: "\(name).png"))
    }
  }
}
