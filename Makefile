PROJECT := inkear/inkear.xcodeproj
SCHEME := inkear
SIMULATOR ?= iPhone 18 Pro
UI_TESTS ?= inkearUITests
DEVICE ?= 00008140-000815391ED2801C
SUPERTONIC := tools/coreml/build/supertonic-3

.DEFAULT_GOAL := help
.PHONY: help generate build build-device install uninstall test test-fast test-ui lint clean

help: ## List the targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-13s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

generate: $(PROJECT)/project.pbxproj ## Regenerate the Xcode project when inkear/project.yml changed

build: generate | $(SUPERTONIC) ## Build the app for the iOS Simulator
	xcodebuild -project $(PROJECT) -scheme $(SCHEME) -destination 'generic/platform=iOS Simulator' -quiet build

build-device: generate | $(SUPERTONIC) ## Build the app for iPhone, unsigned
	xcodebuild -project $(PROJECT) -scheme $(SCHEME) -destination 'generic/platform=iOS' \
		CODE_SIGNING_ALLOWED=NO -quiet build

install: generate | $(SUPERTONIC) ## Build the app signed and install it on DEVICE
	xcodebuild -project $(PROJECT) -scheme $(SCHEME) -destination 'id=$(DEVICE)' \
		-derivedDataPath build -allowProvisioningUpdates -quiet build
	xcrun devicectl device install app --device $(DEVICE) build/Build/Products/Debug-iphoneos/inkear.app

uninstall: ## Remove the app and all its data from DEVICE
	xcrun devicectl device uninstall app --device $(DEVICE) dev.gabfssilva.inkear

test: ## Run the InkearKit tests on the Mac
	cd InkearKit && swift test

test-fast: ## Run the InkearKit tests but the corpus
	cd InkearKit && swift test --skip CorpusTests

# Xcode 27 collects a simulator diagnose after passing UI tests too, which runs into its 10-minute
# timeout; the result bundle keeps the test logs and screenshots without it.
test-ui: generate | $(SUPERTONIC) ## Run the UI tests on SIMULATOR; UI_TESTS narrows them
	xcodebuild -project $(PROJECT) -scheme $(SCHEME) -destination 'platform=iOS Simulator,name=$(SIMULATOR)' \
		-only-testing:$(UI_TESTS) -collect-test-diagnostics never test

lint: ## Lint the Python tools
	uvx ruff@0.17.0 check tools

clean: ## Remove the build products of the app and of InkearKit
	xcodebuild -project $(PROJECT) -scheme $(SCHEME) -quiet clean
	cd InkearKit && swift package clean

$(PROJECT)/project.pbxproj: inkear/project.yml
	cd inkear && xcodegen generate

# The app bundles Supertonic from here (inkear/Scripts/bundle-supertonic.sh).
$(SUPERTONIC):
	cd tools/coreml && uv run supertonic-convert
