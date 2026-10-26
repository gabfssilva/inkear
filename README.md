# inkear

inkear reads PDFs aloud on iOS with on-device text-to-speech on the Apple Neural Engine. It
keeps reading in the background and navigates by the document's structure: pages, sections,
paragraphs, and sentences.

## Features

- **PDF analysis** — text layer or Vision OCR, reading order, sections from the outline, the
  printed table of contents, or layout headings.
- **On-device TTS engines** — downloaded on demand from Hugging Face: Kokoro, Pocket TTS,
  MOSS-TTS-Nano, Qwen3-TTS, Magpie, OmniVoice, Chatterbox, and Phonon. Supertonic ships inside
  the app.
- **Streaming narration** — audio starts while it is still being generated, with speed control,
  quality levels where supported, a sleep timer, and lock-screen controls.
- **Background reading** — keeps reading while the app is in the background.

## Layout

| Path | What it is |
|---|---|
| `inkear/` | The SwiftUI app; `project.yml` generates the Xcode project (`xcodegen generate`) |
| `InkearKit/` | A Swift package with all the logic, tested on the Mac |
| `tools/` | uv projects that convert models and publish them to Hugging Face under `itsmechico/` |
| `corpus/` | Open PDFs in pt-BR and English used by the tests |
| `design/` | App icon and App Store texts |

## Build and test

```sh
make build          # app for the simulator
make build-device   # app for iPhone, unsigned
make install         # signed build, installed on DEVICE
make test            # InkearKit tests on the Mac
make test-fast       # the same, less the corpus tests

# Or by hand
cd inkear && xcodegen generate
cd InkearKit && swift test
```

The Xcode project and every build artifact are generated and gitignored. See `CLAUDE.md`
for the details.

## Voice models

The downloadable engines are published as Core ML/Core AI packages under `itsmechico/` on
Hugging Face. Each model has its own license — read the model card before distributing:

- MIT / Apache-2.0: Kokoro, Pocket TTS, MOSS, Qwen3, Chatterbox
- NVIDIA Open Model License: Magpie
- CC BY-NC 4.0: OmniVoice, Phonon (non-commercial use)

## License

MIT — see [LICENSE](LICENSE).
