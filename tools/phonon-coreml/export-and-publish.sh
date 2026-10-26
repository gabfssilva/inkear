#!/bin/sh
# Export the Phonon 40M Core ML bundles, one per language, and publish them to Hugging Face;
# optionally rebuild and publish PhononCore.xcframework, the runtime that loads them.
#
# Requirements:
# - cargo (Rust) and the `hf` CLI, logged in (`hf auth login`), with a token that can read
#   gated repos. The checkpoints gradium/phonon-40m-<language> are gated: accept the terms on
#   each page first (approval is automatic).
# - Network access (this escapes the workspace sandbox).
#
# Both the bundles and the runtime come from xn-ptts v0.4.2 with steps.patch applied, which
# makes the number of flow steps per frame a run-time choice: the bundles gain a `flow-step`
# graph, and upstream's runtime does not load them.
#
# Environment:
# - SRC:       the xn-ptts checkout (cloned and patched here if absent).
# - OUT:       where the exported bundles go, one directory per language
#              (default: <workspace>/.context/phonon).
# - VOICES:    a directory of the voice catalog's files named "<locale> · <Name>.safetensors";
#              each bundle takes those of its language.
# - LANGUAGES: default "pt en fr de es".
# - RUNTIME:   set to 1 to also rebuild and publish the runtime to itsmechico/phonon-core-runtime.
#
# After publishing, set each model's revision and size in PhononEngine.models (the commit is
# `hf api models <repo> | jq -r .sha`), and after a new runtime, the URL and checksum of the
# PhononCore binary target in InkearKit/Package.swift.

set -eu

WORKSPACE=${WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}
SRC=${SRC:-"$WORKSPACE/.context/phonon/xn-ptts"}
OUT=${OUT:-"$WORKSPACE/.context/phonon"}
VOICES=${VOICES:-"$WORKSPACE/.context/phonon/voices-locale"}
LANGUAGES=${LANGUAGES:-"pt en fr de es"}
PATCH="$WORKSPACE/tools/phonon-coreml/steps.patch"

if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch v0.4.2 https://github.com/gradium-ai/xn-ptts.git "$SRC"
  git -C "$SRC" apply "$PATCH"
fi

cd "$SRC"
cargo build --release -p ptts --example export_coreml --features hub

for language in $LANGUAGES; do
  voices="$OUT/voices-$language"
  rm -rf "$voices" && mkdir -p "$voices"
  for file in "$VOICES/$language"*; do ln -s "$file" "$voices/"; done
  bundle="$OUT/coreml-40m-$language"
  rm -rf "$bundle"
  target/release/examples/export_coreml "$bundle" \
    --model "hf://gradium/phonon-40m-$language" --voices "$voices"
  repo="itsmechico/phonon-40m-$language-coreml"
  hf upload "$repo" "$bundle"
  echo "published: $repo (commit: $(hf api models "$repo" | jq -r .sha))"
done

if [ "${RUNTIME:-0}" = 1 ]; then
  ios/build-xcframework.sh
  hf upload itsmechico/phonon-core-runtime ios/PhononTTS/PhononCore.xcframework.zip \
    PhononCore.xcframework.zip
  echo "runtime checksum: $(swift package compute-checksum ios/PhononTTS/PhononCore.xcframework.zip)"
fi
