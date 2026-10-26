#!/bin/sh
# Export the Phonon 40M multilingual Core ML bundle and publish it to Hugging Face.
#
# Requirements:
# - cargo (Rust) and the `hf` CLI, logged in (`hf auth login`), with a token that can read
#   gated repos. The model gradium/phonon-40m-multilingual is gated: accept its terms on its
#   page first (approval is automatic).
# - Network access (this escapes the workspace sandbox).
#
# Environment:
# - SRC:    the xn-ptts checkout (cloned here if absent), pinned to v0.4.2.
# - OUT:    where the exported bundle goes (default: <workspace>/.context/phonon/bundle).
# - MODEL:  the checkpoint repo, default hf://gradium/phonon-40m-multilingual.
# - REPO:   the public destination repo, default itsmechico/phonon-40m-multilingual-coreml.
#
# After publishing, read the commit SHA (hf api models <repo> | jq -r .sha) and set it as
# PhononEngine.revision, and the sum of the bundle sizes as downloadSize.

set -eu

WORKSPACE=${WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}
SRC=${SRC:-"$WORKSPACE/.context/phonon/xn-ptts"}
OUT=${OUT:-"$WORKSPACE/.context/phonon/bundle"}
MODEL=${MODEL:-hf://gradium/phonon-40m-multilingual}
REPO=${REPO:-itsmechico/phonon-40m-multilingual-coreml}

if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch v0.4.2 \
    https://github.com/gradium-ai/xn-ptts.git "$SRC"
fi

cd "$SRC"
cargo run --release -p ptts --example export_coreml --features hub -- \
  "$OUT" --model "$MODEL"

hf upload "$REPO" "$OUT"
echo "published: $REPO (commit: $(hf api models "$REPO" | jq -r .sha))"
