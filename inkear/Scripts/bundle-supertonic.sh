#!/bin/sh
# Compiles the Supertonic Core ML packages into the app bundle, next to the assets the engine reads.
set -eu

source="$SRCROOT/../tools/coreml/build/supertonic-3"
destination="$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH/supertonic-3"
mkdir -p "$destination"

for stage in duration_predictor text_encoder vector_estimator vocoder; do
  if [ ! -d "$destination/$stage.mlmodelc" ] || [ "$source/$stage.mlpackage" -nt "$destination/$stage.mlmodelc" ]; then
    rm -rf "$destination/$stage.mlmodelc"
    xcrun coremlcompiler compile "$source/$stage.mlpackage" "$destination"
  fi
done

cp "$source/tts.json" "$source/unicode_indexer.json" "$destination/"
rsync -a --delete "$source/voice_styles/" "$destination/voice_styles/"
