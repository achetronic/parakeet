#!/usr/bin/env bash
set -euo pipefail

# Build a Parakeet diarization model image from export/diarization/assets.
# Usage: build-image.sh <int8|fp32> <image> <tag> [--push]

mode=$1
image=$2
tag=$3
push=${4:-}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
assets="$root/export/diarization/assets"
stage="$root/export/.image-diarization/$mode"

rm -rf "$root/export/.image-diarization"
mkdir -p "$stage"

cp "$assets/config.json" \
   "$assets/preprocessor_config.json" \
   "$assets/processor_config.json" \
   "$stage/"

case "$mode" in
  int8)
    cp "$assets/model_quantized.onnx" \
       "$assets/model_quantized.onnx_data" \
       "$stage/"
    ;;
  fp32)
    cp "$assets/model.onnx" "$stage/"
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 1
    ;;
esac

docker build -f "$root/export/diarization/Dockerfile.models" -t "$image:$tag" "$stage"

if [ "$push" = "--push" ]; then
  docker push "$image:$tag"
  docker tag "$image:$tag" "$image:latest"
  docker push "$image:latest"
fi
