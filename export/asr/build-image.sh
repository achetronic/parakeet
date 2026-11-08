#!/usr/bin/env bash
set -euo pipefail

# Build a Parakeet model image from export/asr/assets.
# Usage: build-image.sh <int8|fp32> <image> <tag> [--push]

mode=$1
image=$2
tag=$3
push=${4:-}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
assets="$root/export/asr/assets"
stage="$root/export/.image/$mode"

rm -rf "$root/export/.image"
mkdir -p "$stage"

cp "$assets/config.json" "$assets/vocab.txt" "$stage/"

case "$mode" in
  int8)
    cp "$assets/encoder-full.int8.onnx" \
       "$assets/encoder-local.int8.onnx" \
       "$assets/decoder_joint-model.int8.onnx" \
       "$stage/"
    ;;
  fp32)
    cp "$assets/encoder-full.onnx" \
       "$assets/encoder-full.onnx.data" \
       "$assets/encoder-local.onnx" \
       "$assets/encoder-local.onnx.data" \
       "$assets/decoder_joint-model.onnx" \
       "$stage/"
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 1
    ;;
esac

docker build -f "$root/export/asr/Dockerfile.models" -t "$image:$tag" "$stage"

if [ "$push" = "--push" ]; then
  docker push "$image:$tag"
fi
