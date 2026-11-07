#!/usr/bin/env bash
set -euo pipefail

# Build a Parakeet model image from training/asr/assets.
# Usage: build-image.sh <int8|fp32> <image> <tag> [--push]

mode=$1
image=$2
tag=$3
push=${4:-}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
assets="$root/training/asr/assets"
stage="$root/training/.image/$mode"

rm -rf "$root/training/.image"
mkdir -p "$stage"

cp "$assets/config.json" "$assets/vocab.txt" "$stage/"

case "$mode" in
  int8)
    cp "$assets/encoder-model.int8.onnx" \
       "$assets/encoder-local.int8.onnx" \
       "$assets/decoder_joint-model.int8.onnx" \
       "$stage/"
    ;;
  fp32)
    cp "$assets/encoder-model.onnx" \
       "$assets/encoder-model.onnx.data" \
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

docker build -f "$root/training/Dockerfile.models" -t "$image:$tag" "$stage"

if [ "$push" = "--push" ]; then
  docker push "$image:$tag"
fi
