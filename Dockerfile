# Build stage - using debian for CGO support (required by onnxruntime_go)
FROM golang:1.25-bookworm AS builder

# Install ONNX Runtime for build
# NOTE: onnxruntime_go v1.30.1 targets the ONNX Runtime 1.25.x C API (ORT_API_VERSION 25)
ARG ONNXRUNTIME_VERSION=1.25.1
ARG TARGETARCH
RUN ARCH=$([ "$TARGETARCH" = "arm64" ] && echo "aarch64" || echo "x64") && \
    curl -L -o /tmp/onnxruntime.tgz \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ARCH}-${ONNXRUNTIME_VERSION}.tgz" && \
    tar -xzf /tmp/onnxruntime.tgz -C /opt && \
    mv /opt/onnxruntime-linux-${ARCH}-${ONNXRUNTIME_VERSION} /opt/onnxruntime && \
    rm /tmp/onnxruntime.tgz

ENV CGO_ENABLED=1
ENV CGO_LDFLAGS="-L/opt/onnxruntime/lib"
ENV CGO_CFLAGS="-I/opt/onnxruntime/include"
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

WORKDIR /app

# Copy go mod files first for better caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build binary with CGO
RUN go build -ldflags="-s -w" -o parakeet .

# Models stage: downloads the models via the Makefile recipes so the runtime
# stage only receives the model files (no make, curl or Makefile in the final
# image).
FROM debian:bookworm-slim AS models
ARG MODEL_PRECISION=int8
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Makefile /app/Makefile
RUN make MODELS_DIR=/models model-download-asr-${MODEL_PRECISION} model-download-vad model-download-diarization-${MODEL_PRECISION}

# Runtime stage
FROM debian:bookworm-slim

# Install ffmpeg (used for non-WAV audio conversion). No curl here: the ONNX
# Runtime library is copied from the builder stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set library path
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib
ENV ONNXRUNTIME_LIB=/opt/onnxruntime/lib/libonnxruntime.so

WORKDIR /app

# Copy the ONNX Runtime library (from the builder), the models (from the
# models stage) and the binary.
COPY --from=builder /opt/onnxruntime/lib /opt/onnxruntime/lib
COPY --from=models /models /models
COPY --from=builder /app/parakeet /app/parakeet

# Expose port
EXPOSE 5092

# Runtime defaults as env vars, not CMD args: Docker replaces the whole CMD when
# the user passes their own flags, so a default baked into CMD is lost on any
# override. Every flag reads its PARAKEET_* env var, and a flag passed on the
# command line still wins over the env var.
ARG MODEL_PRECISION=int8
ENV PARAKEET_PORT=5092
ENV PARAKEET_MODELS=/models
ENV PARAKEET_MODEL_PRECISION=${MODEL_PRECISION}

ENTRYPOINT ["/app/parakeet"]
