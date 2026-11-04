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

# Runtime stage
FROM debian:bookworm-slim

# Model precision: "int8" (default, ~670MB) or "fp32" (~2.5GB)
ARG MODEL_PRECISION=int8

# Install ONNX Runtime, make (model download via the Makefile) and ffmpeg
# (used for non-WAV audio conversion)
ARG ONNXRUNTIME_VERSION=1.25.1
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    make \
    && ARCH=$([ "$TARGETARCH" = "arm64" ] && echo "aarch64" || echo "x64") \
    && curl -L -o /tmp/onnxruntime.tgz \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ARCH}-${ONNXRUNTIME_VERSION}.tgz" \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt \
    && mv /opt/onnxruntime-linux-${ARCH}-${ONNXRUNTIME_VERSION} /opt/onnxruntime \
    && rm /tmp/onnxruntime.tgz \
    && rm -rf /var/lib/apt/lists/*

# Set library path
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib
ENV ONNXRUNTIME_LIB=/opt/onnxruntime/lib/libonnxruntime.so

WORKDIR /app

# Download and embed models via the Makefile recipes (single source of truth
# for the model URLs). models-int8/-fp32 chain in Silero VAD and the
# Nemotron-3-Diarization model (gated at runtime by -enable-diarization).
# Downloaded before copying the binary so the multi-GB model layer stays
# cached across code changes.
COPY Makefile /app/Makefile
RUN make MODELS_DIR=/models models-${MODEL_PRECISION} && \
    apt-get purge -y curl make && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /app/parakeet /app/parakeet

# Expose port
EXPOSE 5092

# Runtime defaults as env vars, not CMD args: Docker replaces the whole CMD when
# the user passes their own flags, so a default baked into CMD is lost on any
# override. Every flag reads its PARAKEET_* env var, and a flag passed on the
# command line still wins over the env var.
ENV PARAKEET_PORT=5092
ENV PARAKEET_MODELS=/models

ENTRYPOINT ["/app/parakeet"]
