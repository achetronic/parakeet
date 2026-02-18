# Build stage - using debian for CGO support (required by onnxruntime_go)
FROM golang:1.25-bookworm AS builder

# Install ONNX Runtime for build
# NOTE: onnxruntime_go v1.19.0 is compatible with ONNX Runtime 1.21.x
ARG ONNXRUNTIME_VERSION=1.21.0
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

# Install ONNX Runtime
ARG ONNXRUNTIME_VERSION=1.21.0
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
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

# Copy binary from builder
COPY --from=builder /app/parakeet /app/parakeet

# Download and embed models based on MODEL_PRECISION arg
# int8: encoder-model.int8.onnx, decoder_joint-model.int8.onnx (~670MB total)
# fp32: encoder-model.onnx, encoder-model.onnx.data, decoder_joint-model.onnx (~2.5GB total)
RUN mkdir -p /models && \
    curl -L -o /models/config.json "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/config.json" && \
    curl -L -o /models/vocab.txt "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt" && \
    curl -L -o /models/nemo128.onnx "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/nemo128.onnx" && \
    if [ "$MODEL_PRECISION" = "fp32" ]; then \
        curl -L -o /models/encoder-model.onnx "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx" && \
        curl -L -o /models/encoder-model.onnx.data "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx.data" && \
        curl -L -o /models/decoder_joint-model.onnx "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.onnx"; \
    else \
        curl -L -o /models/encoder-model.int8.onnx "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.int8.onnx" && \
        curl -L -o /models/decoder_joint-model.int8.onnx "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.int8.onnx"; \
    fi && \
    apt-get remove -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Expose port
EXPOSE 5092

# Run
ENTRYPOINT ["/app/parakeet"]
CMD ["-port", "5092", "-models", "/models"]
