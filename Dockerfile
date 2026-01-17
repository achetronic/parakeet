# Build stage
FROM golang:1.22-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /app

# Copy go mod files first for better caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build binary
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o parakeet-server .

# Runtime stage
FROM debian:bookworm-slim

# Install ONNX Runtime
ARG ONNXRUNTIME_VERSION=1.17.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && curl -L -o /tmp/onnxruntime.tgz \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt \
    && mv /opt/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION} /opt/onnxruntime \
    && rm /tmp/onnxruntime.tgz \
    && apt-get remove -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Set library path
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib
ENV ONNXRUNTIME_LIB=/opt/onnxruntime/lib/libonnxruntime.so

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/parakeet-server /app/parakeet-server

# Create models directory
RUN mkdir -p /models

# Expose port
EXPOSE 5092

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5092/health || exit 1

# Run
ENTRYPOINT ["/app/parakeet-server"]
CMD ["-port", "5092", "-models", "/models"]
