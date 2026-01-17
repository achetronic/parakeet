# Parakeet Makefile

# Variables
BIN_DIR := ./bin
BINARY_NAME := $(BIN_DIR)/parakeet
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
LDFLAGS := -ldflags "-s -w -X main.Version=$(VERSION) -X main.Commit=$(COMMIT) -X main.BuildDate=$(BUILD_DATE)"

# Go parameters
GOCMD := go
GOBUILD := $(GOCMD) build
GOTEST := $(GOCMD) test
GOMOD := $(GOCMD) mod
GOFMT := gofmt
GOVET := $(GOCMD) vet

# Docker parameters
DOCKER_IMAGE := parakeet
DOCKER_TAG ?= $(VERSION)

# Directories
MODELS_DIR := ./models

.PHONY: all build clean test fmt vet lint run help
.PHONY: docker-build docker-run docker-push
.PHONY: models models-int8 models-fp32
.PHONY: release release-linux release-darwin release-windows
.PHONY: deps-onnxruntime

# Auto-detect ONNX Runtime library path
ONNXRUNTIME_LIB ?= $(shell \
	if [ -f /usr/lib/libonnxruntime.so ]; then echo /usr/lib/libonnxruntime.so; \
	elif [ -f /usr/local/lib/libonnxruntime.so ]; then echo /usr/local/lib/libonnxruntime.so; \
	elif [ -f /opt/onnxruntime/lib/libonnxruntime.so ]; then echo /opt/onnxruntime/lib/libonnxruntime.so; \
	elif ls /usr/lib/x86_64-linux-gnu/libonnxruntime.so* 1>/dev/null 2>&1; then ls /usr/lib/x86_64-linux-gnu/libonnxruntime.so* 2>/dev/null | head -1; \
	elif ls /usr/lib/aarch64-linux-gnu/libonnxruntime.so* 1>/dev/null 2>&1; then ls /usr/lib/aarch64-linux-gnu/libonnxruntime.so* 2>/dev/null | head -1; \
	fi)

# Default target
all: build

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

## Build targets

build: $(BIN_DIR) ## Build the binary
	$(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME) .

build-static: $(BIN_DIR) ## Build a statically linked binary
	CGO_ENABLED=0 $(GOBUILD) $(LDFLAGS) -a -installsuffix cgo -o $(BINARY_NAME) .

## Development targets

run: build ## Build and run the server
	@if [ -z "$(ONNXRUNTIME_LIB)" ]; then \
		echo "Error: ONNX Runtime library not found. Run 'make deps-onnxruntime' to install it."; \
		exit 1; \
	fi
	ONNXRUNTIME_LIB=$(ONNXRUNTIME_LIB) ./$(BINARY_NAME) --debug=true

run-dev: build ## Run with custom port for development
	@if [ -z "$(ONNXRUNTIME_LIB)" ]; then \
		echo "Error: ONNX Runtime library not found. Run 'make deps-onnxruntime' to install it."; \
		exit 1; \
	fi
	ONNXRUNTIME_LIB=$(ONNXRUNTIME_LIB) ./$(BINARY_NAME) -port 5092 -models $(MODELS_DIR)

clean: ## Remove build artifacts
	rm -rf $(BIN_DIR)

## Code quality targets

fmt: ## Format source code
	$(GOFMT) -s -w .

vet: ## Run go vet
	$(GOVET) ./...

lint: vet fmt ## Run all linters

test: ## Run tests
	$(GOTEST) -v ./...

test-coverage: ## Run tests with coverage
	$(GOTEST) -v -coverprofile=coverage.out ./...
	$(GOCMD) tool cover -html=coverage.out -o coverage.html

## Dependency targets

deps: ## Download dependencies
	$(GOMOD) download

deps-tidy: ## Tidy dependencies
	$(GOMOD) tidy

deps-verify: ## Verify dependencies
	$(GOMOD) verify

ONNXRUNTIME_VERSION ?= 1.16.3
deps-onnxruntime: ## Install ONNX Runtime library
	@echo "Installing ONNX Runtime $(ONNXRUNTIME_VERSION)..."
	@ARCH=$$(uname -m); \
	if [ "$$ARCH" = "x86_64" ]; then ARCH_NAME="x64"; \
	elif [ "$$ARCH" = "aarch64" ]; then ARCH_NAME="aarch64"; \
	else echo "Unsupported architecture: $$ARCH"; exit 1; fi; \
	curl -L -o /tmp/onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v$(ONNXRUNTIME_VERSION)/onnxruntime-linux-$$ARCH_NAME-$(ONNXRUNTIME_VERSION).tgz" && \
	tar -xzf /tmp/onnxruntime.tgz -C /tmp && \
	sudo cp /tmp/onnxruntime-linux-$$ARCH_NAME-$(ONNXRUNTIME_VERSION)/lib/* /usr/local/lib/ && \
	sudo ldconfig && \
	rm -rf /tmp/onnxruntime.tgz /tmp/onnxruntime-linux-$$ARCH_NAME-$(ONNXRUNTIME_VERSION) && \
	echo "ONNX Runtime installed to /usr/local/lib/"

## Model targets
# Models are downloaded from HuggingFace: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx
# ONNX conversion by Ivan Googol Stupakov (https://github.com/istupakov)

models: models-int8 ## Download models (default: int8)

models-int8: ## Download int8 quantized models (~670MB)
	@mkdir -p $(MODELS_DIR)
	@echo "Downloading Parakeet TDT int8 models from https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx ..."
	@curl -L -o $(MODELS_DIR)/config.json "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/config.json"
	@curl -L -o $(MODELS_DIR)/vocab.txt "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt"
	@curl -L -o $(MODELS_DIR)/nemo128.onnx "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/nemo128.onnx"
	@curl -L -o $(MODELS_DIR)/encoder-model.int8.onnx "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.int8.onnx"
	@curl -L -o $(MODELS_DIR)/decoder_joint-model.int8.onnx "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.int8.onnx"
	@echo "Models downloaded to $(MODELS_DIR)"

models-fp32: ## Download fp32 full precision models (~2.5GB)
	@mkdir -p $(MODELS_DIR)
	@echo "Downloading Parakeet TDT fp32 models from https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx ..."
	@curl -L -o $(MODELS_DIR)/config.json "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/config.json"
	@curl -L -o $(MODELS_DIR)/vocab.txt "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt"
	@curl -L -o $(MODELS_DIR)/nemo128.onnx "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/nemo128.onnx"
	@curl -L -o $(MODELS_DIR)/encoder-model.onnx "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx"
	@curl -L -o $(MODELS_DIR)/encoder-model.onnx.data "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx.data"
	@curl -L -o $(MODELS_DIR)/decoder_joint-model.onnx "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.onnx"
	@echo "Models downloaded to $(MODELS_DIR)"

## Docker targets

docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_IMAGE):latest

docker-run: ## Run Docker container
	docker run --rm -p 5092:5092 -v $(PWD)/models:/models $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-push: ## Push Docker image to registry
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_IMAGE):latest

## Release targets

release: $(BIN_DIR) release-linux release-darwin release-windows ## Build release binaries for all platforms

release-linux: $(BIN_DIR) ## Build Linux binaries
	GOOS=linux GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME)-linux-amd64 .
	GOOS=linux GOARCH=arm64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME)-linux-arm64 .

release-darwin: $(BIN_DIR) ## Build macOS binaries
	GOOS=darwin GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME)-darwin-amd64 .
	GOOS=darwin GOARCH=arm64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME)-darwin-arm64 .

release-windows: $(BIN_DIR) ## Build Windows binary
	GOOS=windows GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME)-windows-amd64.exe .

## Help target

help: ## Show this help message
	@echo "Parakeet - Makefile targets"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
