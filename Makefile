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
DOCKER_IMAGE := ghcr.io/achetronic/parakeet
DOCKER_TAG ?= $(VERSION)

# Directories
MODELS_DIR := ./models

.PHONY: all build clean test fmt vet lint run help
.PHONY: docker-build-int8 docker-build-fp32 docker-build-cuda docker-run-int8 docker-run-fp32 docker-run-cuda docker-push
.PHONY: model-download model-download-asr model-download-asr-int8 model-download-asr-fp32 model-download-vad model-download-diarization model-download-diarization-int8 model-download-diarization-fp32
.PHONY: model-training-asr-venv model-training-asr-export model-build-oci-images model-push-oci-images
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

################################################################################
## Build targets
################################################################################

build: $(BIN_DIR) ## Build the binary
	$(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME) .

# NOTE: Static build (CGO_ENABLED=0) is not possible because onnxruntime_go requires CGO

################################################################################
## Development targets
################################################################################

run: build ## Build and run the server
	@if [ -z "$(ONNXRUNTIME_LIB)" ]; then \
		echo "Error: ONNX Runtime library not found. Run 'make deps-onnxruntime' to install it."; \
		exit 1; \
	fi
	ONNXRUNTIME_LIB=$(ONNXRUNTIME_LIB) ./$(BINARY_NAME) -log-level=debug

run-dev: build ## Run with custom port for development
	@if [ -z "$(ONNXRUNTIME_LIB)" ]; then \
		echo "Error: ONNX Runtime library not found. Run 'make deps-onnxruntime' to install it."; \
		exit 1; \
	fi
	ONNXRUNTIME_LIB=$(ONNXRUNTIME_LIB) ./$(BINARY_NAME) -port 5092 -models $(MODELS_DIR) -log-level=debug

clean: ## Remove build artifacts
	rm -rf $(BIN_DIR)

################################################################################
## Code quality targets
################################################################################

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

################################################################################
## Dependency targets
################################################################################

deps: ## Download dependencies
	$(GOMOD) download

deps-tidy: ## Tidy dependencies
	$(GOMOD) tidy

deps-verify: ## Verify dependencies
	$(GOMOD) verify

ONNXRUNTIME_VERSION ?= 1.25.1
deps-onnxruntime: ## Install ONNX Runtime library
	@echo "Installing ONNX Runtime $(ONNXRUNTIME_VERSION)..."
	@ARCH=$$(uname -m); \
	if [ "$$ARCH" = "x86_64" ]; then ARCH_NAME="x64"; \
	elif [ "$$ARCH" = "aarch64" ]; then ARCH_NAME="aarch64"; \
	else echo "Unsupported architecture: $$ARCH"; exit 1; fi; \
	curl -L -o /tmp/onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v$(ONNXRUNTIME_VERSION)/onnxruntime-linux-$$ARCH_NAME-$(ONNXRUNTIME_VERSION).tgz" && \
	tar -xzf /tmp/onnxruntime.tgz -C /tmp && \
	sudo cp /tmp/onnxruntime-linux-$$ARCH_NAME-$(ONNXRUNTIME_VERSION)/lib/*.so* /usr/local/lib/ && \
	sudo ldconfig && \
	rm -rf /tmp/onnxruntime.tgz /tmp/onnxruntime-linux-$$ARCH_NAME-$(ONNXRUNTIME_VERSION) && \
	echo "ONNX Runtime installed to /usr/local/lib/"

################################################################################
## Model targets
################################################################################
# Full model lifecycle in three stages: train (export assets), package (build
# + push OCI images), and download (fetch prebuilt assets).

### Training (export ONNX assets from checkpoints)
# Each model family gets its own venv and export targets. asr is the only one
# today; add vad/ or diarization/ under training/ and mirror these targets.
TRAINING_PYTHON ?= training/venv/bin/python

model-training-asr-venv: ## Create the training venv with ASR dependencies
	@uv venv --python 3.12 training/venv
	@uv pip install --python training/venv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cu126
	@uv pip install --python training/venv/bin/python -r training/asr/requirements.txt

model-training-asr-export: ## Regenerate ASR ONNX assets from the NVIDIA checkpoint
	@test -x $(TRAINING_PYTHON) || (echo "training venv missing, run: make model-training-asr-venv" && exit 1)
	@$(TRAINING_PYTHON) training/asr/export.py --out training/asr/assets
	@echo "Assets written to training/asr/assets/"

### Package (build + publish OCI model images)
# The model images carry the ASR encoder+decoder split (int8/fp32). They are
# ASR-specific for now; a future family gets its own image targets if needed.
MODEL_IMAGE_REPO ?= ghcr.io/achetronic/parakeet-models
MODEL_TAG ?= latest

model-build-oci-images: model-training-asr-export ## Build int8 + fp32 model images locally (no push)
	@./training/asr/build-image.sh int8 "$(MODEL_IMAGE_REPO)-int8" "$(MODEL_TAG)"
	@./training/asr/build-image.sh fp32 "$(MODEL_IMAGE_REPO)-fp32" "$(MODEL_TAG)"

model-push-oci-images: model-training-asr-export ## Build and push int8 + fp32 model images to GHCR
	@./training/asr/build-image.sh int8 "$(MODEL_IMAGE_REPO)-int8" "$(MODEL_TAG)" --push
	@./training/asr/build-image.sh fp32 "$(MODEL_IMAGE_REPO)-fp32" "$(MODEL_TAG)" --push

### Download (fetch prebuilt models)
# models/ is organized by model family first, precision second:
#   models/asr/{int8,fp32}, models/vad, models/diarization/{int8,fp32}.
# VAD has no precision. model-download pulls everything.

# ASR encoder + decoder assets. Default to the community conversion; point
# MODELS_REPO at your own repo to self-host (see training/README.md).
MODELS_REPO ?= istupakov/parakeet-tdt-0.6b-v3-onnx
MODELS_ASR_INT8_DIR := $(MODELS_DIR)/asr/int8
MODELS_ASR_FP32_DIR := $(MODELS_DIR)/asr/fp32
MODELS_VAD_DIR := $(MODELS_DIR)/vad

model-download: model-download-asr model-download-vad model-download-diarization ## Download every model (ASR, VAD, diarization, both precisions)

model-download-asr: model-download-asr-int8 model-download-asr-fp32 ## Download ASR models (int8 + fp32)

model-download-asr-int8: ## Download ASR int8 models (~670MB)
	@mkdir -p $(MODELS_ASR_INT8_DIR)
	@echo "Downloading Parakeet TDT int8 models from https://huggingface.co/$(MODELS_REPO) ..."
	@curl -L -o $(MODELS_ASR_INT8_DIR)/config.json "https://huggingface.co/$(MODELS_REPO)/resolve/main/config.json"
	@curl -L -o $(MODELS_ASR_INT8_DIR)/vocab.txt "https://huggingface.co/$(MODELS_REPO)/resolve/main/vocab.txt"
	@curl -L -o $(MODELS_ASR_INT8_DIR)/nemo128.onnx "https://huggingface.co/$(MODELS_REPO)/resolve/main/nemo128.onnx"
	@curl -L -o $(MODELS_ASR_INT8_DIR)/encoder-model.int8.onnx "https://huggingface.co/$(MODELS_REPO)/resolve/main/encoder-model.int8.onnx"
	@curl -L -o $(MODELS_ASR_INT8_DIR)/decoder_joint-model.int8.onnx "https://huggingface.co/$(MODELS_REPO)/resolve/main/decoder_joint-model.int8.onnx"
	@echo "ASR int8 models downloaded to $(MODELS_ASR_INT8_DIR)"

model-download-asr-fp32: ## Download ASR fp32 models (~2.5GB)
	@mkdir -p $(MODELS_ASR_FP32_DIR)
	@echo "Downloading Parakeet TDT fp32 models from https://huggingface.co/$(MODELS_REPO) ..."
	@curl -L -o $(MODELS_ASR_FP32_DIR)/config.json "https://huggingface.co/$(MODELS_REPO)/resolve/main/config.json"
	@curl -L -o $(MODELS_ASR_FP32_DIR)/vocab.txt "https://huggingface.co/$(MODELS_REPO)/resolve/main/vocab.txt"
	@curl -L -o $(MODELS_ASR_FP32_DIR)/nemo128.onnx "https://huggingface.co/$(MODELS_REPO)/resolve/main/nemo128.onnx"
	@curl -L -o $(MODELS_ASR_FP32_DIR)/encoder-model.onnx "https://huggingface.co/$(MODELS_REPO)/resolve/main/encoder-model.onnx"
	@curl -L -o $(MODELS_ASR_FP32_DIR)/encoder-model.onnx.data "https://huggingface.co/$(MODELS_REPO)/resolve/main/encoder-model.onnx.data"
	@curl -L -o $(MODELS_ASR_FP32_DIR)/decoder_joint-model.onnx "https://huggingface.co/$(MODELS_REPO)/resolve/main/decoder_joint-model.onnx"
	@echo "ASR fp32 models downloaded to $(MODELS_ASR_FP32_DIR)"

# Silero VAD model for chunk-boundary selection (long-audio mode). Pinned to a
# specific release tag and sha256 so the download is reproducible and verified.
# snakers4/silero-vad is MIT licensed. NOTE: at v6.2.1 the ONNX file lives at
# src/silero_vad/data/silero_vad.onnx (the older files/silero_vad.onnx path is
# gone), so the URL points there. VAD is precision-independent, so it lives
# directly under models/vad/.
SILERO_VAD_VERSION ?= v6.2.1
SILERO_VAD_SHA256 ?= 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
SILERO_VAD_URL := https://github.com/snakers4/silero-vad/raw/$(SILERO_VAD_VERSION)/src/silero_vad/data/silero_vad.onnx

model-download-vad: ## Download and verify the Silero VAD model (MIT)
	@mkdir -p $(MODELS_VAD_DIR)
	@echo "Downloading Silero VAD $(SILERO_VAD_VERSION) (MIT) ..."
	@curl -L -o $(MODELS_VAD_DIR)/silero_vad.onnx "$(SILERO_VAD_URL)"
	@echo "$(SILERO_VAD_SHA256)  $(MODELS_VAD_DIR)/silero_vad.onnx" | sha256sum -c -
	@echo "Silero VAD model verified"

# .onnx + .onnx_data must stay together (external weights, relative path). Ref:
# https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX
DIARIZATION_REPO := onnx-community/Nemotron-3-Diarization-ONNX
DIARIZATION_INT8_DIR := $(MODELS_DIR)/diarization/int8
DIARIZATION_FP32_DIR := $(MODELS_DIR)/diarization/fp32

model-download-diarization: model-download-diarization-int8 model-download-diarization-fp32 ## Download diarization models (int8 + fp32)

model-download-diarization-int8: ## Download int8 diarization model (~120MB)
	@mkdir -p $(DIARIZATION_INT8_DIR)
	@echo "Downloading Nemotron-3-Diarization int8 model from https://huggingface.co/$(DIARIZATION_REPO) ..."
	@curl -L -o $(DIARIZATION_INT8_DIR)/config.json "https://huggingface.co/$(DIARIZATION_REPO)/resolve/main/config.json"
	@curl -L -o $(DIARIZATION_INT8_DIR)/preprocessor_config.json "https://huggingface.co/$(DIARIZATION_REPO)/resolve/main/preprocessor_config.json"
	@curl -L -o $(DIARIZATION_INT8_DIR)/processor_config.json "https://huggingface.co/$(DIARIZATION_REPO)/resolve/main/processor_config.json"
	@curl -L -o $(DIARIZATION_INT8_DIR)/model_quantized.onnx "https://huggingface.co/$(DIARIZATION_REPO)/resolve/main/onnx/model_quantized.onnx"
	@curl -L -o $(DIARIZATION_INT8_DIR)/model_quantized.onnx_data "https://huggingface.co/$(DIARIZATION_REPO)/resolve/main/onnx/model_quantized.onnx_data"
	@echo "Diarization int8 model downloaded to $(DIARIZATION_INT8_DIR)"

model-download-diarization-fp32: ## Download fp32 diarization model (~400MB)
	@mkdir -p $(DIARIZATION_FP32_DIR)
	@echo "Downloading Nemotron-3-Diarization fp32 model from https://huggingface.co/$(DIARIZATION_REPO) ..."
	@curl -L -o $(DIARIZATION_FP32_DIR)/config.json "https://huggingface.co/$(DIARIZATION_REPO)/resolve/main/config.json"
	@curl -L -o $(DIARIZATION_FP32_DIR)/preprocessor_config.json "https://huggingface.co/$(DIARIZATION_REPO)/resolve/main/preprocessor_config.json"
	@curl -L -o $(DIARIZATION_FP32_DIR)/processor_config.json "https://huggingface.co/$(DIARIZATION_REPO)/resolve/main/processor_config.json"
	@curl -L -o $(DIARIZATION_FP32_DIR)/model.onnx "https://huggingface.co/$(DIARIZATION_REPO)/resolve/main/onnx/model.onnx"
	@curl -L -o $(DIARIZATION_FP32_DIR)/model.onnx_data "https://huggingface.co/$(DIARIZATION_REPO)/resolve/main/onnx/model.onnx_data"
	@echo "Diarization fp32 model downloaded to $(DIARIZATION_FP32_DIR)"

################################################################################
## Docker targets
################################################################################

docker-build-int8: ## Build Docker image with embedded int8 models (~670MB)
	docker build --build-arg MODEL_PRECISION=int8 -t $(DOCKER_IMAGE):$(DOCKER_TAG)-int8 .
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG)-int8 $(DOCKER_IMAGE):latest-int8

docker-build-fp32: ## Build Docker image with embedded fp32 models (~2.5GB)
	docker build --build-arg MODEL_PRECISION=fp32 -t $(DOCKER_IMAGE):$(DOCKER_TAG)-fp32 .
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG)-fp32 $(DOCKER_IMAGE):latest-fp32

docker-build-cuda: ## Build CUDA/GPU Docker image with embedded fp32 models (linux/amd64)
	docker build -f Dockerfile.cuda --build-arg MODEL_PRECISION=fp32 -t $(DOCKER_IMAGE):$(DOCKER_TAG)-cuda .
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG)-cuda $(DOCKER_IMAGE):latest-cuda

docker-run-int8: ## Run Docker container with embedded int8 models
	docker run --rm -p 5092:5092 $(DOCKER_IMAGE):$(DOCKER_TAG)-int8

docker-run-fp32: ## Run Docker container with embedded fp32 models
	docker run --rm -p 5092:5092 $(DOCKER_IMAGE):$(DOCKER_TAG)-fp32

docker-run-cuda: ## Run CUDA/GPU Docker container (requires nvidia-container-toolkit)
	docker run --rm --gpus all -p 5092:5092 $(DOCKER_IMAGE):$(DOCKER_TAG)-cuda

docker-push: ## Push Docker images to registry
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)-int8
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)-fp32
	docker push $(DOCKER_IMAGE):latest-int8
	docker push $(DOCKER_IMAGE):latest-fp32

################################################################################
## Release targets
################################################################################

release: $(BIN_DIR) release-linux release-darwin release-windows ## Build release binaries for all platforms

release-linux: $(BIN_DIR) ## Build Linux binaries
	GOOS=linux GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME)-linux-amd64 .
	GOOS=linux GOARCH=arm64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME)-linux-arm64 .

release-darwin: $(BIN_DIR) ## Build macOS binaries
	GOOS=darwin GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME)-darwin-amd64 .
	GOOS=darwin GOARCH=arm64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME)-darwin-arm64 .

release-windows: $(BIN_DIR) ## Build Windows binary
	GOOS=windows GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME)-windows-amd64.exe .

################################################################################
## Help target
################################################################################

help: ## Show this help message
	@echo "Parakeet ASR Server - Makefile targets"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "\033[1mBuild:\033[0m"
	@echo "  \033[36mbuild\033[0m               Build the binary to ./bin/parakeet"
	@echo "  \033[36mclean\033[0m               Remove build artifacts"
	@echo ""
	@echo "\033[1mDevelopment:\033[0m"
	@echo "  \033[36mrun\033[0m                 Build and run the server with debug mode"
	@echo "  \033[36mrun-dev\033[0m             Run with custom port (5092) for development"
	@echo ""
	@echo "\033[1mCode Quality:\033[0m"
	@echo "  \033[36mfmt\033[0m                 Format source code"
	@echo "  \033[36mvet\033[0m                 Run go vet"
	@echo "  \033[36mlint\033[0m                Run all linters (vet + fmt)"
	@echo "  \033[36mtest\033[0m                Run tests"
	@echo "  \033[36mtest-coverage\033[0m       Run tests with coverage report"
	@echo ""
	@echo "\033[1mDependencies:\033[0m"
	@echo "  \033[36mdeps\033[0m                Download Go dependencies"
	@echo "  \033[36mdeps-tidy\033[0m           Tidy dependencies"
	@echo "  \033[36mdeps-verify\033[0m         Verify dependencies"
	@echo "  \033[36mdeps-onnxruntime\033[0m    Install ONNX Runtime library (requires sudo)"
	@echo ""
	@echo "\033[1mModels:\033[0m"
	@echo "  \033[36mmodel-training-asr-venv\033[0m  Create the training venv with ASR deps"
	@echo "  \033[36mmodel-training-asr-export\033[0m  Regenerate ASR ONNX assets from the NVIDIA checkpoint"
	@echo "  \033[36mmodel-build-oci-images\033[0m  Build int8 + fp32 model images locally"
	@echo "  \033[36mmodel-push-oci-images\033[0m  Build and push int8 + fp32 model images to GHCR"
	@echo "  \033[36mmodel-download\033[0m  Download every model (ASR + VAD + diarization, both precisions)"
	@echo "  \033[36mmodel-download-asr\033[0m  Download ASR models (int8 + fp32)"
	@echo "  \033[36mmodel-download-asr-int8\033[0m  Download ASR int8 models (~670MB)"
	@echo "  \033[36mmodel-download-asr-fp32\033[0m  Download ASR fp32 models (~2.5GB)"
	@echo "  \033[36mmodel-download-vad\033[0m  Download and verify the Silero VAD model (MIT)"
	@echo "  \033[36mmodel-download-diarization\033[0m  Download diarization models (int8 + fp32)"
	@echo "  \033[36mmodel-download-diarization-int8\033[0m  Download int8 diarization model (~120MB)"
	@echo "  \033[36mmodel-download-diarization-fp32\033[0m  Download fp32 diarization model (~400MB)"
	@echo ""
	@echo "\033[1mDocker:\033[0m"
	@echo "  \033[36mdocker-build-int8\033[0m   Build Docker image with int8 models"
	@echo "  \033[36mdocker-build-fp32\033[0m   Build Docker image with fp32 models"
	@echo "  \033[36mdocker-build-cuda\033[0m   Build CUDA/GPU image with fp32 models"
	@echo "  \033[36mdocker-run-int8\033[0m     Run Docker container with int8 models"
	@echo "  \033[36mdocker-run-fp32\033[0m     Run Docker container with fp32 models"
	@echo "  \033[36mdocker-run-cuda\033[0m     Run CUDA/GPU container (needs --gpus all)"
	@echo "  \033[36mdocker-push\033[0m         Push Docker images to registry"
	@echo ""
	@echo "\033[1mRelease:\033[0m"
	@echo "  \033[36mrelease\033[0m             Build release binaries for all platforms"
	@echo "  \033[36mrelease-linux\033[0m       Build Linux binaries (amd64/arm64)"
	@echo "  \033[36mrelease-darwin\033[0m      Build macOS binaries (amd64/arm64)"
	@echo "  \033[36mrelease-windows\033[0m     Build Windows binary (amd64)"
	@echo ""
	@echo "\033[1mOther:\033[0m"
	@echo "  \033[36mhelp\033[0m                Show this help message"
