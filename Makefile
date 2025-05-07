# OCR Benchmarking Makefile

# Variables
SEED := 42
SAMPLE_SIZE := 10
SOURCE_DIR := dataset/testing_data
SAMPLE_DIR := dataset/sample_test
METHODS := tesseract
VERTICAL_TOLERANCE := 15
VLM_MODEL := google/gemini-2.5-flash-preview

# Python environment
PYTHON := uv run 
PIP := pip

# Help command
.PHONY: help
help:
	@echo "OCR Benchmarking Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  help              - Show this help message"
	@echo "  install           - Install dependencies"
	@echo "  sample            - Create a sample dataset"
	@echo "  ground-truth      - Generate ground truth from annotations"
	@echo "  ground-truth-vlm  - Generate ground truth using VLM models (requires OPENROUTER_API_KEY)"
	@echo "  benchmark         - Run OCR benchmark"
	@echo "  eval              - Evaluate saved results against ground truth"
	@echo "  eval-vlm          - Evaluate saved results against VLM ground truth"
	@echo "  clean             - Remove generated files"
	@echo "  all               - Run the complete pipeline"
	@echo "  all-vlm           - Run the complete pipeline with VLM ground truth"
	@echo ""
	@echo "Example usage:"
	@echo "  make sample SEED=123 SAMPLE_SIZE=5"
	@echo "  make ground-truth VERTICAL_TOLERANCE=20"
	@echo "  make ground-truth-vlm VLM_MODEL=\"anthropic/claude-3-5-sonnet\""
	@echo "  make benchmark METHODS=\"tesseract easyocr\""

# Install dependencies
.PHONY: install
install:
	$(PIP) install uv
	uv sync

# Create a sample dataset
.PHONY: sample
sample:
	$(PYTHON) sample_dataset.py --source-dir $(SOURCE_DIR) --dest-dir $(SAMPLE_DIR) --sample-size $(SAMPLE_SIZE) --seed $(SEED)

# Generate ground truth from annotations
.PHONY: ground-truth
ground-truth:
	$(PYTHON) generate_ground_truth.py --annotations-dir $(SAMPLE_DIR)/annotations --output-file $(SAMPLE_DIR)/ground_truth.json --vertical-tolerance $(VERTICAL_TOLERANCE)

# Generate ground truth using VLM models
.PHONY: ground-truth-vlm
ground-truth-vlm:
	$(PYTHON) generate_vlm_ground_truth.py --image-dir $(SAMPLE_DIR)/images --output-file $(SAMPLE_DIR)/ground_truth_vlm.json --model $(VLM_MODEL)

.PHONY: benchmark
only-extract-text:
	$(PYTHON) run_benchmark.py --image-dir $(SAMPLE_DIR)/images --ground-truth $(SAMPLE_DIR)/ground_truth.json --methods $(METHODS) --save-results --skip-eval


# Run OCR benchmark
.PHONY: benchmark
benchmark:
	$(PYTHON) run_benchmark.py --image-dir $(SAMPLE_DIR)/images --ground-truth $(SAMPLE_DIR)/ground_truth.json --methods $(METHODS) --save-results

# Evaluate saved results
.PHONY: eval
eval:
	$(PYTHON) run_benchmark.py --eval-only --ground-truth $(SAMPLE_DIR)/ground_truth.json

# Evaluate against VLM ground truth
.PHONY: eval-vlm
eval-vlm:
	$(PYTHON) run_benchmark.py --eval-only --ground-truth $(SAMPLE_DIR)/ground_truth_vlm.json

# Clean generated files
.PHONY: clean
clean:
	rm -rf results/*
	rm -f $(SAMPLE_DIR)/ground_truth.json
	rm -f $(SAMPLE_DIR)/ground_truth_vlm.json
	rm -f $(SAMPLE_DIR)/sample_info.json

# Run complete pipeline
.PHONY: all
all: sample ground-truth benchmark

# Run complete pipeline with VLM ground truth
.PHONY: all-vlm
all-vlm: sample ground-truth-vlm benchmark 