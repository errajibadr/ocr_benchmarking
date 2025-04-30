# OCR Benchmarking

A proof-of-concept project to benchmark various OCR (Optical Character Recognition) techniques on scanned document datasets.

## Datasets used 

OCR datasets repository : https://github.com/xinke-wang/OCRDatasets?tab=readme-ov-file
this project Dataset : https://guillaumejaume.github.io/FUNSD/download/

## Project Overview

This project evaluates and compares different OCR methods:

1. Traditional OCR libraries:
   - Tesseract
   - EasyOCR
   - PaddleOCR

2. Vision Language Models (VLMs):
   - Qwen-2.5-VL
   - OlmOCR
   - Mistral OCR
   - Google Gemini Flash

The project takes a phased approach, exploring one OCR method at a time to thoroughly evaluate its performance, resource requirements, and privacy implications.

## Key Features

- Standardized benchmarking framework for fair comparison
- Privacy-first approach with preference for local processing
- Comprehensive metrics collection (accuracy, speed, resource usage)
- Modular architecture allowing easy addition of new OCR methods

## Setup Instructions

### Prerequisites

- Python 3.12+
- macOS environment (for local testing)
- Google Colab account (for GPU acceleration testing)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ocr_benchmarking.git
   cd ocr_benchmarking
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies using uv:
   ```
   pip install uv
   uv pip install -e .
   ```

### Dataset Setup

Place your scanned document images in the following directories:
- `dataset/training_data/` - For training and calibration
- `dataset/testing_data/` - For benchmark evaluation

## Usage

Each OCR method has its own module and can be run independently:

```python
from src.benchmark.runner import BenchmarkRunner
from src.ocr.traditional.tesseract import TesseractProcessor

# Run benchmark on Tesseract
runner = BenchmarkRunner()
results = runner.run(TesseractProcessor(), dataset_path="dataset/testing_data/")
runner.visualize_results(results)
```

## Project Structure

```
ocr_benchmarking/
├── dataset/                # Document images for benchmarking
│   ├── training_data/      # Training dataset samples
│   └── testing_data/       # Testing dataset samples
├── memory_bank/            # Project memory and context
├── src/
│   ├── benchmark/          # Benchmarking framework
│   ├── ocr/                # OCR implementation modules
│   │   ├── traditional/    # Traditional OCR methods
│   │   └── vlm/            # Vision Language Model methods
│   └── utils/              # Shared utilities
├── notebooks/              # Jupyter notebooks for experiments
├── results/                # Benchmark results
└── tests/                  # Unit tests
```

## License

MIT
