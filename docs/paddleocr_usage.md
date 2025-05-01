# PaddleOCR Setup and Usage Guide

## Overview

PaddleOCR is a powerful OCR toolkit developed by Baidu, featuring multilingual text detection and recognition capabilities. It provides state-of-the-art models optimized for various scenarios including general text, handwritten text, and document analysis.

## Installation

To use PaddleOCR with this benchmarking framework, you need to install the required packages:

```bash
# Using UV package manager (recommended)
uv pip install paddlepaddle paddleocr

# Or using standard pip
pip install paddlepaddle paddleocr
```

**Note**: For GPU acceleration, you may need to install the GPU version:

```bash
# For CUDA support
uv pip install paddlepaddle-gpu paddleocr
```

## Features

PaddleOCR provides several key advantages:

1. **Multilingual Support**: Recognizes text in 80+ languages
2. **Advanced Capabilities**: Handles complex layouts, tables, and forms
3. **Rotation Detection**: Automatically detects text orientation
4. **Structural Analysis**: Extracts document structure information
5. **Customizable Pipeline**: Configure text detection, recognition, and orientation separately

## Using the PaddleOCR Adapter

### Basic Usage

```python
from src.ocr.traditional.paddleocr import PaddleOCRProcessor

# Initialize with default settings (English language, CPU processing)
processor = PaddleOCRProcessor()

# Process an image
result = processor.process_image("path/to/image.jpg")
print(f"Extracted text: {result.text}")
print(f"Confidence: {result.confidence}")
```

### Advanced Configuration

```python
# Initialize with custom settings
processor = PaddleOCRProcessor(
    langs=["ch"],           # Chinese language
    use_gpu=True,           # Use GPU acceleration
    use_angle_cls=True,     # Enable rotation detection
    preprocess=True,        # Apply preprocessing
    use_mp=True,            # Use multi-processing
    total_process_num=4,    # Number of processes
    show_log=False,         # Hide detailed logs
)
```

## Command-Line Example

The example script provides a convenient way to test PaddleOCR:

```bash
# Basic usage
python examples/paddleocr_example.py --dataset dataset/sample_images

# With GPU acceleration
python examples/paddleocr_example.py --dataset dataset/sample_images --use-gpu

# Multi-language support
python examples/paddleocr_example.py --dataset dataset/sample_images --lang ch

# Create ground truth template
python examples/paddleocr_example.py --dataset dataset/sample_images --create-ground-truth

# Benchmark with ground truth
python examples/paddleocr_example.py --dataset dataset/sample_images --ground-truth dataset/sample_images/ground_truth.json
```

## Available Language Codes

PaddleOCR supports many languages including:

- `en`: English
- `ch`: Chinese (Simplified)
- `japan`: Japanese
- `korean`: Korean
- `fr`: French
- `german`: German
- `it`: Italian
- `es`: Spanish
- `pt`: Portuguese
- `ru`: Russian
- `ar`: Arabic
- `hi`: Hindi
- `ug`: Uighur
- `fa`: Persian
- `ur`: Urdu
- `rs`: Serbian
- `oc`: Occitan
- `mr`: Marathi
- `ne`: Nepali
- `vi`: Vietnamese

## Performance Considerations

- PaddleOCR performs best with GPU acceleration for large datasets
- For complex layouts, enable angle classification (`use_angle_cls=True`)
- Using multi-processing can improve throughput on multi-core systems
- First-time use may be slower due to model downloading

## Comparing with Other OCR Methods

When benchmarking, consider comparing PaddleOCR with other methods like Tesseract and EasyOCR:

```bash
# Run benchmarks for all OCR processors
python examples/tesseract_example.py --dataset dataset/sample_images
python examples/easyocr_example.py --dataset dataset/sample_images
python examples/paddleocr_example.py --dataset dataset/sample_images

# Compare results
python examples/compare_results.py --results-dir results
```

## Troubleshooting

- **ImportError**: Ensure paddlepaddle and paddleocr are installed correctly
- **CUDA Errors**: Check CUDA and cuDNN compatibility if using GPU
- **Memory Issues**: Reduce image size or disable GPU for large images
- **Slow Performance**: Try enabling multi-processing or using GPU 