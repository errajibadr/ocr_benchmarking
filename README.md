# OCR Benchmarking

This repository contains scripts to benchmark different OCR (Optical Character Recognition) methods against a ground truth dataset generated using Gemini 2.5 Flash.

## Dataset Structure

The sample dataset should be organized as follows:

```
dataset/
  sample/
    images/
      image1.png
      image2.png
      ...
    ground_truth.json
```

The `ground_truth.json` file maps image filenames to the text extracted by Gemini 2.5 Flash.



## Quick Start with Google Colab

The easiest way to run the comparison is with Google Colab:

1. Open a new notebook in Google Colab: [Open in Colab](https://colab.research.google.com)
2. Copy the contents of `colab_notebook.py` into cells in the notebook
   - Each section marked with `# %%` should be a new cell
   - Markdown sections marked with `# %% [markdown]` should be converted to text cells
3. Run the cells in order to process your images

**Important Note**: `colab_notebook.py` is NOT meant to be run directly as a Python script. It's specially formatted for copying into Google Colab, with cell markers and Colab-specific imports.

## Local Setup

### Prerequisites

- Python 3.8+
- Required libraries (install with `pip install -r requirements.txt`)
- Tesseract OCR installed on your system

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ocr_benchmarking.git
   cd ocr_benchmarking
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. For MMOCR (optional):
   ```bash
   pip install -U openmim
   mim install mmocr
   ```

4. Ensure you have Tesseract OCR installed:
   - **Linux**: `apt-get install tesseract-ocr`
   - **MacOS**: `brew install tesseract`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)



## OCR Methods Included

This benchmark compares the following OCR methods:

1. **Tesseract OCR** - Traditional open-source OCR engine
2. **EasyOCR** - Deep learning based OCR
3. **PaddleOCR** - Efficient OCR system by Baidu
4. **Keras-OCR** - Packaged OCR based on CRAFT text detector and Keras CRNN recognizer
5. **MMOCR** - PyTorch-based OCR toolkit from OpenMMLab
6. **DocTR** - Document Text Recognition from Hugging Face

Additionally, the repository includes adapters for the following cloud OCR services (requires API keys):
- **Azure Computer Vision Read API** - Microsoft's OCR service
- **Amazon Textract** - AWS OCR service for documents

## Evaluation Metrics

The benchmark evaluates OCR methods using:

1. **Text similarity** - How similar the extracted text is to the ground truth using SequenceMatcher
2. **Word Error Rate (WER)** - Measures the edit distance between words (lower is better)
3. **Character Error Rate (CER)** - Measures the edit distance between characters (lower is better) 
4. **Common Word Accuracy** - Percentage of reference words that appear in the extracted text
5. **Processing time** - How long each method takes to process images
6. **Success rate** - Percentage of images successfully processed

## Running the Benchmark

Simple usage:
```bash
python run_benchmark.py
```

With specific OCR methods:
```bash
python run_benchmark.py --methods tesseract easyocr paddleocr 
```

### Saving and Loading OCR Results

You can save OCR results to avoid reprocessing images:

```bash
# Process images and save results
python run_benchmark.py --save-results --results-file my_results.json

# Load saved results and evaluate
python run_benchmark.py --load-results --results-file my_results.json

# Evaluate only (same as --load-results)
python run_benchmark.py --eval-only --results-file my_results.json
```

This is useful for:
- Running different evaluations on the same OCR results
- Sharing OCR results with others without sharing the original images
- Comparing different OCR implementations without reprocessing images

## Adding Your Own OCR Methods

To add a new OCR method, edit the `ocr_methods.py` file and add a new function:

```python
def ocr_your_method(image_path: str) -> str:
    """Extract text from image using your method"""
    # Your implementation here
    return extracted_text
```

Then add it to the `OCR_METHODS` dictionary at the bottom of the file.

## Cloud OCR Services Configuration

To use cloud OCR services, set the required environment variables:

### Azure Computer Vision
```bash
export AZURE_VISION_KEY="your_api_key"
export AZURE_VISION_ENDPOINT="your_endpoint"
```

### Amazon Textract
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION_NAME="your_region"
```

Then uncomment the relevant lines in the `OCR_METHODS` dictionary in `ocr_methods.py`.

## Outputs

Results are saved in the `results/` directory:
- Individual JSON files with extracted text for each method
- Complete OCR results with extracted text and processing times (when using `--save-results`)
- Visualization plots comparing performance metrics
- Similarity scores compared to ground truth
- A summary table highlighting the best-performing methods
- JSON export of evaluation metrics
