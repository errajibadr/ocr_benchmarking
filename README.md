# OCR Benchmarking

This repository contains scripts to benchmark different OCR (Optical Character Recognition) methods against a ground truth dataset generated using Gemini 2.5 Flash.

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

3. Ensure you have Tesseract OCR installed:
   - **Linux**: `apt-get install tesseract-ocr`
   - **MacOS**: `brew install tesseract`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

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

## OCR Methods Included

This benchmark compares the following OCR methods:

1. **Tesseract OCR** - Traditional open-source OCR engine
2. **EasyOCR** - Deep learning based OCR
3. **PaddleOCR** - Efficient OCR system by Baidu
4. **Keras-OCR** - Packaged OCR based on CRAFT text detector and Keras CRNN recognizer

## Evaluation Metrics

The benchmark evaluates OCR methods using:

1. **Text similarity** - How similar the extracted text is to the ground truth using SequenceMatcher
2. **Word Error Rate (WER)** - Measures the edit distance between words (lower is better)
3. **Character Error Rate (CER)** - Measures the edit distance between characters (lower is better) 
4. **Common Word Accuracy** - Percentage of reference words that appear in the extracted text
5. **Processing time** - How long each method takes to process images
6. **Success rate** - Percentage of images successfully processed

## Running the Benchmark

```bash
python run_benchmark.py
```

Or, if you'd like to use specific OCR methods:

```bash
python run_benchmark.py --methods tesseract easyocr
```

## Adding Your Own OCR Methods

To add a new OCR method, edit the `ocr_methods.py` file and add a new function:

```python
def ocr_your_method(image_path: str) -> str:
    """Extract text from image using your method"""
    # Your implementation here
    return extracted_text
```

Then add it to the `OCR_METHODS` dictionary at the bottom of the file.

## Outputs

Results are saved in the `results/` directory:
- JSON files with extracted text for each method
- Visualization plots comparing performance metrics
- Similarity scores compared to ground truth
- A summary table highlighting the best-performing methods
- JSON export of evaluation metrics
