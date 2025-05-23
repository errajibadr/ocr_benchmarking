# OCR Benchmarking

A comprehensive toolkit for benchmarking various OCR (Optical Character Recognition) methods against ground truth data from FUNSD dataset annotations.

## Overview

This repository contains tools to:
1. Sample datasets for reproducible benchmarking
2. Generate ground truth from FUNSD document annotations 
3. Run and evaluate various OCR methods
4. Visualize and compare results

## Dataset

The benchmark uses the [FUNSD (Form Understanding in Noisy Scanned Documents)](https://guillaumejaume.github.io/FUNSD/) dataset, which consists of noisy scanned forms with annotations for text, layout, and form understanding tasks.

### Dataset Structure

The full dataset is organized as follows:

```
dataset/
  testing_data/
    images/         # PNG files of scanned forms
    annotations/    # JSON annotations with text fields, bounding boxes, etc.
```

The benchmarking toolkit can create reproducible samples from this dataset:

```
dataset/
  sample/
    images/         # Sampled subset of images
    annotations/    # Corresponding annotations
    ground_truth.json  # Extracted text for evaluation
    sample_info.json   # Sampling parameters for reproducibility
```

## OCR Methods

This benchmark compares the following OCR methods:

| Method | Type | Description | Requirements |
|--------|------|-------------|-------------|
| Tesseract | Traditional | Industry-standard open-source OCR engine | `pip install pytesseract opencv-python` |
| EasyOCR | Deep Learning | Multi-language OCR using CRAFT text detector and CRNN recognizer | `pip install easyocr` |
| PaddleOCR | Deep Learning | Efficient OCR system by Baidu | `pip install paddlepaddle paddleocr` |
| DocTR | Deep Learning | Document Text Recognition from Hugging Face | `pip install python-doctr` |
| Docling | Deep Learning | Document processor with OCR capabilities | `pip install docling` |
| KerasOCR | Deep Learning | Uses CRAFT text detector and Keras CRNN recognizer | `pip install keras-ocr` |
| Amazon Textract | Cloud API | AWS OCR service for documents | AWS credentials + `pip install boto3` |
| VLM Models | Vision-Language | OpenRouter vision models (Qwen, Mistral, Pixtral) | OpenRouter API key + `pip install openai` |

## Quick Start

### Prerequisites

- Python 3.8+
- Tesseract OCR installed on your system if using that method
- Required Python packages (install with `pip install -r requirements.txt`)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ocr_benchmarking.git
   cd ocr_benchmarking
   ```

2. Install dependencies:
   ```bash
   make install
   ```

### Creating a Sample Dataset

To create a reproducible sample of the dataset:

```bash
make sample SEED=42 SAMPLE_SIZE=10
```

This will randomly select 10 images from the dataset using seed 42 for reproducibility.

### Generating Ground Truth

Generate ground truth text from the FUNSD annotations:

```bash
make ground-truth
```

For more control over how text is extracted from annotations, you can run the script directly:

```bash
python generate_ground_truth.py --annotations-dir dataset/sample/annotations --output-file dataset/sample/ground_truth.json --vertical-tolerance 15
```

The `--vertical-tolerance` parameter controls how text elements are grouped into lines based on their vertical position. A higher value will group more elements into the same line, while a lower value will create more separate lines.

Use the `--debug` flag to see detailed information about how text elements are grouped into lines:

```bash
python generate_ground_truth.py --debug
```

### Using VLM Models as Ground Truth

You can also generate ground truth using Vision-Language Models (VLMs) via OpenRouter:

```bash
make ground-truth-vlm
```

This requires an OpenRouter API key set in your environment:

```bash
export OPENROUTER_API_KEY="your_api_key"
```

You can specify a different VLM model:

```bash
make ground-truth-vlm VLM_MODEL="anthropic/claude-3-5-sonnet"
```

Or run the script directly with more options:

```bash
python generate_vlm_ground_truth.py --image-dir dataset/sample/images --output-file dataset/sample/ground_truth_vlm.json --model "anthropic/claude-3-5-sonnet" --retries 3
```

### Comparing Different Ground Truth Sources

You can evaluate OCR methods against different ground truth sources:

```bash
# Evaluate against annotation-based ground truth
make eval

# Evaluate against VLM-based ground truth
make eval-vlm
```

This allows you to compare how different OCR methods perform against different ground truth standards.

### Running the Benchmark

Run the benchmark with selected OCR methods:

```bash
make benchmark METHODS="tesseract easyocr paddleocr"
```

This will automatically save the results to a file named based on your sample directory (e.g., `results/result_sample_test.json`).

### Extracting Text Only (Without Evaluation)

If you just want to extract text from images without evaluating against a ground truth:

```bash
make only-extract-text METHODS="tesseract"
```

This will save the results to a file named `result_<sample_name>.json` in the results directory. For example, if your sample directory is `dataset/sample_test`, the results will be saved to `results/result_sample_test.json`.

You can specify a different sample directory:

```bash
make only-extract-text METHODS="tesseract" SAMPLE_DIR="dataset/my_custom_sample"
```

This is useful when you want to:
- Process images with OCR methods without having ground truth available
- Generate OCR results to share with others
- Batch process a set of images with multiple OCR methods

### Evaluating Results

To evaluate previously saved results against a ground truth:

```bash
make eval
```

By default, this uses the results file based on your sample directory name. You can specify a different sample directory or results file:

```bash
# Evaluate results for a specific sample directory
make eval SAMPLE_DIR="dataset/my_custom_sample"

# Evaluate a specific results file against the ground truth
make eval RESULT_FILE="result_custom.json"
```

To evaluate against VLM-generated ground truth instead:

```bash
make eval-vlm
```

The same options apply for specifying custom sample directories or result files:

```bash
make eval-vlm SAMPLE_DIR="dataset/my_custom_sample"
make eval-vlm RESULT_FILE="result_custom.json"
```

### Full Pipeline

Run the entire pipeline (sample, ground truth, and benchmark):

```bash
make all
```

## Advanced Usage

### Manually Running Individual Steps

#### 1. Sample Dataset

```bash
python sample_dataset.py --source-dir dataset/testing_data --dest-dir dataset/sample --sample-size 10 --seed 42
```

#### 2. Generate Ground Truth

```bash
python generate_ground_truth.py --annotations-dir dataset/sample/annotations --output-file dataset/sample/ground_truth.json
```

#### 3. Run Benchmark

```bash
python run_benchmark.py --image-dir dataset/sample/images --ground-truth dataset/sample/ground_truth.json --methods tesseract easyocr paddleocr
```

### Saving and Loading Results

You can save OCR results to avoid reprocessing images:

```bash
python run_benchmark.py --save-results --results-file my_results.json
```

Load saved results and evaluate:

```bash
python run_benchmark.py --load-results --results-file my_results.json
```

Evaluate only (same as --load-results):

```bash
python run_benchmark.py --eval-only --results-file my_results.json
```

## Evaluation Metrics

The benchmark evaluates OCR methods using several complementary metrics to provide a comprehensive understanding of performance:

### Text Similarity

**Description**: Measures how similar the extracted text is to the ground truth using Python's `difflib.SequenceMatcher`.

**Calculation**: The ratio of matching elements to the total number of elements in both sequences, returning a value between 0.0 (no similarity) and 1.0 (perfect match).

**Use case**: Provides a general measure of overall text similarity that accounts for additions, deletions, and substitutions.

### Word Error Rate (WER)

**Description**: A common metric in speech recognition and OCR that measures the edit distance between words.

**Calculation**: Calculated as:
```
WER = (S + D + I) / N
```
Where:
- S = number of substituted words
- D = number of deleted words
- I = number of inserted words
- N = total number of words in the reference text

**Use case**: Lower is better. WER is particularly useful for assessing how many word-level corrections would be needed to transform the OCR output into the ground truth.

### Character Error Rate (CER)

**Description**: Similar to WER but at the character level, which provides finer-grained assessment.

**Calculation**: Calculated as:
```
CER = (S + D + I) / N
```
Where:
- S = number of substituted characters
- D = number of deleted characters
- I = number of inserted characters
- N = total number of characters in the reference text

**Use case**: Lower is better. CER is useful for languages where word boundaries are not clear or when character-level accuracy is important.

### Common Word Accuracy

**Description**: Percentage of reference words that appear in the extracted text.

**Calculation**: The ratio of words from the ground truth that appear in the OCR output, regardless of order or frequency.

**Use case**: Useful for scenarios where the presence of key terms is more important than their exact positioning or order.

### Processing Time

**Description**: How long each method takes to process images.

**Calculation**: Measured in seconds per image.

**Use case**: Important for real-time applications or when processing large volumes of documents.


## Interpreting Results

When interpreting the benchmark results, consider:

1. **Task-specific priorities**: For some applications, accuracy might be more important than speed, while for others, the inverse might be true.

2. **Document type sensitivity**: Some OCR methods perform better on certain document types (handwritten, printed, forms, etc.).

3. **Language considerations**: Performance can vary significantly depending on the language and script.

4. **Error patterns**: Look beyond the raw metrics to understand the types of errors each method makes.

5. **Ground truth quality**: Remember that the evaluation is only as good as the ground truth it's compared against. FUNSD annotations and VLM-generated ground truths may have different characteristics.

## Visualization

The benchmark generates several visualizations to help interpret results:

1. **Comparison charts**: Bar charts comparing all OCR methods across each metric.

2. **Heatmaps**: Show where each method excels or struggles.

3. **Time vs. Accuracy plots**: Help identify optimal methods balancing speed and accuracy.

4. **Per-image results**: Detailed metrics for each image to identify patterns based on document type or complexity.

## Adding Your Own OCR Methods

To add a new OCR method, edit the `ocr_methods.py` file and add a function with the following signature:

```python
def ocr_your_method(image_path: str) -> str:
    """Extract text from image using your method
    
    Installation: !pip install your-requirements
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text from the image
    """
    # Your implementation here
    return extracted_text
```

Then add it to the `OCR_METHODS` dictionary at the bottom of the file:

```python
OCR_METHODS = {
    # Existing methods...
    "your_method": ocr_your_method,
}
```

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

### OpenRouter (for Vision-Language Models)
```bash
export OPENROUTER_API_KEY="your_api_key"
```

## Results

Results are saved in the `results/` directory:
- Individual JSON files with extracted text for each method
- Complete OCR results with extracted text and processing times
- Visualization plots comparing performance metrics
- Similarity scores compared to ground truth
- A summary table highlighting the best-performing methods
- JSON export of evaluation metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/) - For providing the dataset of forms
- Various OCR libraries and their maintainers
