#!/usr/bin/env python3
"""Example script demonstrating Tesseract OCR processing using the benchmarking framework."""

import argparse
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from src.ocr.traditional.tesseract import TesseractProcessor
from src.utils.data_loader import create_ground_truth_template


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Tesseract OCR on a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/testing_data/images",
        help="Path to directory containing images for OCR",
    )
    parser.add_argument(
        "--lang", type=str, default="eng", help="Language(s) for Tesseract OCR (comma-separated)"
    )
    parser.add_argument("--no-preprocess", action="store_true", help="Disable image preprocessing")
    parser.add_argument(
        "--create-ground-truth", action="store_true", help="Create a ground truth template file"
    )
    parser.add_argument("--ground-truth", type=str, help="Path to ground truth JSON file")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit processing to specified number of images (0 for all)",
    )
    parser.add_argument(
        "--no-save-text", action="store_true", help="Don't save extracted text to separate files"
    )

    return parser.parse_args()


def main():
    """Run the example."""
    args = parse_args()

    # Check if dataset directory exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_path}")
        return 1

    # Create ground truth template if requested
    if args.create_ground_truth:
        ground_truth_dir = dataset_path.parent
        output_file = ground_truth_dir / "ground_truth.json"
        create_ground_truth_template(dataset_path, output_file)
        print(f"\nCreated ground truth template: {output_file}")
        print("\nIMPORTANT: For accuracy testing, please:")
        print("1. Edit the ground truth file to add the correct text for each image")
        print("2. Run the benchmark again with: --ground-truth", output_file)
        print("\nFormat of the ground truth file:")
        print("  {")
        print('    "image1.png": "Correct text for image 1",')
        print('    "image2.png": "Correct text for image 2",')
        print("    ...")
        print("  }")
        return 0

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    os.makedirs(results_dir, exist_ok=True)

    # Create benchmark configuration
    config = BenchmarkConfig(
        dataset_path=dataset_path,
        ground_truth_path=Path(args.ground_truth) if args.ground_truth else None,
        save_results=True,
        save_extracted_text=not args.no_save_text,
        results_dir=results_dir,
        verbose=True,
    )

    # Initialize benchmark runner
    runner = BenchmarkRunner(config)

    # Initialize Tesseract OCR processor
    processor = TesseractProcessor(lang=args.lang, preprocess=not args.no_preprocess)

    # Print processor details
    print(f"Running benchmark with {processor.name} OCR")
    print(f"Language: {args.lang}")
    print(f"Preprocessing: {'disabled' if args.no_preprocess else 'enabled'}")
    print(f"Privacy level: {processor.privacy_level}")
    print(
        f"Ground truth: {'None (accuracy metrics will not be calculated)' if not args.ground_truth else args.ground_truth}"
    )

    # Run benchmark
    print(f"\nProcessing images in {dataset_path}...")
    result = runner.run(processor)

    # Print summary
    print("\nBenchmark Results:")
    if args.ground_truth:
        print(f"Average character accuracy: {result.avg_character_accuracy:.4f}")
        print(f"Average word accuracy: {result.avg_word_accuracy:.4f}")
    else:
        print("Accuracy metrics not calculated (no ground truth provided)")
        print("To calculate accuracy, create a ground truth file with --create-ground-truth")

    print(f"Average processing time: {result.avg_processing_time_sec:.2f} seconds per image")
    print(f"Average peak memory usage: {result.avg_peak_memory_mb:.2f} MB")

    print(f"\nResults saved to: {results_dir}")
    if not args.no_save_text:
        print(f"Extracted text saved to: {results_dir}/extracted_texts/")
        print(
            f"You can review the extracted text in: {results_dir}/extracted_texts/{processor.name.lower()}_extracted_text.txt"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
