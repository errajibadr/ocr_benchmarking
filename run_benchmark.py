#!/usr/bin/env python3
"""
OCR Benchmark Runner - Runs comparison between OCR methods and ground truth
"""

import argparse
import os
import pathlib
import sys

from ocr_comparison import (
    evaluate_results,
    load_ground_truth,
    process_dataset,
    save_results,
    visualize_results,
)
from ocr_methods import OCR_METHODS


def parse_args():
    """Parse command line arguments for the benchmark runner"""
    parser = argparse.ArgumentParser(description="OCR Benchmarking Tool")

    parser.add_argument(
        "--image-dir",
        type=str,
        default="dataset/sample/images",
        help="Directory containing images to process",
    )

    parser.add_argument(
        "--ground-truth",
        type=str,
        default="dataset/sample/ground_truth.json",
        help="Path to ground truth JSON file",
    )

    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results"
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["tesseract", "easyocr", "paddleocr", "kerasocr"],
        choices=list(OCR_METHODS.keys()),
        help="OCR methods to benchmark",
    )

    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")

    return parser.parse_args()


def main():
    """Main entry point for benchmark runner"""
    args = parse_args()

    # Check if paths exist
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory {args.image_dir} does not exist")
        return 1

    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file {args.ground_truth} does not exist")
        return 1

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load ground truth data
    print("Loading ground truth data...")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded ground truth for {len(ground_truth)} images")

    # Get selected OCR methods
    selected_methods = []
    for method_name in args.methods:
        if method_name in OCR_METHODS:
            selected_methods.append(OCR_METHODS[method_name])
        else:
            print(f"Warning: OCR method {method_name} not found, skipping")

    if not selected_methods:
        print("Error: No valid OCR methods selected")
        return 1

    # Process images with selected OCR methods
    print(f"\nProcessing images with {len(selected_methods)} OCR methods...")
    results = process_dataset(args.image_dir, selected_methods, args.limit)

    # Save results
    print(f"\nSaving results to {args.output_dir}...")
    save_results(results, args.output_dir)

    # Evaluate results
    print("\nEvaluating results...")
    evaluation = evaluate_results(results, ground_truth)

    # Visualize comparison
    print("\nCreating visualizations...")
    visualize_results(evaluation, args.output_dir)

    print(f"\nAll done! Results saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
