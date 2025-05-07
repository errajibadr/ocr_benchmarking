#!/usr/bin/env python3
"""
OCR Benchmark Runner - Runs comparison between OCR methods and ground truth
"""

import argparse
import os
import sys

from ocr_comparison import (
    evaluate_results,
    load_ground_truth,
    load_ocr_results,
    process_dataset,
    save_extracted_text,
    save_ocr_results,
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

    parser.add_argument(
        "--save-results", action="store_true", help="Save full OCR results for later reuse"
    )

    parser.add_argument(
        "--results-file",
        type=str,
        default="ocr_results.json",
        help="Filename to save/load OCR results",
    )

    parser.add_argument(
        "--load-results",
        action="store_true",
        help="Load OCR results from file instead of processing images",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip processing and only run evaluation on previously saved results",
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip processing and only save extracted text to file",
    )

    return parser.parse_args()


def main():
    """Main entry point for benchmark runner"""
    args = parse_args()

    # Check if loading previous results
    if args.eval_only:
        results_path = os.path.join(args.output_dir, args.results_file)
        if not os.path.exists(results_path):
            print(f"Error: Results file {results_path} does not exist")
            return 1

        # Load previously saved OCR results
        print(f"Loading OCR results from {results_path}...")
        results = load_ocr_results(results_path)

    else:
        # Check if paths exist
        if not os.path.exists(args.image_dir):
            print(f"Error: Image directory {args.image_dir} does not exist")
            return 1

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

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

        # Save individual extracted text results
        print(f"\nSaving extracted text to {args.output_dir}...")
        save_extracted_text(results, args.output_dir)

        # Save complete OCR results if requested
        if args.save_results:
            print(f"\nSaving complete OCR results to {args.results_file}...")
            save_ocr_results(results, args.output_dir, args.results_file)

    # Check if ground truth file exists
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file {args.ground_truth} does not exist")
        return 1

    if not args.skip_eval:
        # Load ground truth data
        print("\nLoading ground truth data...")
        ground_truth = load_ground_truth(args.ground_truth)
        print(f"Loaded ground truth for {len(ground_truth)} images")

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
