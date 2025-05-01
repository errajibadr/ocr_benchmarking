#!/usr/bin/env python3
"""Script for comparing results from different OCR engines."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_result_file(result_file: Path) -> Dict:
    """Load result file and return data as dictionary.

    Args:
        result_file: Path to result JSON file

    Returns:
        Dictionary with result data
    """
    if not result_file.exists():
        print(f"Error: Result file not found: {result_file}")
        sys.exit(1)

    with open(result_file, "r") as f:
        return json.load(f)


def load_extracted_text(text_file: Path) -> Dict[str, str]:
    """Load extracted text from JSON file.

    Args:
        text_file: Path to extracted text JSON file

    Returns:
        Dictionary mapping image names to extracted text
    """
    if not text_file.exists():
        print(f"Error: Extracted text file not found: {text_file}")
        sys.exit(1)

    with open(text_file, "r") as f:
        return json.load(f)


def compare_performance(results: Dict[str, Dict]) -> None:
    """Compare performance metrics between OCR engines.

    Args:
        results: Dictionary mapping OCR engine names to their result data
    """
    performance_data = []

    for engine, data in results.items():
        performance_data.append(
            {
                "OCR Engine": engine,
                "Processing Time (sec/image)": data.get("avg_processing_time_sec", 0),
                "Memory Usage (MB)": data.get("avg_peak_memory_mb", 0),
                "Character Accuracy": data.get("avg_character_accuracy", float("nan")),
                "Word Accuracy": data.get("avg_word_accuracy", float("nan")),
            }
        )

    # Create DataFrame for comparison
    df = pd.DataFrame(performance_data)

    # Set OCR Engine as index
    df = df.set_index("OCR Engine")

    # Display comparison table
    print("\nPerformance Comparison:")
    print(df)

    # Create bar chart for processing time and memory usage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    df["Processing Time (sec/image)"].plot(kind="bar", ax=ax1, color="blue", alpha=0.7)
    ax1.set_title("Processing Time Comparison")
    ax1.set_ylabel("Seconds per Image")

    df["Memory Usage (MB)"].plot(kind="bar", ax=ax2, color="green", alpha=0.7)
    ax2.set_title("Memory Usage Comparison")
    ax2.set_ylabel("MB")

    # Set layout and save figure
    plt.tight_layout()
    plt.savefig("results/performance_comparison.png")
    print("\nPerformance comparison chart saved to: results/performance_comparison.png")

    # Create bar chart for accuracy if available
    if not df["Character Accuracy"].isna().all():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        df["Character Accuracy"].plot(kind="bar", ax=ax1, color="purple", alpha=0.7)
        ax1.set_title("Character Accuracy Comparison")
        ax1.set_ylabel("Accuracy (0-1)")

        df["Word Accuracy"].plot(kind="bar", ax=ax2, color="orange", alpha=0.7)
        ax2.set_title("Word Accuracy Comparison")
        ax2.set_ylabel("Accuracy (0-1)")

        plt.tight_layout()
        plt.savefig("results/accuracy_comparison.png")
        print("Accuracy comparison chart saved to: results/accuracy_comparison.png")


def compare_extracted_text(texts: Dict[str, Dict[str, str]]) -> None:
    """Compare extracted text between OCR engines.

    Args:
        texts: Dictionary mapping OCR engine names to their extracted text dictionaries
    """
    # Get list of images that all engines have processed
    common_images = set.intersection(*[set(engine_texts.keys()) for engine_texts in texts.values()])

    # Create comparison file
    comparison_path = Path("results/text_comparison.txt")

    with open(comparison_path, "w") as f:
        for image in sorted(common_images):
            f.write(f"==== {image} ====\n\n")

            for engine, engine_texts in texts.items():
                f.write(f"{engine} Output:\n")
                f.write(f"{engine_texts.get(image, 'No text extracted')}\n\n")

            f.write("=" * 50 + "\n\n")

    print(f"\nText comparison file saved to: {comparison_path}")


def main():
    """Run the comparison script."""
    parser = argparse.ArgumentParser(description="Compare OCR engine results.")
    parser.add_argument(
        "--engines",
        type=str,
        nargs="+",
        default=["tesseract", "easyocr"],
        help="OCR engine names to compare (default: tesseract easyocr)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory containing result files"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load result data for each engine
    results = {}
    extracted_texts = {}

    for engine in args.engines:
        # Load benchmark results
        result_file = results_dir / f"{engine.lower()}_results.json"
        if result_file.exists():
            results[engine] = load_result_file(result_file)
            print(f"Loaded results for {engine}")
        else:
            print(f"Warning: No result file found for {engine}")

        # Load extracted text
        text_file = results_dir / "extracted_texts" / f"{engine.lower()}_extracted_text.json"
        if text_file.exists():
            extracted_texts[engine] = load_extracted_text(text_file)
            print(f"Loaded extracted text for {engine}")
        else:
            print(f"Warning: No extracted text file found for {engine}")

    if not results:
        print("Error: No result files found for any of the specified engines")
        return 1

    # Perform comparisons
    compare_performance(results)

    if extracted_texts:
        compare_extracted_text(extracted_texts)

    return 0


if __name__ == "__main__":
    sys.exit(main())
