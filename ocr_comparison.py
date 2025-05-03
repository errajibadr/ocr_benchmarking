#!/usr/bin/env python3
"""
OCR Comparison Script - Compares different OCR techniques against Gemini ground truth
"""

import json
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


@dataclass
class OCRResult:
    """Stores the result of an OCR method along with performance metrics"""

    method_name: str
    extracted_text: Dict[str, str]
    processing_time: Dict[str, float]


def load_ground_truth(path: str = "dataset/sample/ground_truth.json") -> Dict[str, str]:
    """Load the ground truth data from Gemini API"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_dataset(
    image_dir: str, ocr_methods: List, sample_limit: Optional[int] = None
) -> List[OCRResult]:
    """Process all images in the dataset with each OCR method

    Args:
        image_dir: Directory containing images to process
        ocr_methods: List of OCR method functions to apply
        sample_limit: Optional limit on number of images to process

    Returns:
        List of OCRResult for each method
    """
    results = []
    image_paths = list(pathlib.Path(image_dir).glob("*.png"))

    if sample_limit and sample_limit < len(image_paths):
        image_paths = image_paths[:sample_limit]

    for method in ocr_methods:
        extracted_text = {}
        processing_time = {}

        for img_path in tqdm(image_paths, desc=f"Processing with {method.__name__}"):
            start_time = time.time()
            try:
                text = method(str(img_path))
                elapsed = time.time() - start_time

                extracted_text[img_path.name] = text
                processing_time[img_path.name] = elapsed
            except Exception as e:
                print(f"Error processing {img_path.name} with {method.__name__}: {str(e)}")
                extracted_text[img_path.name] = f"ERROR: {str(e)}"
                processing_time[img_path.name] = -1

        results.append(
            OCRResult(
                method_name=method.__name__,
                extracted_text=extracted_text,
                processing_time=processing_time,
            )
        )

    return results


def evaluate_results(results: List[OCRResult], ground_truth: Dict[str, str]) -> Dict:
    """Evaluates OCR results against ground truth

    For demonstration purposes, this just compares text length and processing time
    """
    evaluation = {}

    for result in results:
        method_stats = {
            "avg_processing_time": 0,
            "text_length_ratio": {},
            "file_counts": {"success": 0, "error": 0},
        }

        # Calculate metrics
        valid_times = [t for t in result.processing_time.values() if t > 0]
        if valid_times:
            method_stats["avg_processing_time"] = sum(valid_times) / len(valid_times)

        for filename, text in result.extracted_text.items():
            if filename in ground_truth and not text.startswith("ERROR:"):
                if ground_truth[filename]:
                    # Simple ratio of extracted text length to ground truth length
                    ratio = len(text) / len(ground_truth[filename])
                    method_stats["text_length_ratio"][filename] = ratio
                    method_stats["file_counts"]["success"] += 1
                else:
                    method_stats["text_length_ratio"][filename] = 0
                    method_stats["file_counts"]["error"] += 1
            else:
                method_stats["file_counts"]["error"] += 1

        evaluation[result.method_name] = method_stats

    return evaluation


def save_results(results: List[OCRResult], output_dir: str = "results"):
    """Save OCR results to json files"""
    os.makedirs(output_dir, exist_ok=True)

    for result in results:
        output_file = os.path.join(output_dir, f"{result.method_name}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.extracted_text, f, indent=2, ensure_ascii=False)

        print(f"Saved results for {result.method_name} to {output_file}")


def visualize_results(evaluation: Dict, output_dir: str = "results"):
    """Create visualizations for OCR comparison"""
    os.makedirs(output_dir, exist_ok=True)

    # Processing time comparison
    methods = list(evaluation.keys())
    proc_times = [evaluation[m]["avg_processing_time"] for m in methods]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=methods, y=proc_times)
    plt.title("Average Processing Time by OCR Method")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "processing_time_comparison.png"))

    # Text length ratio comparison (boxplot)
    plt.figure(figsize=(10, 6))
    data = []
    labels = []

    for method in methods:
        ratios = list(evaluation[method]["text_length_ratio"].values())
        if ratios:  # Only add if we have data
            data.append(ratios)
            labels.append(method)

    if data:
        plt.boxplot(data)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.title("Text Length Ratio (Extracted / Ground Truth)")
        plt.ylabel("Ratio")
        plt.axhline(y=1.0, color="r", linestyle="--")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "text_length_comparison.png"))

    # Success rate
    success_rates = []
    for m in methods:
        success = evaluation[m]["file_counts"]["success"]
        error = evaluation[m]["file_counts"]["error"]
        total = success + error
        if total > 0:
            success_rates.append(success / total)
        else:
            success_rates.append(0)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=methods, y=success_rates)
    plt.title("Success Rate by OCR Method")
    plt.ylabel("Success Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rate_comparison.png"))
