#!/usr/bin/env python3
"""
OCR Comparison Script - Compares different OCR techniques against Gemini ground truth
"""

import json
import os
import pathlib
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
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


def text_similarity(text1: str, text2: str) -> float:
    """Compute sequence similarity ratio between two strings"""
    return SequenceMatcher(None, text1, text2).ratio()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis texts

    WER = (S + D + I) / N
    where:
    S is the number of substitutions
    D is the number of deletions
    I is the number of insertions
    N is the number of words in the reference
    """
    # Normalize and split into words
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Initialize the distance matrix
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)

    # Source prefixes can be transformed into empty string by dropping all chars
    for i in range(len(ref_words) + 1):
        d[i, 0] = i

    # Target prefixes can be reached from empty source by inserting chars
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    # Fill the distance matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,  # Deletion
                d[i, j - 1] + 1,  # Insertion
                d[i - 1, j - 1] + cost,  # Substitution
            )

    # The last element contains the edit distance
    edit_distance = d[len(ref_words), len(hyp_words)]

    # Calculate WER (avoid division by zero)
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    return min(1.0, edit_distance / len(ref_words))


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER) between reference and hypothesis texts

    CER = (S + D + I) / N
    where:
    S is the number of character substitutions
    D is the number of character deletions
    I is the number of character insertions
    N is the number of characters in the reference
    """
    # Normalize
    ref_chars = reference.lower()
    hyp_chars = hypothesis.lower()

    # Initialize the distance matrix
    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1), dtype=np.int32)

    # Source prefixes can be transformed into empty string by dropping all chars
    for i in range(len(ref_chars) + 1):
        d[i, 0] = i

    # Target prefixes can be reached from empty source by inserting chars
    for j in range(len(hyp_chars) + 1):
        d[0, j] = j

    # Fill the distance matrix
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,  # Deletion
                d[i, j - 1] + 1,  # Insertion
                d[i - 1, j - 1] + cost,  # Substitution
            )

    # The last element contains the edit distance
    edit_distance = d[len(ref_chars), len(hyp_chars)]

    # Calculate CER (avoid division by zero)
    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0

    return min(1.0, edit_distance / len(ref_chars))


def common_word_accuracy(reference: str, hypothesis: str) -> float:
    """Calculate the percentage of words in reference that also appear in hypothesis"""
    # Normalize and split into words
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())

    if not ref_words:
        return 0.0

    # Count words in common
    common_words = ref_words.intersection(hyp_words)

    return len(common_words) / len(ref_words)


def normalize_text(text: str) -> str:
    """Normalize text for better comparison

    - Lowercase
    - Remove extra whitespace
    - Remove punctuation
    """
    import re

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def evaluate_results(results: List[OCRResult], ground_truth: Dict[str, str]) -> Dict:
    """Evaluates OCR results against ground truth using multiple metrics

    Metrics:
    - Text similarity ratio
    - Word Error Rate (WER)
    - Character Error Rate (CER)
    - Common Word Accuracy
    - Text length ratio
    - Processing time
    """
    evaluation = {}

    for result in results:
        method_stats = {
            "avg_processing_time": 0,
            "text_length_ratio": {},
            "text_similarity": {},
            "word_error_rate": {},
            "character_error_rate": {},
            "common_word_accuracy": {},
            "file_counts": {"success": 0, "error": 0},
            "avg_metrics": {"similarity": 0, "wer": 0, "cer": 0, "word_accuracy": 0},
        }

        # Calculate metrics
        valid_times = [t for t in result.processing_time.values() if t > 0]
        if valid_times:
            method_stats["avg_processing_time"] = sum(valid_times) / len(valid_times)

        valid_files = []

        for filename, extracted_text in result.extracted_text.items():
            if filename in ground_truth and not extracted_text.startswith("ERROR:"):
                truth = ground_truth[filename]
                if truth:  # Ensure ground truth is not empty
                    # Calculate metrics

                    # Normalize texts for better comparison
                    norm_extracted = normalize_text(extracted_text)
                    norm_truth = normalize_text(truth)

                    # Basic ratio
                    method_stats["text_length_ratio"][filename] = len(extracted_text) / len(truth)

                    # Text similarity (SequenceMatcher)
                    method_stats["text_similarity"][filename] = text_similarity(
                        norm_extracted, norm_truth
                    )

                    # Word Error Rate
                    method_stats["word_error_rate"][filename] = word_error_rate(
                        norm_truth, norm_extracted
                    )

                    # Character Error Rate
                    method_stats["character_error_rate"][filename] = character_error_rate(
                        norm_truth, norm_extracted
                    )

                    # Common Word Accuracy
                    method_stats["common_word_accuracy"][filename] = common_word_accuracy(
                        norm_truth, norm_extracted
                    )

                    method_stats["file_counts"]["success"] += 1
                    valid_files.append(filename)
                else:
                    method_stats["file_counts"]["error"] += 1
            else:
                method_stats["file_counts"]["error"] += 1

        # Calculate average metrics if we have valid files
        if valid_files:
            method_stats["avg_metrics"]["similarity"] = np.mean(
                [method_stats["text_similarity"][f] for f in valid_files]
            )
            method_stats["avg_metrics"]["wer"] = np.mean(
                [method_stats["word_error_rate"][f] for f in valid_files]
            )
            method_stats["avg_metrics"]["cer"] = np.mean(
                [method_stats["character_error_rate"][f] for f in valid_files]
            )
            method_stats["avg_metrics"]["word_accuracy"] = np.mean(
                [method_stats["common_word_accuracy"][f] for f in valid_files]
            )

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
    plt.show()

    # Advanced metrics comparison (4 metrics in a 2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Text similarity (higher is better)
    similarity_scores = [evaluation[m]["avg_metrics"]["similarity"] for m in methods]
    sns.barplot(x=methods, y=similarity_scores, ax=axes[0, 0], palette="Blues_d")
    axes[0, 0].set_title("Text Similarity (higher is better)")
    axes[0, 0].set_ylabel("Similarity Score (0-1)")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Word Error Rate (lower is better)
    wer_scores = [evaluation[m]["avg_metrics"]["wer"] for m in methods]
    sns.barplot(x=methods, y=wer_scores, ax=axes[0, 1], palette="Reds_d")
    axes[0, 1].set_title("Word Error Rate (lower is better)")
    axes[0, 1].set_ylabel("WER (0-1)")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Character Error Rate (lower is better)
    cer_scores = [evaluation[m]["avg_metrics"]["cer"] for m in methods]
    sns.barplot(x=methods, y=cer_scores, ax=axes[1, 0], palette="Reds_d")
    axes[1, 0].set_title("Character Error Rate (lower is better)")
    axes[1, 0].set_ylabel("CER (0-1)")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Common Word Accuracy (higher is better)
    word_acc_scores = [evaluation[m]["avg_metrics"]["word_accuracy"] for m in methods]
    sns.barplot(x=methods, y=word_acc_scores, ax=axes[1, 1], palette="Greens_d")
    axes[1, 1].set_title("Common Word Accuracy (higher is better)")
    axes[1, 1].set_ylabel("Word Accuracy (0-1)")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "advanced_metrics_comparison.png"))
    plt.show()

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
        plt.show()

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
    plt.show()

    # Create a summary table visualization
    summary_data = {
        "Method": methods,
        "Similarity": [round(evaluation[m]["avg_metrics"]["similarity"], 3) for m in methods],
        "WER": [round(evaluation[m]["avg_metrics"]["wer"], 3) for m in methods],
        "CER": [round(evaluation[m]["avg_metrics"]["cer"], 3) for m in methods],
        "Word Accuracy": [round(evaluation[m]["avg_metrics"]["word_accuracy"], 3) for m in methods],
        "Success Rate": [round(rate, 3) for rate in success_rates],
        "Avg Time (s)": [round(evaluation[m]["avg_processing_time"], 3) for m in methods],
    }

    fig, ax = plt.subplots(figsize=(12, len(methods) * 0.8 + 1.5))
    ax.axis("off")

    # Create table
    col_labels = list(summary_data.keys())
    table_data = []
    for i in range(len(methods)):
        row = []
        for col in col_labels:
            row.append(summary_data[col][i])
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color best values
    for col in ["Similarity", "Word Accuracy"]:
        # Higher is better
        col_idx = col_labels.index(col)
        values = [summary_data[col][i] for i in range(len(methods))]
        best_idx = int(np.argmax(values))
        cell = table.get_celld()[(best_idx + 1, col_idx)]
        cell.set_facecolor("#d6ffe8")  # Light green

    for col in ["WER", "CER"]:
        # Lower is better
        col_idx = col_labels.index(col)
        values = [summary_data[col][i] for i in range(len(methods))]
        best_idx = int(np.argmin(values))
        cell = table.get_celld()[(best_idx + 1, col_idx)]
        cell.set_facecolor("#d6ffe8")  # Light green

    plt.title("OCR Methods Comparison Summary", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_table.png"), bbox_inches="tight")
    plt.show()

    # Save the evaluation results as JSON
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                m: {
                    "avg_metrics": evaluation[m]["avg_metrics"],
                    "success_rate": success_rates[i],
                    "avg_processing_time": evaluation[m]["avg_processing_time"],
                }
                for i, m in enumerate(methods)
            },
            f,
            indent=2,
        )
