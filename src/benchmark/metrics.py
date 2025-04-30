"""Metrics for evaluating OCR processors."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import Levenshtein
import numpy as np
from pydantic import BaseModel


class MetricType(str, Enum):
    """Types of metrics for OCR evaluation."""

    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    RESOURCE = "resource"


class OCRAccuracyMetrics(BaseModel):
    """Accuracy metrics for OCR evaluation."""

    character_accuracy: float
    word_accuracy: float
    levenshtein_distance: float
    normalized_levenshtein_distance: float


class OCRPerformanceMetrics(BaseModel):
    """Performance metrics for OCR evaluation."""

    processing_time_sec: float
    pages_per_minute: float
    characters_per_second: float


class OCRResourceMetrics(BaseModel):
    """Resource usage metrics for OCR evaluation."""

    peak_memory_usage_mb: float
    avg_cpu_percent: float
    gpu_memory_usage_mb: Optional[float] = None
    avg_gpu_percent: Optional[float] = None


class OCREvaluationResult(BaseModel):
    """Combined evaluation result for an OCR processor."""

    processor_name: str
    accuracy_metrics: OCRAccuracyMetrics
    performance_metrics: OCRPerformanceMetrics
    resource_metrics: OCRResourceMetrics
    total_score: float


def calculate_character_accuracy(ground_truth: str, prediction: str) -> float:
    """Calculate character-level accuracy.

    Args:
        ground_truth: Ground truth text
        prediction: Predicted text

    Returns:
        Character-level accuracy as a float between 0 and 1
    """
    if not ground_truth:
        return 1.0 if not prediction else 0.0

    distance = Levenshtein.distance(ground_truth, prediction)
    max_len = max(len(ground_truth), len(prediction))

    return 1.0 - (distance / max_len) if max_len > 0 else 1.0


def calculate_word_accuracy(ground_truth: str, prediction: str) -> float:
    """Calculate word-level accuracy.

    Args:
        ground_truth: Ground truth text
        prediction: Predicted text

    Returns:
        Word-level accuracy as a float between 0 and 1
    """
    if not ground_truth:
        return 1.0 if not prediction else 0.0

    ground_truth_words = ground_truth.split()
    prediction_words = prediction.split()

    if not ground_truth_words:
        return 1.0 if not prediction_words else 0.0

    # Count correct words
    correct = 0
    ground_truth_word_count = len(ground_truth_words)
    prediction_word_count = len(prediction_words)

    for word in prediction_words:
        if word in ground_truth_words:
            correct += 1
            ground_truth_words.remove(word)  # Remove to handle duplicates correctly

    precision = correct / prediction_word_count if prediction_word_count > 0 else 0
    recall = correct / ground_truth_word_count if ground_truth_word_count > 0 else 0

    # F1 score as the word accuracy
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def evaluate_ocr_accuracy(ground_truth: str, prediction: str) -> OCRAccuracyMetrics:
    """Evaluate OCR accuracy against ground truth.

    Args:
        ground_truth: Ground truth text
        prediction: Predicted text

    Returns:
        OCRAccuracyMetrics object with various accuracy metrics
    """
    character_accuracy = calculate_character_accuracy(ground_truth, prediction)
    word_accuracy = calculate_word_accuracy(ground_truth, prediction)

    levenshtein_distance = Levenshtein.distance(ground_truth, prediction)
    max_len = max(len(ground_truth), len(prediction))
    normalized_levenshtein_distance = levenshtein_distance / max_len if max_len > 0 else 0

    return OCRAccuracyMetrics(
        character_accuracy=character_accuracy,
        word_accuracy=word_accuracy,
        levenshtein_distance=float(levenshtein_distance),
        normalized_levenshtein_distance=normalized_levenshtein_distance,
    )


class MetricsCalculator:
    """Calculator for OCR metrics."""

    @staticmethod
    def calculate_accuracy_metrics(ground_truth: str, prediction: str) -> OCRAccuracyMetrics:
        """Calculate accuracy metrics.

        Args:
            ground_truth: Ground truth text
            prediction: Predicted text

        Returns:
            OCRAccuracyMetrics object
        """
        return evaluate_ocr_accuracy(ground_truth, prediction)

    @staticmethod
    def calculate_performance_metrics(
        processing_time_sec: float, text_length: int, num_pages: int = 1
    ) -> OCRPerformanceMetrics:
        """Calculate performance metrics.

        Args:
            processing_time_sec: Processing time in seconds
            text_length: Length of processed text in characters
            num_pages: Number of pages processed

        Returns:
            OCRPerformanceMetrics object
        """
        pages_per_minute = (num_pages / processing_time_sec) * 60 if processing_time_sec > 0 else 0
        chars_per_second = text_length / processing_time_sec if processing_time_sec > 0 else 0

        return OCRPerformanceMetrics(
            processing_time_sec=processing_time_sec,
            pages_per_minute=pages_per_minute,
            characters_per_second=chars_per_second,
        )

    @staticmethod
    def calculate_resource_metrics(
        peak_memory_mb: float,
        avg_cpu_percent: float,
        gpu_memory_mb: Optional[float] = None,
        avg_gpu_percent: Optional[float] = None,
    ) -> OCRResourceMetrics:
        """Calculate resource usage metrics.

        Args:
            peak_memory_mb: Peak memory usage in MB
            avg_cpu_percent: Average CPU usage percentage
            gpu_memory_mb: GPU memory usage in MB (if applicable)
            avg_gpu_percent: Average GPU usage percentage (if applicable)

        Returns:
            OCRResourceMetrics object
        """
        return OCRResourceMetrics(
            peak_memory_usage_mb=peak_memory_mb,
            avg_cpu_percent=avg_cpu_percent,
            gpu_memory_usage_mb=gpu_memory_mb,
            avg_gpu_percent=avg_gpu_percent,
        )
