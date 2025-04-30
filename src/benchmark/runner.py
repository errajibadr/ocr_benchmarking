"""Benchmark runner for OCR processors."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
from pydantic import BaseModel

from src.benchmark.metrics import (
    MetricsCalculator,
    OCRAccuracyMetrics,
    OCREvaluationResult,
    OCRPerformanceMetrics,
    OCRResourceMetrics,
)
from src.ocr.base import OCRProcessor, OCRResult
from src.utils.data_loader import load_ground_truth, load_images


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark runs."""

    dataset_path: Path
    ground_truth_path: Optional[Path] = None
    save_results: bool = True
    save_extracted_text: bool = True
    results_dir: Path = Path("results")
    verbose: bool = True


class BenchmarkResult(BaseModel):
    """Result of a benchmark run."""

    processor_name: str
    results: List[OCREvaluationResult]
    avg_character_accuracy: float
    avg_word_accuracy: float
    avg_processing_time_sec: float
    avg_peak_memory_mb: float
    metadata: Dict = {}
    extracted_texts: Dict[str, str] = {}


class BenchmarkRunner:
    """Runner for OCR benchmarks."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize benchmark runner.

        Args:
            config: Optional configuration for the benchmark run
        """
        self.config = config or BenchmarkConfig(dataset_path=Path("dataset/testing_data"))
        os.makedirs(self.config.results_dir, exist_ok=True)

    def run(
        self, processor: OCRProcessor, dataset_path: Optional[Union[str, Path]] = None
    ) -> BenchmarkResult:
        """Run benchmark on an OCR processor.

        Args:
            processor: OCR processor to benchmark
            dataset_path: Optional override for dataset path

        Returns:
            BenchmarkResult object with evaluation metrics
        """
        if dataset_path:
            self.config.dataset_path = Path(dataset_path)

        if self.config.verbose:
            print(f"Running benchmark for {processor.name}...")
            print(f"Dataset path: {self.config.dataset_path}")

        # Load images and ground truth data
        images = load_images(self.config.dataset_path)
        ground_truths = {}

        if self.config.ground_truth_path:
            ground_truths = load_ground_truth(self.config.ground_truth_path)

        evaluation_results = []
        extracted_texts = {}

        # Process each image and evaluate
        for image_path in images:
            # Monitor memory and CPU usage
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB

            # Process the image
            start_time = time.time()
            result = processor.process_image(image_path)
            end_time = time.time()

            # Store extracted text
            image_name = os.path.basename(image_path)
            extracted_texts[image_name] = result.text

            # Measure resource usage
            end_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = end_memory - start_memory

            # Get ground truth if available
            ground_truth = ground_truths.get(image_name, "")

            # Calculate metrics
            accuracy_metrics = (
                MetricsCalculator.calculate_accuracy_metrics(ground_truth, result.text)
                if ground_truth
                else OCRAccuracyMetrics(
                    character_accuracy=0.0,
                    word_accuracy=0.0,
                    levenshtein_distance=0.0,
                    normalized_levenshtein_distance=0.0,
                )
            )

            performance_metrics = MetricsCalculator.calculate_performance_metrics(
                result.processing_time_sec, len(result.text)
            )

            resource_metrics = OCRResourceMetrics(
                peak_memory_usage_mb=peak_memory,
                avg_cpu_percent=psutil.cpu_percent(),
                gpu_memory_usage_mb=None,
                avg_gpu_percent=None,
            )

            # Calculate total score (weighted average of metrics)
            total_score = (
                accuracy_metrics.character_accuracy * 0.4
                + accuracy_metrics.word_accuracy * 0.4
                + (1.0 - performance_metrics.processing_time_sec / 10.0) * 0.2
            )
            total_score = max(0.0, min(1.0, total_score))  # Clamp to [0, 1]

            evaluation_result = OCREvaluationResult(
                processor_name=processor.name,
                accuracy_metrics=accuracy_metrics,
                performance_metrics=performance_metrics,
                resource_metrics=resource_metrics,
                total_score=total_score,
            )

            evaluation_results.append(evaluation_result)

            if self.config.verbose:
                print(f"Processed {image_name}: Score = {total_score:.4f}")

        # Calculate averages for summary
        avg_char_accuracy = np.mean(
            [r.accuracy_metrics.character_accuracy for r in evaluation_results]
        )
        avg_word_accuracy = np.mean([r.accuracy_metrics.word_accuracy for r in evaluation_results])
        avg_time = np.mean([r.performance_metrics.processing_time_sec for r in evaluation_results])
        avg_memory = np.mean([r.resource_metrics.peak_memory_usage_mb for r in evaluation_results])

        # Create benchmark result
        benchmark_result = BenchmarkResult(
            processor_name=processor.name,
            results=evaluation_results,
            avg_character_accuracy=float(avg_char_accuracy),
            avg_word_accuracy=float(avg_word_accuracy),
            avg_processing_time_sec=float(avg_time),
            avg_peak_memory_mb=float(avg_memory),
            metadata={
                "capabilities": processor.get_capabilities().model_dump(),
                "resource_requirements": processor.get_resource_requirements().model_dump(),
                "privacy_level": processor.privacy_level,
            },
            extracted_texts=extracted_texts,
        )

        # Save results if configured
        if self.config.save_results:
            self._save_results(benchmark_result)

        # Save extracted text to a separate file for easier review
        if self.config.save_extracted_text:
            self._save_extracted_text(benchmark_result)

        return benchmark_result

    def compare(self, results: List[BenchmarkResult]) -> Dict:
        """Compare multiple benchmark results.

        Args:
            results: List of BenchmarkResult objects to compare

        Returns:
            Dictionary with comparison data
        """
        comparison = {
            "processors": [r.processor_name for r in results],
            "accuracy": {
                "character": [r.avg_character_accuracy for r in results],
                "word": [r.avg_word_accuracy for r in results],
            },
            "performance": {
                "time": [r.avg_processing_time_sec for r in results],
                "memory": [r.avg_peak_memory_mb for r in results],
            },
            "privacy_levels": [r.metadata.get("privacy_level") for r in results],
        }

        return comparison

    def _save_results(self, result: BenchmarkResult) -> None:
        """Save benchmark results to file.

        Args:
            result: BenchmarkResult to save
        """
        result_path = self.config.results_dir / f"{result.processor_name.lower()}_results.json"

        # Use model_dump() with Pydantic v2 instead of json() with kwargs
        result_dict = result.model_dump()

        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        if self.config.verbose:
            print(f"Results saved to {result_path}")

    def _save_extracted_text(self, result: BenchmarkResult) -> None:
        """Save extracted text to a readable file.

        Args:
            result: BenchmarkResult containing extracted texts
        """
        # Create a text file for extracted texts
        text_dir = self.config.results_dir / "extracted_texts"
        os.makedirs(text_dir, exist_ok=True)

        text_path = text_dir / f"{result.processor_name.lower()}_extracted_text.txt"

        # Save in a readable format with clear separation between images
        with open(text_path, "w", encoding="utf-8") as f:
            for image_name, text in result.extracted_texts.items():
                f.write(f"==== {image_name} ====\n\n")
                f.write(text.strip() if text else "[No text extracted]")
                f.write("\n\n" + "=" * 50 + "\n\n")

        # Also save as JSON for programmatic access
        json_path = text_dir / f"{result.processor_name.lower()}_extracted_text.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.extracted_texts, f, indent=2, ensure_ascii=False)

        if self.config.verbose:
            print(f"Extracted texts saved to {text_path}")
            print(f"Extracted texts saved as JSON to {json_path}")

    def visualize_results(self, result: Union[BenchmarkResult, List[BenchmarkResult]]) -> None:
        """Visualize benchmark results.

        Args:
            result: BenchmarkResult or list of BenchmarkResult objects
        """
        # This is a placeholder for visualization functionality
        # In a real implementation, this would generate charts and graphs
        print("Visualization functionality not implemented yet.")
