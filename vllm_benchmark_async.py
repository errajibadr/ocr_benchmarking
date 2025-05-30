#!/usr/bin/env python3
"""
VLLM OCR Benchmarking Tool with Async OpenAI Client

Enhanced version using the async OpenAI client for better integration
with VLLM endpoints and automatic token metrics collection.
"""

import asyncio
import base64
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


def setup_logger() -> logging.Logger:
    """Setup logger for the benchmarking module."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


logger = setup_logger()


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""

    base_url: str
    api_key: str
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    concurrent_requests: int = 10
    total_requests: Optional[int] = None
    duration_seconds: Optional[int] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    warmup_requests: int = 5
    max_tokens: int = 2000
    temperature: float = 0.1


@dataclass
class RequestResult:
    """Result of a single OCR request."""

    success: bool
    response_time: float
    response_length: int = 0
    error_message: str = ""
    timestamp: float = field(default_factory=time.time)
    # Token usage information
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class BenchmarkMetrics:
    """Comprehensive metrics from benchmark run."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    response_times: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)
    # Token usage metrics
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second."""
        if self.total_duration == 0:
            return 0.0
        return self.successful_requests / self.total_duration

    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def min_response_time(self) -> float:
        """Calculate minimum response time."""
        if not self.response_times:
            return 0.0
        return min(self.response_times)

    @property
    def max_response_time(self) -> float:
        """Calculate maximum response time."""
        if not self.response_times:
            return 0.0
        return max(self.response_times)

    @property
    def p50_response_time(self) -> float:
        """Calculate 50th percentile response time."""
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)

    @property
    def p95_response_time(self) -> float:
        """Calculate 95th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]

    @property
    def p99_response_time(self) -> float:
        """Calculate 99th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.99 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]

    @property
    def prompt_tokens_per_second(self) -> float:
        """Calculate prompt tokens per second."""
        if self.total_duration == 0:
            return 0.0
        return self.total_prompt_tokens / self.total_duration

    @property
    def completion_tokens_per_second(self) -> float:
        """Calculate completion tokens per second."""
        if self.total_duration == 0:
            return 0.0
        return self.total_completion_tokens / self.total_duration

    @property
    def total_tokens_per_second(self) -> float:
        """Calculate total tokens per second."""
        if self.total_duration == 0:
            return 0.0
        return self.total_tokens / self.total_duration

    @property
    def avg_prompt_tokens_per_request(self) -> float:
        """Calculate average prompt tokens per request."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_prompt_tokens / self.successful_requests

    @property
    def avg_completion_tokens_per_request(self) -> float:
        """Calculate average completion tokens per request."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_completion_tokens / self.successful_requests

    @property
    def avg_total_tokens_per_request(self) -> float:
        """Calculate average total tokens per request."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_tokens / self.successful_requests


class AsyncVLLMOCR:
    """Async VLLM OCR client using OpenAI async client."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize the async OCR client.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.client: Optional[AsyncOpenAI] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.close()

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def estimate_tokens(self, text: str, image_path: str) -> tuple[int, int]:
        """Estimate token counts when not provided by API.

        Args:
            text: The text content to tokenize
            image_path: Path to the image file

        Returns:
            Tuple of (prompt_tokens, completion_tokens)
        """
        try:
            import tiktoken

            # Use GPT-4 tokenizer as approximation
            encoding = tiktoken.encoding_for_model("gpt-4")

            # Estimate prompt tokens (system message + image tokens)
            system_message = (
                "You are a professional OCR assistant. Extract ALL visible text from images "
                "while preserving the original document structure and formatting."
            )
            text_tokens = len(encoding.encode(system_message))

            # Estimate image tokens (typical vision models use ~85-170 tokens per image)
            image_tokens = 150  # Conservative estimate

            prompt_tokens = text_tokens + image_tokens
            completion_tokens = len(encoding.encode(text)) if text else 0

            return prompt_tokens, completion_tokens

        except ImportError:
            # Fallback: rough estimation based on character count
            system_message = (
                "You are a professional OCR assistant. Extract ALL visible text from images "
                "while preserving the original document structure and formatting."
            )
            prompt_tokens = len(system_message) // 4 + 150  # 150 for image
            completion_tokens = len(text) // 4 if text else 0

            return prompt_tokens, completion_tokens

    async def ocr_request(self, image_path: str) -> RequestResult:
        """Perform a single OCR request with retry logic.

        Args:
            image_path: Path to the image file

        Returns:
            RequestResult with timing and success information
        """
        start_time = time.time()

        for attempt in range(self.config.retry_attempts):
            try:
                image_data = self.encode_image(image_path)

                response = await self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional OCR assistant. Your task is to extract ALL visible text "
                                "from images while preserving the original document structure and formatting.\n\n"
                                "**OUTPUT REQUIREMENTS**:\n"
                                "- Extract every piece of readable text including headers, body text, captions, footnotes\n"
                                "- Maintain spatial relationships and reading order\n"
                                "- Use markdown for tables: | Column 1 | Column 2 |\n"
                                "- Preserve lists, numbering, and indentation\n"
                                "- Mark unclear text as [UNCLEAR: best_guess]\n"
                                "- Mark illegible text as [ILLEGIBLE]\n\n"
                                "**QUALITY STANDARDS**: Accuracy over speed, preserve original formatting."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract all text from this image:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                                },
                            ],
                        },
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )

                response_time = time.time() - start_time
                content = response.choices[0].message.content or ""

                # Extract token usage if available
                usage = response.usage
                if usage:
                    prompt_tokens = usage.prompt_tokens
                    completion_tokens = usage.completion_tokens
                    total_tokens = usage.total_tokens
                else:
                    # Fallback to estimation
                    estimated_prompt, estimated_completion = self.estimate_tokens(
                        content, image_path
                    )
                    prompt_tokens = estimated_prompt
                    completion_tokens = estimated_completion
                    total_tokens = prompt_tokens + completion_tokens

                return RequestResult(
                    success=True,
                    response_time=response_time,
                    response_length=len(content),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            except Exception as e:
                response_time = time.time() - start_time
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                return RequestResult(
                    success=False, response_time=response_time, error_message=str(e)
                )

        # Should not reach here, but just in case
        return RequestResult(
            success=False,
            response_time=time.time() - start_time,
            error_message="Max retries exceeded",
        )


class BenchmarkRunner:
    """Main benchmark runner for VLLM OCR testing."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.metrics = BenchmarkMetrics()

    async def warmup(self, image_path: str) -> None:
        """Perform warmup requests to stabilize the service.

        Args:
            image_path: Path to the test image
        """
        logger.info(f"Performing {self.config.warmup_requests} warmup requests...")

        async with AsyncVLLMOCR(self.config) as ocr_client:
            tasks = [ocr_client.ocr_request(image_path) for _ in range(self.config.warmup_requests)]
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Warmup completed")

    async def run_fixed_requests(self, image_path: str) -> BenchmarkMetrics:
        """Run benchmark with a fixed number of requests.

        Args:
            image_path: Path to the test image

        Returns:
            BenchmarkMetrics with test results
        """
        logger.info(
            f"Starting benchmark: {self.config.total_requests} requests with {self.config.concurrent_requests} concurrent"
        )

        start_time = time.time()

        async with AsyncVLLMOCR(self.config) as ocr_client:
            semaphore = asyncio.Semaphore(self.config.concurrent_requests)

            async def limited_request():
                async with semaphore:
                    return await ocr_client.ocr_request(image_path)

            tasks = [limited_request() for _ in range(self.config.total_requests or 0)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()

        return self._process_results(results, end_time - start_time)

    async def run_duration_based(self, image_path: str) -> BenchmarkMetrics:
        """Run benchmark for a specific duration.

        Args:
            image_path: Path to the test image

        Returns:
            BenchmarkMetrics with test results
        """
        logger.info(
            f"Starting benchmark: {self.config.duration_seconds}s duration with {self.config.concurrent_requests} concurrent"
        )

        start_time = time.time()
        end_time = start_time + (self.config.duration_seconds or 0)
        results = []

        async with AsyncVLLMOCR(self.config) as ocr_client:
            semaphore = asyncio.Semaphore(self.config.concurrent_requests)

            async def limited_request():
                async with semaphore:
                    return await ocr_client.ocr_request(image_path)

            while time.time() < end_time:
                batch_size = min(
                    self.config.concurrent_requests,
                    max(1, int((end_time - time.time()) * self.config.concurrent_requests)),
                )

                if batch_size <= 0:
                    break

                tasks = [limited_request() for _ in range(batch_size)]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(batch_results)

                # Small delay to prevent overwhelming the server
                await asyncio.sleep(0.1)

        actual_duration = time.time() - start_time
        return self._process_results(results, actual_duration)

    def _process_results(
        self, results: Sequence[Union[RequestResult, BaseException]], duration: float
    ) -> BenchmarkMetrics:
        """Process benchmark results and calculate metrics.

        Args:
            results: List of request results or exceptions
            duration: Total benchmark duration

        Returns:
            BenchmarkMetrics with calculated statistics
        """
        metrics = BenchmarkMetrics()
        metrics.total_duration = duration
        metrics.total_requests = len(results)

        for result in results:
            if isinstance(result, Exception):
                metrics.failed_requests += 1
                error_type = type(result).__name__
                metrics.error_counts[error_type] = metrics.error_counts.get(error_type, 0) + 1
            elif isinstance(result, RequestResult):
                if result.success:
                    metrics.successful_requests += 1
                    metrics.response_times.append(result.response_time)
                    # Accumulate token usage
                    metrics.total_prompt_tokens += result.prompt_tokens
                    metrics.total_completion_tokens += result.completion_tokens
                    metrics.total_tokens += result.total_tokens
                else:
                    metrics.failed_requests += 1
                    error_type = (
                        result.error_message.split(":")[0]
                        if ":" in result.error_message
                        else result.error_message
                    )
                    metrics.error_counts[error_type] = metrics.error_counts.get(error_type, 0) + 1

        return metrics

    async def run_benchmark(self, image_path: str, skip_warmup: bool = False) -> BenchmarkMetrics:
        """Run the complete benchmark test.

        Args:
            image_path: Path to the test image
            skip_warmup: Whether to skip warmup requests

        Returns:
            BenchmarkMetrics with test results
        """
        # Validate image path
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Warmup
        if not skip_warmup:
            await self.warmup(image_path)

        # Run benchmark
        if self.config.total_requests:
            return await self.run_fixed_requests(image_path)
        elif self.config.duration_seconds:
            return await self.run_duration_based(image_path)
        else:
            raise ValueError("Either total_requests or duration_seconds must be specified")

    def print_results(self, metrics: BenchmarkMetrics) -> None:
        """Print formatted benchmark results.

        Args:
            metrics: Benchmark metrics to display
        """
        print("\n" + "=" * 70)
        print("VLLM OCR BENCHMARK RESULTS (Async OpenAI Client)")
        print("=" * 70)

        print(f"\nOverall Performance:")
        print(f"  Total Requests:      {metrics.total_requests}")
        print(f"  Successful Requests: {metrics.successful_requests}")
        print(f"  Failed Requests:     {metrics.failed_requests}")
        print(f"  Success Rate:        {metrics.success_rate:.2f}%")
        print(f"  Total Duration:      {metrics.total_duration:.2f}s")
        print(f"  Requests/Second:     {metrics.requests_per_second:.2f}")

        if metrics.response_times:
            print(f"\nLatency Statistics (seconds):")
            print(f"  Average:             {metrics.avg_response_time:.3f}")
            print(f"  Minimum:             {metrics.min_response_time:.3f}")
            print(f"  Maximum:             {metrics.max_response_time:.3f}")
            print(f"  50th Percentile:     {metrics.p50_response_time:.3f}")
            print(f"  95th Percentile:     {metrics.p95_response_time:.3f}")
            print(f"  99th Percentile:     {metrics.p99_response_time:.3f}")

        if metrics.total_tokens > 0:
            print(f"\nToken Statistics:")
            print(f"  Total Tokens:        {metrics.total_tokens:,}")
            print(f"  Prompt Tokens:       {metrics.total_prompt_tokens:,}")
            print(f"  Completion Tokens:   {metrics.total_completion_tokens:,}")
            print(f"  Tokens/Second:       {metrics.total_tokens_per_second:.2f}")
            print(f"  Prompt Tokens/s:     {metrics.prompt_tokens_per_second:.2f}")
            print(f"  Completion Tokens/s: {metrics.completion_tokens_per_second:.2f}")
            print(f"\nAverage Tokens per Request:")
            print(f"  Total:               {metrics.avg_total_tokens_per_request:.1f}")
            print(f"  Prompt:              {metrics.avg_prompt_tokens_per_request:.1f}")
            print(f"  Completion:          {metrics.avg_completion_tokens_per_request:.1f}")

        if metrics.error_counts:
            print(f"\nError Summary:")
            for error_type, count in metrics.error_counts.items():
                print(f"  {error_type}: {count}")

        print("=" * 70)

    def save_results(self, metrics: BenchmarkMetrics, output_path: str) -> None:
        """Save benchmark results to JSON file.

        Args:
            metrics: Benchmark metrics to save
            output_path: Path to save the results
        """
        results_data = {
            "config": {
                "base_url": self.config.base_url,
                "model_name": self.config.model_name,
                "concurrent_requests": self.config.concurrent_requests,
                "total_requests": self.config.total_requests,
                "duration_seconds": self.config.duration_seconds,
                "timeout_seconds": self.config.timeout_seconds,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
            "metrics": {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "success_rate": metrics.success_rate,
                "total_duration": metrics.total_duration,
                "requests_per_second": metrics.requests_per_second,
                "avg_response_time": metrics.avg_response_time,
                "min_response_time": metrics.min_response_time,
                "max_response_time": metrics.max_response_time,
                "p50_response_time": metrics.p50_response_time,
                "p95_response_time": metrics.p95_response_time,
                "p99_response_time": metrics.p99_response_time,
                "error_counts": metrics.error_counts,
                "response_times": metrics.response_times,
                # Token metrics
                "total_tokens": metrics.total_tokens,
                "total_prompt_tokens": metrics.total_prompt_tokens,
                "total_completion_tokens": metrics.total_completion_tokens,
                "total_tokens_per_second": metrics.total_tokens_per_second,
                "prompt_tokens_per_second": metrics.prompt_tokens_per_second,
                "completion_tokens_per_second": metrics.completion_tokens_per_second,
                "avg_total_tokens_per_request": metrics.avg_total_tokens_per_request,
                "avg_prompt_tokens_per_request": metrics.avg_prompt_tokens_per_request,
                "avg_completion_tokens_per_request": metrics.avg_completion_tokens_per_request,
            },
            "timestamp": time.time(),
            "client_type": "async_openai",
        }

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to {output_path}")


async def main():
    """Example usage of the async benchmark runner."""
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv(override=True)

    # Configuration
    base_url = os.getenv("RUNPOD_BASE_URL")
    api_key = os.getenv("RUNPOD_API_KEY")

    if not base_url or not api_key:
        logger.error("RUNPOD_BASE_URL and RUNPOD_API_KEY environment variables must be set")
        return

    config = BenchmarkConfig(
        base_url=base_url,
        api_key=api_key,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        concurrent_requests=5,
        total_requests=20,
        timeout_seconds=30,
        max_tokens=2000,
        temperature=0.1,
    )

    # Run benchmark
    runner = BenchmarkRunner(config)

    try:
        metrics = await runner.run_benchmark("dataset/sample/images/82200067_0069.png")
        runner.print_results(metrics)
        runner.save_results(metrics, "async_benchmark_results.json")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
