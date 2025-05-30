#!/usr/bin/env python3
"""
CLI interface for VLLM OCR Benchmarking Tool

Provides easy command-line access to benchmark VLLM OCR endpoints
for latency and throughput testing.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from vllm_benchmark import BenchmarkConfig, BenchmarkRunner

load_dotenv(override=True)


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Setup logger with appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def validate_environment() -> tuple[str, str]:
    """Validate required environment variables.

    Returns:
        Tuple of (base_url, api_key)

    Raises:
        SystemExit: If required environment variables are missing
    """
    base_url = os.getenv("RUNPOD_BASE_URL")
    api_key = os.getenv("RUNPOD_API_KEY")

    if not base_url:
        print("ERROR: RUNPOD_BASE_URL environment variable is required")
        print("Set it with: export RUNPOD_BASE_URL='your_endpoint_url'")
        sys.exit(1)

    if not api_key:
        print("ERROR: RUNPOD_API_KEY environment variable is required")
        print("Set it with: export RUNPOD_API_KEY='your_api_key'")
        sys.exit(1)

    return base_url, api_key


def validate_image_path(image_path: str) -> str:
    """Validate image path exists.

    Args:
        image_path: Path to image file

    Returns:
        Validated image path

    Raises:
        SystemExit: If image file doesn't exist
    """
    if not Path(image_path).exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)

    return image_path


async def run_benchmark_command(args: argparse.Namespace) -> None:
    """Run the benchmark with provided arguments.

    Args:
        args: Parsed command line arguments
    """
    logger = setup_logger(args.verbose)

    # Validate environment
    base_url, api_key = validate_environment()

    # Validate image path
    image_path = validate_image_path(args.image)

    # Create configuration
    config = BenchmarkConfig(
        base_url=base_url,
        api_key=api_key,
        model_name=args.model,
        concurrent_requests=args.concurrent,
        total_requests=args.requests,
        duration_seconds=args.duration,
        timeout_seconds=args.timeout,
        retry_attempts=args.retries,
        retry_delay=args.retry_delay,
        warmup_requests=args.warmup,
    )

    # Validate configuration
    if not config.total_requests and not config.duration_seconds:
        print("ERROR: Either --requests or --duration must be specified")
        sys.exit(1)

    if config.total_requests and config.duration_seconds:
        print("ERROR: Cannot specify both --requests and --duration")
        sys.exit(1)

    # Run benchmark
    runner = BenchmarkRunner(config)

    try:
        logger.info("Starting VLLM OCR benchmark...")
        metrics = await runner.run_benchmark(image_path, skip_warmup=args.no_warmup)

        # Print results
        runner.print_results(metrics)

        # Save results if requested
        if args.output:
            runner.save_results(metrics, args.output)
            logger.info(f"Results saved to {args.output}")

        # Exit with error code if success rate is below threshold
        if args.min_success_rate and metrics.success_rate < args.min_success_rate:
            logger.error(
                f"Success rate {metrics.success_rate:.2f}% below threshold {args.min_success_rate}%"
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="VLLM OCR Benchmarking Tool - Test latency and throughput",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick latency test (10 requests, 1 concurrent)
  python benchmark_cli.py --image test.png --requests 10 --concurrent 1

  # Throughput test (50 concurrent for 60 seconds)
  python benchmark_cli.py --image test.png --duration 60 --concurrent 50

  # Stress test with custom timeout
  python benchmark_cli.py --image test.png --requests 100 --concurrent 20 --timeout 60

  # Save results to file
  python benchmark_cli.py --image test.png --requests 50 --concurrent 10 --output results.json

Environment Variables:
  RUNPOD_BASE_URL    - Your VLLM endpoint URL (required)
  RUNPOD_API_KEY     - Your API key (required)
        """,
    )

    # Required arguments
    parser.add_argument("--image", "-i", required=True, help="Path to the test image file")

    # Test configuration
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument("--requests", "-r", type=int, help="Total number of requests to send")
    test_group.add_argument("--duration", "-d", type=int, help="Duration to run test in seconds")

    # Concurrency settings
    parser.add_argument(
        "--concurrent",
        "-c",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)",
    )

    # Model settings
    parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        help="Model name to use (default: Qwen/Qwen2.5-VL-3B-Instruct-AWQ)",
    )

    # Timeout and retry settings
    parser.add_argument(
        "--timeout", "-t", type=int, default=30, help="Request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Number of retry attempts (default: 3)"
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay between retries in seconds (default: 1.0)",
    )

    # Warmup settings
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup requests (default: 5)"
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup requests")

    # Output settings
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument(
        "--min-success-rate",
        type=float,
        help="Minimum success rate percentage (exit with error if below)",
    )

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    return parser


load_dotenv(override=True)


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Run the benchmark
    asyncio.run(run_benchmark_command(args))


if __name__ == "__main__":
    main()
