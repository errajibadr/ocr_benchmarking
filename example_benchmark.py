#!/usr/bin/env python3
"""
Example usage of VLLM OCR Benchmarking Tool

This script demonstrates different benchmarking scenarios:
1. Latency testing (single request at a time)
2. Throughput testing (multiple concurrent requests)
3. Stress testing (high concurrency)
4. Duration-based testing
"""

import asyncio
import os

from dotenv import load_dotenv

from vllm_benchmark import BenchmarkConfig, BenchmarkRunner


async def latency_test():
    """Test latency with single concurrent request."""
    print("\n" + "=" * 50)
    print("LATENCY TEST - Single Request Performance")
    print("=" * 50)

    config = BenchmarkConfig(
        base_url=os.getenv("RUNPOD_BASE_URL"),
        api_key=os.getenv("RUNPOD_API_KEY"),
        concurrent_requests=1,  # Single request at a time
        total_requests=10,  # 10 requests total
        timeout_seconds=30,
    )

    runner = BenchmarkRunner(config)
    metrics = await runner.run_benchmark("dataset/sample/images/82200067_0069.png")
    runner.print_results(metrics)
    runner.save_results(metrics, "latency_test_results.json")


async def throughput_test():
    """Test throughput with multiple concurrent requests."""
    print("\n" + "=" * 50)
    print("THROUGHPUT TEST - Multiple Concurrent Requests")
    print("=" * 50)

    config = BenchmarkConfig(
        base_url=os.getenv("RUNPOD_BASE_URL"),
        api_key=os.getenv("RUNPOD_API_KEY"),
        concurrent_requests=10,  # 10 concurrent requests
        total_requests=50,  # 50 requests total
        timeout_seconds=30,
    )

    runner = BenchmarkRunner(config)
    metrics = await runner.run_benchmark("dataset/sample/images/82200067_0069.png")
    runner.print_results(metrics)
    runner.save_results(metrics, "throughput_test_results.json")


async def stress_test():
    """Test with high concurrency to find limits."""
    print("\n" + "=" * 50)
    print("STRESS TEST - High Concurrency")
    print("=" * 50)

    config = BenchmarkConfig(
        base_url=os.getenv("RUNPOD_BASE_URL"),
        api_key=os.getenv("RUNPOD_API_KEY"),
        concurrent_requests=25,  # 25 concurrent requests
        total_requests=100,  # 100 requests total
        timeout_seconds=60,  # Longer timeout for stress test
        retry_attempts=5,  # More retries for stress conditions
    )

    runner = BenchmarkRunner(config)
    metrics = await runner.run_benchmark("dataset/sample/images/82200067_0069.png")
    runner.print_results(metrics)
    runner.save_results(metrics, "stress_test_results.json")


async def duration_test():
    """Test for a specific duration to measure sustained performance."""
    print("\n" + "=" * 50)
    print("DURATION TEST - Sustained Performance")
    print("=" * 50)

    config = BenchmarkConfig(
        base_url=os.getenv("RUNPOD_BASE_URL"),
        api_key=os.getenv("RUNPOD_API_KEY"),
        concurrent_requests=15,  # 15 concurrent requests
        duration_seconds=60,  # Run for 60 seconds
        timeout_seconds=30,
    )

    runner = BenchmarkRunner(config)
    metrics = await runner.run_benchmark("dataset/sample/images/82200067_0069.png")
    runner.print_results(metrics)
    runner.save_results(metrics, "duration_test_results.json")


async def comparative_test():
    """Run multiple tests with different concurrency levels for comparison."""
    print("\n" + "=" * 50)
    print("COMPARATIVE TEST - Different Concurrency Levels")
    print("=" * 50)

    concurrency_levels = [1, 5, 10, 20]
    results = {}

    for concurrency in concurrency_levels:
        print(f"\nTesting with {concurrency} concurrent requests...")

        config = BenchmarkConfig(
            base_url=os.getenv("RUNPOD_BASE_URL"),
            api_key=os.getenv("RUNPOD_API_KEY"),
            concurrent_requests=concurrency,
            total_requests=30,  # Fixed number of requests for fair comparison
            timeout_seconds=30,
            warmup_requests=3,  # Fewer warmup requests for faster testing
        )

        runner = BenchmarkRunner(config)
        metrics = await runner.run_benchmark("dataset/sample/images/82200067_0069.png")

        results[concurrency] = {
            "requests_per_second": metrics.requests_per_second,
            "avg_response_time": metrics.avg_response_time,
            "success_rate": metrics.success_rate,
        }

        print(f"  RPS: {metrics.requests_per_second:.2f}")
        print(f"  Avg Response Time: {metrics.avg_response_time:.3f}s")
        print(f"  Success Rate: {metrics.success_rate:.1f}%")

    # Print comparison summary
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"{'Concurrency':<12} {'RPS':<8} {'Avg Time':<10} {'Success %':<10}")
    print("-" * 50)

    for concurrency, data in results.items():
        print(
            f"{concurrency:<12} {data['requests_per_second']:<8.2f} "
            f"{data['avg_response_time']:<10.3f} {data['success_rate']:<10.1f}"
        )


async def main():
    """Run all benchmark examples."""
    # Load environment variables
    load_dotenv(override=True)

    # Validate environment
    if not os.getenv("RUNPOD_BASE_URL") or not os.getenv("RUNPOD_API_KEY"):
        print("ERROR: Please set RUNPOD_BASE_URL and RUNPOD_API_KEY environment variables")
        return

    print("VLLM OCR Benchmarking Examples")
    print("This will run several benchmark scenarios...")

    try:
        # Run different test scenarios
        await latency_test()
        await throughput_test()
        await stress_test()
        await duration_test()
        await comparative_test()

        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED")
        print("=" * 50)
        print("Check the generated JSON files for detailed results:")
        print("- latency_test_results.json")
        print("- throughput_test_results.json")
        print("- stress_test_results.json")
        print("- duration_test_results.json")

    except Exception as e:
        print(f"Error running benchmarks: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
