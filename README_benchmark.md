# VLLM OCR Benchmarking Tool

A comprehensive tool for testing latency and throughput of VLLM OCR endpoints with concurrent requests, detailed metrics, and performance analysis.

## Features

- **Concurrent Request Testing**: Run multiple OCR requests simultaneously to test throughput
- **Latency Analysis**: Measure response times with detailed percentile statistics (P50, P95, P99)
- **Throughput Measurement**: Calculate requests per second under different load conditions
- **Flexible Test Modes**: Fixed request count or duration-based testing
- **Comprehensive Metrics**: Success rates, error tracking, and detailed performance statistics
- **CLI Interface**: Easy-to-use command-line interface with extensive options
- **Results Export**: Save detailed results to JSON for further analysis
- **Retry Logic**: Built-in retry mechanism for handling transient failures
- **Warmup Support**: Optional warmup requests to stabilize service performance

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_benchmark.txt
```

2. Set up your environment variables:
```bash
export RUNPOD_BASE_URL="your_vllm_endpoint_url"
export RUNPOD_API_KEY="your_api_key"
```

Or create a `.env` file:
```
RUNPOD_BASE_URL=your_vllm_endpoint_url
RUNPOD_API_KEY=your_api_key
```

## Quick Start

### Command Line Interface

The easiest way to run benchmarks is using the CLI:

```bash
# Quick latency test (10 requests, 1 concurrent)
python benchmark_cli.py --image test.png --requests 10 --concurrent 1

# Throughput test (50 concurrent for 60 seconds)
python benchmark_cli.py --image test.png --duration 60 --concurrent 50

# Stress test with custom timeout
python benchmark_cli.py --image test.png --requests 100 --concurrent 20 --timeout 60
```

### Programmatic Usage

```python
import asyncio
from vllm_benchmark import BenchmarkConfig, BenchmarkRunner

async def run_benchmark():
    config = BenchmarkConfig(
        base_url="your_endpoint_url",
        api_key="your_api_key",
        concurrent_requests=10,
        total_requests=50,
        timeout_seconds=30
    )
    
    runner = BenchmarkRunner(config)
    metrics = await runner.run_benchmark("path/to/image.png")
    runner.print_results(metrics)

asyncio.run(run_benchmark())
```

## CLI Options

### Required Arguments
- `--image, -i`: Path to the test image file

### Test Configuration (choose one)
- `--requests, -r`: Total number of requests to send
- `--duration, -d`: Duration to run test in seconds

### Performance Settings
- `--concurrent, -c`: Number of concurrent requests (default: 10)
- `--timeout, -t`: Request timeout in seconds (default: 30)
- `--retries`: Number of retry attempts (default: 3)
- `--retry-delay`: Delay between retries in seconds (default: 1.0)

### Model Settings
- `--model, -m`: Model name to use (default: Qwen/Qwen2.5-VL-3B-Instruct-AWQ)

### Warmup Settings
- `--warmup`: Number of warmup requests (default: 5)
- `--no-warmup`: Skip warmup requests

### Output Settings
- `--output, -o`: Save results to JSON file
- `--min-success-rate`: Minimum success rate percentage (exit with error if below)
- `--verbose, -v`: Enable verbose logging

## Example Usage Scenarios

### 1. Latency Testing
Test individual request performance:
```bash
python benchmark_cli.py --image test.png --requests 10 --concurrent 1 --output latency_results.json
```

### 2. Throughput Testing
Test maximum throughput with multiple concurrent requests:
```bash
python benchmark_cli.py --image test.png --requests 100 --concurrent 20 --output throughput_results.json
```

### 3. Stress Testing
Test system limits with high concurrency:
```bash
python benchmark_cli.py --image test.png --requests 200 --concurrent 50 --timeout 60 --output stress_results.json
```

### 4. Duration-Based Testing
Test sustained performance over time:
```bash
python benchmark_cli.py --image test.png --duration 300 --concurrent 15 --output duration_results.json
```

### 5. CI/CD Integration
Test with minimum success rate requirement:
```bash
python benchmark_cli.py --image test.png --requests 50 --concurrent 10 --min-success-rate 95
```

## Example Scripts

### Run All Test Scenarios
```bash
python example_benchmark.py
```

This will run:
- Latency test (1 concurrent, 10 requests)
- Throughput test (10 concurrent, 50 requests)
- Stress test (25 concurrent, 100 requests)
- Duration test (15 concurrent, 60 seconds)
- Comparative test (different concurrency levels)

## Metrics Explained

### Performance Metrics
- **Requests/Second (RPS)**: Number of successful requests processed per second
- **Success Rate**: Percentage of requests that completed successfully
- **Total Duration**: Total time taken for the benchmark

### Latency Metrics
- **Average Response Time**: Mean response time across all successful requests
- **Minimum/Maximum**: Fastest and slowest response times
- **50th Percentile (P50)**: Median response time
- **95th Percentile (P95)**: 95% of requests completed within this time
- **99th Percentile (P99)**: 99% of requests completed within this time

### Token Metrics
- **Total Tokens/Second**: Overall token processing rate
- **Prompt Tokens/Second**: Input token processing rate
- **Completion Tokens/Second**: Output token generation rate
- **Average Tokens per Request**: Mean token usage per request
- **Token Efficiency**: Prompt to completion token ratio

### Error Tracking
- **Failed Requests**: Number of requests that failed
- **Error Types**: Breakdown of different error types encountered

## Output Format

Results are saved in JSON format with the following structure:

```json
{
  "config": {
    "base_url": "endpoint_url",
    "model_name": "model_name",
    "concurrent_requests": 10,
    "total_requests": 50,
    "timeout_seconds": 30
  },
  "metrics": {
    "total_requests": 50,
    "successful_requests": 48,
    "failed_requests": 2,
    "success_rate": 96.0,
    "total_duration": 25.5,
    "requests_per_second": 1.88,
    "avg_response_time": 5.32,
    "min_response_time": 3.21,
    "max_response_time": 8.45,
    "p50_response_time": 5.12,
    "p95_response_time": 7.89,
    "p99_response_time": 8.23,
    "error_counts": {
      "TimeoutError": 2
    },
    "response_times": [3.21, 4.56, ...],
    "total_tokens": 12500,
    "total_prompt_tokens": 7200,
    "total_completion_tokens": 5300,
    "total_tokens_per_second": 490.2,
    "prompt_tokens_per_second": 282.4,
    "completion_tokens_per_second": 207.8,
    "avg_total_tokens_per_request": 260.4,
    "avg_prompt_tokens_per_request": 150.0,
    "avg_completion_tokens_per_request": 110.4
  },
  "timestamp": 1703123456.789
}
```

## Best Practices

### 1. Warmup Requests
Always include warmup requests (default: 5) to allow the service to stabilize before measurement.

### 2. Appropriate Timeouts
Set timeouts based on your expected response times. For OCR tasks, 30-60 seconds is typically appropriate.

### 3. Gradual Load Increase
Start with low concurrency and gradually increase to find the optimal performance point.

### 4. Multiple Test Runs
Run multiple tests and average the results for more reliable measurements.

### 5. Monitor Resource Usage
Monitor your server's CPU, memory, and GPU usage during benchmarks to identify bottlenecks.

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify `RUNPOD_BASE_URL` and `RUNPOD_API_KEY` are set correctly
   - Check network connectivity to the endpoint

2. **Timeout Errors**
   - Increase `--timeout` value
   - Reduce `--concurrent` to lower server load

3. **High Error Rates**
   - Check server capacity and scaling settings
   - Reduce concurrency level
   - Increase retry attempts and delay

4. **Import Errors**
   - Install dependencies: `pip install -r requirements_benchmark.txt`
   - Ensure Python 3.8+ is being used

### Performance Optimization

1. **For Latency Testing**: Use `--concurrent 1` to measure single-request performance
2. **For Throughput Testing**: Start with low concurrency and increase gradually
3. **For Stress Testing**: Monitor server resources and increase concurrency until performance degrades

## Contributing

Feel free to submit issues and enhancement requests. When contributing:

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all linter checks pass

## License

This tool is provided as-is for benchmarking purposes. Ensure you have appropriate permissions to test your VLLM endpoints. 