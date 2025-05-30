#!/usr/bin/env python3
"""
Token Per Second Calculation Analysis and Improved Implementation

This script explains the difference between throughput metrics and generation speed metrics,
and provides an improved implementation for more accurate token/s measurements.
"""

import statistics
from dataclasses import dataclass
from typing import List


@dataclass
class ImprovedRequestResult:
    """Enhanced request result with detailed timing."""

    success: bool
    total_response_time: float  # Total time from request start to completion
    first_token_time: float  # Time to first token (TTFT)
    generation_time: float  # Time spent generating tokens (after first token)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TokenMetricsExplainer:
    """Explains different token/s calculation approaches."""

    def __init__(self):
        self.results: List[ImprovedRequestResult] = []
        self.total_benchmark_duration = 0.0

    def add_result(self, result: ImprovedRequestResult):
        """Add a request result."""
        self.results.append(result)

    def explain_current_implementation(self):
        """Explain what the current implementation actually measures."""
        print("🔍 CURRENT IMPLEMENTATION ANALYSIS")
        print("=" * 50)

        total_prompt_tokens = sum(r.prompt_tokens for r in self.results if r.success)
        total_completion_tokens = sum(r.completion_tokens for r in self.results if r.success)

        # This is what we currently calculate
        current_prompt_tps = total_prompt_tokens / self.total_benchmark_duration
        current_completion_tps = total_completion_tokens / self.total_benchmark_duration

        print(f"Current 'Prompt Tokens/s': {current_prompt_tps:.2f}")
        print(f"Current 'Completion Tokens/s': {current_completion_tps:.2f}")
        print()
        print("❌ PROBLEMS with current approach:")
        print("1. Prompt tokens aren't 'generated' - they're processed instantly")
        print("2. Includes wait time, network latency, and queue time")
        print("3. Doesn't reflect actual model generation speed")
        print("4. Misleading for performance optimization")
        print()

    def calculate_true_generation_speed(self):
        """Calculate actual token generation speed."""
        print("✅ IMPROVED METRICS")
        print("=" * 50)

        successful_results = [r for r in self.results if r.success]

        if not successful_results:
            print("No successful results to analyze")
            return

        # 1. THROUGHPUT METRICS (useful for capacity planning)
        total_prompt_tokens = sum(r.prompt_tokens for r in successful_results)
        total_completion_tokens = sum(r.completion_tokens for r in successful_results)

        throughput_prompt_tps = total_prompt_tokens / self.total_benchmark_duration
        throughput_completion_tps = total_completion_tokens / self.total_benchmark_duration

        print("📊 THROUGHPUT METRICS (Capacity Planning)")
        print(f"  Prompt Token Throughput:     {throughput_prompt_tps:.2f} tokens/s")
        print(f"  Completion Token Throughput: {throughput_completion_tps:.2f} tokens/s")
        print("  → How many tokens the system processes per second overall")
        print()

        # 2. GENERATION SPEED METRICS (actual model performance)
        generation_speeds = []
        for result in successful_results:
            if result.generation_time > 0 and result.completion_tokens > 0:
                speed = result.completion_tokens / result.generation_time
                generation_speeds.append(speed)

        if generation_speeds:
            avg_generation_speed = statistics.mean(generation_speeds)
            min_generation_speed = min(generation_speeds)
            max_generation_speed = max(generation_speeds)
            p50_generation_speed = statistics.median(generation_speeds)

            print("🚀 GENERATION SPEED METRICS (Model Performance)")
            print(f"  Average Generation Speed:    {avg_generation_speed:.2f} tokens/s")
            print(f"  Min Generation Speed:        {min_generation_speed:.2f} tokens/s")
            print(f"  Max Generation Speed:        {max_generation_speed:.2f} tokens/s")
            print(f"  P50 Generation Speed:        {p50_generation_speed:.2f} tokens/s")
            print("  → Actual speed the model generates tokens")
            print()

        # 3. LATENCY METRICS
        ttft_times = [r.first_token_time for r in successful_results if r.first_token_time > 0]
        if ttft_times:
            avg_ttft = statistics.mean(ttft_times)
            p95_ttft = sorted(ttft_times)[int(0.95 * len(ttft_times))]

            print("⏱️  LATENCY METRICS")
            print(f"  Average Time to First Token: {avg_ttft:.3f}s")
            print(f"  P95 Time to First Token:     {p95_ttft:.3f}s")
            print("  → How quickly the model starts responding")
            print()

    def explain_why_different(self):
        """Explain why the metrics are different."""
        print("🤔 WHY ARE THESE DIFFERENT?")
        print("=" * 50)
        print()
        print("THROUGHPUT vs GENERATION SPEED:")
        print()
        print("📈 Throughput = Total tokens / Total benchmark time")
        print("   - Includes: Network latency, queue time, processing overhead")
        print("   - Useful for: Capacity planning, cost estimation")
        print("   - Example: 'My system can handle 500 tokens/s of load'")
        print()
        print("🚀 Generation Speed = Completion tokens / Pure generation time")
        print("   - Measures: Only the token generation phase")
        print("   - Useful for: Model optimization, comparing models")
        print("   - Example: 'This model generates text at 50 tokens/s'")
        print()
        print("REAL EXAMPLE:")
        print("- Request takes 10 seconds total")
        print("- 2 seconds for processing prompt + first token")
        print("- 8 seconds generating 400 completion tokens")
        print()
        print("Throughput calculation: 400 tokens / 10 seconds = 40 tokens/s")
        print("Generation speed: 400 tokens / 8 seconds = 50 tokens/s")
        print()
        print("Both are correct but measure different things!")


def demonstrate_with_example():
    """Demonstrate with a realistic example."""
    print("🎯 REALISTIC EXAMPLE")
    print("=" * 50)

    explainer = TokenMetricsExplainer()
    explainer.total_benchmark_duration = 30.0  # 30 second benchmark

    # Simulate 5 requests with realistic timings
    example_results = [
        ImprovedRequestResult(
            success=True,
            total_response_time=6.0,
            first_token_time=1.5,  # 1.5s to first token
            generation_time=4.5,  # 4.5s generating tokens
            prompt_tokens=150,
            completion_tokens=225,  # 225 tokens in 4.5s = 50 tokens/s generation
            total_tokens=375,
        ),
        ImprovedRequestResult(
            success=True,
            total_response_time=8.0,
            first_token_time=2.0,
            generation_time=6.0,  # 300 tokens in 6s = 50 tokens/s generation
            prompt_tokens=150,
            completion_tokens=300,
            total_tokens=450,
        ),
        ImprovedRequestResult(
            success=True,
            total_response_time=5.0,
            first_token_time=1.0,
            generation_time=4.0,  # 200 tokens in 4s = 50 tokens/s generation
            prompt_tokens=150,
            completion_tokens=200,
            total_tokens=350,
        ),
        ImprovedRequestResult(
            success=True,
            total_response_time=7.0,
            first_token_time=1.8,
            generation_time=5.2,  # 260 tokens in 5.2s = 50 tokens/s generation
            prompt_tokens=150,
            completion_tokens=260,
            total_tokens=410,
        ),
        ImprovedRequestResult(
            success=True,
            total_response_time=4.0,
            first_token_time=0.8,
            generation_time=3.2,  # 160 tokens in 3.2s = 50 tokens/s generation
            prompt_tokens=150,
            completion_tokens=160,
            total_tokens=310,
        ),
    ]

    for result in example_results:
        explainer.add_result(result)

    explainer.explain_current_implementation()
    explainer.calculate_true_generation_speed()
    explainer.explain_why_different()


if __name__ == "__main__":
    demonstrate_with_example()
