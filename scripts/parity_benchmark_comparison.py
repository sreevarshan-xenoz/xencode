#!/usr/bin/env python3
"""Run both Rust and Python benchmarks, compare results for parity."""
import subprocess
import json
import sys

def run_rust_bench(bench_name: str) -> dict:
    """Run a cargo benchmark and extract the time."""
    try:
        result = subprocess.run(
            ["cargo", "bench", "--", "--json"],
            capture_output=True, text=True, timeout=120, cwd="rust"
        )
        # Parse JSON lines for the specific benchmark
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if data.get("type") == "bench" and bench_name in data.get("name", ""):
                    return {
                        "time_ns": data.get("typical_estimate", {}).get("nanoseconds", 0),
                        "iterations": data.get("iterations", 0),
                    }
            except json.JSONDecodeError:
                continue
        return {"error": f"Benchmark '{bench_name}' not found in output"}
    except subprocess.TimeoutExpired:
        return {"error": "Rust benchmark timed out"}
    except FileNotFoundError:
        return {"error": "Cargo not found"}


def run_python_bench(function_name: str, module: str) -> dict:
    """Run a Python benchmark using timeit."""
    import timeit
    try:
        # Import the module and run the function
        setup = f"from {module} import {function_name}"
        stmt = f"{function_name}()"
        number = 100
        total_time = timeit.timeit(stmt, setup=setup, number=number)
        return {
            "total_time_s": total_time,
            "avg_time_ms": (total_time / number) * 1000,
            "iterations": number,
        }
    except Exception as e:
        return {"error": str(e)}


def format_comparison(name: str, rust_result: dict, python_result: dict) -> str:
    """Format a comparison line."""
    if "error" in rust_result:
        return f"  {name}: Rust error - {rust_result['error']}"
    if "error" in python_result:
        return f"  {name}: Python error - {python_result['error']}"

    rust_time = rust_result.get("time_ns", 0) / 1_000_000  # ns to ms
    python_time = python_result.get("avg_time_ms", 0)

    if rust_time > 0 and python_time > 0:
        speedup = python_time / rust_time
        return f"  {name}: Rust={rust_time:.3f}ms  Python={python_time:.3f}ms  ({speedup:.1f}x faster)"
    return f"  {name}: Rust={rust_time:.3f}ms  Python={python_time:.3f}ms"


def main():
    """Run parity benchmarks and print comparison."""
    print("=" * 60)
    print("  Xencode Rust vs Python Parity Benchmarks")
    print("=" * 60)
    print()

    # Comparison 1: Workspace scan
    print("📁 Workspace Scan:")
    rust_scan = run_rust_bench("workspace_scan")
    python_scan = run_python_bench("benchmark_workspace_scan", "xencode_core")
    print(format_comparison("  workspace_scan", rust_scan, python_scan))
    print()

    # Comparison 2: Config load
    print("⚙️  Config Load:")
    rust_config = run_rust_bench("config_load")
    python_config = run_python_bench("benchmark_config_load", "xencode_config")
    print(format_comparison("  config_load", rust_config, python_config))
    print()

    # Comparison 3: Code analysis
    print("🔍 Code Analysis:")
    rust_analysis = run_rust_bench("code_analysis")
    python_analysis = run_python_bench("benchmark_code_analysis", "xencode_code_analysis_system")
    print(format_comparison("  code_analysis", rust_analysis, python_analysis))
    print()

    # Summary
    print("=" * 60)
    print("  Expected: Rust should be 3-10x faster than Python")
    print("  Benchmarks may vary based on system, Ollama availability, etc.")
    print("=" * 60)


if __name__ == "__main__":
    main()
