#!/usr/bin/env python3
"""
Test Execution Loop

Automated test execution system with parallel execution, coverage reporting,
and iterative test fix capabilities.

Features:
- Run generated tests automatically
- Capture and analyze test failures
- Iterate on test fixes (bounded retries)
- Test coverage reporting
- Parallel test execution
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ExecutionMode(Enum):
    """Test execution mode"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"


@dataclass
class TestResult:
    """Result of a single test execution"""
    test_id: str
    test_name: str
    test_file: str
    status: TestStatus
    duration_ms: float
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    output: str = ""
    assertions_passed: int = 0
    assertions_failed: int = 0
    coverage_percent: float = 0.0
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestExecutionResult:
    """Result of a test execution run"""
    execution_id: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_ms: float
    coverage_report: Optional[Dict[str, Any]] = None
    test_results: List[TestResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    parallel_workers: int = 1
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    retry_summary: Dict[str, int] = field(default_factory=dict)


@dataclass
class CoverageReport:
    """Test coverage report"""
    total_lines: int
    covered_lines: int
    missing_lines: List[int]
    excluded_lines: List[int]
    percent_covered: float
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    functions: Dict[str, bool] = field(default_factory=dict)
    classes: Dict[str, bool] = field(default_factory=dict)
    branches_covered: int = 0
    branches_total: int = 0


@dataclass
class ExecutionConfig:
    """Configuration for test execution"""
    mode: ExecutionMode = ExecutionMode.PARALLEL
    max_workers: int = 4
    timeout_seconds: float = 300.0
    retry_failed: bool = True
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    collect_coverage: bool = True
    coverage_source: Optional[List[str]] = None
    coverage_omit: Optional[List[str]] = None
    verbose: bool = True
    stop_on_failure: bool = False
    parallel_threshold: int = 5  # Use parallel if more than this many tests
    environment: Dict[str, str] = field(default_factory=dict)
    pytest_args: List[str] = field(default_factory=list)
    unittest_args: List[str] = field(default_factory=list)


class CoverageCollector:
    """Collects and reports test coverage"""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.cov: Optional[coverage.Coverage] = None
        self.is_running = False

    def start(self) -> None:
        """Start coverage collection"""
        if not COVERAGE_AVAILABLE:
            return

        self.cov = coverage.Coverage(
            source=self.config.coverage_source,
            omit=self.config.coverage_omit,
            branch=True,
            concurrency=["thread", "multiprocessing"]
        )
        self.cov.start()
        self.is_running = True

    def stop(self) -> None:
        """Stop coverage collection"""
        if self.cov and self.is_running:
            self.cov.stop()
            self.is_running = False

    def save(self, data_file: Optional[str] = None) -> None:
        """Save coverage data"""
        if self.cov:
            self.cov.save(data_file)

    def load(self, data_file: Optional[str] = None) -> None:
        """Load coverage data"""
        if self.cov:
            self.cov.load(data_file)

    def get_report(self) -> CoverageReport:
        """Generate coverage report"""
        if not self.cov:
            return CoverageReport(
                total_lines=0,
                covered_lines=0,
                missing_lines=[],
                excluded_lines=[],
                percent_covered=0.0
            )

        try:
            # Get coverage data - handle different coverage API versions
            try:
                analysis = self.cov.analysis([])  # Pass empty list for all files
            except TypeError:
                # Older API might not need argument
                analysis = self.cov.analysis()
        except Exception:
            return CoverageReport(
                total_lines=0,
                covered_lines=0,
                missing_lines=[],
                excluded_lines=[],
                percent_covered=0.0
            )
            
        report = CoverageReport(
            total_lines=0,
            covered_lines=0,
            missing_lines=[],
            excluded_lines=[],
            percent_covered=0.0
        )

        for file_path in analysis:
            try:
                lines = self.cov.analysis2(file_path)
                covered = self.cov.covered_lines(file_path)

                file_total = len(lines)
                file_covered = len(covered) if covered else 0
                file_missing = list(set(lines) - set(covered)) if covered else lines

                report.total_lines += file_total
                report.covered_lines += file_covered
                report.missing_lines.extend(file_missing)

                report.files[file_path] = {
                    "total": file_total,
                    "covered": file_covered,
                    "missing": file_missing,
                    "percent": (file_covered / file_total * 100) if file_total > 0 else 0
                }
            except Exception:
                continue

        # Calculate overall percentage
        if report.total_lines > 0:
            report.percent_covered = (report.covered_lines / report.total_lines) * 100

        return report

    def get_html_report(self, directory: str = "htmlcov") -> str:
        """Generate HTML coverage report"""
        if not self.cov:
            return ""

        self.cov.html_report(directory=directory)
        return os.path.join(directory, "index.html")

    def get_xml_report(self, outfile: str = "coverage.xml") -> str:
        """Generate XML coverage report"""
        if not self.cov:
            return ""

        self.cov.xml_report(outfile=outfile)
        return outfile


class TestRunner:
    """Main test execution engine"""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.coverage_collector = CoverageCollector(self.config)
        self.results: List[TestResult] = []
        self.current_execution: Optional[TestExecutionResult] = None
        self._stop_requested = False
        self._executor: Optional[ThreadPoolExecutor] = None

    def execute(
        self,
        test_files: List[str],
        config: Optional[ExecutionConfig] = None
    ) -> TestExecutionResult:
        """Execute tests"""
        if config:
            self.config = config

        execution_id = str(uuid.uuid4())
        start_time = datetime.now()

        # Initialize result
        self.current_execution = TestExecutionResult(
            execution_id=execution_id,
            total_tests=len(test_files),
            passed=0,
            failed=0,
            skipped=0,
            errors=0,
            duration_ms=0,
            parallel_workers=self.config.max_workers,
            execution_mode=self.config.mode
        )

        # Determine execution mode
        if len(test_files) <= self.config.parallel_threshold:
            self.config.mode = ExecutionMode.SEQUENTIAL
        else:
            self.config.mode = ExecutionMode.PARALLEL

        # Start coverage collection if enabled
        if self.config.collect_coverage and COVERAGE_AVAILABLE:
            self.coverage_collector.start()

        try:
            # Execute based on mode
            if self.config.mode == ExecutionMode.SEQUENTIAL:
                self._execute_sequential(test_files)
            elif self.config.mode == ExecutionMode.PARALLEL:
                self._execute_parallel(test_files)
            else:
                self._execute_sequential(test_files)  # Fallback to sequential

        finally:
            # Stop coverage collection
            if self.config.collect_coverage and COVERAGE_AVAILABLE:
                self.coverage_collector.stop()
                self.current_execution.coverage_report = self._get_coverage_data()

        # Calculate final statistics
        end_time = datetime.now()
        self.current_execution.end_time = end_time
        self.current_execution.duration_ms = (end_time - start_time).total_seconds() * 1000
        self.current_execution.test_results = self.results

        # Update counts
        for result in self.results:
            if result.status == TestStatus.PASSED:
                self.current_execution.passed += 1
            elif result.status == TestStatus.FAILED:
                self.current_execution.failed += 1
            elif result.status == TestStatus.SKIPPED:
                self.current_execution.skipped += 1
            elif result.status == TestStatus.ERROR:
                self.current_execution.errors += 1

        return self.current_execution

    def _execute_sequential(self, test_files: List[str]) -> None:
        """Execute tests sequentially"""
        for test_file in test_files:
            if self._stop_requested:
                break

            if self.config.stop_on_failure and self.current_execution.failed > 0:
                break

            result = self._execute_single_test(test_file)
            self.results.append(result)

    def _execute_parallel(self, test_files: List[str]) -> None:
        """Execute tests in parallel using thread pool"""
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        try:
            futures = {
                self._executor.submit(self._execute_single_test, test_file): test_file
                for test_file in test_files
            }

            for future in as_completed(futures, timeout=self.config.timeout_seconds):
                if self._stop_requested:
                    break

                test_file = futures[future]

                try:
                    result = future.result()
                    self.results.append(result)

                    if self.config.stop_on_failure and result.status == TestStatus.FAILED:
                        self._stop_requested = True
                except Exception as e:
                    # Create error result
                    result = TestResult(
                        test_id=str(uuid.uuid4()),
                        test_name=Path(test_file).stem,
                        test_file=test_file,
                        status=TestStatus.ERROR,
                        duration_ms=0,
                        error_message=str(e)
                    )
                    self.results.append(result)

        except TimeoutError:
            # Handle timeout
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
            raise
        finally:
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None

    def _execute_single_test(self, test_file: str) -> TestResult:
        """Execute a single test file"""
        test_id = str(uuid.uuid4())
        test_name = Path(test_file).stem
        start_time = time.time()

        result = TestResult(
            test_id=test_id,
            test_name=test_name,
            test_file=test_file,
            status=TestStatus.PENDING,
            duration_ms=0
        )

        try:
            # Determine test framework and run appropriate command
            if self._is_pytest_file(test_file):
                result = self._run_pytest(test_file, result)
            elif self._is_unittest_file(test_file):
                result = self._run_unittest(test_file, result)
            else:
                # Default to pytest
                result = self._run_pytest(test_file, result)

        except subprocess.TimeoutExpired:
            result.status = TestStatus.TIMEOUT
            result.error_message = f"Test timed out after {self.config.timeout_seconds}s"
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)

        # Calculate duration
        end_time = time.time()
        result.duration_ms = (end_time - start_time) * 1000

        # Handle retries
        if (self.config.retry_failed and
            result.status == TestStatus.FAILED and
            result.retry_count < self.config.max_retries):
            result = self._retry_test(test_file, result)

        return result

    def _run_pytest(self, test_file: str, result: TestResult) -> TestResult:
        """Run pytest on a test file"""
        if not PYTEST_AVAILABLE:
            # Fallback to subprocess
            return self._run_pytest_subprocess(test_file, result)

        # Run pytest programmatically
        args = [
            test_file,
            "-v",
            "--tb=short",
            "--json-report",
            "--json-report-file=none"
        ]

        # Add custom args
        args.extend(self.config.pytest_args)

        # Capture output
        import io
        from contextlib import redirect_stdout, redirect_stderr

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        exit_code = 0

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exit_code = pytest.main(args)
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            return result

        result.output = stdout_capture.getvalue()
        result.error_message = stderr_capture.getvalue() if exit_code != 0 else None

        if exit_code == 0:
            result.status = TestStatus.PASSED
        else:
            result.status = TestStatus.FAILED

        return result

    def _run_pytest_subprocess(self, test_file: str, result: TestResult) -> TestResult:
        """Run pytest using subprocess"""
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--tb=short",
            "--color=no"
        ]

        cmd.extend(self.config.pytest_args)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                env={**os.environ, **self.config.environment}
            )

            result.output = proc.stdout
            result.error_message = proc.stderr

            if proc.returncode == 0:
                result.status = TestStatus.PASSED
            else:
                result.status = TestStatus.FAILED

        except subprocess.TimeoutExpired:
            result.status = TestStatus.TIMEOUT
            result.error_message = f"Test timed out after {self.config.timeout_seconds}s"

        return result

    def _run_unittest(self, test_file: str, result: TestResult) -> TestResult:
        """Run unittest on a test file"""
        cmd = [
            sys.executable, "-m", "unittest",
            "-v",
            test_file
        ]

        cmd.extend(self.config.unittest_args)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                env={**os.environ, **self.config.environment}
            )

            result.output = proc.stdout
            result.error_message = proc.stderr

            if proc.returncode == 0:
                result.status = TestStatus.PASSED
            else:
                result.status = TestStatus.FAILED

        except subprocess.TimeoutExpired:
            result.status = TestStatus.TIMEOUT
            result.error_message = f"Test timed out after {self.config.timeout_seconds}s"

        return result

    def _retry_test(self, test_file: str, result: TestResult) -> TestResult:
        """Retry a failed test"""
        retry_count = 0

        while (retry_count < self.config.max_retries and
               result.status == TestStatus.FAILED):
            retry_count += 1
            result.retry_count = retry_count

            # Wait before retry
            time.sleep(self.config.retry_delay_seconds)

            # Re-run test
            if self._is_pytest_file(test_file):
                result = self._run_pytest(test_file, result)
            else:
                result = self._run_unittest(test_file, result)

        return result

    def _is_pytest_file(self, test_file: str) -> bool:
        """Check if file contains pytest tests"""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                return 'import pytest' in content or 'def test_' in content
        except Exception:
            return False

    def _is_unittest_file(self, test_file: str) -> bool:
        """Check if file contains unittest tests"""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                return 'import unittest' in content or 'unittest.TestCase' in content
        except Exception:
            return False

    def _get_coverage_data(self) -> Optional[Dict[str, Any]]:
        """Get coverage data"""
        if not COVERAGE_AVAILABLE or not self.coverage_collector.cov:
            return None

        report = self.coverage_collector.get_report()

        return {
            "total_lines": report.total_lines,
            "covered_lines": report.covered_lines,
            "missing_lines": report.missing_lines[:100],  # Limit for response size
            "percent_covered": report.percent_covered,
            "files": {
                path: {
                    "percent": data["percent"],
                    "covered": data["covered"],
                    "total": data["total"]
                }
                for path, data in list(report.files.items())[:20]  # Limit files
            }
        }

    def stop(self) -> None:
        """Stop test execution"""
        self._stop_requested = True

    def get_results(self) -> List[TestResult]:
        """Get all test results"""
        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        if not self.current_execution:
            return {}

        return {
            "execution_id": self.current_execution.execution_id,
            "total_tests": self.current_execution.total_tests,
            "passed": self.current_execution.passed,
            "failed": self.current_execution.failed,
            "skipped": self.current_execution.skipped,
            "errors": self.current_execution.errors,
            "duration_ms": self.current_execution.duration_ms,
            "success_rate": (
                self.current_execution.passed / self.current_execution.total_tests * 100
                if self.current_execution.total_tests > 0 else 0
            ),
            "coverage_percent": (
                self.current_execution.coverage_report.get("percent_covered", 0)
                if self.current_execution.coverage_report else 0
            )
        }


class TestExecutionLoop:
    """
    Iterative test execution loop with automatic fix attempts.

    Runs tests, analyzes failures, applies fixes, and re-runs
    until all tests pass or max iterations reached.
    """

    def __init__(
        self,
        runner: Optional[TestRunner] = None,
        max_iterations: int = 5,
        fix_strategy: Optional[Callable] = None
    ):
        self.runner = runner or TestRunner()
        self.max_iterations = max_iterations
        self.fix_strategy = fix_strategy or self._default_fix_strategy
        self.iteration_history: List[Dict[str, Any]] = []
        self._current_iteration = 0

    def run(
        self,
        test_files: List[str],
        source_files: Optional[List[str]] = None,
        config: Optional[ExecutionConfig] = None
    ) -> Dict[str, Any]:
        """
        Run the test execution loop.

        Args:
            test_files: List of test files to execute
            source_files: Optional list of source files for coverage
            config: Execution configuration

        Returns:
            Dictionary with execution results and history
        """
        if config:
            self.runner.config = config

        if source_files:
            self.runner.config.coverage_source = source_files

        results = {
            "success": False,
            "iterations": [],
            "final_result": None,
            "total_duration_ms": 0,
            "stopping_reason": None
        }

        start_time = time.time()

        for iteration in range(self.max_iterations):
            self._current_iteration = iteration + 1

            # Run tests
            execution_result = self.runner.execute(test_files)

            # Record iteration
            iteration_data = {
                "iteration": iteration + 1,
                "timestamp": datetime.now().isoformat(),
                "passed": execution_result.passed,
                "failed": execution_result.failed,
                "errors": execution_result.errors,
                "skipped": execution_result.skipped,
                "duration_ms": execution_result.duration_ms,
                "coverage": execution_result.coverage_report,
                "failures": self._extract_failures(execution_result)
            }
            results["iterations"].append(iteration_data)
            self.iteration_history.append(iteration_data)

            # Check if all tests passed
            if execution_result.failed == 0 and execution_result.errors == 0:
                results["success"] = True
                results["stopping_reason"] = "all_tests_passed"
                break

            # Check if we should continue
            if iteration >= self.max_iterations - 1:
                results["stopping_reason"] = "max_iterations_reached"
                break

            # Apply fixes
            fix_result = self.fix_strategy(execution_result, source_files)

            if not fix_result.get("success"):
                results["stopping_reason"] = "fix_strategy_failed"
                break

            # Update test files if they were modified
            if fix_result.get("modified_files"):
                test_files = fix_result["modified_files"]

        # Calculate total duration
        end_time = time.time()
        results["total_duration_ms"] = (end_time - start_time) * 1000
        results["final_result"] = self.runner.get_summary()

        return results

    def _extract_failures(self, execution_result: TestExecutionResult) -> List[Dict[str, Any]]:
        """Extract failure information from execution result"""
        failures = []

        for result in execution_result.test_results:
            if result.status in (TestStatus.FAILED, TestStatus.ERROR):
                failures.append({
                    "test_name": result.test_name,
                    "test_file": result.test_file,
                    "status": result.status.value,
                    "error_message": result.error_message,
                    "retry_count": result.retry_count
                })

        return failures

    def _default_fix_strategy(
        self,
        execution_result: TestExecutionResult,
        source_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Default fix strategy - analyzes failures and suggests fixes.

        In a full implementation, this would integrate with an AI agent
        to automatically fix test issues.
        """
        # For now, just report that no automatic fixes are applied
        return {
            "success": True,
            "modified_files": [],
            "fixes_applied": 0,
            "message": "No automatic fixes applied in default strategy"
        }

    def get_iteration_history(self) -> List[Dict[str, Any]]:
        """Get iteration history"""
        return self.iteration_history

    def reset(self) -> None:
        """Reset the execution loop"""
        self.iteration_history = []
        self._current_iteration = 0


class ParallelTestExecutor:
    """
    Advanced parallel test executor with work stealing and load balancing.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor: Optional[ProcessPoolExecutor] = None
        self.results_queue: Queue = Queue()
        self.tasks_queue: Queue = Queue()

    def execute(
        self,
        test_files: List[str],
        timeout: float = 300.0
    ) -> List[TestResult]:
        """Execute tests in parallel with load balancing"""
        results = []

        # Add tasks to queue
        for test_file in test_files:
            self.tasks_queue.put(test_file)

        # Create worker threads
        workers = []
        for _ in range(min(self.max_workers, len(test_files))):
            worker = threading.Thread(target=self._worker)
            worker.start()
            workers.append(worker)

        # Wait for all tasks to complete
        self.tasks_queue.join()

        # Stop workers
        for _ in workers:
            self.tasks_queue.put(None)

        for worker in workers:
            worker.join(timeout=10)

        # Collect results
        while not self.results_queue.empty():
            results.append(self.results_queue.get())

        return results

    def _worker(self) -> None:
        """Worker thread that processes tasks from queue"""
        while True:
            try:
                test_file = self.tasks_queue.get(timeout=1)

                if test_file is None:
                    self.tasks_queue.task_done()
                    break

                # Execute test
                result = self._execute_test(test_file)
                self.results_queue.put(result)

                self.tasks_queue.task_done()

            except Empty:
                continue
            except Exception:
                pass

    def _execute_test(self, test_file: str) -> TestResult:
        """Execute a single test"""
        test_id = str(uuid.uuid4())
        test_name = Path(test_file).stem
        start_time = time.time()

        result = TestResult(
            test_id=test_id,
            test_name=test_name,
            test_file=test_file,
            status=TestStatus.PENDING,
            duration_ms=0
        )

        try:
            cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            result.output = proc.stdout
            result.error_message = proc.stderr
            result.status = TestStatus.PASSED if proc.returncode == 0 else TestStatus.FAILED

        except subprocess.TimeoutExpired:
            result.status = TestStatus.TIMEOUT
            result.error_message = "Test timed out"
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)

        result.duration_ms = (time.time() - start_time) * 1000

        return result


def create_test_runner(config: Optional[ExecutionConfig] = None) -> TestRunner:
    """Factory function to create a test runner"""
    return TestRunner(config)


def create_execution_loop(
    runner: Optional[TestRunner] = None,
    max_iterations: int = 5
) -> TestExecutionLoop:
    """Factory function to create a test execution loop"""
    return TestExecutionLoop(runner, max_iterations)


def run_tests(
    test_files: List[str],
    parallel: bool = True,
    coverage: bool = True,
    timeout: float = 300.0
) -> TestExecutionResult:
    """Convenience function to run tests"""
    config = ExecutionConfig(
        mode=ExecutionMode.PARALLEL if parallel else ExecutionMode.SEQUENTIAL,
        max_workers=4,
        timeout_seconds=timeout,
        collect_coverage=coverage
    )

    runner = TestRunner(config)
    return runner.execute(test_files)


if __name__ == "__main__":
    # Example usage
    print("Test Runner - Example Usage")
    print("=" * 50)

    # Create runner
    config = ExecutionConfig(
        mode=ExecutionMode.PARALLEL,
        max_workers=2,
        collect_coverage=False,
        verbose=True
    )

    runner = TestRunner(config)

    # Example test files (these would be actual test files)
    test_files = [
        "tests/example/test_sample.py"
    ]

    print(f"Would execute {len(test_files)} test file(s)")
    print(f"Mode: {config.mode.value}")
    print(f"Max workers: {config.max_workers}")
    print(f"Coverage: {config.collect_coverage}")
