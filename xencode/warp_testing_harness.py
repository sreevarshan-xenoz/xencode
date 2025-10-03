#!/usr/bin/env python3
"""
Warp Terminal Testing Harness

Comprehensive testing framework for validating command execution,
parsing correctness, and performance under load.
"""

import subprocess
import time
import random
import threading
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text


@dataclass
class TestResult:
    """Result of a test command execution"""
    command: str
    success: bool
    duration_ms: int
    error: Optional[str] = None
    output_size: int = 0
    exit_code: int = 0
    parsed_correctly: bool = True


class CommandTestingHarness:
    """Testing harness for simulating command execution and validation"""
    
    def __init__(self):
        self.console = Console()
        
        # Test commands categorized by type
        self.test_commands = {
            "basic": [
                "echo 'Hello, World!'",
                "date",
                "whoami", 
                "pwd",
                "uptime"
            ],
            "file_operations": [
                "ls -la",
                "ls -lh",
                "find . -name '*.py' | head -10",
                "cat /etc/os-release",
                "df -h",
                "free -m"
            ],
            "git_operations": [
                "git status",
                "git log --oneline -5",
                "git branch",
                "git remote -v"
            ],
            "system_info": [
                "ps aux | head -10",
                "top -b -n1 | head -10",
                "netstat -tuln | head -10",
                "lscpu | head -10"
            ],
            "development": [
                "python --version",
                "pip list | head -10",
                "npm --version",
                "node --version",
                "docker --version"
            ],
            "network": [
                "ping -c 3 8.8.8.8",
                "curl -s https://httpbin.org/json",
                "wget --spider -q https://google.com"
            ]
        }
        
        # Flatten all commands for easy access
        self.all_commands = []
        for category, commands in self.test_commands.items():
            self.all_commands.extend(commands)
    
    def run_stress_test(self, terminal, num_commands: int = 50, 
                       max_workers: int = 5) -> List[TestResult]:
        """Run a stress test with multiple commands"""
        results = []
        
        self.console.print(f"[bold blue]Starting stress test with {num_commands} commands...[/bold blue]")
        
        def execute_command(cmd):
            start_time = time.time()
            try:
                # Execute the command using the terminal
                block = terminal.run_command_streaming(cmd)
                
                # Wait for completion (with timeout)
                timeout = 30  # 30 seconds
                elapsed = 0
                check_interval = 0.1  # Check every 100ms
                
                while block.metadata.get('exit_code') is None and elapsed < timeout:
                    time.sleep(check_interval)
                    elapsed += check_interval
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                if block.metadata.get('exit_code') is None:
                    # Command timed out
                    return TestResult(
                        command=cmd,
                        success=False,
                        duration_ms=duration_ms,
                        error="Command timed out",
                        exit_code=-1
                    )
                else:
                    # Command completed
                    output_size = len(str(block.output_data.get('data', '')))
                    exit_code = block.metadata.get('exit_code', -1)
                    
                    # Check if parsing was successful
                    parsed_correctly = block.output_data.get('type') != 'error'
                    
                    return TestResult(
                        command=cmd,
                        success=exit_code == 0,
                        duration_ms=duration_ms,
                        error=block.metadata.get('error'),
                        output_size=output_size,
                        exit_code=exit_code,
                        parsed_correctly=parsed_correctly
                    )
                    
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                return TestResult(
                    command=cmd,
                    success=False,
                    duration_ms=duration_ms,
                    error=str(e),
                    exit_code=-1,
                    parsed_correctly=False
                )
        
        # Execute commands in parallel (with a limit)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            
            task = progress.add_task("Executing commands...", total=num_commands)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for i in range(num_commands):
                    # Select a random command
                    cmd = random.choice(self.all_commands)
                    future = executor.submit(execute_command, cmd)
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    result = future.result()
                    results.append(result)
                    progress.update(task, advance=1)
        
        return results
    
    def run_parser_validation_test(self, terminal) -> Dict[str, List[TestResult]]:
        """Test parser accuracy for different command types"""
        results_by_category = {}
        
        self.console.print("[bold blue]Running parser validation tests...[/bold blue]")
        
        for category, commands in self.test_commands.items():
            category_results = []
            
            self.console.print(f"Testing {category} commands...")
            
            for cmd in commands:
                try:
                    block = terminal.run_command(cmd)
                    
                    # Validate parsing based on command type
                    parsed_correctly = self._validate_parsing(cmd, block.output_data)
                    
                    result = TestResult(
                        command=cmd,
                        success=block.metadata.get('exit_code') == 0,
                        duration_ms=block.metadata.get('duration_ms', 0),
                        error=block.metadata.get('error'),
                        output_size=len(str(block.output_data.get('data', ''))),
                        exit_code=block.metadata.get('exit_code', -1),
                        parsed_correctly=parsed_correctly
                    )
                    
                    category_results.append(result)
                    
                except Exception as e:
                    result = TestResult(
                        command=cmd,
                        success=False,
                        duration_ms=0,
                        error=str(e),
                        exit_code=-1,
                        parsed_correctly=False
                    )
                    category_results.append(result)
            
            results_by_category[category] = category_results
        
        return results_by_category
    
    def _validate_parsing(self, command: str, output_data: Dict[str, Any]) -> bool:
        """Validate that parsing was done correctly for the command type"""
        cmd_type = command.split()[0] if command.split() else ""
        output_type = output_data.get("type", "")
        
        # Validation rules for different command types
        if cmd_type == "git":
            if "status" in command:
                return output_type == "git_status"
            elif "log" in command:
                return output_type == "git_log"
            else:
                return output_type in ["git", "text"]
        
        elif cmd_type == "ls":
            return output_type == "file_list"
        
        elif cmd_type == "ps":
            return output_type == "process_list"
        
        elif cmd_type == "docker":
            if "ps" in command:
                return output_type == "process_list"
            elif "images" in command:
                return output_type == "docker_images"
            else:
                return output_type in ["docker", "text"]
        
        elif cmd_type in ["curl", "wget"] and "json" in command:
            return output_type == "json"
        
        # Default: any parsing is acceptable for unknown commands
        return True
    
    def generate_report(self, results: List[TestResult]) -> str:
        """Generate a comprehensive report from test results"""
        total_commands = len(results)
        successful_commands = sum(1 for r in results if r.success)
        failed_commands = total_commands - successful_commands
        parsing_errors = sum(1 for r in results if not r.parsed_correctly)
        
        avg_duration = sum(r.duration_ms for r in results) / total_commands if total_commands > 0 else 0
        max_duration = max(r.duration_ms for r in results) if results else 0
        min_duration = min(r.duration_ms for r in results) if results else 0
        
        total_output_size = sum(r.output_size for r in results)
        avg_output_size = total_output_size / total_commands if total_commands > 0 else 0
        
        # Performance categories
        fast_commands = sum(1 for r in results if r.duration_ms < 100)
        medium_commands = sum(1 for r in results if 100 <= r.duration_ms < 1000)
        slow_commands = sum(1 for r in results if r.duration_ms >= 1000)
        
        report = f"""# Xencode Warp Terminal Test Report

## Summary
- **Total Commands**: {total_commands}
- **Successful**: {successful_commands} ({successful_commands/total_commands*100:.1f}%)
- **Failed**: {failed_commands} ({failed_commands/total_commands*100:.1f}%)
- **Parsing Errors**: {parsing_errors} ({parsing_errors/total_commands*100:.1f}%)

## Performance Metrics
- **Average Duration**: {avg_duration:.2f}ms
- **Min Duration**: {min_duration}ms
- **Max Duration**: {max_duration}ms
- **Average Output Size**: {avg_output_size:.2f} bytes

## Performance Distribution
- **Fast (<100ms)**: {fast_commands} ({fast_commands/total_commands*100:.1f}%)
- **Medium (100-1000ms)**: {medium_commands} ({medium_commands/total_commands*100:.1f}%)
- **Slow (>1000ms)**: {slow_commands} ({slow_commands/total_commands*100:.1f}%)

## Failed Commands
"""
        
        for result in results:
            if not result.success:
                report += f"- `{result.command}`: {result.error}\n"
        
        if parsing_errors > 0:
            report += "\n## Parsing Errors\n"
            for result in results:
                if not result.parsed_correctly:
                    report += f"- `{result.command}`: Parsing failed\n"
        
        return report
    
    def display_results_table(self, results: List[TestResult]):
        """Display results in a formatted table"""
        table = Table(title="Command Execution Results")
        
        table.add_column("Command", style="cyan", width=30)
        table.add_column("Status", style="green", width=10)
        table.add_column("Duration", style="yellow", width=10)
        table.add_column("Output Size", style="blue", width=12)
        table.add_column("Parsed", style="magenta", width=8)
        
        for result in results:
            status = "✅ Pass" if result.success else "❌ Fail"
            parsed = "✅" if result.parsed_correctly else "❌"
            
            table.add_row(
                result.command[:27] + "..." if len(result.command) > 30 else result.command,
                status,
                f"{result.duration_ms}ms",
                f"{result.output_size}B",
                parsed
            )
        
        self.console.print(table)
    
    def run_performance_benchmark(self, terminal, iterations: int = 10) -> Dict[str, float]:
        """Run performance benchmarks for different command types"""
        benchmarks = {}
        
        self.console.print(f"[bold blue]Running performance benchmarks ({iterations} iterations)...[/bold blue]")
        
        for category, commands in self.test_commands.items():
            category_times = []
            
            for _ in range(iterations):
                for cmd in commands[:2]:  # Test first 2 commands in each category
                    start_time = time.time()
                    try:
                        block = terminal.run_command(cmd)
                        duration = time.time() - start_time
                        category_times.append(duration * 1000)  # Convert to ms
                    except Exception:
                        pass  # Skip failed commands in benchmark
            
            if category_times:
                benchmarks[category] = sum(category_times) / len(category_times)
        
        return benchmarks


# Usage example and main function
def run_comprehensive_test():
    """Run a comprehensive test suite on the Warp terminal"""
    from xencode.warp_terminal import WarpTerminal, example_ai_suggester
    
    console = Console()
    
    # Initialize terminal
    terminal = WarpTerminal(ai_suggester=example_ai_suggester)
    harness = CommandTestingHarness()
    
    console.print(Panel.fit(
        "[bold green]Xencode Warp Terminal Test Suite[/bold green]\n\n"
        "Running comprehensive tests for performance, reliability, and parsing accuracy.",
        title="Test Suite",
        border_style="green"
    ))
    
    # 1. Stress test
    console.print("\n[bold]1. Stress Test[/bold]")
    stress_results = harness.run_stress_test(terminal, num_commands=25)
    
    # 2. Parser validation
    console.print("\n[bold]2. Parser Validation Test[/bold]")
    parser_results = harness.run_parser_validation_test(terminal)
    
    # 3. Performance benchmark
    console.print("\n[bold]3. Performance Benchmark[/bold]")
    benchmark_results = harness.run_performance_benchmark(terminal)
    
    # Display results
    console.print("\n[bold]Stress Test Results:[/bold]")
    harness.display_results_table(stress_results)
    
    # Generate and display report
    console.print("\n[bold]Test Report:[/bold]")
    report = harness.generate_report(stress_results)
    console.print(Panel(report, title="Test Report", border_style="blue"))
    
    # Display benchmark results
    console.print("\n[bold]Performance Benchmarks:[/bold]")
    benchmark_table = Table(title="Average Command Duration by Category")
    benchmark_table.add_column("Category", style="cyan")
    benchmark_table.add_column("Avg Duration (ms)", style="yellow")
    
    for category, avg_time in benchmark_results.items():
        benchmark_table.add_row(category, f"{avg_time:.2f}")
    
    console.print(benchmark_table)
    
    return stress_results, parser_results, benchmark_results


if __name__ == "__main__":
    run_comprehensive_test()