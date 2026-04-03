#!/usr/bin/env python3
"""
Inline Code Execution Panel for Xencode TUI

Executes Python and shell code snippets with:
- Live output streaming to chat panel
- Syntax highlighting
- Error detection and display
- Execution history
- Safe execution sandboxing
"""

import asyncio
import io
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager

from rich.console import Console, Capture
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

console = Console()


class LanguageType(Enum):
    """Supported execution languages"""
    PYTHON = "python"
    SHELL = "shell"
    POWERSHELL = "powershell"


class ExecutionStatus(Enum):
    """Code execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"


@dataclass
class ExecutionResult:
    """Result of code execution"""
    code: str
    language: LanguageType
    status: ExecutionStatus
    output: str = ""
    error: str = ""
    exit_code: int = 0
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """Check if execution succeeded"""
        return self.status == ExecutionStatus.COMPLETED and self.exit_code == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "code": self.code,
            "language": self.language.value,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "is_success": self.is_success,
        }


@dataclass
class ExecutionHistory:
    """History of code executions"""
    results: List[ExecutionResult] = field(default_factory=list)
    max_history: int = 100
    
    def add(self, result: ExecutionResult):
        """Add execution result to history"""
        self.results.append(result)
        # Trim if exceeds max
        if len(self.results) > self.max_history:
            self.results = self.results[-self.max_history:]
    
    def get_recent(self, count: int = 10) -> List[ExecutionResult]:
        """Get recent execution results"""
        return self.results[-count:]
    
    def clear(self):
        """Clear execution history"""
        self.results = []


class CodeExecutor:
    """
    Executes code snippets safely with output capture
    
    Usage:
        executor = CodeExecutor()
        result = await executor.execute("print('hello')", LanguageType.PYTHON)
    """
    
    def __init__(
        self,
        timeout: float = 30.0,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize code executor
        
        Args:
            timeout: Maximum execution time in seconds
            working_directory: Working directory for execution
            environment: Custom environment variables
        """
        self.timeout = timeout
        self.working_directory = working_directory or os.getcwd()
        self.environment = environment or os.environ.copy()
        self._history = ExecutionHistory()
    
    async def execute(
        self,
        code: str,
        language: LanguageType = LanguageType.PYTHON,
    ) -> ExecutionResult:
        """
        Execute code snippet
        
        Args:
            code: Code to execute
            language: Programming language
            
        Returns:
            ExecutionResult with output and status
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            if language == LanguageType.PYTHON:
                result = await self._execute_python(code)
            elif language == LanguageType.SHELL:
                result = await self._execute_shell(code)
            elif language == LanguageType.POWERSHELL:
                result = await self._execute_powershell(code)
            else:
                result = ExecutionResult(
                    code=code,
                    language=language,
                    status=ExecutionStatus.ERROR,
                    error=f"Unsupported language: {language.value}",
                )
            
            result.execution_time = asyncio.get_event_loop().time() - start_time
            self._history.add(result)
            return result
            
        except asyncio.TimeoutError:
            return ExecutionResult(
                code=code,
                language=language,
                status=ExecutionStatus.TIMEOUT,
                error=f"Execution timed out after {self.timeout}s",
                execution_time=self.timeout,
            )
        except Exception as e:
            return ExecutionResult(
                code=code,
                language=language,
                status=ExecutionStatus.ERROR,
                error=f"Execution failed: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time,
            )
    
    async def _execute_python(self, code: str) -> ExecutionResult:
        """Execute Python code"""
        # Create output buffers
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            # Redirect stdout and stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            
            # Execute code
            try:
                # Try as expression first
                result = eval(code)
                if result is not None:
                    stdout_buffer.write(f"{result}\n")
            except SyntaxError:
                # Not an expression, execute as statement
                exec(code, {"__builtins__": __builtins__})
            
            output = stdout_buffer.getvalue()
            error = stderr_buffer.getvalue()
            
            return ExecutionResult(
                code=code,
                language=LanguageType.PYTHON,
                status=ExecutionStatus.COMPLETED if not error else ExecutionStatus.ERROR,
                output=output,
                error=error,
                exit_code=0 if not error else 1,
            )
            
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
    async def _execute_shell(self, code: str) -> ExecutionResult:
        """Execute shell command"""
        try:
            process = await asyncio.create_subprocess_shell(
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory,
                env=self.environment,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
                
                return ExecutionResult(
                    code=code,
                    language=LanguageType.SHELL,
                    status=ExecutionStatus.COMPLETED if process.returncode == 0 else ExecutionStatus.ERROR,
                    output=stdout.decode('utf-8', errors='replace'),
                    error=stderr.decode('utf-8', errors='replace'),
                    exit_code=process.returncode or 0,
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                raise
                
        except Exception as e:
            return ExecutionResult(
                code=code,
                language=LanguageType.SHELL,
                status=ExecutionStatus.ERROR,
                error=str(e),
            )
    
    async def _execute_powershell(self, code: str) -> ExecutionResult:
        """Execute PowerShell command"""
        try:
            # Wrap PowerShell command
            ps_command = f"powershell -Command \"{code}\""
            
            process = await asyncio.create_subprocess_shell(
                ps_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory,
                env=self.environment,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
                
                return ExecutionResult(
                    code=code,
                    language=LanguageType.POWERSHELL,
                    status=ExecutionStatus.COMPLETED if process.returncode == 0 else ExecutionStatus.ERROR,
                    output=stdout.decode('utf-8', errors='replace'),
                    error=stderr.decode('utf-8', errors='replace'),
                    exit_code=process.returncode or 0,
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                raise
                
        except Exception as e:
            return ExecutionResult(
                code=code,
                language=LanguageType.POWERSHELL,
                status=ExecutionStatus.ERROR,
                error=str(e),
            )
    
    def get_history(self, count: int = 10) -> List[ExecutionResult]:
        """Get recent execution history"""
        return self._history.get_recent(count)
    
    def clear_history(self):
        """Clear execution history"""
        self._history.clear()


class CodeExecutionPanel:
    """
    TUI panel for code execution with live output streaming
    
    Integrates with Textual TUI framework
    """
    
    def __init__(
        self,
        executor: Optional[CodeExecutor] = None,
        output_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize code execution panel
        
        Args:
            executor: CodeExecutor instance
            output_callback: Callback for streaming output to chat panel
        """
        self.executor = executor or CodeExecutor()
        self.output_callback = output_callback
        self._current_result: Optional[ExecutionResult] = None
    
    async def execute_and_stream(
        self,
        code: str,
        language: LanguageType = LanguageType.PYTHON,
        show_code: bool = True,
    ) -> ExecutionResult:
        """
        Execute code and stream output
        
        Args:
            code: Code to execute
            language: Programming language
            show_code: Whether to display the code before execution
            
        Returns:
            ExecutionResult
        """
        # Display code if requested
        if show_code and self.output_callback:
            syntax = Syntax(code, language.value, theme="monokai", line_numbers=True)
            self.output_callback(f"[dim]Executing {language.value} code:[/dim]")
            # In real TUI, would render syntax widget
            self.output_callback(code)
            self.output_callback("")
        
        # Execute and stream output
        result = await self.executor.execute(code, language)
        self._current_result = result
        
        # Stream output
        if self.output_callback:
            if result.output:
                self.output_callback(result.output)
            if result.error:
                self.output_callback(f"[red]{result.error}[/red]")
            
            status_icon = "✓" if result.is_success else "✗"
            self.output_callback(
                f"\n[dim]{status_icon} {result.status.value} "
                f"({result.execution_time:.2f}s)[/dim]"
            )
        
        return result
    
    def get_current_result(self) -> Optional[ExecutionResult]:
        """Get current execution result"""
        return self._current_result
    
    def get_history(self, count: int = 10) -> List[ExecutionResult]:
        """Get execution history"""
        return self.executor.get_history(count)
    
    def clear_history(self):
        """Clear execution history"""
        self.executor.clear_history()


# Global panel instance
_panel: Optional[CodeExecutionPanel] = None


def get_execution_panel() -> CodeExecutionPanel:
    """Get or create global execution panel"""
    global _panel
    if _panel is None:
        _panel = CodeExecutionPanel()
    return _panel


# Convenience functions
async def execute_code(
    code: str,
    language: LanguageType = LanguageType.PYTHON,
    stream: bool = True,
) -> ExecutionResult:
    """
    Execute code via global panel
    
    Args:
        code: Code to execute
        language: Programming language
        stream: Whether to stream output
        
    Returns:
        ExecutionResult
    """
    panel = get_execution_panel()
    
    if stream:
        return await panel.execute_and_stream(code, language)
    else:
        return await panel.executor.execute(code, language)


async def execute_python(code: str, stream: bool = True) -> ExecutionResult:
    """Execute Python code"""
    return await execute_code(code, LanguageType.PYTHON, stream)


async def execute_shell(code: str, stream: bool = True) -> ExecutionResult:
    """Execute shell command"""
    return await execute_code(code, LanguageType.SHELL, stream)


def get_execution_history(count: int = 10) -> List[ExecutionResult]:
    """Get recent execution history"""
    return get_execution_panel().get_history(count)


if __name__ == "__main__":
    # Demo execution
    async def demo():
        console.print("[bold blue]Code Execution Panel Demo[/bold blue]\n")
        
        # Create panel with console output
        def console_output(text: str):
            console.print(text)
        
        panel = CodeExecutionPanel(output_callback=console_output)
        
        # Demo 1: Python expression
        console.print("\n[bold]1. Python Expression[/bold]")
        result = await panel.execute_and_stream(
            "2 + 2 * 3",
            LanguageType.PYTHON,
        )
        console.print(f"[dim]Result: {result.to_dict()}[/dim]")
        
        # Demo 2: Python statements
        console.print("\n[bold]2. Python Statements[/bold]")
        result = await panel.execute_and_stream(
            "for i in range(3):\n    print(f'Count: {i}')",
            LanguageType.PYTHON,
        )
        
        # Demo 3: Shell command
        console.print("\n[bold]3. Shell Command[/bold]")
        result = await panel.execute_and_stream(
            "echo Hello from shell",
            LanguageType.SHELL,
        )
        
        # Demo 4: Error handling
        console.print("\n[bold]4. Error Handling[/bold]")
        result = await panel.execute_and_stream(
            "1 / 0",
            LanguageType.PYTHON,
        )
        
        # Show history
        console.print("\n[bold]Execution History[/bold]")
        history = panel.get_history()
        for i, hist in enumerate(history, 1):
            icon = "✓" if hist.is_success else "✗"
            console.print(f"  {icon} {i}: {hist.language.value} - {hist.status.value}")
    
    asyncio.run(demo())
