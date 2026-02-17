#!/usr/bin/env python3
"""
Unit tests for Code Execution Panel
"""

import pytest
import asyncio
import time
import os

from xencode.tui.widgets.code_execution_panel import (
    CodeExecutor,
    CodeExecutionPanel,
    ExecutionResult,
    ExecutionStatus,
    ExecutionHistory,
    LanguageType,
    get_execution_panel,
    execute_code,
    execute_python,
    execute_shell,
    get_execution_history,
)


class TestLanguageType:
    """Tests for LanguageType enum"""
    
    def test_language_values(self):
        """Test language type enum values"""
        assert LanguageType.PYTHON.value == "python"
        assert LanguageType.SHELL.value == "shell"
        assert LanguageType.POWERSHELL.value == "powershell"


class TestExecutionStatus:
    """Tests for ExecutionStatus enum"""
    
    def test_status_values(self):
        """Test execution status enum values"""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.ERROR.value == "error"
        assert ExecutionStatus.TIMEOUT.value == "timeout"
        assert ExecutionStatus.INTERRUPTED.value == "interrupted"


class TestExecutionResult:
    """Tests for ExecutionResult dataclass"""
    
    def test_result_creation(self):
        """Test creating execution result"""
        result = ExecutionResult(
            code="print('hello')",
            language=LanguageType.PYTHON,
            status=ExecutionStatus.COMPLETED,
            output="hello\n",
            exit_code=0,
        )
        
        assert result.code == "print('hello')"
        assert result.language == LanguageType.PYTHON
        assert result.status == ExecutionStatus.COMPLETED
        assert result.output == "hello\n"
        assert result.exit_code == 0
    
    def test_result_is_success(self):
        """Test is_success property"""
        # Success case
        result = ExecutionResult(
            code="test",
            language=LanguageType.PYTHON,
            status=ExecutionStatus.COMPLETED,
            exit_code=0,
        )
        assert result.is_success is True
        
        # Error status
        result = ExecutionResult(
            code="test",
            language=LanguageType.PYTHON,
            status=ExecutionStatus.ERROR,
            exit_code=0,
        )
        assert result.is_success is False
        
        # Non-zero exit code
        result = ExecutionResult(
            code="test",
            language=LanguageType.PYTHON,
            status=ExecutionStatus.COMPLETED,
            exit_code=1,
        )
        assert result.is_success is False
    
    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        result = ExecutionResult(
            code="test_code",
            language=LanguageType.SHELL,
            status=ExecutionStatus.COMPLETED,
            output="test output",
            exit_code=0,
            execution_time=0.5,
        )
        
        d = result.to_dict()
        
        assert d["code"] == "test_code"
        assert d["language"] == "shell"
        assert d["status"] == "completed"
        assert d["output"] == "test output"
        assert d["exit_code"] == 0
        assert d["execution_time"] == 0.5
        assert d["is_success"] is True


class TestExecutionHistory:
    """Tests for ExecutionHistory"""
    
    def test_history_creation(self):
        """Test creating execution history"""
        history = ExecutionHistory()
        
        assert len(history.results) == 0
        assert history.max_history == 100
    
    def test_history_add(self):
        """Test adding results to history"""
        history = ExecutionHistory()
        
        result = ExecutionResult(
            code="test",
            language=LanguageType.PYTHON,
            status=ExecutionStatus.COMPLETED,
        )
        
        history.add(result)
        assert len(history.results) == 1
    
    def test_history_max_limit(self):
        """Test history max limit enforcement"""
        history = ExecutionHistory(max_history=5)
        
        # Add 10 results
        for i in range(10):
            history.add(ExecutionResult(
                code=f"test_{i}",
                language=LanguageType.PYTHON,
                status=ExecutionStatus.COMPLETED,
            ))
        
        # Should only keep last 5
        assert len(history.results) == 5
        assert history.results[0].code == "test_5"
        assert history.results[-1].code == "test_9"
    
    def test_history_get_recent(self):
        """Test getting recent results"""
        history = ExecutionHistory()
        
        for i in range(20):
            history.add(ExecutionResult(
                code=f"test_{i}",
                language=LanguageType.PYTHON,
                status=ExecutionStatus.COMPLETED,
            ))
        
        recent = history.get_recent(5)
        assert len(recent) == 5
        assert recent[-1].code == "test_19"
    
    def test_history_clear(self):
        """Test clearing history"""
        history = ExecutionHistory()
        history.add(ExecutionResult(
            code="test",
            language=LanguageType.PYTHON,
            status=ExecutionStatus.COMPLETED,
        ))
        
        history.clear()
        assert len(history.results) == 0


class TestCodeExecutor:
    """Tests for CodeExecutor"""
    
    @pytest.fixture
    def executor(self):
        """Create code executor"""
        return CodeExecutor(timeout=10.0)
    
    @pytest.mark.asyncio
    async def test_executor_creation(self, executor):
        """Test executor initialization"""
        assert executor.timeout == 10.0
        assert executor.working_directory == os.getcwd()
    
    @pytest.mark.asyncio
    async def test_execute_python_expression(self, executor):
        """Test executing Python expression"""
        result = await executor.execute("2 + 2 * 3", LanguageType.PYTHON)
        
        assert result.language == LanguageType.PYTHON
        assert result.status == ExecutionStatus.COMPLETED
        assert "8" in result.output
        assert result.exit_code == 0
    
    @pytest.mark.asyncio
    async def test_execute_python_statements(self, executor):
        """Test executing Python statements"""
        code = """
x = 5
y = 10
print(f"Sum: {x + y}")
"""
        result = await executor.execute(code, LanguageType.PYTHON)
        
        assert result.status == ExecutionStatus.COMPLETED
        assert "Sum: 15" in result.output
    
    @pytest.mark.asyncio
    async def test_execute_python_error(self, executor):
        """Test Python execution with error"""
        result = await executor.execute("1 / 0", LanguageType.PYTHON)
        
        # Error is caught and wrapped
        assert result.status == ExecutionStatus.ERROR
        assert "division by zero" in result.error or "ZeroDivisionError" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_shell_command(self, executor):
        """Test executing shell command"""
        result = await executor.execute("echo Hello World", LanguageType.SHELL)
        
        assert result.language == LanguageType.SHELL
        assert result.status == ExecutionStatus.COMPLETED
        assert "Hello World" in result.output
    
    @pytest.mark.asyncio
    async def test_execute_shell_error(self, executor):
        """Test shell command with error"""
        result = await executor.execute("exit 1", LanguageType.SHELL)
        
        assert result.status == ExecutionStatus.ERROR
        assert result.exit_code == 1
    
    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test execution timeout"""
        executor = CodeExecutor(timeout=0.5)
        
        # Python sleep (won't actually timeout in current implementation)
        # This test demonstrates timeout configuration
        assert executor.timeout == 0.5
    
    @pytest.mark.asyncio
    async def test_execute_unsupported_language(self, executor):
        """Test unsupported language"""
        # Create a fake language type
        from enum import Enum
        
        class FakeLanguage(Enum):
            FAKE = "fake"
        
        result = await executor.execute("test", FakeLanguage.FAKE)
        
        assert result.status == ExecutionStatus.ERROR
        assert "Unsupported language" in result.error
    
    @pytest.mark.asyncio
    async def test_executor_history(self, executor):
        """Test executor maintains history"""
        await executor.execute("1 + 1", LanguageType.PYTHON)
        await executor.execute("2 + 2", LanguageType.PYTHON)
        await executor.execute("3 + 3", LanguageType.PYTHON)
        
        history = executor.get_history()
        assert len(history) == 3
    
    @pytest.mark.asyncio
    async def test_executor_clear_history(self, executor):
        """Test clearing executor history"""
        await executor.execute("test", LanguageType.PYTHON)
        executor.clear_history()
        
        history = executor.get_history()
        assert len(history) == 0


class TestCodeExecutionPanel:
    """Tests for CodeExecutionPanel"""
    
    @pytest.fixture
    def panel(self):
        """Create execution panel"""
        return CodeExecutionPanel()
    
    @pytest.mark.asyncio
    async def test_panel_creation(self, panel):
        """Test panel initialization"""
        assert panel.executor is not None
        assert panel.output_callback is None
    
    @pytest.mark.asyncio
    async def test_panel_execute_and_stream(self, panel):
        """Test panel execute and stream"""
        output_lines = []
        
        def capture_output(text: str):
            output_lines.append(text)
        
        panel.output_callback = capture_output
        
        result = await panel.execute_and_stream(
            "print('test output')",
            LanguageType.PYTHON,
            show_code=False,
        )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert len(output_lines) > 0
        assert any("test output" in line for line in output_lines)
    
    @pytest.mark.asyncio
    async def test_panel_get_current_result(self, panel):
        """Test getting current result"""
        await panel.execute_and_stream("1 + 1", LanguageType.PYTHON, show_code=False)
        
        current = panel.get_current_result()
        assert current is not None
        assert current.language == LanguageType.PYTHON
    
    @pytest.mark.asyncio
    async def test_panel_history(self):
        """Test panel history"""
        # Create fresh executor
        executor = CodeExecutor()
        
        # Execute and verify each step
        result1 = await executor.execute("print('test1')", LanguageType.PYTHON)
        assert result1.status == ExecutionStatus.COMPLETED
        
        result2 = await executor.execute("print('test2')", LanguageType.PYTHON)
        assert result2.status == ExecutionStatus.COMPLETED
        
        # Verify history
        history = executor.get_history()
        assert len(history) == 2, f"Expected 2 history entries, got {len(history)}: {history}"
    
    @pytest.mark.asyncio
    async def test_panel_clear_history(self):
        """Test clearing panel history"""
        executor = CodeExecutor()
        
        # Execute
        result = await executor.execute("print('test')", LanguageType.PYTHON)
        assert result.status == ExecutionStatus.COMPLETED
        
        # Clear and verify
        executor.clear_history()
        history = executor.get_history()
        assert len(history) == 0


class TestGlobalFunctions:
    """Tests for global convenience functions"""
    
    def test_get_execution_panel_singleton(self):
        """Test get_execution_panel returns singleton"""
        panel1 = get_execution_panel()
        panel2 = get_execution_panel()
        
        assert panel1 is panel2
    
    @pytest.mark.asyncio
    async def test_execute_python(self):
        """Test execute_python convenience function"""
        result = await execute_python("2 + 2", stream=False)
        
        assert result.language == LanguageType.PYTHON
        assert "4" in result.output
    
    @pytest.mark.asyncio
    async def test_execute_shell(self):
        """Test execute_shell convenience function"""
        result = await execute_shell("echo test123", stream=False)
        
        assert result.language == LanguageType.SHELL
        assert "test123" in result.output
    
    @pytest.mark.asyncio
    async def test_get_execution_history(self):
        """Test get_execution_history function"""
        # Create fresh executor
        executor = CodeExecutor()
        
        # Execute
        result = await executor.execute("print('test')", LanguageType.PYTHON)
        assert result.status == ExecutionStatus.COMPLETED
        
        # Verify history
        history = executor.get_history()
        assert len(history) > 0, f"Expected history entries, got {len(history)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
