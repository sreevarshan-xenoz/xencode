#!/usr/bin/env python3
"""
Unit tests for Agent Workflow Orchestrator
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from xencode.agentic.workflow_orchestrator import (
    WorkflowOrchestrator,
    TaskContext,
    WorkflowState,
    StopReason,
    FailureType,
    get_orchestrator,
    execute_workflow,
)


class TestWorkflowState:
    """Tests for WorkflowState enum"""
    
    def test_state_values(self):
        """Test workflow state enum values"""
        assert WorkflowState.IDLE.value == "idle"
        assert WorkflowState.PLANNING.value == "planning"
        assert WorkflowState.EDITING.value == "editing"
        assert WorkflowState.TESTING.value == "testing"
        assert WorkflowState.FIXING.value == "fixing"
        assert WorkflowState.COMPLETED.value == "completed"
        assert WorkflowState.FAILED.value == "failed"
        assert WorkflowState.STOPPED.value == "stopped"


class TestStopReason:
    """Tests for StopReason enum"""
    
    def test_stop_reason_values(self):
        """Test stop reason enum values"""
        assert StopReason.PASS.value == "pass"
        assert StopReason.RISK.value == "risk"
        assert StopReason.CAP.value == "cap"
        assert StopReason.UNRECOVERABLE.value == "unrecoverable"
        assert StopReason.TIMEOUT.value == "timeout"
        assert StopReason.USER_STOP.value == "user_stop"


class TestFailureType:
    """Tests for FailureType enum"""
    
    def test_failure_type_values(self):
        """Test failure type enum values"""
        assert FailureType.SYNTAX_ERROR.value == "syntax_error"
        assert FailureType.TEST_FAILURE.value == "test_failure"
        assert FailureType.RUNTIME_ERROR.value == "runtime_error"
        assert FailureType.TIMEOUT.value == "timeout"
        assert FailureType.RISKY_OPERATION.value == "risky_operation"
        assert FailureType.UNKNOWN.value == "unknown"


class TestTaskContext:
    """Tests for TaskContext dataclass"""
    
    def test_context_creation(self):
        """Test creating task context"""
        context = TaskContext(
            task_id="test-001",
            original_prompt="Test task",
        )
        
        assert context.task_id == "test-001"
        assert context.original_prompt == "Test task"
        assert context.current_state == WorkflowState.IDLE
        assert context.iteration_count == 0
        assert context.max_iterations == 5
        assert context.stop_reason is None
        assert context.is_running is True
    
    def test_context_with_custom_max_iterations(self):
        """Test context with custom max iterations"""
        context = TaskContext(
            task_id="test-002",
            original_prompt="Test",
            max_iterations=10,
        )
        
        assert context.max_iterations == 10
        assert context.can_continue is True
    
    def test_context_elapsed_time(self):
        """Test elapsed time calculation"""
        context = TaskContext(
            task_id="test-003",
            original_prompt="Test",
            start_time=time.time() - 5.0,  # 5 seconds ago
        )
        
        elapsed = context.elapsed_time
        assert elapsed >= 5.0
        assert elapsed < 6.0
    
    def test_context_can_continue(self):
        """Test can_continue property"""
        context = TaskContext(
            task_id="test-004",
            original_prompt="Test",
            max_iterations=3,
        )
        
        # Should be able to continue initially
        assert context.can_continue is True
        
        # Hit iteration cap
        context.iteration_count = 3
        assert context.can_continue is False
        
        # Set to completed state
        context.iteration_count = 0
        context.current_state = WorkflowState.COMPLETED
        assert context.can_continue is False
    
    def test_context_to_dict(self):
        """Test converting context to dictionary"""
        context = TaskContext(
            task_id="test-005",
            original_prompt="Test task",
            max_iterations=10,
        )
        
        d = context.to_dict()
        
        assert d["task_id"] == "test-005"
        assert d["original_prompt"] == "Test task"
        assert d["current_state"] == "idle"
        assert d["iteration_count"] == 0
        assert d["max_iterations"] == 10
        assert d["is_running"] is True


class TestWorkflowOrchestrator:
    """Tests for WorkflowOrchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create test orchestrator"""
        return WorkflowOrchestrator(max_iterations=3, timeout_seconds=60.0)
    
    @pytest.mark.asyncio
    async def test_orchestrator_creation(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.max_iterations == 3
        assert orchestrator.timeout_seconds == 60.0
        assert len(orchestrator._active_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_execute_task_basic(self, orchestrator):
        """Test basic task execution"""
        result = await orchestrator.execute_task(
            task_id="test-001",
            prompt="Test task",
        )
        
        assert result.task_id == "test-001"
        assert result.original_prompt == "Test task"
        assert result.iteration_count > 0
        assert result.end_time is not None
    
    @pytest.mark.asyncio
    async def test_execute_task_with_callbacks(self):
        """Test task execution with callbacks"""
        # Mock callbacks
        model_callback = AsyncMock(return_value="Test plan")
        edit_callback = AsyncMock(return_value={"file": "test.py", "action": "edit"})
        test_callback = AsyncMock(return_value={"success": True})
        
        orchestrator = WorkflowOrchestrator(
            max_iterations=3,
            model_callback=model_callback,
            edit_callback=edit_callback,
            test_callback=test_callback,
        )
        
        result = await orchestrator.execute_task(
            task_id="test-002",
            prompt="Test with callbacks",
        )
        
        # Verify callbacks were called
        assert model_callback.called
        assert edit_callback.called
        assert test_callback.called
        
        # Verify result
        assert result.task_id == "test-002"
        assert result.plan == "Test plan"
        assert len(result.edits_made) > 0
        assert len(result.test_results) > 0
    
    @pytest.mark.asyncio
    async def test_execute_task_test_failure_then_pass(self):
        """Test task that fails tests then passes"""
        call_count = 0
        
        async def test_callback():
            nonlocal call_count
            call_count += 1
            # Fail first time, pass second time
            return {"success": call_count > 1}
        
        orchestrator = WorkflowOrchestrator(
            max_iterations=5,
            test_callback=test_callback,
        )
        
        result = await orchestrator.execute_task(
            task_id="test-003",
            prompt="Test with retry",
        )
        
        # Should have run at least 2 iterations
        assert result.iteration_count >= 2
        assert result.stop_reason == StopReason.PASS
    
    @pytest.mark.asyncio
    async def test_execute_task_iteration_cap(self):
        """Test task that hits iteration cap"""
        async def test_callback():
            # Always fail
            return {"success": False, "errors": ["Test failed"]}
        
        orchestrator = WorkflowOrchestrator(
            max_iterations=3,
            test_callback=test_callback,
        )
        
        result = await orchestrator.execute_task(
            task_id="test-004",
            prompt="Test iteration cap",
        )
        
        assert result.iteration_count == 3
        assert result.stop_reason == StopReason.CAP
        assert result.current_state == WorkflowState.STOPPED
    
    @pytest.mark.asyncio
    async def test_execute_task_timeout(self):
        """Test task that times out"""
        async def slow_model_callback(prompt):
            await asyncio.sleep(0.2)
            return "Slow plan"
        
        orchestrator = WorkflowOrchestrator(
            max_iterations=10,
            timeout_seconds=0.1,  # 100ms timeout
            model_callback=slow_model_callback,
            edit_callback=AsyncMock(return_value={}),
            test_callback=AsyncMock(return_value={"success": False}),
        )
        
        result = await orchestrator.execute_task(
            task_id="test-005",
            prompt="Test timeout",
        )
        
        assert result.stop_reason == StopReason.TIMEOUT
        assert result.failure_type == FailureType.TIMEOUT
    
    @pytest.mark.asyncio
    async def test_execute_task_risk_detection(self):
        """Test task that detects risky operation"""
        async def risky_edit_callback(prompt):
            raise Exception("This operation is risky and unsafe")
        
        orchestrator = WorkflowOrchestrator(
            max_iterations=3,
            edit_callback=risky_edit_callback,
        )
        
        result = await orchestrator.execute_task(
            task_id="test-006",
            prompt="Test risk detection",
        )
        
        assert result.stop_reason == StopReason.RISK
        assert result.failure_type == FailureType.RISKY_OPERATION
    
    @pytest.mark.asyncio
    async def test_get_task(self, orchestrator):
        """Test getting task by ID"""
        await orchestrator.execute_task(
            task_id="test-007",
            prompt="Test",
        )
        
        task = orchestrator.get_task("test-007")
        assert task is not None
        assert task.task_id == "test-007"
    
    @pytest.mark.asyncio
    async def test_get_all_tasks(self, orchestrator):
        """Test getting all tasks"""
        await orchestrator.execute_task("task-1", "Test 1")
        await orchestrator.execute_task("task-2", "Test 2")
        await orchestrator.execute_task("task-3", "Test 3")
        
        all_tasks = orchestrator.get_all_tasks()
        assert len(all_tasks) == 3
    
    @pytest.mark.asyncio
    async def test_stop_task(self, orchestrator):
        """Test manually stopping a task"""
        # Start a task in background
        async def slow_task():
            return await orchestrator.execute_task(
                task_id="test-008",
                prompt="Slow task",
            )
        
        # Don't await, just start
        task = asyncio.create_task(slow_task())
        
        # Give it a moment to start
        await asyncio.sleep(0.01)
        
        # Stop it
        stopped = await orchestrator.stop_task("test-008")
        
        # Wait for task to complete
        result = await task
        
        # Task should be stopped or completed before stop could take effect
        assert stopped or result.current_state != WorkflowState.IDLE
    
    @pytest.mark.asyncio
    async def test_task_error_logging(self):
        """Test that errors are properly logged"""
        error_count = 0
        
        async def failing_test_callback():
            nonlocal error_count
            error_count += 1
            return {"success": False, "errors": [f"Error {error_count}"]}
        
        orchestrator = WorkflowOrchestrator(
            max_iterations=3,
            test_callback=failing_test_callback,
        )
        
        result = await orchestrator.execute_task(
            task_id="test-009",
            prompt="Test error logging",
        )
        
        # Should have logged errors
        assert len(result.error_log) > 0


class TestGetOrchestrator:
    """Tests for get_orchestrator helper"""
    
    def test_get_orchestrator_singleton(self):
        """Test that get_orchestrator returns singleton"""
        # Reset singleton
        import xencode.agentic.workflow_orchestrator as wo
        wo._orchestrator = None
        
        orch1 = get_orchestrator()
        orch2 = get_orchestrator()
        
        assert orch1 is orch2
    
    def test_get_orchestrator_with_params(self):
        """Test get_orchestrator with custom params"""
        # Reset singleton
        import xencode.agentic.workflow_orchestrator as wo
        wo._orchestrator = None
        
        orch = get_orchestrator(max_iterations=10, timeout_seconds=120.0)
        
        assert orch.max_iterations == 10
        assert orch.timeout_seconds == 120.0


class TestExecuteWorkflow:
    """Tests for execute_workflow convenience function"""
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        """Test execute_workflow function"""
        result = await execute_workflow(
            task_id="func-test-001",
            prompt="Test via function",
        )
        
        assert result.task_id == "func-test-001"
        assert result.original_prompt == "Test via function"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
