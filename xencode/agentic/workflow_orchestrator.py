#!/usr/bin/env python3
"""
Agent Workflow Orchestrator MVP

Implements autonomous Plan → Edit → Test → Fix loop with:
- Bounded iterations
- Failure classification
- Multiple stop reasons (pass, risk, cap, unrecoverable)
- Task context tracking
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from rich.console import Console

console = Console()


class WorkflowState(Enum):
    """Workflow execution states"""
    IDLE = "idle"
    PLANNING = "planning"
    EDITING = "editing"
    TESTING = "testing"
    FIXING = "fixing"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class StopReason(Enum):
    """Reasons for workflow termination"""
    PASS = "pass"  # Task completed successfully
    RISK = "risk"  # Risky operation detected, requires human intervention
    CAP = "cap"  # Iteration cap reached
    UNRECOVERABLE = "unrecoverable"  # Unrecoverable error
    TIMEOUT = "timeout"  # Execution timeout
    USER_STOP = "user_stop"  # User manually stopped


class FailureType(Enum):
    """Types of failures that can occur"""
    SYNTAX_ERROR = "syntax_error"
    TEST_FAILURE = "test_failure"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    RISKY_OPERATION = "risky_operation"
    UNKNOWN = "unknown"


@dataclass
class TaskContext:
    """
    Context object for tracking task execution state
    
    Attributes:
        task_id: Unique identifier for this task
        original_prompt: The original user request
        current_state: Current workflow state
        iteration_count: Number of Plan→Edit→Test→Fix cycles
        max_iterations: Maximum allowed iterations
        stop_reason: Why the workflow stopped (if applicable)
        failure_type: Type of failure (if applicable)
        plan: Current execution plan
        edits_made: List of file edits performed
        test_results: Results of test executions
        error_log: Log of errors encountered
        start_time: When the task started
        end_time: When the task ended
        metadata: Additional context/metadata
    """
    task_id: str
    original_prompt: str
    current_state: WorkflowState = WorkflowState.IDLE
    iteration_count: int = 0
    max_iterations: int = 5
    stop_reason: Optional[StopReason] = None
    failure_type: Optional[FailureType] = None
    plan: Optional[str] = None
    edits_made: List[Dict[str, Any]] = field(default_factory=list)
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    error_log: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed execution time"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def is_running(self) -> bool:
        """Check if workflow is still running"""
        return self.current_state not in [
            WorkflowState.COMPLETED,
            WorkflowState.FAILED,
            WorkflowState.STOPPED
        ]
    
    @property
    def can_continue(self) -> bool:
        """Check if workflow can continue iterating"""
        if not self.is_running:
            return False
        if self.iteration_count >= self.max_iterations:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "task_id": self.task_id,
            "original_prompt": self.original_prompt,
            "current_state": self.current_state.value,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "plan": self.plan,
            "edits_made": self.edits_made,
            "test_results": self.test_results,
            "error_log": self.error_log,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_time": self.elapsed_time,
            "is_running": self.is_running,
            "metadata": self.metadata,
        }


class WorkflowOrchestrator:
    """
    Main workflow orchestrator for autonomous agent tasks
    
    Implements Plan → Edit → Test → Fix loop with bounded iterations
    and comprehensive failure handling.
    
    Usage:
        orchestrator = WorkflowOrchestrator()
        result = await orchestrator.execute_task(
            task_id="task-001",
            prompt="Add a function to calculate fibonacci numbers",
            workspace_path="/path/to/project"
        )
    """
    
    def __init__(
        self,
        max_iterations: int = 5,
        timeout_seconds: float = 300.0,
        model_callback: Optional[Callable] = None,
        edit_callback: Optional[Callable] = None,
        test_callback: Optional[Callable] = None,
    ):
        """
        Initialize workflow orchestrator
        
        Args:
            max_iterations: Maximum Plan→Edit→Test→Fix cycles
            timeout_seconds: Maximum execution time
            model_callback: Callback for LLM calls (plan/fix generation)
            edit_callback: Callback for applying code edits
            test_callback: Callback for running tests
        """
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.model_callback = model_callback
        self.edit_callback = edit_callback
        self.test_callback = test_callback
        
        # Active tasks
        self._active_tasks: Dict[str, TaskContext] = {}
    
    async def execute_task(
        self,
        task_id: str,
        prompt: str,
        workspace_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskContext:
        """
        Execute a complete workflow task
        
        Args:
            task_id: Unique identifier for this task
            prompt: User's task description
            workspace_path: Path to project workspace
            context: Additional context/metadata
            
        Returns:
            TaskContext with execution results
        """
        # Create task context
        task_context = TaskContext(
            task_id=task_id,
            original_prompt=prompt,
            max_iterations=self.max_iterations,
            start_time=time.time(),
            metadata=context or {},
        )
        
        if workspace_path:
            task_context.metadata["workspace_path"] = workspace_path
        
        self._active_tasks[task_id] = task_context
        
        console.print(f"\n[bold blue]Starting Task: {task_id}[/bold blue]")
        console.print(f"[dim]Prompt: {prompt}[/dim]\n")
        
        try:
            # Execute workflow loop
            await self._workflow_loop(task_context)
        except Exception as e:
            task_context.current_state = WorkflowState.FAILED
            task_context.stop_reason = StopReason.UNRECOVERABLE
            task_context.failure_type = FailureType.UNKNOWN
            task_context.error_log.append(f"Unexpected error: {str(e)}")
            console.print(f"[red]Task failed with error: {e}[/red]")
        finally:
            task_context.end_time = time.time()
            self._log_completion(task_context)
        
        return task_context
    
    async def _workflow_loop(self, context: TaskContext) -> None:
        """
        Main workflow loop: Plan → Edit → Test → Fix
        
        Continues until task passes, fails, or hits iteration cap
        """
        while context.can_continue:
            context.iteration_count += 1
            console.print(f"\n[bold cyan]=== Iteration {context.iteration_count}/{context.max_iterations} ===[/bold cyan]\n")
            
            # Check timeout
            if context.elapsed_time > self.timeout_seconds:
                context.current_state = WorkflowState.FAILED
                context.stop_reason = StopReason.TIMEOUT
                context.failure_type = FailureType.TIMEOUT
                context.error_log.append(f"Timeout after {context.elapsed_time:.1f}s")
                return
            
            # Phase 1: Planning
            console.print("[bold]Phase 1: Planning[/bold]")
            context.current_state = WorkflowState.PLANNING
            plan_success = await self._plan_phase(context)
            
            if not plan_success:
                context.current_state = WorkflowState.FAILED
                context.stop_reason = StopReason.UNRECOVERABLE
                return
            
            # Phase 2: Editing
            console.print("\n[bold]Phase 2: Editing[/bold]")
            context.current_state = WorkflowState.EDITING
            edit_success = await self._edit_phase(context)
            
            if not edit_success:
                # Check if it's a risky operation
                if context.failure_type == FailureType.RISKY_OPERATION:
                    context.current_state = WorkflowState.STOPPED
                    context.stop_reason = StopReason.RISK
                    return
                context.current_state = WorkflowState.FAILED
                context.stop_reason = StopReason.UNRECOVERABLE
                return
            
            # Phase 3: Testing
            console.print("\n[bold]Phase 3: Testing[/bold]")
            context.current_state = WorkflowState.TESTING
            test_result = await self._test_phase(context)
            
            if test_result == "pass":
                context.current_state = WorkflowState.COMPLETED
                context.stop_reason = StopReason.PASS
                console.print("\n[green]✓ Task completed successfully![/green]")
                return
            
            # Phase 4: Fixing (if tests failed)
            console.print("\n[bold]Phase 4: Fixing[/bold]")
            context.current_state = WorkflowState.FIXING
            fix_success = await self._fix_phase(context)
            
            if not fix_success:
                context.current_state = WorkflowState.FAILED
                context.stop_reason = StopReason.UNRECOVERABLE
                return
        
        # If we exit the loop, we hit the iteration cap
        context.current_state = WorkflowState.STOPPED
        context.stop_reason = StopReason.CAP
        console.print(f"\n[yellow]⚠ Stopped: Reached maximum iterations ({context.max_iterations})[/yellow]")
    
    async def _plan_phase(self, context: TaskContext) -> bool:
        """
        Planning phase: Generate execution plan
        
        Returns:
            True if planning succeeded
        """
        console.print("Generating execution plan...")
        
        if self.model_callback:
            try:
                # Call LLM to generate plan
                plan_prompt = self._build_plan_prompt(context)
                plan = await self.model_callback(plan_prompt)
                context.plan = plan
                console.print(f"[green]✓ Plan generated[/green]")
                console.print(f"[dim]{plan}[/dim]")
                return True
            except Exception as e:
                context.error_log.append(f"Planning failed: {str(e)}")
                console.print(f"[red]✗ Planning failed: {e}[/red]")
                return False
        else:
            # Default: use iteration count as simple plan
            context.plan = f"Iteration {context.iteration_count}: Analyze task and implement solution"
            console.print(f"[green]✓ Plan: {context.plan}[/green]")
            return True
    
    async def _edit_phase(self, context: TaskContext) -> bool:
        """
        Editing phase: Apply code changes
        
        Returns:
            True if editing succeeded
        """
        console.print("Applying code edits...")
        
        if self.edit_callback:
            try:
                # Call LLM to generate edits
                edit_prompt = self._build_edit_prompt(context)
                edits = await self.edit_callback(edit_prompt)
                
                # Apply edits (callback should handle actual file modification)
                if isinstance(edits, list):
                    context.edits_made.extend(edits)
                else:
                    context.edits_made.append(edits)
                
                console.print(f"[green]✓ Edits applied ({len(edits) if isinstance(edits, list) else 1} changes)[/green]")
                return True
            except Exception as e:
                error_msg = str(e)
                context.error_log.append(f"Editing failed: {error_msg}")
                
                # Check for risky operations
                if any(term in error_msg.lower() for term in ["risky", "dangerous", "unsafe"]):
                    context.failure_type = FailureType.RISKY_OPERATION
                    console.print(f"[red]✗ Risky operation detected[/red]")
                else:
                    console.print(f"[red]✗ Editing failed: {e}[/red]")
                return False
        else:
            # No callback - simulate edit
            context.edits_made.append({
                "file": "simulated_file.py",
                "action": "edit",
                "description": "Simulated edit (no callback configured)",
            })
            console.print("[yellow]⚠ No edit callback configured - simulating edit[/yellow]")
            return True
    
    async def _test_phase(self, context: TaskContext) -> str:
        """
        Testing phase: Run tests and validate changes
        
        Returns:
            "pass" if tests pass, "fail" otherwise
        """
        console.print("Running tests...")
        
        if self.test_callback:
            try:
                # Run tests via callback
                test_results = await self.test_callback()
                context.test_results.append(test_results)
                
                if test_results.get("success", False):
                    console.print("[green]✓ All tests passed[/green]")
                    return "pass"
                else:
                    errors = test_results.get("errors", ["Unknown test failure"])
                    context.error_log.extend(errors)
                    context.failure_type = FailureType.TEST_FAILURE
                    console.print(f"[red]✗ Tests failed: {errors[0]}[/red]")
                    return "fail"
            except Exception as e:
                context.error_log.append(f"Testing failed: {str(e)}")
                context.failure_type = FailureType.RUNTIME_ERROR
                console.print(f"[red]✗ Testing failed: {e}[/red]")
                return "fail"
        else:
            # No callback - simulate test result
            # For demo, pass on first iteration, fail on others
            if context.iteration_count == 1:
                console.print("[yellow]⚠ No test callback - simulating test failure[/yellow]")
                context.failure_type = FailureType.TEST_FAILURE
                context.error_log.append("Simulated test failure (no callback configured)")
                return "fail"
            else:
                console.print("[green]✓ Simulated test pass[/green]")
                return "pass"
    
    async def _fix_phase(self, context: TaskContext) -> bool:
        """
        Fixing phase: Generate and apply fixes for test failures
        
        Returns:
            True if fixing succeeded
        """
        console.print("Generating fixes...")
        
        if self.model_callback:
            try:
                # Call LLM to generate fix
                fix_prompt = self._build_fix_prompt(context)
                fix = await self.model_callback(fix_prompt)
                
                # Apply fix via edit callback
                if self.edit_callback:
                    edits = await self.edit_callback(fix_prompt)
                    if isinstance(edits, list):
                        context.edits_made.extend(edits)
                    else:
                        context.edits_made.append(edits)
                
                console.print("[green]✓ Fix applied[/green]")
                return True
            except Exception as e:
                context.error_log.append(f"Fixing failed: {str(e)}")
                console.print(f"[red]✗ Fixing failed: {e}[/red]")
                return False
        else:
            console.print("[yellow]⚠ No model callback - simulating fix[/yellow]")
            context.edits_made.append({
                "action": "fix",
                "description": "Simulated fix (no callback configured)",
            })
            return True
    
    def _build_plan_prompt(self, context: TaskContext) -> str:
        """Build prompt for planning phase"""
        return f"""Task: {context.original_prompt}

Iteration: {context.iteration_count}/{context.max_iterations}

Please analyze this task and create a step-by-step plan to implement it.
Include:
1. Files that need to be modified
2. Changes to make
3. Tests to run
4. Potential risks to consider

Be specific and actionable."""
    
    def _build_edit_prompt(self, context: TaskContext) -> str:
        """Build prompt for edit phase"""
        return f"""Task: {context.original_prompt}

Plan: {context.plan}

Please implement the plan by generating the necessary code changes.
For each file:
1. Show the file path
2. Show the complete updated content or a clear diff
3. Explain the changes made

Be precise and ensure the code is syntactically correct."""
    
    def _build_fix_prompt(self, context: TaskContext) -> str:
        """Build prompt for fix phase"""
        errors = "\n".join(context.error_log[-3:])  # Last 3 errors
        
        return f"""Task: {context.original_prompt}

Previous attempts failed with these errors:
{errors}

Please analyze the errors and generate a fix.
Include:
1. Root cause of the failure
2. Specific code changes needed
3. Updated file content

Be thorough in addressing all error conditions."""
    
    def _log_completion(self, context: TaskContext) -> None:
        """Log task completion summary"""
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Task Completion Summary[/bold]")
        console.print(f"{'='*60}")
        console.print(f"Task ID: {context.task_id}")
        console.print(f"Status: {context.current_state.value}")
        console.print(f"Stop Reason: {context.stop_reason.value if context.stop_reason else 'N/A'}")
        console.print(f"Iterations: {context.iteration_count}/{context.max_iterations}")
        console.print(f"Elapsed Time: {context.elapsed_time:.2f}s")
        console.print(f"Edits Made: {len(context.edits_made)}")
        console.print(f"Tests Run: {len(context.test_results)}")
        
        if context.error_log:
            console.print(f"\n[red]Errors ({len(context.error_log)}):[/red]")
            for error in context.error_log[-5:]:  # Show last 5 errors
                console.print(f"  • {error}")
        
        console.print(f"{'='*60}\n")
    
    def get_task(self, task_id: str) -> Optional[TaskContext]:
        """Get task context by ID"""
        return self._active_tasks.get(task_id)
    
    def get_all_tasks(self) -> List[TaskContext]:
        """Get all task contexts"""
        return list(self._active_tasks.values())
    
    async def stop_task(self, task_id: str) -> bool:
        """Manually stop a running task"""
        context = self.get_task(task_id)
        if not context or not context.is_running:
            return False
        
        context.current_state = WorkflowState.STOPPED
        context.stop_reason = StopReason.USER_STOP
        context.end_time = time.time()
        
        console.print(f"[yellow]Task {task_id} stopped by user[/yellow]")
        return True


# Global orchestrator instance
_orchestrator: Optional[WorkflowOrchestrator] = None


def get_orchestrator(
    max_iterations: int = 5,
    timeout_seconds: float = 300.0,
) -> WorkflowOrchestrator:
    """Get or create global orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = WorkflowOrchestrator(
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
        )
    return _orchestrator


async def execute_workflow(
    task_id: str,
    prompt: str,
    workspace_path: Optional[str] = None,
    model_callback: Optional[Callable] = None,
    edit_callback: Optional[Callable] = None,
    test_callback: Optional[Callable] = None,
) -> TaskContext:
    """
    Convenience function to execute a workflow task
    
    Args:
        task_id: Unique task identifier
        prompt: Task description
        workspace_path: Path to project
        model_callback: LLM callback
        edit_callback: Edit application callback
        test_callback: Test execution callback
        
    Returns:
        TaskContext with results
    """
    orchestrator = WorkflowOrchestrator(
        model_callback=model_callback,
        edit_callback=edit_callback,
        test_callback=test_callback,
    )
    
    return await orchestrator.execute_task(
        task_id=task_id,
        prompt=prompt,
        workspace_path=workspace_path,
    )


if __name__ == "__main__":
    # Demo execution
    async def demo():
        console.print("[bold blue]Agent Workflow Orchestrator Demo[/bold blue]\n")
        
        # Demo without callbacks (simulated execution)
        orchestrator = WorkflowOrchestrator(max_iterations=3)
        
        result = await orchestrator.execute_task(
            task_id="demo-001",
            prompt="Add a hello world function",
            workspace_path="/tmp/demo",
        )
        
        console.print("\n[bold]Result:[/bold]")
        console.print(f"Status: {result.current_state.value}")
        console.print(f"Stop Reason: {result.stop_reason.value if result.stop_reason else 'N/A'}")
        console.print(f"Iterations: {result.iteration_count}")
    
    asyncio.run(demo())
