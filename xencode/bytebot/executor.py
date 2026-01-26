"""
Executor - Executes plan steps safely

This component executes individual steps from a plan graph with safety
validation and proper error handling.
"""

import subprocess
import threading
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import os
import tempfile
from dataclasses import dataclass

from .terminal_cognition_layer import TerminalCognitionLayer, CommandResult


@dataclass
class ExecutionResult:
    """Result of executing a single plan step"""
    step_id: str
    status: str  # success, failed, error, blocked, skipped
    result: Any
    command: str
    duration: float
    timestamp: datetime
    error_message: Optional[str] = None
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class Executor:
    """
    Executor component that safely executes plan steps
    """
    
    def __init__(self, terminal_layer: TerminalCognitionLayer = None):
        self.terminal_layer = terminal_layer or TerminalCognitionLayer()
        self.execution_history = []
        self.active_executions = {}
        self.max_concurrent_executions = 1  # For now, execute sequentially
    
    def execute_step(self, step: Dict[str, Any], context: Dict[str, Any] = None) -> ExecutionResult:
        """
        Execute a single step from the plan
        
        Args:
            step: The step to execute (with id, command, type, etc.)
            context: Context information for the execution
            
        Returns:
            ExecutionResult with execution details
        """
        step_id = step.get("id", str(uuid.uuid4()))
        command = step.get("command", "")
        step_type = step.get("type", "command")
        
        # Record start of execution
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        
        self.active_executions[execution_id] = {
            "step_id": step_id,
            "command": command,
            "start_time": start_time
        }
        
        try:
            # Update context if provided
            if context:
                # In a real implementation, we'd update the terminal layer's context
                pass
            
            # Execute based on step type
            if step_type == "command":
                result = self._execute_command_step(step, context)
            elif step_type == "validation":
                result = self._execute_validation_step(step, context)
            elif step_type == "conditional":
                result = self._execute_conditional_step(step, context)
            else:
                # Default to command execution for unknown types
                result = self._execute_command_step(step, context)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Create execution result
            execution_result = ExecutionResult(
                step_id=step_id,
                status=result.get("status", "success"),
                result=result.get("result", ""),
                command=command,
                duration=duration,
                timestamp=datetime.now(),
                error_message=result.get("error", None),
                exit_code=result.get("exit_code", None),
                stdout=result.get("stdout", None),
                stderr=result.get("stderr", None)
            )
            
            # Add to history
            self.execution_history.append(execution_result)
            
            # Remove from active executions
            del self.active_executions[execution_id]
            
            return execution_result
            
        except Exception as e:
            duration = time.time() - start_time
            error_result = ExecutionResult(
                step_id=step_id,
                status="error",
                result=f"Execution error: {str(e)}",
                command=command,
                duration=duration,
                timestamp=datetime.now(),
                error_message=str(e)
            )
            
            # Add to history
            self.execution_history.append(error_result)
            
            # Remove from active executions
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            return error_result
    
    def _execute_command_step(self, step: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a command-type step"""
        command = step.get("command", "")
        
        if not command or command == "SAFE_GUARD_TRIGGERED":
            return {
                "status": "blocked",
                "result": "Command was blocked by safety guard",
                "error": "Command blocked by safety mechanism"
            }
        
        # Execute the command using the terminal layer
        command_result = self.terminal_layer.execute_command_safe(command)
        
        if command_result.success:
            return {
                "status": "success",
                "result": command_result.stdout,
                "exit_code": command_result.exit_code,
                "stdout": command_result.stdout,
                "stderr": command_result.stderr
            }
        else:
            return {
                "status": "failed",
                "result": command_result.stderr or "Command execution failed",
                "error": command_result.stderr,
                "exit_code": command_result.exit_code,
                "stdout": command_result.stdout,
                "stderr": command_result.stderr
            }
    
    def _execute_validation_step(self, step: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a validation-type step"""
        command = step.get("command", "")
        original_command = step.get("metadata", {}).get("original_command", "")
        
        # For validation steps, we might just check if the original command is safe
        if original_command:
            # Validate the original command using the terminal layer
            validation = self.terminal_layer.validate_command(original_command)
            
            if validation["valid"]:
                return {
                    "status": "success",
                    "result": f"Validation passed for command: {original_command}",
                    "validation_details": validation
                }
            else:
                return {
                    "status": "failed",
                    "result": f"Validation failed for command: {original_command}",
                    "error": f"Validation errors: {validation['errors']}",
                    "validation_details": validation
                }
        else:
            # If no original command to validate, just return success
            return {
                "status": "success",
                "result": "Validation step completed (no command to validate)"
            }
    
    def _execute_conditional_step(self, step: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a conditional-type step"""
        # For now, just execute the command associated with the conditional
        command = step.get("command", "")
        
        if not command:
            return {
                "status": "skipped",
                "result": "No command to execute for conditional step"
            }
        
        # Execute the command
        command_result = self.terminal_layer.execute_command_safe(command)
        
        if command_result.success:
            return {
                "status": "success",
                "result": command_result.stdout,
                "exit_code": command_result.exit_code,
                "stdout": command_result.stdout,
                "stderr": command_result.stderr
            }
        else:
            return {
                "status": "failed",
                "result": command_result.stderr or "Conditional command execution failed",
                "error": command_result.stderr,
                "exit_code": command_result.exit_code,
                "stdout": command_result.stdout,
                "stderr": command_result.stderr
            }
    
    def execute_plan_sequential(self, plan: Dict[str, Any], context: Dict[str, Any] = None) -> List[ExecutionResult]:
        """
        Execute a plan sequentially, respecting dependencies
        
        Args:
            plan: The plan graph to execute
            context: Context information for the execution
            
        Returns:
            List of ExecutionResults for each step
        """
        steps = plan.get("steps", [])
        dependencies = plan.get("dependencies", [])  # List of (from_id, to_id) tuples
        
        # Build dependency graph
        dependency_map = {}
        dependents_map = {}
        
        for step in steps:
            step_id = step["id"]
            dependency_map[step_id] = set()
            dependents_map[step_id] = set()
        
        for from_id, to_id in dependencies:
            dependency_map[to_id].add(from_id)
            dependents_map[from_id].add(to_id)
        
        # Execute steps in order respecting dependencies
        results = []
        completed = set()
        remaining_steps = set(step["id"] for step in steps)
        
        while remaining_steps:
            # Find steps whose dependencies are all completed
            ready_steps = []
            for step_id in remaining_steps:
                if dependency_map[step_id].issubset(completed):
                    # Find the actual step data
                    step_data = next((s for s in steps if s["id"] == step_id), None)
                    if step_data:
                        ready_steps.append(step_data)
            
            if not ready_steps:
                # Circular dependency or missing dependency
                raise Exception(f"Unable to execute plan: remaining steps {remaining_steps} have unmet dependencies")
            
            # Execute ready steps (for now, just execute one at a time)
            step_to_execute = ready_steps[0]
            result = self.execute_step(step_to_execute, context)
            results.append(result)
            
            # Mark as completed
            completed.add(step_to_execute["id"])
            remaining_steps.remove(step_to_execute["id"])
        
        return results
    
    def execute_plan_with_timeout(self, plan: Dict[str, Any], context: Dict[str, Any] = None, 
                                 timeout_seconds: int = 300) -> List[ExecutionResult]:
        """
        Execute a plan with a timeout
        
        Args:
            plan: The plan graph to execute
            context: Context information for the execution
            timeout_seconds: Maximum time to spend executing the plan
            
        Returns:
            List of ExecutionResults for completed steps
        """
        start_time = time.time()
        
        def execution_worker():
            try:
                return self.execute_plan_sequential(plan, context)
            except Exception as e:
                # Return partial results if available
                return getattr(self, '_partial_results', [])
        
        # Execute in a separate thread with timeout
        execution_thread = threading.Thread(target=lambda: setattr(self, '_thread_result', execution_worker()))
        execution_thread.daemon = True
        execution_thread.start()
        execution_thread.join(timeout=timeout_seconds)
        
        if execution_thread.is_alive():
            # Execution timed out
            return [{
                "status": "timeout",
                "result": f"Plan execution timed out after {timeout_seconds} seconds",
                "step_id": "timeout_marker",
                "command": "timeout",
                "duration": timeout_seconds,
                "timestamp": datetime.now()
            }]
        
        return getattr(self, '_thread_result', [])
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about execution history"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "error_executions": 0,
                "average_duration": 0.0,
                "total_duration": 0.0
            }
        
        total = len(self.execution_history)
        successful = len([r for r in self.execution_history if r.status == "success"])
        failed = len([r for r in self.execution_history if r.status == "failed"])
        errors = len([r for r in self.execution_history if r.status == "error"])
        total_duration = sum(r.duration for r in self.execution_history)
        avg_duration = total_duration / total if total > 0 else 0.0
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": failed,
            "error_executions": errors,
            "average_duration": avg_duration,
            "total_duration": total_duration
        }
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution"""
        if execution_id in self.active_executions:
            # In a real implementation, we would terminate the process
            # For now, we'll just remove it from active executions
            del self.active_executions[execution_id]
            return True
        return False
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get list of currently active executions"""
        active_list = []
        for exec_id, exec_info in self.active_executions.items():
            duration = time.time() - exec_info["start_time"]
            active_list.append({
                "execution_id": exec_id,
                "step_id": exec_info["step_id"],
                "command": exec_info["command"],
                "duration": duration,
                "start_time": exec_info["start_time"]
            })
        return active_list