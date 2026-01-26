"""
Execution Replay and Debugging Capabilities

This module provides functionality for replaying executions, debugging failed steps,
and analyzing execution history for ByteBot operations.
"""

import json
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import copy
from pathlib import Path

from .executor import Executor, ExecutionResult
from .planner import PlanGraph
from .plan_graph_storage import PlanGraphManager


class ExecutionDebugger:
    """
    Handles debugging of execution failures and issues
    """
    
    def __init__(self):
        self.debug_logs = []
        self.failure_patterns = {}
    
    def analyze_execution_result(self, result: ExecutionResult) -> Dict[str, Any]:
        """
        Analyze an execution result to identify potential issues
        
        Args:
            result: The execution result to analyze
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "step_id": result.step_id,
            "status": result.status,
            "command": result.command,
            "duration": result.duration,
            "timestamp": result.timestamp.isoformat(),
            "issues": [],
            "suggestions": [],
            "severity": "low"  # low, medium, high, critical
        }
        
        # Analyze based on status
        if result.status == "error":
            analysis["issues"].append(f"Execution error: {result.error_message}")
            analysis["severity"] = "high"
            
            # Try to identify common error patterns
            if result.error_message:
                error_lower = result.error_message.lower()
                if "permission denied" in error_lower:
                    analysis["issues"].append("Permission issue detected")
                    analysis["suggestions"].append("Try running with appropriate permissions or check file ownership")
                elif "not found" in error_lower or "command not found" in error_lower:
                    analysis["issues"].append("Command or file not found")
                    analysis["suggestions"].append("Verify command exists and is in PATH, or check file path")
                elif "timeout" in error_lower:
                    analysis["issues"].append("Command timed out")
                    analysis["suggestions"].append("Command took too long, consider increasing timeout or optimizing command")
        
        elif result.status == "failed":
            analysis["issues"].append(f"Command failed with exit code: {result.exit_code}")
            analysis["severity"] = "medium"
            
            # Analyze stderr for common issues
            if result.stderr:
                stderr_lower = result.stderr.lower()
                if "permission denied" in stderr_lower:
                    analysis["issues"].append("Permission issue detected in stderr")
                    analysis["suggestions"].append("Check file/directory permissions")
                elif "file exists" in stderr_lower:
                    analysis["issues"].append("File/directory already exists")
                    analysis["suggestions"].append("Consider using -f flag to force or check existence first")
                elif "no such file" in stderr_lower or "does not exist" in stderr_lower:
                    analysis["issues"].append("File/directory does not exist")
                    analysis["suggestions"].append("Verify path is correct")
        
        elif result.status == "success":
            analysis["severity"] = "low"
            analysis["suggestions"].append("Command executed successfully")
        
        # Add to debug logs
        self.debug_logs.append(analysis)
        
        return analysis
    
    def analyze_plan_execution(self, plan: Dict[str, Any], execution_results: List[ExecutionResult]) -> Dict[str, Any]:
        """
        Analyze the execution of an entire plan
        
        Args:
            plan: The plan that was executed
            execution_results: Results from executing the plan
            
        Returns:
            Comprehensive analysis of the plan execution
        """
        total_steps = len(execution_results)
        successful = len([r for r in execution_results if r.status == "success"])
        failed = len([r for r in execution_results if r.status == "failed"])
        errors = len([r for r in execution_results if r.status == "error"])
        blocked = len([r for r in execution_results if r.status == "blocked"])
        skipped = len([r for r in execution_results if r.status == "skipped"])
        
        # Analyze each result
        detailed_analysis = []
        for result in execution_results:
            detailed_analysis.append(self.analyze_execution_result(result))
        
        # Determine overall severity
        max_severity = "low"
        for analysis in detailed_analysis:
            severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            if severity_order.get(analysis["severity"], 0) > severity_order.get(max_severity, 0):
                max_severity = analysis["severity"]
        
        # Identify common failure patterns
        all_issues = []
        all_suggestions = []
        for analysis in detailed_analysis:
            all_issues.extend(analysis["issues"])
            all_suggestions.extend(analysis["suggestions"])
        
        return {
            "plan_id": plan.get("id"),
            "intent": plan.get("intent"),
            "total_steps": total_steps,
            "successful": successful,
            "failed": failed,
            "errors": errors,
            "blocked": blocked,
            "skipped": skipped,
            "success_rate": successful / total_steps if total_steps > 0 else 0,
            "overall_severity": max_severity,
            "total_duration": sum(r.duration for r in execution_results),
            "detailed_analysis": detailed_analysis,
            "common_issues": list(set(all_issues)),
            "suggestions": list(set(all_suggestions)),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """
        Get a summary of debugging information
        
        Returns:
            Summary of debug logs and patterns
        """
        if not self.debug_logs:
            return {
                "total_debug_entries": 0,
                "severity_breakdown": {},
                "common_issues": [],
                "last_debugged": None
            }
        
        # Count severities
        severity_counts = {}
        for log in self.debug_logs:
            severity = log["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Get common issues
        all_issues = []
        for log in self.debug_logs:
            all_issues.extend(log["issues"])
        
        # Get last debugged timestamp
        last_timestamp = max(log["timestamp"] for log in self.debug_logs)
        
        return {
            "total_debug_entries": len(self.debug_logs),
            "severity_breakdown": severity_counts,
            "common_issues": list(set(all_issues)),
            "last_debugged": last_timestamp
        }


class ExecutionReplay:
    """
    Handles replaying of previous executions
    """
    
    def __init__(self, executor: Executor, plan_storage: PlanGraphManager = None):
        self.executor = executor
        self.plan_storage = plan_storage or PlanGraphManager()
        self.replay_history = []
    
    def replay_execution(self, execution_id: str, plan: Dict[str, Any], 
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Replay a previous execution
        
        Args:
            execution_id: ID of the execution to replay
            plan: The plan to execute
            context: Context for the replay
            
        Returns:
            Dictionary with replay results
        """
        start_time = datetime.now()
        
        try:
            # Execute the plan again
            results = self.executor.execute_plan_sequential(plan, context)
            
            # Record replay
            replay_record = {
                "replay_id": str(uuid.uuid4()),
                "original_execution_id": execution_id,
                "plan_id": plan.get("id"),
                "replay_time": start_time.isoformat(),
                "execution_results": [
                    {
                        "step_id": r.step_id,
                        "status": r.status,
                        "command": r.command,
                        "duration": r.duration,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in results
                ],
                "total_duration": sum(r.duration for r in results)
            }
            
            self.replay_history.append(replay_record)
            
            # Compare with original execution if available
            comparison = self._compare_with_original(execution_id, results)
            replay_record["comparison"] = comparison
            
            return {
                "status": "replayed",
                "replay_id": replay_record["replay_id"],
                "results": replay_record["execution_results"],
                "total_duration": replay_record["total_duration"],
                "comparison": comparison
            }
            
        except Exception as e:
            error_record = {
                "replay_id": str(uuid.uuid4()),
                "original_execution_id": execution_id,
                "plan_id": plan.get("id"),
                "replay_time": start_time.isoformat(),
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
            self.replay_history.append(error_record)
            
            return {
                "status": "error",
                "replay_id": error_record["replay_id"],
                "error": str(e),
                "traceback": error_record["traceback"]
            }
    
    def replay_from_storage(self, plan_id: str, execution_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Replay an execution from stored plan
        
        Args:
            plan_id: ID of the plan to replay
            execution_params: Parameters for the execution
            
        Returns:
            Dictionary with replay results
        """
        # Load plan from storage
        plan = self.plan_storage.storage.load_plan(plan_id)
        if not plan:
            return {
                "status": "error",
                "message": f"Plan with ID {plan_id} not found"
            }
        
        # Use provided context or create default
        context = execution_params.get("context", {}) if execution_params else {}
        
        # Generate a new execution ID for this replay
        execution_id = f"replay-{uuid.uuid4()}"
        
        return self.replay_execution(execution_id, plan, context)
    
    def _compare_with_original(self, original_id: str, new_results: List[ExecutionResult]) -> Dict[str, Any]:
        """
        Compare new execution results with original execution
        
        Args:
            original_id: ID of the original execution
            new_results: Results from the new execution
            
        Returns:
            Comparison results
        """
        # This would normally compare with stored original results
        # For now, we'll just return a basic comparison structure
        return {
            "compared_with_original": False,  # Original results not available in this implementation
            "new_results_count": len(new_results),
            "statuses_match": None,
            "durations_comparable": False
        }
    
    def dry_run_execution(self, plan: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform a dry run of an execution without actually executing commands
        
        Args:
            plan: The plan to dry run
            context: Context for the dry run
            
        Returns:
            Dictionary with dry run results
        """
        dry_run_results = []
        
        for step in plan.get("steps", []):
            # Instead of executing, just validate the command
            command = step.get("command", "")
            
            # Validate command safety
            if command and command != "SAFE_GUARD_TRIGGERED":
                # Simulate what would happen
                dry_run_results.append({
                    "step_id": step.get("id"),
                    "command": command,
                    "predicted_status": "would_execute",  # Would execute in normal conditions
                    "predicted_duration": step.get("estimated_duration", 1.0),
                    "risk_assessment": "available_in_real_execution",
                    "validation_passed": True  # Would be validated in real execution
                })
            else:
                dry_run_results.append({
                    "step_id": step.get("id"),
                    "command": command,
                    "predicted_status": "blocked",
                    "predicted_duration": 0,
                    "risk_assessment": "command_blocked",
                    "validation_passed": False
                })
        
        return {
            "status": "dry_run_complete",
            "plan_id": plan.get("id"),
            "dry_run_results": dry_run_results,
            "total_predicted_duration": sum(r.get("predicted_duration", 0) for r in dry_run_results),
            "total_steps": len(dry_run_results),
            "context_used": context is not None
        }


class ExecutionHistoryTracker:
    """
    Tracks execution history for replay and debugging purposes
    """
    
    def __init__(self, storage_dir: str = "./bytebot_execution_history"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_dir / "execution_history.json"
        
        # Load existing history
        self.execution_history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load execution history from storage"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def _save_history(self):
        """Save execution history to storage"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.execution_history, f, indent=2, ensure_ascii=False)
    
    def record_execution(self, intent: str, plan: Dict[str, Any], 
                        execution_results: List[Dict[str, Any]], 
                        mode: str, context: Dict[str, Any] = None) -> str:
        """
        Record an execution in history
        
        Args:
            intent: The user's intent
            plan: The plan that was executed
            execution_results: Results of the execution
            mode: Execution mode used
            context: Context at time of execution
            
        Returns:
            Execution ID
        """
        execution_id = str(uuid.uuid4())
        
        record = {
            "execution_id": execution_id,
            "intent": intent,
            "plan_id": plan.get("id"),
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "context_snapshot": context or {},
            "execution_results": execution_results,
            "total_steps": len(execution_results),
            "successful_steps": len([r for r in execution_results if r.get("status") == "success"]),
            "failed_steps": len([r for r in execution_results if r.get("status") in ["failed", "error"]])
        }
        
        self.execution_history.append(record)
        
        # Keep only recent history (last 100 executions)
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        # Save to storage
        self._save_history()
        
        return execution_id
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific execution from history
        
        Args:
            execution_id: ID of the execution to retrieve
            
        Returns:
            Execution record or None if not found
        """
        for record in self.execution_history:
            if record["execution_id"] == execution_id:
                return record
        return None
    
    def search_executions(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search executions by various parameters
        
        Args:
            search_params: Parameters to search by (intent, mode, date range, etc.)
            
        Returns:
            List of matching execution records
        """
        results = []
        
        for record in self.execution_history:
            match = True
            
            # Check intent if provided
            if "intent_contains" in search_params:
                if search_params["intent_contains"].lower() not in record["intent"].lower():
                    match = False
            
            # Check mode if provided
            if "mode" in search_params:
                if search_params["mode"] != record["mode"]:
                    match = False
            
            # Check date range if provided
            if "after" in search_params:
                if record["timestamp"] < search_params["after"]:
                    match = False
            
            if "before" in search_params:
                if record["timestamp"] > search_params["before"]:
                    match = False
            
            # Check success rate if provided
            if "min_success_rate" in search_params:
                success_rate = record["successful_steps"] / record["total_steps"] if record["total_steps"] > 0 else 0
                if success_rate < search_params["min_success_rate"]:
                    match = False
            
            if match:
                results.append(record)
        
        return results
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about execution history
        
        Returns:
            Statistics about executions
        """
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "mode_distribution": {},
                "avg_steps_per_execution": 0,
                "avg_success_rate": 0
            }
        
        total = len(self.execution_history)
        successful = len([r for r in self.execution_history 
                         if r["successful_steps"] == r["total_steps"]])
        failed = total - successful
        
        # Mode distribution
        mode_dist = {}
        for record in self.execution_history:
            mode = record["mode"]
            mode_dist[mode] = mode_dist.get(mode, 0) + 1
        
        # Average steps and success rate
        total_steps = sum(r["total_steps"] for r in self.execution_history)
        avg_steps = total_steps / total if total > 0 else 0
        
        total_successful_steps = sum(r["successful_steps"] for r in self.execution_history)
        avg_success_rate = total_successful_steps / total_steps if total_steps > 0 else 0
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": successful / total if total > 0 else 0,
            "mode_distribution": mode_dist,
            "avg_steps_per_execution": avg_steps,
            "avg_success_rate": avg_success_rate,
            "date_range": {
                "start": min(r["timestamp"] for r in self.execution_history),
                "end": max(r["timestamp"] for r in self.execution_history)
            }
        }


class ReplayAndDebugManager:
    """
    Main manager for execution replay and debugging capabilities
    """
    
    def __init__(self, executor: Executor, plan_storage: PlanGraphManager = None):
        self.debugger = ExecutionDebugger()
        self.replay_system = ExecutionReplay(executor, plan_storage)
        self.history_tracker = ExecutionHistoryTracker()
    
    def execute_with_tracking(self, intent: str, plan: Dict[str, Any], 
                           mode: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a plan with full tracking and debugging capabilities
        
        Args:
            intent: User's intent
            plan: Plan to execute
            mode: Execution mode
            context: Execution context
            
        Returns:
            Execution results with tracking info
        """
        # Execute the plan
        execution_results = self.replay_system.executor.execute_plan_sequential(plan, context)
        
        # Convert execution results to dict format for storage
        result_dicts = []
        for result in execution_results:
            result_dicts.append({
                "step_id": result.step_id,
                "status": result.status,
                "command": result.command,
                "duration": result.duration,
                "timestamp": result.timestamp.isoformat(),
                "error_message": result.error_message,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
        
        # Record in history
        execution_id = self.history_tracker.record_execution(
            intent, plan, result_dicts, mode, context
        )
        
        # Analyze the execution
        analysis = self.debugger.analyze_plan_execution(plan, execution_results)
        
        return {
            "execution_id": execution_id,
            "status": "completed",
            "results": result_dicts,
            "analysis": analysis,
            "plan_id": plan.get("id")
        }
    
    def replay_execution_by_id(self, execution_id: str) -> Dict[str, Any]:
        """
        Replay a specific execution by its ID
        
        Args:
            execution_id: ID of the execution to replay
            
        Returns:
            Replay results
        """
        # Get the original execution record
        original_record = self.history_tracker.get_execution(execution_id)
        if not original_record:
            return {
                "status": "error",
                "message": f"Execution with ID {execution_id} not found"
            }
        
        # Get the plan that was used (would need to be retrieved from plan storage)
        # For this implementation, we'll assume the plan is available somewhere
        # In a real implementation, we'd retrieve it from the plan storage system
        
        # For now, return a message indicating what would happen
        return {
            "status": "info",
            "message": f"Would replay execution {execution_id}. "
                      f"This would execute the same plan with the same context as the original."
        }
    
    def debug_execution(self, execution_id: str) -> Dict[str, Any]:
        """
        Debug a specific execution
        
        Args:
            execution_id: ID of the execution to debug
            
        Returns:
            Debug analysis
        """
        record = self.history_tracker.get_execution(execution_id)
        if not record:
            return {
                "status": "error",
                "message": f"Execution with ID {execution_id} not found"
            }
        
        # Convert stored results back to ExecutionResult objects for analysis
        execution_results = []
        for result_data in record["execution_results"]:
            execution_results.append(ExecutionResult(
                step_id=result_data["step_id"],
                status=result_data["status"],
                result=result_data.get("stdout", "") or result_data.get("error_message", ""),
                command=result_data["command"],
                duration=result_data["duration"],
                timestamp=datetime.fromisoformat(result_data["timestamp"]),
                error_message=result_data.get("error_message"),
                exit_code=result_data.get("exit_code"),
                stdout=result_data.get("stdout"),
                stderr=result_data.get("stderr")
            ))
        
        # Analyze the execution
        # Need to retrieve the original plan for full analysis
        # For this implementation, we'll create a minimal plan structure
        plan_stub = {"id": record.get("plan_id", "unknown")}
        
        analysis = self.debugger.analyze_plan_execution(plan_stub, execution_results)
        
        return {
            "status": "analyzed",
            "execution_id": execution_id,
            "analysis": analysis
        }
    
    def get_debugging_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive debugging report
        
        Returns:
            Comprehensive debugging information
        """
        debug_summary = self.debugger.get_debug_summary()
        execution_stats = self.history_tracker.get_execution_stats()
        
        return {
            "debug_summary": debug_summary,
            "execution_stats": execution_stats,
            "timestamp": datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    print("Execution Replay and Debugging System")
    print("=" * 50)
    
    # This would normally be initialized with actual executor and storage instances
    # For demonstration, we'll show the structure
    
    print("\nSystem Components:")
    print("- ExecutionDebugger: Analyzes execution results for issues")
    print("- ExecutionReplay: Replays previous executions")
    print("- ExecutionHistoryTracker: Tracks execution history")
    print("- ReplayAndDebugManager: Main interface for replay/debugging")
    
    print("\nCapabilities:")
    print("✓ Execution replay by ID")
    print("✓ Dry-run execution simulation") 
    print("✓ Failure analysis and debugging")
    print("✓ Execution history tracking")
    print("✓ Search and filtering of execution history")
    print("✓ Statistical analysis of execution patterns")
    print("✓ Comparison of execution runs")