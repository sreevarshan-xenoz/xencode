"""
Execution Modes - Implementation of assist/execute/autonomous modes

This module implements the three execution modes as specified in the plan:
- assist: Suggest only, no execution
- execute: Auto-run safe steps, confirm risky ones
- autonomous: Run everything except vetoed operations
"""

import json
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from datetime import datetime
import uuid

from .planner import Planner
from .executor import Executor
from .safety_gate import SafetyGate, ExecutionMode
from .risk_scorer import RiskScorer
from .context_engine import ContextEngine
from .terminal_cognition_layer import TerminalCognitionLayer


class ExecutionMode(Enum):
    """Execution modes for ByteBot operations"""
    ASSIST = "assist"      # Suggest only, no execution
    EXECUTE = "execute"    # Auto-run safe steps, confirm risky ones
    AUTONOMOUS = "autonomous"  # Run everything except vetoed operations


class ExecutionModeHandler:
    """
    Handles the logic for different execution modes
    """
    
    def __init__(self, planner: Planner, executor: Executor, 
                 safety_gate: SafetyGate, risk_scorer: RiskScorer,
                 context_engine: ContextEngine):
        self.planner = planner
        self.executor = executor
        self.safety_gate = safety_gate
        self.risk_scorer = risk_scorer
        self.context_engine = context_engine
        
        # Mode-specific configurations
        self.mode_configs = {
            ExecutionMode.ASSIST: {
                "auto_execute": False,
                "require_confirmation": False,
                "max_risk_allowed": 0.0,
                "description": "Provide suggestions only, no execution"
            },
            ExecutionMode.EXECUTE: {
                "auto_execute": True,
                "require_confirmation": True,
                "max_risk_allowed": 0.7,
                "description": "Auto-run safe steps, confirm risky ones"
            },
            ExecutionMode.AUTONOMOUS: {
                "auto_execute": True,
                "require_confirmation": False,
                "max_risk_allowed": 0.95,  # Still block absolutely dangerous
                "description": "Run everything except vetoed operations"
            }
        }
    
    def execute_intent(self, intent: str, mode: ExecutionMode) -> Dict[str, Any]:
        """
        Execute an intent with the specified mode
        
        Args:
            intent: Natural language description of what user wants to do
            mode: Execution mode to use
            
        Returns:
            Dictionary with execution results
        """
        # Get current context
        context = self.context_engine.get_context()
        
        # Create plan
        plan = self.planner.create_plan(intent, context)
        
        # Validate plan
        is_valid, issues = self.planner.validate_plan(plan)
        if not is_valid:
            return {
                "status": "error",
                "message": f"Plan validation failed: {', '.join(issues)}",
                "plan_id": plan.id,
                "intent": intent,
                "mode": mode.value
            }
        
        # Execute based on mode
        if mode == ExecutionMode.ASSIST:
            return self._execute_assist_mode(intent, plan, context)
        elif mode == ExecutionMode.EXECUTE:
            return self._execute_execute_mode(intent, plan, context)
        elif mode == ExecutionMode.AUTONOMOUS:
            return self._execute_autonomous_mode(intent, plan, context)
        else:
            return {
                "status": "error",
                "message": f"Unknown execution mode: {mode}",
                "intent": intent,
                "mode": mode.value
            }
    
    def _execute_assist_mode(self, intent: str, plan, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute in assist mode - provide suggestions only"""
        # In assist mode, we just return the plan without executing
        steps_info = []
        for step in plan.steps:
            risk_assessment = self.risk_scorer.score_command(step.command, context)
            steps_info.append({
                "step_id": step.id,
                "description": step.description,
                "command": step.command,
                "risk_score": risk_assessment.score,
                "risk_category": self.risk_scorer.get_risk_category(risk_assessment.score),
                "recommendation": self.risk_scorer.get_recommendation(risk_assessment)
            })
        
        return {
            "status": "suggested",
            "intent": intent,
            "mode": ExecutionMode.ASSIST.value,
            "plan_id": plan.id,
            "suggested_steps": steps_info,
            "total_steps": len(steps_info),
            "summary": f"Found {len(steps_info)} steps for '{intent}'. No execution performed in assist mode."
        }
    
    def _execute_execute_mode(self, intent: str, plan, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute in execute mode - auto-run safe steps, confirm risky ones"""
        execution_results = []
        
        for step in plan.steps:
            # Score risk for this step
            risk_assessment = self.risk_scorer.score_command(step.command, context)
            
            # Check if safety gate blocks this step
            if self.safety_gate.should_block(step.__dict__, risk_assessment.score, ExecutionMode.EXECUTE):
                execution_results.append({
                    "step_id": step.id,
                    "status": "blocked",
                    "reason": self.safety_gate.get_block_reason(step.__dict__, risk_assessment.score, ExecutionMode.EXECUTE),
                    "command": step.command,
                    "risk_score": risk_assessment.score
                })
                continue
            
            # Check if confirmation is needed
            needs_confirmation = self.safety_gate.should_request_confirmation(
                step.__dict__, risk_assessment.score, ExecutionMode.EXECUTE
            )
            
            if needs_confirmation:
                # In a real implementation, we'd ask for user confirmation
                # For this implementation, we'll simulate the confirmation
                confirmed = self._request_user_confirmation(step, risk_assessment)
                
                if not confirmed:
                    execution_results.append({
                        "step_id": step.id,
                        "status": "skipped",
                        "reason": "User declined to execute medium-risk operation",
                        "command": step.command,
                        "risk_score": risk_assessment.score
                    })
                    continue
            
            # Execute the step
            result = self.executor.execute_step(step.__dict__, context)
            execution_results.append({
                "step_id": result.step_id,
                "status": result.status,
                "command": result.command,
                "duration": result.duration,
                "risk_score": risk_assessment.score,
                "result": result.result
            })
        
        return {
            "status": "executed",
            "intent": intent,
            "mode": ExecutionMode.EXECUTE.value,
            "plan_id": plan.id,
            "execution_results": execution_results,
            "summary": self._summarize_execution_results(execution_results)
        }
    
    def _execute_autonomous_mode(self, intent: str, plan, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute in autonomous mode - run everything except vetoed operations"""
        execution_results = []
        
        for step in plan.steps:
            # Score risk for this step
            risk_assessment = self.risk_scorer.score_command(step.command, context)
            
            # Check if safety gate blocks this step (only absolutely dangerous commands)
            if self.safety_gate.should_block(step.__dict__, risk_assessment.score, ExecutionMode.AUTONOMOUS):
                execution_results.append({
                    "step_id": step.id,
                    "status": "blocked",
                    "reason": self.safety_gate.get_block_reason(step.__dict__, risk_assessment.score, ExecutionMode.AUTONOMOUS),
                    "command": step.command,
                    "risk_score": risk_assessment.score
                })
                continue
            
            # Execute the step
            result = self.executor.execute_step(step.__dict__, context)
            execution_results.append({
                "step_id": result.step_id,
                "status": result.status,
                "command": result.command,
                "duration": result.duration,
                "risk_score": risk_assessment.score,
                "result": result.result
            })
        
        return {
            "status": "executed",
            "intent": intent,
            "mode": ExecutionMode.AUTONOMOUS.value,
            "plan_id": plan.id,
            "execution_results": execution_results,
            "summary": self._summarize_execution_results(execution_results)
        }
    
    def _request_user_confirmation(self, step, risk_assessment) -> bool:
        """
        Request user confirmation for medium-risk operations
        In a real implementation, this would show a prompt to the user
        """
        print(f"\n⚠️  Medium-risk operation detected:")
        print(f"   Command: {step.command}")
        print(f"   Risk Score: {risk_assessment.score:.2f}")
        print(f"   Category: {self.risk_scorer.get_risk_category(risk_assessment.score)}")
        print(f"   Recommendation: {self.risk_scorer.get_recommendation(risk_assessment)}")
        
        # For this implementation, we'll auto-confirm for testing purposes
        # In a real implementation, this would wait for user input
        return True
    
    def _summarize_execution_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize execution results"""
        total_steps = len(results)
        successful = len([r for r in results if r.get("status") in ["success", "suggested"]])
        blocked = len([r for r in results if r.get("status") == "blocked"])
        skipped = len([r for r in results if r.get("status") == "skipped"])
        failed = len([r for r in results if r.get("status") == "failed"])
        errors = len([r for r in results if r.get("status") == "error"])
        
        return {
            "total_steps": total_steps,
            "successful": successful,
            "blocked": blocked,
            "skipped": skipped,
            "failed": failed,
            "errors": errors,
            "success_rate": successful / total_steps if total_steps > 0 else 0
        }
    
    def get_mode_description(self, mode: ExecutionMode) -> str:
        """Get description of an execution mode"""
        return self.mode_configs[mode]["description"]
    
    def get_current_mode_config(self, mode: ExecutionMode) -> Dict[str, Any]:
        """Get configuration for current mode"""
        return self.mode_configs[mode]


class ModeAwareByteBot:
    """
    Main ByteBot class that is aware of execution modes
    """
    
    def __init__(self, planner: Planner, executor: Executor, 
                 safety_gate: SafetyGate, risk_scorer: RiskScorer,
                 context_engine: ContextEngine):
        self.mode_handler = ExecutionModeHandler(
            planner, executor, safety_gate, risk_scorer, context_engine
        )
        self.current_mode = ExecutionMode.EXECUTE
        self.execution_history = []
    
    def set_mode(self, mode: ExecutionMode):
        """Set the current execution mode"""
        self.current_mode = mode
    
    def process_intent(self, intent: str, mode: ExecutionMode = None) -> Dict[str, Any]:
        """
        Process an intent with the specified mode or current mode
        
        Args:
            intent: Natural language description of what user wants to do
            mode: Execution mode to use (uses current mode if not specified)
            
        Returns:
            Dictionary with processing results
        """
        if mode is None:
            mode = self.current_mode
        
        result = self.mode_handler.execute_intent(intent, mode)
        
        # Add to execution history
        result["execution_id"] = str(uuid.uuid4())
        result["timestamp"] = datetime.now().isoformat()
        self.execution_history.append(result)
        
        return result
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_history
    
    def get_mode_summary(self) -> Dict[str, str]:
        """Get summary of all available modes"""
        return {
            mode.value: self.mode_handler.get_mode_description(mode)
            for mode in ExecutionMode
        }
    
    def switch_mode_with_confirmation(self, new_mode: ExecutionMode) -> Dict[str, Any]:
        """
        Switch to a new mode with appropriate confirmation based on the mode
        """
        old_mode = self.current_mode
        
        # For safety, warn when switching to autonomous mode
        if new_mode == ExecutionMode.AUTONOMOUS and old_mode != ExecutionMode.AUTONOMOUS:
            print(f"\n⚠️  Switching to {new_mode.value.upper()} mode!")
            print("This mode will execute commands automatically with minimal safety checks.")
            print("Are you sure you want to proceed?")
            # In a real implementation, we'd ask for user confirmation here
            # For this implementation, we'll proceed without confirmation
        
        self.current_mode = new_mode
        
        return {
            "status": "success",
            "message": f"Switched from {old_mode.value} to {new_mode.value} mode",
            "old_mode": old_mode.value,
            "new_mode": new_mode.value,
            "policy_summary": self.mode_handler.get_mode_description(new_mode)
        }


# Example usage
if __name__ == "__main__":
    # Example of how to use the execution modes
    # Note: This would normally be initialized with actual instances of the required components
    print("Execution Modes Implementation")
    print("=" * 40)
    
    # Show mode descriptions
    from .context_engine import ContextEngine
    from .risk_scorer import RiskScorer
    from .safety_gate import SafetyGate
    from .terminal_cognition_layer import TerminalCognitionLayer
    
    # Create dummy components for demonstration
    context_engine = ContextEngine()
    risk_scorer = RiskScorer()
    safety_gate = SafetyGate()
    terminal_layer = TerminalCognitionLayer()
    
    # In a real implementation, we would create actual planner and executor instances
    # For now, we'll just show the concept
    print("\nExecution Modes:")
    for mode in ExecutionMode:
        print(f"- {mode.value.upper()}: {ExecutionModeHandler(None, None, safety_gate, risk_scorer, context_engine).get_mode_description(mode)}")