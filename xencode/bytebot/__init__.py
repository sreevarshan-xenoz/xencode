"""
ByteBot Terminal Cognition Layer - Core Module

Implements the ByteBotEngine as specified in the integration plan:
- Single-brain, multi-tool architecture
- Execution modes (assist/execute/autonomous)
- Risk-based safety validation
- Plan graph generation and execution
- Context-aware command execution
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json
import subprocess
import platform
from datetime import datetime
import uuid

from ..core import ModelManager
from ..shell_genie.genie import ShellGenie

# Import the new components
from .context_engine import ContextEngine
from .planner import Planner
from .executor import Executor
from .safety_gate import SafetyGate, ExecutionMode
from .risk_scorer import RiskScorer
from .execution_modes import ModeAwareByteBot
from .plan_graph_storage import PlanGraphManager
from .replay_debug import ReplayAndDebugManager


class ByteBotEngine:
    """
    The core ByteBot engine implementing single-brain, multi-tool architecture
    with execution modes, risk scoring, and plan graph generation.
    """

    def __init__(self, model_manager=None, tool_registry=None):
        self.model_manager = model_manager or ModelManager()
        self.tool_registry = tool_registry or {}

        # Initialize the new components
        self.context_engine = ContextEngine()
        self.risk_scorer = RiskScorer()
        self.safety_gate = SafetyGate()

        # Initialize planner with context engine and risk scorer
        self.planner = Planner(self.context_engine, self.risk_scorer)

        # Initialize executor with terminal cognition layer
        from .terminal_cognition_layer import TerminalCognitionLayer
        terminal_layer = TerminalCognitionLayer(self.model_manager)
        self.executor = Executor(terminal_layer)

        # Initialize plan storage and management
        self.plan_manager = PlanGraphManager()

        # Initialize execution replay and debugging
        self.replay_debug_manager = ReplayAndDebugManager(self.executor, self.plan_manager)

        # Initialize mode-aware ByteBot
        self.mode_aware_bot = ModeAwareByteBot(
            self.planner, self.executor, self.safety_gate,
            self.risk_scorer, self.context_engine
        )

        # Current execution context
        self.current_mode = ExecutionMode.EXECUTE
        self.execution_history = []

    def process_intent(self, intent: str, mode: str = "execute") -> Dict[str, Any]:
        """
        Process user intent with specified execution mode

        Args:
            intent: Natural language description of what user wants to do
            mode: Execution mode ('assist', 'execute', 'autonomous')

        Returns:
            Dictionary containing plan, execution results, and status
        """
        # Convert string mode to enum
        try:
            execution_mode = ExecutionMode(mode)
        except ValueError:
            execution_mode = ExecutionMode.EXECUTE

        # Use the mode-aware bot to process the intent
        return self.mode_aware_bot.process_intent(intent, execution_mode)

    def execute_plan_graph(self, plan_graph: Dict[str, Any], mode: ExecutionMode = None) -> Dict[str, Any]:
        """
        Execute a plan graph with safety validation and risk scoring
        """
        if mode is None:
            mode = self.current_mode

        execution_results = []
        execution_id = str(uuid.uuid4())

        for step in plan_graph.get("steps", []):
            # Score risk for this step
            risk_score = self.risk_scorer.score_command(step.get("command", ""),
                                                      self.context_engine.get_context())

            # Apply safety gate
            if self.safety_gate.should_block(step, risk_score, mode):
                execution_results.append({
                    "step_id": step.get("id"),
                    "status": "blocked",
                    "reason": "Safety gate blocked execution",
                    "risk_score": risk_score
                })
                continue

            # Execute based on mode and risk
            if mode == ExecutionMode.AUTONOMOUS:
                # In autonomous mode, execute everything that passes safety gate
                result = self.executor.execute_step(step)
            else:
                # In execute mode, confirm medium-risk operations
                if risk_score >= 0.3 and risk_score < 0.7:
                    # Ask for confirmation for medium-risk operations
                    confirmed = self._confirm_medium_risk(step, risk_score)
                    if not confirmed:
                        execution_results.append({
                            "step_id": step.get("id"),
                            "status": "skipped",
                            "reason": "User declined medium-risk operation",
                            "risk_score": risk_score
                        })
                        continue

                result = self.executor.execute_step(step)

            execution_results.append(result)

        # Record execution in history
        execution_record = {
            "id": execution_id,
            "timestamp": datetime.now().isoformat(),
            "plan": plan_graph,
            "results": execution_results,
            "mode": mode.value
        }
        self.execution_history.append(execution_record)

        return {
            "execution_id": execution_id,
            "results": execution_results,
            "summary": self._summarize_execution(execution_results)
        }

    def replay_execution(self, execution_id: str) -> Dict[str, Any]:
        """
        Re-execute a previous task for debugging
        """
        # Use the replay manager to handle execution replay
        return self.replay_debug_manager.replay_execution_by_id(execution_id)

    def _generate_suggestions(self, plan_graph: Dict[str, Any]) -> List[str]:
        """Generate suggestions for the planned steps"""
        suggestions = []
        for step in plan_graph.get("steps", []):
            command = step.get("command", "")
            description = step.get("description", "")
            suggestions.append(f"{description}: {command}")
        return suggestions

    def _confirm_medium_risk(self, step: Dict[str, Any], risk_score: float) -> bool:
        """Ask user to confirm medium-risk operations"""
        # In a real implementation, this would show a prompt to the user
        # For now, we'll return True to continue execution
        print(f"Medium-risk operation detected (score: {risk_score}): {step.get('command')}")
        print(f"Description: {step.get('description', 'No description')}")
        # In a real implementation, we'd ask for user confirmation here
        return True  # For now, auto-confirm for testing

    def _summarize_execution(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize execution results"""
        total_steps = len(results)
        successful = len([r for r in results if r.get("status") == "success"])
        blocked = len([r for r in results if r.get("status") == "blocked"])
        skipped = len([r for r in results if r.get("status") == "skipped"])

        return {
            "total_steps": total_steps,
            "successful": successful,
            "blocked": blocked,
            "skipped": skipped,
            "success_rate": successful / total_steps if total_steps > 0 else 0
        }

    def set_mode(self, mode: ExecutionMode):
        """Set the current execution mode"""
        self.mode_aware_bot.set_mode(mode)
        self.current_mode = mode

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.mode_aware_bot.get_execution_history()

    def save_plan(self, plan: Dict[str, Any]) -> str:
        """Save a plan to storage"""
        return self.plan_manager.storage.save_plan(plan)

    def load_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Load a plan from storage"""
        return self.plan_manager.storage.load_plan(plan_id)

    def debug_execution(self, execution_id: str) -> Dict[str, Any]:
        """Debug a specific execution"""
        return self.replay_debug_manager.debug_execution(execution_id)

    def get_debugging_report(self) -> Dict[str, Any]:
        """Get a comprehensive debugging report"""
        return self.replay_debug_manager.get_debugging_report()