"""
Planner - Generates plan graphs from user intents

This component takes user intents and creates structured plan graphs
that can be executed by the Executor component.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
from dataclasses import dataclass

from ..shell_genie.genie import ShellGenie
from .context_engine import ContextEngine
from .risk_scorer import RiskScorer


@dataclass
class PlanStep:
    """Represents a single step in a plan"""
    id: str
    type: str  # command, conditional, loop, etc.
    description: str
    command: str
    dependencies: List[str]  # IDs of steps this step depends on
    estimated_risk: float
    estimated_duration: float  # in seconds
    metadata: Dict[str, Any]


@dataclass
class PlanGraph:
    """Represents a complete plan graph"""
    id: str
    intent: str
    steps: List[PlanStep]
    dependencies: List[Tuple[str, str]]  # (from_step_id, to_step_id)
    context: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]


class Planner:
    """
    Planner component that generates plan graphs from user intents
    """
    
    def __init__(self, context_engine: ContextEngine = None, risk_scorer: RiskScorer = None):
        self.shell_genie = ShellGenie()
        self.context_engine = context_engine or ContextEngine()
        self.risk_scorer = risk_scorer or RiskScorer()
        self.plan_history = []
    
    def create_plan(self, intent: str, context: Dict[str, Any] = None) -> PlanGraph:
        """
        Create a plan graph from user intent and context
        
        Args:
            intent: Natural language description of what user wants to do
            context: Context information to consider in planning
            
        Returns:
            PlanGraph with structured steps
        """
        if context is None:
            context = self.context_engine.get_context()
        
        # Analyze the intent to determine the type of plan needed
        plan_type = self._analyze_intent_type(intent)
        
        # Generate plan based on type
        if plan_type == "simple_command":
            plan_graph = self._create_simple_command_plan(intent, context)
        elif plan_type == "file_operation":
            plan_graph = self._create_file_operation_plan(intent, context)
        elif plan_type == "git_operation":
            plan_graph = self._create_git_operation_plan(intent, context)
        elif plan_type == "development_task":
            plan_graph = self._create_development_task_plan(intent, context)
        else:
            # Default to simple command plan
            plan_graph = self._create_simple_command_plan(intent, context)
        
        # Calculate risk for each step
        for step in plan_graph.steps:
            risk_assessment = self.risk_scorer.score_command(step.command, context)
            step.estimated_risk = risk_assessment.score
        
        # Store in history
        self.plan_history.append(plan_graph)
        
        return plan_graph
    
    def _analyze_intent_type(self, intent: str) -> str:
        """Analyze intent to determine the type of plan needed"""
        intent_lower = intent.lower()
        
        # Check for file operations
        file_keywords = ["file", "create", "delete", "copy", "move", "rename", "edit", "modify", "read", "write"]
        if any(keyword in intent_lower for keyword in file_keywords):
            return "file_operation"
        
        # Check for git operations
        git_keywords = ["git", "commit", "push", "pull", "checkout", "branch", "merge", "clone", "status"]
        if any(keyword in intent_lower for keyword in git_keywords):
            return "git_operation"
        
        # Check for development tasks
        dev_keywords = ["build", "compile", "test", "run", "install", "deploy", "start", "stop", "restart"]
        if any(keyword in intent_lower for keyword in dev_keywords):
            return "development_task"
        
        # Default to simple command
        return "simple_command"
    
    def _create_simple_command_plan(self, intent: str, context: Dict[str, Any]) -> PlanGraph:
        """Create a simple command execution plan"""
        # Use ShellGenie to generate the command
        command, explanation = self.shell_genie.generate_command(intent)
        
        # Create a single-step plan
        step = PlanStep(
            id=str(uuid.uuid4()),
            type="command",
            description=explanation,
            command=command,
            dependencies=[],
            estimated_risk=0.0,  # Will be calculated later
            estimated_duration=2.0,  # Default estimate
            metadata={
                "intent": intent,
                "context_used": True
            }
        )
        
        plan_graph = PlanGraph(
            id=str(uuid.uuid4()),
            intent=intent,
            steps=[step],
            dependencies=[],
            context=context,
            timestamp=datetime.now(),
            metadata={
                "plan_type": "simple_command",
                "generated_by": "ByteBot Planner",
                "context_included": True
            }
        )
        
        return plan_graph
    
    def _create_file_operation_plan(self, intent: str, context: Dict[str, Any]) -> PlanGraph:
        """Create a plan for file operations"""
        # Use ShellGenie to generate the command
        command, explanation = self.shell_genie.generate_command(intent)
        
        # For file operations, we might want to add validation steps
        steps = []
        
        # Validation step - check if operation is safe
        validation_step = PlanStep(
            id=str(uuid.uuid4()),
            type="validation",
            description="Validate file operation safety",
            command=f"# Validate: {command}",  # Not a real command, just for tracking
            dependencies=[],
            estimated_risk=0.0,
            estimated_duration=0.5,
            metadata={
                "validation_type": "file_operation",
                "original_command": command
            }
        )
        
        # Actual operation step
        operation_step = PlanStep(
            id=str(uuid.uuid4()),
            type="command",
            description=explanation,
            command=command,
            dependencies=[validation_step.id],
            estimated_risk=0.0,  # Will be calculated later
            estimated_duration=2.0,
            metadata={
                "intent": intent,
                "operation_type": "file"
            }
        )
        
        steps = [validation_step, operation_step]
        
        plan_graph = PlanGraph(
            id=str(uuid.uuid4()),
            intent=intent,
            steps=steps,
            dependencies=[(validation_step.id, operation_step.id)],
            context=context,
            timestamp=datetime.now(),
            metadata={
                "plan_type": "file_operation",
                "generated_by": "ByteBot Planner",
                "context_included": True
            }
        )
        
        return plan_graph
    
    def _create_git_operation_plan(self, intent: str, context: Dict[str, Any]) -> PlanGraph:
        """Create a plan for git operations"""
        # Use ShellGenie to generate the command
        command, explanation = self.shell_genie.generate_command(intent)
        
        steps = []
        
        # Check git status first
        git_status_step = PlanStep(
            id=str(uuid.uuid4()),
            type="command",
            description="Check current git status",
            command="git status --porcelain",
            dependencies=[],
            estimated_risk=0.05,
            estimated_duration=1.0,
            metadata={
                "purpose": "git_status_check",
                "intent": intent
            }
        )
        
        # Perform the requested git operation
        git_operation_step = PlanStep(
            id=str(uuid.uuid4()),
            type="command",
            description=explanation,
            command=command,
            dependencies=[git_status_step.id],
            estimated_risk=0.0,  # Will be calculated later
            estimated_duration=3.0,
            metadata={
                "intent": intent,
                "operation_type": "git"
            }
        )
        
        # Verify the operation succeeded
        verify_step = PlanStep(
            id=str(uuid.uuid4()),
            type="command",
            description="Verify git operation result",
            command="git status --porcelain",
            dependencies=[git_operation_step.id],
            estimated_risk=0.05,
            estimated_duration=1.0,
            metadata={
                "purpose": "verification",
                "original_intent": intent
            }
        )
        
        steps = [git_status_step, git_operation_step, verify_step]
        dependencies = [
            (git_status_step.id, git_operation_step.id),
            (git_operation_step.id, verify_step.id)
        ]
        
        plan_graph = PlanGraph(
            id=str(uuid.uuid4()),
            intent=intent,
            steps=steps,
            dependencies=dependencies,
            context=context,
            timestamp=datetime.now(),
            metadata={
                "plan_type": "git_operation",
                "generated_by": "ByteBot Planner",
                "context_included": True
            }
        )
        
        return plan_graph
    
    def _create_development_task_plan(self, intent: str, context: Dict[str, Any]) -> PlanGraph:
        """Create a plan for development tasks"""
        # Use ShellGenie to generate the command
        command, explanation = self.shell_genie.generate_command(intent)
        
        steps = []
        
        # Check if we're in a project directory
        project_check_step = PlanStep(
            id=str(uuid.uuid4()),
            type="command",
            description="Check project type and status",
            command=self._get_project_check_command(context),
            dependencies=[],
            estimated_risk=0.05,
            estimated_duration=1.0,
            metadata={
                "purpose": "project_check",
                "intent": intent
            }
        )
        
        # Perform the development task
        dev_task_step = PlanStep(
            id=str(uuid.uuid4()),
            type="command",
            description=explanation,
            command=command,
            dependencies=[project_check_step.id],
            estimated_risk=0.0,  # Will be calculated later
            estimated_duration=5.0,  # Development tasks typically take longer
            metadata={
                "intent": intent,
                "operation_type": "development"
            }
        )
        
        # Check results if this is a test or build operation
        if any(word in intent.lower() for word in ["test", "build", "compile"]):
            result_check_step = PlanStep(
                id=str(uuid.uuid4()),
                type="command",
                description="Check development task results",
                command=self._get_result_check_command(intent),
                dependencies=[dev_task_step.id],
                estimated_risk=0.05,
                estimated_duration=2.0,
                metadata={
                    "purpose": "result_check",
                    "original_intent": intent
                }
            )
            
            steps = [project_check_step, dev_task_step, result_check_step]
            dependencies = [
                (project_check_step.id, dev_task_step.id),
                (dev_task_step.id, result_check_step.id)
            ]
        else:
            steps = [project_check_step, dev_task_step]
            dependencies = [(project_check_step.id, dev_task_step.id)]
        
        plan_graph = PlanGraph(
            id=str(uuid.uuid4()),
            intent=intent,
            steps=steps,
            dependencies=dependencies,
            context=context,
            timestamp=datetime.now(),
            metadata={
                "plan_type": "development_task",
                "generated_by": "ByteBot Planner",
                "context_included": True
            }
        )
        
        return plan_graph
    
    def _get_project_check_command(self, context: Dict[str, Any]) -> str:
        """Get appropriate command to check project status based on context"""
        project_type = context.get("project_info", {}).get("type", "unknown")
        
        if project_type == "python":
            return "python --version && pip list --format=freeze | head -10"
        elif project_type == "nodejs":
            return "node --version && npm list --depth=0 | head -10"
        elif project_type == "maven":
            return "mvn --version && ls -la pom.xml"
        elif project_type == "gradle":
            return "gradle --version && ls -la build.gradle"
        else:
            return "pwd && ls -la"
    
    def _get_result_check_command(self, intent: str) -> str:
        """Get appropriate command to check results based on intent"""
        if "test" in intent.lower():
            # If running tests, check test reports or recent test output
            return "ls -la && find . -name '*test*' -type d | head -5"
        elif "build" in intent.lower() or "compile" in intent.lower():
            # If building, check for build artifacts
            return "ls -la dist/ build/ out/ 2>/dev/null || echo 'No standard build directories found'"
        else:
            return "echo 'Operation completed'"
    
    def validate_plan(self, plan: PlanGraph) -> Tuple[bool, List[str]]:
        """
        Validate a plan for correctness and safety
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for circular dependencies
        if self._has_circular_dependencies(plan):
            issues.append("Plan has circular dependencies")
        
        # Check that all dependency IDs exist
        step_ids = {step.id for step in plan.steps}
        for dep_from, dep_to in plan.dependencies:
            if dep_from not in step_ids or dep_to not in step_ids:
                issues.append(f"Dependency references non-existent step ID")
        
        # Check for dangerous commands
        for step in plan.steps:
            if step.type == "command":
                risk_assessment = self.risk_scorer.score_command(step.command, plan.context)
                if risk_assessment.is_dangerous:
                    issues.append(f"Step {step.id} contains dangerous command: {step.command}")
        
        return len(issues) == 0, issues
    
    def _has_circular_dependencies(self, plan: PlanGraph) -> bool:
        """Check if the plan has circular dependencies using DFS"""
        # Build adjacency list
        adj_list = {step.id: [] for step in plan.steps}
        for dep_from, dep_to in plan.dependencies:
            adj_list[dep_from].append(dep_to)
        
        # Colors: 0 = white (unvisited), 1 = gray (visiting), 2 = black (visited)
        color = {step.id: 0 for step in plan.steps}
        
        def dfs(node):
            color[node] = 1  # Mark as visiting
            for neighbor in adj_list[node]:
                if color[neighbor] == 1:  # Back edge found
                    return True
                if color[neighbor] == 0 and dfs(neighbor):
                    return True
            color[node] = 2  # Mark as visited
            return False
        
        for step in plan.steps:
            if color[step.id] == 0:
                if dfs(step.id):
                    return True
        return False
    
    def optimize_plan(self, plan: PlanGraph) -> PlanGraph:
        """
        Optimize a plan by reordering steps for efficiency
        """
        # For now, just return the original plan
        # In the future, we could reorder steps to minimize dependencies
        # or group similar operations together
        return plan
    
    def serialize_plan(self, plan: PlanGraph) -> str:
        """Serialize a plan to JSON string"""
        plan_dict = {
            "id": plan.id,
            "intent": plan.intent,
            "steps": [
                {
                    "id": step.id,
                    "type": step.type,
                    "description": step.description,
                    "command": step.command,
                    "dependencies": step.dependencies,
                    "estimated_risk": step.estimated_risk,
                    "estimated_duration": step.estimated_duration,
                    "metadata": step.metadata
                }
                for step in plan.steps
            ],
            "dependencies": plan.dependencies,
            "context": plan.context,
            "timestamp": plan.timestamp.isoformat(),
            "metadata": plan.metadata
        }
        return json.dumps(plan_dict, indent=2)
    
    def deserialize_plan(self, plan_json: str) -> PlanGraph:
        """Deserialize a plan from JSON string"""
        plan_dict = json.loads(plan_json)
        
        steps = [
            PlanStep(
                id=step_data["id"],
                type=step_data["type"],
                description=step_data["description"],
                command=step_data["command"],
                dependencies=step_data["dependencies"],
                estimated_risk=step_data["estimated_risk"],
                estimated_duration=step_data["estimated_duration"],
                metadata=step_data["metadata"]
            )
            for step_data in plan_dict["steps"]
        ]
        
        return PlanGraph(
            id=plan_dict["id"],
            intent=plan_dict["intent"],
            steps=steps,
            dependencies=plan_dict["dependencies"],
            context=plan_dict["context"],
            timestamp=datetime.fromisoformat(plan_dict["timestamp"]),
            metadata=plan_dict["metadata"]
        )
    
    def get_plan_summary(self, plan: PlanGraph) -> Dict[str, Any]:
        """Get a summary of the plan"""
        total_risk = sum(step.estimated_risk for step in plan.steps)
        avg_risk = total_risk / len(plan.steps) if plan.steps else 0
        
        return {
            "id": plan.id,
            "intent": plan.intent,
            "step_count": len(plan.steps),
            "total_estimated_duration": sum(step.estimated_duration for step in plan.steps),
            "average_risk": avg_risk,
            "max_risk": max((step.estimated_risk for step in plan.steps), default=0),
            "plan_type": plan.metadata.get("plan_type", "unknown"),
            "timestamp": plan.timestamp.isoformat()
        }