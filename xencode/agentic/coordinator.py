"""Multi-agent coordination system."""

from typing import Dict, List, Optional, Any
from enum import Enum
import time

from .manager import LangChainManager


class AgentType(Enum):
    """Types of specialized agents."""
    CODE = "code"
    RESEARCH = "research"
    EXECUTION = "execution"
    GENERAL = "general"


class Agent:
    """Individual agent with specialized capabilities."""
    
    def __init__(self, agent_type: AgentType, model_name: str, base_url: str = "http://localhost:11434"):
        self.agent_type = agent_type
        self.manager = LangChainManager(
            model_name=model_name,
            base_url=base_url,
            use_memory=True,
            db_path=f"agent_{agent_type.value}_memory.db"
        )
    
    def execute(self, task: str) -> Dict[str, Any]:
        """Execute a task and return result."""
        start_time = time.time()
        
        result = self.manager.run_agent(task)
        
        return {
            "agent_type": self.agent_type.value,
            "task": task,
            "result": result,
            "execution_time": time.time() - start_time
        }


class AgentCoordinator:
    """Coordinates multiple specialized agents."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.agents: Dict[AgentType, Agent] = {}
        self.base_url = base_url
        self._init_agents()
    
    def _init_agents(self):
        """Initialize specialized agents."""
        # Code agent - uses codellama
        self.agents[AgentType.CODE] = Agent(
            agent_type=AgentType.CODE,
            model_name="codellama:7b",
            base_url=self.base_url
        )
        
        # Research agent - uses mistral
        self.agents[AgentType.RESEARCH] = Agent(
            agent_type=AgentType.RESEARCH,
            model_name="mistral:7b",
            base_url=self.base_url
        )
        
        # Execution agent - uses qwen3 (fast and reliable)
        self.agents[AgentType.EXECUTION] = Agent(
            agent_type=AgentType.EXECUTION,
            model_name="qwen3:4b",
            base_url=self.base_url
        )
        
        # General agent - uses qwen3
        self.agents[AgentType.GENERAL] = Agent(
            agent_type=AgentType.GENERAL,
            model_name="qwen3:4b",
            base_url=self.base_url
        )
    
    def classify_task(self, task: str) -> AgentType:
        """Classify task to determine which agent should handle it."""
        task_lower = task.lower()
        
        # Code-related keywords
        code_keywords = ["code", "function", "class", "debug", "program", "script", 
                        "python", "javascript", "refactor", "algorithm"]
        if any(keyword in task_lower for keyword in code_keywords):
            return AgentType.CODE
        
        # Research keywords
        research_keywords = ["search", "find", "research", "look up", "information",
                            "web", "google", "learn about"]
        if any(keyword in task_lower for keyword in research_keywords):
            return AgentType.RESEARCH
        
        # Execution keywords
        execution_keywords = ["run", "execute", "command", "terminal", "shell",
                             "file", "create", "delete", "write"]
        if any(keyword in task_lower for keyword in execution_keywords):
            return AgentType.EXECUTION
        
        return AgentType.GENERAL
    
    def delegate_task(self, task: str, agent_type: Optional[AgentType] = None) -> Dict[str, Any]:
        """Delegate a task to the appropriate agent."""
        if agent_type is None:
            agent_type = self.classify_task(task)
        
        agent = self.agents.get(agent_type)
        if not agent:
            # Fallback to general agent
            agent = self.agents[AgentType.GENERAL]
        
        result = agent.execute(task)
        result["selected_agent"] = agent_type.value
        
        return result
    
    def multi_agent_task(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tasks with different agents."""
        results = []
        
        for task_spec in tasks:
            task = task_spec.get("task", "")
            agent_type_str = task_spec.get("agent_type")
            
            agent_type = None
            if agent_type_str:
                try:
                    agent_type = AgentType(agent_type_str)
                except ValueError:
                    pass
            
            result = self.delegate_task(task, agent_type)
            results.append(result)
        
        return results
    
    def collaborative_task(self, task: str, subtasks: List[str]) -> Dict[str, Any]:
        """Break down a complex task into subtasks and coordinate agents."""
        print(f"Main task: {task}")
        print(f"Breaking into {len(subtasks)} subtasks...")
        
        subtask_results = []
        
        for i, subtask in enumerate(subtasks, 1):
            print(f"\nSubtask {i}/{len(subtasks)}: {subtask}")
            result = self.delegate_task(subtask)
            subtask_results.append(result)
            print(f"Completed by {result['selected_agent']} agent")
        
        # Aggregate results
        final_result = {
            "main_task": task,
            "subtasks": subtasks,
            "subtask_results": subtask_results,
            "summary": self._generate_summary(subtask_results)
        }
        
        return final_result
    
    def _generate_summary(self, subtask_results: List[Dict[str, Any]]) -> str:
        """Generate a summary of subtask results."""
        summary_parts = []
        
        for i, result in enumerate(subtask_results, 1):
            summary_parts.append(
                f"Subtask {i} ({result['selected_agent']} agent): {result['result'][:100]}..."
            )
        
        return "\n".join(summary_parts)
