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
    PLANNING = "planning"


class Agent:
    """Individual agent with specialized capabilities."""
    
    def __init__(self, agent_type: AgentType, model_name: str, base_url: str = "http://localhost:11434", use_rag: bool = False):
        self.agent_type = agent_type
        self.manager = LangChainManager(
            model_name=model_name,
            base_url=base_url,
            use_memory=True,
            db_path=f"agent_{agent_type.value}_memory.db",
            use_rag=use_rag
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

    def __init__(self, base_url: str = "http://localhost:11434", enable_dynamic_routing: bool = True):
        self.agents: Dict[AgentType, Agent] = {}
        self.base_url = base_url
        self.enable_dynamic_routing = enable_dynamic_routing
        self.task_history: List[Dict[str, Any]] = []
        self._init_agents()

    def _init_agents(self):
        """Initialize specialized agents."""
        # Code agent - uses codellama (with RAG!)
        self.agents[AgentType.CODE] = Agent(
            agent_type=AgentType.CODE,
            model_name="codellama:7b",
            base_url=self.base_url,
            use_rag=True
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

        # Planning agent - uses a reasoning-focused model
        self.agents[AgentType.PLANNING] = Agent(
            agent_type=AgentType.PLANNING,
            model_name="llama3.1:8b",
            base_url=self.base_url
        )

    def add_agent(self, agent_type: AgentType, model_name: str, use_rag: bool = False):
        """Dynamically add a new agent to the coordinator."""
        self.agents[agent_type] = Agent(
            agent_type=agent_type,
            model_name=model_name,
            base_url=self.base_url,
            use_rag=use_rag
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

    def sequential_collaboration(self, task: str, agent_sequence: List[AgentType]) -> Dict[str, Any]:
        """Execute a task through a sequence of agents, passing results between them."""
        print(f"Sequential task: {task}")
        print(f"Agent sequence: {[agent.value for agent in agent_sequence]}")

        current_result = task
        intermediate_results = []

        for i, agent_type in enumerate(agent_sequence):
            agent = self.agents.get(agent_type)
            if not agent:
                print(f"Warning: Agent {agent_type} not found, skipping")
                continue

            print(f"\nStep {i+1}/{len(agent_sequence)}: {agent_type.value} agent")

            # Create task for this agent based on previous result
            step_task = current_result if i == 0 else f"Previous result: {current_result}\n\nTask: {task}"

            result = agent.execute(step_task)
            current_result = result['result']
            intermediate_results.append({
                'step': i+1,
                'agent': agent_type.value,
                'input': step_task[:200] + "..." if len(step_task) > 200 else step_task,
                'output': current_result[:200] + "..." if len(current_result) > 200 else current_result,
                'execution_time': result['execution_time']
            })
            print(f"Completed by {result['agent_type']} agent")

        final_result = {
            "original_task": task,
            "agent_sequence": [agent.value for agent in agent_sequence],
            "intermediate_results": intermediate_results,
            "final_result": current_result,
            "total_steps": len(intermediate_results)
        }

        return final_result

    def parallel_collaboration(self, task: str, participating_agents: List[AgentType]) -> Dict[str, Any]:
        """Execute a task in parallel across multiple agents and synthesize results."""
        print(f"Parallel task: {task}")
        print(f"Participating agents: {[agent.value for agent in participating_agents]}")

        # Create tasks for each agent (potentially customized per agent type)
        agent_tasks = []
        for agent_type in participating_agents:
            agent_task = {
                "task": task,
                "agent_type": agent_type
            }
            agent_tasks.append(agent_task)

        # Execute in parallel using multi_agent_task
        results = self.multi_agent_task(agent_tasks)

        # Synthesize results
        synthesized_result = self._synthesize_parallel_results(task, results)

        final_result = {
            "original_task": task,
            "participating_agents": [agent.value for agent in participating_agents],
            "individual_results": results,
            "synthesized_result": synthesized_result,
            "total_agents": len(participating_agents)
        }

        return final_result

    def _synthesize_parallel_results(self, original_task: str, results: List[Dict[str, Any]]) -> str:
        """Synthesize results from parallel agent execution."""
        # For now, we'll use a simple approach - could be enhanced with more sophisticated synthesis

        # Join all results with agent labels
        synthesis_parts = [f"Original Task: {original_task}", "Results from participating agents:"]
        for i, result in enumerate(results):
            agent_type = result.get('selected_agent', 'unknown')
            result_text = result.get('result', 'No result returned')
            synthesis_parts.append(f"\n{agent_type.upper()} AGENT:\n{result_text}")

        return "\n".join(synthesis_parts)

    def adaptive_collaboration(self, task: str) -> Dict[str, Any]:
        """Determine the optimal collaboration strategy based on task characteristics."""
        # Analyze the task to determine the best approach
        task_complexity = self._estimate_task_complexity(task)
        task_domain = self._identify_task_domain(task)

        print(f"Task analysis - Complexity: {task_complexity}, Domain: {task_domain}")

        if task_complexity == "high" and task_domain in ["code", "technical"]:
            # Use sequential approach: Planning -> Code -> General for review
            agent_sequence = [AgentType.PLANNING, AgentType.CODE, AgentType.GENERAL]
            return self.sequential_collaboration(task, agent_sequence)
        elif task_complexity == "medium" and len(self.agents) > 2:
            # Use parallel approach with relevant agents
            relevant_agents = self._get_relevant_agents(task)
            return self.parallel_collaboration(task, relevant_agents)
        else:
            # Use standard delegation
            result = self.delegate_task(task)
            return {
                "original_task": task,
                "approach": "standard_delegation",
                "result": result
            }

    def _estimate_task_complexity(self, task: str) -> str:
        """Estimate task complexity based on keywords and length."""
        words = task.split()
        length = len(words)

        # Keywords that suggest complexity
        complex_keywords = ["implement", "design", "architecture", "debug", "optimize", "refactor",
                          "integrate", "configure", "deploy", "security", "performance", "algorithm"]

        complex_count = sum(1 for word in words if word.lower() in complex_keywords)

        if length > 50 or complex_count >= 3:
            return "high"
        elif length > 20 or complex_count >= 1:
            return "medium"
        else:
            return "low"

    def _identify_task_domain(self, task: str) -> str:
        """Identify the domain of the task."""
        task_lower = task.lower()

        if any(word in task_lower for word in ["code", "function", "class", "debug", "program", "script",
                                             "python", "javascript", "java", "c++", "algorithm", "data structure"]):
            return "code"
        elif any(word in task_lower for word in ["research", "find", "search", "information", "study",
                                               "learn", "about", "history", "science", "facts"]):
            return "research"
        elif any(word in task_lower for word in ["run", "execute", "command", "terminal", "shell",
                                               "file", "create", "delete", "write", "read"]):
            return "execution"
        elif any(word in task_lower for word in ["plan", "strategy", "approach", "method", "way",
                                               "how", "should", "could"]):
            return "planning"
        else:
            return "general"

    def _get_relevant_agents(self, task: str) -> List[AgentType]:
        """Get agents most relevant to the task."""
        domain = self._identify_task_domain(task)

        # Define agent relevance by domain
        domain_agents = {
            "code": [AgentType.CODE, AgentType.RESEARCH, AgentType.GENERAL],
            "research": [AgentType.RESEARCH, AgentType.GENERAL],
            "execution": [AgentType.EXECUTION, AgentType.GENERAL],
            "planning": [AgentType.PLANNING, AgentType.GENERAL],
            "general": [AgentType.GENERAL, AgentType.RESEARCH]
        }

        return domain_agents.get(domain, [AgentType.GENERAL])

    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get statistics about collaboration activities."""
        total_tasks = len(self.task_history)

        agent_usage = {}
        for task_record in self.task_history:
            agent_type = task_record.get('selected_agent', 'unknown')
            agent_usage[agent_type] = agent_usage.get(agent_type, 0) + 1

        return {
            "total_collaborations": total_tasks,
            "agent_utilization": agent_usage,
            "available_agents": [agent.value for agent in self.agents.keys()],
            "average_execution_time": sum(
                task_record.get('execution_time', 0) for task_record in self.task_history
            ) / total_tasks if total_tasks > 0 else 0
        }
