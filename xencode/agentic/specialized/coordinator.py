"""
Coordinator for specialized agents in Xencode
"""
from typing import Dict, List, Optional, Any
from enum import Enum
from . import (
    SpecializedAgentFactory, SpecializedAgentType, SpecializedAgent,
    DataScienceAgent, WebDevelopmentAgent, SecurityAnalysisAgent,
    DevOpsAgent, TestingAgent, DocumentationAgent
)


class SpecializedAgentCoordinator:
    """Coordinates specialized agents for domain-specific tasks."""
    
    def __init__(self):
        self.agents: Dict[SpecializedAgentType, SpecializedAgent] = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all specialized agents."""
        factory = SpecializedAgentFactory()
        
        for agent_type in SpecializedAgentFactory.get_available_agent_types():
            try:
                agent = factory.create_agent(agent_type)
                self.agents[agent_type] = agent
            except Exception as e:
                print(f"Warning: Could not initialize {agent_type.value} agent: {e}")
    
    def get_agent(self, agent_type: SpecializedAgentType) -> Optional[SpecializedAgent]:
        """Get a specialized agent by type."""
        return self.agents.get(agent_type)
    
    def classify_task(self, task: str) -> Optional[SpecializedAgentType]:
        """Classify a task to determine the most appropriate specialized agent."""
        task_lower = task.lower()
        
        # Data science keywords
        data_science_keywords = [
            "data", "analysis", "dataset", "pandas", "numpy", "matplotlib", 
            "plot", "visualization", "ml", "machine learning", "model", 
            "prediction", "statistical", "regression", "classification",
            "clustering", "analytics", "dashboard", "statistics"
        ]
        if any(keyword in task_lower for keyword in data_science_keywords):
            return SpecializedAgentType.DATA_SCIENCE
        
        # Web development keywords
        web_dev_keywords = [
            "html", "css", "javascript", "react", "vue", "angular", 
            "frontend", "backend", "api", "rest", "graphql", "express",
            "node", "database", "sql", "mongodb", "postgresql", "website",
            "web app", "spa", "responsive", "framework", "library"
        ]
        if any(keyword in task_lower for keyword in web_dev_keywords):
            return SpecializedAgentType.WEB_DEVELOPMENT
        
        # Security analysis keywords
        security_keywords = [
            "security", "vulnerability", "exploit", "attack", "defense",
            "penetration", "pentest", "scan", "secure", "authentication",
            "authorization", "encryption", "ssl", "tls", "firewall",
            "malware", "threat", "risk", "compliance", "owasp"
        ]
        if any(keyword in task_lower for keyword in security_keywords):
            return SpecializedAgentType.SECURITY_ANALYSIS
        
        # DevOps keywords
        devops_keywords = [
            "ci/cd", "pipeline", "docker", "kubernetes", "container",
            "deployment", "infrastructure", "terraform", "aws", "azure",
            "gcp", "cloud", "monitoring", "logging", "alerting", "scaling",
            "automation", "ansible", "jenkins", "gitlab", "github actions"
        ]
        if any(keyword in task_lower for keyword in devops_keywords):
            return SpecializedAgentType.DEVOPS
        
        # Testing keywords
        testing_keywords = [
            "test", "testing", "unit test", "integration test", "e2e",
            "automated test", "qa", "quality", "coverage", "tdd", "bdd",
            "mock", "stub", "assertion", "verification", "validation",
            "bug", "issue", "defect", "selenium", "pytest", "unittest"
        ]
        if any(keyword in task_lower for keyword in testing_keywords):
            return SpecializedAgentType.TESTING
        
        # Documentation keywords
        documentation_keywords = [
            "documentation", "doc", "readme", "tutorial", "guide", 
            "manual", "api doc", "reference", "explanation", 
            "instruction", "howto", "walkthrough", "example"
        ]
        if any(keyword in task_lower for keyword in documentation_keywords):
            return SpecializedAgentType.DOCUMENTATION
        
        # If no specific match, return None to indicate general processing
        return None
    
    async def process_task(self, task: str) -> str:
        """Process a task using the most appropriate specialized agent."""
        agent_type = self.classify_task(task)
        
        if agent_type and agent_type in self.agents:
            agent = self.agents[agent_type]
            print(f"Routing task to {agent_type.value} agent...")
            return await agent.process_request(task)
        else:
            # If no specialized agent is appropriate, return a message
            # In a real implementation, this might route to a general agent
            return f"No specialized agent found for task: {task}. Consider using a general agent or specifying a domain."
    
    def get_all_agent_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of all specialized agents."""
        capabilities = {}
        for agent_type, agent in self.agents.items():
            capabilities[agent_type.value] = agent.get_capabilities()
        return capabilities
    
    def add_custom_agent(self, agent_type: SpecializedAgentType, agent: SpecializedAgent):
        """Add a custom specialized agent."""
        self.agents[agent_type] = agent
    
    async def collaborative_task(self, task: str, agent_types: List[SpecializedAgentType]) -> Dict[str, Any]:
        """Process a task collaboratively using multiple specialized agents."""
        results = {}
        
        for agent_type in agent_types:
            if agent_type in self.agents:
                agent = self.agents[agent_type]
                print(f"Processing with {agent_type.value} agent...")
                result = await agent.process_request(task)
                results[agent_type.value] = result
            else:
                results[agent_type.value] = f"Agent {agent_type.value} not available"
        
        # Synthesize results
        synthesized_result = self._synthesize_results(task, results)
        
        return {
            "original_task": task,
            "agent_results": results,
            "synthesized_result": synthesized_result
        }
    
    def _synthesize_results(self, original_task: str, results: Dict[str, str]) -> str:
        """Synthesize results from multiple specialized agents."""
        synthesis_parts = [f"SYNTHESIZED RESPONSE FOR: {original_task}", ""]
        
        for agent_type, result in results.items():
            synthesis_parts.append(f"=== {agent_type.upper()} PERSPECTIVE ===")
            synthesis_parts.append(result)
            synthesis_parts.append("")  # Empty line for separation
        
        synthesis_parts.append("=== SYNTHESIS ===")
        synthesis_parts.append("Based on the perspectives above, here is a comprehensive solution...")
        
        return "\n".join(synthesis_parts)