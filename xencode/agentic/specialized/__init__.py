"""
Specialized agents for specific domains in Xencode
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from ...core.models import ModelManager
from ..manager import LangChainManager
from ..tools import ReadFileTool, WriteFileTool, ExecuteCommandTool
from ..advanced_tools import (
    GitStatusTool, GitDiffTool, GitLogTool, GitCommitTool,
    WebSearchTool, CodeAnalysisTool
)
from ..enhanced_tools import (
    GitBranchTool, GitPushTool, GitPullTool,
    FindFileTool, FileStatTool, DependencyAnalysisTool,
    SystemInfoTool, ProcessInfoTool, WebSearchDetailedTool
)


class SpecializedAgentType(Enum):
    """Types of specialized agents."""
    DATA_SCIENCE = "data_science"
    WEB_DEVELOPMENT = "web_development"
    SECURITY_ANALYSIS = "security_analysis"
    DEVOPS = "devops"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


@dataclass
class AgentProfile:
    """Profile for a specialized agent."""
    name: str
    description: str
    primary_model: str
    fallback_models: List[str]
    specialized_tools: List[str]
    expertise_domains: List[str]
    personality_traits: List[str]


class SpecializedAgent:
    """Base class for specialized agents."""
    
    def __init__(self, agent_type: SpecializedAgentType, profile: AgentProfile):
        self.agent_type = agent_type
        self.profile = profile
        self.model_manager = ModelManager()
        self.langchain_manager = LangChainManager(
            model_name=profile.primary_model,
            use_memory=True,
            db_path=f"agent_{agent_type.value}_memory.db"
        )
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize tools specific to this agent type."""
        # Base tools that all agents have
        base_tools = [
            ReadFileTool(),
            WriteFileTool(),
            ExecuteCommandTool()
        ]
        
        # Add specialized tools based on agent type
        specialized_tools = self._get_specialized_tools()
        return base_tools + specialized_tools
    
    def _get_specialized_tools(self):
        """Get specialized tools for this agent type."""
        # This will be overridden by subclasses
        return []
    
    async def process_request(self, request: str) -> str:
        """Process a request using the specialized agent."""
        # This will be implemented by subclasses
        raise NotImplementedError
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get the agent's capabilities."""
        return {
            "type": self.agent_type.value,
            "name": self.profile.name,
            "description": self.profile.description,
            "primary_model": self.profile.primary_model,
            "specialized_tools": [tool.name for tool in self.tools],
            "expertise_domains": self.profile.expertise_domains
        }


class DataScienceAgent(SpecializedAgent):
    """Specialized agent for data science tasks."""
    
    def __init__(self):
        profile = AgentProfile(
            name="Data Science Assistant",
            description="Expert in data analysis, visualization, and ML model development",
            primary_model="codellama:7b",  # Good for code generation
            fallback_models=["llama3.1:8b", "mistral:7b"],
            specialized_tools=[
                "pandas_operations", "matplotlib_visualization", 
                "sklearn_modeling", "jupyter_notebook"
            ],
            expertise_domains=["data analysis", "machine learning", "statistics", "visualization"],
            personality_traits=["analytical", "detail-oriented", "methodical"]
        )
        super().__init__(SpecializedAgentType.DATA_SCIENCE, profile)
    
    def _get_specialized_tools(self):
        """Get tools specific to data science."""
        # In a real implementation, these would be actual LangChain tools
        # For now, we'll use enhanced tools that are relevant
        return [
            FindFileTool(),  # Find data files
            FileStatTool(),  # Get file info for datasets
            CodeAnalysisTool(),  # Analyze data processing code
            WebSearchDetailedTool()  # Search for data science techniques
        ]
    
    async def process_request(self, request: str) -> str:
        """Process a data science request."""
        # Enhanced processing for data science tasks
        enhanced_prompt = f"""
        As a data science expert, please help with this request: {request}
        
        Consider the following in your response:
        - Statistical validity of approaches
        - Appropriate visualization techniques
        - Data preprocessing requirements
        - Model selection criteria
        - Evaluation metrics
        """
        
        return self.langchain_manager.run_agent(enhanced_prompt)


class WebDevelopmentAgent(SpecializedAgent):
    """Specialized agent for web development tasks."""
    
    def __init__(self):
        profile = AgentProfile(
            name="Web Development Assistant",
            description="Expert in frontend, backend, and full-stack web development",
            primary_model="codellama:7b",  # Good for code generation
            fallback_models=["llama3.1:8b", "mistral:7b"],
            specialized_tools=[
                "html_css_js", "react_vue_angular", "nodejs_express",
                "database_design", "api_development", "deployment"
            ],
            expertise_domains=["frontend", "backend", "full-stack", "devops"],
            personality_traits=["practical", "up-to-date", "performance-focused"]
        )
        super().__init__(SpecializedAgentType.WEB_DEVELOPMENT, profile)
    
    def _get_specialized_tools(self):
        """Get tools specific to web development."""
        return [
            FindFileTool(),  # Find web files
            FileStatTool(),  # Check file permissions for web servers
            CodeAnalysisTool(),  # Analyze web code
            DependencyAnalysisTool(),  # Analyze package dependencies
            GitStatusTool(),  # Check git status for deployments
            GitDiffTool(),  # Review changes before deployment
            WebSearchDetailedTool()  # Search for web technologies
        ]
    
    async def process_request(self, request: str) -> str:
        """Process a web development request."""
        enhanced_prompt = f"""
        As a web development expert, please help with this request: {request}
        
        Consider the following in your response:
        - Modern web development practices
        - Cross-browser compatibility
        - Performance optimization
        - Security best practices
        - Responsive design principles
        """
        
        return self.langchain_manager.run_agent(enhanced_prompt)


class SecurityAnalysisAgent(SpecializedAgent):
    """Specialized agent for security analysis tasks."""
    
    def __init__(self):
        profile = AgentProfile(
            name="Security Analysis Assistant",
            description="Expert in security analysis, vulnerability assessment, and secure coding",
            primary_model="llama3.1:8b",  # Good for analytical tasks
            fallback_models=["mistral:7b", "codellama:7b"],
            specialized_tools=[
                "vulnerability_scanning", "penetration_testing", 
                "secure_coding", "compliance_checking"
            ],
            expertise_domains=["vulnerability assessment", "secure coding", "compliance", "threat modeling"],
            personality_traits=["cautious", "thorough", "proactive"]
        )
        super().__init__(SpecializedAgentType.SECURITY_ANALYSIS, profile)
    
    def _get_specialized_tools(self):
        """Get tools specific to security analysis."""
        return [
            CodeAnalysisTool(),  # Analyze code for security issues
            DependencyAnalysisTool(),  # Check for vulnerable dependencies
            FindFileTool(),  # Find configuration files
            FileStatTool(),  # Check file permissions
            SystemInfoTool(),  # Get system information for security assessment
            ProcessInfoTool(),  # Check running processes
            WebSearchDetailedTool()  # Search for security advisories
        ]
    
    async def process_request(self, request: str) -> str:
        """Process a security analysis request."""
        enhanced_prompt = f"""
        As a security analysis expert, please help with this request: {request}
        
        Consider the following in your response:
        - Potential security vulnerabilities
        - Best practices for mitigation
        - Compliance requirements
        - Attack vectors to consider
        - Security testing recommendations
        """
        
        return self.langchain_manager.run_agent(enhanced_prompt)


class DevOpsAgent(SpecializedAgent):
    """Specialized agent for DevOps tasks."""
    
    def __init__(self):
        profile = AgentProfile(
            name="DevOps Assistant",
            description="Expert in CI/CD, infrastructure, and deployment automation",
            primary_model="llama3.1:8b",  # Good for procedural tasks
            fallback_models=["mistral:7b", "codellama:7b"],
            specialized_tools=[
                "ci_cd_pipeline", "containerization", "cloud_deployment",
                "monitoring_alerting", "infrastructure_as_code"
            ],
            expertise_domains=["CI/CD", "containers", "cloud", "monitoring", "automation"],
            personality_traits=["automated", "reliable", "scalable"]
        )
        super().__init__(SpecializedAgentType.DEVOPS, profile)
    
    def _get_specialized_tools(self):
        """Get tools specific to DevOps."""
        return [
            ExecuteCommandTool(),  # Execute DevOps commands
            FindFileTool(),  # Find configuration files
            FileStatTool(),  # Check file permissions
            GitStatusTool(),  # Check git status
            GitDiffTool(),  # Review changes
            GitBranchTool(),  # Manage branches
            GitPushTool(),  # Push changes
            GitPullTool(),  # Pull changes
            SystemInfoTool(),  # Get system info for deployment
            ProcessInfoTool(),  # Check running services
            WebSearchDetailedTool()  # Search for DevOps practices
        ]
    
    async def process_request(self, request: str) -> str:
        """Process a DevOps request."""
        enhanced_prompt = f"""
        As a DevOps expert, please help with this request: {request}
        
        Consider the following in your response:
        - Infrastructure as code
        - CI/CD pipeline optimization
        - Containerization strategies
        - Cloud deployment options
        - Monitoring and alerting
        """
        
        return self.langchain_manager.run_agent(enhanced_prompt)


class TestingAgent(SpecializedAgent):
    """Specialized agent for testing tasks."""
    
    def __init__(self):
        profile = AgentProfile(
            name="Testing Assistant",
            description="Expert in test automation, QA, and quality assurance",
            primary_model="codellama:7b",  # Good for code generation
            fallback_models=["llama3.1:8b", "mistral:7b"],
            specialized_tools=[
                "test_automation", "unit_testing", "integration_testing",
                "performance_testing", "test_coverage"
            ],
            expertise_domains=["unit testing", "integration testing", "QA", "test automation"],
            personality_traits=["thorough", "quality-focused", "systematic"]
        )
        super().__init__(SpecializedAgentType.TESTING, profile)
    
    def _get_specialized_tools(self):
        """Get tools specific to testing."""
        return [
            CodeAnalysisTool(),  # Analyze code for testability
            FindFileTool(),  # Find test files
            FileStatTool(),  # Check test file properties
            DependencyAnalysisTool(),  # Analyze test dependencies
            WebSearchDetailedTool()  # Search for testing strategies
        ]
    
    async def process_request(self, request: str) -> str:
        """Process a testing request."""
        enhanced_prompt = f"""
        As a testing expert, please help with this request: {request}
        
        Consider the following in your response:
        - Test strategy recommendations
        - Test coverage optimization
        - Different testing types (unit, integration, e2e)
        - Test data management
        - Continuous testing practices
        """
        
        return self.langchain_manager.run_agent(enhanced_prompt)


class DocumentationAgent(SpecializedAgent):
    """Specialized agent for documentation tasks."""
    
    def __init__(self):
        profile = AgentProfile(
            name="Documentation Assistant",
            description="Expert in technical writing, API documentation, and knowledge management",
            primary_model="llama3.1:8b",  # Good for writing tasks
            fallback_models=["mistral:7b", "codellama:7b"],
            specialized_tools=[
                "tech_writing", "api_documentation", "knowledge_management",
                "style_guidelines", "content_review"
            ],
            expertise_domains=["technical writing", "API docs", "knowledge management", "content strategy"],
            personality_traits=["clear", "concise", "organized"]
        )
        super().__init__(SpecializedAgentType.DOCUMENTATION, profile)
    
    def _get_specialized_tools(self):
        """Get tools specific to documentation."""
        return [
            ReadFileTool(),  # Read existing docs
            WriteFileTool(),  # Write new docs
            CodeAnalysisTool(),  # Extract code documentation
            FindFileTool(),  # Find documentation files
            FileStatTool(),  # Check doc file properties
            WebSearchDetailedTool()  # Search for documentation best practices
        ]
    
    async def process_request(self, request: str) -> str:
        """Process a documentation request."""
        enhanced_prompt = f"""
        As a documentation expert, please help with this request: {request}
        
        Consider the following in your response:
        - Audience-appropriate language
        - Clear structure and organization
        - Examples and use cases
        - Accessibility and usability
        - Maintenance and versioning
        """
        
        return self.langchain_manager.run_agent(enhanced_prompt)


class SpecializedAgentFactory:
    """Factory for creating specialized agents."""

    _agent_registry: Dict[SpecializedAgentType, type] = {
        SpecializedAgentType.DATA_SCIENCE: DataScienceAgent,
        SpecializedAgentType.WEB_DEVELOPMENT: WebDevelopmentAgent,
        SpecializedAgentType.SECURITY_ANALYSIS: SecurityAnalysisAgent,
        SpecializedAgentType.DEVOPS: DevOpsAgent,
        SpecializedAgentType.TESTING: TestingAgent,
        SpecializedAgentType.DOCUMENTATION: DocumentationAgent,
    }

    @classmethod
    def create_agent(cls, agent_type: SpecializedAgentType) -> SpecializedAgent:
        """Create a specialized agent of the given type."""
        if agent_type not in cls._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_class = cls._agent_registry[agent_type]
        return agent_class()

    @classmethod
    def get_available_agent_types(cls) -> List[SpecializedAgentType]:
        """Get list of available agent types."""
        return list(cls._agent_registry.keys())

    @classmethod
    def get_agent_capabilities(cls, agent_type: SpecializedAgentType) -> Dict[str, Any]:
        """Get capabilities of a specific agent type."""
        agent = cls.create_agent(agent_type)
        return agent.get_capabilities()


# Export all classes
__all__ = [
    'SpecializedAgentType',
    'AgentProfile',
    'SpecializedAgent',
    'DataScienceAgent',
    'WebDevelopmentAgent',
    'SecurityAnalysisAgent',
    'DevOpsAgent',
    'TestingAgent',
    'DocumentationAgent',
    'SpecializedAgentFactory'
]