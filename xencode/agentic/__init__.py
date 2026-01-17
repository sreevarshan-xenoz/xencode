"""
Agentic capabilities for Xencode using LangChain.
"""
from .manager import LangChainManager
from .tools import ReadFileTool, WriteFileTool, ExecuteCommandTool
from .advanced_tools import (
    GitStatusTool,
    GitDiffTool,
    GitLogTool,
    GitCommitTool,
    WebSearchTool,
    CodeAnalysisTool,
    ToolRegistry
)
from .enhanced_tools import (
    GitBranchTool,
    GitPushTool,
    GitPullTool,
    FindFileTool,
    FileStatTool,
    DependencyAnalysisTool,
    SystemInfoTool,
    ProcessInfoTool,
    WebSearchDetailedTool,
    EnhancedToolRegistry
)
from .coordinator import AgentCoordinator, AgentType
from .ensemble_integration import EnsembleChain, ModelCouncil, create_ensemble_chain, create_model_council

__all__ = [
    "LangChainManager",
    "ReadFileTool",
    "WriteFileTool",
    "ExecuteCommandTool",
    "GitStatusTool",
    "GitDiffTool",
    "GitLogTool",
    "GitCommitTool",
    "WebSearchTool",
    "CodeAnalysisTool",
    "ToolRegistry",
    "GitBranchTool",
    "GitPushTool",
    "GitPullTool",
    "FindFileTool",
    "FileStatTool",
    "DependencyAnalysisTool",
    "SystemInfoTool",
    "ProcessInfoTool",
    "WebSearchDetailedTool",
    "EnhancedToolRegistry",
    "AgentCoordinator",
    "AgentType",
    "EnsembleChain",
    "ModelCouncil",
    "create_ensemble_chain",
    "create_model_council"
]
