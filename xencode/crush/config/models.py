"""Configuration data models for Crush integration."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ModelType(str, Enum):
    """Model type enumeration."""
    LARGE = "large"
    SMALL = "small"


@dataclass
class SelectedModel:
    """Selected model configuration."""
    provider: str
    model: str
    
    def __post_init__(self):
        """Validate model configuration."""
        if not self.provider:
            raise ValueError("Provider is required for model configuration")
        if not self.model:
            raise ValueError("Model name is required for model configuration")


@dataclass
class ProviderConfig:
    """AI provider configuration."""
    api_key: str
    base_url: Optional[str] = None
    models: List[str] = field(default_factory=list)
    price_per_1m_input: float = 0.0
    price_per_1m_output: float = 0.0
    price_per_1m_cached_input: float = 0.0
    price_per_1m_cached_output: float = 0.0
    timeout: int = 60
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate provider configuration."""
        if not self.api_key:
            raise ValueError("API key is required for provider configuration")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")


@dataclass
class LSPConfig:
    """Language Server Protocol configuration."""
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    file_patterns: List[str] = field(default_factory=list)
    root_markers: List[str] = field(default_factory=list)
    initialization_options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate LSP configuration."""
        if not self.command:
            raise ValueError("Command is required for LSP configuration")


@dataclass
class MCPConfig:
    """Model Context Protocol configuration."""
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    disabled: bool = False
    auto_approve: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate MCP configuration."""
        if not self.command:
            raise ValueError("Command is required for MCP configuration")


@dataclass
class Options:
    """General options configuration."""
    auto_summarize: bool = True
    max_context_tokens: int = 100000
    summary_model: Optional[str] = None
    working_directory: Optional[str] = None
    data_directory: Optional[str] = None
    log_level: str = "INFO"
    command_timeout: int = 60
    background_job_timeout: int = 300
    permission_timeout: int = 300
    
    def __post_init__(self):
        """Validate options configuration."""
        if self.max_context_tokens <= 0:
            raise ValueError("Max context tokens must be positive")
        if self.command_timeout <= 0:
            raise ValueError("Command timeout must be positive")
        if self.background_job_timeout <= 0:
            raise ValueError("Background job timeout must be positive")
        if self.permission_timeout <= 0:
            raise ValueError("Permission timeout must be positive")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log level: {self.log_level}")


@dataclass
class Permissions:
    """Permission configuration."""
    skip_requests: bool = False
    allowed_tools: List[str] = field(default_factory=list)
    allowed_directories: List[str] = field(default_factory=list)
    blocked_commands: List[str] = field(default_factory=lambda: [
        "sudo", "rm -rf", "dd", "mkfs", "fdisk", "format",
        "del /f", "rmdir /s", "> /dev/sda"
    ])
    
    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is in the allowed list."""
        return tool_name in self.allowed_tools
    
    def is_command_blocked(self, command: str) -> bool:
        """Check if a command contains blocked patterns."""
        command_lower = command.lower()
        return any(blocked in command_lower for blocked in self.blocked_commands)


@dataclass
class Agent:
    """Agent configuration."""
    name: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    model: Optional[str] = None
    
    def __post_init__(self):
        """Validate agent configuration."""
        if not self.name:
            raise ValueError("Agent name is required")
        if not self.system_prompt:
            raise ValueError("System prompt is required for agent")


@dataclass
class Config:
    """Main configuration class."""
    models: Dict[str, SelectedModel] = field(default_factory=dict)
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    lsp: Dict[str, LSPConfig] = field(default_factory=dict)
    mcp: Dict[str, MCPConfig] = field(default_factory=dict)
    options: Options = field(default_factory=Options)
    permissions: Permissions = field(default_factory=Permissions)
    agents: Dict[str, Agent] = field(default_factory=dict)
    _working_dir: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate configuration."""
        # Ensure we have at least large and small models configured
        if "large" not in self.models and self.providers:
            raise ValueError("Large model must be configured")
        if "small" not in self.models and self.providers:
            raise ValueError("Small model must be configured")
        
        # Validate that model providers exist
        for model_type, selected_model in self.models.items():
            if selected_model.provider not in self.providers:
                raise ValueError(
                    f"Provider '{selected_model.provider}' for {model_type} model "
                    f"not found in providers configuration"
                )
    
    def working_dir(self) -> str:
        """Get working directory."""
        if self._working_dir:
            return self._working_dir
        if self.options.working_directory:
            return self.options.working_directory
        import os
        return os.getcwd()
    
    def is_configured(self) -> bool:
        """Check if at least one provider is configured."""
        return len(self.providers) > 0 and len(self.models) >= 2
    
    def get_large_model(self) -> Optional[SelectedModel]:
        """Get large model configuration."""
        return self.models.get("large")
    
    def get_small_model(self) -> Optional[SelectedModel]:
        """Get small model configuration."""
        return self.models.get("small")
    
    def get_provider(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get provider configuration by name."""
        return self.providers.get(provider_name)
    
    def get_lsp_client(self, language: str) -> Optional[LSPConfig]:
        """Get LSP client configuration for a language."""
        return self.lsp.get(language)
    
    def get_mcp_server(self, server_name: str) -> Optional[MCPConfig]:
        """Get MCP server configuration by name."""
        return self.mcp.get(server_name)
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Get agent configuration by name."""
        return self.agents.get(agent_name)
    
    def resolve(self, value: str) -> str:
        """Resolve environment variables in config values.
        
        This is a convenience method that creates a resolver and resolves the value.
        
        Args:
            value: Value to resolve
            
        Returns:
            Resolved value
        """
        from xencode.crush.config.resolver import EnvironmentResolver
        resolver = EnvironmentResolver()
        return resolver.resolve(value)
