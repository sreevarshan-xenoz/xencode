"""Configuration validation for Crush integration."""

import re
from typing import List, Optional, Dict, Any
from xencode.crush.config.models import (
    Config,
    ProviderConfig,
    LSPConfig,
    MCPConfig,
    Options,
    Permissions,
    SelectedModel,
    Agent,
)


class ValidationError(Exception):
    """Configuration validation error."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Field name that failed validation
        """
        self.field = field
        if field:
            super().__init__(f"Validation error in '{field}': {message}")
        else:
            super().__init__(f"Validation error: {message}")


class ConfigValidator:
    """Validates configuration objects."""
    
    # API key patterns for validation
    API_KEY_PATTERNS = {
        "openai": re.compile(r'^sk-[A-Za-z0-9]{20,}$'),
        "anthropic": re.compile(r'^sk-ant-[A-Za-z0-9\-_]{20,}$'),
    }
    
    # URL pattern for base_url validation
    URL_PATTERN = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    
    @classmethod
    def validate(cls, config: Config) -> List[str]:
        """Validate entire configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation warnings (empty if all valid)
            
        Raises:
            ValidationError: If configuration has critical errors
        """
        validator = cls()
        warnings = []
        
        # Validate that we have providers
        if not config.providers:
            raise ValidationError(
                "At least one provider must be configured",
                field="providers"
            )
        
        # Validate that we have models
        if not config.models:
            raise ValidationError(
                "At least one model must be configured",
                field="models"
            )
        
        # Validate models
        for model_type, selected_model in config.models.items():
            try:
                validator.validate_selected_model(selected_model, model_type)
            except ValidationError as e:
                raise ValidationError(
                    str(e),
                    field=f"models.{model_type}"
                )
            
            # Check that provider exists
            if selected_model.provider not in config.providers:
                raise ValidationError(
                    f"Provider '{selected_model.provider}' not found in providers",
                    field=f"models.{model_type}.provider"
                )
        
        # Validate providers
        for provider_name, provider_config in config.providers.items():
            try:
                provider_warnings = validator.validate_provider(
                    provider_config,
                    provider_name
                )
                warnings.extend(provider_warnings)
            except ValidationError as e:
                raise ValidationError(
                    str(e),
                    field=f"providers.{provider_name}"
                )
        
        # Validate LSP clients
        for language, lsp_config in config.lsp.items():
            try:
                validator.validate_lsp(lsp_config, language)
            except ValidationError as e:
                raise ValidationError(
                    str(e),
                    field=f"lsp.{language}"
                )
        
        # Validate MCP servers
        for server_name, mcp_config in config.mcp.items():
            try:
                validator.validate_mcp(mcp_config, server_name)
            except ValidationError as e:
                raise ValidationError(
                    str(e),
                    field=f"mcp.{server_name}"
                )
        
        # Validate options
        try:
            validator.validate_options(config.options)
        except ValidationError as e:
            raise ValidationError(
                str(e),
                field="options"
            )
        
        # Validate permissions
        try:
            validator.validate_permissions(config.permissions)
        except ValidationError as e:
            raise ValidationError(
                str(e),
                field="permissions"
            )
        
        # Validate agents
        for agent_name, agent_config in config.agents.items():
            try:
                validator.validate_agent(agent_config, agent_name)
            except ValidationError as e:
                raise ValidationError(
                    str(e),
                    field=f"agents.{agent_name}"
                )
        
        return warnings
    
    def validate_selected_model(
        self,
        model: SelectedModel,
        model_type: str
    ) -> None:
        """Validate selected model configuration.
        
        Args:
            model: Selected model to validate
            model_type: Type of model (large, small, etc.)
            
        Raises:
            ValidationError: If model is invalid
        """
        if not model.provider:
            raise ValidationError(
                f"Provider is required for {model_type} model"
            )
        
        if not model.model:
            raise ValidationError(
                f"Model name is required for {model_type} model"
            )
    
    def validate_provider(
        self,
        provider: ProviderConfig,
        provider_name: str
    ) -> List[str]:
        """Validate provider configuration.
        
        Args:
            provider: Provider configuration to validate
            provider_name: Name of the provider
            
        Returns:
            List of validation warnings
            
        Raises:
            ValidationError: If provider is invalid
        """
        warnings = []
        
        # Validate API key
        if not provider.api_key:
            raise ValidationError(
                f"API key is required for provider '{provider_name}'"
            )
        
        # Check API key format (warning only)
        if provider_name.lower() in self.API_KEY_PATTERNS:
            pattern = self.API_KEY_PATTERNS[provider_name.lower()]
            if not pattern.match(provider.api_key):
                warnings.append(
                    f"API key for provider '{provider_name}' does not match "
                    f"expected format. This may cause authentication errors."
                )
        
        # Validate base URL if provided
        if provider.base_url:
            if not self.URL_PATTERN.match(provider.base_url):
                raise ValidationError(
                    f"Invalid base URL for provider '{provider_name}': "
                    f"{provider.base_url}"
                )
        
        # Validate timeout
        if provider.timeout <= 0:
            raise ValidationError(
                f"Timeout must be positive for provider '{provider_name}'"
            )
        
        # Validate max retries
        if provider.max_retries < 0:
            raise ValidationError(
                f"Max retries cannot be negative for provider '{provider_name}'"
            )
        
        # Validate pricing (warnings only)
        if provider.price_per_1m_input < 0:
            warnings.append(
                f"Negative input price for provider '{provider_name}'"
            )
        
        if provider.price_per_1m_output < 0:
            warnings.append(
                f"Negative output price for provider '{provider_name}'"
            )
        
        return warnings
    
    def validate_lsp(self, lsp: LSPConfig, language: str) -> None:
        """Validate LSP configuration.
        
        Args:
            lsp: LSP configuration to validate
            language: Language name
            
        Raises:
            ValidationError: If LSP config is invalid
        """
        if not lsp.command:
            raise ValidationError(
                f"Command is required for LSP client '{language}'"
            )
        
        # Validate that command is not empty
        if not lsp.command.strip():
            raise ValidationError(
                f"Command cannot be empty for LSP client '{language}'"
            )
    
    def validate_mcp(self, mcp: MCPConfig, server_name: str) -> None:
        """Validate MCP configuration.
        
        Args:
            mcp: MCP configuration to validate
            server_name: Server name
            
        Raises:
            ValidationError: If MCP config is invalid
        """
        if not mcp.command:
            raise ValidationError(
                f"Command is required for MCP server '{server_name}'"
            )
        
        # Validate that command is not empty
        if not mcp.command.strip():
            raise ValidationError(
                f"Command cannot be empty for MCP server '{server_name}'"
            )
    
    def validate_options(self, options: Options) -> None:
        """Validate options configuration.
        
        Args:
            options: Options to validate
            
        Raises:
            ValidationError: If options are invalid
        """
        if options.max_context_tokens <= 0:
            raise ValidationError(
                "Max context tokens must be positive"
            )
        
        if options.command_timeout <= 0:
            raise ValidationError(
                "Command timeout must be positive"
            )
        
        if options.background_job_timeout <= 0:
            raise ValidationError(
                "Background job timeout must be positive"
            )
        
        if options.permission_timeout <= 0:
            raise ValidationError(
                "Permission timeout must be positive"
            )
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if options.log_level not in valid_log_levels:
            raise ValidationError(
                f"Invalid log level: {options.log_level}. "
                f"Must be one of: {', '.join(valid_log_levels)}"
            )
    
    def validate_permissions(self, permissions: Permissions) -> None:
        """Validate permissions configuration.
        
        Args:
            permissions: Permissions to validate
            
        Raises:
            ValidationError: If permissions are invalid
        """
        # Validate that blocked commands is a list
        if not isinstance(permissions.blocked_commands, list):
            raise ValidationError(
                "Blocked commands must be a list"
            )
        
        # Validate that allowed tools is a list
        if not isinstance(permissions.allowed_tools, list):
            raise ValidationError(
                "Allowed tools must be a list"
            )
        
        # Validate that allowed directories is a list
        if not isinstance(permissions.allowed_directories, list):
            raise ValidationError(
                "Allowed directories must be a list"
            )
    
    def validate_agent(self, agent: Agent, agent_name: str) -> None:
        """Validate agent configuration.
        
        Args:
            agent: Agent configuration to validate
            agent_name: Agent name
            
        Raises:
            ValidationError: If agent is invalid
        """
        if not agent.name:
            raise ValidationError(
                f"Name is required for agent '{agent_name}'"
            )
        
        if not agent.system_prompt:
            raise ValidationError(
                f"System prompt is required for agent '{agent_name}'"
            )
        
        if not agent.system_prompt.strip():
            raise ValidationError(
                f"System prompt cannot be empty for agent '{agent_name}'"
            )
