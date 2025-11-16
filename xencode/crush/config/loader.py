"""Configuration loader for Crush integration."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
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
from xencode.crush.config.resolver import EnvironmentResolver


class ConfigurationError(Exception):
    """Configuration error exception."""
    pass


class ConfigLoader:
    """Loads and merges configuration from multiple sources."""
    
    PROJECT_CONFIG_NAME = ".xencode.json"
    USER_CONFIG_PATH = "~/.xencode/config.json"
    
    def __init__(self, resolver: Optional[EnvironmentResolver] = None):
        """Initialize configuration loader.
        
        Args:
            resolver: Environment variable resolver (creates default if None)
        """
        self.resolver = resolver or EnvironmentResolver()
    
    @classmethod
    def load(
        cls,
        working_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        resolver: Optional[EnvironmentResolver] = None,
        validate: bool = True
    ) -> Config:
        """Load configuration from files.
        
        Args:
            working_dir: Working directory to search for project config
            data_dir: Data directory for storing database and logs
            resolver: Environment variable resolver
            validate: Whether to validate configuration (default: True)
            
        Returns:
            Loaded and merged configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        loader = cls(resolver)
        
        # Determine working directory
        if working_dir is None:
            working_dir = os.getcwd()
        working_dir = os.path.abspath(working_dir)
        
        # Load configurations
        user_config = loader._load_user_config()
        project_config = loader._load_project_config(working_dir)
        
        # Merge configurations (project takes precedence)
        merged_data = loader._merge_configs(user_config, project_config)
        
        # Resolve environment variables
        resolved_data = loader._resolve_config(merged_data)
        
        # Build Config object
        config = loader._build_config(resolved_data, working_dir, data_dir)
        
        # Validate configuration if requested
        if validate:
            warnings = loader._validate_config(config)
            if warnings:
                import logging
                logger = logging.getLogger(__name__)
                for warning in warnings:
                    logger.warning(f"Configuration warning: {warning}")
        
        return config
    
    def _load_user_config(self) -> Dict[str, Any]:
        """Load user-level configuration.
        
        Returns:
            User configuration dictionary (empty if not found)
        """
        user_config_path = Path(self.USER_CONFIG_PATH).expanduser()
        
        if not user_config_path.exists():
            return {}
        
        try:
            with open(user_config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in user config {user_config_path}: {e}"
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load user config {user_config_path}: {e}"
            )
    
    def _load_project_config(self, working_dir: str) -> Dict[str, Any]:
        """Load project-level configuration.
        
        Args:
            working_dir: Directory to search for project config
            
        Returns:
            Project configuration dictionary (empty if not found)
        """
        project_config_path = Path(working_dir) / self.PROJECT_CONFIG_NAME
        
        if not project_config_path.exists():
            return {}
        
        try:
            with open(project_config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in project config {project_config_path}: {e}"
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load project config {project_config_path}: {e}"
            )
    
    def _merge_configs(
        self,
        user_config: Dict[str, Any],
        project_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge user and project configurations.
        
        Project configuration takes precedence over user configuration.
        For nested dictionaries, merge recursively.
        
        Args:
            user_config: User-level configuration
            project_config: Project-level configuration
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        # Start with user config
        for key, value in user_config.items():
            merged[key] = value
        
        # Override with project config
        for key, value in project_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                # Override completely
                merged[key] = value
        
        return merged
    
    def _merge_dicts(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _resolve_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve environment variables in configuration.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            Configuration with resolved environment variables
        """
        return self._resolve_dict(config_data)
    
    def _resolve_dict(self, data: Any) -> Any:
        """Recursively resolve environment variables in data structure.
        
        Args:
            data: Data to resolve (dict, list, str, or other)
            
        Returns:
            Data with resolved environment variables
        """
        if isinstance(data, dict):
            return {key: self._resolve_dict(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._resolve_dict(item) for item in data]
        elif isinstance(data, str):
            return self.resolver.resolve(data)
        else:
            return data
    
    def _build_config(
        self,
        config_data: Dict[str, Any],
        working_dir: str,
        data_dir: Optional[str]
    ) -> Config:
        """Build Config object from configuration data.
        
        Args:
            config_data: Resolved configuration dictionary
            working_dir: Working directory
            data_dir: Data directory
            
        Returns:
            Config object
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Parse models
            models = {}
            if "models" in config_data:
                for model_type, model_data in config_data["models"].items():
                    if isinstance(model_data, dict):
                        models[model_type] = SelectedModel(**model_data)
                    else:
                        raise ConfigurationError(
                            f"Invalid model configuration for '{model_type}'"
                        )
            
            # Parse providers
            providers = {}
            if "providers" in config_data:
                for provider_name, provider_data in config_data["providers"].items():
                    if isinstance(provider_data, dict):
                        providers[provider_name] = ProviderConfig(**provider_data)
                    else:
                        raise ConfigurationError(
                            f"Invalid provider configuration for '{provider_name}'"
                        )
            
            # Parse LSP clients
            lsp = {}
            if "lsp" in config_data:
                for language, lsp_data in config_data["lsp"].items():
                    if isinstance(lsp_data, dict):
                        lsp[language] = LSPConfig(**lsp_data)
                    else:
                        raise ConfigurationError(
                            f"Invalid LSP configuration for '{language}'"
                        )
            
            # Parse MCP servers
            mcp = {}
            if "mcp" in config_data:
                for server_name, mcp_data in config_data["mcp"].items():
                    if isinstance(mcp_data, dict):
                        mcp[server_name] = MCPConfig(**mcp_data)
                    else:
                        raise ConfigurationError(
                            f"Invalid MCP configuration for '{server_name}'"
                        )
            
            # Parse options
            options_data = config_data.get("options", {})
            if data_dir:
                options_data["data_directory"] = data_dir
            if working_dir:
                options_data["working_directory"] = working_dir
            options = Options(**options_data)
            
            # Parse permissions
            permissions_data = config_data.get("permissions", {})
            permissions = Permissions(**permissions_data)
            
            # Parse agents
            agents = {}
            if "agents" in config_data:
                for agent_name, agent_data in config_data["agents"].items():
                    if isinstance(agent_data, dict):
                        agents[agent_name] = Agent(**agent_data)
                    else:
                        raise ConfigurationError(
                            f"Invalid agent configuration for '{agent_name}'"
                        )
            
            # Create Config object
            config = Config(
                models=models,
                providers=providers,
                lsp=lsp,
                mcp=mcp,
                options=options,
                permissions=permissions,
                agents=agents,
                _working_dir=working_dir
            )
            
            return config
            
        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration structure: {e}")
        except ValueError as e:
            raise ConfigurationError(f"Invalid configuration value: {e}")
    
    def _validate_config(self, config: Config) -> List[str]:
        """Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation warnings
            
        Raises:
            ConfigurationError: If configuration has critical errors
        """
        # Import here to avoid circular dependency
        from xencode.crush.config.validator import ConfigValidator, ValidationError
        
        try:
            warnings = ConfigValidator.validate(config)
            return warnings
        except ValidationError as e:
            raise ConfigurationError(str(e))
