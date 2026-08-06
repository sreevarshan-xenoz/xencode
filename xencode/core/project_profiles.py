#!/usr/bin/env python3
"""
Per-Project Model Profiles

Auto-switch model/provider policy by workspace profile (`.xencode.json`).

Features:
- Project-specific model configuration
- Automatic profile detection and loading
- Model/provider policy inheritance
- Environment variable substitution
- Profile validation and migration
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

console = Console()


@dataclass
class ModelProfile:
    """
    Model profile configuration for a project

    Attributes:
        name: Profile name
        default_model: Default model for general tasks
        code_model: Model for code generation/review
        chat_model: Model for casual chat
        reasoning_model: Model for complex reasoning
        provider_priority: Ordered list of preferred providers
        fallback_chain: Fallback model chain
        context_window: Context window size to use
        max_tokens: Maximum tokens per request
        temperature: Default temperature
        cost_budget: Monthly cost budget (USD)
        local_first: Prefer local models when available
    """
    name: str
    default_model: str = "qwen2.5:7b"
    code_model: str = "qwen3-coder-next-instruct"
    chat_model: str = "llama3.2:3b"
    reasoning_model: str = "qwen-max"
    provider_priority: List[str] = field(default_factory=lambda: ["local_ollama", "cloud_qwen"])
    fallback_chain: List[str] = field(default_factory=list)
    context_window: int = 128000
    max_tokens: int = 4096
    temperature: float = 0.7
    cost_budget: float = 10.0
    local_first: bool = True
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'default_model': self.default_model,
            'code_model': self.code_model,
            'chat_model': self.chat_model,
            'reasoning_model': self.reasoning_model,
            'provider_priority': self.provider_priority,
            'fallback_chain': self.fallback_chain,
            'context_window': self.context_window,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'cost_budget': self.cost_budget,
            'local_first': self.local_first,
            'enabled': self.enabled,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelProfile':
        return cls(
            name=data.get('name', 'default'),
            default_model=data.get('default_model', 'qwen2.5:7b'),
            code_model=data.get('code_model', 'qwen3-coder-next-instruct'),
            chat_model=data.get('chat_model', 'llama3.2:3b'),
            reasoning_model=data.get('reasoning_model', 'qwen-max'),
            provider_priority=data.get('provider_priority', ['local_ollama', 'cloud_qwen']),
            fallback_chain=data.get('fallback_chain', []),
            context_window=data.get('context_window', 128000),
            max_tokens=data.get('max_tokens', 4096),
            temperature=data.get('temperature', 0.7),
            cost_budget=data.get('cost_budget', 10.0),
            local_first=data.get('local_first', True),
            enabled=data.get('enabled', True),
            metadata=data.get('metadata', {}),
        )

    def get_model_for_task(self, task_type: str) -> str:
        """Get model for specific task type"""
        task_model_map = {
            'code': self.code_model,
            'code_generation': self.code_model,
            'code_review': self.code_model,
            'debugging': self.code_model,
            'chat': self.chat_model,
            'reasoning': self.reasoning_model,
            'analysis': self.reasoning_model,
            'general': self.default_model,
            'documentation': self.default_model,
        }

        return task_model_map.get(task_type.lower(), self.default_model)


@dataclass
class ProjectConfig:
    """
    Complete project configuration

    Attributes:
        project_root: Root directory of the project
        profile: Active model profile
        providers: Provider configurations
        options: General options
        permissions: Permission settings
        agents: Agent configurations
    """
    project_root: Path
    profile: ModelProfile
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    permissions: Dict[str, Any] = field(default_factory=dict)
    agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'profile': self.profile.to_dict(),
            'providers': self.providers,
            'options': self.options,
            'permissions': self.permissions,
            'agents': self.agents,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], project_root: Path) -> 'ProjectConfig':
        return cls(
            project_root=project_root,
            profile=ModelProfile.from_dict(data.get('profile', {})),
            providers=data.get('providers', {}),
            options=data.get('options', {}),
            permissions=data.get('permissions', {}),
            agents=data.get('agents', {}),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else datetime.now(),
            version=data.get('version', '2.0'),
        )


class ProjectProfileManager:
    """
    Manages per-project model profiles and configurations

    Features:
    - Auto-detect project root
    - Load/save `.xencode.json` configuration
    - Profile inheritance and overrides
    - Environment variable substitution
    - Configuration validation
    """

    CONFIG_FILENAME = ".xencode.json"
    EXAMPLE_CONFIG_FILENAME = ".xencode.example.json"

    # Default profiles
    DEFAULT_PROFILES = {
        'default': ModelProfile(
            name='default',
            default_model='qwen2.5:7b',
            code_model='qwen3-coder-next-instruct',
            chat_model='llama3.2:3b',
            reasoning_model='qwen-max',
            local_first=True,
        ),
        'cloud_optimized': ModelProfile(
            name='cloud_optimized',
            default_model='qwen-turbo',
            code_model='qwen3-coder-next-instruct',
            chat_model='qwen-turbo',
            reasoning_model='qwen-max',
            provider_priority=['cloud_qwen', 'local_ollama'],
            local_first=False,
            cost_budget=5.0,
        ),
        'local_only': ModelProfile(
            name='local_only',
            default_model='qwen2.5:7b',
            code_model='qwen2.5-coder:7b',
            chat_model='llama3.2:3b',
            reasoning_model='qwen2.5:72b',
            provider_priority=['local_ollama'],
            local_first=True,
            cost_budget=0.0,
        ),
        'high_performance': ModelProfile(
            name='high_performance',
            default_model='qwen-plus',
            code_model='qwen3-coder-next-instruct',
            chat_model='qwen-plus',
            reasoning_model='qwen-max',
            provider_priority=['cloud_qwen', 'cloud_anthropic'],
            local_first=False,
            cost_budget=50.0,
            context_window=256000,
        ),
    }

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize profile manager

        Args:
            project_root: Optional project root directory
        """
        self.project_root = project_root or Path.cwd()
        self.config_path = self.project_root / self.CONFIG_FILENAME
        self.config: Optional[ProjectConfig] = None

        # Profile cache
        self._profile_cache: Dict[str, ModelProfile] = {}

        # Load configuration if exists
        if self.config_path.exists():
            self.load_config()

    def _substitute_env_vars(self, value: Any) -> Any:
        """Substitute environment variables in configuration values"""
        if isinstance(value, str):
            # Match ${VAR_NAME} or $VAR_NAME patterns
            pattern = r'\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)'

            def replace(match):
                var_name = match.group(1) or match.group(2)
                return os.environ.get(var_name, match.group(0))

            return re.sub(pattern, replace, value)
        elif isinstance(value, dict):
            return {k: self._substitute_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_env_vars(v) for v in value]
        return value

    def _detect_project_root(self, start_path: Path) -> Optional[Path]:
        """Detect project root by looking for config files"""
        current = start_path

        while current != current.parent:
            # Look for config file
            if (current / self.CONFIG_FILENAME).exists():
                return current

            # Look for project markers
            markers = [
                'pyproject.toml', 'setup.py', 'requirements.txt',
                'package.json', 'Cargo.toml', 'go.mod', 'pom.xml',
                '.git',
            ]

            for marker in markers:
                if (current / marker).exists():
                    return current

            current = current.parent

        return None

    def load_config(self, config_path: Optional[Path] = None) -> Optional[ProjectConfig]:
        """
        Load configuration from file

        Args:
            config_path: Optional custom config path

        Returns:
            Loaded ProjectConfig or None
        """
        path = config_path or self.config_path

        if not path.exists():
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Substitute environment variables
            data = self._substitute_env_vars(data)

            # Create config object
            project_root = path.parent
            self.config = ProjectConfig.from_dict(data, project_root)

            console.print(f"[green]✓ Loaded project config from {path}[/green]")

            return self.config

        except json.JSONDecodeError as e:
            console.print(f"[red]✗ Failed to parse config: {e}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]✗ Failed to load config: {e}[/red]")
            return None

    def save_config(self, config: Optional[ProjectConfig] = None) -> bool:
        """
        Save configuration to file

        Args:
            config: Optional config to save (uses self.config if not provided)

        Returns:
            True if saved successfully
        """
        config = config or self.config

        if not config:
            return False

        try:
            path = self.config_path
            path.parent.mkdir(parents=True, exist_ok=True)

            # Update timestamp
            config.updated_at = datetime.now()

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

            console.print(f"[green]✓ Saved project config to {path}[/green]")
            return True

        except Exception as e:
            console.print(f"[red]✗ Failed to save config: {e}[/red]")
            return False

    def create_default_config(
        self,
        profile_name: str = 'default',
        overwrite: bool = False,
    ) -> Optional[ProjectConfig]:
        """
        Create default configuration file

        Args:
            profile_name: Name of default profile to use
            overwrite: Overwrite existing config

        Returns:
            Created ProjectConfig or None
        """
        if self.config_path.exists() and not overwrite:
            console.print(f"[yellow]⚠ Config already exists at {self.config_path}[/yellow]")
            return self.config

        # Get profile
        profile = self.DEFAULT_PROFILES.get(profile_name, self.DEFAULT_PROFILES['default'])

        # Create config
        config = ProjectConfig(
            project_root=self.project_root,
            profile=profile,
            providers={
                'local_ollama': {
                    'base_url': 'http://localhost:11434',
                    'timeout': 60,
                    'max_retries': 3,
                },
                'cloud_qwen': {
                    'api_key': '${QWEN_API_KEY}',
                    'base_url': 'https://chat.qwen.ai/api/v1',
                    'timeout': 120,
                    'max_retries': 3,
                },
            },
            options={
                'auto_summarize': True,
                'max_context_tokens': profile.context_window,
                'log_level': 'INFO',
                'command_timeout': 60,
            },
            permissions={
                'skip_requests': False,
                'allowed_tools': ['view', 'grep', 'ls', 'edit'],
                'blocked_commands': [
                    'sudo', 'rm -rf', 'dd', 'mkfs', 'fdisk',
                ],
            },
            agents={
                'default': {
                    'name': 'default',
                    'system_prompt': 'You are a helpful AI coding assistant.',
                    'tools': ['bash', 'view', 'edit', 'write', 'grep', 'ls'],
                },
            },
        )

        self.config = config
        self.save_config()

        return config

    def get_profile(self, profile_name: Optional[str] = None) -> ModelProfile:
        """
        Get model profile

        Args:
            profile_name: Optional profile name (uses config default if not provided)

        Returns:
            ModelProfile instance
        """
        # Check cache
        if profile_name and profile_name in self._profile_cache:
            return self._profile_cache[profile_name]

        # Use config profile if no name provided
        if not profile_name:
            if self.config:
                return self.config.profile
            return self.DEFAULT_PROFILES['default']

        # Check if it's a default profile
        if profile_name in self.DEFAULT_PROFILES:
            return self.DEFAULT_PROFILES[profile_name]

        # Check if config has custom profiles
        if self.config and 'profiles' in self.config.options:
            profiles = self.config.options['profiles']
            if profile_name in profiles:
                profile_data = profiles[profile_name]
                profile = ModelProfile.from_dict(profile_data)
                self._profile_cache[profile_name] = profile
                return profile

        # Not found
        console.print(f"[yellow]⚠ Profile '{profile_name}' not found, using default[/yellow]")
        return self.DEFAULT_PROFILES['default']

    def set_profile(self, profile: ModelProfile):
        """Set active profile in config"""
        if not self.config:
            self.config = ProjectConfig(
                project_root=self.project_root,
                profile=profile,
            )
        else:
            self.config.profile = profile
            self.config.updated_at = datetime.now()

        self._profile_cache[profile.name] = profile

    def get_model_for_task(self, task_type: str) -> str:
        """Get model for specific task type from active profile"""
        profile = self.get_profile()
        return profile.get_model_for_task(task_type)

    def get_provider_priority(self) -> List[str]:
        """Get provider priority list from active profile"""
        profile = self.get_profile()
        return profile.provider_priority

    def is_local_first(self) -> bool:
        """Check if local models should be preferred"""
        profile = self.get_profile()
        return profile.local_first

    def get_cost_budget(self) -> float:
        """Get monthly cost budget"""
        profile = self.get_profile()
        return profile.cost_budget

    def validate_config(self) -> List[str]:
        """
        Validate current configuration

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self.config:
            errors.append("No configuration loaded")
            return errors

        # Validate profile
        profile = self.config.profile
        if not profile.name:
            errors.append("Profile name is required")
        if profile.cost_budget < 0:
            errors.append("Cost budget must be non-negative")
        if profile.context_window <= 0:
            errors.append("Context window must be positive")

        # Validate providers
        for provider_name, provider_config in self.config.providers.items():
            if 'api_key' in provider_config:
                api_key = provider_config['api_key']
                if not api_key or api_key.startswith('${'):
                    errors.append(f"Provider '{provider_name}' has unset API key")

        # Validate options
        options = self.config.options
        if options.get('max_context_tokens', 0) <= 0:
            errors.append("max_context_tokens must be positive")
        if options.get('command_timeout', 0) <= 0:
            errors.append("command_timeout must be positive")

        return errors

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        if not self.config:
            return {'status': 'not_loaded'}

        profile = self.config.profile

        return {
            'status': 'loaded',
            'project_root': str(self.config.project_root),
            'profile_name': profile.name,
            'default_model': profile.default_model,
            'code_model': profile.code_model,
            'local_first': profile.local_first,
            'cost_budget': profile.cost_budget,
            'context_window': profile.context_window,
            'providers_configured': list(self.config.providers.keys()),
            'config_path': str(self.config_path),
            'last_updated': self.config.updated_at.isoformat(),
        }


# Global manager instance
_manager: Optional[ProjectProfileManager] = None


def get_profile_manager(project_root: Optional[Path] = None) -> ProjectProfileManager:
    """Get or create global profile manager"""
    global _manager
    if _manager is None or (project_root and _manager.project_root != project_root):
        _manager = ProjectProfileManager(project_root)
    return _manager


def get_active_profile() -> ModelProfile:
    """Get active model profile"""
    manager = get_profile_manager()
    return manager.get_profile()


def get_model_for_task(task_type: str) -> str:
    """Get model for task type"""
    manager = get_profile_manager()
    return manager.get_model_for_task(task_type)


def load_project_config(project_root: Optional[Path] = None) -> Optional[ProjectConfig]:
    """Load project config"""
    manager = get_profile_manager(project_root)
    return manager.load_config()


if __name__ == "__main__":
    # Demo
    console.print("[bold blue]Per-Project Model Profiles Demo[/bold blue]\n")

    # Create manager
    manager = ProjectProfileManager()

    # Show config summary
    summary = manager.get_config_summary()
    console.print("[bold]Config Summary:[/bold]")
    for key, value in summary.items():
        console.print(f"  {key}: {value}")

    # Show available profiles
    console.print("\n[bold]Available Profiles:[/bold]")
    for name, profile in ProjectProfileManager.DEFAULT_PROFILES.items():
        console.print(f"  - {name}: {profile.default_model} (local_first={profile.local_first})")

    # Validate config
    errors = manager.validate_config()
    if errors:
        console.print("\n[red]Validation Errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
    else:
        console.print("\n[green]✓ Configuration is valid[/green]")
