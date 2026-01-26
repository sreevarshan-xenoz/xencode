#!/usr/bin/env python3
"""
Smart Configuration Management System for Xencode Phase 2

Advanced configuration system with YAML/TOML support, environment variables,
user profiles, and intelligent defaults with validation.
"""

import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
import json
import yaml
import configparser
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.table import Table
from rich.panel import Panel
from enum import Enum

# Try to import toml, fall back to tomli for Python < 3.11
try:
    import tomllib
    TOML_AVAILABLE = True
except ImportError:
    try:
        import tomli as tomllib
        TOML_AVAILABLE = True
    except ImportError:
        try:
            import toml as tomllib
            TOML_AVAILABLE = True
        except ImportError:
            TOML_AVAILABLE = False
            try:
                import sys
                if sys.platform.startswith('win'):
                    toml_console = Console(
                        force_terminal=True,
                        force_interactive=True,
                        color_system="windows",
                        legacy_windows=False,
                        encoding="utf-8"
                    )
                else:
                    toml_console = Console()
            except Exception:
                toml_console = Console()
            toml_console.print("[yellow]WARNING TOML support not available. Install 'tomli' for Python < 3.11 or 'toml' package[/yellow]")

# Initialize Rich console with proper encoding handling for Windows
try:
    import sys

    # On Windows, handle encoding issues with Rich console
    if sys.platform.startswith('win'):
        # Force UTF-8 encoding for Rich console on Windows and disable problematic features
        console = Console(
            force_terminal=True,
            force_interactive=True,
            color_system="windows",
            legacy_windows=False,  # Important: Disable legacy Windows console
            encoding="utf-8",
            record=True  # Enable recording to handle encoding issues
        )
    else:
        console = Console()
except Exception:
    # Fallback to basic console if there are issues
    console = Console()


class ConfigFormat(Enum):
    """Supported configuration file formats"""
    YAML = "yaml"
    TOML = "toml"
    JSON = "json"
    INI = "ini"


@dataclass
class ModelConfig:
    """AI model configuration"""
    name: str = "llama3.1:8b"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    system_prompt: str = "You are a helpful AI coding assistant."
    context_length: int = 4096
    
    def validate(self) -> List[str]:
        """Validate model configuration"""
        errors = []
        if not 0.0 <= self.temperature <= 2.0:
            errors.append("Temperature must be between 0.0 and 2.0")
        if not 1 <= self.max_tokens <= 8192:
            errors.append("Max tokens must be between 1 and 8192")
        if not 0.0 <= self.top_p <= 1.0:
            errors.append("Top-p must be between 0.0 and 1.0")
        if not 1 <= self.top_k <= 100:
            errors.append("Top-k must be between 1 and 100")
        if not 0.8 <= self.repeat_penalty <= 1.5:
            errors.append("Repeat penalty must be between 0.8 and 1.5")
        return errors


@dataclass 
class CacheConfig:
    """Cache system configuration"""
    enabled: bool = True
    memory_cache_mb: int = 256
    disk_cache_mb: int = 1024
    max_age_days: int = 7
    compression_enabled: bool = True
    auto_cleanup: bool = True
    
    def validate(self) -> List[str]:
        """Validate cache configuration"""
        errors = []
        if not 32 <= self.memory_cache_mb <= 2048:
            errors.append("Memory cache must be between 32MB and 2048MB")
        if not 100 <= self.disk_cache_mb <= 10240:
            errors.append("Disk cache must be between 100MB and 10GB")
        if not 1 <= self.max_age_days <= 365:
            errors.append("Max age must be between 1 and 365 days")
        return errors


@dataclass
class SecurityConfig:
    """Security and privacy configuration"""
    enabled: bool = True
    scan_code: bool = True
    check_dependencies: bool = True
    sandbox_execution: bool = True
    log_conversations: bool = False
    anonymize_logs: bool = True
    max_log_size_mb: int = 100
    
    def validate(self) -> List[str]:
        """Validate security configuration"""
        errors = []
        if not 10 <= self.max_log_size_mb <= 1000:
            errors.append("Max log size must be between 10MB and 1000MB")
        return errors


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    async_processing: bool = True
    max_concurrent_requests: int = 5
    request_timeout_seconds: int = 120
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    resource_monitoring: bool = True
    auto_optimization: bool = True
    
    def validate(self) -> List[str]:
        """Validate performance configuration"""
        errors = []
        if not 1 <= self.max_concurrent_requests <= 20:
            errors.append("Max concurrent requests must be between 1 and 20")
        if not 10 <= self.request_timeout_seconds <= 600:
            errors.append("Request timeout must be between 10 and 600 seconds")
        if not 0 <= self.retry_attempts <= 10:
            errors.append("Retry attempts must be between 0 and 10")
        if not 0.1 <= self.retry_delay_seconds <= 60.0:
            errors.append("Retry delay must be between 0.1 and 60 seconds")
        return errors


@dataclass
class UIConfig:
    """User interface configuration"""
    theme: str = "dark"
    color_output: bool = True
    show_progress: bool = True
    verbose_mode: bool = False
    auto_save: bool = True
    editor_integration: bool = True
    
    def validate(self) -> List[str]:
        """Validate UI configuration"""
        errors = []
        if self.theme not in ["dark", "light", "auto"]:
            errors.append("Theme must be 'dark', 'light', or 'auto'")
        return errors


@dataclass
class APIKeysConfig:
    """API keys configuration for external providers"""
    openai_api_key: str = ""
    google_gemini_api_key: str = ""
    openrouter_api_key: str = ""
    anthropic_api_key: str = ""
    huggingface_api_key: str = ""

    def validate(self) -> List[str]:
        """Validate API key configuration"""
        errors = []
        # Add validation if needed
        return errors


@dataclass
class XencodeConfig:
    """Complete Xencode configuration"""
    version: str = "2.0.0"
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate all configuration sections"""
        validation_results = {}

        sections = {
            "model": self.model,
            "cache": self.cache,
            "security": self.security,
            "performance": self.performance,
            "ui": self.ui,
            "api_keys": self.api_keys
        }

        for section_name, section_config in sections.items():
            if hasattr(section_config, 'validate'):
                errors = section_config.validate()
                if errors:
                    validation_results[section_name] = errors

        return validation_results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'XencodeConfig':
        """Create configuration from dictionary"""
        config = cls()

        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'cache' in data:
            config.cache = CacheConfig(**data['cache'])
        if 'security' in data:
            config.security = SecurityConfig(**data['security'])
        if 'performance' in data:
            config.performance = PerformanceConfig(**data['performance'])
        if 'ui' in data:
            config.ui = UIConfig(**data['ui'])
        if 'api_keys' in data:
            config.api_keys = APIKeysConfig(**data['api_keys'])
        if 'custom' in data:
            config.custom = data['custom']
        if 'version' in data:
            config.version = data['version']

        return config


class ConfigurationManager:
    """Smart configuration manager with multiple format support"""
    
    DEFAULT_CONFIG_PATHS = [
        Path.home() / ".xencode" / "config.yaml",
        Path.home() / ".xencode" / "config.toml", 
        Path.home() / ".xencode" / "config.json",
        Path.home() / ".config" / "xencode" / "config.yaml",
        Path("xencode.yaml"),
        Path("xencode.toml"),
        Path(".xencode.yaml"),
    ]
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.config: Optional[XencodeConfig] = None
        self.config_format: Optional[ConfigFormat] = None
        self.environment_overrides: Dict[str, Any] = {}
        
        # Load environment variables
        self._load_environment_overrides()
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        env_mapping = {
            'XENCODE_MODEL_NAME': ('model', 'name'),
            'XENCODE_MODEL_TEMPERATURE': ('model', 'temperature', float),
            'XENCODE_MODEL_MAX_TOKENS': ('model', 'max_tokens', int),
            'XENCODE_CACHE_ENABLED': ('cache', 'enabled', lambda x: x.lower() == 'true'),
            'XENCODE_CACHE_MEMORY_MB': ('cache', 'memory_cache_mb', int),
            'XENCODE_CACHE_DISK_MB': ('cache', 'disk_cache_mb', int),
            'XENCODE_SECURITY_ENABLED': ('security', 'enabled', lambda x: x.lower() == 'true'),
            'XENCODE_VERBOSE': ('ui', 'verbose_mode', lambda x: x.lower() == 'true'),
            'XENCODE_THEME': ('ui', 'theme'),
            'OPENAI_API_KEY': ('api_keys', 'openai_api_key'),
            'GOOGLE_GEMINI_API_KEY': ('api_keys', 'google_gemini_api_key'),
            'OPENROUTER_API_KEY': ('api_keys', 'openrouter_api_key'),
            'ANTHROPIC_API_KEY': ('api_keys', 'anthropic_api_key'),
            'HUGGINGFACE_API_KEY': ('api_keys', 'huggingface_api_key'),
        }
        
        for env_var, config_path in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Parse value if converter provided
                if len(config_path) > 2:
                    converter = config_path[2]
                    try:
                        value = converter(value)
                    except (ValueError, TypeError):
                        console.print(f"[yellow]⚠️  Invalid environment variable {env_var}: {value}[/yellow]")
                        continue
                
                # Store override
                section, key = config_path[0], config_path[1]
                if section not in self.environment_overrides:
                    self.environment_overrides[section] = {}
                self.environment_overrides[section][key] = value
    
    def find_config_file(self) -> Optional[Path]:
        """Find configuration file in standard locations"""
        if self.config_path and self.config_path.exists():
            return self.config_path
        
        for path in self.DEFAULT_CONFIG_PATHS:
            if path.exists():
                return path
        
        return None
    
    def detect_format(self, config_path: Path) -> ConfigFormat:
        """Detect configuration file format"""
        suffix = config_path.suffix.lower()
        
        if suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        elif suffix == '.toml':
            return ConfigFormat.TOML
        elif suffix == '.json':
            return ConfigFormat.JSON
        elif suffix in ['.ini', '.cfg']:
            return ConfigFormat.INI
        else:
            # Try to detect by content
            try:
                with open(config_path, 'r') as f:
                    content = f.read().strip()
                
                if content.startswith('{') or content.startswith('['):
                    return ConfigFormat.JSON
                elif '=' in content and '[' in content:
                    return ConfigFormat.INI
                elif content.startswith('---') or ':' in content:
                    return ConfigFormat.YAML
                else:
                    return ConfigFormat.TOML
            except (OSError, IOError, UnicodeDecodeError):
                return ConfigFormat.YAML  # Default
    
    def load_config(self, config_path: Optional[Path] = None) -> XencodeConfig:
        """Load configuration from file or create default"""
        # Find config file
        config_file = config_path or self.find_config_file()
        
        if config_file:
            try:
                self.config_format = self.detect_format(config_file)
                self.config_path = config_file
                
                # Load based on format
                if self.config_format == ConfigFormat.YAML:
                    config_data = self._load_yaml(config_file)
                elif self.config_format == ConfigFormat.TOML:
                    config_data = self._load_toml(config_file)
                elif self.config_format == ConfigFormat.JSON:
                    config_data = self._load_json(config_file)
                elif self.config_format == ConfigFormat.INI:
                    config_data = self._load_ini(config_file)
                else:
                    raise ValueError(f"Unsupported format: {self.config_format}")
                
                self.config = XencodeConfig.from_dict(config_data)
                console.print(f"[green]OK Configuration loaded from {config_file}[/green]")
                
            except Exception as e:
                console.print(f"[red]ERROR Error loading config: {e}[/red]")
                console.print("[yellow]Using default configuration[/yellow]")
                self.config = XencodeConfig()
        else:
            # Create default configuration
            self.config = XencodeConfig()
            console.print("[yellow]⚠️  No config file found, using defaults[/yellow]")
        
        # Apply environment overrides
        self._apply_environment_overrides()
        
        # Validate configuration
        self._validate_and_fix_config()
        
        return self.config
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration"""
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _load_toml(self, path: Path) -> Dict[str, Any]:
        """Load TOML configuration"""
        if not TOML_AVAILABLE:
            raise ImportError("TOML support not available. Install 'tomli' package.")
        
        with open(path, 'rb') as f:
            return tomllib.load(f)
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_ini(self, path: Path) -> Dict[str, Any]:
        """Load INI configuration"""
        config = configparser.ConfigParser()
        config.read(path)
        
        # Convert to nested dictionary
        result = {}
        for section_name in config.sections():
            result[section_name] = dict(config[section_name])
        
        return result
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides to configuration"""
        if not self.config:
            return
        
        for section_name, overrides in self.environment_overrides.items():
            if hasattr(self.config, section_name):
                section = getattr(self.config, section_name)
                for key, value in overrides.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                        console.print(f"[blue]🔧 Environment override: {section_name}.{key} = {value}[/blue]")
    
    def _validate_and_fix_config(self):
        """Validate configuration and fix common issues"""
        if not self.config:
            return
        
        validation_errors = self.config.validate()
        
        if validation_errors:
            console.print("[yellow]⚠️  Configuration validation warnings:[/yellow]")
            for section, errors in validation_errors.items():
                console.print(f"  {section}:")
                for error in errors:
                    console.print(f"    • {error}")
            
            # Auto-fix common issues
            self._auto_fix_config()
    
    def _auto_fix_config(self):
        """Automatically fix common configuration issues"""
        if not self.config:
            return
        
        # Fix model config
        if self.config.model.temperature < 0.0:
            self.config.model.temperature = 0.0
        elif self.config.model.temperature > 2.0:
            self.config.model.temperature = 2.0
        
        if self.config.model.max_tokens < 1:
            self.config.model.max_tokens = 1024
        elif self.config.model.max_tokens > 8192:
            self.config.model.max_tokens = 8192
        
        # Fix cache config
        if self.config.cache.memory_cache_mb < 32:
            self.config.cache.memory_cache_mb = 32
        elif self.config.cache.memory_cache_mb > 2048:
            self.config.cache.memory_cache_mb = 2048
        
        console.print("[green]OK Configuration auto-fixed[/green]")
    
    def save_config(self, config_path: Optional[Path] = None, 
                   format: Optional[ConfigFormat] = None) -> bool:
        """Save configuration to file"""
        if not self.config:
            return False
        
        save_path = config_path or self.config_path
        save_format = format or self.config_format or ConfigFormat.YAML
        
        # Ensure directory exists
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Create default path
            save_path = Path.home() / ".xencode" / f"config.{save_format.value}"
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = self.config.to_dict()
            
            # Save based on format
            if save_format == ConfigFormat.YAML:
                self._save_yaml(save_path, config_dict)
            elif save_format == ConfigFormat.TOML:
                self._save_toml(save_path, config_dict)
            elif save_format == ConfigFormat.JSON:
                self._save_json(save_path, config_dict)
            elif save_format == ConfigFormat.INI:
                self._save_ini(save_path, config_dict)
            
            self.config_path = save_path
            self.config_format = save_format
            console.print(f"[green]OK Configuration saved to {save_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]ERROR Error saving config: {e}[/red]")
            return False
    
    def _save_yaml(self, path: Path, data: Dict[str, Any]):
        """Save YAML configuration"""
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    
    def _save_toml(self, path: Path, data: Dict[str, Any]):
        """Save TOML configuration"""
        if not TOML_AVAILABLE:
            raise ImportError("TOML support not available.")
        
        try:
            # Try tomli-w for writing
            import tomli_w
            with open(path, 'wb') as f:
                tomli_w.dump(data, f)
        except ImportError:
            # Fall back to basic TOML writing (limited)
            with open(path, 'w') as f:
                f.write(self._dict_to_toml(data))
    
    def _dict_to_toml(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Basic TOML serialization fallback"""
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"\n{prefix}[{key}]")
                lines.append(self._dict_to_toml(value, indent + 1))
            elif isinstance(value, str):
                lines.append(f'{prefix}{key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f'{prefix}{key} = {str(value).lower()}')
            else:
                lines.append(f'{prefix}{key} = {value}')
        
        return "\n".join(lines)
    
    def _save_json(self, path: Path, data: Dict[str, Any]):
        """Save JSON configuration"""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_ini(self, path: Path, data: Dict[str, Any]):
        """Save INI configuration"""
        config = configparser.ConfigParser()
        
        for section_name, section_data in data.items():
            if isinstance(section_data, dict):
                config.add_section(section_name)
                for key, value in section_data.items():
                    config.set(section_name, key, str(value))
        
        with open(path, 'w') as f:
            config.write(f)
    
    def get_config(self) -> XencodeConfig:
        """Get current configuration"""
        if self.config is None:
            self.config = self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any], save: bool = True) -> bool:
        """Update configuration with new values"""
        if not self.config:
            self.config = XencodeConfig()
        
        try:
            # Apply updates
            for section_name, section_updates in updates.items():
                if hasattr(self.config, section_name):
                    section = getattr(self.config, section_name)
                    if isinstance(section_updates, dict):
                        for key, value in section_updates.items():
                            if hasattr(section, key):
                                setattr(section, key, value)
                    else:
                        setattr(self.config, section_name, section_updates)
            
            # Validate
            validation_errors = self.config.validate()
            if validation_errors:
                console.print("[yellow]⚠️  Configuration validation warnings after update[/yellow]")
                self._auto_fix_config()
            
            # Save if requested
            if save:
                return self.save_config()
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error updating config: {e}[/red]")
            return False
    
    def show_config(self):
        """Display current configuration"""
        if not self.config:
            console.print("[red]No configuration loaded[/red]")
            return

        table = Table(title="🔧 Xencode Configuration")
        table.add_column("Section", style="cyan")
        table.add_column("Setting", style="white")
        table.add_column("Value", style="green")

        config_dict = self.config.to_dict()

        for section_name, section_data in config_dict.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if section_name == "model" and key == "system_prompt":
                        # Truncate long system prompts
                        display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    elif section_name == "api_keys":
                        # Mask API keys for security
                        if "api_key" in key.lower():
                            display_value = "*" * len(str(value)) if value else "<not set>"
                        else:
                            display_value = str(value)
                    else:
                        display_value = str(value)

                    table.add_row(section_name, key, display_value)
            else:
                table.add_row(section_name, "", str(section_data))

        console.print(table)
    
    def interactive_setup(self) -> XencodeConfig:
        """Interactive configuration setup wizard"""
        console.print("[bold blue]🔧 Xencode Configuration Setup[/bold blue]\n")

        config = XencodeConfig()

        # API Keys configuration
        console.print("[bold]API Keys Configuration:[/bold]")
        config.api_keys.openai_api_key = Prompt.ask("OpenAI API Key (optional)", password=True, default="")
        config.api_keys.google_gemini_api_key = Prompt.ask("Google Gemini API Key (optional)", password=True, default="")
        config.api_keys.openrouter_api_key = Prompt.ask("OpenRouter API Key (optional)", password=True, default="")
        config.api_keys.anthropic_api_key = Prompt.ask("Anthropic API Key (optional)", password=True, default="")
        config.api_keys.huggingface_api_key = Prompt.ask("HuggingFace API Key (optional)", password=True, default="")

        # Model configuration
        console.print("\n[bold]Model Configuration:[/bold]")
        config.model.name = Prompt.ask("Model name", default=config.model.name)
        config.model.temperature = FloatPrompt.ask("Temperature (0.0-2.0)", default=config.model.temperature)
        config.model.max_tokens = IntPrompt.ask("Max tokens", default=config.model.max_tokens)

        # Cache configuration
        console.print("\n[bold]Cache Configuration:[/bold]")
        config.cache.enabled = Confirm.ask("Enable caching", default=config.cache.enabled)
        if config.cache.enabled:
            config.cache.memory_cache_mb = IntPrompt.ask("Memory cache (MB)", default=config.cache.memory_cache_mb)
            config.cache.disk_cache_mb = IntPrompt.ask("Disk cache (MB)", default=config.cache.disk_cache_mb)

        # Security configuration
        console.print("\n[bold]Security Configuration:[/bold]")
        config.security.enabled = Confirm.ask("Enable security features", default=config.security.enabled)
        if config.security.enabled:
            config.security.scan_code = Confirm.ask("Scan code for vulnerabilities", default=config.security.scan_code)
            config.security.sandbox_execution = Confirm.ask("Sandbox code execution", default=config.security.sandbox_execution)

        # UI configuration
        console.print("\n[bold]UI Configuration:[/bold]")
        theme_choices = ["dark", "light", "auto"]
        config.ui.theme = Prompt.ask("Theme", choices=theme_choices, default=config.ui.theme)
        config.ui.verbose_mode = Confirm.ask("Verbose mode", default=config.ui.verbose_mode)

        self.config = config

        # Save configuration
        if Confirm.ask("\nSave configuration"):
            format_choices = ["yaml", "toml", "json"]
            format_choice = Prompt.ask("Configuration format", choices=format_choices, default="yaml")
            self.save_config(format=ConfigFormat(format_choice))

        return config


# Global configuration manager
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigurationManager:
    """Get or create global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_path)
    return _config_manager


def get_config(config_path: Optional[Path] = None) -> XencodeConfig:
    """Get current configuration"""
    manager = get_config_manager(config_path)
    return manager.get_config()


if __name__ == "__main__":
    # Demo and testing
    manager = ConfigurationManager()
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        # Interactive setup
        config = manager.interactive_setup()
    else:
        # Load and display config
        config = manager.load_config()
        manager.show_config()
        
        # Show environment overrides
        if manager.environment_overrides:
            console.print("\n[bold blue]Environment Overrides:[/bold blue]")
            for section, overrides in manager.environment_overrides.items():
                for key, value in overrides.items():
                    console.print(f"  {section}.{key} = {value}")