"""
Plugin configuration system for Xencode
Manages plugin settings and activation
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import yaml


@dataclass
class PluginConfig:
    """Configuration for a single plugin."""
    name: str
    enabled: bool = True
    settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.settings is None:
            self.settings = {}


class PluginConfigManager:
    """Manages plugin configurations."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".xencode" / "plugin_config.json"
        self.configs: Dict[str, PluginConfig] = {}
        self.load_configs()
    
    def load_configs(self):
        """Load plugin configurations from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for name, config_data in data.items():
                    config = PluginConfig(
                        name=name,
                        enabled=config_data.get('enabled', True),
                        settings=config_data.get('settings', {})
                    )
                    self.configs[name] = config
            except Exception as e:
                print(f"Error loading plugin configs: {e}")
        else:
            # Create default config if file doesn't exist
            self.save_configs()
    
    def save_configs(self):
        """Save plugin configurations to file."""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert configs to dictionary
            data = {}
            for name, config in self.configs.items():
                data[name] = {
                    'enabled': config.enabled,
                    'settings': config.settings
                }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving plugin configs: {e}")
    
    def get_config(self, plugin_name: str) -> Optional[PluginConfig]:
        """Get configuration for a plugin."""
        return self.configs.get(plugin_name)
    
    def set_config(self, plugin_name: str, enabled: bool = True, settings: Optional[Dict[str, Any]] = None):
        """Set configuration for a plugin."""
        if settings is None:
            settings = {}
        
        config = PluginConfig(
            name=plugin_name,
            enabled=enabled,
            settings=settings
        )
        self.configs[plugin_name] = config
        self.save_configs()
    
    def enable_plugin(self, plugin_name: str):
        """Enable a plugin."""
        config = self.get_config(plugin_name)
        if config:
            config.enabled = True
        else:
            config = PluginConfig(name=plugin_name, enabled=True)
            self.configs[plugin_name] = config
        self.save_configs()
    
    def disable_plugin(self, plugin_name: str):
        """Disable a plugin."""
        config = self.get_config(plugin_name)
        if config:
            config.enabled = False
        else:
            config = PluginConfig(name=plugin_name, enabled=False)
            self.configs[plugin_name] = config
        self.save_configs()
    
    def update_settings(self, plugin_name: str, settings: Dict[str, Any]):
        """Update settings for a plugin."""
        config = self.get_config(plugin_name)
        if config:
            config.settings.update(settings)
        else:
            config = PluginConfig(name=plugin_name, enabled=True, settings=settings)
            self.configs[plugin_name] = config
        self.save_configs()
    
    def get_enabled_plugins(self) -> List[str]:
        """Get list of enabled plugins."""
        return [name for name, config in self.configs.items() if config.enabled]
    
    def get_all_plugin_names(self) -> List[str]:
        """Get all plugin names."""
        return list(self.configs.keys())


# Create a global instance
_config_manager = PluginConfigManager()


def get_plugin_config_manager() -> PluginConfigManager:
    """Get the global plugin config manager."""
    return _config_manager


def is_plugin_enabled(plugin_name: str) -> bool:
    """Check if a plugin is enabled."""
    config = _config_manager.get_config(plugin_name)
    return config.enabled if config else False


def enable_plugin(plugin_name: str):
    """Enable a plugin."""
    _config_manager.enable_plugin(plugin_name)


def disable_plugin(plugin_name: str):
    """Disable a plugin."""
    _config_manager.disable_plugin(plugin_name)


__all__ = [
    'PluginConfig',
    'PluginConfigManager',
    'get_plugin_config_manager',
    'is_plugin_enabled',
    'enable_plugin',
    'disable_plugin'
]