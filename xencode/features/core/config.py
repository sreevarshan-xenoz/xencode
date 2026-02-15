"""
Feature Configuration System

Configuration management for features.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..base import FeatureConfig


@dataclass
class FeatureSystemConfig:
    """System-wide feature configuration"""
    features_dir: str = "xencode/features"
    config_dir: str = ".xencode"
    default_enabled: bool = True
    auto_load: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSystemConfig':
        """Create config from dictionary"""
        return cls(
            features_dir=data.get('features_dir', 'xencode/features'),
            config_dir=data.get('config_dir', '.xencode'),
            default_enabled=data.get('default_enabled', True),
            auto_load=data.get('auto_load', True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'features_dir': self.features_dir,
            'config_dir': self.config_dir,
            'default_enabled': self.default_enabled,
            'auto_load': self.auto_load
        }


class FeatureConfigManager:
    """Manager for feature configuration"""
    
    def __init__(self, system_config: FeatureSystemConfig = None):
        self.system_config = system_config or FeatureSystemConfig()
        self._config_path = Path(self.system_config.config_dir) / "features.json"
        self._user_config: Dict[str, Any] = {}
        self._load_user_config()
    
    def _load_user_config(self) -> None:
        """Load user configuration from file"""
        if self._config_path.exists():
            try:
                with open(self._config_path, 'r') as f:
                    self._user_config = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._user_config = {}
    
    def _save_user_config(self) -> None:
        """Save user configuration to file"""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self._config_path, 'w') as f:
            json.dump(self._user_config, f, indent=2)
    
    def get_feature_config(self, feature_name: str) -> Dict[str, Any]:
        """Get configuration for a specific feature"""
        return self._user_config.get(feature_name, {})
    
    def set_feature_config(self, feature_name: str, config: Dict[str, Any]) -> None:
        """Set configuration for a specific feature"""
        self._user_config[feature_name] = config
        self._save_user_config()
    
    def get_enabled_features(self) -> list:
        """Get list of enabled features"""
        enabled = []
        
        for feature_name, config in self._user_config.items():
            if config.get('enabled', True):
                enabled.append(feature_name)
        
        return enabled
    
    def set_feature_enabled(self, feature_name: str, enabled: bool) -> None:
        """Enable or disable a feature"""
        if feature_name not in self._user_config:
            self._user_config[feature_name] = {}
        
        self._user_config[feature_name]['enabled'] = enabled
        self._save_user_config()
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all feature configurations"""
        return self._user_config.copy()
    
    def create_feature_config(self, feature_name: str) -> FeatureConfig:
        """Create a FeatureConfig for a feature"""
        user_config = self.get_feature_config(feature_name)
        
        return FeatureConfig(
            name=feature_name,
            enabled=user_config.get('enabled', self.system_config.default_enabled),
            version=user_config.get('version', '1.0.0'),
            config=user_config.get('config', {}),
            dependencies=user_config.get('dependencies', [])
        )
