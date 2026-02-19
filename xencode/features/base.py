"""
Feature Base Classes

Base classes and interfaces for all Xencode features.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path


class FeatureStatus(Enum):
    """Status of a feature"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    INITIALIZING = "initializing"
    ERROR = "error"
    DEPRECATED = "deprecated"


class FeatureError(Exception):
    """Base exception for feature-related errors"""
    pass


@dataclass
class FeatureConfig:
    """Configuration for a feature"""
    name: str
    enabled: bool = True
    version: str = "1.0.0"
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureConfig':
        """Create config from dictionary"""
        return cls(
            name=data.get('name', ''),
            enabled=data.get('enabled', True),
            version=data.get('version', '1.0.0'),
            config=data.get('config', {}),
            dependencies=data.get('dependencies', [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'version': self.version,
            'config': self.config,
            'dependencies': self.dependencies
        }


class FeatureBase(ABC):
    """Base class for all Xencode features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.status = FeatureStatus.DISABLED
        self._initialized = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Feature name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Feature description"""
        pass
    
    @property
    def version(self) -> str:
        """Feature version"""
        return self.config.version
    
    @property
    def is_enabled(self) -> bool:
        """Check if feature is enabled"""
        return self.config.enabled
    
    @property
    def is_initialized(self) -> bool:
        """Check if feature is initialized"""
        return self._initialized
    
    async def initialize(self) -> bool:
        """Initialize the feature"""
        try:
            self.status = FeatureStatus.INITIALIZING
            await self._initialize()
            self._initialized = True
            self.status = FeatureStatus.ENABLED
            return True
        except Exception as e:
            self.status = FeatureStatus.ERROR
            raise FeatureError(f"Failed to initialize feature {self.name}: {str(e)}")
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Internal initialization logic"""
        pass
    
    async def shutdown(self) -> None:
        """Shutdown the feature"""
        await self._shutdown()
        self._initialized = False
        self.status = FeatureStatus.DISABLED
    
    async def _shutdown(self) -> None:
        """Internal shutdown logic"""
        pass
    
    def get_status(self) -> FeatureStatus:
        """Get current feature status"""
        return self.status
    
    def get_config(self) -> FeatureConfig:
        """Get feature configuration"""
        return self.config
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update feature configuration"""
        self.config.config.update(config)
    
    @abstractmethod
    def get_cli_commands(self) -> List[Any]:
        """Get CLI commands for this feature"""
        pass
    
    @abstractmethod
    def get_tui_components(self) -> List[Any]:
        """Get TUI components for this feature"""
        pass
    
    def get_api_endpoints(self) -> List[Any]:
        """Get API endpoints for this feature"""
        return []
    
    def track_analytics(self, event: str, properties: Dict[str, Any] = None) -> None:
        """Track analytics for this feature"""
        try:
            from xencode.analytics.event_tracker import event_tracker, EventCategory
            
            # Add feature context to properties
            feature_properties = {
                'feature_name': self.name,
                'feature_version': self.version,
                'feature_status': self.status.value,
                **(properties or {})
            }
            
            # Track the event
            event_tracker.track_event(
                event_type=f"feature_{self.name}_{event}",
                category=EventCategory.USER_ACTION,
                properties=feature_properties,
                tags=['feature', self.name, event]
            )
        except Exception as e:
            # Silently fail if analytics is not available
            pass
