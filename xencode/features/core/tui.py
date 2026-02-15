"""
Feature TUI Integration

TUI component manager for features.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from ..base import FeatureBase, FeatureConfig


class FeatureTUIManager:
    """Manager for feature TUI components"""
    
    def __init__(self):
        self.components: Dict[str, Any] = {}
        self.panels: Dict[str, Any] = {}
        self.views: Dict[str, Any] = {}
    
    def register_component(self, feature_name: str, component: Any) -> None:
        """Register a TUI component for a feature"""
        self.components[feature_name] = component
    
    def register_panel(self, feature_name: str, panel: Any) -> None:
        """Register a TUI panel for a feature"""
        self.panels[feature_name] = panel
    
    def register_view(self, feature_name: str, view: Any) -> None:
        """Register a TUI view for a feature"""
        self.views[feature_name] = view
    
    def get_component(self, feature_name: str) -> Optional[Any]:
        """Get a registered component"""
        return self.components.get(feature_name)
    
    def get_panel(self, feature_name: str) -> Optional[Any]:
        """Get a registered panel"""
        return self.panels.get(feature_name)
    
    def get_view(self, feature_name: str) -> Optional[Any]:
        """Get a registered view"""
        return self.views.get(feature_name)
    
    def get_all_components(self) -> Dict[str, Any]:
        """Get all registered components"""
        return self.components.copy()
    
    def get_all_panels(self) -> Dict[str, Any]:
        """Get all registered panels"""
        return self.panels.copy()
    
    def get_all_views(self) -> Dict[str, Any]:
        """Get all registered views"""
        return self.views.copy()
    
    def create_status_indicator(self, feature: FeatureBase) -> str:
        """Create a status indicator for a feature"""
        status = feature.get_status()
        
        if status == FeatureStatus.ENABLED:
            return "[green]●[/green]"
        elif status == FeatureStatus.DISABLED:
            return "[gray]○[/gray]"
        elif status == FeatureStatus.ERROR:
            return "[red]●[/red]"
        elif status == FeatureStatus.INITIALIZING:
            return "[yellow]●[/yellow]"
        else:
            return "[gray]○[/gray]"
    
    def create_navigation(self, features: List[str]) -> Dict[str, str]:
        """Create navigation mappings for features"""
        navigation = {}
        
        for i, feature in enumerate(features, 1):
            navigation[str(i)] = feature
            navigation[feature.lower()] = feature
        
        return navigation
