"""
Feature Manager

Manages feature discovery, loading, and lifecycle.
"""

import importlib
import pkgutil
from typing import Dict, List, Optional, Type
from pathlib import Path

from .base import FeatureBase, FeatureConfig, FeatureStatus, FeatureError
from .core.schema import schema_validator


class FeatureManager:
    """Manager for all Xencode features"""
    
    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path('.xencode/features.json')
        self.features: Dict[str, FeatureBase] = {}
        self._feature_classes: Dict[str, Type[FeatureBase]] = {}
        self._load_feature_classes()
    
    def _load_feature_classes(self) -> None:
        """Load all feature classes from the features module"""
        import xencode.features
        
        # Get the features package path
        features_path = Path(xencode.features.__file__).parent
        
        # Walk through all submodules
        for _, module_name, _ in pkgutil.walk_packages(
            path=[str(features_path)],
            prefix='xencode.features.'
        ):
            try:
                # Skip base module and core modules
                if (module_name.endswith('.base') or 
                    module_name.endswith('__init__') or
                    '.core.' in module_name or
                    module_name.endswith('.core')):
                    continue
                
                # Import the module
                module = importlib.import_module(module_name)
                
                # Find feature classes in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    
                    # Check if it's a feature class (subclass of FeatureBase)
                    if (isinstance(attr, type) and 
                        issubclass(attr, FeatureBase) and 
                        attr != FeatureBase):
                        # Get the feature name from the class
                        feature_name = attr_name.lower().replace('feature', '')
                        self._feature_classes[feature_name] = attr
                        
            except (ImportError, AttributeError) as e:
                # Skip modules that can't be imported
                continue
    
    def get_available_features(self) -> List[str]:
        """Get list of available feature names"""
        return list(self._feature_classes.keys())
    
    def get_feature_class(self, name: str) -> Optional[Type[FeatureBase]]:
        """Get feature class by name"""
        return self._feature_classes.get(name)
    
    def load_feature(self, name: str, config: FeatureConfig = None) -> Optional[FeatureBase]:
        """Load a feature by name"""
        if name in self.features:
            return self.features[name]
        
        feature_class = self.get_feature_class(name)
        if not feature_class:
            return None
        
        # Create config if not provided
        if config is None:
            config = FeatureConfig(name=name)
        
        # Validate and apply defaults from schema
        if config.config:
            valid, errors = schema_validator.validate_config(name, config.config)
            if not valid:
                raise FeatureError(f"Invalid configuration for feature '{name}': {', '.join(errors)}")
            
            # Apply defaults
            config.config = schema_validator.apply_defaults(name, config.config)
        
        # Create feature instance
        feature = feature_class(config)
        self.features[name] = feature
        
        return feature
    
    async def initialize_feature(self, name: str, config: FeatureConfig = None) -> bool:
        """Initialize a feature by name"""
        feature = self.load_feature(name, config)
        if not feature:
            return False
        
        return await feature.initialize()
    
    async def initialize_all_features(self) -> Dict[str, bool]:
        """Initialize all available features"""
        results = {}
        
        for feature_name in self._feature_classes.keys():
            try:
                results[feature_name] = await self.initialize_feature(feature_name)
            except Exception as e:
                results[feature_name] = False
        
        return results
    
    async def shutdown_feature(self, name: str) -> bool:
        """Shutdown a feature by name"""
        feature = self.features.get(name)
        if not feature:
            return False
        
        await feature.shutdown()
        return True
    
    async def shutdown_all_features(self) -> Dict[str, bool]:
        """Shutdown all loaded features"""
        results = {}
        
        for feature_name in list(self.features.keys()):
            try:
                results[feature_name] = await self.shutdown_feature(feature_name)
            except Exception as e:
                results[feature_name] = False
        
        return results
    
    def get_feature(self, name: str) -> Optional[FeatureBase]:
        """Get a loaded feature by name"""
        return self.features.get(name)
    
    def get_all_features(self) -> Dict[str, FeatureBase]:
        """Get all loaded features"""
        return self.features.copy()
    
    def get_enabled_features(self) -> Dict[str, FeatureBase]:
        """Get all enabled features"""
        return {
            name: feature 
            for name, feature in self.features.items() 
            if feature.is_enabled
        }
    
    def get_features_by_status(self, status: FeatureStatus) -> List[FeatureBase]:
        """Get features by status"""
        return [
            feature 
            for feature in self.features.values() 
            if feature.get_status() == status
        ]
    
    def reload_feature(self, name: str) -> bool:
        """Reload a feature (useful for development)"""
        if name not in self.features:
            return False
        
        # Get the feature's module
        feature_class = self._feature_classes.get(name)
        if not feature_class:
            return False
        
        # Reload the module
        module = importlib.import_module(feature_class.__module__)
        importlib.reload(module)
        
        # Update the feature class
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, FeatureBase) and 
                attr != FeatureBase):
                feature_name = attr_name.lower().replace('feature', '')
                if feature_name == name:
                    self._feature_classes[name] = attr
                    return True
        
        return False
