"""
Feature API Integration

API endpoints for feature management.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..base import FeatureBase, FeatureConfig, FeatureStatus


class FeatureInfo(BaseModel):
    """Feature information model"""
    name: str
    status: str
    version: str
    enabled: bool
    initialized: bool
    description: Optional[str] = None


class FeatureConfigUpdate(BaseModel):
    """Feature configuration update model"""
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None


class FeatureAPI:
    """API endpoints for feature management"""
    
    def __init__(self, feature_manager):
        self.feature_manager = feature_manager
        self.router = APIRouter(prefix="/features", tags=["features"])
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup API routes"""
        self.router.get("/", response_model=List[FeatureInfo])(self._list_features)
        self.router.get("/{feature_name}", response_model=FeatureInfo)(self._get_feature)
        self.router.post("/{feature_name}/enable")(self._enable_feature)
        self.router.post("/{feature_name}/disable")(self._disable_feature)
        self.router.put("/{feature_name}/config")(self._update_config)
        self.router.get("/{feature_name}/status")(self._get_status)
    
    async def _list_features(self) -> List[FeatureInfo]:
        """List all features"""
        features = []
        
        for feature_name in self.feature_manager.get_available_features():
            feature = self.feature_manager.get_feature(feature_name)
            if feature:
                features.append(FeatureInfo(
                    name=feature_name,
                    status=feature.get_status().value,
                    version=feature.version,
                    enabled=feature.is_enabled,
                    initialized=feature.is_initialized,
                    description=feature.description
                ))
        
        return features
    
    async def _get_feature(self, feature_name: str) -> FeatureInfo:
        """Get feature information"""
        feature = self.feature_manager.get_feature(feature_name)
        
        if not feature:
            raise HTTPException(status_code=404, detail=f"Feature '{feature_name}' not found")
        
        return FeatureInfo(
            name=feature_name,
            status=feature.get_status().value,
            version=feature.version,
            enabled=feature.is_enabled,
            initialized=feature.is_initialized,
            description=feature.description
        )
    
    async def _enable_feature(self, feature_name: str) -> Dict[str, bool]:
        """Enable a feature"""
        success = await self.feature_manager.initialize_feature(feature_name)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to enable feature '{feature_name}'")
        
        return {"success": True}
    
    async def _disable_feature(self, feature_name: str) -> Dict[str, bool]:
        """Disable a feature"""
        success = await self.feature_manager.shutdown_feature(feature_name)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to disable feature '{feature_name}'")
        
        return {"success": True}
    
    async def _update_config(self, feature_name: str, config_update: FeatureConfigUpdate) -> Dict[str, Any]:
        """Update feature configuration"""
        feature = self.feature_manager.get_feature(feature_name)
        
        if not feature:
            raise HTTPException(status_code=404, detail=f"Feature '{feature_name}' not found")
        
        if config_update.enabled is not None:
            feature.config.enabled = config_update.enabled
        
        if config_update.config is not None:
            feature.update_config(config_update.config)
        
        return feature.config.to_dict()
    
    async def _get_status(self, feature_name: str) -> Dict[str, Any]:
        """Get feature status"""
        feature = self.feature_manager.get_feature(feature_name)
        
        if not feature:
            raise HTTPException(status_code=404, detail=f"Feature '{feature_name}' not found")
        
        return {
            "name": feature_name,
            "status": feature.get_status().value,
            "enabled": feature.is_enabled,
            "initialized": feature.is_initialized
        }
