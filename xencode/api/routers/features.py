#!/usr/bin/env python3
"""
Features API Router

FastAPI router for feature management endpoints including configuration, status, and control.
Provides REST API access to all Xencode features with authentication support.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Body, status
from pydantic import BaseModel, Field

from xencode.api.auth import (
    verify_jwt_token,
    get_current_user,
    verify_collaboration_auth,
    verify_token_optional
)

router = APIRouter()


# Pydantic models for API
class FeatureConfigModel(BaseModel):
    """Feature configuration model"""
    name: str = Field(..., description="Feature name")
    enabled: bool = Field(True, description="Whether the feature is enabled")
    version: str = Field("1.0.0", description="Feature version")
    config: Dict[str, Any] = Field(default_factory=dict, description="Feature-specific configuration")
    dependencies: List[str] = Field(default_factory=list, description="Feature dependencies")


class FeatureStatusModel(BaseModel):
    """Feature status model"""
    name: str
    status: str
    enabled: bool
    initialized: bool
    version: str
    description: str


class FeatureListResponse(BaseModel):
    """Response for listing features"""
    features: List[FeatureStatusModel]
    total: int
    timestamp: datetime


class FeatureDetailResponse(BaseModel):
    """Detailed feature information"""
    name: str
    description: str
    version: str
    status: str
    enabled: bool
    initialized: bool
    config: Dict[str, Any]
    dependencies: List[str]
    cli_commands: List[str]
    api_endpoints: List[str]


class FeatureOperationResponse(BaseModel):
    """Response for feature operations"""
    success: bool
    message: str
    feature_name: str
    timestamp: datetime


class FeatureAnalyticsModel(BaseModel):
    """Feature analytics data"""
    feature_name: str
    usage_count: int
    last_used: Optional[datetime]
    error_count: int
    average_response_time_ms: float


# Dependency for authentication
async def verify_token(payload: Dict[str, Any] = Depends(verify_jwt_token)) -> Dict[str, Any]:
    """Verify JWT token for authenticated endpoints"""
    return payload


# Dependency to get feature manager
async def get_feature_manager():
    """Get the feature manager instance"""
    try:
        from xencode.features.manager import FeatureManager
        return FeatureManager()
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature manager not available"
        )


@router.get("/", response_model=FeatureListResponse)
async def list_features(
    enabled_only: bool = False,
    manager = Depends(get_feature_manager)
):
    """
    List all available features
    
    - **enabled_only**: If true, only return enabled features
    """
    try:
        features = manager.get_all_features()
        
        if enabled_only:
            features = manager.get_enabled_features()
        
        feature_list = [
            FeatureStatusModel(
                name=feature.name,
                status=feature.get_status().value,
                enabled=feature.is_enabled,
                initialized=feature.is_initialized,
                version=feature.version,
                description=feature.description
            )
            for feature in features.values()
        ]
        
        return FeatureListResponse(
            features=feature_list,
            total=len(feature_list),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list features: {str(e)}"
        )


@router.get("/{feature_name}", response_model=FeatureDetailResponse)
async def get_feature(
    feature_name: str,
    manager = Depends(get_feature_manager)
):
    """
    Get detailed information about a specific feature
    
    - **feature_name**: Name of the feature to retrieve
    """
    try:
        feature = manager.get_feature(feature_name)
        
        if not feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feature '{feature_name}' not found"
            )
        
        # Get CLI commands (simplified)
        cli_commands = []
        try:
            commands = feature.get_cli_commands()
            cli_commands = [str(cmd) for cmd in commands] if commands else []
        except Exception:
                pass  # Silently ignore
# Get API endpoints (simplified)
        api_endpoints = []
        try:
            endpoints = feature.get_api_endpoints()
            api_endpoints = [str(ep) for ep in endpoints] if endpoints else []
        except Exception:
                pass  # Silently ignore

        return FeatureDetailResponse(
            name=feature.name,
            description=feature.description,
            version=feature.version,
            status=feature.get_status().value,
            enabled=feature.is_enabled,
            initialized=feature.is_initialized,
            config=feature.get_config().config,
            dependencies=feature.get_config().dependencies,
            cli_commands=cli_commands,
            api_endpoints=api_endpoints
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feature: {str(e)}"
        )


@router.post("/{feature_name}/enable", response_model=FeatureOperationResponse)
async def enable_feature(
    feature_name: str,
    config: Optional[FeatureConfigModel] = None,
    manager = Depends(get_feature_manager)
):
    """
    Enable a feature
    
    - **feature_name**: Name of the feature to enable
    - **config**: Optional configuration for the feature
    """
    try:
        # Load feature if not already loaded
        feature = manager.get_feature(feature_name)
        
        if not feature:
            from xencode.features.base import FeatureConfig
            
            feature_config = FeatureConfig(
                name=feature_name,
                enabled=True,
                version=config.version if config else "1.0.0",
                config=config.config if config else {},
                dependencies=config.dependencies if config else []
            )
            
            success = await manager.initialize_feature(feature_name, feature_config)
        else:
            # Update config if provided
            if config:
                feature.update_config(config.config)
            
            # Initialize if not already initialized
            if not feature.is_initialized:
                success = await feature.initialize()
            else:
                success = True
        
        if success:
            return FeatureOperationResponse(
                success=True,
                message=f"Feature '{feature_name}' enabled successfully",
                feature_name=feature_name,
                timestamp=datetime.now()
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to enable feature '{feature_name}'"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable feature: {str(e)}"
        )


@router.post("/{feature_name}/disable", response_model=FeatureOperationResponse)
async def disable_feature(
    feature_name: str,
    manager = Depends(get_feature_manager)
):
    """
    Disable a feature
    
    - **feature_name**: Name of the feature to disable
    """
    try:
        success = await manager.shutdown_feature(feature_name)
        
        if success:
            return FeatureOperationResponse(
                success=True,
                message=f"Feature '{feature_name}' disabled successfully",
                feature_name=feature_name,
                timestamp=datetime.now()
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feature '{feature_name}' not found or already disabled"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable feature: {str(e)}"
        )


@router.put("/{feature_name}/config", response_model=FeatureOperationResponse)
async def update_feature_config(
    feature_name: str,
    config: Dict[str, Any] = Body(...),
    manager = Depends(get_feature_manager)
):
    """
    Update feature configuration
    
    - **feature_name**: Name of the feature to configure
    - **config**: New configuration values
    """
    try:
        feature = manager.get_feature(feature_name)
        
        if not feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feature '{feature_name}' not found"
            )
        
        feature.update_config(config)
        
        return FeatureOperationResponse(
            success=True,
            message=f"Configuration for '{feature_name}' updated successfully",
            feature_name=feature_name,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}"
        )


@router.get("/{feature_name}/status", response_model=FeatureStatusModel)
async def get_feature_status(
    feature_name: str,
    manager = Depends(get_feature_manager)
):
    """
    Get current status of a feature
    
    - **feature_name**: Name of the feature
    """
    try:
        feature = manager.get_feature(feature_name)
        
        if not feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feature '{feature_name}' not found"
            )
        
        return FeatureStatusModel(
            name=feature.name,
            status=feature.get_status().value,
            enabled=feature.is_enabled,
            initialized=feature.is_initialized,
            version=feature.version,
            description=feature.description
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feature status: {str(e)}"
        )


# Collaborative features endpoints (require authentication)
@router.post("/{feature_name}/collaborate/start", response_model=FeatureOperationResponse)
async def start_collaboration(
    feature_name: str,
    room_id: str = Body(..., embed=True),
    user: Dict[str, Any] = Depends(verify_collaboration_auth),
    manager = Depends(get_feature_manager)
):
    """
    Start a collaborative session for a feature (requires authentication)

    - **feature_name**: Name of the feature
    - **room_id**: Collaboration room identifier
    - **Authorization**: Bearer token required
    """
    try:
        feature = manager.get_feature(feature_name)

        if not feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feature '{feature_name}' not found"
            )

        # Check if feature supports collaboration
        if feature_name != 'collaborative_coding':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Feature '{feature_name}' does not support collaboration"
            )

        # Get user info from JWT payload
        user_id = user.get('user_id')
        username = user.get('username')

        # Import collaboration manager
        from xencode.collaboration.manager import CollaborationManager
        
        collab_manager = CollaborationManager()
        
        # Start collaboration session
        session = await collab_manager.start_session(
            owner_id=user_id,
            owner_username=username,
            room_id=room_id
        )

        return FeatureOperationResponse(
            success=True,
            message=f"Collaboration session started for '{feature_name}' in room '{room_id}'",
            feature_name=feature_name,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start collaboration: {str(e)}"
        )


@router.get("/{feature_name}/analytics", response_model=FeatureAnalyticsModel, dependencies=[Depends(verify_jwt_token)])
async def get_feature_analytics(
    feature_name: str,
    manager = Depends(get_feature_manager)
):
    """
    Get analytics data for a feature (requires authentication)

    - **feature_name**: Name of the feature
    """
    try:
        feature = manager.get_feature(feature_name)

        if not feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feature '{feature_name}' not found"
            )

        # Get analytics from feature if available
        # For now, return basic metrics from feature state
        feature_status = feature.get_status()
        
        return FeatureAnalyticsModel(
            feature_name=feature_name,
            usage_count=getattr(feature, 'usage_count', 0),
            last_used=getattr(feature, 'last_used', None),
            error_count=getattr(feature, 'error_count', 0),
            average_response_time_ms=getattr(feature, 'avg_response_time', 0.0)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )


router.tags = ["Features"]
