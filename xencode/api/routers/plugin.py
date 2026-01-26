#!/usr/bin/env python3
"""
Plugin System API Router

FastAPI router for plugin management endpoints including marketplace integration,
plugin execution, monitoring, and comprehensive plugin lifecycle management.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import plugin system components
try:
    from ...plugin_system import (
        PluginManager, PluginMetadata, PluginStatus, PluginError,
        SecurityLevel, PluginPermission
    )
    from ...plugins.marketplace_client import MarketplaceClient
    PLUGIN_COMPONENTS_AVAILABLE = True
except ImportError:
    PLUGIN_COMPONENTS_AVAILABLE = False

router = APIRouter()


# Pydantic models for API
class PluginInfo(BaseModel):
    """Plugin information"""
    id: str
    name: str
    version: str
    author: str
    description: str
    license: str = "MIT"
    category: str = "general"
    tags: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    
    # Status information
    status: str = "available"  # available, installed, enabled, disabled, error
    installed: bool = False
    enabled: bool = False
    installed_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # Marketplace information
    downloads: int = 0
    rating: float = 0.0
    reviews_count: int = 0
    marketplace_url: Optional[str] = None
    
    # Runtime information
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    execution_count: int = 0
    last_executed: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None


class PluginInstallRequest(BaseModel):
    """Request to install plugin"""
    plugin_id: str
    version: Optional[str] = None
    source: str = "marketplace"  # marketplace, file, url
    url: Optional[str] = None
    verify_signature: bool = True
    auto_enable: bool = True


class PluginUpdateRequest(BaseModel):
    """Request to update plugin"""
    version: Optional[str] = None
    auto_restart: bool = True


class PluginExecuteRequest(BaseModel):
    """Request to execute plugin"""
    method: str
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 30
    async_execution: bool = False


class PluginSearchRequest(BaseModel):
    """Request to search plugins"""
    query: str = ""
    category: str = ""
    tags: List[str] = Field(default_factory=list)
    sort_by: str = "downloads"  # downloads, rating, name, updated
    limit: int = 50
    offset: int = 0


class PluginConfigUpdate(BaseModel):
    """Plugin configuration update"""
    config: Dict[str, Any]
    restart_required: bool = False


class PluginExecutionResult(BaseModel):
    """Plugin execution result"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: int
    memory_used_mb: float
    timestamp: datetime


class PluginStats(BaseModel):
    """Plugin statistics"""
    plugin_id: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time_ms: float
    total_memory_used_mb: float
    uptime_hours: float
    last_24h_executions: int
    error_rate_percent: float


class MarketplaceInfo(BaseModel):
    """Marketplace information"""
    total_plugins: int
    categories: List[str]
    featured_plugins: List[str]
    recent_updates: List[Dict[str, Any]]
    marketplace_status: str


# Dependency to get plugin manager
async def get_plugin_manager():
    """Dependency to get plugin manager"""
    if not PLUGIN_COMPONENTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Plugin system not available")
    
    try:
        # For now, return a mock manager - in production this would be a singleton
        return PluginManager()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plugin manager: {e}")


# Dependency to get marketplace client
async def get_marketplace_client():
    """Dependency to get marketplace client"""
    if not PLUGIN_COMPONENTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Marketplace client not available")
    
    try:
        return MarketplaceClient()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get marketplace client: {e}")


@router.get("/", response_model=List[PluginInfo])
async def list_plugins(
    status: Optional[str] = None,
    category: Optional[str] = None,
    plugin_manager = Depends(get_plugin_manager)
):
    """List installed and available plugins"""
    try:
        if PLUGIN_COMPONENTS_AVAILABLE:
            plugins = await plugin_manager.list_plugins()
            
            # Convert to API format
            plugin_list = []
            for plugin in plugins:
                plugin_info = PluginInfo(
                    id=plugin.metadata.name,
                    name=plugin.metadata.name,
                    version=plugin.metadata.version,
                    author=plugin.metadata.author,
                    description=plugin.metadata.description,
                    license=plugin.metadata.license,
                    permissions=plugin.permissions,
                    dependencies=plugin.metadata.dependencies,
                    status=plugin.status.value,
                    installed=plugin.status != PluginStatus.AVAILABLE,
                    enabled=plugin.status == PluginStatus.ENABLED,
                    installed_at=plugin.installed_at,
                    last_updated=plugin.last_updated
                )
                
                # Apply filters
                if status and plugin_info.status != status:
                    continue
                if category and plugin_info.category != category:
                    continue
                
                plugin_list.append(plugin_info)
            
            return plugin_list
        else:
            # Mock implementation for testing
            return [
                PluginInfo(
                    id="file-operations",
                    name="File Operations",
                    version="1.0.0",
                    author="Xencode Team",
                    description="File system operations plugin",
                    status="installed",
                    installed=True,
                    enabled=True
                )
            ]
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list plugins: {e}")


@router.get("/{plugin_id}", response_model=PluginInfo)
async def get_plugin(
    plugin_id: str,
    plugin_manager = Depends(get_plugin_manager)
):
    """Get detailed information about a specific plugin"""
    try:
        if PLUGIN_COMPONENTS_AVAILABLE:
            plugin = await plugin_manager.get_plugin(plugin_id)
            if not plugin:
                raise HTTPException(status_code=404, detail="Plugin not found")
            
            return PluginInfo(
                id=plugin.metadata.name,
                name=plugin.metadata.name,
                version=plugin.metadata.version,
                author=plugin.metadata.author,
                description=plugin.metadata.description,
                license=plugin.metadata.license,
                permissions=plugin.permissions,
                dependencies=plugin.metadata.dependencies,
                status=plugin.status.value,
                installed=plugin.status != PluginStatus.AVAILABLE,
                enabled=plugin.status == PluginStatus.ENABLED,
                installed_at=plugin.installed_at,
                last_updated=plugin.last_updated,
                execution_count=plugin.execution_count,
                last_executed=plugin.last_executed,
                error_count=plugin.error_count
            )
        else:
            # Mock implementation
            if plugin_id == "file-operations":
                return PluginInfo(
                    id="file-operations",
                    name="File Operations",
                    version="1.0.0",
                    author="Xencode Team",
                    description="File system operations plugin",
                    status="installed",
                    installed=True,
                    enabled=True
                )
            else:
                raise HTTPException(status_code=404, detail="Plugin not found")
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plugin: {e}")


@router.post("/install")
async def install_plugin(
    request: PluginInstallRequest,
    background_tasks: BackgroundTasks,
    plugin_manager = Depends(get_plugin_manager)
):
    """Install a plugin from marketplace, file, or URL"""
    try:
        if not PLUGIN_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Plugin installation not available")
        
        # Start installation in background
        installation_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            install_plugin_background,
            plugin_manager,
            request,
            installation_id
        )
        
        return {
            "message": f"Plugin {request.plugin_id} installation started",
            "installation_id": installation_id,
            "status": "installing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start plugin installation: {e}")


@router.post("/upload")
async def upload_plugin(
    file: UploadFile = File(...),
    auto_enable: bool = Form(True),
    verify_signature: bool = Form(True),
    plugin_manager = Depends(get_plugin_manager)
):
    """Upload and install a plugin from file"""
    try:
        if not PLUGIN_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Plugin upload not available")
        
        # Validate file type
        if not file.filename.endswith(('.zip', '.tar.gz', '.xencode')):
            raise HTTPException(status_code=400, detail="Invalid plugin file format")
        
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{file.filename}")
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Install plugin
        result = await plugin_manager.install_from_file(
            temp_path,
            verify_signature=verify_signature,
            auto_enable=auto_enable
        )
        
        # Clean up temp file
        temp_path.unlink(missing_ok=True)
        
        return {
            "message": f"Plugin {result.name} installed successfully",
            "plugin_id": result.name,
            "version": result.version,
            "status": "installed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload plugin: {e}")


@router.put("/{plugin_id}")
async def update_plugin(
    plugin_id: str,
    request: PluginUpdateRequest,
    background_tasks: BackgroundTasks,
    plugin_manager = Depends(get_plugin_manager)
):
    """Update a plugin to a new version"""
    try:
        if not PLUGIN_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Plugin update not available")
        
        plugin = await plugin_manager.get_plugin(plugin_id)
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")
        
        # Start update in background
        update_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            update_plugin_background,
            plugin_manager,
            plugin_id,
            request,
            update_id
        )
        
        return {
            "message": f"Plugin {plugin_id} update started",
            "update_id": update_id,
            "current_version": plugin.metadata.version,
            "target_version": request.version or "latest"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update plugin: {e}")


@router.post("/{plugin_id}/enable")
async def enable_plugin(
    plugin_id: str,
    plugin_manager = Depends(get_plugin_manager)
):
    """Enable a plugin"""
    try:
        if not PLUGIN_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Plugin management not available")
        
        success = await plugin_manager.enable_plugin(plugin_id)
        if not success:
            raise HTTPException(status_code=404, detail="Plugin not found or cannot be enabled")
        
        return {
            "message": f"Plugin {plugin_id} enabled successfully",
            "plugin_id": plugin_id,
            "status": "enabled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable plugin: {e}")


@router.post("/{plugin_id}/disable")
async def disable_plugin(
    plugin_id: str,
    plugin_manager = Depends(get_plugin_manager)
):
    """Disable a plugin"""
    try:
        if not PLUGIN_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Plugin management not available")
        
        success = await plugin_manager.disable_plugin(plugin_id)
        if not success:
            raise HTTPException(status_code=404, detail="Plugin not found or cannot be disabled")
        
        return {
            "message": f"Plugin {plugin_id} disabled successfully",
            "plugin_id": plugin_id,
            "status": "disabled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable plugin: {e}")


@router.delete("/{plugin_id}")
async def uninstall_plugin(
    plugin_id: str,
    force: bool = False,
    plugin_manager = Depends(get_plugin_manager)
):
    """Uninstall a plugin"""
    try:
        if not PLUGIN_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Plugin management not available")
        
        success = await plugin_manager.uninstall_plugin(plugin_id, force=force)
        if not success:
            raise HTTPException(status_code=404, detail="Plugin not found or cannot be uninstalled")
        
        return {
            "message": f"Plugin {plugin_id} uninstalled successfully",
            "plugin_id": plugin_id,
            "status": "uninstalled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to uninstall plugin: {e}")


@router.post("/{plugin_id}/execute", response_model=PluginExecutionResult)
async def execute_plugin(
    plugin_id: str,
    request: PluginExecuteRequest,
    plugin_manager = Depends(get_plugin_manager)
):
    """Execute a plugin method"""
    try:
        if not PLUGIN_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Plugin execution not available")
        
        start_time = datetime.now()
        
        # Execute plugin method
        result = await plugin_manager.execute_plugin(
            plugin_id,
            request.method,
            *request.args,
            timeout=request.timeout_seconds,
            **request.kwargs
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get approximate memory usage (this is a simplified approach)
        import psutil
        current_process = psutil.Process()
        memory_used_mb = current_process.memory_info().rss / 1024 / 1024

        return PluginExecutionResult(
            success=True,
            result=result,
            execution_time_ms=int(execution_time),
            memory_used_mb=memory_used_mb,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PluginExecutionResult(
            success=False,
            error=str(e),
            execution_time_ms=int(execution_time),
            memory_used_mb=0.0,
            timestamp=datetime.now()
        )


@router.get("/{plugin_id}/config")
async def get_plugin_config(
    plugin_id: str,
    plugin_manager = Depends(get_plugin_manager)
):
    """Get plugin configuration"""
    try:
        if not PLUGIN_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Plugin management not available")
        
        config = await plugin_manager.get_plugin_config(plugin_id)
        if config is None:
            raise HTTPException(status_code=404, detail="Plugin not found")
        
        return {
            "plugin_id": plugin_id,
            "config": config,
            "last_updated": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plugin config: {e}")


@router.put("/{plugin_id}/config")
async def update_plugin_config(
    plugin_id: str,
    request: PluginConfigUpdate,
    plugin_manager = Depends(get_plugin_manager)
):
    """Update plugin configuration"""
    try:
        if not PLUGIN_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Plugin management not available")
        
        success = await plugin_manager.update_plugin_config(
            plugin_id,
            request.config,
            restart_if_needed=request.restart_required
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Plugin not found")
        
        return {
            "message": f"Plugin {plugin_id} configuration updated",
            "plugin_id": plugin_id,
            "restart_required": request.restart_required
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update plugin config: {e}")


@router.get("/{plugin_id}/stats", response_model=PluginStats)
async def get_plugin_stats(
    plugin_id: str,
    plugin_manager = Depends(get_plugin_manager)
):
    """Get plugin execution statistics"""
    try:
        if not PLUGIN_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Plugin management not available")
        
        stats = await plugin_manager.get_plugin_stats(plugin_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Plugin not found")
        
        return PluginStats(
            plugin_id=plugin_id,
            total_executions=stats.get('total_executions', 0),
            successful_executions=stats.get('successful_executions', 0),
            failed_executions=stats.get('failed_executions', 0),
            average_execution_time_ms=stats.get('avg_execution_time_ms', 0.0),
            total_memory_used_mb=stats.get('total_memory_mb', 0.0),
            uptime_hours=stats.get('uptime_hours', 0.0),
            last_24h_executions=stats.get('last_24h_executions', 0),
            error_rate_percent=stats.get('error_rate_percent', 0.0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plugin stats: {e}")


@router.get("/{plugin_id}/logs")
async def get_plugin_logs(
    plugin_id: str,
    lines: int = 100,
    level: str = "INFO",
    plugin_manager = Depends(get_plugin_manager)
):
    """Get plugin logs"""
    try:
        if not PLUGIN_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Plugin management not available")
        
        logs = await plugin_manager.get_plugin_logs(plugin_id, lines=lines, level=level)
        
        return {
            "plugin_id": plugin_id,
            "logs": logs,
            "lines_returned": len(logs),
            "level": level
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plugin logs: {e}")


# Marketplace endpoints
@router.get("/marketplace/info", response_model=MarketplaceInfo)
async def get_marketplace_info(
    marketplace_client = Depends(get_marketplace_client)
):
    """Get marketplace information"""
    try:
        async with marketplace_client as client:
            info = await client.get_marketplace_info()
            
            return MarketplaceInfo(
                total_plugins=info.get('total_plugins', 0),
                categories=info.get('categories', []),
                featured_plugins=info.get('featured_plugins', []),
                recent_updates=info.get('recent_updates', []),
                marketplace_status=info.get('status', 'online')
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get marketplace info: {e}")


@router.post("/marketplace/search")
async def search_marketplace(
    request: PluginSearchRequest,
    marketplace_client = Depends(get_marketplace_client)
):
    """Search plugins in marketplace"""
    try:
        async with marketplace_client as client:
            results = await client.search_plugins(
                query=request.query,
                tags=request.tags,
                category=request.category,
                sort_by=request.sort_by,
                limit=request.limit,
                offset=request.offset
            )
            
            # Convert to API format
            plugins = []
            for result in results:
                plugin_info = PluginInfo(
                    id=result['id'],
                    name=result['name'],
                    version=result['version'],
                    author=result['author'],
                    description=result['description'],
                    category=result.get('category', 'general'),
                    tags=result.get('tags', []),
                    downloads=result.get('downloads', 0),
                    rating=result.get('rating', 0.0),
                    reviews_count=result.get('reviews_count', 0),
                    marketplace_url=result.get('url'),
                    status="available"
                )
                plugins.append(plugin_info)
            
            return {
                "plugins": plugins,
                "total_results": len(plugins),
                "query": request.query,
                "has_more": len(plugins) == request.limit
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search marketplace: {e}")


@router.get("/marketplace/categories")
async def get_marketplace_categories(
    marketplace_client = Depends(get_marketplace_client)
):
    """Get available plugin categories"""
    try:
        async with marketplace_client as client:
            categories = await client.get_categories()
            
            return {
                "categories": categories,
                "total_count": len(categories)
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {e}")


@router.get("/system/status")
async def get_plugin_system_status(
    plugin_manager = Depends(get_plugin_manager)
):
    """Get plugin system status"""
    try:
        if not PLUGIN_COMPONENTS_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "Plugin system components not available"
            }
        
        status = await plugin_manager.get_system_status()
        
        return {
            "status": "healthy",
            "total_plugins": status.get('total_plugins', 0),
            "enabled_plugins": status.get('enabled_plugins', 0),
            "disabled_plugins": status.get('disabled_plugins', 0),
            "failed_plugins": status.get('failed_plugins', 0),
            "memory_usage_mb": status.get('memory_usage_mb', 0.0),
            "uptime_hours": status.get('uptime_hours', 0.0),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")


# Background tasks
async def install_plugin_background(
    plugin_manager,
    request: PluginInstallRequest,
    installation_id: str
):
    """Background task for plugin installation"""
    try:
        if request.source == "marketplace":
            await plugin_manager.install_from_marketplace(
                request.plugin_id,
                version=request.version,
                verify_signature=request.verify_signature,
                auto_enable=request.auto_enable
            )
        elif request.source == "url" and request.url:
            await plugin_manager.install_from_url(
                request.url,
                verify_signature=request.verify_signature,
                auto_enable=request.auto_enable
            )
        else:
            raise ValueError(f"Unsupported installation source: {request.source}")

        # Update installation status in database/cache
        try:
            # In a real implementation, this would update the database
            print(f"Plugin installation status updated for {request.plugin_id}")
        except Exception as status_error:
            print(f"Warning: Could not update installation status: {status_error}")

    except Exception as e:
        # Log installation error
        import logging
        logging.error(f"Plugin installation failed: {e}", exc_info=True)
        print(f"Plugin installation failed: {e}")


async def update_plugin_background(
    plugin_manager,
    plugin_id: str,
    request: PluginUpdateRequest,
    update_id: str
):
    """Background task for plugin update"""
    try:
        await plugin_manager.update_plugin(
            plugin_id,
            version=request.version,
            auto_restart=request.auto_restart
        )

        # Update status in database/cache
        try:
            # In a real implementation, this would update the database
            print(f"Plugin update status updated for {plugin_id}")
        except Exception as status_error:
            print(f"Warning: Could not update plugin status: {status_error}")

    except Exception as e:
        # Log update error
        import logging
        logging.error(f"Plugin update failed: {e}", exc_info=True)
        print(f"Plugin update failed: {e}")


router.tags = ["Plugins"]