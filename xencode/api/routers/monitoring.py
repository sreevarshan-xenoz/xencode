#!/usr/bin/env python3
"""
Monitoring API Router

FastAPI router for monitoring, performance metrics, and resource management endpoints.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field

# Import monitoring components
try:
    from ...monitoring.resource_manager import get_resource_manager, ResourceType, CleanupPriority
    from ...monitoring.performance_optimizer import PerformanceOptimizer
    from ...performance_monitoring_dashboard import PerformanceMonitoringDashboard
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

router = APIRouter()


# Pydantic models for API responses
class ResourceUsageResponse(BaseModel):
    """Resource usage information"""
    resource_type: str
    current_usage: float
    peak_usage: float
    unit: str
    timestamp: datetime
    limit_soft: Optional[float] = None
    limit_hard: Optional[float] = None


class CleanupResultResponse(BaseModel):
    """Cleanup operation result"""
    tasks_executed: int
    tasks_successful: int
    memory_freed_mb: float
    errors: List[str]
    timestamp: datetime


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    response_time_avg: float
    cache_hit_rate: float
    active_connections: int


class SystemHealthResponse(BaseModel):
    """System health status"""
    status: str = Field(..., description="Overall system health status")
    timestamp: datetime
    uptime_seconds: float
    components: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]


class ResourceLimitRequest(BaseModel):
    """Request to update resource limits"""
    resource_type: str
    soft_limit: float
    hard_limit: float
    unit: str = "percentage"
    enabled: bool = True


# Dependency to get resource manager
async def get_resource_manager_dep():
    """Dependency to get resource manager"""
    if not MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Monitoring components not available")
    
    try:
        return await get_resource_manager()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get resource manager: {e}")


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    resource_manager = Depends(get_resource_manager_dep)
):
    """Get comprehensive system health status"""
    try:
        # Get resource usage
        resource_usage = await resource_manager.get_resource_usage()
        
        # Check for violations
        violations = await resource_manager.check_resource_limits()
        
        # Get statistics
        stats = resource_manager.get_statistics()
        
        # Determine overall health status
        status = "healthy"
        if violations:
            critical_violations = [v for v in violations if v.current_usage >= v.limit.hard_limit]
            if critical_violations:
                status = "critical"
            else:
                status = "warning"
        
        # Generate recommendations
        recommendations = []
        if violations:
            for violation in violations:
                recommendations.append(
                    f"Resource {violation.resource_type.value} is at {violation.current_usage:.1f}% "
                    f"(limit: {violation.limit.soft_limit:.1f}%/{violation.limit.hard_limit:.1f}%)"
                )
        
        if stats["cleanup_stats"]["total_cleanups"] == 0:
            recommendations.append("Consider running cleanup operations to optimize performance")
        
        return SystemHealthResponse(
            status=status,
            timestamp=datetime.now(),
            uptime_seconds=0,  # TODO: Calculate actual uptime
            components={
                "resource_manager": "healthy",
                "memory_tracker": "healthy" if resource_manager.memory_tracker.tracking_enabled else "disabled",
                "gc_manager": "healthy",
                "temp_file_manager": "healthy"
            },
            alerts=[
                {
                    "type": "resource_violation",
                    "resource": v.resource_type.value,
                    "current": v.current_usage,
                    "limit": v.limit.hard_limit,
                    "severity": "critical" if v.current_usage >= v.limit.hard_limit else "warning"
                }
                for v in violations
            ],
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {e}")


@router.get("/resources", response_model=List[ResourceUsageResponse])
async def get_resource_usage(
    resource_manager = Depends(get_resource_manager_dep)
):
    """Get current resource usage for all monitored resources"""
    try:
        usage_data = await resource_manager.get_resource_usage()
        
        return [
            ResourceUsageResponse(
                resource_type=resource_type.value,
                current_usage=usage.current_usage,
                peak_usage=usage.peak_usage,
                unit=usage.limit.unit if usage.limit else "unknown",
                timestamp=usage.timestamp,
                limit_soft=usage.limit.soft_limit if usage.limit else None,
                limit_hard=usage.limit.hard_limit if usage.limit else None
            )
            for resource_type, usage in usage_data.items()
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get resource usage: {e}")


@router.get("/resources/{resource_type}")
async def get_specific_resource_usage(
    resource_type: str,
    resource_manager = Depends(get_resource_manager_dep)
):
    """Get usage information for a specific resource type"""
    try:
        # Convert string to ResourceType enum
        try:
            resource_enum = ResourceType(resource_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid resource type: {resource_type}")
        
        usage_data = await resource_manager.get_resource_usage()
        
        if resource_enum not in usage_data:
            raise HTTPException(status_code=404, detail=f"Resource type {resource_type} not found")
        
        usage = usage_data[resource_enum]
        
        return ResourceUsageResponse(
            resource_type=resource_enum.value,
            current_usage=usage.current_usage,
            peak_usage=usage.peak_usage,
            unit=usage.limit.unit if usage.limit else "unknown",
            timestamp=usage.timestamp,
            limit_soft=usage.limit.soft_limit if usage.limit else None,
            limit_hard=usage.limit.hard_limit if usage.limit else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get resource usage: {e}")


@router.post("/cleanup", response_model=CleanupResultResponse)
async def trigger_cleanup(
    priority: str = Query("medium", description="Cleanup priority: low, medium, high, critical"),
    background_tasks: BackgroundTasks = None,
    resource_manager = Depends(get_resource_manager_dep)
):
    """Trigger resource cleanup operations"""
    try:
        # Convert string to CleanupPriority enum
        try:
            priority_enum = CleanupPriority(priority.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")
        
        # Trigger cleanup
        results = await resource_manager.trigger_cleanup(priority_enum)
        
        return CleanupResultResponse(
            tasks_executed=results["tasks_executed"],
            tasks_successful=results["tasks_successful"],
            memory_freed_mb=results["memory_freed_mb"],
            errors=results["errors"],
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger cleanup: {e}")


@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    resource_manager = Depends(get_resource_manager_dep)
):
    """Get current performance metrics"""
    try:
        # Get resource usage
        usage_data = await resource_manager.get_resource_usage()
        
        # Extract key metrics
        memory_usage = usage_data.get(ResourceType.MEMORY)
        
        # Get cache statistics if available
        cache_hit_rate = 0.0
        try:
            from ...cache.multimodal_cache import get_multimodal_cache
            cache_system = await get_multimodal_cache()
            cache_stats = await cache_system.get_cache_statistics()
            cache_hit_rate = cache_stats.get("base_cache", {}).get("hit_rate", 0.0)
        except:
            pass
        
        return PerformanceMetricsResponse(
            timestamp=datetime.now(),
            cpu_usage=0.0,  # TODO: Get actual CPU usage
            memory_usage=memory_usage.current_usage if memory_usage else 0.0,
            disk_usage=usage_data.get(ResourceType.TEMPORARY_FILES, {}).current_usage or 0.0,
            response_time_avg=0.0,  # TODO: Calculate from request logs
            cache_hit_rate=cache_hit_rate,
            active_connections=0  # TODO: Track active connections
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {e}")


@router.get("/statistics")
async def get_monitoring_statistics(
    resource_manager = Depends(get_resource_manager_dep)
):
    """Get comprehensive monitoring statistics"""
    try:
        stats = resource_manager.get_statistics()
        return {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {e}")


@router.post("/limits/{resource_type}")
async def update_resource_limit(
    resource_type: str,
    limit_request: ResourceLimitRequest,
    resource_manager = Depends(get_resource_manager_dep)
):
    """Update resource limits for a specific resource type"""
    try:
        # Convert string to ResourceType enum
        try:
            resource_enum = ResourceType(resource_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid resource type: {resource_type}")
        
        # Create new resource limit
        from ...monitoring.resource_manager import ResourceLimit
        
        new_limit = ResourceLimit(
            resource_type=resource_enum,
            soft_limit=limit_request.soft_limit,
            hard_limit=limit_request.hard_limit,
            unit=limit_request.unit,
            enabled=limit_request.enabled
        )
        
        # Update the limit
        resource_manager.resource_limits[resource_enum] = new_limit
        
        return {
            "message": f"Resource limit updated for {resource_type}",
            "resource_type": resource_type,
            "soft_limit": limit_request.soft_limit,
            "hard_limit": limit_request.hard_limit,
            "unit": limit_request.unit,
            "enabled": limit_request.enabled,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update resource limit: {e}")


@router.get("/alerts")
async def get_active_alerts(
    resource_manager = Depends(get_resource_manager_dep)
):
    """Get active monitoring alerts"""
    try:
        # Check for resource violations
        violations = await resource_manager.check_resource_limits()
        
        alerts = []
        for violation in violations:
            severity = "critical" if violation.current_usage >= violation.limit.hard_limit else "warning"
            
            alerts.append({
                "id": f"{violation.resource_type.value}_{severity}",
                "type": "resource_violation",
                "severity": severity,
                "resource_type": violation.resource_type.value,
                "current_usage": violation.current_usage,
                "limit": violation.limit.hard_limit if severity == "critical" else violation.limit.soft_limit,
                "unit": violation.limit.unit,
                "timestamp": violation.timestamp.isoformat(),
                "message": f"Resource {violation.resource_type.value} usage is {violation.current_usage:.1f}{violation.limit.unit}"
            })
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "critical_count": len([a for a in alerts if a["severity"] == "critical"]),
            "warning_count": len([a for a in alerts if a["severity"] == "warning"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {e}")


@router.post("/memory/snapshot")
async def take_memory_snapshot(
    label: str = Query("api_request", description="Label for the memory snapshot"),
    resource_manager = Depends(get_resource_manager_dep)
):
    """Take a memory snapshot for analysis"""
    try:
        snapshot = resource_manager.memory_tracker.take_snapshot(label)
        
        if snapshot is None:
            raise HTTPException(status_code=400, detail="Memory tracking is not enabled")
        
        return {
            "message": "Memory snapshot taken successfully",
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "snapshot_count": len(resource_manager.memory_tracker.snapshots)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to take memory snapshot: {e}")


@router.get("/memory/analysis")
async def get_memory_analysis(
    resource_manager = Depends(get_resource_manager_dep)
):
    """Get memory growth analysis"""
    try:
        analysis = resource_manager.memory_tracker.analyze_memory_growth()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "current_usage": resource_manager.memory_tracker.get_current_memory_usage()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory analysis: {e}")


# Add router tags and metadata
router.tags = ["Monitoring"]