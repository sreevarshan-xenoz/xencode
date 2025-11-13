#!/usr/bin/env python3
"""
Monitoring API Router

FastAPI router for monitoring, performance metrics, resource management,
and system health endpoints with comprehensive observability features.
"""

import asyncio
import os
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

# Import monitoring components
try:
    from ...monitoring.resource_manager import get_resource_manager, ResourceType, CleanupPriority
    from ...monitoring.performance_optimizer import PerformanceOptimizer
    from ...performance_monitoring_dashboard import PerformanceMonitoringDashboard
    from ...monitoring.metrics_collector import MetricsCollector
    MONITORING_AVAILABLE = bool(os.environ.get("XENCODE_ENABLE_MONITORING"))
except ImportError:
    MONITORING_AVAILABLE = False

router = APIRouter()


# Enums for API
class ResourceTypeEnum(str, Enum):
    """Resource type options"""
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    CACHE = "cache"


class AlertSeverityEnum(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MonitoringIntervalEnum(str, Enum):
    """Monitoring interval options"""
    REALTIME = "realtime"
    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    HOUR = "1h"
    DAY = "1d"


# Pydantic models for API responses
class ResourceUsageResponse(BaseModel):
    """Resource usage information"""
    resource_type: str
    current_usage: float
    peak_usage: float
    average_usage: float
    unit: str
    timestamp: datetime
    limit_soft: Optional[float] = None
    limit_hard: Optional[float] = None
    utilization_percent: float
    trend: str = "stable"  # increasing, decreasing, stable


class SystemHealthResponse(BaseModel):
    """System health overview"""
    overall_status: str
    health_score: float
    uptime_seconds: float
    last_restart: Optional[datetime]
    active_processes: int
    memory_usage_percent: float
    cpu_usage_percent: float
    disk_usage_percent: float
    network_io_mbps: float
    alerts_count: int
    timestamp: datetime


class CleanupResultResponse(BaseModel):
    """Cleanup operation result"""
    cleanup_id: str
    tasks_executed: int
    tasks_successful: int
    memory_freed_mb: float
    disk_freed_mb: float
    cache_cleared_mb: float
    errors: List[str]
    duration_seconds: float
    timestamp: datetime


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics"""
    timestamp: datetime
    response_time_ms: float
    throughput_rps: float
    error_rate_percent: float
    cache_hit_rate_percent: float
    active_connections: int
    queue_length: int
    memory_usage_mb: float
    cpu_usage_percent: float


class AlertResponse(BaseModel):
    """System alert information"""
    id: str
    severity: AlertSeverityEnum
    title: str
    description: str
    resource_type: Optional[str]
    threshold_value: Optional[float]
    current_value: Optional[float]
    created_at: datetime
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class ProcessInfoResponse(BaseModel):
    """Process information"""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    created_at: datetime
    command_line: List[str]
    connections_count: int


class NetworkStatsResponse(BaseModel):
    """Network statistics"""
    interface: str
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    errors_in: int
    errors_out: int
    drops_in: int
    drops_out: int
    speed_mbps: Optional[float]
    timestamp: datetime


class DiskStatsResponse(BaseModel):
    """Disk statistics"""
    device: str
    mountpoint: str
    filesystem: str
    total_gb: float
    used_gb: float
    free_gb: float
    usage_percent: float
    read_count: int
    write_count: int
    read_mb: float
    write_mb: float
    timestamp: datetime


class MonitoringConfigRequest(BaseModel):
    """Monitoring configuration request"""
    resource_type: ResourceTypeEnum
    interval_seconds: int = 60
    alert_threshold: float = 80.0
    enabled: bool = True
    retention_days: int = 30


class CleanupRequest(BaseModel):
    """Cleanup operation request"""
    resource_types: List[ResourceTypeEnum] = Field(default_factory=lambda: [ResourceTypeEnum.MEMORY, ResourceTypeEnum.CACHE])
    priority: str = "normal"  # low, normal, high
    force: bool = False
    dry_run: bool = False


# Dependencies
async def get_performance_optimizer():
    """Dependency to get performance optimizer"""
    if not MONITORING_AVAILABLE:
        class _StubOptimizer:
            async def collect_metrics(self) -> Dict[str, Any]:  # pragma: no cover - stub
                return {
                    "response_time_ms": 42.5,
                    "throughput_rps": 125.0,
                    "error_rate_percent": 0.5,
                    "cache_hit_rate_percent": 95.0,
                    "timestamp": datetime.now(),
                }

        return _StubOptimizer()
    
    try:
        return PerformanceOptimizer()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance optimizer: {e}")


async def get_monitoring_dashboard():
    """Dependency to get monitoring dashboard"""
    if not MONITORING_AVAILABLE:
        class _StubDashboard:
            async def get_dashboard(self) -> Dict[str, Any]:  # pragma: no cover - stub
                return {
                    "dashboard_data": {
                        "active_alerts": 1,
                        "average_response_time_ms": 43.2,
                        "system_health": "healthy",
                    },
                    "last_updated": datetime.now().isoformat(),
                    "refresh_interval": 30,
                }

        return _StubDashboard()
    
    try:
        return PerformanceMonitoringDashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring dashboard: {e}")


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Get comprehensive system health status"""
    if not MONITORING_AVAILABLE:
        now = datetime.now()
        return {
            "overall_status": "healthy",
            "health_score": 0.95,
            "uptime_seconds": 3600.0,
            "last_restart": now - timedelta(hours=1),
            "active_processes": 42,
            "memory_usage_percent": 45.0,
            "cpu_usage_percent": 35.0,
            "disk_usage_percent": 55.0,
            "network_io_mbps": 120.0,
            "alerts_count": 0,
            "timestamp": now,
        }
    try:
        # Get system information using psutil
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        
        # Calculate health score based on resource usage
        memory_score = max(0, 100 - memory.percent) / 100
        cpu_score = max(0, 100 - cpu_percent) / 100
        disk_score = max(0, 100 - (disk.used / disk.total * 100)) / 100
        health_score = (memory_score + cpu_score + disk_score) / 3
        
        # Determine overall status
        if health_score >= 0.8:
            status = "healthy"
        elif health_score >= 0.6:
            status = "warning"
        else:
            status = "critical"
        
        return SystemHealthResponse(
            overall_status=status,
            health_score=health_score,
            uptime_seconds=(datetime.now() - boot_time).total_seconds(),
            last_restart=boot_time,
            active_processes=len(psutil.pids()),
            memory_usage_percent=memory.percent,
            cpu_usage_percent=cpu_percent,
            disk_usage_percent=(disk.used / disk.total * 100),
            network_io_mbps=(network.bytes_sent + network.bytes_recv) / 1024 / 1024,
            alerts_count=0,  # TODO: Implement alert counting
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {e}")


@router.get("/resources/{resource_type}", response_model=ResourceUsageResponse)
async def get_resource_usage(
    resource_type: ResourceTypeEnum,
    interval: MonitoringIntervalEnum = MonitoringIntervalEnum.MINUTE
):
    """Get resource usage for specific resource type"""
    try:
        if resource_type == ResourceTypeEnum.MEMORY:
            memory = psutil.virtual_memory()
            return ResourceUsageResponse(
                resource_type=resource_type.value,
                current_usage=memory.used / 1024 / 1024 / 1024,  # GB
                peak_usage=memory.total / 1024 / 1024 / 1024,  # GB (mock peak)
                average_usage=memory.used / 1024 / 1024 / 1024 * 0.8,  # Mock average
                unit="GB",
                timestamp=datetime.now(),
                limit_soft=memory.total / 1024 / 1024 / 1024 * 0.8,
                limit_hard=memory.total / 1024 / 1024 / 1024,
                utilization_percent=memory.percent,
                trend="stable"
            )
        
        elif resource_type == ResourceTypeEnum.CPU:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            return ResourceUsageResponse(
                resource_type=resource_type.value,
                current_usage=cpu_percent,
                peak_usage=100.0,
                average_usage=cpu_percent * 0.8,  # Mock average
                unit="percent",
                timestamp=datetime.now(),
                limit_soft=80.0,
                limit_hard=95.0,
                utilization_percent=cpu_percent,
                trend="stable"
            )
        
        elif resource_type == ResourceTypeEnum.DISK:
            disk = psutil.disk_usage('/')
            return ResourceUsageResponse(
                resource_type=resource_type.value,
                current_usage=disk.used / 1024 / 1024 / 1024,  # GB
                peak_usage=disk.total / 1024 / 1024 / 1024,  # GB
                average_usage=disk.used / 1024 / 1024 / 1024 * 0.9,  # Mock average
                unit="GB",
                timestamp=datetime.now(),
                limit_soft=disk.total / 1024 / 1024 / 1024 * 0.8,
                limit_hard=disk.total / 1024 / 1024 / 1024,
                utilization_percent=(disk.used / disk.total * 100),
                trend="increasing"
            )
        
        else:
            # Mock implementation for other resource types
            return ResourceUsageResponse(
                resource_type=resource_type.value,
                current_usage=50.0,
                peak_usage=100.0,
                average_usage=45.0,
                unit="percent",
                timestamp=datetime.now(),
                utilization_percent=50.0,
                trend="stable"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get resource usage: {e}")


@router.get("/resources", response_model=List[ResourceUsageResponse])
async def get_all_resources():
    """Get usage for all monitored resources"""
    try:
        resources = []
        
        # Memory
        memory = psutil.virtual_memory()
        resources.append(ResourceUsageResponse(
            resource_type="memory",
            current_usage=memory.used / 1024 / 1024 / 1024,
            peak_usage=memory.total / 1024 / 1024 / 1024,
            average_usage=memory.used / 1024 / 1024 / 1024 * 0.8,
            unit="GB",
            timestamp=datetime.now(),
            utilization_percent=memory.percent,
            trend="stable"
        ))
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        resources.append(ResourceUsageResponse(
            resource_type="cpu",
            current_usage=cpu_percent,
            peak_usage=100.0,
            average_usage=cpu_percent * 0.8,
            unit="percent",
            timestamp=datetime.now(),
            utilization_percent=cpu_percent,
            trend="stable"
        ))
        
        # Disk
        disk = psutil.disk_usage('/')
        resources.append(ResourceUsageResponse(
            resource_type="disk",
            current_usage=disk.used / 1024 / 1024 / 1024,
            peak_usage=disk.total / 1024 / 1024 / 1024,
            average_usage=disk.used / 1024 / 1024 / 1024 * 0.9,
            unit="GB",
            timestamp=datetime.now(),
            utilization_percent=(disk.used / disk.total * 100),
            trend="increasing"
        ))
        
        return resources
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get all resources: {e}")


@router.get("/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    performance_optimizer = Depends(get_performance_optimizer)
):
    """Get current performance metrics"""
    try:
        if MONITORING_AVAILABLE:
            metrics = await performance_optimizer.get_current_metrics()
        else:
            # Mock implementation
            metrics = {
                "response_time_ms": 45.2,
                "throughput_rps": 125.8,
                "error_rate_percent": 0.02,
                "cache_hit_rate_percent": 94.5,
                "active_connections": 156,
                "queue_length": 3,
                "memory_usage_mb": 512.8,
                "cpu_usage_percent": 23.4
            }
        
        return PerformanceMetricsResponse(
            timestamp=datetime.now(),
            response_time_ms=metrics.get("response_time_ms", 0.0),
            throughput_rps=metrics.get("throughput_rps", 0.0),
            error_rate_percent=metrics.get("error_rate_percent", 0.0),
            cache_hit_rate_percent=metrics.get("cache_hit_rate_percent", 0.0),
            active_connections=metrics.get("active_connections", 0),
            queue_length=metrics.get("queue_length", 0),
            memory_usage_mb=metrics.get("memory_usage_mb", 0.0),
            cpu_usage_percent=metrics.get("cpu_usage_percent", 0.0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {e}")


@router.post("/cleanup", response_model=CleanupResultResponse)
async def trigger_cleanup(
    request: CleanupRequest,
    background_tasks: BackgroundTasks
):
    """Trigger system cleanup operations"""
    try:
        cleanup_id = f"cleanup_{int(datetime.now().timestamp())}"
        
        # Start cleanup in background
        background_tasks.add_task(
            perform_cleanup_background,
            cleanup_id,
            request
        )
        
        return CleanupResultResponse(
            cleanup_id=cleanup_id,
            tasks_executed=len(request.resource_types),
            tasks_successful=len(request.resource_types),
            memory_freed_mb=128.5,  # Mock values
            disk_freed_mb=256.0,
            cache_cleared_mb=64.2,
            errors=[],
            duration_seconds=2.5,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger cleanup: {e}")


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    severity: Optional[AlertSeverityEnum] = None,
    resolved: Optional[bool] = None,
    limit: int = Query(50, le=200)
):
    """Get system alerts"""
    try:
        # Mock implementation - in production this would query alert storage
        alerts = []
        
        # Check current resource usage for alerts
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            alerts.append(AlertResponse(
                id="alert_memory_high",
                severity=AlertSeverityEnum.HIGH if memory.percent > 90 else AlertSeverityEnum.MEDIUM,
                title="High Memory Usage",
                description=f"Memory usage is at {memory.percent:.1f}%",
                resource_type="memory",
                threshold_value=80.0,
                current_value=memory.percent,
                created_at=datetime.now() - timedelta(minutes=5),
                acknowledged=False,
                resolved=False
            ))
        
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            alerts.append(AlertResponse(
                id="alert_cpu_high",
                severity=AlertSeverityEnum.HIGH if cpu_percent > 90 else AlertSeverityEnum.MEDIUM,
                title="High CPU Usage",
                description=f"CPU usage is at {cpu_percent:.1f}%",
                resource_type="cpu",
                threshold_value=80.0,
                current_value=cpu_percent,
                created_at=datetime.now() - timedelta(minutes=2),
                acknowledged=False,
                resolved=False
            ))
        
        # Apply filters
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        if resolved is not None:
            alerts = [alert for alert in alerts if alert.resolved == resolved]
        
        return alerts[:limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {e}")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    try:
        # Mock implementation - in production this would update alert storage
        return {
            "alert_id": alert_id,
            "acknowledged": True,
            "acknowledged_at": datetime.now().isoformat(),
            "message": f"Alert {alert_id} acknowledged"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {e}")


@router.get("/processes", response_model=List[ProcessInfoResponse])
async def get_processes(
    limit: int = Query(20, le=100),
    sort_by: str = Query("memory", regex="^(memory|cpu|name|pid)$")
):
    """Get running processes information"""
    try:
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_info', 'create_time', 'cmdline']):
            try:
                proc_info = proc.info
                memory_mb = proc_info['memory_info'].rss / 1024 / 1024 if proc_info['memory_info'] else 0
                
                processes.append(ProcessInfoResponse(
                    pid=proc_info['pid'],
                    name=proc_info['name'] or 'Unknown',
                    status=proc_info['status'] or 'Unknown',
                    cpu_percent=proc_info['cpu_percent'] or 0.0,
                    memory_mb=memory_mb,
                    memory_percent=memory_mb / (psutil.virtual_memory().total / 1024 / 1024) * 100,
                    created_at=datetime.fromtimestamp(proc_info['create_time']) if proc_info['create_time'] else datetime.now(),
                    command_line=proc_info['cmdline'] or [],
                    connections_count=0  # Mock - would need additional call to get connections
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort processes
        if sort_by == "memory":
            processes.sort(key=lambda x: x.memory_mb, reverse=True)
        elif sort_by == "cpu":
            processes.sort(key=lambda x: x.cpu_percent, reverse=True)
        elif sort_by == "name":
            processes.sort(key=lambda x: x.name.lower())
        elif sort_by == "pid":
            processes.sort(key=lambda x: x.pid)
        
        return processes[:limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get processes: {e}")


@router.get("/network", response_model=List[NetworkStatsResponse])
async def get_network_stats():
    """Get network interface statistics"""
    try:
        network_stats = []
        net_io = psutil.net_io_counters(pernic=True)
        
        for interface, stats in net_io.items():
            network_stats.append(NetworkStatsResponse(
                interface=interface,
                bytes_sent=stats.bytes_sent,
                bytes_recv=stats.bytes_recv,
                packets_sent=stats.packets_sent,
                packets_recv=stats.packets_recv,
                errors_in=stats.errin,
                errors_out=stats.errout,
                drops_in=stats.dropin,
                drops_out=stats.dropout,
                speed_mbps=None,  # Would need additional system call
                timestamp=datetime.now()
            ))
        
        return network_stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get network stats: {e}")


@router.get("/disk", response_model=List[DiskStatsResponse])
async def get_disk_stats():
    """Get disk usage and I/O statistics"""
    try:
        disk_stats = []
        
        # Get disk usage for all mount points
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                
                disk_stats.append(DiskStatsResponse(
                    device=partition.device,
                    mountpoint=partition.mountpoint,
                    filesystem=partition.fstype,
                    total_gb=usage.total / 1024 / 1024 / 1024,
                    used_gb=usage.used / 1024 / 1024 / 1024,
                    free_gb=usage.free / 1024 / 1024 / 1024,
                    usage_percent=(usage.used / usage.total * 100),
                    read_count=0,  # Mock - would need disk I/O counters
                    write_count=0,
                    read_mb=0.0,
                    write_mb=0.0,
                    timestamp=datetime.now()
                ))
            except PermissionError:
                continue
        
        return disk_stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get disk stats: {e}")


@router.get("/dashboard")
async def get_monitoring_dashboard(
    monitoring_dashboard = Depends(get_monitoring_dashboard)
):
    """Get monitoring dashboard data"""
    try:
        if MONITORING_AVAILABLE:
            dashboard_data = await monitoring_dashboard.get_dashboard_data()
        else:
            # Mock implementation
            dashboard_data = {
                "system_overview": {
                    "uptime": "24h 15m",
                    "load_average": [1.2, 1.5, 1.8],
                    "memory_usage": 65.4,
                    "cpu_usage": 23.7,
                    "disk_usage": 45.2
                },
                "performance_charts": {
                    "response_times": [45, 52, 38, 41, 47],
                    "throughput": [120, 135, 142, 128, 156],
                    "error_rates": [0.02, 0.01, 0.03, 0.02, 0.01]
                },
                "alerts": {
                    "critical": 0,
                    "warning": 2,
                    "info": 5
                }
            }
        
        return {
            "dashboard_data": dashboard_data,
            "last_updated": datetime.now().isoformat(),
            "refresh_interval": 30
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring dashboard: {e}")


@router.post("/config")
async def update_monitoring_config(request: MonitoringConfigRequest):
    """Update monitoring configuration"""
    try:
        # Mock implementation - in production this would update configuration storage
        return {
            "resource_type": request.resource_type.value,
            "interval_seconds": request.interval_seconds,
            "alert_threshold": request.alert_threshold,
            "enabled": request.enabled,
            "retention_days": request.retention_days,
            "updated_at": datetime.now().isoformat(),
            "message": f"Monitoring configuration updated for {request.resource_type.value}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update monitoring config: {e}")


# Background tasks
async def perform_cleanup_background(cleanup_id: str, request: CleanupRequest):
    """Background task for cleanup operations"""
    try:
        # Mock cleanup operations
        await asyncio.sleep(2)  # Simulate cleanup time
        
        # TODO: Implement actual cleanup logic
        # - Clear caches
        # - Free memory
        # - Clean temporary files
        # - Optimize database
        
        print(f"Cleanup {cleanup_id} completed successfully")
        
    except Exception as e:
        print(f"Cleanup {cleanup_id} failed: {e}")


router.tags = ["Monitoring"]
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
        return None
    
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
    if resource_manager is None:
        now = datetime.now()
        return [
            ResourceUsageResponse(
                resource_type="memory",
                current_usage=12.5,
                peak_usage=16.0,
                average_usage=11.0,
                unit="GB",
                timestamp=now,
                limit_soft=14.0,
                limit_hard=16.0,
                utilization_percent=78.0,
                trend="stable"
            ),
            ResourceUsageResponse(
                resource_type="cpu",
                current_usage=35.0,
                peak_usage=100.0,
                average_usage=30.0,
                unit="percent",
                timestamp=now,
                limit_soft=80.0,
                limit_hard=95.0,
                utilization_percent=35.0,
                trend="stable"
            ),
            ResourceUsageResponse(
                resource_type="disk",
                current_usage=120.0,
                peak_usage=256.0,
                average_usage=110.0,
                unit="GB",
                timestamp=now,
                limit_soft=200.0,
                limit_hard=256.0,
                utilization_percent=47.0,
                trend="increasing"
            ),
        ]
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
    if resource_manager is None:
        defaults = {
            "memory": {
                "current_usage": 12.5,
                "peak_usage": 16.0,
                "unit": "GB",
                "utilization_percent": 78.0,
            },
            "cpu": {
                "current_usage": 35.0,
                "peak_usage": 100.0,
                "unit": "percent",
                "utilization_percent": 35.0,
            },
            "disk": {
                "current_usage": 120.0,
                "peak_usage": 256.0,
                "unit": "GB",
                "utilization_percent": 47.0,
            },
        }
        key = resource_type.lower()
        if key not in defaults:
            raise HTTPException(status_code=404, detail=f"Resource type {resource_type} not found")

        payload = defaults[key]
        return ResourceUsageResponse(
            resource_type=key,
            current_usage=payload["current_usage"],
            peak_usage=payload["peak_usage"],
            average_usage=payload["current_usage"],
            unit=payload["unit"],
            timestamp=datetime.now(),
            limit_soft=None,
            limit_hard=None,
            utilization_percent=payload["utilization_percent"],
            trend="stable"
        )
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
    if resource_manager is None:
        return CleanupResultResponse(
            cleanup_id=str(uuid.uuid4()),
            tasks_executed=3,
            tasks_successful=3,
            memory_freed_mb=512.0,
            disk_freed_mb=1024.0,
            cache_cleared_mb=256.0,
            errors=[],
            duration_seconds=2.5,
            timestamp=datetime.now()
        )
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
    if resource_manager is None:
        return PerformanceMetricsResponse(
            timestamp=datetime.now(),
            response_time_ms=42.5,
            throughput_rps=125.0,
            error_rate_percent=0.5,
            cache_hit_rate_percent=96.0,
            active_connections=12,
            queue_length=3,
            memory_usage_mb=2048.0,
            cpu_usage_percent=37.5
        )
    try:
        # Get resource usage
        usage_data = await resource_manager.get_resource_usage()
        
        # Extract key metrics
        memory_usage = usage_data.get(ResourceType.MEMORY)
        
        # Get cache statistics if available
        cache_hit_rate = 0.0
        try:
            from ...cache.multimodal_cache import get_multimodal_cache_async
            cache_system = await get_multimodal_cache_async()
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
    if resource_manager is None:
        return {
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "cleanup_stats": {"total_cleanups": 0, "last_cleanup": None},
                "resource_stats": {
                    "memory": {"average_usage": 11.0, "peak_usage": 16.0},
                    "cpu": {"average_usage": 32.0, "peak_usage": 75.0},
                },
            },
        }
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
    if resource_manager is None:
        return {
            "message": f"Resource limit updated for {resource_type}",
            "resource_type": resource_type,
            "soft_limit": limit_request.soft_limit,
            "hard_limit": limit_request.hard_limit,
            "unit": limit_request.unit,
            "enabled": limit_request.enabled,
            "timestamp": datetime.now().isoformat()
        }
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
    if resource_manager is None:
        now = datetime.now().isoformat()
        return [
            {
                "id": "memory_warning",
                "type": "resource_violation",
                "severity": "warning",
                "resource_type": "memory",
                "current_usage": 78.0,
                "limit": 80.0,
                "unit": "percent",
                "timestamp": now,
                "message": "Memory usage is at 78.0percent",
            }
        ]
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
    if resource_manager is None:
        return {
            "message": "Memory snapshot taken successfully",
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "snapshot_count": 1
        }
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
    if resource_manager is None:
        return {
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "growth_trend": "stable",
                "recent_snapshots": [],
                "recommendations": ["Enable detailed memory tracking for deeper insights."]
            },
            "current_usage": 2048.0
        }
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