#!/usr/bin/env python3
"""
Monitoring API Router

FastAPI router for monitoring, performance metrics, resource management,
and system health endpoints with comprehensive observability features.
"""

import asyncio
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

# Import monitoring components
try:
    from ...monitoring.performance_optimizer import PerformanceOptimizer
    from ...monitoring.resource_manager import (
        ResourceType,
        get_resource_manager,
    )
    from ...performance_monitoring_dashboard import PerformanceMonitoringDashboard
    MONITORING_AVAILABLE = bool(os.environ.get("XENCODE_ENABLE_MONITORING"))
except ImportError:
    MONITORING_AVAILABLE = False

# Import benchmark components
try:
    from ...monitoring.benchmark_engine import BenchmarkEngine
    from ...monitoring.benchmark_recommendations import get_recommendations_engine
    from ...monitoring.benchmark_store import BenchmarkStore
    from ...monitoring.benchmark_suites import get_benchmark_suites
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

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


# Benchmark Pydantic models
class BenchmarkRunRequest(BaseModel):
    """Request to run benchmark suite"""
    suite_name: str = "code_generation"
    providers: List[Dict[str, str]] = Field(default_factory=lambda: [{"provider": "ollama", "model": "llama3.2"}])
    concurrent: bool = True
    custom_tasks: Optional[List[Dict[str, Any]]] = None


class BenchmarkResultResponse(BaseModel):
    """Benchmark result"""
    run_id: str
    task_name: str
    task_type: str
    provider: str
    model: str
    latency_ms: float
    tokens_per_sec: float
    accuracy_score: float
    cost_per_request: float
    success: bool
    timestamp: str


class BenchmarkComparisonResponse(BaseModel):
    """Provider comparison"""
    task_type: str
    providers: List[Dict[str, Any]]
    best_provider: str
    best_model: str


class BenchmarkRecommendationResponse(BaseModel):
    """Model recommendation"""
    task_type: str
    use_case: str
    recommended_provider: str
    recommended_model: str
    confidence: float
    reasons: List[str]
    alternatives: List[Dict[str, Any]]
    tradeoffs: Dict[str, str]


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
        raise HTTPException(status_code=500, detail=f"Failed to get performance optimizer: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring dashboard: {e}")  from e


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
            alerts_count=1 if memory.percent > 90 else 0,  # Basic alert counting based on resource thresholds
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to get resource usage: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to get all resources: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to trigger cleanup: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {e}")  from e


@router.get("/processes", response_model=List[ProcessInfoResponse])
async def get_processes(
    limit: int = Query(20, le=100),
    sort_by: str = Query("memory", pattern="^(memory|cpu|name|pid)$")
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
        raise HTTPException(status_code=500, detail=f"Failed to get processes: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to get network stats: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to get disk stats: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring dashboard: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to update monitoring config: {e}")  from e


# Background tasks
async def perform_cleanup_background(cleanup_id: str, request: CleanupRequest):
    """Background task for cleanup operations"""
    try:
        # Actual cleanup operations
        await asyncio.sleep(1)  # Simulate cleanup time

        # Clear caches if available
        try:
            from ...cache.multimodal_cache import get_multimodal_cache_async
            cache_system = await get_multimodal_cache_async()
            if hasattr(cache_system, 'clear_expired'):
                await cache_system.clear_expired()
        except ImportError:
            pass  # Cache system not available
        except Exception:
            pass  # Ignore cache clearing errors

        # Clean temporary files
        import tempfile
        temp_dir = tempfile.gettempdir()
        try:
            # Clean up temporary files older than 1 day
            import os
            import time
            current_time = time.time()
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    # Remove files older than 1 day
                    if current_time - os.path.getmtime(file_path) > 86400:
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass  # Ignore file removal errors
        except Exception:
            pass  # Ignore temp file cleanup errors

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
        raise HTTPException(status_code=500, detail=f"Failed to get resource manager: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {e}")  from e


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
            raise HTTPException(status_code=400, detail=f"Invalid resource type: {resource_type}")  from None

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
        raise HTTPException(status_code=500, detail=f"Failed to update resource limit: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to take memory snapshot: {e}")  from e


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
        raise HTTPException(status_code=500, detail=f"Failed to get memory analysis: {e}")  from e


# ============================================================================
# Provider Health Endpoints
# ============================================================================

class ProviderHealthStatusEnum(str, Enum):
    """Provider health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    OFFLINE = "offline"


class ProviderHealthResponse(BaseModel):
    """Provider health information"""
    provider: str
    status: str
    endpoint: Optional[str] = None
    last_checked: Optional[str] = None
    last_success: Optional[str] = None
    uptime_percentage: float = 0.0
    latency: Dict[str, Any] = Field(default_factory=dict)
    errors: Dict[str, Any] = Field(default_factory=dict)
    usage: Dict[str, Any] = Field(default_factory=dict)
    model_count: int = 0
    available_models: List[str] = Field(default_factory=list)


class ProviderHealthSummaryResponse(BaseModel):
    """Provider health summary"""
    timestamp: str
    providers: Dict[str, ProviderHealthResponse]
    overall_status: str
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    recommended_provider: str


@router.get("/providers/health", response_model=ProviderHealthSummaryResponse, tags=["providers"])
async def get_provider_health_summary():
    """
    Get health summary for all configured providers
    
    Returns real-time health status including:
    - Provider availability
    - Response latency metrics
    - Error rates and tracking
    - Usage quotas and limits
    - Model availability
    """
    try:
        from ...monitoring.provider_health import get_health_monitor

        monitor = get_health_monitor()

        # Check all providers
        import asyncio

        from ...monitoring.provider_health import ProviderType

        async def check_all():
            tasks = [
                monitor.check_provider_health(provider)
                for provider in ProviderType
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        await check_all()

        # Get summary
        summary = monitor.get_health_summary()

        # Get recommendation
        recommended = monitor.get_recommended_provider()

        return ProviderHealthSummaryResponse(
            timestamp=summary['timestamp'],
            providers={
                k: ProviderHealthResponse(**v)
                for k, v in summary['providers'].items()
            },
            overall_status=summary['overall_status'],
            healthy_count=summary['healthy_count'],
            degraded_count=summary['degraded_count'],
            unhealthy_count=summary['unhealthy_count'],
            recommended_provider=recommended.value,
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Provider health monitoring not available"
        )  from None
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get provider health: {e}"
        )  from e


@router.get("/providers/{provider_name}/health", response_model=ProviderHealthResponse, tags=["providers"])
async def get_provider_health(provider_name: str):
    """
    Get detailed health information for a specific provider
    
    Args:
        provider_name: Provider identifier (e.g., 'local_ollama', 'cloud_qwen')
    """
    try:
        from ...monitoring.provider_health import ProviderType, get_health_monitor

        monitor = get_health_monitor()

        # Find provider
        try:
            provider = ProviderType(provider_name)
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown provider: {provider_name}"
            )  from None

        # Check health
        health = await monitor.check_provider_health(provider)

        return ProviderHealthResponse(
            provider=health.provider.value,
            status=health.status.value,
            endpoint=health.endpoint,
            last_checked=health.last_checked.isoformat() if health.last_checked else None,
            last_success=health.last_success.isoformat() if health.last_success else None,
            uptime_percentage=health.uptime_percentage,
            latency=health.latency.to_dict(),
            errors=health.errors.to_dict(),
            usage=health.usage.to_dict(),
            model_count=health.model_count,
            available_models=health.available_models,
        )

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Provider health monitoring not available"
        )  from None
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get provider health: {e}"
        )  from e


@router.post("/providers/{provider_name}/check", tags=["providers"])
async def check_provider(provider_name: str):
    """
    Force an immediate health check for a provider
    
    Args:
        provider_name: Provider identifier
    """
    try:
        from ...monitoring.provider_health import ProviderType, get_health_monitor

        monitor = get_health_monitor()

        try:
            provider = ProviderType(provider_name)
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown provider: {provider_name}"
            )  from None

        health = await monitor.check_provider_health(provider)

        return {
            "provider": health.provider.value,
            "status": health.status.value,
            "latency_ms": health.latency.current_ms,
            "error_rate": health.errors.error_rate,
            "checked_at": health.last_checked.isoformat() if health.last_checked else None,
        }

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Provider health monitoring not available"
        )  from None
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check provider: {e}"
        )  from e


@router.get("/providers/recommend", tags=["providers"])
async def get_recommended_provider(task_type: str = Query(default="general")):
    """
    Get recommended provider based on current health and task type
    
    Args:
        task_type: Type of task (general, code, chat, reasoning)
    """
    try:
        from ...monitoring.provider_health import get_health_monitor

        monitor = get_health_monitor()
        recommended = monitor.get_recommended_provider(task_type)

        health = monitor.get_provider_health(recommended)

        return {
            "provider": recommended.value,
            "status": health.status.value if health else "unknown",
            "reason": "Best combination of latency, error rate, and uptime",
            "latency_avg_ms": health.latency.avg_ms if health else None,
            "error_rate": health.errors.error_rate if health else None,
            "uptime_percentage": health.uptime_percentage if health else None,
        }

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Provider health monitoring not available"
        )  from None
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recommendation: {e}"
        )  from e


@router.get("/providers/latency/trends", tags=["providers"])
async def get_latency_trends():
    """
    Get latency trends for all providers
    
    Returns historical latency data and trend analysis.
    """
    try:
        from ...monitoring.provider_health import ProviderType, get_health_monitor

        monitor = get_health_monitor()

        trends = {}
        for provider in ProviderType:
            health = monitor.get_provider_health(provider)
            if health:
                trends[provider.value] = {
                    "current_ms": health.latency.current_ms,
                    "avg_ms": health.latency.avg_ms,
                    "min_ms": health.latency.min_ms,
                    "max_ms": health.latency.max_ms,
                    "p50_ms": health.latency.p50_ms,
                    "p95_ms": health.latency.p95_ms,
                    "trend": health.latency.get_trend(),
                    "sample_count": health.latency.sample_count,
                }

        return {
            "timestamp": datetime.now().isoformat(),
            "trends": trends,
        }

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Provider health monitoring not available"
        )  from None
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get latency trends: {e}"
        )  from e


@router.get("/providers/errors", tags=["providers"])
async def get_provider_errors(hours: int = Query(default=24, ge=1, le=168)):
    """
    Get error summary for all providers
    
    Args:
        hours: Number of hours to look back (1-168)
    """
    try:
        from ...monitoring.provider_health import ProviderType, get_health_monitor

        monitor = get_health_monitor()

        errors = {}
        for provider in ProviderType:
            health = monitor.get_provider_health(provider)
            if health and health.errors.total_errors > 0:
                errors[provider.value] = {
                    "total_errors": health.errors.total_errors,
                    "error_rate": health.errors.error_rate,
                    "error_types": health.errors.error_types,
                    "last_error_time": health.errors.last_error_time.isoformat() if health.errors.last_error_time else None,
                    "last_error_message": health.errors.last_error_message,
                    "consecutive_errors": health.errors.consecutive_errors,
                }

        return {
            "timestamp": datetime.now().isoformat(),
            "hours": hours,
            "errors": errors,
        }

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Provider health monitoring not available"
        )  from None
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get error summary: {e}"
        )  from e


# Benchmark Endpoints
@router.post("/benchmarks/run", response_model=Dict[str, Any], tags=["benchmarks"])
async def run_benchmark_suite(request: BenchmarkRunRequest):
    """
    Run a benchmark suite
    
    Executes benchmark tasks against specified providers and models.
    Results are stored for analysis and recommendations.
    """
    if not BENCHMARK_AVAILABLE:
        return {
            "error": "Benchmark module not available",
            "run_id": "mock_run",
            "status": "mock",
        }

    try:
        engine = BenchmarkEngine()
        suites = get_benchmark_suites()

        # Get tasks from suite
        tasks = suites.get_suite(request.suite_name)

        if not tasks:
            raise HTTPException(status_code=400, detail=f"Unknown suite: {request.suite_name}")

        # Add custom tasks if provided
        if request.custom_tasks:
            for custom in request.custom_tasks:
                task = suites.create_custom_task(
                    name=custom.get("name", "custom"),
                    task_type=custom.get("task_type", "general"),
                    prompt=custom.get("prompt", ""),
                    expected_output=custom.get("expected_output"),
                )
                tasks.append(task)

        # Parse providers
        providers = [(p["provider"], p["model"]) for p in request.providers]

        # Run benchmark
        summary = await engine.run_benchmark_suite(
            tasks=tasks,
            providers=providers,
            suite_name=request.suite_name,
            concurrent=request.concurrent,
        )

        # Close engine session
        await engine.close()

        return {
            "run_id": summary["run_id"],
            "status": "completed",
            "suite_name": summary["suite_name"],
            "total_tasks": summary["total_tasks"],
            "successful": summary["successful"],
            "failed": summary["failed"],
            "avg_latency_ms": summary.get("avg_latency_ms"),
            "avg_accuracy_score": summary.get("avg_accuracy_score"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {e}")  from e


@router.get("/benchmarks/results", response_model=List[BenchmarkResultResponse], tags=["benchmarks"])
async def get_benchmark_results(
    run_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    task_type: Optional[str] = None,
    limit: int = 100,
):
    """Get benchmark results with optional filters"""
    if not BENCHMARK_AVAILABLE:
        return []

    try:
        store = BenchmarkStore()

        if run_id:
            # Get results for specific run
            run = store.get_run(run_id)
            if not run:
                raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

            # Query would need run_id filter - simplified for now
            results = store.get_results_by_task_type("general", limit=limit)
        elif provider and model:
            results = store.get_results_by_model(provider, model, limit)
        elif provider:
            results = store.get_results_by_provider(provider, limit)
        elif task_type:
            results = store.get_results_by_task_type(task_type, limit)
        else:
            # Get recent results from all providers
            results = store.get_results_by_task_type("general", limit)

        return [
            BenchmarkResultResponse(
                run_id=r.get("run_id", ""),
                task_name=r.get("task_name", ""),
                task_type=r.get("task_type", ""),
                provider=r.get("provider", ""),
                model=r.get("model", ""),
                latency_ms=r.get("latency_ms", 0),
                tokens_per_sec=r.get("tokens_per_sec", 0),
                accuracy_score=r.get("accuracy_score", 0),
                cost_per_request=r.get("cost_per_request", 0),
                success=r.get("success", False),
                timestamp=r.get("timestamp", ""),
            )
            for r in results
        ]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get results: {e}")  from e


@router.get("/benchmarks/comparison", response_model=Dict[str, Any], tags=["benchmarks"])
async def compare_providers(
    task_type: Optional[str] = None,
):
    """Compare provider performance"""
    if not BENCHMARK_AVAILABLE:
        return {
            "task_type": task_type or "general",
            "providers": [],
            "best_provider": "ollama",
            "best_model": "llama3.2",
        }

    try:
        store = BenchmarkStore()
        recommendations = get_recommendations_engine(store)

        # Get analysis
        analysis = recommendations.get_cost_quality_analysis(task_type or "general")

        if "error" in analysis:
            return {
                "task_type": task_type or "general",
                "providers": [],
                "best_provider": "unknown",
                "best_model": "unknown",
                "message": analysis["error"],
            }

        # Format comparison
        providers = []
        for model in analysis.get("all_models", []):
            providers.append({
                "provider": model["provider"],
                "model": model["model"],
                "overall_score": model["overall_score"],
                "performance_score": model["performance_score"],
                "quality_score": model["quality_score"],
                "cost_score": model["cost_score"],
                "avg_latency_ms": model["avg_latency_ms"],
                "avg_accuracy": model["avg_accuracy"],
                "avg_cost": model["avg_cost_per_request"],
            })

        best = providers[0] if providers else {}

        return {
            "task_type": task_type or "general",
            "providers": providers,
            "best_provider": best.get("provider", "unknown"),
            "best_model": best.get("model", "unknown"),
            "pareto_optimal_count": analysis.get("pareto_optimal", 0),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")  from e


@router.get("/benchmarks/recommendations", response_model=BenchmarkRecommendationResponse, tags=["benchmarks"])
async def get_recommendations(
    task_type: str = "code_generation",
    use_case: str = "production",
    max_budget: Optional[float] = None,
    max_latency_ms: Optional[float] = None,
    min_accuracy: Optional[float] = None,
):
    """Get model recommendations based on benchmarks"""
    if not BENCHMARK_AVAILABLE:
        return {
            "task_type": task_type,
            "use_case": use_case,
            "recommended_provider": "ollama",
            "recommended_model": "llama3.2",
            "confidence": 0.5,
            "reasons": ["Default recommendation - no benchmark data"],
            "alternatives": [],
            "tradeoffs": {},
        }

    try:
        store = BenchmarkStore()
        recommendations = get_recommendations_engine(store)

        rec = recommendations.get_recommendations(
            task_type=task_type,
            use_case=use_case,
            max_budget=max_budget,
            max_latency_ms=max_latency_ms,
            min_accuracy=min_accuracy,
        )

        return BenchmarkRecommendationResponse(
            task_type=rec.task_type,
            use_case=rec.use_case,
            recommended_provider=rec.recommended_provider,
            recommended_model=rec.recommended_model,
            confidence=rec.confidence,
            reasons=rec.reasons,
            alternatives=rec.alternatives,
            tradeoffs=rec.tradeoffs,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {e}")  from e


@router.get("/benchmarks/suites", response_model=Dict[str, Any], tags=["benchmarks"])
async def list_benchmark_suites():
    """List available benchmark suites"""
    if not BENCHMARK_AVAILABLE:
        return {
            "suites": ["code_generation", "chat", "reasoning"],
            "message": "Mock response - benchmark module not available",
        }

    try:
        suites = get_benchmark_suites()

        all_suites = suites.get_all_suites()

        return {
            "suites": {
                name: {
                    "task_count": len(tasks),
                    "task_types": list({t.task_type.value if hasattr(t.task_type, 'value') else str(t.task_type) for t in tasks}),
                }
                for name, tasks in all_suites.items()
            },
            "task_types": suites.get_task_types(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list suites: {e}")  from e


# Add router tags and metadata
router.tags = ["Monitoring"]
