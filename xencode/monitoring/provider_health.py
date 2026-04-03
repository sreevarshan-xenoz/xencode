#!/usr/bin/env python3
"""
Provider Health Dashboard

Live monitoring of provider status, latency, errors, and expiry.
Provides real-time health metrics across all configured providers.

Features:
- Real-time provider status monitoring
- Latency tracking and trending
- Error rate monitoring and alerting
- Token usage and quota tracking
- Provider comparison and recommendations
- Historical health data persistence
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import aiohttp

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class HealthStatus(Enum):
    """Provider health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    OFFLINE = "offline"


class ProviderType(Enum):
    """Supported provider types"""
    LOCAL_OLLAMA = "local_ollama"
    CLOUD_QWEN = "cloud_qwen"
    CLOUD_OPENROUTER = "cloud_openrouter"
    CLOUD_ANTHROPIC = "cloud_anthropic"
    CLOUD_OPENAI = "cloud_openai"


@dataclass
class LatencyMetrics:
    """Latency tracking metrics"""
    current_ms: float = 0.0
    avg_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    sample_count: int = 0
    
    # Trend tracking
    samples: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_sample(self, latency_ms: float):
        """Add a latency sample and update metrics"""
        self.samples.append(latency_ms)
        self.sample_count += 1
        
        # Update current
        self.current_ms = latency_ms
        
        # Update min/max
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        
        # Update average
        self.avg_ms = sum(self.samples) / len(self.samples)
        
        # Calculate percentiles
        sorted_samples = sorted(self.samples)
        n = len(sorted_samples)
        if n > 0:
            self.p50_ms = sorted_samples[int(n * 0.50)]
            self.p95_ms = sorted_samples[int(n * 0.95)] if n >= 20 else self.max_ms
            self.p99_ms = sorted_samples[int(n * 0.99)] if n >= 100 else self.max_ms
    
    def get_trend(self) -> str:
        """Get latency trend direction"""
        if len(self.samples) < 10:
            return "stable"
        
        recent_avg = sum(list(self.samples)[-10:]) / 10
        older_avg = sum(list(self.samples)[:10]) / 10
        
        if recent_avg > older_avg * 1.2:
            return "increasing"
        elif recent_avg < older_avg * 0.8:
            return "decreasing"
        return "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_ms': self.current_ms,
            'avg_ms': self.avg_ms,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'p50_ms': self.p50_ms,
            'p95_ms': self.p95_ms,
            'p99_ms': self.p99_ms,
            'sample_count': self.sample_count,
            'trend': self.get_trend(),
        }


@dataclass
class ErrorMetrics:
    """Error tracking metrics"""
    total_errors: int = 0
    errors_last_hour: int = 0
    errors_last_24h: int = 0
    error_rate: float = 0.0  # Percentage
    consecutive_errors: int = 0
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None
    
    # Error breakdown by type
    error_types: Dict[str, int] = field(default_factory=dict)
    
    # Time-based tracking
    error_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def record_error(self, error_type: str, error_message: str):
        """Record an error occurrence"""
        now = datetime.now()
        
        self.total_errors += 1
        self.consecutive_errors += 1
        self.last_error_time = now
        self.last_error_message = error_message
        
        # Track error type
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        # Add to history
        self.error_history.append({
            'timestamp': now,
            'type': error_type,
            'message': error_message,
        })
        
        # Update time-based counts
        self._update_time_based_counts()
    
    def record_success(self):
        """Record a successful request"""
        self.consecutive_errors = 0
    
    def _update_time_based_counts(self):
        """Update time-based error counts"""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(hours=24)
        
        self.errors_last_hour = sum(
            1 for e in self.error_history
            if e['timestamp'] > one_hour_ago
        )
        self.errors_last_24h = sum(
            1 for e in self.error_history
            if e['timestamp'] > one_day_ago
        )
    
    def calculate_error_rate(self, total_requests: int):
        """Calculate error rate percentage"""
        if total_requests > 0:
            self.error_rate = (self.total_errors / total_requests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_errors': self.total_errors,
            'errors_last_hour': self.errors_last_hour,
            'errors_last_24h': self.errors_last_24h,
            'error_rate': self.error_rate,
            'consecutive_errors': self.consecutive_errors,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'last_error_message': self.last_error_message,
            'error_types': self.error_types,
        }


@dataclass
class UsageMetrics:
    """Token usage and quota tracking"""
    tokens_used_today: int = 0
    tokens_used_total: int = 0
    requests_today: int = 0
    requests_total: int = 0
    quota_limit: Optional[int] = None
    quota_remaining: Optional[int] = None
    quota_reset_time: Optional[datetime] = None
    estimated_cost_today: float = 0.0
    estimated_cost_total: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tokens_used_today': self.tokens_used_today,
            'tokens_used_total': self.tokens_used_total,
            'requests_today': self.requests_today,
            'requests_total': self.requests_total,
            'quota_limit': self.quota_limit,
            'quota_remaining': self.quota_remaining,
            'quota_reset_time': self.quota_reset_time.isoformat() if self.quota_reset_time else None,
            'estimated_cost_today': self.estimated_cost_today,
            'estimated_cost_total': self.estimated_cost_total,
        }


@dataclass
class ProviderHealth:
    """Complete health information for a provider"""
    provider: ProviderType
    status: HealthStatus = HealthStatus.UNKNOWN
    endpoint: Optional[str] = None
    last_checked: Optional[datetime] = None
    last_success: Optional[datetime] = None
    uptime_percentage: float = 0.0
    
    # Metrics
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    errors: ErrorMetrics = field(default_factory=ErrorMetrics)
    usage: UsageMetrics = field(default_factory=UsageMetrics)
    
    # Configuration
    model_count: int = 0
    available_models: List[str] = field(default_factory=list)
    
    # Metadata
    version: Optional[str] = None
    region: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'provider': self.provider.value,
            'status': self.status.value,
            'endpoint': self.endpoint,
            'last_checked': self.last_checked.isoformat() if self.last_checked else None,
            'last_success': self.last_success.isoformat() if self.last_success else None,
            'uptime_percentage': self.uptime_percentage,
            'latency': self.latency.to_dict(),
            'errors': self.errors.to_dict(),
            'usage': self.usage.to_dict(),
            'model_count': self.model_count,
            'available_models': self.available_models,
            'version': self.version,
            'region': self.region,
        }


class ProviderHealthMonitor:
    """
    Monitors health of all configured providers
    
    Features:
    - Periodic health checks
    - Latency monitoring
    - Error tracking
    - Usage quota monitoring
    - Health history persistence
    """
    
    # Default health check endpoints
    HEALTH_ENDPOINTS = {
        ProviderType.LOCAL_OLLAMA: "http://localhost:11434/api/tags",
        ProviderType.CLOUD_QWEN: "https://chat.qwen.ai/api/v1/models",
        ProviderType.CLOUD_OPENROUTER: "https://openrouter.ai/api/v1/models",
        ProviderType.CLOUD_ANTHROPIC: "https://api.anthropic.com/v1/models",
        ProviderType.CLOUD_OPENAI: "https://api.openai.com/v1/models",
    }
    
    # Health check intervals (seconds)
    CHECK_INTERVALS = {
        HealthStatus.HEALTHY: 60,
        HealthStatus.DEGRADED: 30,
        HealthStatus.UNHEALTHY: 10,
        HealthStatus.UNKNOWN: 30,
        HealthStatus.OFFLINE: 300,
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize provider health monitor
        
        Args:
            config_path: Optional path to provider configuration
        """
        self.config_path = config_path or Path.home() / ".xencode" / "provider_config.json"
        self.health_data_path = Path.home() / ".xencode" / "provider_health.json"
        
        # Provider health tracking
        self.providers: Dict[ProviderType, ProviderHealth] = {}
        self._health_checks: Dict[ProviderType, asyncio.Task] = {}
        self._running = False
        
        # Configuration
        self.provider_configs: Dict[ProviderType, Dict[str, Any]] = {}
        
        # Callbacks for health status changes
        self._status_callbacks: List[callable] = []
        
        # Load configuration
        self._load_config()
        self._load_health_history()
        
        # Initialize providers
        self._initialize_providers()
    
    def _load_config(self):
        """Load provider configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                self.provider_configs = data.get('providers', {})
                console.print(f"[green]✓ Loaded provider configuration[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ Failed to load provider config: {e}[/yellow]")
        else:
            # Use defaults
            console.print("[blue]ℹ Using default provider configuration[/blue]")
    
    def _load_health_history(self):
        """Load historical health data"""
        if self.health_data_path.exists():
            try:
                with open(self.health_data_path, 'r') as f:
                    data = json.load(f)
                
                # Restore latency samples and error history
                # (Simplified for now, full restoration would rebuild metrics)
                console.print(f"[green]✓ Loaded health history[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ Failed to load health history: {e}[/yellow]")
    
    def _save_health_data(self):
        """Persist current health data"""
        try:
            self.health_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'last_updated': datetime.now().isoformat(),
                'providers': {
                    p.value: h.to_dict()
                    for p, h in self.providers.items()
                },
            }
            
            with open(self.health_data_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to save health data: {e}[/yellow]")
    
    def _initialize_providers(self):
        """Initialize health tracking for configured providers"""
        for provider_type in ProviderType:
            config = self.provider_configs.get(provider_type.value, {})
            
            health = ProviderHealth(
                provider=provider_type,
                endpoint=config.get('base_url', self.HEALTH_ENDPOINTS.get(provider_type)),
                region=config.get('region'),
            )
            
            self.providers[provider_type] = health
            console.print(f"[blue]ℹ Initialized health tracking for {provider_type.value}[/blue]")
    
    def register_status_callback(self, callback: callable):
        """Register callback for health status changes"""
        self._status_callbacks.append(callback)
    
    def _notify_status_change(
        self,
        provider: ProviderType,
        old_status: HealthStatus,
        new_status: HealthStatus,
    ):
        """Notify callbacks of status change"""
        for callback in self._status_callbacks:
            try:
                callback(provider, old_status, new_status)
            except Exception as e:
                console.print(f"[yellow]⚠ Status callback error: {e}[/yellow]")
    
    async def check_provider_health(
        self,
        provider: ProviderType,
        timeout: int = 10,
    ) -> ProviderHealth:
        """
        Check health of a single provider
        
        Args:
            provider: Provider to check
            timeout: Request timeout in seconds
        
        Returns:
            Updated ProviderHealth instance
        """
        health = self.providers[provider]
        endpoint = health.endpoint
        
        if not endpoint:
            health.status = HealthStatus.UNKNOWN
            return health
        
        start_time = time.time()
        
        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                async with session.get(endpoint) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        # Success
                        health.status = HealthStatus.HEALTHY
                        health.last_success = datetime.now()
                        health.errors.record_success()
                        
                        # Try to extract model list
                        try:
                            data = await response.json()
                            if isinstance(data, dict) and 'data' in data:
                                health.available_models = [
                                    m.get('id', '') for m in data['data']
                                ][:20]  # Limit to 20 models
                                health.model_count = len(health.available_models)
                        except Exception:
                            pass
                    else:
                        # HTTP error
                        health.status = HealthStatus.DEGRADED
                        health.errors.record_error(
                            f'http_{response.status}',
                            f'HTTP {response.status} from {endpoint}',
                        )
                    
                    # Update latency
                    health.latency.add_sample(latency_ms)
                    
        except asyncio.TimeoutError:
            health.status = HealthStatus.UNHEALTHY
            health.errors.record_error('timeout', f'Timeout connecting to {endpoint}')
            
        except aiohttp.ClientConnectionError as e:
            health.status = HealthStatus.OFFLINE
            health.errors.record_error('connection_error', str(e))
            
        except Exception as e:
            health.status = HealthStatus.UNHEALTHY
            health.errors.record_error('unknown', str(e))
        
        # Update metadata
        health.last_checked = datetime.now()
        
        # Calculate uptime (simplified)
        total_checks = health.errors.total_errors + health.errors.consecutive_errors
        if total_checks > 0:
            health.uptime_percentage = (
                (total_checks - health.errors.total_errors) / total_checks
            ) * 100
        
        # Update error rate
        health.errors.calculate_error_rate(health.usage.requests_total)
        
        # Check for status change
        # (In production, track previous status and notify callbacks)
        
        return health
    
    async def start_monitoring(self, interval: int = 30):
        """
        Start continuous health monitoring
        
        Args:
            interval: Base check interval in seconds
        """
        self._running = True
        console.print("[green]✓ Starting provider health monitoring[/green]")
        
        async def monitor_loop():
            while self._running:
                tasks = []
                for provider in ProviderType:
                    # Adjust interval based on status
                    health = self.providers[provider]
                    check_interval = self.CHECK_INTERVALS.get(
                        health.status,
                        interval,
                    )
                    
                    # Check if it's time for this provider
                    if (health.last_checked is None or
                        datetime.now() - (health.last_checked or datetime.now()) 
                        >= timedelta(seconds=check_interval)):
                        tasks.append(self.check_provider_health(provider))
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, Exception):
                            console.print(f"[red]✗ Health check failed: {result}[/red]")
                
                # Save health data periodically
                self._save_health_data()
                
                await asyncio.sleep(interval)
        
        # Start monitoring loop
        asyncio.create_task(monitor_loop())
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        
        # Cancel active health checks
        for task in self._health_checks.values():
            task.cancel()
        
        # Save final health data
        self._save_health_data()
        console.print("[blue]ℹ Provider health monitoring stopped[/blue]")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of all provider health"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'providers': {},
            'overall_status': HealthStatus.HEALTHY.value,
            'healthy_count': 0,
            'degraded_count': 0,
            'unhealthy_count': 0,
        }
        
        for provider, health in self.providers.items():
            summary['providers'][provider.value] = health.to_dict()
            
            if health.status == HealthStatus.HEALTHY:
                summary['healthy_count'] += 1
            elif health.status == HealthStatus.DEGRADED:
                summary['degraded_count'] += 1
            elif health.status in [HealthStatus.UNHEALTHY, HealthStatus.OFFLINE]:
                summary['unhealthy_count'] += 1
        
        # Determine overall status
        if summary['unhealthy_count'] > 0:
            summary['overall_status'] = HealthStatus.UNHEALTHY.value
        elif summary['degraded_count'] > 0:
            summary['overall_status'] = HealthStatus.DEGRADED.value
        
        return summary
    
    def get_provider_health(self, provider: ProviderType) -> Optional[ProviderHealth]:
        """Get health for specific provider"""
        return self.providers.get(provider)
    
    def get_recommended_provider(self, task_type: str = "general") -> ProviderType:
        """
        Get recommended provider based on current health
        
        Args:
            task_type: Type of task (affects selection criteria)
        
        Returns:
            Recommended ProviderType
        """
        # Filter healthy providers
        healthy_providers = [
            (p, h) for p, h in self.providers.items()
            if h.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        ]
        
        if not healthy_providers:
            # Fallback to first provider
            return list(ProviderType)[0]
        
        # Score providers
        scored = []
        for provider, health in healthy_providers:
            score = 0
            
            # Status weight
            if health.status == HealthStatus.HEALTHY:
                score += 100
            else:
                score += 50
            
            # Latency score (lower is better)
            if health.latency.avg_ms > 0:
                latency_score = max(0, 100 - (health.latency.avg_ms / 100))
                score += latency_score
            
            # Error rate penalty
            score -= health.errors.error_rate * 10
            
            # Uptime bonus
            score += health.uptime_percentage * 0.5
            
            scored.append((provider, score))
        
        # Return highest scored provider
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    
    def generate_dashboard_table(self) -> Table:
        """Generate Rich table for dashboard display"""
        table = Table(title="Provider Health Dashboard")
        
        table.add_column("Provider", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Error Rate", justify="right")
        table.add_column("Uptime", justify="right")
        table.add_column("Last Check", justify="center")
        
        for provider, health in self.providers.items():
            # Status indicator
            status_icons = {
                HealthStatus.HEALTHY: "[green]●[/green]",
                HealthStatus.DEGRADED: "[yellow]●[/yellow]",
                HealthStatus.UNHEALTHY: "[red]●[/red]",
                HealthStatus.OFFLINE: "[red]○[/red]",
                HealthStatus.UNKNOWN: "[gray]●[/gray]",
            }
            status_str = status_icons.get(health.status, "?")
            
            # Latency with trend
            latency_str = f"{health.latency.current_ms:.0f}"
            trend = health.latency.get_trend()
            if trend == "increasing":
                latency_str += " ↑"
            elif trend == "decreasing":
                latency_str += " ↓"
            
            # Error rate
            error_str = f"{health.errors.error_rate:.1f}%"
            if health.errors.error_rate > 5:
                error_str = f"[red]{error_str}[/red]"
            
            # Uptime
            uptime_str = f"{health.uptime_percentage:.1f}%"
            
            # Last check
            last_check = "Never"
            if health.last_checked:
                delta = datetime.now() - health.last_checked
                seconds = int(delta.total_seconds())
                if seconds < 60:
                    last_check = f"{seconds}s ago"
                elif seconds < 3600:
                    last_check = f"{seconds // 60}m ago"
                else:
                    last_check = f"{seconds // 3600}h ago"
            
            table.add_row(
                provider.value,
                status_str,
                latency_str,
                error_str,
                uptime_str,
                last_check,
            )
        
        return table


# Global monitor instance
_monitor: Optional[ProviderHealthMonitor] = None


def get_health_monitor(config_path: Optional[Path] = None) -> ProviderHealthMonitor:
    """Get or create global health monitor"""
    global _monitor
    if _monitor is None:
        _monitor = ProviderHealthMonitor(config_path)
    return _monitor


def check_all_providers() -> Dict[str, Any]:
    """
    Check all providers synchronously (for CLI use)
    
    Returns:
        Health summary dictionary
    """
    monitor = get_health_monitor()
    
    async def check_all():
        tasks = [
            monitor.check_provider_health(provider)
            for provider in ProviderType
        ]
        await asyncio.gather(*tasks)
        return monitor.get_health_summary()
    
    return asyncio.run(check_all())


def show_health_dashboard():
    """Display health dashboard in terminal"""
    monitor = get_health_monitor()
    
    async def check_and_display():
        # Check all providers
        tasks = [
            monitor.check_provider_health(provider)
            for provider in ProviderType
        ]
        await asyncio.gather(*tasks)
        
        # Display table
        table = monitor.generate_dashboard_table()
        console.print(table)
        
        # Show recommendations
        recommended = monitor.get_recommended_provider()
        console.print(
            Panel(
                f"[green]Recommended Provider:[/green] {recommended.value}",
                title="Recommendation",
            )
        )
    
    asyncio.run(check_and_display())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--dashboard":
        show_health_dashboard()
    elif len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        # Start continuous monitoring
        monitor = get_health_monitor()
        
        async def run_monitor():
            await monitor.start_monitoring(interval=10)
            try:
                while True:
                    summary = monitor.get_health_summary()
                    console.print(f"\n[bold]Overall Status:[/bold] {summary['overall_status']}")
                    console.print(f"  Healthy: {summary['healthy_count']}")
                    console.print(f"  Degraded: {summary['degraded_count']}")
                    console.print(f"  Unhealthy: {summary['unhealthy_count']}")
                    await asyncio.sleep(30)
            except KeyboardInterrupt:
                await monitor.stop_monitoring()
        
        asyncio.run(run_monitor())
    else:
        console.print("Provider Health Dashboard")
        console.print("Usage:")
        console.print("  python -m xencode.monitoring.provider_health --dashboard")
        console.print("  python -m xencode.monitoring.provider_health --monitor")
