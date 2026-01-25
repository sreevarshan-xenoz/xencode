#!/usr/bin/env python3
"""
Dynamic Cache Sizing System for Xencode

Automatically adjusts cache sizes based on available system memory
and current usage patterns for optimal performance.
"""

import asyncio
import psutil
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import json
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class MemoryProfile:
    """System memory profile for cache sizing decisions"""
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_percent: float
    swap_total_gb: float
    swap_used_gb: float
    process_memory_mb: float


@dataclass
class CacheConfiguration:
    """Cache configuration with dynamic sizing"""
    memory_cache_mb: int
    disk_cache_mb: int
    max_item_size_kb: int
    ttl_seconds: int
    compression_enabled: bool
    profile_timestamp: float


class DynamicCacheSizer:
    """Determines optimal cache sizes based on system resources"""

    def __init__(self):
        self.profile: Optional[MemoryProfile] = None
        self.configuration: Optional[CacheConfiguration] = None
        self.last_update: float = 0
        self.update_interval: int = 300  # Update every 5 minutes

    def get_memory_profile(self) -> MemoryProfile:
        """Get current system memory profile"""
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        
        # Get current process memory usage
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / (1024 * 1024)

        profile = MemoryProfile(
            total_memory_gb=vm.total / (1024**3),
            available_memory_gb=vm.available / (1024**3),
            used_memory_gb=vm.used / (1024**3),
            memory_percent=vm.percent,
            swap_total_gb=sm.total / (1024**3),
            swap_used_gb=sm.used / (1024**3),
            process_memory_mb=process_memory_mb
        )
        
        return profile

    def calculate_optimal_sizes(self, profile: MemoryProfile) -> CacheConfiguration:
        """Calculate optimal cache sizes based on memory profile"""
        
        # Base allocation percentages based on system size
        if profile.total_memory_gb >= 32:
            # Large system: allocate more aggressively
            memory_cache_percent = 0.15  # 15% of available memory
            disk_cache_gb = 4  # 4GB disk cache
        elif profile.total_memory_gb >= 16:
            # Medium system: balanced allocation
            memory_cache_percent = 0.10  # 10% of available memory
            disk_cache_gb = 2  # 2GB disk cache
        elif profile.total_memory_gb >= 8:
            # Small system: conservative allocation
            memory_cache_percent = 0.08  # 8% of available memory
            disk_cache_gb = 1  # 1GB disk cache
        else:
            # Very small system: minimal allocation
            memory_cache_percent = 0.05  # 5% of available memory
            disk_cache_gb = 0.5  # 512MB disk cache

        # Calculate memory cache size
        available_for_cache = profile.available_memory_gb * memory_cache_percent
        memory_cache_mb = max(64, int(available_for_cache * 1024))  # Minimum 64MB

        # Adjust based on memory pressure
        if profile.memory_percent > 80:
            # High memory pressure: reduce cache sizes
            memory_cache_mb = max(32, int(memory_cache_mb * 0.5))  # Reduce by 50%
            disk_cache_gb = max(0.25, disk_cache_gb * 0.5)  # Reduce by 50%

        disk_cache_mb = int(disk_cache_gb * 1024)

        # Determine other parameters based on system capabilities
        max_item_size_kb = 1024  # 1MB max item size (adjustable based on use case)
        
        # TTL based on system size (larger systems can afford longer TTLs)
        ttl_seconds = 3600 * 24 * 7  # 1 week default
        if profile.total_memory_gb < 8:
            ttl_seconds = 3600 * 24  # 1 day for small systems

        # Enable compression on smaller systems to save memory
        compression_enabled = profile.total_memory_gb < 16

        config = CacheConfiguration(
            memory_cache_mb=memory_cache_mb,
            disk_cache_mb=disk_cache_mb,
            max_item_size_kb=max_item_size_kb,
            ttl_seconds=ttl_seconds,
            compression_enabled=compression_enabled,
            profile_timestamp=time.time()
        )

        return config

    def get_current_configuration(self) -> CacheConfiguration:
        """Get current cache configuration, updating if necessary"""
        current_time = time.time()
        
        # Update if needed
        if (self.configuration is None or 
            current_time - self.last_update > self.update_interval):
            self.profile = self.get_memory_profile()
            self.configuration = self.calculate_optimal_sizes(self.profile)
            self.last_update = current_time

        return self.configuration

    def get_advice(self) -> str:
        """Get human-readable advice about cache sizing"""
        config = self.get_current_configuration()
        profile = self.profile or self.get_memory_profile()

        advice_parts = [
            f"System Memory: {profile.total_memory_gb:.1f}GB total, {profile.available_memory_gb:.1f}GB available",
            f"Recommended Memory Cache: {config.memory_cache_mb}MB",
            f"Recommended Disk Cache: {config.disk_cache_mb}MB",
            f"Compression: {'Enabled' if config.compression_enabled else 'Disabled'}"
        ]

        if profile.memory_percent > 80:
            advice_parts.append("⚠️ High memory pressure detected - caches reduced")
        elif profile.memory_percent < 30:
            advice_parts.append("✅ Low memory pressure - optimal sizing applied")

        return "\n".join(advice_parts)


class AdaptiveCacheManager:
    """Cache manager that adapts to system conditions"""

    def __init__(self, base_cache_dir: Optional[Path] = None):
        self.base_cache_dir = base_cache_dir or Path.home() / ".xencode" / "adaptive_cache"
        self.sizer = DynamicCacheSizer()
        self.cache_instances = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False

    async def initialize(self):
        """Initialize the adaptive cache manager"""
        console.print("[blue]🔄 Initializing adaptive cache manager...[/blue]")
        
        # Get initial configuration
        config = self.sizer.get_current_configuration()
        
        console.print(f"[green]✅ Cache configuration loaded:[/green]")
        console.print(f"   Memory: {config.memory_cache_mb}MB")
        console.print(f"   Disk: {config.disk_cache_mb}MB")
        console.print(f"   Compression: {'Yes' if config.compression_enabled else 'No'}")

    async def get_optimal_cache(self, cache_name: str = "default"):
        """Get a cache instance with optimal sizing"""
        config = self.sizer.get_current_configuration()
        
        # Import the improved cache system we created earlier
        try:
            from .improved_cache_system import ImprovedHybridCacheManager
            cache = ImprovedHybridCacheManager(
                memory_cache_mb=config.memory_cache_mb,
                disk_cache_mb=config.disk_cache_mb,
                cache_dir=self.base_cache_dir / cache_name
            )
            self.cache_instances[cache_name] = cache
            return cache
        except ImportError:
            # Fallback to basic cache if improved version not available
            from .advanced_cache_system import HybridCacheManager
            cache = HybridCacheManager(
                memory_cache_mb=config.memory_cache_mb,
                disk_cache_mb=config.disk_cache_mb,
                cache_dir=self.base_cache_dir / cache_name
            )
            self.cache_instances[cache_name] = cache
            return cache

    async def update_cache_sizes(self):
        """Dynamically update cache sizes based on current system conditions"""
        if not self.running:
            return

        config = self.sizer.get_current_configuration()
        
        # Update all existing cache instances
        for name, cache in self.cache_instances.items():
            # Note: This would require modifying the cache implementation to support resizing
            # For now, we'll just log the recommendation
            console.print(f"[yellow]💡 Cache '{name}' should be resized to {config.memory_cache_mb}MB memory, {config.disk_cache_mb}MB disk[/yellow]")

    async def start_monitoring(self):
        """Start background monitoring of system resources"""
        if self.running:
            return

        self.running = True
        console.print("[blue]👀 Starting cache monitoring...[/blue]")

        async def monitor_loop():
            while self.running:
                try:
                    # Update cache sizes periodically
                    await self.update_cache_sizes()
                    await asyncio.sleep(300)  # Check every 5 minutes
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    console.print(f"[red]Error in monitoring loop: {e}[/red]")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying

        self.monitoring_task = asyncio.create_task(monitor_loop())

    async def stop_monitoring(self):
        """Stop background monitoring"""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

    def get_system_report(self) -> Dict[str, Any]:
        """Get a comprehensive system report"""
        profile = self.sizer.get_memory_profile()
        config = self.sizer.get_current_configuration()

        return {
            "timestamp": time.time(),
            "memory_profile": {
                "total_gb": profile.total_memory_gb,
                "available_gb": profile.available_memory_gb,
                "used_gb": profile.used_memory_gb,
                "percent": profile.memory_percent,
                "process_memory_mb": profile.process_memory_mb
            },
            "cache_configuration": {
                "memory_cache_mb": config.memory_cache_mb,
                "disk_cache_mb": config.disk_cache_mb,
                "compression_enabled": config.compression_enabled,
                "ttl_seconds": config.ttl_seconds
            },
            "recommendations": self.sizer.get_advice()
        }

    def display_system_report(self):
        """Display system report in a formatted table"""
        report = self.get_system_report()
        
        console.print(Panel(
            f"[bold]System Memory Profile[/bold]\n"
            f"Total Memory: {report['memory_profile']['total_gb']:.1f} GB\n"
            f"Available Memory: {report['memory_profile']['available_gb']:.1f} GB\n"
            f"Used Memory: {report['memory_profile']['used_gb']:.1f} GB ({report['memory_profile']['percent']:.1f}%)\n"
            f"Process Memory: {report['memory_profile']['process_memory_mb']:.1f} MB",
            title="Memory Overview",
            border_style="blue"
        ))

        console.print(Panel(
            f"[bold]Cache Configuration[/bold]\n"
            f"Memory Cache: {report['cache_configuration']['memory_cache_mb']} MB\n"
            f"Disk Cache: {report['cache_configuration']['disk_cache_mb']} MB\n"
            f"Compression: {'Enabled' if report['cache_configuration']['compression_enabled'] else 'Disabled'}\n"
            f"TTL: {report['cache_configuration']['ttl_seconds']} seconds",
            title="Cache Settings",
            border_style="green"
        ))

        console.print(Panel(
            f"[bold]Recommendations[/bold]\n{report['recommendations']}",
            title="Optimization Advice",
            border_style="yellow"
        ))


# Global adaptive cache manager instance
_adaptive_cache_manager: Optional[AdaptiveCacheManager] = None


async def get_adaptive_cache_manager() -> AdaptiveCacheManager:
    """Get or create the global adaptive cache manager"""
    global _adaptive_cache_manager
    if _adaptive_cache_manager is None:
        _adaptive_cache_manager = AdaptiveCacheManager()
        await _adaptive_cache_manager.initialize()
    return _adaptive_cache_manager


async def get_adaptive_cache(cache_name: str = "default"):
    """Get an optimally sized cache instance"""
    manager = await get_adaptive_cache_manager()
    return await manager.get_optimal_cache(cache_name)


# Example usage
if __name__ == "__main__":
    async def demo():
        console.print("[bold blue]🚀 Dynamic Cache Sizing Demo[/bold blue]")
        
        # Create adaptive cache manager
        manager = AdaptiveCacheManager()
        await manager.initialize()
        
        # Display system report
        manager.display_system_report()
        
        # Get an optimally sized cache
        cache = await manager.get_optimal_cache("demo_cache")
        console.print(f"[green]✅ Got optimally sized cache[/green]")
        
        # Start monitoring
        await manager.start_monitoring()
        console.print(f"[blue]✅ Monitoring started[/blue]")
        
        # Simulate some work
        await asyncio.sleep(2)
        
        # Stop monitoring
        await manager.stop_monitoring()
        console.print(f"[yellow]✅ Monitoring stopped[/yellow]")

    # Don't run the demo by default
    # asyncio.run(demo())