#!/usr/bin/env python3
"""
Resource Monitor for Xencode Phase 2 Integration

Monitors system resources and adapts feature availability based on hardware
capabilities. Implements progressive enhancement strategy with 3-tier scanning.

Requirements: 6.1, 6.6, 7.1, 7.2, 7.3
"""

import os
import psutil
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class FeatureLevel(Enum):
    """System feature levels based on hardware capabilities"""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"


class ResourcePressure(Enum):
    """Resource pressure levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemProfile:
    """System hardware profile and capabilities"""
    ram_gb: float
    cpu_cores: int
    storage_gb: float
    feature_level: FeatureLevel
    
    # Performance characteristics
    cpu_freq_mhz: float = 0.0
    ram_speed_mhz: float = 0.0
    storage_type: str = "unknown"  # "ssd", "hdd", "nvme"
    
    # Calculated thresholds
    max_context_size_mb: int = 50
    max_concurrent_operations: int = 2
    scan_batch_size: int = 50
    
    def __post_init__(self):
        """Calculate adaptive thresholds based on hardware"""
        # Context size limits based on RAM
        if self.ram_gb >= 16:
            self.max_context_size_mb = 200
        elif self.ram_gb >= 8:
            self.max_context_size_mb = 100
        elif self.ram_gb >= 4:
            self.max_context_size_mb = 50
        else:
            self.max_context_size_mb = 25
        
        # Concurrent operations based on CPU cores
        if self.cpu_cores >= 8:
            self.max_concurrent_operations = 4
        elif self.cpu_cores >= 4:
            self.max_concurrent_operations = 2
        else:
            self.max_concurrent_operations = 1
        
        # Batch size based on combined RAM and CPU
        if self.ram_gb >= 8 and self.cpu_cores >= 4:
            self.scan_batch_size = 100
        elif self.ram_gb >= 4 and self.cpu_cores >= 2:
            self.scan_batch_size = 50
        else:
            self.scan_batch_size = 25


@dataclass
class ResourceUsage:
    """Current resource usage metrics"""
    ram_mb: int
    ram_percent: float
    cpu_percent: float
    storage_mb: int
    storage_percent: float
    
    # Pressure indicators
    is_throttled: bool = False
    pressure_level: ResourcePressure = ResourcePressure.LOW
    
    # Performance metrics
    io_wait_percent: float = 0.0
    load_average: float = 0.0
    
    def __post_init__(self):
        """Calculate pressure level based on usage"""
        # Determine pressure based on multiple factors
        pressure_score = 0
        
        # RAM pressure (most critical for context operations)
        if self.ram_percent >= 90:
            pressure_score += 3
        elif self.ram_percent >= 80:
            pressure_score += 2
        elif self.ram_percent >= 70:
            pressure_score += 1
        
        # CPU pressure
        if self.cpu_percent >= 90:
            pressure_score += 2
        elif self.cpu_percent >= 80:
            pressure_score += 1
        
        # Storage pressure
        if self.storage_percent >= 95:
            pressure_score += 2
        elif self.storage_percent >= 90:
            pressure_score += 1
        
        # I/O wait pressure
        if self.io_wait_percent >= 50:
            pressure_score += 2
        elif self.io_wait_percent >= 25:
            pressure_score += 1
        
        # Determine pressure level
        if pressure_score >= 6:
            self.pressure_level = ResourcePressure.CRITICAL
            self.is_throttled = True
        elif pressure_score >= 4:
            self.pressure_level = ResourcePressure.HIGH
            self.is_throttled = True
        elif pressure_score >= 2:
            self.pressure_level = ResourcePressure.MEDIUM
        else:
            self.pressure_level = ResourcePressure.LOW


@dataclass
class ScanStrategy:
    """Adaptive scanning strategy based on system capabilities"""
    depth: int
    batch_size: int
    max_file_size_kb: int
    skip_types: List[str]
    compression: str  # "none", "semantic", "advanced"
    pause_between_batches_ms: int = 100
    
    # Performance tuning
    parallel_processing: bool = True
    memory_limit_mb: int = 100
    timeout_seconds: int = 30
    
    # Progress reporting
    show_progress: bool = True
    progress_update_interval: int = 10  # files


@dataclass
class ProgressReport:
    """Detailed progress reporting with resource breakdown"""
    current_files: int
    total_files: int
    current_size_mb: float
    total_size_mb: float
    
    # File type breakdown
    file_types: Dict[str, int] = field(default_factory=dict)
    
    # Resource metrics
    memory_usage_mb: int = 0
    cpu_usage_percent: float = 0.0
    
    # Time estimates
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    
    # Throttling indicators
    is_throttled: bool = False
    throttle_reason: str = ""
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.current_files / self.total_files) * 100
    
    @property
    def size_progress_percent(self) -> float:
        """Calculate size-based progress percentage"""
        if self.total_size_mb == 0:
            return 0.0
        return (self.current_size_mb / self.total_size_mb) * 100


class HardwareProfiler:
    """Detects and profiles system hardware capabilities"""
    
    def __init__(self):
        self._profile_cache = None
        self._cache_timestamp = 0
        self._cache_ttl_seconds = 300  # Cache for 5 minutes
    
    def get_system_profile(self, force_refresh: bool = False) -> SystemProfile:
        """
        Get comprehensive system profile with caching
        
        Args:
            force_refresh: Force re-profiling ignoring cache
            
        Returns:
            SystemProfile with hardware capabilities
        """
        current_time = time.time()
        
        # Use cache if available and not expired
        if (not force_refresh and 
            self._profile_cache and 
            current_time - self._cache_timestamp < self._cache_ttl_seconds):
            return self._profile_cache
        
        # Profile system hardware
        profile = self._profile_hardware()
        
        # Cache results
        self._profile_cache = profile
        self._cache_timestamp = current_time
        
        return profile
    
    def _profile_hardware(self) -> SystemProfile:
        """Profile system hardware capabilities"""
        # Get basic system info
        memory = psutil.virtual_memory()
        ram_gb = memory.total / (1024**3)
        
        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
        
        # Get storage info for current directory
        disk_usage = psutil.disk_usage('/')
        storage_gb = disk_usage.total / (1024**3)
        
        # Determine feature level based on hardware
        feature_level = self._determine_feature_level(ram_gb, cpu_cores)
        
        # Create base profile
        profile = SystemProfile(
            ram_gb=ram_gb,
            cpu_cores=cpu_cores,
            storage_gb=storage_gb,
            feature_level=feature_level
        )
        
        # Add detailed characteristics
        self._add_detailed_characteristics(profile)
        
        return profile
    
    def _determine_feature_level(self, ram_gb: float, cpu_cores: int) -> FeatureLevel:
        """Determine feature level based on hardware specs"""
        # Advanced: High-end systems
        if ram_gb >= 16 and cpu_cores >= 8:
            return FeatureLevel.ADVANCED
        
        # Standard: Mid-range systems
        elif ram_gb >= 8 and cpu_cores >= 4:
            return FeatureLevel.STANDARD
        
        # Basic: Low-end or resource-constrained systems
        else:
            return FeatureLevel.BASIC
    
    def _add_detailed_characteristics(self, profile: SystemProfile):
        """Add detailed hardware characteristics to profile"""
        try:
            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                profile.cpu_freq_mhz = cpu_freq.current or cpu_freq.max or 0.0
            
            # Storage type detection (simplified)
            profile.storage_type = self._detect_storage_type()
            
        except Exception:
            # Best effort - continue with defaults if detection fails
            pass
    
    def _detect_storage_type(self) -> str:
        """Detect storage type (SSD vs HDD)"""
        try:
            # Check for SSD indicators in /proc/mounts or similar
            # This is a simplified detection - real implementation would be more robust
            
            # Check if running on SSD (Linux-specific)
            if os.path.exists('/sys/block'):
                for device in os.listdir('/sys/block'):
                    if device.startswith(('sd', 'nvme')):
                        rotational_file = f'/sys/block/{device}/queue/rotational'
                        if os.path.exists(rotational_file):
                            with open(rotational_file, 'r') as f:
                                if f.read().strip() == '0':
                                    if device.startswith('nvme'):
                                        return "nvme"
                                    else:
                                        return "ssd"
                                else:
                                    return "hdd"
            
            return "unknown"
            
        except Exception:
            return "unknown"


class PerformanceThrottler:
    """Adjusts system behavior based on resource pressure"""
    
    def __init__(self, system_profile: SystemProfile):
        self.system_profile = system_profile
        self.throttle_history = []
        self.max_history_size = 10
    
    def should_throttle_operations(self, current_usage: ResourceUsage) -> bool:
        """Determine if operations should be throttled"""
        return current_usage.is_throttled or current_usage.pressure_level in [
            ResourcePressure.HIGH, ResourcePressure.CRITICAL
        ]
    
    def adjust_scan_strategy(self, base_strategy: ScanStrategy, 
                           current_usage: ResourceUsage) -> ScanStrategy:
        """
        Adjust scanning strategy based on current resource pressure
        
        Args:
            base_strategy: Base scanning strategy
            current_usage: Current resource usage
            
        Returns:
            Adjusted scanning strategy
        """
        adjusted = ScanStrategy(
            depth=base_strategy.depth,
            batch_size=base_strategy.batch_size,
            max_file_size_kb=base_strategy.max_file_size_kb,
            skip_types=base_strategy.skip_types.copy(),
            compression=base_strategy.compression,
            pause_between_batches_ms=base_strategy.pause_between_batches_ms,
            parallel_processing=base_strategy.parallel_processing,
            memory_limit_mb=base_strategy.memory_limit_mb,
            timeout_seconds=base_strategy.timeout_seconds,
            show_progress=base_strategy.show_progress,
            progress_update_interval=base_strategy.progress_update_interval
        )
        
        # Adjust based on pressure level
        if current_usage.pressure_level == ResourcePressure.CRITICAL:
            # Severe throttling
            adjusted.batch_size = max(10, adjusted.batch_size // 4)
            adjusted.max_file_size_kb = min(100, adjusted.max_file_size_kb // 2)
            adjusted.pause_between_batches_ms = 500
            adjusted.parallel_processing = False
            adjusted.memory_limit_mb = min(25, adjusted.memory_limit_mb // 2)
            adjusted.timeout_seconds = min(10, adjusted.timeout_seconds // 2)
            
        elif current_usage.pressure_level == ResourcePressure.HIGH:
            # Moderate throttling
            adjusted.batch_size = max(25, adjusted.batch_size // 2)
            adjusted.max_file_size_kb = adjusted.max_file_size_kb // 2
            adjusted.pause_between_batches_ms = 250
            adjusted.parallel_processing = self.system_profile.cpu_cores > 2
            adjusted.memory_limit_mb = adjusted.memory_limit_mb // 2
            
        elif current_usage.pressure_level == ResourcePressure.MEDIUM:
            # Light throttling
            adjusted.batch_size = max(adjusted.batch_size // 2, 25)
            adjusted.pause_between_batches_ms = 150
        
        # Record throttling decision
        self.throttle_history.append({
            'timestamp': time.time(),
            'pressure_level': current_usage.pressure_level,
            'throttled': current_usage.is_throttled,
            'batch_size': adjusted.batch_size
        })
        
        # Maintain history size
        if len(self.throttle_history) > self.max_history_size:
            self.throttle_history.pop(0)
        
        return adjusted


class ProgressReporter:
    """Provides detailed progress feedback with resource monitoring"""
    
    def __init__(self, system_profile: SystemProfile):
        self.system_profile = system_profile
        self.start_time = time.time()
        self.last_update_time = 0
        self.update_interval_seconds = 1.0
    
    def generate_progress_report(self, current: int, total: int, 
                               current_size_mb: float, total_size_mb: float,
                               file_types: Dict[str, int],
                               strategy: ScanStrategy) -> ProgressReport:
        """
        Generate comprehensive progress report
        
        Args:
            current: Current file count
            total: Total file count
            current_size_mb: Current processed size in MB
            total_size_mb: Total size to process in MB
            file_types: File type breakdown
            strategy: Current scanning strategy
            
        Returns:
            Detailed progress report
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Get current resource usage
        memory_usage = self._get_current_memory_usage()
        cpu_usage = self._get_current_cpu_usage()
        
        # Calculate time estimates
        if current > 0:
            rate = current / elapsed
            remaining = (total - current) / rate if rate > 0 else 0
        else:
            remaining = 0
        
        # Check for throttling
        is_throttled = strategy.pause_between_batches_ms > 100
        throttle_reason = self._get_throttle_reason(strategy)
        
        return ProgressReport(
            current_files=current,
            total_files=total,
            current_size_mb=current_size_mb,
            total_size_mb=total_size_mb,
            file_types=file_types.copy(),
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=remaining,
            is_throttled=is_throttled,
            throttle_reason=throttle_reason
        )
    
    def _get_current_memory_usage(self) -> int:
        """Get current process memory usage in MB"""
        try:
            process = psutil.Process()
            return int(process.memory_info().rss / (1024 * 1024))
        except Exception:
            return 0
    
    def _get_current_cpu_usage(self) -> float:
        """Get current process CPU usage percentage"""
        try:
            process = psutil.Process()
            return process.cpu_percent()
        except Exception:
            return 0.0
    
    def _get_throttle_reason(self, strategy: ScanStrategy) -> str:
        """Determine reason for throttling"""
        reasons = []
        
        if strategy.batch_size < 50:
            reasons.append("small batches")
        
        if strategy.pause_between_batches_ms > 200:
            reasons.append("extended pauses")
        
        if not strategy.parallel_processing:
            reasons.append("sequential processing")
        
        if strategy.memory_limit_mb < 50:
            reasons.append("memory limits")
        
        return ", ".join(reasons) if reasons else ""
    
    def should_update_progress(self) -> bool:
        """Check if progress should be updated based on interval"""
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval_seconds:
            self.last_update_time = current_time
            return True
        return False
    
    def format_progress_display(self, report: ProgressReport) -> str:
        """Format progress report for display"""
        lines = []
        
        # Progress bar
        progress_bar = self._create_progress_bar(report.progress_percent)
        lines.append(f"Progress: {progress_bar} {report.progress_percent:.1f}%")
        
        # File and size info
        lines.append(f"Files: {report.current_files:,}/{report.total_files:,}")
        lines.append(f"Size: {report.current_size_mb:.1f}/{report.total_size_mb:.1f} MB")
        
        # Resource usage
        lines.append(f"Memory: {report.memory_usage_mb} MB | CPU: {report.cpu_usage_percent:.1f}%")
        
        # Time estimates
        if report.estimated_remaining_seconds > 0:
            remaining_str = self._format_duration(report.estimated_remaining_seconds)
            lines.append(f"ETA: {remaining_str}")
        
        # Throttling info
        if report.is_throttled:
            lines.append(f"⚠️ Throttled: {report.throttle_reason}")
        
        # File type breakdown (top 3)
        if report.file_types:
            top_types = sorted(report.file_types.items(), key=lambda x: x[1], reverse=True)[:3]
            type_str = ", ".join([f"{ext}: {count}" for ext, count in top_types])
            lines.append(f"Types: {type_str}")
        
        return "\n".join(lines)
    
    def _create_progress_bar(self, percent: float, width: int = 20) -> str:
        """Create ASCII progress bar"""
        filled = int(width * percent / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


class ResourceMonitor:
    """
    Main resource monitoring coordinator that adapts feature availability
    based on system capabilities and current resource usage
    """
    
    def __init__(self):
        self.hardware_profiler = HardwareProfiler()
        self.system_profile = self.hardware_profiler.get_system_profile()
        self.throttler = PerformanceThrottler(self.system_profile)
        self.progress_reporter = ProgressReporter(self.system_profile)
        
        # Monitoring state
        self._monitoring_active = False
        self._monitor_thread = None
        self._usage_history = []
        self._max_history_size = 60  # Keep 1 minute of history at 1s intervals
    
    def get_system_profile(self) -> SystemProfile:
        """Get current system profile"""
        return self.system_profile
    
    def monitor_resource_usage(self) -> ResourceUsage:
        """Get current resource usage metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            ram_mb = int(memory.used / (1024 * 1024))
            ram_percent = memory.percent
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Storage usage
            disk = psutil.disk_usage('/')
            storage_mb = int(disk.used / (1024 * 1024))
            storage_percent = (disk.used / disk.total) * 100
            
            # I/O wait (Linux-specific)
            io_wait_percent = 0.0
            try:
                cpu_times = psutil.cpu_times_percent(interval=0.1)
                if hasattr(cpu_times, 'iowait'):
                    io_wait_percent = cpu_times.iowait
            except Exception:
                pass
            
            # Load average (Unix-specific)
            load_average = 0.0
            try:
                load_average = os.getloadavg()[0]
            except (OSError, AttributeError):
                pass
            
            usage = ResourceUsage(
                ram_mb=ram_mb,
                ram_percent=ram_percent,
                cpu_percent=cpu_percent,
                storage_mb=storage_mb,
                storage_percent=storage_percent,
                io_wait_percent=io_wait_percent,
                load_average=load_average
            )
            
            # Add to history
            self._usage_history.append({
                'timestamp': time.time(),
                'usage': usage
            })
            
            # Maintain history size
            if len(self._usage_history) > self._max_history_size:
                self._usage_history.pop(0)
            
            return usage
            
        except Exception:
            # Return safe defaults on error
            return ResourceUsage(
                ram_mb=0,
                ram_percent=0.0,
                cpu_percent=0.0,
                storage_mb=0,
                storage_percent=0.0
            )
    
    def should_throttle_features(self) -> bool:
        """Check if features should be throttled due to resource pressure"""
        current_usage = self.monitor_resource_usage()
        return self.throttler.should_throttle_operations(current_usage)
    
    def get_recommended_feature_set(self) -> Dict[str, bool]:
        """Get recommended feature set based on current system state"""
        current_usage = self.monitor_resource_usage()
        
        # Base recommendations on system profile
        recommendations = {
            'context_scanning': True,
            'semantic_compression': self.system_profile.feature_level != FeatureLevel.BASIC,
            'parallel_processing': self.system_profile.cpu_cores > 2,
            'background_monitoring': self.system_profile.feature_level == FeatureLevel.ADVANCED,
            'detailed_progress': True,
            'model_stability_testing': True
        }
        
        # Adjust based on current pressure
        if current_usage.pressure_level in [ResourcePressure.HIGH, ResourcePressure.CRITICAL]:
            recommendations.update({
                'semantic_compression': False,
                'parallel_processing': False,
                'background_monitoring': False,
                'detailed_progress': False
            })
        
        return recommendations
    
    def get_scan_strategy(self) -> ScanStrategy:
        """Get adaptive scanning strategy based on system capabilities"""
        # Base strategy on system profile
        if self.system_profile.feature_level == FeatureLevel.ADVANCED:
            strategy = ScanStrategy(
                depth=10,
                batch_size=100,
                max_file_size_kb=1000,
                skip_types=['.exe', '.dll', '.so', '.dylib'],
                compression="advanced",
                pause_between_batches_ms=50,
                parallel_processing=True,
                memory_limit_mb=200,
                timeout_seconds=60
            )
        elif self.system_profile.feature_level == FeatureLevel.STANDARD:
            strategy = ScanStrategy(
                depth=5,
                batch_size=50,
                max_file_size_kb=500,
                skip_types=['.exe', '.dll', '.so', '.dylib', '.bin'],
                compression="semantic",
                pause_between_batches_ms=100,
                parallel_processing=True,
                memory_limit_mb=100,
                timeout_seconds=30
            )
        else:  # BASIC
            strategy = ScanStrategy(
                depth=3,
                batch_size=25,
                max_file_size_kb=250,
                skip_types=['.exe', '.dll', '.so', '.dylib', '.bin', '.img', '.iso'],
                compression="none",
                pause_between_batches_ms=200,
                parallel_processing=False,
                memory_limit_mb=50,
                timeout_seconds=15
            )
        
        return strategy
    
    def adjust_strategy_for_pressure(self, current_strategy: ScanStrategy) -> ScanStrategy:
        """Adjust strategy based on current resource pressure"""
        current_usage = self.monitor_resource_usage()
        return self.throttler.adjust_scan_strategy(current_strategy, current_usage)
    
    def generate_progress_report(self, current: int, total: int, 
                               current_size_mb: float, total_size_mb: float,
                               file_types: Dict[str, int],
                               strategy: ScanStrategy) -> ProgressReport:
        """Generate detailed progress report"""
        return self.progress_reporter.generate_progress_report(
            current, total, current_size_mb, total_size_mb, file_types, strategy
        )
    
    def start_background_monitoring(self, interval_seconds: float = 1.0):
        """Start background resource monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        def monitor_loop():
            while self._monitoring_active:
                try:
                    self.monitor_resource_usage()
                    time.sleep(interval_seconds)
                except Exception:
                    time.sleep(interval_seconds)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_background_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary for debugging"""
        current_usage = self.monitor_resource_usage()
        
        return {
            'system_profile': {
                'ram_gb': self.system_profile.ram_gb,
                'cpu_cores': self.system_profile.cpu_cores,
                'storage_gb': self.system_profile.storage_gb,
                'feature_level': self.system_profile.feature_level.value,
                'storage_type': self.system_profile.storage_type
            },
            'current_usage': {
                'ram_percent': current_usage.ram_percent,
                'cpu_percent': current_usage.cpu_percent,
                'storage_percent': current_usage.storage_percent,
                'pressure_level': current_usage.pressure_level.value,
                'is_throttled': current_usage.is_throttled
            },
            'recommendations': self.get_recommended_feature_set(),
            'scan_strategy': {
                'depth': self.get_scan_strategy().depth,
                'batch_size': self.get_scan_strategy().batch_size,
                'compression': self.get_scan_strategy().compression
            }
        }


def main():
    """Demo function for ResourceMonitor"""
    print("🔍 Resource Monitor Demo")
    print("=" * 40)
    
    # Initialize resource monitor
    monitor = ResourceMonitor()
    
    # Show system profile
    profile = monitor.get_system_profile()
    print(f"\n💻 System Profile:")
    print(f"  RAM: {profile.ram_gb:.1f} GB")
    print(f"  CPU Cores: {profile.cpu_cores}")
    print(f"  Storage: {profile.storage_gb:.1f} GB")
    print(f"  Feature Level: {profile.feature_level.value.upper()}")
    print(f"  Storage Type: {profile.storage_type}")
    
    # Show current usage
    usage = monitor.monitor_resource_usage()
    print(f"\n📊 Current Usage:")
    print(f"  RAM: {usage.ram_percent:.1f}% ({usage.ram_mb:,} MB)")
    print(f"  CPU: {usage.cpu_percent:.1f}%")
    print(f"  Storage: {usage.storage_percent:.1f}%")
    print(f"  Pressure: {usage.pressure_level.value.upper()}")
    
    # Show recommendations
    recommendations = monitor.get_recommended_feature_set()
    print(f"\n🎯 Feature Recommendations:")
    for feature, enabled in recommendations.items():
        status = "✅ Enabled" if enabled else "❌ Disabled"
        print(f"  {feature}: {status}")
    
    # Show scan strategy
    strategy = monitor.get_scan_strategy()
    print(f"\n🔍 Scan Strategy:")
    print(f"  Depth: {strategy.depth}")
    print(f"  Batch Size: {strategy.batch_size}")
    print(f"  Max File Size: {strategy.max_file_size_kb} KB")
    print(f"  Compression: {strategy.compression}")
    print(f"  Parallel: {strategy.parallel_processing}")
    
    print(f"\n✅ Resource Monitor ready!")


if __name__ == "__main__":
    main()