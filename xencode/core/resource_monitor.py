"""
Resource monitoring utilities for Xencode
Provides memory usage monitoring and resource management
"""
import psutil
import os
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryUsage:
    """Data class to represent memory usage statistics"""
    rss: int  # Resident Set Size - physical memory currently used by the process
    vms: int  # Virtual Memory Size - virtual memory used by the process
    percent: float  # Percentage of memory used relative to total system memory
    available: int  # Available memory in bytes
    timestamp: datetime


@dataclass
class CPUUsage:
    """Data class to represent CPU usage statistics"""
    percent: float  # CPU usage percentage
    count: int  # Number of logical CPUs
    load_avg: tuple  # Load average over 1, 5, and 15 minutes
    timestamp: datetime


class ResourceManager:
    """Manages system resources and monitors usage"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.history: Dict[str, list] = {
            'memory': [],
            'cpu': [],
            'disk': []
        }
    
    def get_memory_usage(self) -> MemoryUsage:
        """Get current memory usage statistics for the process"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        virtual_memory = psutil.virtual_memory()
        
        return MemoryUsage(
            rss=memory_info.rss,
            vms=memory_info.vms,
            percent=memory_percent,
            available=virtual_memory.available,
            timestamp=datetime.now()
        )
    
    def get_cpu_usage(self) -> CPUUsage:
        """Get current CPU usage statistics for the process"""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg()
        
        return CPUUsage(
            percent=cpu_percent,
            count=cpu_count,
            load_avg=load_avg,
            timestamp=datetime.now()
        )
    
    def get_disk_usage(self, path: str = ".") -> Dict[str, Any]:
        """Get disk usage statistics for a given path"""
        disk_usage = psutil.disk_usage(path)
        
        return {
            'total': disk_usage.total,
            'used': disk_usage.used,
            'free': disk_usage.free,
            'percent_used': (disk_usage.used / disk_usage.total) * 100,
            'timestamp': datetime.now()
        }
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get a comprehensive system resource overview"""
        memory = self.get_memory_usage()
        cpu = self.get_cpu_usage()
        disk = self.get_disk_usage()
        
        return {
            'memory': {
                'rss_mb': round(memory.rss / 1024 / 1024, 2),
                'vms_mb': round(memory.vms / 1024 / 1024, 2),
                'percent': round(memory.percent, 2),
                'available_mb': round(memory.available / 1024 / 1024, 2)
            },
            'cpu': {
                'percent': round(cpu.percent, 2),
                'count': cpu.count,
                'load_avg': [round(x, 2) for x in cpu.load_avg]
            },
            'disk': {
                'total_gb': round(disk['total'] / 1024**3, 2),
                'used_gb': round(disk['used'] / 1024**3, 2),
                'free_gb': round(disk['free'] / 1024**3, 2),
                'percent_used': round(disk['percent_used'], 2)
            },
            'timestamp': memory.timestamp
        }
    
    def monitor_resources(self, duration: int = 10, interval: int = 1) -> Dict[str, list]:
        """Monitor resources over a period of time
        
        Args:
            duration: Duration to monitor in seconds
            interval: Interval between measurements in seconds
            
        Returns:
            Dictionary containing historical resource usage data
        """
        self.history = {'memory': [], 'cpu': [], 'disk': []}
        
        for _ in range(duration):
            memory = self.get_memory_usage()
            cpu = self.get_cpu_usage()
            disk = self.get_disk_usage()
            
            self.history['memory'].append({
                'rss_mb': round(memory.rss / 1024 / 1024, 2),
                'percent': round(memory.percent, 2),
                'timestamp': memory.timestamp.isoformat()
            })
            
            self.history['cpu'].append({
                'percent': round(cpu.percent, 2),
                'timestamp': cpu.timestamp.isoformat()
            })
            
            self.history['disk'].append({
                'percent_used': round(disk['percent_used'], 2),
                'timestamp': disk['timestamp'].isoformat()
            })
            
            time.sleep(interval)
        
        return self.history
    
    def is_memory_usage_high(self, threshold: float = 80.0) -> bool:
        """Check if memory usage is above a threshold
        
        Args:
            threshold: Memory usage percentage threshold
            
        Returns:
            True if memory usage is above threshold, False otherwise
        """
        memory = self.get_memory_usage()
        return memory.percent > threshold
    
    def is_cpu_usage_high(self, threshold: float = 80.0) -> bool:
        """Check if CPU usage is above a threshold
        
        Args:
            threshold: CPU usage percentage threshold
            
        Returns:
            True if CPU usage is above threshold, False otherwise
        """
        cpu = self.get_cpu_usage()
        return cpu.percent > threshold
    
    def get_peak_memory_usage(self) -> Optional[float]:
        """Get the peak memory usage recorded in history
        
        Returns:
            Peak memory usage percentage or None if no history
        """
        if not self.history['memory']:
            return None
        return max([record['percent'] for record in self.history['memory']])
    
    def get_average_cpu_usage(self) -> Optional[float]:
        """Get the average CPU usage recorded in history
        
        Returns:
            Average CPU usage percentage or None if no history
        """
        if not self.history['cpu']:
            return None
        return sum([record['percent'] for record in self.history['cpu']]) / len(self.history['cpu'])


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get or create a singleton resource manager instance"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def get_current_memory_usage() -> Dict[str, Any]:
    """Get current memory usage as a dictionary"""
    rm = get_resource_manager()
    return rm.get_system_overview()['memory']


def get_current_cpu_usage() -> Dict[str, Any]:
    """Get current CPU usage as a dictionary"""
    rm = get_resource_manager()
    return rm.get_system_overview()['cpu']


def get_current_disk_usage(path: str = ".") -> Dict[str, Any]:
    """Get current disk usage for a path as a dictionary"""
    rm = get_resource_manager()
    return rm.get_disk_usage(path)


def is_system_stressed(memory_threshold: float = 85.0, cpu_threshold: float = 85.0) -> bool:
    """Check if the system is under stress based on resource usage
    
    Args:
        memory_threshold: Memory usage percentage threshold
        cpu_threshold: CPU usage percentage threshold
        
    Returns:
        True if system is stressed, False otherwise
    """
    rm = get_resource_manager()
    return rm.is_memory_usage_high(memory_threshold) or rm.is_cpu_usage_high(cpu_threshold)


def get_system_health_report() -> Dict[str, Any]:
    """Get a comprehensive system health report
    
    Returns:
        Dictionary containing system health information
    """
    rm = get_resource_manager()
    overview = rm.get_system_overview()
    
    return {
        'status': 'STRESSED' if is_system_stressed() else 'HEALTHY',
        'overview': overview,
        'recommendations': _generate_recommendations(overview)
    }


def _generate_recommendations(overview: Dict[str, Any]) -> list:
    """Generate recommendations based on system overview
    
    Args:
        overview: System overview dictionary
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    if overview['memory']['percent'] > 85:
        recommendations.append("High memory usage detected. Consider optimizing memory-intensive operations.")
    elif overview['memory']['percent'] > 70:
        recommendations.append("Moderate memory usage. Monitor for potential issues.")
    
    if overview['cpu']['percent'] > 85:
        recommendations.append("High CPU usage detected. Consider optimizing computational operations.")
    elif overview['cpu']['percent'] > 70:
        recommendations.append("Moderate CPU usage. Monitor for potential issues.")
    
    if overview['disk']['percent_used'] > 90:
        recommendations.append("Disk usage is critically high. Free up disk space.")
    elif overview['disk']['percent_used'] > 75:
        recommendations.append("Disk usage is high. Consider cleaning up unnecessary files.")
    
    if not recommendations:
        recommendations.append("System resources are within normal ranges.")
    
    return recommendations