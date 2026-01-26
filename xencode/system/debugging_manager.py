#!/usr/bin/env python3
"""
Comprehensive Logging and Debugging System for Xencode Warp Terminal

Implements structured logging, debugging tools, and diagnostic capabilities
for troubleshooting and performance monitoring.
"""

import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import json
import gzip
import shutil
from functools import wraps

import psutil


@dataclass
class LogEntry:
    """Structure for log entries"""
    timestamp: float
    level: str
    module: str
    function: str
    message: str
    extra: Dict[str, Any]


class XencodeFormatter(logging.Formatter):
    """Custom formatter for Xencode logs with structured data"""
    
    def format(self, record):
        # Create a structured log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread": record.thread,
            "thread_name": record.threadName
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class XencodeLogger:
    """Enhanced logger for Xencode with multiple output formats"""
    
    def __init__(self, name: str = "xencode", log_dir: Optional[Path] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        
        if log_dir is None:
            log_dir = Path.home() / ".xencode" / "logs"
        
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_handlers()
        self.logger.setLevel(logging.DEBUG)
        
        # Thread-safe log storage for debugging
        self.debug_logs = []
        self._debug_lock = threading.Lock()
        
    def _setup_handlers(self):
        """Setup log handlers for different outputs"""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler for real-time debugging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logs
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = XencodeFormatter()
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Structured log file for analysis
        structured_log_file = self.log_dir / f"{self.name}_structured.log"
        structured_handler = logging.handlers.RotatingFileHandler(
            structured_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=3
        )
        structured_handler.setLevel(logging.DEBUG)
        structured_formatter = XencodeFormatter()
        structured_handler.setFormatter(structured_formatter)
        self.logger.addHandler(structured_handler)
    
    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message with extra context"""
        self._log_with_extra(logging.DEBUG, msg, extra, **kwargs)
    
    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message with extra context"""
        self._log_with_extra(logging.INFO, msg, extra, **kwargs)
    
    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message with extra context"""
        self._log_with_extra(logging.WARNING, msg, extra, **kwargs)
    
    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log error message with extra context"""
        self._log_with_extra(logging.ERROR, msg, extra, **kwargs)
    
    def critical(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log critical message with extra context"""
        self._log_with_extra(logging.CRITICAL, msg, extra, **kwargs)
    
    def _log_with_extra(self, level: int, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Internal method to log with extra context"""
        extra_dict = extra or {}
        extra_dict.update(kwargs)
        
        # Add to debug logs if enabled
        if hasattr(self, '_debug_lock'):
            with self._debug_lock:
                self.debug_logs.append(LogEntry(
                    timestamp=time.time(),
                    level=logging.getLevelName(level),
                    module=self.name,
                    function="unknown",  # This would be the calling function
                    message=msg,
                    extra=extra_dict
                ))
        
        # Create a custom record with extra fields
        if extra_dict:
            self.logger.log(level, msg, extra={'extra_fields': extra_dict})
        else:
            self.logger.log(level, msg)
    
    def get_recent_logs(self, count: int = 100) -> List[LogEntry]:
        """Get recent logs from memory"""
        with self._debug_lock:
            return self.debug_logs[-count:]
    
    def export_logs(self, output_file: Path, compress: bool = True):
        """Export logs to a file"""
        logs = self.get_recent_logs(count=1000)  # Get last 1000 logs
        
        with open(output_file, 'w') as f:
            for log in logs:
                f.write(json.dumps({
                    "timestamp": log.timestamp,
                    "level": log.level,
                    "module": log.module,
                    "function": log.function,
                    "message": log.message,
                    "extra": log.extra
                }) + '\n')
        
        if compress:
            compressed_file = output_file.with_suffix(output_file.suffix + '.gz')
            with open(output_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            output_file.unlink()  # Remove uncompressed file
            return compressed_file
        
        return output_file


class DebuggingManager:
    """Centralized debugging and diagnostic manager"""
    
    def __init__(self):
        self.logger = XencodeLogger("debug")
        self.performance_monitors = {}
        self.resource_monitors = {}
        self.diagnostics = {}
        self._profiling_enabled = False
        self._profile_data = {}
        
    def enable_profiling(self):
        """Enable performance profiling"""
        self._profiling_enabled = True
        self.logger.info("Performance profiling enabled")
    
    def disable_profiling(self):
        """Disable performance profiling"""
        self._profiling_enabled = False
        self.logger.info("Performance profiling disabled")
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator to profile function execution"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__qualname__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self._profiling_enabled:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    raise
                finally:
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    # Store profile data
                    if name not in self._profile_data:
                        self._profile_data[name] = {
                            "call_count": 0,
                            "total_time": 0,
                            "avg_time": 0,
                            "min_time": float('inf'),
                            "max_time": 0,
                            "total_memory_delta": 0,
                            "avg_memory_delta": 0,
                            "success_count": 0,
                            "failure_count": 0
                        }
                    
                    profile = self._profile_data[name]
                    profile["call_count"] += 1
                    profile["total_time"] += execution_time
                    profile["avg_time"] = profile["total_time"] / profile["call_count"]
                    profile["min_time"] = min(profile["min_time"], execution_time)
                    profile["max_time"] = max(profile["max_time"], execution_time)
                    profile["total_memory_delta"] += memory_delta
                    profile["avg_memory_delta"] = profile["total_memory_delta"] / profile["call_count"]
                    
                    if success:
                        profile["success_count"] += 1
                    else:
                        profile["failure_count"] += 1
                    
                    # Log performance data
                    self.logger.debug(
                        f"Function {name} executed",
                        extra={
                            "execution_time_ms": execution_time * 1000,
                            "memory_delta_mb": memory_delta,
                            "success": success
                        }
                    )
                
                return result
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def start_performance_monitor(self, name: str, interval: float = 1.0):
        """Start monitoring performance metrics"""
        if name in self.performance_monitors:
            self.stop_performance_monitor(name)
        
        def monitor():
            while name in self.performance_monitors:
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_percent = psutil.virtual_memory().percent
                    disk_percent = psutil.disk_usage('/').percent
                    
                    self.logger.debug(
                        f"Performance metrics for {name}",
                        extra={
                            "cpu_percent": cpu_percent,
                            "memory_percent": memory_percent,
                            "disk_percent": disk_percent,
                            "timestamp": time.time()
                        }
                    )
                    
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Performance monitor {name} error: {e}")
                    break
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.performance_monitors[name] = monitor_thread
        monitor_thread.start()
    
    def stop_performance_monitor(self, name: str):
        """Stop a performance monitor"""
        if name in self.performance_monitors:
            del self.performance_monitors[name]
    
    def start_resource_monitor(self, name: str, interval: float = 5.0):
        """Start monitoring system resources"""
        if name in self.resource_monitors:
            self.stop_resource_monitor(name)
        
        def monitor():
            while name in self.resource_monitors:
                try:
                    # Get detailed system information
                    cpu_freq = psutil.cpu_freq()
                    memory = psutil.virtual_memory()
                    swap = psutil.swap_memory()
                    disk_io = psutil.disk_io_counters()
                    net_io = psutil.net_io_counters()
                    
                    resource_data = {
                        "cpu_count": psutil.cpu_count(),
                        "cpu_percent": psutil.cpu_percent(percpu=True),
                        "cpu_freq_current": cpu_freq.current if cpu_freq else None,
                        "cpu_freq_max": cpu_freq.max if cpu_freq else None,
                        "memory_total": memory.total,
                        "memory_available": memory.available,
                        "memory_percent": memory.percent,
                        "swap_total": swap.total,
                        "swap_used": swap.used,
                        "swap_percent": swap.percent,
                        "disk_total": psutil.disk_usage('/').total,
                        "disk_used": psutil.disk_usage('/').used,
                        "disk_free": psutil.disk_usage('/').free,
                        "disk_percent": psutil.disk_usage('/').percent,
                        "process_count": len(psutil.pids()),
                        "boot_time": psutil.boot_time(),
                        "timestamp": time.time()
                    }
                    
                    self.diagnostics[f"resource_{name}"] = resource_data
                    
                    self.logger.debug(
                        f"Resource metrics for {name}",
                        extra=resource_data
                    )
                    
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Resource monitor {name} error: {e}")
                    break
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.resource_monitors[name] = monitor_thread
        monitor_thread.start()
    
    def stop_resource_monitor(self, name: str):
        """Stop a resource monitor"""
        if name in self.resource_monitors:
            del self.resource_monitors[name]
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics"""
        diagnostics = {
            "timestamp": time.time(),
            "system_info": self._get_system_info(),
            "process_info": self._get_process_info(),
            "performance_stats": self._get_performance_stats(),
            "profile_data": self._get_profile_data(),
            "log_stats": self._get_log_stats()
        }
        
        self.diagnostics.update(diagnostics)
        return diagnostics
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            uname = os.uname()
            return {
                "system": uname.sysname,
                "node": uname.nodename,
                "release": uname.release,
                "version": uname.version,
                "machine": uname.machine,
                "platform": sys.platform,
                "python_version": sys.version,
                "python_implementation": sys.implementation.name,
                "cwd": os.getcwd(),
                "home_dir": os.path.expanduser("~"),
                "temp_dir": tempfile.gettempdir()
            }
        except:
            return {"error": "Could not get system info"}
    
    def _get_process_info(self) -> Dict[str, Any]:
        """Get process information"""
        try:
            process = psutil.Process(os.getpid())
            return {
                "pid": process.pid,
                "ppid": process.ppid(),
                "status": process.status(),
                "username": process.username(),
                "create_time": process.create_time(),
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None,
                "memory_info": process.memory_info()._asdict(),
                "cpu_times": process.cpu_times()._asdict(),
                "open_files": [f.path for f in process.open_files()] if process.open_files() else [],
                "connections": [conn._asdict() for conn in process.connections()]
            }
        except:
            return {"error": "Could not get process info"}
    
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "boot_time": psutil.boot_time(),
            "uptime_seconds": time.time() - psutil.boot_time()
        }
    
    def _get_profile_data(self) -> Dict[str, Any]:
        """Get profiling data"""
        return self._profile_data
    
    def _get_log_stats(self) -> Dict[str, Any]:
        """Get log statistics"""
        return {
            "log_count": len(getattr(self.logger, 'debug_logs', [])),
            "logger_name": self.logger.name,
            "log_directory": str(self.logger.log_dir)
        }
    
    def export_diagnostics(self, output_dir: Path) -> Path:
        """Export diagnostics to a file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run diagnostics
        diag_data = self.run_diagnostics()
        
        # Write to file
        diag_file = output_dir / f"diagnostics_{int(time.time())}.json"
        with open(diag_file, 'w') as f:
            json.dump(diag_data, f, indent=2, default=str)
        
        # Also export logs
        log_file = output_dir / f"logs_{int(time.time())}.log.gz"
        self.logger.export_logs(log_file.parent / log_file.name.replace('.gz', ''), compress=True)
        
        return diag_file


class DiagnosticDecorator:
    """Class for diagnostic decorators"""
    
    def __init__(self, debugging_manager: DebuggingManager):
        self.debugging_manager = debugging_manager
    
    def trace_calls(self, func):
        """Decorator to trace function calls"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            self.debugging_manager.logger.debug(
                f"Entering {func_name}",
                extra={
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                    "args_types": [type(arg).__name__ for arg in args],
                    "kwargs_keys": list(kwargs.keys())
                }
            )
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                self.debugging_manager.logger.debug(
                    f"Exiting {func_name}",
                    extra={
                        "duration_ms": duration * 1000,
                        "result_type": type(result).__name__ if result is not None else "None"
                    }
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                self.debugging_manager.logger.error(
                    f"Exception in {func_name}",
                    extra={
                        "duration_ms": duration * 1000,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e)
                    }
                )
                raise
        
        return wrapper


# Global debugging manager instance
_debugging_manager: Optional[DebuggingManager] = None


def get_debugging_manager() -> DebuggingManager:
    """Get the global debugging manager instance"""
    global _debugging_manager
    if _debugging_manager is None:
        _debugging_manager = DebuggingManager()
    return _debugging_manager


def debug_function(func):
    """Decorator to add debugging to a function"""
    dm = get_debugging_manager()
    return dm.profile_function()(func)


def trace_function(func):
    """Decorator to trace a function"""
    dm = get_debugging_manager()
    dd = DiagnosticDecorator(dm)
    return dd.trace_calls(func)


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        
        # Get debugging manager
        dm = get_debugging_manager()
        
        print("Testing Debugging and Logging System...")
        
        # Enable profiling
        dm.enable_profiling()
        
        # Start monitors
        dm.start_performance_monitor("main", interval=2.0)
        dm.start_resource_monitor("system", interval=5.0)
        
        # Test profiling decorator
        @dm.profile_function("test_function")
        def test_function(x, y):
            time.sleep(0.1)  # Simulate work
            return x + y
        
        # Test tracing decorator
        dd = DiagnosticDecorator(dm)
        
        @dd.trace_calls
        def traced_function(a, b):
            return a * b
        
        # Run some tests
        print("\n1. Testing profiled function:")
        for i in range(3):
            result = test_function(i, i+1)
            print(f"  test_function({i}, {i+1}) = {result}")
        
        print("\n2. Testing traced function:")
        for i in range(2):
            result = traced_function(i, i+2)
            print(f"  traced_function({i}, {i+2}) = {result}")
        
        # Wait a bit for monitors to collect data
        time.sleep(3)
        
        # Run diagnostics
        print("\n3. Running diagnostics:")
        diagnostics = dm.run_diagnostics()
        print(f"  System: {diagnostics['system_info']['system']}")
        print(f"  CPU usage: {diagnostics['performance_stats']['cpu_percent']}%")
        print(f"  Memory usage: {diagnostics['performance_stats']['memory_percent']}%")
        
        # Show profile data
        print("\n4. Profile data:")
        for func_name, stats in diagnostics['profile_data'].items():
            print(f"  {func_name}: {stats['call_count']} calls, "
                  f"avg {stats['avg_time']*1000:.2f}ms, "
                  f"{stats['success_count']} successes")
        
        # Export diagnostics
        print("\n5. Exporting diagnostics:")
        export_path = dm.export_diagnostics(temp_path)
        print(f"  Diagnostics exported to: {export_path}")
        
        # Stop monitors
        dm.stop_performance_monitor("main")
        dm.stop_resource_monitor("system")
        
        print("\nDebugging system test completed!")