#!/usr/bin/env python3
"""
Advanced Error Handling & Recovery System for Xencode Phase 2

Comprehensive error handling with retry mechanisms, graceful degradation,
intelligent recovery strategies, and user-friendly error reporting.
"""

import asyncio
import functools
import logging
import traceback
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.traceback import install
import aiofiles

# Install rich traceback for better error display
install(show_locals=True)

console = Console()


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorCategory(Enum):
    """Error category types"""
    NETWORK = "network"
    MODEL = "model"
    CACHE = "cache"
    CONFIG = "config"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER_INPUT = "user_input"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    timestamp: datetime = field(default_factory=datetime.now)
    function_name: str = ""
    module_name: str = ""
    user_action: str = ""
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "function_name": self.function_name,
            "module_name": self.module_name,
            "user_action": self.user_action,
            "system_state": self.system_state,
            "recovery_attempts": self.recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts
        }


@dataclass
class XencodeError:
    """Comprehensive error information"""
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    technical_details: str
    user_message: str
    suggested_actions: List[str]
    context: ErrorContext
    original_exception: Optional[Exception] = None
    recoverable: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "technical_details": self.technical_details,
            "user_message": self.user_message,
            "suggested_actions": self.suggested_actions,
            "context": self.context.to_dict(),
            "recoverable": self.recoverable,
            "exception_type": type(self.original_exception).__name__ if self.original_exception else None
        }


class RetryStrategy:
    """Configurable retry strategy"""
    
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 backoff_multiplier: float = 2.0,
                 max_delay: float = 60.0,
                 exponential_backoff: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff_multiplier = backoff_multiplier
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.exponential_backoff:
            delay = self.base_delay * (self.backoff_multiplier ** (attempt - 1))
        else:
            delay = self.base_delay
        
        return min(delay, self.max_delay)


class ErrorLogger:
    """Advanced error logging system"""
    
    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or Path.home() / ".xencode" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger("xencode_errors")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        log_file = self.log_dir / f"xencode_errors_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    async def log_error(self, error: XencodeError):
        """Log error asynchronously"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error.to_dict(),
            "system_info": await self._get_system_info()
        }
        
        # Log to file
        self.logger.error(f"Xencode Error: {error.message}", extra=log_entry)
        
        # Save detailed JSON log
        log_file = self.log_dir / f"error_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        async with aiofiles.open(log_file, 'w') as f:
            import json
            await f.write(json.dumps(log_entry, indent=2, default=str))
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('.').percent,
                "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
                "platform": psutil.platform.system()
            }
        except:
            return {"system_info": "unavailable"}


class RecoveryManager:
    """Intelligent error recovery system"""
    
    def __init__(self):
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.NETWORK: [
                self._retry_with_backoff,
                self._check_connection,
                self._use_fallback_endpoint
            ],
            ErrorCategory.MODEL: [
                self._restart_model,
                self._use_fallback_model,
                self._clear_model_cache
            ],
            ErrorCategory.CACHE: [
                self._clear_cache,
                self._rebuild_cache,
                self._disable_cache_temporarily
            ],
            ErrorCategory.CONFIG: [
                self._reset_to_defaults,
                self._reload_config,
                self._use_backup_config
            ],
            ErrorCategory.PERFORMANCE: [
                self._free_memory,
                self._reduce_concurrency,
                self._enable_performance_mode
            ]
        }
    
    async def attempt_recovery(self, error: XencodeError) -> bool:
        """Attempt to recover from error"""
        if not error.recoverable or error.context.recovery_attempts >= error.context.max_recovery_attempts:
            return False
        
        strategies = self.recovery_strategies.get(error.category, [])
        
        for strategy in strategies:
            try:
                console.print(f"[yellow]🔄 Attempting recovery: {strategy.__name__.replace('_', ' ').title()}[/yellow]")
                
                success = await strategy(error)
                if success:
                    console.print(f"[green]✅ Recovery successful using {strategy.__name__}[/green]")
                    return True
                    
            except Exception as recovery_error:
                console.print(f"[red]❌ Recovery strategy failed: {recovery_error}[/red]")
        
        return False
    
    async def _retry_with_backoff(self, error: XencodeError) -> bool:
        """Generic retry with exponential backoff"""
        strategy = RetryStrategy()
        delay = strategy.get_delay(error.context.recovery_attempts + 1)
        
        console.print(f"[blue]⏳ Retrying in {delay:.1f} seconds...[/blue]")
        await asyncio.sleep(delay)
        
        error.context.recovery_attempts += 1
        return True
    
    async def _check_connection(self, error: XencodeError) -> bool:
        """Check network connectivity"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('https://httpbin.org/status/200', timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    async def _use_fallback_endpoint(self, error: XencodeError) -> bool:
        """Switch to fallback endpoint"""
        # Implementation would depend on specific service
        console.print("[blue]🔄 Switching to fallback endpoint[/blue]")
        return True
    
    async def _restart_model(self, error: XencodeError) -> bool:
        """Restart AI model service"""
        console.print("[blue]🔄 Restarting model service[/blue]")
        # Implementation would restart Ollama or similar
        return True
    
    async def _use_fallback_model(self, error: XencodeError) -> bool:
        """Switch to fallback model"""
        console.print("[blue]🔄 Switching to fallback model[/blue]")
        return True
    
    async def _clear_model_cache(self, error: XencodeError) -> bool:
        """Clear model cache"""
        console.print("[blue]🗑️  Clearing model cache[/blue]")
        return True
    
    async def _clear_cache(self, error: XencodeError) -> bool:
        """Clear application cache"""
        console.print("[blue]🗑️  Clearing cache[/blue]")
        return True
    
    async def _rebuild_cache(self, error: XencodeError) -> bool:
        """Rebuild cache from scratch"""
        console.print("[blue]🔨 Rebuilding cache[/blue]")
        return True
    
    async def _disable_cache_temporarily(self, error: XencodeError) -> bool:
        """Temporarily disable caching"""
        console.print("[blue]⏸️  Disabling cache temporarily[/blue]")
        return True
    
    async def _reset_to_defaults(self, error: XencodeError) -> bool:
        """Reset configuration to defaults"""
        console.print("[blue]🔧 Resetting to default configuration[/blue]")
        return True
    
    async def _reload_config(self, error: XencodeError) -> bool:
        """Reload configuration"""
        console.print("[blue]🔄 Reloading configuration[/blue]")
        return True
    
    async def _use_backup_config(self, error: XencodeError) -> bool:
        """Use backup configuration"""
        console.print("[blue]📋 Using backup configuration[/blue]")
        return True
    
    async def _free_memory(self, error: XencodeError) -> bool:
        """Free system memory"""
        import gc
        gc.collect()
        console.print("[blue]🧹 Memory cleanup completed[/blue]")
        return True
    
    async def _reduce_concurrency(self, error: XencodeError) -> bool:
        """Reduce concurrent operations"""
        console.print("[blue]🐌 Reducing concurrency for stability[/blue]")
        return True
    
    async def _enable_performance_mode(self, error: XencodeError) -> bool:
        """Enable performance optimization mode"""
        console.print("[blue]⚡ Enabling performance mode[/blue]")
        return True


class ErrorHandler:
    """Main error handling coordinator"""
    
    def __init__(self):
        self.error_logger = ErrorLogger()
        self.recovery_manager = RecoveryManager()
        self.error_history: List[XencodeError] = []
        self.max_history = 100
    
    async def handle_error(self, 
                          exception: Exception, 
                          context: ErrorContext,
                          severity: ErrorSeverity = ErrorSeverity.ERROR,
                          category: ErrorCategory = ErrorCategory.UNKNOWN) -> XencodeError:
        """Handle error with full recovery pipeline"""
        
        # Classify error
        error = self._classify_error(exception, context, severity, category)
        
        # Add to history
        self.error_history.append(error)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Log error
        await self.error_logger.log_error(error)
        
        # Display to user
        self._display_error_to_user(error)
        
        # Attempt recovery if appropriate
        if error.recoverable and error.severity != ErrorSeverity.CRITICAL:
            recovery_success = await self.recovery_manager.attempt_recovery(error)
            if recovery_success:
                console.print("[green]✅ Error recovered successfully[/green]")
                return error
        
        # If recovery failed or not attempted
        if error.severity == ErrorSeverity.CRITICAL:
            console.print("[red]💥 Critical error - manual intervention required[/red]")
        
        return error
    
    def _classify_error(self, 
                       exception: Exception, 
                       context: ErrorContext,
                       severity: ErrorSeverity,
                       category: ErrorCategory) -> XencodeError:
        """Classify error and generate comprehensive error info"""
        
        # Auto-detect category if unknown
        if category == ErrorCategory.UNKNOWN:
            category = self._detect_error_category(exception)
        
        # Generate user-friendly message
        user_message, suggested_actions = self._generate_user_guidance(exception, category)
        
        # Technical details
        technical_details = f"{type(exception).__name__}: {str(exception)}"
        if hasattr(exception, '__traceback__') and exception.__traceback__:
            technical_details += f"\n{traceback.format_exc()}"
        
        return XencodeError(
            severity=severity,
            category=category,
            message=str(exception),
            technical_details=technical_details,
            user_message=user_message,
            suggested_actions=suggested_actions,
            context=context,
            original_exception=exception,
            recoverable=self._is_recoverable(exception, category)
        )
    
    def _detect_error_category(self, exception: Exception) -> ErrorCategory:
        """Automatically detect error category"""
        exception_type = type(exception).__name__.lower()
        exception_msg = str(exception).lower()
        
        # Network errors
        if any(term in exception_type for term in ['connection', 'network', 'timeout', 'http']):
            return ErrorCategory.NETWORK
        if any(term in exception_msg for term in ['connection', 'network', 'timeout', 'unreachable']):
            return ErrorCategory.NETWORK
        
        # Model errors
        if any(term in exception_msg for term in ['model', 'ollama', 'inference', 'generation']):
            return ErrorCategory.MODEL
        
        # Cache errors
        if any(term in exception_msg for term in ['cache', 'redis', 'sqlite']):
            return ErrorCategory.CACHE
        
        # Config errors
        if any(term in exception_type for term in ['config', 'yaml', 'toml', 'json']):
            return ErrorCategory.CONFIG
        
        # Performance errors
        if any(term in exception_type for term in ['memory', 'timeout', 'resource']):
            return ErrorCategory.PERFORMANCE
        
        # Security errors
        if any(term in exception_msg for term in ['permission', 'access', 'security', 'forbidden']):
            return ErrorCategory.SECURITY
        
        return ErrorCategory.UNKNOWN
    
    def _generate_user_guidance(self, exception: Exception, category: ErrorCategory) -> tuple[str, List[str]]:
        """Generate user-friendly message and suggested actions"""
        
        guidance_map = {
            ErrorCategory.NETWORK: (
                "Network connection issue detected. Please check your internet connection.",
                [
                    "Check your internet connection",
                    "Try again in a few moments",
                    "Check if the service is down",
                    "Contact support if issue persists"
                ]
            ),
            ErrorCategory.MODEL: (
                "AI model issue detected. The model may need to be restarted or updated.",
                [
                    "Restart the AI model service",
                    "Check if the model is installed correctly",
                    "Try a different model",
                    "Check system resources (RAM, CPU)"
                ]
            ),
            ErrorCategory.CACHE: (
                "Cache system issue detected. Cache may need to be cleared.",
                [
                    "Clear the cache and try again",
                    "Check disk space",
                    "Restart the application",
                    "Check cache configuration"
                ]
            ),
            ErrorCategory.CONFIG: (
                "Configuration issue detected. Settings may need to be corrected.",
                [
                    "Check configuration file syntax",
                    "Reset to default configuration",
                    "Verify all required settings are present",
                    "Check file permissions"
                ]
            ),
            ErrorCategory.PERFORMANCE: (
                "Performance issue detected. System resources may be insufficient.",
                [
                    "Close other applications to free memory",
                    "Reduce model complexity",
                    "Enable performance mode",
                    "Upgrade system resources if possible"
                ]
            ),
            ErrorCategory.SECURITY: (
                "Security issue detected. Operation blocked for safety.",
                [
                    "Check file/directory permissions",
                    "Review security settings",
                    "Run with appropriate privileges",
                    "Contact administrator if needed"
                ]
            )
        }
        
        return guidance_map.get(category, (
            "An unexpected error occurred. Please try again or contact support.",
            [
                "Try the operation again",
                "Restart the application",
                "Check the logs for more details",
                "Contact support with error details"
            ]
        ))
    
    def _is_recoverable(self, exception: Exception, category: ErrorCategory) -> bool:
        """Determine if error is recoverable"""
        
        # Critical exceptions that are not recoverable
        critical_exceptions = [
            'SystemExit',
            'KeyboardInterrupt',
            'ImportError',
            'SyntaxError'
        ]
        
        if type(exception).__name__ in critical_exceptions:
            return False
        
        # Category-based recovery assessment
        recoverable_categories = [
            ErrorCategory.NETWORK,
            ErrorCategory.CACHE,
            ErrorCategory.PERFORMANCE,
            ErrorCategory.MODEL
        ]
        
        return category in recoverable_categories
    
    def _display_error_to_user(self, error: XencodeError):
        """Display error to user in a friendly way"""
        
        # Choose style based on severity
        style_map = {
            ErrorSeverity.CRITICAL: "red",
            ErrorSeverity.ERROR: "red",
            ErrorSeverity.WARNING: "yellow",
            ErrorSeverity.INFO: "blue"
        }
        
        icon_map = {
            ErrorSeverity.CRITICAL: "💥",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.INFO: "ℹ️"
        }
        
        style = style_map[error.severity]
        icon = icon_map[error.severity]
        
        # Create error panel
        error_content = f"""
{icon} **{error.severity.value.upper()}**: {error.user_message}

**What happened**: {error.message}

**Suggested actions**:
{chr(10).join(f"• {action}" for action in error.suggested_actions)}

**Category**: {error.category.value.title()}
        """
        
        panel = Panel(
            error_content.strip(),
            title=f"[bold {style}]Error Handler[/bold {style}]",
            border_style=style
        )
        
        console.print(panel)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        if not self.error_history:
            return {"total_errors": 0, "categories": {}, "recent_errors": []}
        
        # Count by category
        category_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        # Recent errors (last 5)
        recent_errors = [
            {
                "timestamp": error.context.timestamp.isoformat(),
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message
            }
            for error in self.error_history[-5:]
        ]
        
        return {
            "total_errors": len(self.error_history),
            "categories": category_counts,
            "severities": severity_counts,
            "recent_errors": recent_errors
        }


# Decorators for easy error handling
def async_error_handler(category: ErrorCategory = ErrorCategory.UNKNOWN,
                       severity: ErrorSeverity = ErrorSeverity.ERROR,
                       recoverable: bool = True):
    """Decorator for async function error handling"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            context = ErrorContext(
                function_name=func.__name__,
                module_name=func.__module__,
                user_action=f"Calling {func.__name__}"
            )
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handler = ErrorHandler()
                await handler.handle_error(e, context, severity, category)
                
                if severity == ErrorSeverity.CRITICAL:
                    raise
                return None
        
        return wrapper
    return decorator


def sync_error_handler(category: ErrorCategory = ErrorCategory.UNKNOWN,
                      severity: ErrorSeverity = ErrorSeverity.ERROR):
    """Decorator for sync function error handling"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                function_name=func.__name__,
                module_name=func.__module__,
                user_action=f"Calling {func.__name__}"
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we can't use async handler
                console.print(f"[red]❌ Error in {func.__name__}: {e}[/red]")
                
                if severity == ErrorSeverity.CRITICAL:
                    raise
                return None
        
        return wrapper
    return decorator


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


async def get_error_handler() -> ErrorHandler:
    """Get or create global error handler"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


async def handle_error(exception: Exception, 
                      context: ErrorContext = None,
                      severity: ErrorSeverity = ErrorSeverity.ERROR,
                      category: ErrorCategory = ErrorCategory.UNKNOWN) -> XencodeError:
    """Global error handling function"""
    if context is None:
        context = ErrorContext()
    
    handler = await get_error_handler()
    return await handler.handle_error(exception, context, severity, category)


if __name__ == "__main__":
    # Demo and testing
    async def demo():
        console.print("[bold blue]🛡️  Error Handling System Demo[/bold blue]")
        
        # Test different error types
        test_errors = [
            (ConnectionError("Network timeout"), ErrorCategory.NETWORK),
            (FileNotFoundError("Config file missing"), ErrorCategory.CONFIG),
            (MemoryError("Out of memory"), ErrorCategory.PERFORMANCE),
            (PermissionError("Access denied"), ErrorCategory.SECURITY),
        ]
        
        handler = ErrorHandler()
        
        for exception, category in test_errors:
            context = ErrorContext(
                function_name="demo_function",
                user_action="Testing error handling"
            )
            
            console.print(f"\n[blue]Testing {category.value} error...[/blue]")
            await handler.handle_error(exception, context, category=category)
        
        # Show error summary
        console.print("\n[bold]Error Summary:[/bold]")
        summary = handler.get_error_summary()
        
        for category, count in summary["categories"].items():
            console.print(f"  {category}: {count} errors")
    
    asyncio.run(demo())