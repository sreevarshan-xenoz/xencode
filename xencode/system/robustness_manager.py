#!/usr/bin/env python3
"""
Comprehensive Error Handling and Robustness for Xencode Warp Terminal

Implements advanced error handling, timeout management, and recovery mechanisms
for reliable terminal operation.
"""

import asyncio
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import logging
import subprocess
import json
from datetime import datetime

from ..warp_terminal import WarpTerminal, LazyCommandBlock
from ..system.session_manager import get_session_manager


logger = logging.getLogger(__name__)


class CommandTimeoutException(Exception):
    """Raised when a command exceeds its timeout limit"""
    pass


class CommandExecutionError(Exception):
    """Raised when command execution fails"""
    pass


class RecoveryManager:
    """Manages recovery from various failure scenarios"""
    
    def __init__(self):
        self.failure_log = []
        self.max_failures_to_track = 100
        
    def log_failure(self, command: str, error: Exception, context: Dict[str, Any] = None):
        """Log a failure for later analysis"""
        failure_record = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.failure_log.append(failure_record)
        
        # Keep only the most recent failures
        if len(self.failure_log) > self.max_failures_to_track:
            self.failure_log = self.failure_log[-self.max_failures_to_track:]
        
        logger.error(f"Command failed: {command}, Error: {error}")
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get a summary of recent failures"""
        if not self.failure_log:
            return {"total_failures": 0}
        
        error_types = {}
        commands_failed = set()
        
        for failure in self.failure_log:
            error_type = failure["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
            commands_failed.add(failure["command"])
        
        return {
            "total_failures": len(self.failure_log),
            "unique_commands_failed": len(commands_failed),
            "error_distribution": error_types,
            "most_recent_failure": self.failure_log[-1] if self.failure_log else None
        }


class RobustWarpTerminal(WarpTerminal):
    """Enhanced WarpTerminal with comprehensive error handling and robustness features"""
    
    def __init__(self, ai_suggester: Optional[Callable] = None, max_blocks: int = 20):
        super().__init__(ai_suggester, max_blocks)
        self.timeout_seconds = 60  # Default timeout for commands
        self.recovery_manager = RecoveryManager()
        self.session_manager = get_session_manager()
        self.command_retry_attempts = 3
        self.health_check_interval = 30  # seconds
        self.last_health_check = time.time()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    
    def run_command_with_timeout(self, command: str, timeout: Optional[int] = None) -> LazyCommandBlock:
        """Execute a command with timeout handling"""
        if timeout is None:
            timeout = self.timeout_seconds
            
        start_time = time.time()
        
        # Create a block immediately
        block = LazyCommandBlock(
            id=f"cmd_{len(self.command_blocks)+1}",
            command=command,
            output_data={"type": "text", "data": "", "partial": True},
            metadata={"exit_code": None, "duration_ms": None},
            tags=[command.split()[0] if command.split() else "unknown"]
        )
        
        # Add to blocks immediately
        self.command_blocks.append(block)
        
        def execute_with_timeout():
            process = None
            try:
                # Start the process
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                # Wait for completion with timeout
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    exit_code = process.returncode
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                    exit_code = -1
                    raise CommandTimeoutException(f"Command timed out after {timeout} seconds")
                
                # Calculate duration
                duration_ms = int((time.time() - start_time) * 1000)
                
                # Parse the output
                from .warp_terminal import StreamingOutputParser
                output_parser = StreamingOutputParser()
                parsed_output = output_parser.parse_output(command, stdout, exit_code)
                
                # Update block with results
                block.output_data = parsed_output
                block.metadata = {
                    "exit_code": exit_code,
                    "duration_ms": duration_ms,
                    "error": stderr if stderr else None,
                    "timeout": False
                }
                
                # Update tags
                block.tags = self._generate_tags(command, parsed_output)
                
                # Save to session
                self.session_manager.save_command_block(block)
                
            except CommandTimeoutException as e:
                duration_ms = int((time.time() - start_time) * 1000)
                
                block.output_data = {"type": "error", "data": str(e), "partial": False}
                block.metadata = {
                    "exit_code": -1,
                    "duration_ms": duration_ms,
                    "error": str(e),
                    "timeout": True
                }
                block.tags = ["timeout", "error"]
                
                self.recovery_manager.log_failure(command, e, {"timeout": timeout})
                
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                
                block.output_data = {"type": "error", "data": str(e), "partial": False}
                block.metadata = {
                    "exit_code": -1,
                    "duration_ms": duration_ms,
                    "error": str(e),
                    "timeout": False
                }
                block.tags = ["error"]
                
                self.recovery_manager.log_failure(command, e, {"duration": duration_ms})
                
            finally:
                if process:
                    try:
                        process.terminate()
                        process.wait(timeout=1)
                    except Exception:
                        pass
        
        # Execute in background thread
        thread = threading.Thread(target=execute_with_timeout)
        thread.daemon = True
        thread.start()
        
        return block
    
    def run_command_with_retry(self, command: str, max_retries: Optional[int] = None) -> LazyCommandBlock:
        """Execute a command with automatic retry on failure"""
        if max_retries is None:
            max_retries = self.command_retry_attempts
            
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt} for command: {command}")
                    time.sleep(min(2 ** attempt, 10))  # Exponential backoff with cap
                
                block = self.run_command_with_timeout(command)
                
                # Wait briefly to see if the command completed successfully
                time.sleep(0.1)
                
                # Check if the command was successful
                if block.metadata.get("exit_code") == 0:
                    return block
                elif block.metadata.get("timeout"):
                    # Don't retry timeouts
                    return block
                else:
                    # Command failed, continue to retry
                    continue
                    
            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    # Final attempt failed, log and return error block
                    self.recovery_manager.log_failure(command, e, {"attempts": attempt + 1})
                    
                    # Create error block
                    error_block = LazyCommandBlock(
                        id=f"cmd_{len(self.command_blocks)+1}",
                        command=command,
                        output_data={"type": "error", "data": str(e), "partial": False},
                        metadata={
                            "exit_code": -1,
                            "duration_ms": 0,
                            "error": str(e),
                            "retries_exhausted": True
                        },
                        tags=["error", "retry_failed"]
                    )
                    self.command_blocks.append(error_block)
                    return error_block
        
        # This shouldn't happen, but just in case
        error_block = LazyCommandBlock(
            id=f"cmd_{len(self.command_blocks)+1}",
            command=command,
            output_data={"type": "error", "data": f"Command failed after {max_retries} retries", "partial": False},
            metadata={
                "exit_code": -1,
                "duration_ms": 0,
                "error": str(last_exception) if last_exception else "Unknown error",
                "retries_exhausted": True
            },
            tags=["error", "retry_failed"]
        )
        self.command_blocks.append(error_block)
        return error_block
    
    def run_command(self, command: str) -> LazyCommandBlock:
        """Override parent method to use robust execution"""
        return self.run_command_with_retry(command)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the terminal"""
        current_time = time.time()
        
        # Check if we should run a health check
        if current_time - self.last_health_check < self.health_check_interval:
            return {"healthy": True, "last_check": self.last_health_check}
        
        self.last_health_check = current_time
        
        # Perform various health checks
        checks = {
            "session_manager_available": self.session_manager is not None,
            "command_queue_size": len(self.command_blocks),
            "recent_failures": len(self.recovery_manager.failure_log),
            "memory_usage_approx": len(self.command_blocks) * 1024,  # Rough estimate
        }
        
        # Check for any stuck processes (this is a simplified check)
        # In a real implementation, you'd want to check for actual stuck processes
        
        # Get failure summary
        failure_summary = self.recovery_manager.get_failure_summary()
        
        overall_healthy = (
            checks["command_queue_size"] <= self.command_blocks.maxlen and
            failure_summary.get("total_failures", 0) < 10  # Less than 10 failures
        )
        
        health_report = {
            "healthy": overall_healthy,
            "checks": checks,
            "failure_summary": failure_summary,
            "timestamp": current_time
        }
        
        if not overall_healthy:
            logger.warning(f"Health check failed: {health_report}")
        
        return health_report
    
    def recover_from_crash(self, session_id: str) -> List[LazyCommandBlock]:
        """Attempt to recover from a crashed session"""
        logger.info(f"Attempting to recover from session: {session_id}")
        
        recovered_blocks = []
        
        try:
            # Load commands from the crashed session
            commands = self.session_manager.load_session_commands(session_id)
            
            for command_block in commands:
                # Convert to LazyCommandBlock for consistency
                lazy_block = LazyCommandBlock(
                    id=command_block.id,
                    command=command_block.command,
                    output_data=command_block.output_data,
                    metadata=command_block.metadata,
                    tags=command_block.tags
                )
                
                recovered_blocks.append(lazy_block)
                
                # Add to current command blocks
                self.command_blocks.append(lazy_block)
            
            logger.info(f"Recovered {len(recovered_blocks)} commands from session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to recover from session {session_id}: {e}")
            self.recovery_manager.log_failure(f"recover_session_{session_id}", e)
        
        return recovered_blocks
    
    def shutdown(self):
        """Gracefully shut down the terminal"""
        logger.info("Shutting down RobustWarpTerminal...")
        
        # Close the current session
        if hasattr(self, 'session_manager'):
            self.session_manager.close_current_session()
        
        # Perform final health check
        health = self.health_check()
        logger.info(f"Shutdown health report: {health}")
        
        # Log any final failures
        failure_summary = self.recovery_manager.get_failure_summary()
        if failure_summary.get("total_failures", 0) > 0:
            logger.warning(f"Shutdown with {failure_summary['total_failures']} recent failures")
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get system resource information"""
        import psutil
        
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids()),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            return {"error": str(e)}
    
    @contextmanager
    def error_context(self, operation: str, context: Dict[str, Any] = None):
        """Context manager for consistent error handling"""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            duration = time.time() - start_time
            error_context = {
                "operation": operation,
                "duration": duration,
                "context": context or {}
            }
            self.recovery_manager.log_failure(operation, e, error_context)
            raise


# Decorator for adding error handling to functions
def robust_command(func):
    """Decorator to add robust error handling to command functions"""
    def wrapper(self, *args, **kwargs):
        with self.error_context(func.__name__, {"args": args, "kwargs": kwargs}):
            return func(self, *args, **kwargs)
    return wrapper


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override session manager to use temp directory
        from .session_manager import SessionManager
        from xencode import session_manager as session_mod

        temp_db = Path(tmpdir) / "test_sessions.db"
        session_mod._session_manager = SessionManager(temp_db)
        
        # Create robust terminal
        def test_ai_suggester(recent_commands):
            return ["echo 'test suggestion'"]
        
        terminal = RobustWarpTerminal(ai_suggester=test_ai_suggester)
        
        print("Testing RobustWarpTerminal...")
        
        # Test normal command
        print("\n1. Testing normal command:")
        block = terminal.run_command("echo 'Hello, World!'")
        time.sleep(2)  # Give it time to complete
        print(f"Command: {block.command}")
        print(f"Exit code: {block.metadata.get('exit_code')}")
        print(f"Output: {block.output_data.get('data')}")
        
        # Test timeout
        print("\n2. Testing timeout handling:")
        timeout_block = terminal.run_command_with_timeout("sleep 5", timeout=1)
        time.sleep(2)
        print(f"Command: {timeout_block.command}")
        print(f"Timed out: {block.metadata.get('timeout', False)}")
        
        # Test health check
        print("\n3. Testing health check:")
        health = terminal.health_check()
        print(f"Healthy: {health['healthy']}")
        print(f"Checks: {health['checks']}")
        
        # Test system resources
        print("\n4. Testing system resources:")
        resources = terminal.get_system_resources()
        print(f"Resources: {resources}")
        
        # Test failure logging
        print("\n5. Testing failure logging:")
        try:
            # This should fail
            bad_block = terminal.run_command("nonexistent_command_that_does_not_exist")
            time.sleep(1)
        except Exception as e:
            print(f"Caught exception: {e}")
        
        # Show failure summary
        failure_summary = terminal.recovery_manager.get_failure_summary()
        print(f"Failure summary: {failure_summary}")
        
        # Shutdown
        terminal.shutdown()
        print("\nShutdown complete.")