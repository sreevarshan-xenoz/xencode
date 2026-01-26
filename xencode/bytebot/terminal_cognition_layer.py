"""
TerminalCognitionLayer - The actual implementation layer that handles terminal interactions

This layer sits between the ByteBotEngine and the actual terminal/shell operations,
providing a structured interface for command execution, input/output handling,
and terminal state management.
"""

import os
import sys
import subprocess
import threading
import queue
import time
from typing import Dict, Any, Callable, Optional, Generator
from dataclasses import dataclass
from datetime import datetime
import json
import tempfile

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

from ..core import ModelManager
from ..shell_genie.genie import ShellGenie


@dataclass
class TerminalState:
    """Represents the current state of the terminal"""
    current_directory: str
    environment_vars: Dict[str, str]
    user: str
    host: str
    shell_type: str
    timestamp: datetime


@dataclass
class CommandResult:
    """Result of a command execution"""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    timestamp: datetime
    success: bool


class TerminalCognitionLayer:
    """
    The TerminalCognitionLayer handles all terminal interactions,
    command execution, and state management for ByteBot operations.
    """
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager or ModelManager()
        self.shell_genie = ShellGenie(model_name=self.model_manager.current_model)
        self.console = Console()
        self.state = self._get_current_state()
        self.command_history = []
        self.output_queue = queue.Queue()
        self.is_executing = False
        
    def _get_current_state(self) -> TerminalState:
        """Get the current terminal state"""
        import getpass
        import socket
        
        return TerminalState(
            current_directory=os.getcwd(),
            environment_vars=dict(os.environ),
            user=getpass.getuser(),
            host=socket.gethostname(),
            shell_type="PowerShell" if os.name == 'nt' else "bash",
            timestamp=datetime.now()
        )
    
    def update_state(self) -> TerminalState:
        """Update and return the current terminal state"""
        self.state = self._get_current_state()
        return self.state
    
    def execute_command_safe(self, command: str, mode: str = "execute") -> CommandResult:
        """
        Safely execute a command with proper error handling and state tracking
        
        Args:
            command: The command to execute
            mode: Execution mode ('assist', 'execute', 'autonomous')
            
        Returns:
            CommandResult with execution details
        """
        if not command or command == "SAFE_GUARD_TRIGGERED":
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="Command was blocked by safety guard",
                duration=0.0,
                timestamp=datetime.now(),
                success=False
            )
        
        start_time = time.time()
        self.is_executing = True
        
        try:
            # Update state before execution
            self.update_state()
            
            # Execute the command
            if mode == "assist":
                # In assist mode, just return what would be executed
                return CommandResult(
                    command=command,
                    exit_code=0,
                    stdout=f"Would execute: {command}",
                    stderr="",
                    duration=time.time() - start_time,
                    timestamp=datetime.now(),
                    success=True
                )
            else:
                # Actually execute the command
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout
                )
                
                duration = time.time() - start_time
                success = result.returncode == 0
                
                cmd_result = CommandResult(
                    command=command,
                    exit_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    duration=duration,
                    timestamp=datetime.now(),
                    success=success
                )
                
                # Add to history
                self.command_history.append(cmd_result)
                
                return cmd_result
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="Command timed out after 30 seconds",
                duration=duration,
                timestamp=datetime.now(),
                success=False
            )
        except Exception as e:
            duration = time.time() - start_time
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                duration=duration,
                timestamp=datetime.now(),
                success=False
            )
        finally:
            self.is_executing = False
            # Update state after execution
            self.update_state()
    
    def execute_command_streaming(self, command: str, callback: Optional[Callable[[str], None]] = None) -> CommandResult:
        """
        Execute a command with streaming output
        
        Args:
            command: The command to execute
            callback: Optional callback to receive output as it arrives
            
        Returns:
            CommandResult with execution details
        """
        if not command or command == "SAFE_GUARD_TRIGGERED":
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="Command was blocked by safety guard",
                duration=0.0,
                timestamp=datetime.now(),
                success=False
            )
        
        start_time = time.time()
        self.is_executing = True
        
        stdout_buffer = []
        stderr_buffer = []
        
        try:
            # Update state before execution
            self.update_state()
            
            # Create process with streaming
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output streams
            def read_stream(stream, buffer, is_stdout=True):
                for line in iter(stream.readline, ''):
                    buffer.append(line)
                    if callback:
                        callback(f"[STDOUT] {line}" if is_stdout else f"[STDERR] {line}")
                stream.close()
            
            # Create threads to read stdout and stderr
            stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_buffer, True))
            stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_buffer, False))
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Wait for threads to finish
            stdout_thread.join(timeout=5)  # 5 second timeout
            stderr_thread.join(timeout=5)  # 5 second timeout
            
            duration = time.time() - start_time
            success = return_code == 0
            
            cmd_result = CommandResult(
                command=command,
                exit_code=return_code,
                stdout=''.join(stdout_buffer),
                stderr=''.join(stderr_buffer),
                duration=duration,
                timestamp=datetime.now(),
                success=success
            )
            
            # Add to history
            self.command_history.append(cmd_result)
            
            return cmd_result
            
        except Exception as e:
            duration = time.time() - start_time
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                duration=duration,
                timestamp=datetime.now(),
                success=False
            )
        finally:
            self.is_executing = False
            # Update state after execution
            self.update_state()
    
    def get_context_info(self) -> Dict[str, Any]:
        """Get contextual information for planning and execution"""
        # Get git status if in a repo
        git_info = self._get_git_info()
        
        # Get system info
        system_info = {
            "os": os.name,
            "platform": sys.platform,
            "cwd": self.state.current_directory,
            "user": self.state.user,
            "host": self.state.host,
            "shell": self.state.shell_type
        }
        
        # Get recent commands
        recent_commands = [
            {
                "command": cmd.command,
                "success": cmd.success,
                "timestamp": cmd.timestamp.isoformat()
            }
            for cmd in self.command_history[-5:]  # Last 5 commands
        ]
        
        return {
            "system": system_info,
            "git": git_info,
            "recent_commands": recent_commands,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get git repository information"""
        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                cwd=self.state.current_directory,
                timeout=5
            )
            
            if result.returncode != 0:
                return {"is_git_repo": False}
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                cwd=self.state.current_directory,
                timeout=5
            )
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
            
            # Get status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.state.current_directory,
                timeout=5
            )
            has_changes = bool(status_result.stdout.strip())
            
            # Get last commit
            commit_result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%h - %s (%cr)"],
                capture_output=True,
                text=True,
                cwd=self.state.current_directory,
                timeout=5
            )
            last_commit = commit_result.stdout.strip() if commit_result.returncode == 0 else "unknown"
            
            return {
                "is_git_repo": True,
                "current_branch": current_branch,
                "has_changes": has_changes,
                "last_commit": last_commit,
                "repo_path": os.path.dirname(result.stdout.strip())
            }
        except:
            return {"is_git_repo": False}
    
    def suggest_command(self, natural_language: str) -> Dict[str, str]:
        """
        Suggest a command based on natural language description
        
        Args:
            natural_language: Natural language description of desired action
            
        Returns:
            Dictionary with 'command' and 'explanation' keys
        """
        command, explanation = self.shell_genie.generate_command(natural_language)
        return {
            "command": command,
            "explanation": explanation
        }
    
    def display_command_result(self, result: CommandResult, show_output: bool = True):
        """
        Display command result in a user-friendly format
        """
        status_icon = "✅" if result.success else "❌"
        status_color = "green" if result.success else "red"
        
        # Create a panel showing command and result
        result_text = f"[bold]{status_icon} Command Result ({result.duration:.2f}s)[/bold]\n"
        result_text += f"[bold]Command:[/bold] {result.command}\n"
        result_text += f"[bold]Exit Code:[/bold] {result.exit_code}\n"
        
        if show_output and result.stdout:
            result_text += f"\n[bold]Output:[/bold]\n{result.stdout}"
        
        if result.stderr:
            result_text += f"\n[bold red]Errors:[/bold red]\n{result.stderr}"
        
        panel = Panel(
            result_text,
            title="Command Execution",
            border_style=status_color,
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def get_risk_indicators(self, command: str) -> Dict[str, Any]:
        """
        Analyze a command for potential risk indicators
        
        Args:
            command: The command to analyze
            
        Returns:
            Dictionary with risk indicators and severity
        """
        indicators = {
            "high_risk": [],
            "medium_risk": [],
            "low_risk": [],
            "safe": []
        }
        
        cmd_lower = command.lower()
        
        # High risk indicators
        high_risk_patterns = [
            ("rm -rf /", "Deleting root directory"),
            ("format ", "Disk formatting command"),
            ("del /f c:\\", "Force deleting system drive"),
            ("mkfs.", "File system creation (destructive)"),
            ("dd if=", "Direct disk access (potentially destructive)"),
            ("shred ", "Secure file deletion"),
            ("cat /dev/zero", "Potentially destructive disk operation")
        ]
        
        for pattern, description in high_risk_patterns:
            if pattern in cmd_lower:
                indicators["high_risk"].append({"pattern": pattern, "description": description})
        
        # Medium risk indicators
        medium_risk_patterns = [
            ("sudo ", "Privilege escalation"),
            ("rm -rf ", "Recursive deletion"),
            ("mv ", "File/directory move (can overwrite)"),
            ("chmod -R", "Recursive permission change"),
            ("chown -R", "Recursive ownership change"),
            ("kill ", "Process termination"),
            ("pkill ", "Process termination by name"),
            ("systemctl ", "System service management")
        ]
        
        for pattern, description in medium_risk_patterns:
            if pattern in cmd_lower:
                indicators["medium_risk"].append({"pattern": pattern, "description": description})
        
        # Low risk indicators
        low_risk_patterns = [
            ("ls", "List directory contents"),
            ("cat ", "File viewing"),
            ("grep ", "Text search"),
            ("find ", "File search"),
            ("echo ", "Print text"),
            ("mkdir ", "Directory creation"),
            ("touch ", "Update file timestamp/create file")
        ]
        
        for pattern, description in low_risk_patterns:
            if pattern in cmd_lower and not any(med_pattern[0] in cmd_lower for med_pattern in medium_risk_patterns):
                indicators["low_risk"].append({"pattern": pattern, "description": description})
        
        # If no risk indicators found, consider it safe
        if not indicators["high_risk"] and not indicators["medium_risk"]:
            indicators["safe"].append({"pattern": "no_high_risks", "description": "No high or medium risk patterns detected"})
        
        return indicators
    
    def validate_command(self, command: str) -> Dict[str, Any]:
        """
        Validate a command for safety before execution
        
        Args:
            command: The command to validate
            
        Returns:
            Dictionary with validation results
        """
        if not command:
            return {
                "valid": False,
                "errors": ["Empty command"],
                "warnings": [],
                "risk_level": "unknown"
            }
        
        errors = []
        warnings = []
        
        # Check for dangerous patterns
        risk_indicators = self.get_risk_indicators(command)
        
        if risk_indicators["high_risk"]:
            errors.extend([indicator["description"] for indicator in risk_indicators["high_risk"]])
        
        if risk_indicators["medium_risk"]:
            warnings.extend([indicator["description"] for indicator in risk_indicators["medium_risk"]])
        
        # Determine risk level
        if risk_indicators["high_risk"]:
            risk_level = "high"
        elif risk_indicators["medium_risk"]:
            risk_level = "medium"
        elif risk_indicators["low_risk"]:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "risk_level": risk_level,
            "indicators": risk_indicators
        }