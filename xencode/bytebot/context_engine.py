"""
Context Engine - Event-driven context management that actively influences planning

This engine monitors system events and actively updates context that influences
the planning and execution of ByteBot operations.
"""

import os
import subprocess
import threading
import time
from typing import Dict, Any, Callable, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
import tempfile
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ContextEventType(Enum):
    """Types of context events that can occur"""
    GIT_CHECKOUT = "git_checkout"
    FILE_MODIFIED = "file_modified"
    DEPENDENCY_CHANGE = "dependency_change"
    TEST_FAILURE = "test_failure"
    DIRECTORY_CHANGE = "directory_change"
    PROCESS_STARTED = "process_started"
    PROCESS_ENDED = "process_ended"
    NETWORK_ACTIVITY = "network_activity"
    SYSTEM_RESOURCE_CHANGE = "system_resource_change"


@dataclass
class ContextEvent:
    """Represents a context event"""
    event_type: ContextEventType
    data: Dict[str, Any]
    timestamp: datetime
    source: str = "system"


class ContextEngine:
    """
    Event-driven context management that actively influences planning
    """
    
    def __init__(self):
        self.context = self._initialize_context()
        self.event_handlers: List[Callable[[ContextEvent], None]] = []
        self.file_watchers = {}
        self.observer = Observer()
        self.is_monitoring = False
        self.last_git_status = {}
        self.dependency_cache = {}
        self.resource_monitor_thread = None
        self.monitoring_active = False
        
    def _initialize_context(self) -> Dict[str, Any]:
        """Initialize the base context"""
        return {
            "current_directory": os.getcwd(),
            "os_info": self._get_os_info(),
            "shell_type": "PowerShell" if os.name == 'nt' else "bash",
            "git_status": self._get_git_status(),
            "project_info": self._detect_project_type(),
            "dependencies": self._get_dependencies(),
            "system_resources": self._get_system_resources(),
            "recent_events": [],
            "environment": dict(os.environ),
            "user_info": self._get_user_info(),
            "network_status": self._get_network_status(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_os_info(self) -> Dict[str, str]:
        """Get operating system information"""
        import platform
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
    
    def _get_git_status(self) -> Dict[str, Any]:
        """Get current git repository status"""
        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return {"is_git_repo": False}
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                timeout=5
            )
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
            
            # Get status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5
            )
            has_changes = bool(status_result.stdout.strip())
            
            # Get staged files
            staged_result = subprocess.run(
                ["git", "diff", "--name-only", "--cached"],
                capture_output=True,
                text=True,
                timeout=5
            )
            staged_files = staged_result.stdout.strip().split('\n') if staged_result.stdout.strip() else []
            
            # Get untracked files
            untracked_result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                capture_output=True,
                text=True,
                timeout=5
            )
            untracked_files = untracked_result.stdout.strip().split('\n') if untracked_result.stdout.strip() else []
            
            # Get last commit
            commit_result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%H|%s|%cr"],
                capture_output=True,
                text=True,
                timeout=5
            )
            last_commit_parts = commit_result.stdout.strip().split('|') if commit_result.stdout.strip() else []
            last_commit = {
                "hash": last_commit_parts[0] if len(last_commit_parts) > 0 else "unknown",
                "message": last_commit_parts[1] if len(last_commit_parts) > 1 else "unknown",
                "relative_time": last_commit_parts[2] if len(last_commit_parts) > 2 else "unknown"
            } if last_commit_parts else {"hash": "unknown", "message": "unknown", "relative_time": "unknown"}
            
            return {
                "is_git_repo": True,
                "current_branch": current_branch,
                "has_changes": has_changes,
                "staged_files": [f for f in staged_files if f],
                "untracked_files": [f for f in untracked_files if f],
                "last_commit": last_commit,
                "repo_path": os.path.dirname(result.stdout.strip())
            }
        except Exception as e:
            return {"is_git_repo": False, "error": str(e)}
    
    def _detect_project_type(self) -> Dict[str, Any]:
        """Detect the type of project in the current directory"""
        project_types = []
        
        # Check for common project files
        files = os.listdir('.')
        
        if 'package.json' in files:
            project_types.append("nodejs")
            # Get Node.js project info
            try:
                with open('package.json', 'r') as f:
                    pkg = json.load(f)
                    return {
                        "type": "nodejs",
                        "name": pkg.get("name", ""),
                        "version": pkg.get("version", ""),
                        "scripts": list(pkg.get("scripts", {}).keys()),
                        "dependencies": list(pkg.get("dependencies", {}).keys())
                    }
            except:
                pass
        
        if 'requirements.txt' in files or 'pyproject.toml' in files or 'setup.py' in files:
            project_types.append("python")
            # Get Python project info
            deps = []
            if 'requirements.txt' in files:
                try:
                    with open('requirements.txt', 'r') as f:
                        deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                except:
                    pass
            return {
                "type": "python",
                "dependencies": deps
            }
        
        if 'pom.xml' in files:
            project_types.append("maven")
            return {"type": "maven"}
        
        if 'build.gradle' in files:
            project_types.append("gradle")
            return {"type": "gradle"}
        
        if 'go.mod' in files:
            project_types.append("go")
            return {"type": "go"}
        
        if 'Cargo.toml' in files:
            project_types.append("rust")
            return {"type": "rust"}
        
        if 'Dockerfile' in files:
            project_types.append("docker")
            return {"type": "docker"}
        
        return {
            "type": "unknown",
            "detected_types": project_types
        }
    
    def _get_dependencies(self) -> Dict[str, List[str]]:
        """Get project dependencies"""
        project_type = self.context.get("project_info", {}).get("type", "unknown")
        
        if project_type == "python":
            try:
                result = subprocess.run(
                    ["pip", "list", "--format=json"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    return {"pip_packages": json.loads(result.stdout)}
            except:
                pass
        elif project_type == "nodejs":
            try:
                with open('package.json', 'r') as f:
                    pkg = json.load(f)
                    deps = list(pkg.get("dependencies", {}).keys())
                    dev_deps = list(pkg.get("devDependencies", {}).keys())
                    return {
                        "dependencies": deps,
                        "dev_dependencies": dev_deps
                    }
            except:
                pass
        
        return {}
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('.').percent,
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "process_count": len(psutil.pids())
            }
        except ImportError:
            # Fallback if psutil is not available
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "disk_usage": 0,
                "memory_available_gb": 0,
                "process_count": 0
            }
        except Exception:
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "disk_usage": 0,
                "memory_available_gb": 0,
                "process_count": 0
            }
    
    def _get_user_info(self) -> Dict[str, str]:
        """Get user information"""
        import getpass
        import socket
        return {
            "username": getpass.getuser(),
            "hostname": socket.gethostname(),
            "home_dir": os.path.expanduser("~")
        }
    
    def _get_network_status(self) -> Dict[str, Any]:
        """Get basic network status"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            sock.close()
            return {
                "connected": True,
                "local_ip": ip
            }
        except:
            return {
                "connected": False,
                "local_ip": "unknown"
            }
    
    def get_context(self) -> Dict[str, Any]:
        """Get the current context, refreshing it first"""
        self._refresh_context()
        return self.context
    
    def _refresh_context(self):
        """Refresh the context with current system state"""
        old_context = self.context.copy()
        
        # Update current directory
        self.context["current_directory"] = os.getcwd()
        
        # Update git status
        new_git_status = self._get_git_status()
        if new_git_status != old_context.get("git_status"):
            self._trigger_event(ContextEventType.GIT_CHECKOUT, {
                "previous": old_context.get("git_status"),
                "current": new_git_status
            })
        self.context["git_status"] = new_git_status
        
        # Update system resources
        self.context["system_resources"] = self._get_system_resources()
        
        # Update timestamp
        self.context["timestamp"] = datetime.now().isoformat()
    
    def register_event_handler(self, handler: Callable[[ContextEvent], None]):
        """Register a function to handle context events"""
        self.event_handlers.append(handler)
    
    def _trigger_event(self, event_type: ContextEventType, data: Dict[str, Any], source: str = "system"):
        """Trigger a context event"""
        event = ContextEvent(
            event_type=event_type,
            data=data,
            timestamp=datetime.now(),
            source=source
        )
        
        # Add to recent events
        self.context["recent_events"].append({
            "type": event_type.value,
            "data": data,
            "timestamp": event.timestamp.isoformat(),
            "source": source
        })
        
        # Keep only recent events (last 50)
        self.context["recent_events"] = self.context["recent_events"][-50:]
        
        # Call all registered handlers
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in context event handler: {e}")
    
    def start_monitoring(self):
        """Start monitoring for context changes"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        
        # Start file system monitoring
        self._setup_file_monitoring()
        
        # Start resource monitoring in a separate thread
        self.resource_monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.resource_monitor_thread.start()
        
        print("Context monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring for context changes"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        self.observer.stop()
        self.observer.join(timeout=1)
        
        print("Context monitoring stopped")
    
    def _setup_file_monitoring(self):
        """Set up file system monitoring"""
        # Watch the current directory for file changes
        event_handler = FileChangeHandler(self)
        self.observer.schedule(event_handler, path='.', recursive=True)
        self.observer.start()
    
    def _monitor_resources(self):
        """Monitor system resources in a background thread"""
        while self.is_monitoring:
            try:
                # Check system resources periodically
                current_resources = self._get_system_resources()
                old_resources = self.context.get("system_resources", {})
                
                # Check for significant changes in resource usage
                if (abs(current_resources.get("cpu_percent", 0) - old_resources.get("cpu_percent", 0)) > 20 or
                    abs(current_resources.get("memory_percent", 0) - old_resources.get("memory_percent", 0)) > 20):
                    
                    self._trigger_event(ContextEventType.SYSTEM_RESOURCE_CHANGE, {
                        "previous": old_resources,
                        "current": current_resources
                    })
                
                # Update context with new resource info
                self.context["system_resources"] = current_resources
                
                # Sleep for 5 seconds before next check
                time.sleep(5)
            except Exception as e:
                print(f"Error in resource monitoring: {e}")
                time.sleep(5)
    
    def on_git_checkout(self, branch_name: str):
        """Call when a git checkout event occurs"""
        self._trigger_event(ContextEventType.GIT_CHECKOUT, {
            "branch": branch_name,
            "current_directory": self.context["current_directory"]
        })
    
    def on_file_modified(self, file_path: str, change_type: str = "modified"):
        """Call when a file is modified"""
        self._trigger_event(ContextEventType.FILE_MODIFIED, {
            "file_path": file_path,
            "change_type": change_type,
            "current_directory": self.context["current_directory"]
        })
    
    def on_dependency_change(self, dependency_name: str, change_type: str = "added"):
        """Call when a dependency is changed"""
        self._trigger_event(ContextEventType.DEPENDENCY_CHANGE, {
            "dependency": dependency_name,
            "change_type": change_type,
            "current_directory": self.context["current_directory"]
        })
    
    def on_test_failure(self, test_name: str, error_message: str):
        """Call when a test fails"""
        self._trigger_event(ContextEventType.TEST_FAILURE, {
            "test_name": test_name,
            "error_message": error_message,
            "current_directory": self.context["current_directory"]
        })
    
    def enrich_context_for_task(self, task_description: str) -> Dict[str, Any]:
        """Enrich context specifically for a given task"""
        enriched_context = self.context.copy()
        
        # Add task-specific information
        enriched_context["task_relevance"] = {
            "description": task_description,
            "relevant_files": self._find_relevant_files(task_description),
            "related_dependencies": self._find_related_dependencies(task_description)
        }
        
        return enriched_context
    
    def _find_relevant_files(self, task_description: str) -> List[str]:
        """Find files that might be relevant to the task"""
        relevant_files = []
        task_lower = task_description.lower()
        
        # Look for common file patterns based on task description
        if any(word in task_lower for word in ["test", "spec", "check"]):
            # Look for test files
            for root, dirs, files in os.walk("."):
                for file in files:
                    if any(test_pattern in file.lower() for test_pattern in ["test", "spec", "_test", "_spec"]):
                        relevant_files.append(os.path.join(root, file))
        
        if any(word in task_lower for word in ["config", "setting", "env"]):
            # Look for config files
            config_patterns = [".env", "config", "settings", "conf"]
            for root, dirs, files in os.walk("."):
                for file in files:
                    if any(pattern in file.lower() for pattern in config_patterns):
                        relevant_files.append(os.path.join(root, file))
        
        # Limit to first 10 relevant files
        return relevant_files[:10]


class FileChangeHandler(FileSystemEventHandler):
    """Handles file system change events"""
    
    def __init__(self, context_engine: ContextEngine):
        self.context_engine = context_engine
    
    def on_modified(self, event):
        if not event.is_directory:
            self.context_engine.on_file_modified(event.src_path, "modified")
    
    def on_created(self, event):
        if not event.is_directory:
            self.context_engine.on_file_modified(event.src_path, "created")
    
    def on_deleted(self, event):
        if not event.is_directory:
            self.context_engine.on_file_modified(event.src_path, "deleted")
    
    def on_moved(self, event):
        if not event.is_directory:
            self.context_engine.on_file_modified(event.dest_path, "moved")