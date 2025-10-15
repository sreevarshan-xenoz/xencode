#!/usr/bin/env python3
"""
File Operations Plugin

Provides comprehensive file system operations for Xencode including:
- Directory listing and navigation
- File reading (text and binary)
- Content search and pattern matching
- Directory tree visualization
- Lint error detection
- Secure file mutations with RBAC

Key Features:
- Async I/O for performance
- Workspace-scoped operations for security
- RBAC permission checking
- Intelligent caching with TTL
- Audit logging for mutations
- Sandboxed execution
"""

import asyncio
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Generator, AsyncGenerator
import json
import logging
import hashlib

# Import async file operations
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

# Import caching system
try:
    from ..advanced_cache_system import get_cache_manager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# Import audit logging
try:
    from ..audit.audit_logger import get_global_audit_logger, AuditEventType, AuditSeverity
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PluginContext:
    """Context information for plugin execution"""
    workspace_id: str = "default"
    user_id: str = "anonymous"
    workspace_root: Optional[Path] = None
    permissions: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize workspace root if not provided"""
        if self.workspace_root is None:
            self.workspace_root = Path.cwd()
        elif isinstance(self.workspace_root, str):
            self.workspace_root = Path(self.workspace_root)


class FileOperationsPlugin:
    """
    File operations plugin with comprehensive file system capabilities.
    
    Provides secure, workspace-scoped file operations with RBAC, caching,
    and audit logging integration.
    """
    
    def __init__(self, context: PluginContext):
        self.context = context
        self.cache_manager = None
        self.audit_logger = None
        
        # Initialize cache manager if available
        if CACHE_AVAILABLE:
            try:
                asyncio.create_task(self._init_cache())
            except Exception as e:
                logger.warning(f"Failed to initialize cache: {e}")
        
        # Initialize audit logger if available
        if AUDIT_AVAILABLE:
            try:
                self.audit_logger = get_global_audit_logger()
            except Exception as e:
                logger.warning(f"Failed to initialize audit logger: {e}")
        
        # Cache for directory listings (TTL: 5 seconds)
        self._ls_cache: Dict[str, Tuple[Dict[str, List[str]], float]] = {}
        self._cache_ttl = 5.0
    
    async def _init_cache(self):
        """Initialize cache manager asynchronously"""
        try:
            self.cache_manager = await get_cache_manager()
        except Exception as e:
            logger.warning(f"Cache manager initialization failed: {e}")
    
    def _check_permission(self, permission: str) -> bool:
        """Check if user has required permission"""
        return permission in self.context.permissions
    
    def _require_permission(self, permission: str) -> None:
        """Require specific permission or raise PermissionError"""
        if not self._check_permission(permission):
            raise PermissionError(f"Permission '{permission}' required")
    
    def _sandbox_path(self, path: Path) -> Path:
        """Ensure path is within workspace boundaries"""
        if not path.is_absolute():
            path = self.context.workspace_root / path
        
        # Resolve to handle .. and . components
        resolved_path = path.resolve()
        workspace_root = self.context.workspace_root.resolve()
        
        # Check if path is within workspace
        try:
            resolved_path.relative_to(workspace_root)
        except ValueError:
            raise PermissionError(f"Path {path} is outside workspace boundaries")
        
        return resolved_path
    
    def _log_audit_event(self, event_type: str, path: Path, success: bool = True, 
                        error_message: Optional[str] = None, **details):
        """Log audit event if audit logging is available"""
        if self.audit_logger:
            try:
                self.audit_logger.log_event(
                    AuditEventType.DATA_ACCESS if success else AuditEventType.SECURITY_INCIDENT,
                    AuditSeverity.INFO if success else AuditSeverity.WARNING,
                    user_id=self.context.user_id,
                    session_id=self.context.session_id,
                    resource=str(path),
                    action=event_type,
                    success=success,
                    error_message=error_message,
                    **details
                )
            except Exception as e:
                logger.warning(f"Audit logging failed: {e}")
    
    async def ls_dir(self, path: Path) -> Dict[str, List[str]]:
        """
        List directory contents returning files and directories separately.
        
        Args:
            path: Directory path to list
            
        Returns:
            Dict with 'files' and 'dirs' keys containing lists of names
            
        Raises:
            PermissionError: If user lacks file:read permission or path is outside workspace
            FileNotFoundError: If directory doesn't exist
        """
        self._require_permission("file:read")
        
        # Sandbox the path
        safe_path = self._sandbox_path(path)
        
        # Check cache first
        cache_key = str(safe_path)
        if cache_key in self._ls_cache:
            cached_result, cached_time = self._ls_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                self._log_audit_event("ls_dir", safe_path, cache_hit=True)
                return cached_result
        
        try:
            if not safe_path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
            
            if not safe_path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {path}")
            
            # List directory contents
            files = []
            dirs = []
            
            for item in safe_path.iterdir():
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    dirs.append(item.name)
            
            result = {
                "files": sorted(files),
                "dirs": sorted(dirs)
            }
            
            # Cache the result
            self._ls_cache[cache_key] = (result, time.time())
            
            # Clean old cache entries
            self._cleanup_cache()
            
            self._log_audit_event("ls_dir", safe_path, file_count=len(files), dir_count=len(dirs))
            return result
            
        except Exception as e:
            self._log_audit_event("ls_dir", safe_path, success=False, error_message=str(e))
            raise
    
    async def read_file(self, path: Path, binary: bool = False) -> Union[str, bytes]:
        """
        Read file content as text or binary.
        
        Args:
            path: File path to read
            binary: If True, return bytes; if False, return string
            
        Returns:
            File content as string or bytes
            
        Raises:
            PermissionError: If user lacks file:read permission or path is outside workspace
            FileNotFoundError: If file doesn't exist
        """
        self._require_permission("file:read")
        
        # Sandbox the path
        safe_path = self._sandbox_path(path)
        
        try:
            if not safe_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            if not safe_path.is_file():
                raise IsADirectoryError(f"Path is not a file: {path}")
            
            # Read file content
            if AIOFILES_AVAILABLE:
                mode = 'rb' if binary else 'r'
                encoding = None if binary else 'utf-8'
                
                async with aiofiles.open(safe_path, mode=mode, encoding=encoding) as f:
                    content = await f.read()
            else:
                # Fallback to synchronous I/O
                if binary:
                    with open(safe_path, 'rb') as f:
                        content = f.read()
                else:
                    with open(safe_path, 'r', encoding='utf-8') as f:
                        content = f.read()
            
            self._log_audit_event(
                "read_file", 
                safe_path, 
                binary_mode=binary,
                content_size=len(content)
            )
            
            return content
            
        except Exception as e:
            self._log_audit_event("read_file", safe_path, success=False, error_message=str(e))
            raise
    
    async def search_pathnames_only(self, pattern: str, path: Optional[Path] = None) -> List[Path]:
        """
        Search for files matching a pathname pattern.
        
        Args:
            pattern: Glob pattern to match (e.g., "*.py", "test_*.txt")
            path: Directory to search in (defaults to workspace root)
            
        Returns:
            List of matching file paths
        """
        self._require_permission("file:search")
        
        if path is None:
            path = self.context.workspace_root
        
        safe_path = self._sandbox_path(path)
        
        try:
            # Use pathlib glob for pattern matching
            matches = list(safe_path.glob(pattern))
            
            # Filter to only return files within workspace
            workspace_root = self.context.workspace_root.resolve()
            filtered_matches = []
            
            for match in matches:
                try:
                    match.resolve().relative_to(workspace_root)
                    filtered_matches.append(match)
                except ValueError:
                    # Skip files outside workspace
                    continue
            
            self._log_audit_event(
                "search_pathnames", 
                safe_path, 
                pattern=pattern,
                matches_found=len(filtered_matches)
            )
            
            return filtered_matches
            
        except Exception as e:
            self._log_audit_event("search_pathnames", safe_path, success=False, error_message=str(e))
            raise
    
    async def search_for_files(self, query: str, recursive: bool = True, 
                             path: Optional[Path] = None) -> AsyncGenerator[Tuple[Path, str], None]:
        """
        Search for files containing specific content.
        
        Args:
            query: Text to search for in file contents
            recursive: Whether to search subdirectories
            path: Directory to search in (defaults to workspace root)
            
        Yields:
            Tuples of (file_path, matching_line)
        """
        self._require_permission("file:search")
        
        if path is None:
            path = self.context.workspace_root
        
        safe_path = self._sandbox_path(path)
        
        try:
            matches_found = 0
            
            if recursive:
                # Use os.walk for recursive search
                for root, dirs, files in os.walk(safe_path):
                    root_path = Path(root)
                    
                    for file_name in files:
                        file_path = root_path / file_name
                        
                        # Skip binary files and large files
                        try:
                            if file_path.stat().st_size > 10 * 1024 * 1024:  # Skip files > 10MB
                                continue
                            
                            # Try to read as text
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                for line_num, line in enumerate(f, 1):
                                    if query in line:
                                        matches_found += 1
                                        yield (file_path, line.strip())
                                        
                        except (UnicodeDecodeError, PermissionError, OSError):
                            # Skip files that can't be read as text
                            continue
            else:
                # Search only in the specified directory
                for file_path in safe_path.iterdir():
                    if file_path.is_file():
                        try:
                            if file_path.stat().st_size > 10 * 1024 * 1024:  # Skip files > 10MB
                                continue
                            
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                for line_num, line in enumerate(f, 1):
                                    if query in line:
                                        matches_found += 1
                                        yield (file_path, line.strip())
                                        
                        except (UnicodeDecodeError, PermissionError, OSError):
                            continue
            
            self._log_audit_event(
                "search_content", 
                safe_path, 
                query=query,
                recursive=recursive,
                matches_found=matches_found
            )
            
        except Exception as e:
            self._log_audit_event("search_content", safe_path, success=False, error_message=str(e))
            raise
    
    async def search_in_file(self, file_path: Path, query: str) -> List[Tuple[int, str]]:
        """
        Search for content within a specific file.
        
        Args:
            file_path: Path to file to search in
            query: Text to search for
            
        Returns:
            List of tuples (line_number, line_content) for matching lines
        """
        self._require_permission("file:search")
        
        safe_path = self._sandbox_path(file_path)
        
        try:
            if not safe_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not safe_path.is_file():
                raise IsADirectoryError(f"Path is not a file: {file_path}")
            
            matches = []
            
            # Read file and search for query
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(safe_path, 'r', encoding='utf-8', errors='ignore') as f:
                    async for line_num, line in enumerate(f, 1):
                        if query in line:
                            matches.append((line_num, line.strip()))
            else:
                with open(safe_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if query in line:
                            matches.append((line_num, line.strip()))
            
            self._log_audit_event(
                "search_in_file", 
                safe_path, 
                query=query,
                matches_found=len(matches)
            )
            
            return matches
            
        except Exception as e:
            self._log_audit_event("search_in_file", safe_path, success=False, error_message=str(e))
            raise
    
    async def get_dir_tree(self, path: Path, depth: int = 1) -> str:
        """
        Generate ASCII directory tree representation.
        
        Args:
            path: Directory path to generate tree for
            depth: Maximum depth to traverse
            
        Returns:
            ASCII tree representation as string
        """
        self._require_permission("file:view")
        
        safe_path = self._sandbox_path(path)
        
        try:
            if not safe_path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
            
            if not safe_path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {path}")
            
            def _build_tree(current_path: Path, current_depth: int, prefix: str = "") -> List[str]:
                """Recursively build tree representation"""
                if current_depth <= 0:
                    return []
                
                lines = []
                try:
                    items = sorted(current_path.iterdir(), key=lambda x: (x.is_file(), x.name))
                    
                    for i, item in enumerate(items):
                        is_last = i == len(items) - 1
                        
                        # Choose appropriate tree characters
                        if is_last:
                            current_prefix = "└── "
                            next_prefix = prefix + "    "
                        else:
                            current_prefix = "├── "
                            next_prefix = prefix + "│   "
                        
                        # Add current item
                        item_name = item.name
                        if item.is_dir():
                            item_name += "/"
                        
                        lines.append(f"{prefix}{current_prefix}{item_name}")
                        
                        # Recurse into directories
                        if item.is_dir() and current_depth > 1:
                            lines.extend(_build_tree(item, current_depth - 1, next_prefix))
                
                except PermissionError:
                    lines.append(f"{prefix}└── [Permission Denied]")
                
                return lines
            
            # Start with root directory name
            tree_lines = [safe_path.name + "/"]
            tree_lines.extend(_build_tree(safe_path, depth))
            
            result = "\n".join(tree_lines)
            
            self._log_audit_event(
                "get_dir_tree", 
                safe_path, 
                depth=depth,
                lines_generated=len(tree_lines)
            )
            
            return result
            
        except Exception as e:
            self._log_audit_event("get_dir_tree", safe_path, success=False, error_message=str(e))
            raise
    
    async def read_lint_errors(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Read lint errors from a Python file using ruff or flake8.
        
        Args:
            file_path: Path to Python file to lint
            
        Returns:
            List of lint error dictionaries with line, message, severity
        """
        self._require_permission("file:lint")
        
        safe_path = self._sandbox_path(file_path)
        
        try:
            if not safe_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not safe_path.is_file():
                raise IsADirectoryError(f"Path is not a file: {file_path}")
            
            errors = []
            
            # Try ruff first (faster and more modern)
            try:
                result = subprocess.run(
                    ["ruff", "check", "--output-format=json", str(safe_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.stdout:
                    ruff_errors = json.loads(result.stdout)
                    for error in ruff_errors:
                        errors.append({
                            "line": error.get("location", {}).get("row", 0),
                            "column": error.get("location", {}).get("column", 0),
                            "msg": error.get("message", ""),
                            "code": error.get("code", ""),
                            "severity": "error" if error.get("code", "").startswith("E") else "warning"
                        })
                
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
                # Fallback to flake8
                try:
                    result = subprocess.run(
                        ["flake8", "--format=json", str(safe_path)],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    # flake8 doesn't have native JSON output, parse line format
                    for line in result.stdout.strip().split('\n'):
                        if line and ':' in line:
                            parts = line.split(':', 4)
                            if len(parts) >= 4:
                                errors.append({
                                    "line": int(parts[1]) if parts[1].isdigit() else 0,
                                    "column": int(parts[2]) if parts[2].isdigit() else 0,
                                    "msg": parts[3].strip() if len(parts) > 3 else "",
                                    "code": parts[3].split()[0] if len(parts) > 3 and parts[3].split() else "",
                                    "severity": "error"
                                })
                
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                    # No linter available
                    logger.warning("No Python linter (ruff or flake8) available")
            
            self._log_audit_event(
                "read_lint_errors", 
                safe_path, 
                errors_found=len(errors)
            )
            
            return errors
            
        except Exception as e:
            self._log_audit_event("read_lint_errors", safe_path, success=False, error_message=str(e))
            raise
    
    async def create_file_or_folder(self, path: Path, content: Optional[str] = None, 
                                  is_directory: bool = False) -> bool:
        """
        Create a new file or directory.
        
        Args:
            path: Path to create
            content: Content for file (ignored for directories)
            is_directory: If True, create directory; if False, create file
            
        Returns:
            True if creation was successful
        """
        self._require_permission("file:write")
        
        safe_path = self._sandbox_path(path)
        
        try:
            if safe_path.exists():
                raise FileExistsError(f"Path already exists: {path}")
            
            if is_directory:
                # Create directory
                safe_path.mkdir(parents=True, exist_ok=False)
                self._log_audit_event("create_directory", safe_path)
            else:
                # Create file
                safe_path.parent.mkdir(parents=True, exist_ok=True)
                
                if AIOFILES_AVAILABLE and content is not None:
                    async with aiofiles.open(safe_path, 'w', encoding='utf-8') as f:
                        await f.write(content)
                else:
                    with open(safe_path, 'w', encoding='utf-8') as f:
                        f.write(content or "")
                
                self._log_audit_event(
                    "create_file", 
                    safe_path, 
                    content_size=len(content) if content else 0
                )
            
            # Invalidate directory listing cache
            self._invalidate_ls_cache(safe_path.parent)
            
            return True
            
        except Exception as e:
            self._log_audit_event(
                "create_file_or_folder", 
                safe_path, 
                success=False, 
                error_message=str(e),
                is_directory=is_directory
            )
            raise
    
    async def delete_file_or_folder(self, path: Path, recursive: bool = False) -> bool:
        """
        Delete a file or directory.
        
        Args:
            path: Path to delete
            recursive: If True, delete directories recursively
            
        Returns:
            True if deletion was successful
        """
        self._require_permission("file:delete")
        
        safe_path = self._sandbox_path(path)
        
        try:
            if not safe_path.exists():
                raise FileNotFoundError(f"Path not found: {path}")
            
            if safe_path.is_dir():
                if recursive:
                    shutil.rmtree(safe_path)
                else:
                    safe_path.rmdir()  # Only works if directory is empty
                
                self._log_audit_event("delete_directory", safe_path, recursive=recursive)
            else:
                safe_path.unlink()
                self._log_audit_event("delete_file", safe_path)
            
            # Invalidate directory listing cache
            self._invalidate_ls_cache(safe_path.parent)
            
            return True
            
        except Exception as e:
            self._log_audit_event(
                "delete_file_or_folder", 
                safe_path, 
                success=False, 
                error_message=str(e),
                recursive=recursive
            )
            raise
    
    async def edit_file(self, file_path: Path, search: str, replace: str, 
                       backup: bool = True) -> bool:
        """
        Edit file content using search and replace.
        
        Args:
            file_path: Path to file to edit
            search: Text to search for
            replace: Text to replace with
            backup: If True, create backup before editing
            
        Returns:
            True if edit was successful
        """
        self._require_permission("file:write")
        
        safe_path = self._sandbox_path(file_path)
        
        try:
            if not safe_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not safe_path.is_file():
                raise IsADirectoryError(f"Path is not a file: {file_path}")
            
            # Read current content
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(safe_path, 'r', encoding='utf-8') as f:
                    original_content = await f.read()
            else:
                with open(safe_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
            
            # Create backup if requested
            if backup:
                backup_path = safe_path.with_suffix(safe_path.suffix + '.bak')
                if AIOFILES_AVAILABLE:
                    async with aiofiles.open(backup_path, 'w', encoding='utf-8') as f:
                        await f.write(original_content)
                else:
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
            
            # Perform replacement
            new_content = original_content.replace(search, replace)
            
            # Write updated content
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(safe_path, 'w', encoding='utf-8') as f:
                    await f.write(new_content)
            else:
                with open(safe_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            
            replacements_made = original_content.count(search)
            
            self._log_audit_event(
                "edit_file", 
                safe_path, 
                search_text=search,
                replace_text=replace,
                replacements_made=replacements_made,
                backup_created=backup
            )
            
            return True
            
        except Exception as e:
            self._log_audit_event(
                "edit_file", 
                safe_path, 
                success=False, 
                error_message=str(e)
            )
            raise
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, cached_time) in self._ls_cache.items()
            if current_time - cached_time > self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._ls_cache[key]
    
    def _invalidate_ls_cache(self, directory: Path):
        """Invalidate cache entries for a directory"""
        dir_key = str(directory)
        if dir_key in self._ls_cache:
            del self._ls_cache[dir_key]