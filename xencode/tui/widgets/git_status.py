#!/usr/bin/env python3
"""
Git Status Manager for Xencode TUI

Provides real-time Git status tracking and operations.
"""

import asyncio
from pathlib import Path
from typing import Dict, Optional
from enum import Enum
import time

try:
    from git import Repo, InvalidGitRepositoryError, GitCommandError
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


class GitStatus(Enum):
    """Git file status types"""
    UNTRACKED = "?"
    MODIFIED = "M"
    ADDED = "A"
    DELETED = "D"
    RENAMED = "R"
    COPIED = "C"
    UNMERGED = "U"
    CLEAN = " "


class GitStatusManager:
    """Manages Git status for a repository"""
    
    def __init__(self, repo_path: Path, cache_ttl: float = 2.0):
        """Initialize Git status manager
        
        Args:
            repo_path: Path to the repository root
            cache_ttl: Time-to-live for cached status (seconds)
        """
        self.repo_path = repo_path
        self.cache_ttl = cache_ttl
        self._repo: Optional[Repo] = None
        self._status_cache: Dict[str, GitStatus] = {}
        self._last_update: float = 0
        self._is_git_repo = False
        
        self._initialize_repo()
    
    def _initialize_repo(self):
        """Initialize Git repository connection"""
        if not GIT_AVAILABLE:
            return
        
        try:
            self._repo = Repo(self.repo_path, search_parent_directories=True)
            self._is_git_repo = True
        except (InvalidGitRepositoryError, GitCommandError):
            self._is_git_repo = False
    
    @property
    def is_git_repo(self) -> bool:
        """Check if this is a Git repository"""
        return self._is_git_repo
    
    def _should_refresh(self) -> bool:
        """Check if status cache should be refreshed"""
        return time.time() - self._last_update > self.cache_ttl
    
    async def refresh_status(self) -> None:
        """Refresh Git status cache (async)"""
        if not self._is_git_repo or not self._repo:
            return
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._refresh_status_sync)
        except Exception:
            # Silently ignore Git errors (repo might be in invalid state)
            pass
    
    def _refresh_status_sync(self) -> None:
        """Refresh Git status (synchronous)"""
        if not self._repo:
            return
        
        new_cache: Dict[str, GitStatus] = {}
        
        try:
            # Get untracked files
            for item in self._repo.untracked_files:
                new_cache[item] = GitStatus.UNTRACKED
            
            # Get staged files (index vs HEAD)
            try:
                staged = self._repo.index.diff('HEAD')
                for item in staged:
                    if item.new_file:
                        # New file that's been staged
                        new_cache[item.b_path] = GitStatus.ADDED
                    elif item.deleted_file:
                        # File deleted and staged
                        new_cache[item.a_path if item.a_path else item.b_path] = GitStatus.DELETED
                    elif item.renamed_file:
                        # File renamed
                        new_cache[item.b_path] = GitStatus.RENAMED
                    else:
                        # Modified and staged
                        path = item.a_path if item.a_path else item.b_path
                        new_cache[path] = GitStatus.MODIFIED
            except GitCommandError:
                # No HEAD yet (empty repo) or other Git error
                pass
            
            # Get changed files in working tree (not staged)
            changed = self._repo.index.diff(None)
            for item in changed:
                path = item.a_path if item.a_path else item.b_path
                # Only mark as modified in working tree if not already staged
                if path not in new_cache:
                    if item.deleted_file:
                        new_cache[path] = GitStatus.DELETED
                    else:
                        new_cache[path] = GitStatus.MODIFIED
            
            self._status_cache = new_cache
            self._last_update = time.time()
            
        except GitCommandError:
            # Repo might be in an invalid state
            pass
    
    def get_status(self, file_path: Path) -> GitStatus:
        """Get Git status for a file
        
        Args:
            file_path: Absolute path to file
            
        Returns:
            GitStatus enum value
        """
        if not self._is_git_repo:
            return GitStatus.CLEAN
        
        # Refresh cache if needed
        if self._should_refresh():
            # Trigger async refresh but return cached value for now
            asyncio.create_task(self.refresh_status())
        
        try:
            # Make path relative to repo root
            if self._repo:
                repo_root = Path(self._repo.working_dir)
                rel_path = str(file_path.relative_to(repo_root))
                return self._status_cache.get(rel_path, GitStatus.CLEAN)
        except ValueError:
            # Path not relative to repo
            pass
        
        return GitStatus.CLEAN
    
    def get_status_color(self, status: GitStatus) -> str:
        """Get color name for a Git status
        
        Args:
            status: GitStatus enum
            
        Returns:
            Color name for Textual styling
        """
        color_map = {
            GitStatus.UNTRACKED: "dim",
            GitStatus.MODIFIED: "yellow",
            GitStatus.ADDED: "green",
            GitStatus.DELETED: "red",
            GitStatus.RENAMED: "cyan",
            GitStatus.COPIED: "cyan",
            GitStatus.UNMERGED: "red bold",
            GitStatus.CLEAN: "",
        }
        return color_map.get(status, "")
    
    def get_status_icon(self, status: GitStatus) -> str:
        """Get icon/symbol for a Git status
        
        Args:
            status: GitStatus enum
            
        Returns:
            Status icon/symbol
        """
        icon_map = {
            GitStatus.UNTRACKED: "?",
            GitStatus.MODIFIED: "M",
            GitStatus.ADDED: "A",
            GitStatus.DELETED: "D",
            GitStatus.RENAMED: "R",
            GitStatus.COPIED: "C",
            GitStatus.UNMERGED: "U",
            GitStatus.CLEAN: " ",
        }
        return icon_map.get(status, " ")
    
    async def stage_file(self, file_path: Path) -> bool:
        """Stage a file
        
        Args:
            file_path: Path to file to stage
            
        Returns:
            True if successful
        """
        if not self._is_git_repo or not self._repo:
            return False
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._stage_file_sync, file_path)
            await self.refresh_status()
            return True
        except Exception:
            return False
    
    def _stage_file_sync(self, file_path: Path) -> None:
        """Stage a file (synchronous)"""
        if not self._repo:
            return
        
        try:
            repo_root = Path(self._repo.working_dir)
            rel_path = str(file_path.relative_to(repo_root))
            self._repo.index.add([rel_path])
        except (ValueError, GitCommandError):
            pass
    
    async def unstage_file(self, file_path: Path) -> bool:
        """Unstage a file
        
        Args:
            file_path: Path to file to unstage
            
        Returns:
            True if successful
        """
        if not self._is_git_repo or not self._repo:
            return False
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._unstage_file_sync, file_path)
            await self.refresh_status()
            return True
        except Exception:
            return False
    
    def _unstage_file_sync(self, file_path: Path) -> None:
        """Unstage a file (synchronous)"""
        if not self._repo:
            return
        
        try:
            repo_root = Path(self._repo.working_dir)
            rel_path = str(file_path.relative_to(repo_root))
            self._repo.index.reset([rel_path])
        except (ValueError, GitCommandError):
            pass
