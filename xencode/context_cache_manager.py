#!/usr/bin/env python3

import hashlib
import json
import os
import shutil
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

# Cross-platform file locking
if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl


class CacheVersion(Enum):
    """Cache version enumeration for migration support"""

    V1 = 1
    V2 = 2
    CURRENT = V2


@dataclass
class CacheLock:
    """Cache lock information with PID tracking"""

    project_hash: str
    pid: int
    timestamp: datetime
    lock_file_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'project_hash': self.project_hash,
            'pid': self.pid,
            'timestamp': self.timestamp.isoformat(),
            'lock_file_path': self.lock_file_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheLock':
        return cls(
            project_hash=data['project_hash'],
            pid=data['pid'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            lock_file_path=data['lock_file_path'],
        )


@dataclass
class ContextData:
    """Context data structure with versioning support"""

    version: int = CacheVersion.CURRENT.value
    project_hash: str = ""
    project_root: str = ""
    timestamp: datetime = None

    # File analysis
    files: List[Dict[str, Any]] = None
    semantic_index: Dict[str, Any] = None
    file_type_breakdown: Dict[str, int] = None

    # Security info
    excluded_files: List[str] = None
    security_alerts: List[Dict[str, Any]] = None

    # Performance metrics
    scan_duration_ms: int = 0
    total_size_mb: float = 0.0
    estimated_tokens: int = 0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.files is None:
            self.files = []
        if self.semantic_index is None:
            self.semantic_index = {}
        if self.file_type_breakdown is None:
            self.file_type_breakdown = {}
        if self.excluded_files is None:
            self.excluded_files = []
        if self.security_alerts is None:
            self.security_alerts = []

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextData':
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ContextCacheManager:
    """
    Atomic context cache manager with concurrent access protection.

    Implements double-lock pattern with PID tracking to prevent cache corruption
    from multiple xencode instances accessing the same project simultaneously.
    """

    def __init__(self, cache_base_dir: Optional[str] = None):
        """Initialize cache manager with configurable base directory"""
        if cache_base_dir is None:
            cache_base_dir = str(Path.home() / ".xencode" / "context")

        self.cache_base_dir = Path(cache_base_dir)
        self.cache_base_dir.mkdir(parents=True, exist_ok=True)

        # Lock management
        self._active_locks: Dict[str, CacheLock] = {}
        self._lock_fds: Dict[str, int] = {}  # Store file descriptors for active locks
        self._lock_timeout_seconds = 30
        self._cleanup_interval_hours = 24

        # Migration support
        self._migration_handlers = {
            CacheVersion.V1: self._migrate_v1_to_v2,
        }

        # Ensure cleanup on exit
        signal.signal(signal.SIGTERM, self._cleanup_on_exit)
        signal.signal(signal.SIGINT, self._cleanup_on_exit)

    def _cleanup_on_exit(self, signum, frame):
        """Clean up locks on process exit"""
        self.cleanup_stale_locks()

    def _get_project_cache_dir(self, project_hash: str) -> Path:
        """Get cache directory for specific project"""
        cache_dir = self.cache_base_dir / project_hash
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_lock_file_path(self, project_hash: str) -> Path:
        """Get lock file path for project"""
        return self._get_project_cache_dir(project_hash) / ".lock"

    def _get_cache_file_path(self, project_hash: str) -> Path:
        """Get cache file path for project"""
        return self._get_project_cache_dir(project_hash) / "cache.json"

    def _get_backup_file_path(self, project_hash: str) -> Path:
        """Get backup file path for project"""
        return self._get_project_cache_dir(project_hash) / "cache.json.backup"

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 checksum for data integrity validation"""
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _is_pid_running(self, pid: int) -> bool:
        """Check if a process ID is currently running"""
        try:
            return psutil.pid_exists(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def _acquire_file_lock(self, lock_file: Path, timeout: int = 5) -> Optional[int]:
        """Acquire exclusive file lock with timeout (cross-platform)"""
        fd = None
        try:
            # Create lock file if it doesn't exist
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            
            if sys.platform == 'win32':
                # Windows file locking using msvcrt
                # Open file in binary write mode
                fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT | os.O_BINARY)
                
                # Write at least one byte for msvcrt.locking to work
                # Check file size using fstat on the file descriptor
                file_stat = os.fstat(fd)
                if file_stat.st_size == 0:
                    os.write(fd, b'\x00')
                os.lseek(fd, 0, os.SEEK_SET)
                
                # Try to acquire exclusive lock with timeout
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                        return fd
                    except (IOError, OSError):
                        time.sleep(0.1)
            else:
                # Unix file locking using fcntl
                fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT)
                
                # Try to acquire exclusive lock with timeout
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return fd
                    except (IOError, OSError):
                        time.sleep(0.1)

            # Timeout reached
            if fd is not None:
                os.close(fd)
            return None

        except Exception as e:
            if fd is not None:
                try:
                    os.close(fd)
                except:
                    pass
            return None

    def _release_file_lock(self, fd: int):
        """Release file lock (cross-platform)"""
        try:
            if sys.platform == 'win32':
                # Windows file unlocking
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except:
                    pass
            else:
                # Unix file unlocking
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except Exception:
            pass

    def acquire_cache_lock(self, project_hash: str) -> bool:
        """
        Acquire cache lock using double-lock pattern with PID tracking.

        Returns True if lock acquired successfully, False otherwise.
        """
        lock_file = self._get_lock_file_path(project_hash)
        current_pid = os.getpid()

        try:
            # Step 1: Acquire file-level lock
            fd = self._acquire_file_lock(lock_file, timeout=5)
            if fd is None:
                return False

            # Step 2: Check existing lock content
            if lock_file.exists() and lock_file.stat().st_size > 1:  # More than just the lock byte
                try:
                    with open(lock_file, 'r') as f:
                        existing_lock_data = json.load(f)

                    existing_lock = CacheLock.from_dict(existing_lock_data)

                    # Check if existing lock is from a running process
                    if self._is_pid_running(existing_lock.pid):
                        # Check lock age to prevent indefinite locks
                        lock_age = datetime.now() - existing_lock.timestamp
                        if lock_age.total_seconds() < self._lock_timeout_seconds:
                            self._release_file_lock(fd)
                            return False  # Valid lock exists

                    # Stale lock - can be overridden
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Corrupted lock file - can be overridden
                    pass

            # Step 3: Write new lock
            new_lock = CacheLock(
                project_hash=project_hash,
                pid=current_pid,
                timestamp=datetime.now(),
                lock_file_path=str(lock_file),
            )

            # Write lock data through the file descriptor
            lock_data = json.dumps(new_lock.to_dict(), indent=2).encode('utf-8')
            # Truncate file first
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, lock_data)
            # Flush to disk
            os.fsync(fd)

            # Step 4: Store lock in memory with file descriptor
            self._active_locks[project_hash] = new_lock
            # Store the file descriptor to keep the lock held
            self._lock_fds[project_hash] = fd

            return True

        except Exception as e:
            if 'fd' in locals() and fd is not None:
                self._release_file_lock(fd)
            return False

    def release_cache_lock(self, project_hash: str) -> None:
        """Release cache lock for project"""
        try:
            # Release file descriptor if it exists
            if project_hash in self._lock_fds:
                fd = self._lock_fds[project_hash]
                self._release_file_lock(fd)
                del self._lock_fds[project_hash]

            lock_file = self._get_lock_file_path(project_hash)

            # Remove from active locks
            if project_hash in self._active_locks:
                del self._active_locks[project_hash]

            # Remove lock file
            if lock_file.exists():
                lock_file.unlink()

        except Exception:
            # Best effort cleanup
            pass

    def cleanup_stale_locks(self) -> int:
        """Clean up stale locks from dead processes"""
        cleaned_count = 0

        try:
            # Scan all project directories for lock files
            for project_dir in self.cache_base_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                lock_file = project_dir / ".lock"
                if not lock_file.exists():
                    continue

                try:
                    with open(lock_file, 'r') as f:
                        lock_data = json.load(f)

                    lock = CacheLock.from_dict(lock_data)

                    # Check if process is still running
                    if not self._is_pid_running(lock.pid):
                        lock_file.unlink()
                        cleaned_count += 1
                    else:
                        # Check lock age
                        lock_age = datetime.now() - lock.timestamp
                        if lock_age.total_seconds() > self._lock_timeout_seconds:
                            lock_file.unlink()
                            cleaned_count += 1

                except (json.JSONDecodeError, KeyError, ValueError, OSError):
                    # Corrupted or inaccessible lock file
                    try:
                        lock_file.unlink()
                        cleaned_count += 1
                    except OSError:
                        pass

        except Exception:
            pass

        return cleaned_count

    def _migrate_v1_to_v2(self, v1_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate cache data from version 1 to version 2"""
        # V1 to V2 migration: add new fields with defaults
        v2_data = v1_data.copy()
        v2_data['version'] = CacheVersion.V2.value

        # Add new V2 fields if missing
        if 'security_alerts' not in v2_data:
            v2_data['security_alerts'] = []

        if 'estimated_tokens' not in v2_data:
            v2_data['estimated_tokens'] = 0

        if 'semantic_index' not in v2_data:
            v2_data['semantic_index'] = {}

        # Ensure timestamp is properly formatted
        if 'timestamp' in v2_data and not isinstance(v2_data['timestamp'], str):
            v2_data['timestamp'] = datetime.now().isoformat()

        return v2_data

    def migrate_cache_version(self, cache_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate cache data to current version"""
        current_version = cache_data.get('version', CacheVersion.V1.value)

        if current_version == CacheVersion.CURRENT.value:
            return cache_data

        # Apply migrations sequentially
        migrated_data = cache_data.copy()

        for version in CacheVersion:
            if version.value >= current_version and version != CacheVersion.CURRENT:
                if version in self._migration_handlers:
                    migrated_data = self._migration_handlers[version](migrated_data)

        return migrated_data

    def validate_cache_integrity(self, cache_file: Path) -> bool:
        """Validate cache file integrity using checksum"""
        try:
            if not cache_file.exists():
                return False

            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Check if checksum exists
            if 'checksum' not in data:
                return False

            stored_checksum = data.pop('checksum')
            calculated_checksum = self._calculate_checksum(data)

            return stored_checksum == calculated_checksum

        except (json.JSONDecodeError, KeyError, OSError):
            return False

    def _create_backup(self, project_hash: str) -> bool:
        """Create backup of existing cache file"""
        try:
            cache_file = self._get_cache_file_path(project_hash)
            backup_file = self._get_backup_file_path(project_hash)

            if cache_file.exists():
                shutil.copy2(cache_file, backup_file)
                return True

            return False

        except Exception:
            return False

    def _restore_from_backup(self, project_hash: str) -> bool:
        """Restore cache from backup if available"""
        try:
            backup_file = self._get_backup_file_path(project_hash)
            cache_file = self._get_cache_file_path(project_hash)

            if backup_file.exists() and self.validate_cache_integrity(backup_file):
                shutil.copy2(backup_file, cache_file)
                return True

            return False

        except Exception:
            return False

    def save_context(self, project_hash: str, context_data: ContextData) -> bool:
        """
        Save context data with atomic write operations and versioning.

        Returns True if save successful, False otherwise.
        """
        if not self.acquire_cache_lock(project_hash):
            return False

        try:
            # Create backup of existing cache (only if it exists)
            cache_file = self._get_cache_file_path(project_hash)
            if cache_file.exists():
                self._create_backup(project_hash)

            # Prepare data with checksum
            data_dict = context_data.to_dict()
            data_dict['version'] = CacheVersion.CURRENT.value

            # Calculate checksum
            checksum = self._calculate_checksum(data_dict)
            data_with_checksum = data_dict.copy()
            data_with_checksum['checksum'] = checksum

            # Atomic write using temporary file
            temp_file = cache_file.with_suffix('.tmp')

            try:
                # Write to temporary file
                with open(temp_file, 'w') as f:
                    json.dump(data_with_checksum, f, indent=2, ensure_ascii=False)

                # Atomic replace
                temp_file.replace(cache_file)

                return True

            except Exception:
                # Clean up temporary file on error
                if temp_file.exists():
                    temp_file.unlink()
                raise

        except Exception:
            return False

        finally:
            self.release_cache_lock(project_hash)

    def load_context(self, project_hash: str) -> Optional[ContextData]:
        """
        Load context data with integrity validation and migration support.

        Returns ContextData if successful, None otherwise.
        """
        try:
            cache_file = self._get_cache_file_path(project_hash)

            if not cache_file.exists():
                return None

            # Validate integrity
            if not self.validate_cache_integrity(cache_file):
                # Try to restore from backup
                if self._restore_from_backup(project_hash):
                    # Retry with restored file
                    if not self.validate_cache_integrity(cache_file):
                        return None
                else:
                    return None

            # Load and parse data
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Remove checksum for processing
            data.pop('checksum', None)

            # Migrate if necessary
            migrated_data = self.migrate_cache_version(data)

            # Create ContextData object
            context_data = ContextData.from_dict(migrated_data)

            # Check if cache is too old (24 hours)
            cache_age = datetime.now() - context_data.timestamp
            if cache_age > timedelta(hours=self._cleanup_interval_hours):
                # Cache is stale, return None to trigger refresh
                return None

            return context_data

        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return None

    def cleanup_old_caches(self, max_age_hours: int = 24) -> int:
        """Clean up old cache files and return count of cleaned files"""
        cleaned_count = 0
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        try:
            for project_dir in self.cache_base_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                cache_file = project_dir / "cache.json"
                backup_file = project_dir / "cache.json.backup"

                # Check cache file age
                if cache_file.exists():
                    try:
                        file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            cache_file.unlink()
                            cleaned_count += 1
                    except OSError:
                        pass

                # Check backup file age
                if backup_file.exists():
                    try:
                        file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            backup_file.unlink()
                            cleaned_count += 1
                    except OSError:
                        pass

                # Remove empty directories
                try:
                    if not any(project_dir.iterdir()):
                        project_dir.rmdir()
                except OSError:
                    pass

        except Exception:
            pass

        return cleaned_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring and debugging"""
        stats = {
            'total_projects': 0,
            'total_cache_size_mb': 0.0,
            'active_locks': len(self._active_locks),
            'oldest_cache': None,
            'newest_cache': None,
            'cache_base_dir': str(self.cache_base_dir),
        }

        try:
            oldest_time = None
            newest_time = None
            total_size = 0

            for project_dir in self.cache_base_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                cache_file = project_dir / "cache.json"
                if cache_file.exists():
                    stats['total_projects'] += 1

                    # Calculate size
                    try:
                        file_size = cache_file.stat().st_size
                        total_size += file_size
                    except OSError:
                        pass

                    # Track oldest/newest
                    try:
                        file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                        if oldest_time is None or file_mtime < oldest_time:
                            oldest_time = file_mtime
                        if newest_time is None or file_mtime > newest_time:
                            newest_time = file_mtime
                    except OSError:
                        pass

            stats['total_cache_size_mb'] = total_size / (1024 * 1024)
            if oldest_time:
                stats['oldest_cache'] = oldest_time.isoformat()
            if newest_time:
                stats['newest_cache'] = newest_time.isoformat()

        except Exception:
            pass

        return stats


# Global instance for easy access
_cache_manager_instance = None


def get_cache_manager() -> ContextCacheManager:
    """Get global cache manager instance"""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = ContextCacheManager()
    return _cache_manager_instance
