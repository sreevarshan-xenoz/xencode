"""
Resource cleanup utilities for Xencode
Provides mechanisms for cleaning up resources and preventing leaks
"""
import gc
import weakref
import threading
import time
import atexit
from typing import Any, Callable, Dict, List, Optional, Set
from pathlib import Path
import tempfile
import shutil


class ResourceCleanupManager:
    """Manages cleanup of various resources to prevent memory leaks and resource exhaustion"""
    
    def __init__(self):
        self._managed_files: Set[Path] = set()
        self._managed_temp_dirs: Set[Path] = set()
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()
        
        # Register the cleanup function to run at program exit
        atexit.register(self.cleanup_all)
    
    def register_file(self, file_path: Path) -> None:
        """Register a file to be cleaned up
        
        Args:
            file_path: Path to the file to register for cleanup
        """
        with self._lock:
            self._managed_files.add(file_path)
    
    def register_temp_dir(self, dir_path: Path) -> None:
        """Register a temporary directory to be cleaned up
        
        Args:
            dir_path: Path to the temporary directory to register for cleanup
        """
        with self._lock:
            self._managed_temp_dirs.add(dir_path)
    
    def register_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Register a custom cleanup callback function
        
        Args:
            callback: Function to call during cleanup
        """
        with self._lock:
            self._cleanup_callbacks.append(callback)
    
    def unregister_file(self, file_path: Path) -> bool:
        """Unregister a file from cleanup
        
        Args:
            file_path: Path to the file to unregister
            
        Returns:
            True if the file was registered and unregistered, False otherwise
        """
        with self._lock:
            if file_path in self._managed_files:
                self._managed_files.remove(file_path)
                return True
            return False
    
    def cleanup_file(self, file_path: Path) -> bool:
        """Clean up a specific file
        
        Args:
            file_path: Path to the file to clean up
            
        Returns:
            True if the file was cleaned up, False otherwise
        """
        try:
            if file_path.exists():
                file_path.unlink()
                self.unregister_file(file_path)
                return True
            return False
        except Exception:
            return False
    
    def cleanup_temp_dir(self, dir_path: Path) -> bool:
        """Clean up a specific temporary directory
        
        Args:
            dir_path: Path to the temporary directory to clean up
            
        Returns:
            True if the directory was cleaned up, False otherwise
        """
        try:
            if dir_path.exists() and dir_path.is_dir():
                shutil.rmtree(dir_path)
                
                with self._lock:
                    self._managed_temp_dirs.discard(dir_path)
                return True
            return False
        except Exception:
            return False
    
    def cleanup_all(self) -> None:
        """Clean up all registered resources"""
        # Run custom cleanup callbacks first
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception:
                pass  # Ignore errors in cleanup callbacks
        
        # Clean up files
        files_to_remove = []
        for file_path in self._managed_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                files_to_remove.append(file_path)
            except Exception:
                pass  # Ignore errors when cleaning up files
        
        with self._lock:
            for file_path in files_to_remove:
                self._managed_files.discard(file_path)
        
        # Clean up temporary directories
        dirs_to_remove = []
        for dir_path in self._managed_temp_dirs:
            try:
                if dir_path.exists() and dir_path.is_dir():
                    shutil.rmtree(dir_path)
                dirs_to_remove.append(dir_path)
            except Exception:
                pass  # Ignore errors when cleaning up directories
        
        with self._lock:
            for dir_path in dirs_to_remove:
                self._managed_temp_dirs.discard(dir_path)
        
        # Force garbage collection
        gc.collect()
    
    def create_managed_temp_file(self, suffix: str = "", prefix: str = "tmp") -> Path:
        """Create a managed temporary file that will be cleaned up automatically
        
        Args:
            suffix: Suffix for the temporary file
            prefix: Prefix for the temporary file
            
        Returns:
            Path to the created temporary file
        """
        temp_file = Path(tempfile.mktemp(suffix=suffix, prefix=prefix))
        self.register_file(temp_file)
        return temp_file
    
    def create_managed_temp_dir(self, suffix: str = "", prefix: str = "tmp") -> Path:
        """Create a managed temporary directory that will be cleaned up automatically
        
        Args:
            suffix: Suffix for the temporary directory
            prefix: Prefix for the temporary directory
            
        Returns:
            Path to the created temporary directory
        """
        temp_dir = Path(tempfile.mkdtemp(suffix=suffix, prefix=prefix))
        self.register_temp_dir(temp_dir)
        return temp_dir


class WeakReferenceCache:
    """A cache that uses weak references to prevent memory leaks"""
    
    def __init__(self):
        self._cache: Dict[str, weakref.ref] = {}
        self._lock = threading.RLock()
    
    def put(self, key: str, value: Any) -> None:
        """Put a value in the cache with a weak reference
        
        Args:
            key: Key to store the value under
            value: Value to store (should be a deletable object)
        """
        with self._lock:
            # Use a weak reference callback to clean up the cache entry when the object is deleted
            def cleanup_callback(weak_ref):
                with self._lock:
                    # Clean up the cache entry when the referenced object is deleted
                    if key in self._cache and self._cache[key] is weak_ref:
                        del self._cache[key]
            
            self._cache[key] = weakref.ref(value, cleanup_callback)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache
        
        Args:
            key: Key to look up
            
        Returns:
            The cached value if it exists and hasn't been garbage collected, None otherwise
        """
        with self._lock:
            if key in self._cache:
                value_ref = self._cache[key]
                value = value_ref()  # Call the weak reference to get the actual object
                
                if value is not None:
                    return value
                else:
                    # Object has been garbage collected, remove the entry
                    del self._cache[key]
        
        return None
    
    def remove(self, key: str) -> bool:
        """Remove a key from the cache
        
        Args:
            key: Key to remove
            
        Returns:
            True if the key existed and was removed, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from the cache"""
        with self._lock:
            self._cache.clear()
    
    def keys(self) -> List[str]:
        """Get all keys in the cache (may include keys whose objects have been garbage collected)
        
        Returns:
            List of keys in the cache
        """
        with self._lock:
            return list(self._cache.keys())
    
    def size(self) -> int:
        """Get the size of the cache (includes entries for objects that may have been garbage collected)
        
        Returns:
            Number of entries in the cache
        """
        with self._lock:
            return len(self._cache)


class PeriodicCleanupService:
    """A service that performs periodic cleanup operations"""
    
    def __init__(self, cleanup_interval: int = 300):  # 5 minutes default
        """
        Initialize the periodic cleanup service.
        
        Args:
            cleanup_interval: Interval between cleanup operations in seconds
        """
        self.cleanup_interval = cleanup_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self) -> None:
        """Start the periodic cleanup service"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_cleanup_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the periodic cleanup service"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._running = False
    
    def _run_cleanup_loop(self) -> None:
        """Main loop for periodic cleanup"""
        while not self._stop_event.wait(self.cleanup_interval):
            try:
                # Perform cleanup operations
                self._perform_cleanup()
            except Exception:
                # Log the exception but continue running
                pass
    
    def _perform_cleanup(self) -> None:
        """Perform the actual cleanup operations"""
        # Force garbage collection
        collected = gc.collect()
        
        # Could add other cleanup operations here:
        # - Clean up old temporary files
        # - Close idle connections
        # - Flush buffers
        # etc.
        pass


# Global instances
resource_cleanup_manager = ResourceCleanupManager()
weak_cache = WeakReferenceCache()
periodic_cleanup_service = PeriodicCleanupService()


def get_resource_cleanup_manager() -> ResourceCleanupManager:
    """Get the global resource cleanup manager"""
    return resource_cleanup_manager


def get_weak_cache() -> WeakReferenceCache:
    """Get the global weak reference cache"""
    return weak_cache


def get_periodic_cleanup_service() -> PeriodicCleanupService:
    """Get the global periodic cleanup service"""
    return periodic_cleanup_service


def cleanup_resources_on_exit():
    """Explicitly call cleanup when the program exits"""
    resource_cleanup_manager.cleanup_all()
    gc.collect()


def create_temp_file(suffix: str = "", prefix: str = "tmp") -> Path:
    """Create a temporary file that will be cleaned up automatically
    
    Args:
        suffix: Suffix for the temporary file
        prefix: Prefix for the temporary file
        
    Returns:
        Path to the created temporary file
    """
    return resource_cleanup_manager.create_managed_temp_file(suffix, prefix)


def create_temp_dir(suffix: str = "", prefix: str = "tmp") -> Path:
    """Create a temporary directory that will be cleaned up automatically
    
    Args:
        suffix: Suffix for the temporary directory
        prefix: Prefix for the temporary directory
        
    Returns:
        Path to the created temporary directory
    """
    return resource_cleanup_manager.create_managed_temp_dir(suffix, prefix)


# Register the cleanup function to run at program exit
atexit.register(cleanup_resources_on_exit)