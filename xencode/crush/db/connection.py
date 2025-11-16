"""Database connection management for crush."""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator
import logging

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages SQLite database connections with WAL mode and transactions."""
    
    def __init__(self, db_path: str, timeout: float = 30.0):
        """Initialize database connection manager.
        
        Args:
            db_path: Path to SQLite database file
            timeout: Connection timeout in seconds
        """
        self.db_path = db_path
        self.timeout = timeout
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db_directory()
    
    def _ensure_db_directory(self):
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def connect(self) -> sqlite3.Connection:
        """Get or create database connection.
        
        Returns:
            SQLite connection with WAL mode enabled
        """
        if self._conn is None:
            self._conn = self._create_connection()
        
        return self._conn
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with proper configuration.
        
        Returns:
            Configured SQLite connection
        """
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False,  # Allow multi-threaded access
            isolation_level=None  # Autocommit mode for explicit transactions
        )
        
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        
        # Use Row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        logger.info(f"Database connection established: {self.db_path}")
        
        return conn
    
    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("Database connection closed")
    
    def health_check(self) -> bool:
        """Check if database connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            conn = self.connect()
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            return result is not None
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for atomic database transactions.
        
        Yields:
            Database connection within transaction
            
        Example:
            with db.transaction() as conn:
                conn.execute("INSERT INTO ...")
                conn.execute("UPDATE ...")
                # Automatically commits on success, rolls back on error
        """
        conn = self.connect()
        
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Transaction rolled back: {e}")
            raise
    
    def execute_with_retry(
        self,
        sql: str,
        params: tuple = (),
        max_retries: int = 3,
        retry_delay: float = 0.1
    ) -> sqlite3.Cursor:
        """Execute SQL with exponential backoff retry logic.
        
        Args:
            sql: SQL statement to execute
            params: Query parameters
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            
        Returns:
            Cursor with query results
            
        Raises:
            sqlite3.Error: If all retries fail
        """
        conn = self.connect()
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return conn.execute(sql, params)
            except sqlite3.OperationalError as e:
                last_error = e
                if "locked" in str(e).lower():
                    # Database is locked, retry with exponential backoff
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Database locked, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    # Not a lock error, don't retry
                    raise
        
        # All retries failed
        logger.error(f"Failed after {max_retries} retries: {last_error}")
        raise last_error
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ConnectionPool:
    """Simple connection pool for managing multiple database connections."""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        """Initialize connection pool.
        
        Args:
            db_path: Path to SQLite database file
            pool_size: Maximum number of connections in pool
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self._connections: list[DatabaseConnection] = []
    
    def get_connection(self) -> DatabaseConnection:
        """Get a connection from the pool.
        
        Returns:
            Database connection
        """
        # For SQLite with WAL mode, we typically use a single connection
        # This is a simple implementation that can be extended if needed
        if not self._connections:
            conn = DatabaseConnection(self.db_path)
            self._connections.append(conn)
        
        return self._connections[0]
    
    def close_all(self):
        """Close all connections in the pool."""
        for conn in self._connections:
            conn.close()
        self._connections.clear()
        logger.info("All connections closed")


# Global connection instance (can be initialized by application)
_global_connection: Optional[DatabaseConnection] = None


def get_connection(db_path: Optional[str] = None) -> DatabaseConnection:
    """Get global database connection.
    
    Args:
        db_path: Path to database file (required on first call)
        
    Returns:
        Global database connection
        
    Raises:
        ValueError: If db_path not provided on first call
    """
    global _global_connection
    
    if _global_connection is None:
        if db_path is None:
            raise ValueError("db_path required for first connection")
        _global_connection = DatabaseConnection(db_path)
    
    return _global_connection


def close_connection():
    """Close global database connection."""
    global _global_connection
    
    if _global_connection is not None:
        _global_connection.close()
        _global_connection = None
