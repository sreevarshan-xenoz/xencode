"""Database migration system for crush."""

import sqlite3
import time
from pathlib import Path
from typing import List, Tuple


class Migration:
    """Represents a database migration."""
    
    def __init__(self, version: int, description: str, sql: str):
        self.version = version
        self.description = description
        self.sql = sql


# Define migrations in order
MIGRATIONS: List[Migration] = [
    Migration(
        version=1,
        description="Initial schema",
        sql="""
        -- Enable WAL mode for better concurrency
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;
        PRAGMA synchronous=NORMAL;

        -- Sessions table
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            cost REAL DEFAULT 0.0,
            summary_message_id TEXT,
            busy INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);

        -- Messages table
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT,
            provider TEXT,
            created_at INTEGER NOT NULL,
            is_summary INTEGER DEFAULT 0,
            token_count INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at);

        -- File history table
        CREATE TABLE IF NOT EXISTS file_history (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            content BLOB NOT NULL,
            version INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_file_history_path ON file_history(file_path);

        -- Permission requests table
        CREATE TABLE IF NOT EXISTS permission_requests (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            tool_call_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            action TEXT NOT NULL,
            path TEXT,
            description TEXT,
            params TEXT,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            resolved_at INTEGER,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_perm_session_status ON permission_requests(session_id, status);

        -- Schema version tracking
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at INTEGER NOT NULL
        );
        """
    ),
]


class MigrationRunner:
    """Manages database migrations."""
    
    def __init__(self, db_path: str):
        """Initialize migration runner.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_directory()
    
    def _ensure_db_directory(self):
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def get_current_version(self, conn: sqlite3.Connection) -> int:
        """Get current schema version.
        
        Args:
            conn: Database connection
            
        Returns:
            Current schema version, or 0 if no migrations applied
        """
        try:
            cursor = conn.execute(
                "SELECT MAX(version) FROM schema_version"
            )
            result = cursor.fetchone()
            return result[0] if result[0] is not None else 0
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return 0
    
    def get_pending_migrations(self, current_version: int) -> List[Migration]:
        """Get list of pending migrations.
        
        Args:
            current_version: Current schema version
            
        Returns:
            List of migrations to apply
        """
        return [m for m in MIGRATIONS if m.version > current_version]
    
    def apply_migration(
        self,
        conn: sqlite3.Connection,
        migration: Migration
    ) -> None:
        """Apply a single migration.
        
        Args:
            conn: Database connection
            migration: Migration to apply
        """
        print(f"Applying migration {migration.version}: {migration.description}")
        
        # Execute migration SQL
        conn.executescript(migration.sql)
        
        # Record migration
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (migration.version, int(time.time()))
        )
        
        conn.commit()
    
    def run(self) -> Tuple[int, int]:
        """Run all pending migrations.
        
        Returns:
            Tuple of (current_version, migrations_applied)
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys=ON")
        
        try:
            current_version = self.get_current_version(conn)
            pending = self.get_pending_migrations(current_version)
            
            if not pending:
                print(f"Database is up to date (version {current_version})")
                return current_version, 0
            
            print(f"Current version: {current_version}")
            print(f"Pending migrations: {len(pending)}")
            
            for migration in pending:
                self.apply_migration(conn, migration)
            
            new_version = self.get_current_version(conn)
            print(f"Migrations complete. New version: {new_version}")
            
            return new_version, len(pending)
        
        finally:
            conn.close()
    
    def reset(self) -> None:
        """Reset database by dropping all tables.
        
        WARNING: This will delete all data!
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get all tables
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
            # Drop all tables
            for table in tables:
                if table != 'sqlite_sequence':
                    conn.execute(f"DROP TABLE IF EXISTS {table}")
            
            conn.commit()
            print("Database reset complete")
        
        finally:
            conn.close()


def run_migrations(db_path: str) -> Tuple[int, int]:
    """Run database migrations.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        Tuple of (current_version, migrations_applied)
    """
    runner = MigrationRunner(db_path)
    return runner.run()
