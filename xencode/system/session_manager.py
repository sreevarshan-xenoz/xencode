#!/usr/bin/env python3
"""
Session Manager for Xencode Warp Terminal

Handles session persistence, crash recovery, and command history management
with SQLite backend for reliable storage.
"""

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
import pickle
import zlib

from ..warp_terminal import CommandBlock


logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about a terminal session"""
    id: str
    start_time: datetime
    end_time: Optional[datetime]
    command_count: int
    is_active: bool


class SessionManager:
    """Manages terminal sessions with persistence and crash recovery"""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.home() / ".xencode" / "sessions.db"
        
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.connection = None
        self.lock = threading.Lock()
        
        self._init_database()
        self._create_current_session()

    def _init_database(self):
        """Initialize the SQLite database"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Enable WAL mode for better concurrency
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA cache_size=1000")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        
        # Create tables
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                command_count INTEGER,
                is_active BOOLEAN,
                metadata TEXT
            )
        """)
        
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS commands (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                command TEXT,
                input_data BLOB,
                output_data BLOB,
                metadata TEXT,
                timestamp REAL,
                tags TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_commands_session ON commands(session_id);
        """)
        
        self.connection.commit()

    def _create_current_session(self):
        """Create a new session record"""
        self.current_session_id = f"session_{int(time.time())}_{threading.get_ident()}"
        
        with self.lock:
            self.connection.execute(
                "INSERT INTO sessions (id, start_time, command_count, is_active, metadata) VALUES (?, ?, ?, ?, ?)",
                (self.current_session_id, time.time(), 0, True, json.dumps({}))
            )
            self.connection.commit()

    def save_command_block(self, block):
        """Save a command block to the current session"""
        try:
            # Handle both CommandBlock and LazyCommandBlock
            # Extract attributes with fallbacks for LazyCommandBlock
            block_id = getattr(block, 'id', f"lazy_{int(time.time())}")
            command = getattr(block, 'command', '')
            output_data = getattr(block, 'output_data', {})
            metadata = getattr(block, 'metadata', {})
            tags = getattr(block, 'tags', [])

            # Handle input_data - might not exist in LazyCommandBlock
            input_data = getattr(block, 'input_data', {})

            # Handle timestamp - could be datetime object or float
            timestamp_obj = getattr(block, 'timestamp', time.time())
            if isinstance(timestamp_obj, datetime):
                timestamp = timestamp_obj.timestamp()
            elif isinstance(timestamp_obj, (int, float)):
                timestamp = timestamp_obj
            else:
                timestamp = time.time()

            # Serialize the block data
            input_data_compressed = zlib.compress(pickle.dumps(input_data))
            output_data_compressed = zlib.compress(pickle.dumps(output_data))

            with self.lock:
                self.connection.execute(
                    """INSERT INTO commands
                    (id, session_id, command, input_data, output_data, metadata, timestamp, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        block_id,
                        self.current_session_id,
                        command,
                        input_data_compressed,
                        output_data_compressed,
                        json.dumps(metadata),
                        timestamp,
                        json.dumps(tags)
                    )
                )

                # Update session command count
                self.connection.execute(
                    "UPDATE sessions SET command_count = command_count + 1 WHERE id = ?",
                    (self.current_session_id,)
                )

                self.connection.commit()

        except Exception as e:
            logger.error(f"Failed to save command block {getattr(block, 'id', 'unknown')}: {e}")

    def load_session_commands(self, session_id: str) -> List[CommandBlock]:
        """Load all command blocks for a session"""
        try:
            with self.lock:
                cursor = self.connection.execute(
                    """SELECT id, command, input_data, output_data, metadata, timestamp, tags 
                    FROM commands WHERE session_id = ? ORDER BY timestamp""",
                    (session_id,)
                )
                
                blocks = []
                for row in cursor.fetchall():
                    block_id, command, input_data_compressed, output_data_compressed, metadata_str, timestamp, tags_str = row
                    
                    # Decompress and deserialize
                    input_data = pickle.loads(zlib.decompress(input_data_compressed))
                    output_data = pickle.loads(zlib.decompress(output_data_compressed))
                    metadata = json.loads(metadata_str)
                    tags = json.loads(tags_str)
                    
                    block = CommandBlock(
                        id=block_id,
                        command=command,
                        input_data=input_data,
                        output_data=output_data,
                        metadata=metadata,
                        timestamp=datetime.fromtimestamp(timestamp),
                        tags=tags
                    )
                    
                    blocks.append(block)
                
                return blocks
                
        except Exception as e:
            logger.error(f"Failed to load session commands for {session_id}: {e}")
            return []

    def get_recent_sessions(self, limit: int = 10) -> List[SessionInfo]:
        """Get information about recent sessions"""
        try:
            with self.lock:
                cursor = self.connection.execute(
                    """SELECT id, start_time, end_time, command_count, is_active 
                    FROM sessions ORDER BY start_time DESC LIMIT ?""",
                    (limit,)
                )
                
                sessions = []
                for row in cursor.fetchall():
                    session_id, start_time, end_time, command_count, is_active = row
                    
                    session_info = SessionInfo(
                        id=session_id,
                        start_time=datetime.fromtimestamp(start_time),
                        end_time=datetime.fromtimestamp(end_time) if end_time else None,
                        command_count=command_count,
                        is_active=bool(is_active)
                    )
                    
                    sessions.append(session_info)
                
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to get recent sessions: {e}")
            return []

    def close_current_session(self):
        """Mark the current session as closed"""
        try:
            with self.lock:
                self.connection.execute(
                    "UPDATE sessions SET end_time = ?, is_active = ? WHERE id = ?",
                    (time.time(), False, self.current_session_id)
                )
                self.connection.commit()
                
        except Exception as e:
            logger.error(f"Failed to close session {self.current_session_id}: {e}")

    def recover_crashed_sessions(self) -> List[str]:
        """Find and recover crashed sessions (those marked as active but potentially abandoned)"""
        try:
            with self.lock:
                cursor = self.connection.execute(
                    "SELECT id FROM sessions WHERE is_active = 1"
                )
                
                active_session_ids = [row[0] for row in cursor.fetchall()]
                
                # In a real implementation, you might want to check if these sessions
                # are actually still running by checking process IDs or timestamps
                # For now, we'll just return them as potentially crashed
                return active_session_ids
                
        except Exception as e:
            logger.error(f"Failed to recover crashed sessions: {e}")
            return []

    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session"""
        try:
            with self.lock:
                # Get session info
                cursor = self.connection.execute(
                    "SELECT start_time, end_time, command_count, is_active FROM sessions WHERE id = ?",
                    (session_id,)
                )
                
                row = cursor.fetchone()
                if not row:
                    return {}
                
                start_time, end_time, command_count, is_active = row
                
                # Get command statistics
                cursor = self.connection.execute(
                    """SELECT COUNT(*) as total, 
                              AVG(CAST(JSON_EXTRACT(metadata, '$.duration_ms') AS REAL)) as avg_duration,
                              SUM(CASE WHEN JSON_EXTRACT(metadata, '$.exit_code') = 0 THEN 1 ELSE 0 END) as successful
                       FROM commands WHERE session_id = ?""",
                    (session_id,)
                )
                
                stats_row = cursor.fetchone()
                if stats_row:
                    total_commands, avg_duration, successful_commands = stats_row
                else:
                    total_commands = avg_duration = successful_commands = 0
                
                return {
                    "session_id": session_id,
                    "start_time": datetime.fromtimestamp(start_time),
                    "end_time": datetime.fromtimestamp(end_time) if end_time else None,
                    "is_active": bool(is_active),
                    "command_count": command_count,
                    "total_commands": total_commands,
                    "average_duration_ms": avg_duration,
                    "successful_commands": successful_commands,
                    "failure_rate": (total_commands - successful_commands) / total_commands if total_commands > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get session statistics for {session_id}: {e}")
            return {}

    def cleanup_old_sessions(self, days_to_keep: int = 30):
        """Remove sessions older than specified days"""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
            
            with self.lock:
                # Delete old commands first
                self.connection.execute(
                    """DELETE FROM commands WHERE session_id IN 
                    (SELECT id FROM sessions WHERE start_time < ? AND is_active = 0)""",
                    (cutoff_time,)
                )
                
                # Then delete old sessions
                self.connection.execute(
                    "DELETE FROM sessions WHERE start_time < ? AND is_active = 0",
                    (cutoff_time,)
                )
                
                self.connection.commit()
                
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(db_path: Optional[Path] = None) -> SessionManager:
    """Get the global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(db_path)
    return _session_manager


def cleanup_on_exit():
    """Call this function when the application exits to properly close the session"""
    global _session_manager
    if _session_manager:
        _session_manager.close_current_session()
        _session_manager = None


if __name__ == "__main__":
    import tempfile
    
    # Test the session manager
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        sm = SessionManager(db_path)
        
        print(f"Created session: {sm.current_session_id}")
        
        # Create a test command block
        from datetime import datetime
        test_block = CommandBlock(
            id="test_1",
            command="echo 'hello world'",
            input_data={},
            output_data={"type": "text", "data": "hello world"},
            metadata={"exit_code": 0, "duration_ms": 100},
            timestamp=datetime.now(),
            tags=["test", "echo"]
        )
        
        # Save the block
        sm.save_command_block(test_block)
        print("Saved command block")
        
        # Load the session commands
        commands = sm.load_session_commands(sm.current_session_id)
        print(f"Loaded {len(commands)} commands")
        
        # Get session statistics
        stats = sm.get_session_statistics(sm.current_session_id)
        print(f"Session stats: {stats}")
        
        # Close the session
        sm.close_current_session()
        print("Closed session")
        
        # Get recent sessions
        recent = sm.get_recent_sessions()
        print(f"Recent sessions: {len(recent)}")