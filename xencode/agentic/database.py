"""Database layer for agentic system conversation storage."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class AgenticDatabase:
    """Database for storing agent conversations and tool usage."""
    
    def __init__(self, db_path: str = "agentic_memory.db"):
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """Initialize the database and create tables."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_calls TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                token_count INTEGER,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        
        # Tool usage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER,
                tool_name TEXT NOT NULL,
                tool_input TEXT,
                tool_output TEXT,
                success BOOLEAN,
                execution_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (message_id) REFERENCES messages(id)
            )
        """)
        
        self.conn.commit()
    
    def create_conversation(self, session_id: str, model_name: str, metadata: Optional[Dict] = None) -> int:
        """Create a new conversation."""
        cursor = self.conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute(
            "INSERT INTO conversations (session_id, model_name, metadata) VALUES (?, ?, ?)",
            (session_id, model_name, metadata_json)
        )
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_conversation_by_session(self, session_id: str) -> Optional[Dict]:
        """Get conversation by session ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
        token_count: Optional[int] = None
    ) -> int:
        """Add a message to a conversation."""
        cursor = self.conn.cursor()
        
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        
        cursor.execute(
            """INSERT INTO messages (conversation_id, role, content, tool_calls, token_count)
               VALUES (?, ?, ?, ?, ?)""",
            (conversation_id, role, content, tool_calls_json, token_count)
        )
        
        # Update conversation timestamp
        cursor.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,)
        )
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_conversation_messages(
        self,
        conversation_id: int,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict]:
        """Get messages for a conversation."""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp DESC"
        params = [conversation_id]
        
        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def log_tool_usage(
        self,
        message_id: int,
        tool_name: str,
        tool_input: Dict,
        tool_output: str,
        success: bool,
        execution_time: float
    ):
        """Log tool usage."""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """INSERT INTO tool_usage (message_id, tool_name, tool_input, tool_output, success, execution_time)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (message_id, tool_name, json.dumps(tool_input), tool_output, success, execution_time)
        )
        
        self.conn.commit()
    
    def get_tool_usage_stats(self, conversation_id: Optional[int] = None) -> Dict[str, Any]:
        """Get tool usage statistics."""
        cursor = self.conn.cursor()
        
        if conversation_id:
            query = """
                SELECT tool_name, COUNT(*) as count, AVG(execution_time) as avg_time,
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count
                FROM tool_usage tu
                JOIN messages m ON tu.message_id = m.id
                WHERE m.conversation_id = ?
                GROUP BY tool_name
            """
            cursor.execute(query, (conversation_id,))
        else:
            query = """
                SELECT tool_name, COUNT(*) as count, AVG(execution_time) as avg_time,
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count
                FROM tool_usage
                GROUP BY tool_name
            """
            cursor.execute(query)
        
        rows = cursor.fetchall()
        
        return {row['tool_name']: dict(row) for row in rows}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
