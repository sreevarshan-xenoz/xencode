#!/usr/bin/env python3
"""
Benchmark Store

SQLite-based persistence layer for benchmark results.
Provides CRUD operations for benchmark data with indexing for efficient queries.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager


class BenchmarkStore:
    """
    SQLite storage for benchmark results
    
    Features:
    - Persistent benchmark result storage
    - Efficient querying by provider, model, task type
    - Historical data tracking
    - Aggregation queries for analysis
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize benchmark store
        
        Args:
            db_path: Path to SQLite database (default: ~/.xencode/benchmarks.db)
        """
        if db_path is None:
            db_path = str(Path.home() / ".xencode" / "benchmarks.db")
        
        self.db_path = db_path
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Initialize database on each connection (needed for :memory: databases)
        self._init_database_on_connection(conn)
        return conn
    
    def _init_database_on_connection(self, conn: sqlite3.Connection):
        """Initialize database schema on a connection"""
        cursor = conn.cursor()
        
        # Create benchmark_results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                task_type TEXT NOT NULL,
                task_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Performance metrics
                latency_ms REAL,
                tokens_per_sec REAL,
                throughput_rps REAL,
                
                -- Quality metrics
                accuracy_score REAL,
                consistency_score REAL,
                quality_rating REAL,
                
                -- Cost metrics
                cost_per_request REAL,
                cost_per_1k_tokens REAL,
                
                -- Metadata
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                error_message TEXT,
                
                -- Full result JSON for flexibility
                raw_result TEXT
            )
        """)
        
        # Create indexes for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_provider ON benchmark_results(provider)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model ON benchmark_results(model)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_type ON benchmark_results(task_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_run_id ON benchmark_results(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON benchmark_results(created_at)
        """)
        
        # Create benchmark_runs table for tracking runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                suite_name TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT DEFAULT 'running',
                total_tasks INTEGER,
                completed_tasks INTEGER DEFAULT 0,
                config TEXT
            )
        """)
        
        conn.commit()
    
    def _init_database(self):
        """Initialize database schema (for persistent databases)"""
        conn = self._get_connection()
        conn.close()
    
    def save_result(self, result: Dict[str, Any]) -> int:
        """
        Save a benchmark result
        
        Args:
            result: Benchmark result dictionary
            
        Returns:
            Result ID
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            raw_json = json.dumps(result)
            
            cursor.execute("""
                INSERT INTO benchmark_results (
                    run_id, provider, model, task_type, task_name,
                    latency_ms, tokens_per_sec, throughput_rps,
                    accuracy_score, consistency_score, quality_rating,
                    cost_per_request, cost_per_1k_tokens,
                    input_tokens, output_tokens, total_tokens,
                    error_message, raw_result
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.get("run_id", "default"),
                result.get("provider", "unknown"),
                result.get("model", "unknown"),
                result.get("task_type", "general"),
                result.get("task_name"),
                result.get("latency_ms"),
                result.get("tokens_per_sec"),
                result.get("throughput_rps"),
                result.get("accuracy_score"),
                result.get("consistency_score"),
                result.get("quality_rating"),
                result.get("cost_per_request"),
                result.get("cost_per_1k_tokens"),
                result.get("input_tokens"),
                result.get("output_tokens"),
                result.get("total_tokens"),
                result.get("error_message"),
                raw_json,
            ))
            
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def get_result(self, result_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a single benchmark result by ID
        
        Args:
            result_id: Result ID
            
        Returns:
            Result dictionary or None
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT raw_result FROM benchmark_results WHERE id = ?
            """, (result_id,))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row["raw_result"])
            return None
        finally:
            conn.close()
    
    def get_results_by_provider(self, provider: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get results for a specific provider
        
        Args:
            provider: Provider name
            limit: Maximum results to return
            
        Returns:
            List of result dictionaries
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT raw_result FROM benchmark_results 
                WHERE provider = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (provider, limit))
            
            return [json.loads(row["raw_result"]) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_results_by_model(self, provider: str, model: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get results for a specific model"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT raw_result FROM benchmark_results 
                WHERE provider = ? AND model = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (provider, model, limit))
            
            return [json.loads(row["raw_result"]) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_results_by_task_type(self, task_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get results for a specific task type"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT raw_result FROM benchmark_results 
                WHERE task_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (task_type, limit))
            
            return [json.loads(row["raw_result"]) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_aggregate_stats(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get aggregate statistics
        
        Args:
            provider: Filter by provider
            model: Filter by model
            task_type: Filter by task type
            
        Returns:
            Aggregate statistics dictionary
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Build WHERE clause
            conditions = []
            params = []
            
            if provider:
                conditions.append("provider = ?")
                params.append(provider)
            if model:
                conditions.append("model = ?")
                params.append(model)
            if task_type:
                conditions.append("task_type = ?")
                params.append(task_type)
            
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            # Get aggregate metrics
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as count,
                    AVG(latency_ms) as avg_latency,
                    MIN(latency_ms) as min_latency,
                    MAX(latency_ms) as max_latency,
                    AVG(tokens_per_sec) as avg_tokens_per_sec,
                    AVG(accuracy_score) as avg_accuracy,
                    AVG(consistency_score) as avg_consistency,
                    AVG(cost_per_request) as avg_cost
                FROM benchmark_results
                {where_clause}
            """, params)
            
            row = cursor.fetchone()
            
            return {
                "count": row["count"] if row else 0,
                "avg_latency_ms": row["avg_latency"] if row else None,
                "min_latency_ms": row["min_latency"] if row else None,
                "max_latency_ms": row["max_latency"] if row else None,
                "avg_tokens_per_sec": row["avg_tokens_per_sec"] if row else None,
                "avg_accuracy_score": row["avg_accuracy"] if row else None,
                "avg_consistency_score": row["avg_consistency"] if row else None,
                "avg_cost_per_request": row["avg_cost"] if row else None,
            }
        finally:
            conn.close()
    
    def save_run(self, run_id: str, suite_name: str, config: Dict[str, Any], total_tasks: int) -> int:
        """Save benchmark run metadata"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO benchmark_runs (run_id, suite_name, total_tasks, config)
                VALUES (?, ?, ?, ?)
            """, (run_id, suite_name, total_tasks, json.dumps(config)))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def update_run_status(self, run_id: str, status: str, completed_tasks: int):
        """Update benchmark run status"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE benchmark_runs 
                SET status = ?, completed_tasks = ?, completed_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
            """, (status, completed_tasks, run_id))
            conn.commit()
        finally:
            conn.close()
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get benchmark run metadata"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM benchmark_runs WHERE run_id = ?
            """, (run_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "run_id": row["run_id"],
                    "suite_name": row["suite_name"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "status": row["status"],
                    "total_tasks": row["total_tasks"],
                    "completed_tasks": row["completed_tasks"],
                    "config": json.loads(row["config"]) if row["config"] else {},
                }
            return None
        finally:
            conn.close()
    
    def get_recent_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent benchmark runs"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM benchmark_runs 
                ORDER BY started_at DESC 
                LIMIT ?
            """, (limit,))
            
            return [
                {
                    "run_id": row["run_id"],
                    "suite_name": row["suite_name"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "status": row["status"],
                    "total_tasks": row["total_tasks"],
                    "completed_tasks": row["completed_tasks"],
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()
    
    def delete_old_results(self, days: int = 30) -> int:
        """Delete results older than specified days"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM benchmark_results 
                WHERE created_at < datetime('now', ?)
            """, (f'-{days} days',))
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
