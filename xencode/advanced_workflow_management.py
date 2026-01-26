#!/usr/bin/env python3
"""
Advanced Workflow Management System for Xencode

Sophisticated workflow engine for managing complex development tasks,
automated processes, and multi-step operations with dependencies and monitoring.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from pathlib import Path
import sqlite3
import inspect
from concurrent.futures import ThreadPoolExecutor
import logging

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.tree import Tree
import aiofiles

console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowPriority(Enum):
    """Workflow priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class WorkflowDefinition:
    """Definition of a workflow template"""
    id: str
    name: str
    description: str
    tasks: List['TaskDefinition']
    dependencies: Dict[str, List[str]]  # task_id -> list of dependent task_ids
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    version: str = "1.0.0"


@dataclass
class TaskDefinition:
    """Definition of a task within a workflow"""
    id: str
    name: str
    description: str
    function_ref: str  # Reference to function to execute
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300  # 5 minutes default
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    required_resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowInstance:
    """Runtime instance of a workflow"""
    id: str
    definition_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    current_task_id: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    priority: WorkflowPriority = WorkflowPriority.NORMAL
    progress: float = 0.0


@dataclass
class TaskInstance:
    """Runtime instance of a task"""
    id: str
    workflow_id: str
    definition: TaskDefinition
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_time: Optional[float] = None


class WorkflowEngine:
    """Core workflow execution engine"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".xencode" / "workflows.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Registered functions that can be used in workflows
        self.registered_functions: Dict[str, Callable] = {}
        
        # Active workflow instances
        self.workflow_instances: Dict[str, WorkflowInstance] = {}
        self.task_instances: Dict[str, TaskInstance] = {}
        
        # Execution queues
        self.ready_queue: List[str] = []  # workflow IDs ready to execute
        self.running_workflows: List[str] = []  # currently executing workflow IDs
        
        # Thread pool for function execution
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        # Running state
        self.running = False
        self.execution_task: Optional[asyncio.Task] = None

    def _init_database(self):
        """Initialize the workflow database"""
        with sqlite3.connect(self.db_path) as conn:
            # Create workflow definitions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_definitions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    definition TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version TEXT
                )
            """)
            
            # Create workflow instances table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_instances (
                    id TEXT PRIMARY KEY,
                    definition_id TEXT,
                    status TEXT,
                    created_at REAL,
                    started_at REAL,
                    completed_at REAL,
                    current_task_id TEXT,
                    variables TEXT,
                    results TEXT,
                    error TEXT,
                    priority INTEGER,
                    progress REAL,
                    FOREIGN KEY (definition_id) REFERENCES workflow_definitions(id)
                )
            """)
            
            # Create task instances table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_instances (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    task_definition_id TEXT,
                    status TEXT,
                    created_at REAL,
                    started_at REAL,
                    completed_at REAL,
                    result TEXT,
                    error TEXT,
                    retry_count INTEGER,
                    execution_time REAL,
                    FOREIGN KEY (workflow_id) REFERENCES workflow_instances(id)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workflow_def_id ON workflow_instances(definition_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workflow_status ON workflow_instances(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_workflow ON task_instances(workflow_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_status ON task_instances(status)")

    def register_function(self, name: str, func: Callable):
        """Register a function that can be used in workflows"""
        self.registered_functions[name] = func
        logger.info(f"Registered function: {name}")

    def define_workflow(self, definition: WorkflowDefinition) -> str:
        """Define a new workflow template"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO workflow_definitions 
                (id, name, description, definition, metadata, version)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                definition.id,
                definition.name,
                definition.description,
                json.dumps([task.__dict__ for task in definition.tasks]),
                json.dumps(definition.metadata),
                definition.version
            ))
        
        logger.info(f"Defined workflow: {definition.name} ({definition.id})")
        return definition.id

    def create_workflow_instance(self, definition_id: str, 
                               variables: Dict[str, Any] = None,
                               priority: WorkflowPriority = WorkflowPriority.NORMAL) -> str:
        """Create a new workflow instance from a definition"""
        # Load workflow definition
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM workflow_definitions WHERE id = ?", (definition_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Workflow definition {definition_id} not found")
        
        # Create workflow instance
        instance = WorkflowInstance(
            id=f"wf_{uuid.uuid4()}",
            definition_id=definition_id,
            variables=variables or {},
            priority=priority
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO workflow_instances
                (id, definition_id, status, created_at, variables, priority, progress)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                instance.id,
                instance.definition_id,
                instance.status.value,
                instance.created_at,
                json.dumps(instance.variables),
                instance.priority.value,
                instance.progress
            ))
        
        # Create task instances
        self._create_task_instances(instance.id, definition_id)
        
        logger.info(f"Created workflow instance: {instance.id}")
        return instance.id

    def _create_task_instances(self, workflow_id: str, definition_id: str):
        """Create task instances for a workflow"""
        # Load workflow definition
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT definition FROM workflow_definitions WHERE id = ?", (definition_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Workflow definition {definition_id} not found")
        
        task_definitions = json.loads(row[0])
        
        for task_def_data in task_definitions:
            task_def = TaskDefinition(**task_def_data)
            task_instance = TaskInstance(
                id=f"task_{uuid.uuid4()}",
                workflow_id=workflow_id,
                definition=task_def
            )
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO task_instances
                    (id, workflow_id, task_definition_id, status, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    task_instance.id,
                    task_instance.workflow_id,
                    task_instance.definition.id,
                    task_instance.status.value,
                    task_instance.created_at
                ))

    async def start_workflow(self, workflow_id: str):
        """Start executing a workflow instance"""
        # Load workflow instance
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM workflow_instances WHERE id = ?", (workflow_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Workflow instance {workflow_id} not found")
        
        # Update status to running
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE workflow_instances 
                SET status = ?, started_at = ?
                WHERE id = ?
            """, (WorkflowStatus.RUNNING.value, time.time(), workflow_id))
        
        # Add to ready queue
        self.ready_queue.append(workflow_id)
        logger.info(f"Started workflow: {workflow_id}")

    async def _execute_workflow(self, workflow_id: str):
        """Execute a workflow instance"""
        try:
            # Load workflow and tasks
            workflow = await self._get_workflow_instance(workflow_id)
            tasks = await self._get_task_instances(workflow_id)
            
            # Update workflow status
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE workflow_instances 
                    SET status = ?, current_task_id = ?
                    WHERE id = ?
                """, (WorkflowStatus.RUNNING.value, tasks[0].id if tasks else None, workflow_id))
            
            # Execute tasks in order respecting dependencies
            for task in tasks:
                if workflow.status == WorkflowStatus.CANCELLED:
                    break
                    
                await self._execute_task(task.id)
                
                # Update workflow progress
                total_tasks = len(tasks)
                completed_tasks = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
                progress = completed_tasks / total_tasks if total_tasks > 0 else 0
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE workflow_instances 
                        SET progress = ?
                        WHERE id = ?
                    """, (progress, workflow_id))
        
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {e}")
            
            # Mark workflow as failed
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE workflow_instances 
                    SET status = ?, error = ?
                    WHERE id = ?
                """, (WorkflowStatus.FAILED.value, str(e), workflow_id))

    async def _execute_task(self, task_id: str):
        """Execute a single task"""
        # Load task instance
        task = await self._get_task_instance(task_id)
        
        if task.status != TaskStatus.PENDING and task.status != TaskStatus.READY:
            return
        
        # Update task status
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE task_instances 
                SET status = ?, started_at = ?
                WHERE id = ?
            """, (TaskStatus.RUNNING.value, time.time(), task_id))
        
        # Execute the task function
        result = None
        error = None
        
        try:
            # Get the function to execute
            func = self.registered_functions.get(task.definition.function_ref)
            if not func:
                raise ValueError(f"Function {task.definition.function_ref} not registered")
            
            # Prepare arguments
            args = []
            kwargs = task.definition.parameters.copy()
            
            # If function accepts workflow variables, pass them
            sig = inspect.signature(func)
            if 'workflow_vars' in sig.parameters:
                workflow = await self._get_workflow_instance(task.workflow_id)
                kwargs['workflow_vars'] = workflow.variables
            
            # Execute in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                lambda: func(*args, **kwargs)
            )
            
        except Exception as e:
            error = str(e)
            logger.error(f"Task {task_id} failed: {e}")
        
        # Update task status
        new_status = TaskStatus.COMPLETED if error is None else TaskStatus.FAILED
        execution_time = time.time() - task.started_at if task.started_at else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE task_instances 
                SET status = ?, completed_at = ?, result = ?, error = ?, execution_time = ?
                WHERE id = ?
            """, (
                new_status.value,
                time.time() if new_status == TaskStatus.COMPLETED else None,
                json.dumps(result) if result is not None else None,
                error,
                execution_time,
                task_id
            ))
        
        # Update workflow if task failed
        if error:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE workflow_instances 
                    SET status = ?, error = ?
                    WHERE id = ?
                """, (WorkflowStatus.FAILED.value, error, task.workflow_id))

    async def _get_workflow_instance(self, workflow_id: str) -> WorkflowInstance:
        """Get workflow instance from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM workflow_instances WHERE id = ?", (workflow_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Workflow instance {workflow_id} not found")
        
        return WorkflowInstance(
            id=row[0],
            definition_id=row[1],
            status=WorkflowStatus(row[2]),
            created_at=row[3],
            started_at=row[4],
            completed_at=row[5],
            current_task_id=row[6],
            variables=json.loads(row[7]) if row[7] else {},
            results=json.loads(row[8]) if row[8] else {},
            error=row[9],
            priority=WorkflowPriority(row[10]),
            progress=row[11]
        )

    async def _get_task_instances(self, workflow_id: str) -> List[TaskInstance]:
        """Get task instances for a workflow from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM task_instances WHERE workflow_id = ? ORDER BY created_at",
                (workflow_id,)
            )
            rows = cursor.fetchall()
        
        tasks = []
        for row in rows:
            task_def = await self._get_task_definition(row[2])  # task_definition_id
            tasks.append(TaskInstance(
                id=row[0],
                workflow_id=row[1],
                definition=task_def,
                status=TaskStatus(row[3]),
                created_at=row[4],
                started_at=row[5],
                completed_at=row[6],
                result=json.loads(row[7]) if row[7] else None,
                error=row[8],
                retry_count=row[9],
                execution_time=row[10]
            ))
        
        return tasks

    async def _get_task_definition(self, task_def_id: str) -> TaskDefinition:
        """Get task definition from workflow definition"""
        # This is a simplified version - in practice you'd have a separate table for task definitions
        # For now, we'll reconstruct from the workflow definition
        # This is a limitation of the current implementation
        pass

    def get_workflow_status(self, workflow_id: str) -> WorkflowStatus:
        """Get the status of a workflow"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT status FROM workflow_instances WHERE id = ?", (workflow_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return WorkflowStatus(row[0])
        
        raise ValueError(f"Workflow {workflow_id} not found")

    def get_workflow_progress(self, workflow_id: str) -> float:
        """Get the progress of a workflow"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT progress FROM workflow_instances WHERE id = ?", (workflow_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return row[0]
        
        raise ValueError(f"Workflow {workflow_id} not found")

    def cancel_workflow(self, workflow_id: str):
        """Cancel a running workflow"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE workflow_instances 
                SET status = ? 
                WHERE id = ?
            """, (WorkflowStatus.CANCELLED.value, workflow_id))
        
        # Remove from queues if present
        if workflow_id in self.ready_queue:
            self.ready_queue.remove(workflow_id)
        if workflow_id in self.running_workflows:
            self.running_workflows.remove(workflow_id)

    def pause_workflow(self, workflow_id: str):
        """Pause a running workflow"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE workflow_instances 
                SET status = ? 
                WHERE id = ?
            """, (WorkflowStatus.PAUSED.value, workflow_id))

    def resume_workflow(self, workflow_id: str):
        """Resume a paused workflow"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE workflow_instances 
                SET status = ? 
                WHERE id = ?
            """, (WorkflowStatus.RUNNING.value, workflow_id))
        
        # Add back to ready queue
        if workflow_id not in self.ready_queue:
            self.ready_queue.append(workflow_id)

    async def start_execution_loop(self):
        """Start the workflow execution loop"""
        self.running = True
        
        async def execution_loop():
            while self.running:
                try:
                    # Process ready workflows
                    if self.ready_queue:
                        workflow_id = self.ready_queue.pop(0)
                        self.running_workflows.append(workflow_id)
                        
                        # Execute workflow in background
                        asyncio.create_task(self._execute_workflow(workflow_id))
                    
                    # Sleep briefly to avoid busy-waiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in execution loop: {e}")
                    await asyncio.sleep(1)  # Brief pause before continuing
        
        self.execution_task = asyncio.create_task(execution_loop())

    def stop_execution_loop(self):
        """Stop the workflow execution loop"""
        self.running = False
        if self.execution_task:
            self.execution_task.cancel()

    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow system statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Count workflows by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) 
                FROM workflow_instances 
                GROUP BY status
            """)
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Count total workflows
            cursor = conn.execute("SELECT COUNT(*) FROM workflow_instances")
            total_workflows = cursor.fetchone()[0]
            
            # Count total tasks
            cursor = conn.execute("SELECT COUNT(*) FROM task_instances")
            total_tasks = cursor.fetchone()[0]
            
            # Average execution time
            cursor = conn.execute("""
                SELECT AVG(execution_time) 
                FROM task_instances 
                WHERE execution_time IS NOT NULL
            """)
            avg_task_time = cursor.fetchone()[0] or 0.0

        return {
            "total_workflows": total_workflows,
            "total_tasks": total_tasks,
            "status_distribution": status_counts,
            "ready_queue_size": len(self.ready_queue),
            "running_workflows": len(self.running_workflows),
            "average_task_time": avg_task_time,
            "registered_functions": len(self.registered_functions)
        }


class WorkflowScheduler:
    """Schedules workflows for execution"""

    def __init__(self, workflow_engine: WorkflowEngine):
        self.engine = workflow_engine
        self.scheduled_workflows: Dict[str, Dict[str, Any]] = {}
        self.scheduler_task: Optional[asyncio.Task] = None
        self.running = False

    async def schedule_workflow(self, workflow_id: str, 
                              schedule_time: datetime,
                              variables: Dict[str, Any] = None) -> str:
        """Schedule a workflow to run at a specific time"""
        schedule_id = f"schedule_{uuid.uuid4()}"
        
        self.scheduled_workflows[schedule_id] = {
            "workflow_id": workflow_id,
            "schedule_time": schedule_time.timestamp(),
            "variables": variables or {},
            "created_at": time.time()
        }
        
        logger.info(f"Scheduled workflow {workflow_id} for {schedule_time}")
        return schedule_id

    async def schedule_recurring_workflow(self, workflow_id: str,
                                       cron_expression: str,
                                       variables: Dict[str, Any] = None) -> str:
        """Schedule a recurring workflow (simplified cron-like)"""
        # This is a simplified implementation - a real system would use a proper cron parser
        schedule_id = f"recurring_{uuid.uuid4()}"
        
        self.scheduled_workflows[schedule_id] = {
            "workflow_id": workflow_id,
            "cron_expression": cron_expression,
            "variables": variables or {},
            "created_at": time.time(),
            "last_run": None,
            "next_run": time.time() + 3600  # Default: run in 1 hour
        }
        
        logger.info(f"Scheduled recurring workflow {workflow_id} with cron {cron_expression}")
        return schedule_id

    async def start_scheduler(self):
        """Start the scheduler loop"""
        self.running = True
        
        async def scheduler_loop():
            while self.running:
                try:
                    current_time = time.time()
                    
                    # Check for scheduled workflows to run
                    for schedule_id, schedule_info in list(self.scheduled_workflows.items()):
                        schedule_time = schedule_info.get("schedule_time")
                        if schedule_time and current_time >= schedule_time:
                            # Create and start workflow instance
                            wf_instance_id = self.engine.create_workflow_instance(
                                schedule_info["workflow_id"],
                                schedule_info["variables"]
                            )
                            await self.engine.start_workflow(wf_instance_id)
                            
                            # Remove one-time schedules
                            if "schedule_time" in schedule_info:
                                del self.scheduled_workflows[schedule_id]
                            else:
                                # For recurring, calculate next run time
                                # (simplified - would need proper cron parsing in real implementation)
                                schedule_info["next_run"] = current_time + 3600  # Every hour
                    
                    await asyncio.sleep(1)  # Check every second
                    
                except Exception as e:
                    logger.error(f"Error in scheduler: {e}")
                    await asyncio.sleep(5)
        
        self.scheduler_task = asyncio.create_task(scheduler_loop())

    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()


class WorkflowMonitor:
    """Monitors workflow execution and provides insights"""

    def __init__(self, workflow_engine: WorkflowEngine):
        self.engine = workflow_engine
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False

    async def start_monitoring(self):
        """Start monitoring workflow execution"""
        self.running = True
        
        async def monitoring_loop():
            while self.running:
                try:
                    # Log workflow statistics periodically
                    stats = self.engine.get_workflow_stats()
                    logger.info(f"Workflow Stats: {stats}")
                    
                    await asyncio.sleep(30)  # Log every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in monitoring: {e}")
                    await asyncio.sleep(5)
        
        self.monitoring_task = asyncio.create_task(monitoring_loop())

    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()

    def get_execution_insights(self) -> Dict[str, Any]:
        """Get insights about workflow execution"""
        stats = self.engine.get_workflow_stats()
        
        insights = {
            "system_health": "healthy" if stats["ready_queue_size"] < 10 else "overloaded",
            "efficiency": stats["average_task_time"],
            "throughput": stats.get("completed", 0) / max(stats.get("total_workflows", 1), 1),
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Add recommendations based on stats
        if stats["average_task_time"] > 10:  # More than 10 seconds average
            insights["recommendations"].append({
                "type": "performance",
                "message": "Average task execution time is high, consider optimization",
                "priority": "high"
            })
        
        if stats["ready_queue_size"] > 20:  # Queue is getting long
            insights["recommendations"].append({
                "type": "capacity",
                "message": "Ready queue is growing, consider adding more workers",
                "priority": "medium"
            })
        
        return insights


class WorkflowDashboard:
    """Workflow management dashboard"""

    def __init__(self, workflow_engine: WorkflowEngine, 
                 workflow_scheduler: WorkflowScheduler,
                 workflow_monitor: WorkflowMonitor):
        self.engine = workflow_engine
        self.scheduler = workflow_scheduler
        self.monitor = workflow_monitor

    def display_workflow_dashboard(self):
        """Display workflow management dashboard"""
        stats = self.engine.get_workflow_stats()
        insights = self.monitor.get_execution_insights()
        
        console.print(Panel(
            f"[bold blue]Workflow Management Dashboard[/bold blue]\n"
            f"Total Workflows: {stats['total_workflows']}\n"
            f"Total Tasks: {stats['total_tasks']}\n"
            f"Ready Queue: {stats['ready_queue_size']}\n"
            f"Running Workflows: {stats['running_workflows']}\n"
            f"Avg Task Time: {stats['average_task_time']:.2f}s\n"
            f"Registered Functions: {stats['registered_functions']}",
            title="Workflow System Overview",
            border_style="blue"
        ))

        # Display status breakdown
        table = Table(title="Workflow Status Breakdown")
        table.add_column("Status", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Percentage", style="green")

        total = stats['total_workflows']
        for status, count in stats['status_distribution'].items():
            percentage = (count / total * 100) if total > 0 else 0
            table.add_row(status, str(count), f"{percentage:.1f}%")

        console.print(table)

        # Display insights
        insight_panel = Panel(
            f"[bold]System Health:[/bold] {insights['system_health']}\n"
            f"[bold]Efficiency Score:[/bold] {insights['efficiency']:.2f}s avg\n"
            f"[bold]Throughput:[/bold] {insights['throughput']:.2%}\n\n"
            f"[bold]Recommendations:[/bold]\n",
            title="Execution Insights",
            border_style="yellow"
        )

        for rec in insights.get('recommendations', []):
            insight_panel.renderable += f"• {rec['message']} [Priority: {rec['priority']}]\n"

        console.print(insight_panel)


# Example workflow functions
async def example_code_analysis_task(file_path: str, **kwargs) -> Dict[str, Any]:
    """Example task: Analyze code in a file"""
    await asyncio.sleep(1)  # Simulate processing time
    return {
        "file": file_path,
        "lines_of_code": 100,
        "functions": 5,
        "complexity_score": 0.7
    }


async def example_format_code_task(file_path: str, style: str = "black", **kwargs) -> Dict[str, Any]:
    """Example task: Format code in a file"""
    await asyncio.sleep(0.5)  # Simulate processing time
    return {
        "file": file_path,
        "formatted": True,
        "style_applied": style
    }


async def example_test_generation_task(requirements: str, **kwargs) -> Dict[str, Any]:
    """Example task: Generate tests for requirements"""
    await asyncio.sleep(2)  # Simulate processing time
    return {
        "requirements": requirements,
        "tests_generated": 3,
        "coverage_estimate": 0.85
    }


async def demo_workflow_system():
    """Demonstrate the workflow system capabilities"""
    console.print("[bold green]🔄 Initializing Workflow Management System[/bold green]")
    
    # Initialize components
    engine = WorkflowEngine()
    scheduler = WorkflowScheduler(engine)
    monitor = WorkflowMonitor(engine)
    dashboard = WorkflowDashboard(engine, scheduler, monitor)
    
    # Register example functions
    engine.register_function("code_analysis", example_code_analysis_task)
    engine.register_function("format_code", example_format_code_task)
    engine.register_function("generate_tests", example_test_generation_task)
    
    # Define a sample workflow
    code_review_workflow = WorkflowDefinition(
        id="code_review_001",
        name="Code Review Workflow",
        description="Complete code review including analysis, formatting, and test generation",
        tasks=[
            TaskDefinition(
                id="analyze_code",
                name="Analyze Code",
                description="Analyze the provided code file",
                function_ref="code_analysis",
                parameters={"file_path": "/path/to/code.py"}
            ),
            TaskDefinition(
                id="format_code",
                name="Format Code",
                description="Format the code according to style guidelines",
                function_ref="format_code",
                parameters={"file_path": "/path/to/code.py", "style": "black"}
            ),
            TaskDefinition(
                id="generate_tests",
                name="Generate Tests",
                description="Generate unit tests based on requirements",
                function_ref="generate_tests",
                parameters={"requirements": "Code should handle edge cases"}
            )
        ],
        dependencies={},  # No dependencies for this simple example
        metadata={"category": "development", "priority": "normal"}
    )
    
    # Define the workflow
    engine.define_workflow(code_review_workflow)
    
    # Create and start a workflow instance
    console.print("[blue]📋 Creating workflow instance...[/blue]")
    workflow_id = engine.create_workflow_instance(
        code_review_workflow.id,
        variables={"target_file": "/path/to/code.py"}
    )
    
    # Start execution loop
    await engine.start_execution_loop()
    
    console.print(f"[blue]🚀 Starting workflow {workflow_id}...[/blue]")
    await engine.start_workflow(workflow_id)
    
    # Let it run for a bit
    await asyncio.sleep(5)
    
    # Display dashboard
    console.print("\n[bold]📊 Workflow Dashboard:[/bold]")
    dashboard.display_workflow_dashboard()
    
    # Stop execution
    engine.stop_execution_loop()
    
    console.print("[green]✅ Workflow Management System Demo Completed[/green]")


if __name__ == "__main__":
    # Don't run by default to avoid external dependencies
    # asyncio.run(demo_workflow_system())
    pass