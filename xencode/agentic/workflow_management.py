"""
Workflow management system for complex task decomposition in Xencode
"""
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import sqlite3
import threading
from pathlib import Path
from collections import defaultdict, deque
import networkx as nx  # For dependency graph representation


class TaskStatus(Enum):
    """Status of a task in the workflow."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(Enum):
    """Types of tasks in the workflow."""
    SIMPLE = "simple"
    COMPOSITE = "composite"
    DEPENDENT = "dependent"
    PARALLELIZABLE = "parallelizable"


@dataclass
class Subtask:
    """Represents a subtask in a decomposed workflow."""
    subtask_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_task_id: str = ""
    description: str = ""
    agent_requirement: str = ""  # Type of agent needed
    estimated_duration: float = 0.0  # in seconds
    dependencies: List[str] = field(default_factory=list)  # subtask_ids this depends on
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    assigned_agent: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """Represents a complete workflow of decomposed tasks."""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_task: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    subtasks: List[Subtask] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoint_points: List[str] = field(default_factory=list)  # subtask_ids that are checkpoints
    
    def get_ready_subtasks(self) -> List[Subtask]:
        """Get subtasks that are ready to be executed (dependencies satisfied)."""
        ready = []
        for subtask in self.subtasks:
            if subtask.status == TaskStatus.PENDING:
                all_deps_met = all(
                    self.get_subtask(dep_id).status == TaskStatus.COMPLETED 
                    for dep_id in subtask.dependencies
                )
                if all_deps_met:
                    ready.append(subtask)
        return ready
    
    def get_subtask(self, subtask_id: str) -> Optional[Subtask]:
        """Get a specific subtask by ID."""
        for subtask in self.subtasks:
            if subtask.subtask_id == subtask_id:
                return subtask
        return None
    
    def update_subtask_status(self, subtask_id: str, status: TaskStatus, result: Optional[str] = None):
        """Update the status of a subtask."""
        subtask = self.get_subtask(subtask_id)
        if subtask:
            subtask.status = status
            if result:
                subtask.result = result
            if status == TaskStatus.IN_PROGRESS and not subtask.start_time:
                subtask.start_time = datetime.now()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and not subtask.end_time:
                subtask.end_time = datetime.now()
    
    def is_workflow_complete(self) -> bool:
        """Check if all subtasks are completed."""
        return all(subtask.status == TaskStatus.COMPLETED for subtask in self.subtasks)
    
    def get_completion_percentage(self) -> float:
        """Get the percentage of completed subtasks."""
        if not self.subtasks:
            return 0.0
        completed = sum(1 for subtask in self.subtasks if subtask.status == TaskStatus.COMPLETED)
        return (completed / len(self.subtasks)) * 100.0


class TaskDecompositionEngine:
    """Engine for decomposing complex tasks into subtasks."""
    
    def __init__(self):
        self.decomposition_rules: Dict[str, List[Dict[str, Any]]] = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default decomposition rules."""
        self.decomposition_rules = {
            "web_application": [
                {
                    "description": "Design system architecture",
                    "agent_requirement": "planning",
                    "estimated_duration": 3600.0,  # 1 hour
                    "dependencies": [],
                    "priority": TaskPriority.HIGH
                },
                {
                    "description": "Create frontend components",
                    "agent_requirement": "code",
                    "estimated_duration": 7200.0,  # 2 hours
                    "dependencies": ["Design system architecture"],
                    "priority": TaskPriority.HIGH
                },
                {
                    "description": "Implement backend services",
                    "agent_requirement": "code",
                    "estimated_duration": 10800.0,  # 3 hours
                    "dependencies": ["Design system architecture"],
                    "priority": TaskPriority.HIGH
                },
                {
                    "description": "Integrate frontend and backend",
                    "agent_requirement": "code",
                    "estimated_duration": 5400.0,  # 1.5 hours
                    "dependencies": ["Create frontend components", "Implement backend services"],
                    "priority": TaskPriority.HIGH
                },
                {
                    "description": "Test integrated system",
                    "agent_requirement": "testing",
                    "estimated_duration": 7200.0,  # 2 hours
                    "dependencies": ["Integrate frontend and backend"],
                    "priority": TaskPriority.HIGH
                }
            ],
            "data_analysis": [
                {
                    "description": "Load and validate data",
                    "agent_requirement": "data_science",
                    "estimated_duration": 1800.0,  # 30 mins
                    "dependencies": [],
                    "priority": TaskPriority.HIGH
                },
                {
                    "description": "Clean and preprocess data",
                    "agent_requirement": "data_science",
                    "estimated_duration": 3600.0,  # 1 hour
                    "dependencies": ["Load and validate data"],
                    "priority": TaskPriority.HIGH
                },
                {
                    "description": "Perform exploratory analysis",
                    "agent_requirement": "data_science",
                    "estimated_duration": 5400.0,  # 1.5 hours
                    "dependencies": ["Clean and preprocess data"],
                    "priority": TaskPriority.MEDIUM
                },
                {
                    "description": "Build predictive model",
                    "agent_requirement": "data_science",
                    "estimated_duration": 10800.0,  # 3 hours
                    "dependencies": ["Perform exploratory analysis"],
                    "priority": TaskPriority.HIGH
                },
                {
                    "description": "Validate model performance",
                    "agent_requirement": "data_science",
                    "estimated_duration": 3600.0,  # 1 hour
                    "dependencies": ["Build predictive model"],
                    "priority": TaskPriority.HIGH
                },
                {
                    "description": "Generate reports and visualizations",
                    "agent_requirement": "data_science",
                    "estimated_duration": 3600.0,  # 1 hour
                    "dependencies": ["Validate model performance"],
                    "priority": TaskPriority.MEDIUM
                }
            ],
            "security_audit": [
                {
                    "description": "Identify system components",
                    "agent_requirement": "security_analysis",
                    "estimated_duration": 1800.0,  # 30 mins
                    "dependencies": [],
                    "priority": TaskPriority.HIGH
                },
                {
                    "description": "Scan for known vulnerabilities",
                    "agent_requirement": "security_analysis",
                    "estimated_duration": 7200.0,  # 2 hours
                    "dependencies": ["Identify system components"],
                    "priority": TaskPriority.CRITICAL
                },
                {
                    "description": "Assess access controls",
                    "agent_requirement": "security_analysis",
                    "estimated_duration": 5400.0,  # 1.5 hours
                    "dependencies": ["Identify system components"],
                    "priority": TaskPriority.HIGH
                },
                {
                    "description": "Test authentication mechanisms",
                    "agent_requirement": "security_analysis",
                    "estimated_duration": 7200.0,  # 2 hours
                    "dependencies": ["Assess access controls"],
                    "priority": TaskPriority.CRITICAL
                },
                {
                    "description": "Document findings and recommendations",
                    "agent_requirement": "documentation",
                    "estimated_duration": 3600.0,  # 1 hour
                    "dependencies": ["Scan for known vulnerabilities", "Test authentication mechanisms"],
                    "priority": TaskPriority.HIGH
                }
            ]
        }
    
    def decompose_task(self, task_description: str, workflow_id: str) -> Workflow:
        """Decompose a complex task into subtasks based on rules."""
        # Determine task type based on keywords
        task_type = self._classify_task_type(task_description)
        
        # Get decomposition rules for this task type
        rules = self.decomposition_rules.get(task_type, self._generic_decomposition(task_description))
        
        # Create subtasks based on rules
        subtasks = []
        for i, rule in enumerate(rules):
            # Find dependency IDs based on description
            dep_ids = []
            for dep_desc in rule.get("dependencies", []):
                for j, prev_rule in enumerate(rules[:i]):
                    if prev_rule["description"] == dep_desc:
                        # Find the corresponding subtask ID
                        for subtask in subtasks:
                            if subtask.description == dep_desc:
                                dep_ids.append(subtask.subtask_id)
                        break
            
            subtask = Subtask(
                parent_task_id=workflow_id,
                description=rule["description"],
                agent_requirement=rule["agent_requirement"],
                estimated_duration=rule["estimated_duration"],
                dependencies=dep_ids,
                priority=TaskPriority(rule.get("priority", "medium")),
                metadata=rule.get("metadata", {})
            )
            subtasks.append(subtask)
        
        # Create workflow
        workflow = Workflow(
            workflow_id=workflow_id,
            original_task=task_description,
            description=f"Decomposed workflow for: {task_description}",
            subtasks=subtasks,
            priority=self._determine_priority(task_description)
        )
        
        # Set some checkpoint points (every 2nd subtask for demonstration)
        for i in range(1, len(subtasks), 2):
            workflow.checkpoint_points.append(subtasks[i].subtask_id)
        
        return workflow
    
    def _classify_task_type(self, task_description: str) -> str:
        """Classify the task type based on keywords."""
        desc_lower = task_description.lower()
        
        if any(keyword in desc_lower for keyword in ["web", "website", "application", "frontend", "backend", "api"]):
            return "web_application"
        elif any(keyword in desc_lower for keyword in ["data", "analysis", "analyze", "dataset", "model", "predict"]):
            return "data_analysis"
        elif any(keyword in desc_lower for keyword in ["security", "vulnerability", "audit", "secure", "penetration"]):
            return "security_audit"
        else:
            return "web_application"  # Default to web app for demo purposes
    
    def _determine_priority(self, task_description: str) -> TaskPriority:
        """Determine priority based on keywords."""
        desc_lower = task_description.lower()
        
        if any(keyword in desc_lower for keyword in ["urgent", "critical", "emergency", "security", "vulnerability"]):
            return TaskPriority.CRITICAL
        elif any(keyword in desc_lower for keyword in ["important", "high priority", "deadline"]):
            return TaskPriority.HIGH
        else:
            return TaskPriority.MEDIUM
    
    def _generic_decomposition(self, task_description: str) -> List[Dict[str, Any]]:
        """Create a generic decomposition if no specific rules match."""
        return [
            {
                "description": f"Analyze requirements for: {task_description[:50]}",
                "agent_requirement": "planning",
                "estimated_duration": 1800.0,  # 30 mins
                "dependencies": [],
                "priority": "high"
            },
            {
                "description": f"Execute main task: {task_description[:50]}",
                "agent_requirement": "general",
                "estimated_duration": 7200.0,  # 2 hours
                "dependencies": [f"Analyze requirements for: {task_description[:50]}"],
                "priority": "high"
            },
            {
                "description": f"Review and validate results of: {task_description[:50]}",
                "agent_requirement": "testing",
                "estimated_duration": 3600.0,  # 1 hour
                "dependencies": [f"Execute main task: {task_description[:50]}"],
                "priority": "medium"
            }
        ]


class DependencyManager:
    """Manages dependencies between subtasks in workflows."""
    
    def __init__(self):
        self.dependency_graphs: Dict[str, nx.DiGraph] = {}  # workflow_id -> graph
        self.access_lock = threading.RLock()
    
    def create_dependency_graph(self, workflow: Workflow) -> nx.DiGraph:
        """Create a dependency graph for a workflow."""
        graph = nx.DiGraph()
        
        # Add nodes for each subtask
        for subtask in workflow.subtasks:
            graph.add_node(subtask.subtask_id, subtask=subtask)
        
        # Add edges for dependencies
        for subtask in workflow.subtasks:
            for dep_id in subtask.dependencies:
                if dep_id in graph.nodes:
                    graph.add_edge(dep_id, subtask.subtask_id)
        
        self.dependency_graphs[workflow.workflow_id] = graph
        return graph
    
    def get_executable_subtasks(self, workflow: Workflow) -> List[Subtask]:
        """Get subtasks that can be executed (no pending dependencies)."""
        graph = self.dependency_graphs.get(workflow.workflow_id)
        if not graph:
            graph = self.create_dependency_graph(workflow)
        
        executable = []
        for subtask in workflow.subtasks:
            if subtask.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                all_deps_satisfied = True
                for dep_id in subtask.dependencies:
                    dep_subtask = workflow.get_subtask(dep_id)
                    if dep_subtask and dep_subtask.status != TaskStatus.COMPLETED:
                        all_deps_satisfied = False
                        break
                
                if all_deps_satisfied:
                    executable.append(subtask)
        
        return executable
    
    def detect_cycles(self, workflow: Workflow) -> bool:
        """Detect if there are cycles in the dependency graph."""
        graph = self.dependency_graphs.get(workflow.workflow_id)
        if not graph:
            graph = self.create_dependency_graph(workflow)
        
        try:
            # NetworkX raises NetworkXNoCycle if no cycle exists
            cycle = list(nx.find_cycle(graph, orientation="original"))
            return len(cycle) > 0
        except nx.NetworkXNoCycle:
            return False
    
    def get_dependency_path(self, workflow: Workflow, start_subtask_id: str, 
                           end_subtask_id: str) -> Optional[List[str]]:
        """Get the dependency path between two subtasks."""
        graph = self.dependency_graphs.get(workflow.workflow_id)
        if not graph:
            graph = self.create_dependency_graph(workflow)
        
        try:
            path = nx.shortest_path(graph, source=start_subtask_id, target=end_subtask_id)
            return path
        except nx.NetworkXNoPath:
            return None


class CheckpointManager:
    """Manages checkpoints and recovery for workflows."""
    
    def __init__(self, db_path: str = "checkpoints.db"):
        self.db_path = db_path
        self.access_lock = threading.RLock()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize the checkpoints database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create checkpoints table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                workflow_id TEXT,
                subtask_id TEXT,
                status TEXT,
                result TEXT,
                timestamp TEXT,
                metadata TEXT
            )
        ''')
        
        # Create workflow_states table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workflow_states (
                workflow_id TEXT PRIMARY KEY,
                state_data TEXT,
                timestamp TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoint_workflow ON checkpoints(workflow_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoint_subtask ON checkpoints(subtask_id)')
        
        conn.commit()
        conn.close()
    
    def create_checkpoint(self, workflow: Workflow, subtask_id: str) -> str:
        """Create a checkpoint for a specific subtask."""
        checkpoint_id = str(uuid.uuid4())
        subtask = workflow.get_subtask(subtask_id)
        
        if not subtask:
            raise ValueError(f"Subtask {subtask_id} not found in workflow")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO checkpoints
            (checkpoint_id, workflow_id, subtask_id, status, result, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            checkpoint_id,
            workflow.workflow_id,
            subtask_id,
            subtask.status.value,
            subtask.result,
            datetime.now().isoformat(),
            json.dumps(subtask.metadata)
        ))
        
        conn.commit()
        conn.close()
        
        return checkpoint_id
    
    def restore_from_checkpoint(self, workflow: Workflow, checkpoint_id: str) -> bool:
        """Restore workflow state from a checkpoint."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM checkpoints WHERE checkpoint_id = ?', (checkpoint_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return False
        
        # Update the subtask with checkpoint data
        subtask = workflow.get_subtask(row[2])  # subtask_id is at index 2
        if subtask:
            subtask.status = TaskStatus(row[3])  # status is at index 3
            subtask.result = row[4]  # result is at index 4
            return True
        
        return False
    
    def save_workflow_state(self, workflow: Workflow):
        """Save the entire workflow state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize workflow state
        state_data = {
            'workflow_id': workflow.workflow_id,
            'status': workflow.status.value,
            'subtasks': [
                {
                    'subtask_id': st.subtask_id,
                    'status': st.status.value,
                    'result': st.result,
                    'assigned_agent': st.assigned_agent,
                    'start_time': st.start_time.isoformat() if st.start_time else None,
                    'end_time': st.end_time.isoformat() if st.end_time else None,
                    'error_message': st.error_message
                } for st in workflow.subtasks
            ],
            'checkpoint_points': workflow.checkpoint_points,
            'created_at': workflow.created_at.isoformat(),
            'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
            'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None
        }
        
        cursor.execute('''
            INSERT OR REPLACE INTO workflow_states
            (workflow_id, state_data, timestamp)
            VALUES (?, ?, ?)
        ''', (
            workflow.workflow_id,
            json.dumps(state_data),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def restore_workflow_state(self, workflow_id: str) -> Optional[Workflow]:
        """Restore workflow state from saved state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT state_data FROM workflow_states WHERE workflow_id = ?', (workflow_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        state_data = json.loads(row[0])
        
        # Reconstruct workflow
        workflow = Workflow(
            workflow_id=state_data['workflow_id'],
            status=TaskStatus(state_data['status']),
            checkpoint_points=state_data['checkpoint_points'],
            created_at=datetime.fromisoformat(state_data['created_at']),
            started_at=datetime.fromisoformat(state_data['started_at']) if state_data['started_at'] else None,
            completed_at=datetime.fromisoformat(state_data['completed_at']) if state_data['completed_at'] else None
        )
        
        # Reconstruct subtasks
        for subtask_data in state_data['subtasks']:
            subtask = Subtask(
                subtask_id=subtask_data['subtask_id'],
                status=TaskStatus(subtask_data['status']),
                result=subtask_data['result'],
                assigned_agent=subtask_data['assigned_agent'],
                start_time=datetime.fromisoformat(subtask_data['start_time']) if subtask_data['start_time'] else None,
                end_time=datetime.fromisoformat(subtask_data['end_time']) if subtask_data['end_time'] else None,
                error_message=subtask_data['error_message']
            )
            workflow.subtasks.append(subtask)
        
        return workflow


class WorkflowManager:
    """Main manager for workflow operations."""
    
    def __init__(self, db_path: str = "workflows.db"):
        self.db_path = db_path
        self.decomposition_engine = TaskDecompositionEngine()
        self.dependency_manager = DependencyManager()
        self.checkpoint_manager = CheckpointManager()
        self.workflows: Dict[str, Workflow] = {}
        self.access_lock = threading.RLock()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize the workflows database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create workflows table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                original_task TEXT,
                description TEXT,
                status TEXT,
                priority TEXT,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                metadata TEXT
            )
        ''')
        
        # Create subtasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subtasks (
                subtask_id TEXT PRIMARY KEY,
                workflow_id TEXT,
                description TEXT,
                agent_requirement TEXT,
                estimated_duration REAL,
                dependencies TEXT,
                status TEXT,
                priority TEXT,
                assigned_agent TEXT,
                start_time TEXT,
                end_time TEXT,
                result TEXT,
                error_message TEXT,
                metadata TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_workflow_status ON workflows(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_workflow_created ON workflows(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_subtask_workflow ON subtasks(workflow_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_subtask_status ON subtasks(status)')
        
        conn.commit()
        conn.close()
    
    def create_workflow(self, task_description: str) -> Workflow:
        """Create a new workflow by decomposing the task."""
        workflow_id = str(uuid.uuid4())
        
        # Decompose the task
        workflow = self.decomposition_engine.decompose_task(task_description, workflow_id)
        
        # Store in memory
        with self.access_lock:
            self.workflows[workflow_id] = workflow
        
        # Store in database
        self._save_workflow_to_db(workflow)
        
        # Create dependency graph
        self.dependency_manager.create_dependency_graph(workflow)
        
        return workflow
    
    def _save_workflow_to_db(self, workflow: Workflow):
        """Save workflow to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert workflow
        cursor.execute('''
            INSERT OR REPLACE INTO workflows
            (workflow_id, original_task, description, status, priority, created_at, started_at, completed_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            workflow.workflow_id,
            workflow.original_task,
            workflow.description,
            workflow.status.value,
            workflow.priority.value,
            workflow.created_at.isoformat(),
            workflow.started_at.isoformat() if workflow.started_at else None,
            workflow.completed_at.isoformat() if workflow.completed_at else None,
            json.dumps(workflow.metadata)
        ))
        
        # Insert subtasks
        for subtask in workflow.subtasks:
            cursor.execute('''
                INSERT OR REPLACE INTO subtasks
                (subtask_id, workflow_id, description, agent_requirement, estimated_duration, 
                 dependencies, status, priority, assigned_agent, start_time, end_time, result, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                subtask.subtask_id,
                workflow.workflow_id,
                subtask.description,
                subtask.agent_requirement,
                subtask.estimated_duration,
                json.dumps(subtask.dependencies),
                subtask.status.value,
                subtask.priority.value,
                subtask.assigned_agent,
                subtask.start_time.isoformat() if subtask.start_time else None,
                subtask.end_time.isoformat() if subtask.end_time else None,
                subtask.result,
                subtask.error_message,
                json.dumps(subtask.metadata)
            ))
        
        conn.commit()
        conn.close()
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        with self.access_lock:
            if workflow_id in self.workflows:
                return self.workflows[workflow_id]
        
        # Try to load from DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM workflows WHERE workflow_id = ?', (workflow_id,))
        workflow_row = cursor.fetchone()
        
        if workflow_row:
            # Load subtasks
            cursor.execute('SELECT * FROM subtasks WHERE workflow_id = ?', (workflow_id,))
            subtask_rows = cursor.fetchall()
            conn.close()
            
            # Reconstruct workflow
            workflow = Workflow(
                workflow_id=workflow_row[0],
                original_task=workflow_row[1],
                description=workflow_row[2],
                status=TaskStatus(workflow_row[3]),
                priority=TaskPriority(workflow_row[4]),
                created_at=datetime.fromisoformat(workflow_row[5]),
                started_at=datetime.fromisoformat(workflow_row[6]) if workflow_row[6] else None,
                completed_at=datetime.fromisoformat(workflow_row[7]) if workflow_row[7] else None,
                metadata=json.loads(workflow_row[8]) if workflow_row[8] else {}
            )
            
            # Add subtasks
            for subtask_row in subtask_rows:
                subtask = Subtask(
                    subtask_id=subtask_row[0],
                    parent_task_id=subtask_row[1],
                    description=subtask_row[2],
                    agent_requirement=subtask_row[3],
                    estimated_duration=subtask_row[4],
                    dependencies=json.loads(subtask_row[5]) if subtask_row[5] else [],
                    status=TaskStatus(subtask_row[6]),
                    priority=TaskPriority(subtask_row[7]),
                    assigned_agent=subtask_row[8],
                    start_time=datetime.fromisoformat(subtask_row[9]) if subtask_row[9] else None,
                    end_time=datetime.fromisoformat(subtask_row[10]) if subtask_row[10] else None,
                    result=subtask_row[11],
                    error_message=subtask_row[12],
                    metadata=json.loads(subtask_row[13]) if subtask_row[13] else {}
                )
                workflow.subtasks.append(subtask)
            
            # Store in memory
            with self.access_lock:
                self.workflows[workflow_id] = workflow
            
            return workflow
        
        conn.close()
        return None
    
    def get_ready_subtasks(self, workflow_id: str) -> List[Subtask]:
        """Get subtasks that are ready to be executed."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return []
        
        return self.dependency_manager.get_executable_subtasks(workflow)
    
    def update_subtask_status(self, workflow_id: str, subtask_id: str, 
                           status: TaskStatus, result: Optional[str] = None) -> bool:
        """Update the status of a subtask."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False
        
        workflow.update_subtask_status(subtask_id, status, result)
        
        # Update workflow status if needed
        if workflow.is_workflow_complete():
            workflow.status = TaskStatus.COMPLETED
            workflow.completed_at = datetime.now()
        elif any(st.status == TaskStatus.FAILED for st in workflow.subtasks):
            workflow.status = TaskStatus.FAILED
        
        # Save to database
        self._save_workflow_to_db(workflow)
        
        # Save checkpoint if this is a checkpoint subtask
        if subtask_id in workflow.checkpoint_points:
            self.checkpoint_manager.create_checkpoint(workflow, subtask_id)
        
        return True
    
    def assign_agent_to_subtask(self, workflow_id: str, subtask_id: str, agent_id: str) -> bool:
        """Assign an agent to a subtask."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False
        
        subtask = workflow.get_subtask(subtask_id)
        if not subtask:
            return False
        
        subtask.assigned_agent = agent_id
        subtask.status = TaskStatus.IN_PROGRESS
        subtask.start_time = datetime.now()
        
        # Update workflow status
        if workflow.status == TaskStatus.PENDING:
            workflow.status = TaskStatus.IN_PROGRESS
            workflow.started_at = datetime.now()
        
        # Save to database
        self._save_workflow_to_db(workflow)
        return True
    
    def detect_workflow_cycles(self, workflow_id: str) -> bool:
        """Detect cycles in the workflow dependencies."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False
        
        return self.dependency_manager.detect_cycles(workflow)
    
    def get_workflow_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Get progress information for a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return {}
        
        total_subtasks = len(workflow.subtasks)
        completed_subtasks = sum(1 for st in workflow.subtasks if st.status == TaskStatus.COMPLETED)
        failed_subtasks = sum(1 for st in workflow.subtasks if st.status == TaskStatus.FAILED)
        in_progress_subtasks = sum(1 for st in workflow.subtasks if st.status == TaskStatus.IN_PROGRESS)
        
        return {
            'workflow_id': workflow_id,
            'original_task': workflow.original_task,
            'status': workflow.status.value,
            'total_subtasks': total_subtasks,
            'completed_subtasks': completed_subtasks,
            'failed_subtasks': failed_subtasks,
            'in_progress_subtasks': in_progress_subtasks,
            'completion_percentage': workflow.get_completion_percentage(),
            'created_at': workflow.created_at.isoformat(),
            'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
            'estimated_completion_time': self._estimate_completion_time(workflow)
        }
    
    def _estimate_completion_time(self, workflow: Workflow) -> Optional[float]:
        """Estimate remaining completion time for the workflow."""
        if not workflow.subtasks:
            return 0.0
        
        remaining_subtasks = [st for st in workflow.subtasks if st.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]]
        if not remaining_subtasks:
            return 0.0
        
        total_estimated = sum(st.estimated_duration for st in remaining_subtasks)
        
        # Adjust based on progress
        completed_count = sum(1 for st in workflow.subtasks if st.status == TaskStatus.COMPLETED)
        total_count = len(workflow.subtasks)
        
        if completed_count > 0:
            # Calculate average time taken vs estimated
            completed_subtasks = [st for st in workflow.subtasks if st.status == TaskStatus.COMPLETED]
            if completed_subtasks:
                actual_times = []
                for st in completed_subtasks:
                    if st.start_time and st.end_time:
                        actual_times.append((st.end_time - st.start_time).total_seconds())
                
                if actual_times:
                    avg_actual_vs_estimated = sum(actual_times) / len(actual_times) / sum(st.estimated_duration for st in completed_subtasks) / len(completed_subtasks)
                    return total_estimated * avg_actual_vs_estimated
        
        return total_estimated


# Helper functions for common operations
def create_workflow_from_task(task_description: str) -> Workflow:
    """Create a workflow directly from a task description."""
    manager = WorkflowManager()
    return manager.create_workflow(task_description)


def get_next_ready_subtasks(workflow_manager: WorkflowManager, workflow_id: str) -> List[Subtask]:
    """Get the next subtasks that are ready to be executed."""
    return workflow_manager.get_ready_subtasks(workflow_id)