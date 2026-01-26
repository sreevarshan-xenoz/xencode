#!/usr/bin/env python3
"""
Human-in-the-Loop Supervision System for Xencode

Advanced supervision capabilities allowing human oversight and intervention
in AI decision-making processes, with approval workflows and feedback mechanisms.
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
import logging
from enum import IntEnum

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.layout import Layout
from rich.live import Live

console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupervisionLevel(Enum):
    """Levels of human supervision required"""
    AUTOMATIC = "automatic"  # No human oversight
    REVIEW = "review"        # Post-action review
    APPROVAL = "approval"    # Pre-action approval required
    MANUAL = "manual"        # Human performs action directly
    OVERRIDE = "override"    # Human override capability


class DecisionOutcome(Enum):
    """Possible outcomes of human decisions"""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    DEFERRED = "deferred"
    ESCALATED = "escalated"


class FeedbackType(Enum):
    """Types of feedback that can be provided"""
    CORRECTNESS = "correctness"      # Was the output correct?
    QUALITY = "quality"              # How good was the output?
    ETHICS = "ethics"                # Ethical considerations
    ACCURACY = "accuracy"            # Factual accuracy
    RELEVANCE = "relevance"          # Relevance to request
    SAFETY = "safety"                # Safety concerns


class UrgencyLevel(IntEnum):
    """Urgency levels for decisions"""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class DecisionRequest:
    """Request for human decision or approval"""
    id: str
    request_type: str  # e.g., "code_generation", "system_change", "data_access"
    description: str
    context: Dict[str, Any]  # Relevant context for decision
    requested_supervision: SupervisionLevel
    urgency: UrgencyLevel
    created_at: float
    requested_by: str  # Agent or system component
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300  # 5 minutes default


@dataclass
class DecisionResponse:
    """Response to a decision request"""
    request_id: str
    outcome: DecisionOutcome
    approver_id: str
    timestamp: float
    feedback: str = ""
    modified_content: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackRecord:
    """Record of human feedback"""
    id: str
    decision_request_id: str
    feedback_type: FeedbackType
    rating: Optional[float] = None  # 0-1 scale
    comment: str = ""
    created_at: float = field(default_factory=time.time)
    provided_by: str = "human"


@dataclass
class SupervisorProfile:
    """Profile for a human supervisor"""
    id: str
    name: str
    expertise_areas: List[str]
    permissions: List[SupervisionLevel]
    availability: bool = True
    current_assignments: int = 0
    total_decisions: int = 0
    approval_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionApprovalSystem:
    """Core system for managing human approvals and decisions"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".xencode" / "supervision.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Active decision requests
        self.active_requests: Dict[str, DecisionRequest] = {}
        
        # Callbacks for decision events
        self.approval_callbacks: List[Callable] = []
        self.rejection_callbacks: List[Callable] = []
        self.feedback_callbacks: List[Callable] = []
        
        # Supervisors
        self.supervisors: Dict[str, SupervisorProfile] = {}
        
        # Running state
        self.running = False
        self.decision_processor: Optional[asyncio.Task] = None

    def _init_database(self):
        """Initialize the supervision database"""
        with sqlite3.connect(self.db_path) as conn:
            # Create decision requests table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_requests (
                    id TEXT PRIMARY KEY,
                    request_type TEXT,
                    description TEXT,
                    context TEXT,
                    requested_supervision TEXT,
                    urgency INTEGER,
                    created_at REAL,
                    requested_by TEXT,
                    metadata TEXT,
                    timeout_seconds INTEGER,
                    status TEXT DEFAULT 'pending',
                    assigned_supervisor TEXT,
                    expires_at REAL
                )
            """)
            
            # Create decision responses table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_responses (
                    id TEXT PRIMARY KEY,
                    request_id TEXT,
                    outcome TEXT,
                    approver_id TEXT,
                    timestamp REAL,
                    feedback TEXT,
                    modified_content TEXT,
                    metadata TEXT,
                    FOREIGN KEY (request_id) REFERENCES decision_requests(id)
                )
            """)
            
            # Create feedback records table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_records (
                    id TEXT PRIMARY KEY,
                    decision_request_id TEXT,
                    feedback_type TEXT,
                    rating REAL,
                    comment TEXT,
                    created_at REAL,
                    provided_by TEXT,
                    FOREIGN KEY (decision_request_id) REFERENCES decision_requests(id)
                )
            """)
            
            # Create supervisor profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS supervisor_profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    expertise_areas TEXT,
                    permissions TEXT,
                    availability BOOLEAN,
                    current_assignments INTEGER,
                    total_decisions INTEGER,
                    approval_rate REAL,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_status ON decision_requests(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_urgency ON decision_requests(urgency)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_assigned ON decision_requests(assigned_supervisor)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_responses_request ON decision_responses(request_id)")

    def register_approval_callback(self, callback: Callable):
        """Register a callback for approval events"""
        self.approval_callbacks.append(callback)

    def register_rejection_callback(self, callback: Callable):
        """Register a callback for rejection events"""
        self.rejection_callbacks.append(callback)

    def register_feedback_callback(self, callback: Callable):
        """Register a callback for feedback events"""
        self.feedback_callbacks.append(callback)

    def submit_decision_request(self, request: DecisionRequest) -> str:
        """Submit a decision request for human review"""
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO decision_requests
                (id, request_type, description, context, requested_supervision, 
                 urgency, created_at, requested_by, metadata, timeout_seconds, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request.id,
                request.request_type,
                request.description,
                json.dumps(request.context),
                request.requested_supervision.value,
                request.urgency.value,
                request.created_at,
                request.requested_by,
                json.dumps(request.metadata),
                request.timeout_seconds,
                request.created_at + request.timeout_seconds
            ))
        
        # Add to active requests
        self.active_requests[request.id] = request
        
        logger.info(f"Submitted decision request {request.id} for {request.requested_by}")
        return request.id

    def get_pending_requests(self, supervisor_id: Optional[str] = None) -> List[DecisionRequest]:
        """Get pending decision requests"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT id, request_type, description, context, requested_supervision, 
                       urgency, created_at, requested_by, metadata, timeout_seconds
                FROM decision_requests 
                WHERE status = 'pending' AND expires_at > ?
                ORDER BY urgency DESC, created_at ASC
            """
            params = [time.time()]
            
            if supervisor_id:
                # Filter by supervisor permissions
                query += " AND requested_supervision IN (SELECT unnest(permissions) FROM supervisor_profiles WHERE id = ?)"
                params.append(supervisor_id)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        requests = []
        for row in rows:
            request = DecisionRequest(
                id=row[0],
                request_type=row[1],
                description=row[2],
                context=json.loads(row[3]) if row[3] else {},
                requested_supervision=SupervisionLevel(row[4]),
                urgency=UrgencyLevel(row[5]),
                created_at=row[6],
                requested_by=row[7],
                metadata=json.loads(row[8]) if row[8] else {},
                timeout_seconds=row[9]
            )
            requests.append(request)
        
        return requests

    def approve_request(self, request_id: str, approver_id: str, feedback: str = "") -> bool:
        """Approve a decision request"""
        return self._process_decision(request_id, DecisionOutcome.APPROVED, approver_id, feedback)

    def reject_request(self, request_id: str, approver_id: str, feedback: str = "") -> bool:
        """Reject a decision request"""
        return self._process_decision(request_id, DecisionOutcome.REJECTED, approver_id, feedback)

    def defer_request(self, request_id: str, approver_id: str, feedback: str = "") -> bool:
        """Defer a decision request"""
        return self._process_decision(request_id, DecisionOutcome.DEFERRED, approver_id, feedback)

    def escalate_request(self, request_id: str, approver_id: str, feedback: str = "") -> bool:
        """Escalate a decision request"""
        return self._process_decision(request_id, DecisionOutcome.ESCALATED, approver_id, feedback)

    def _process_decision(self, request_id: str, outcome: DecisionOutcome, 
                         approver_id: str, feedback: str = "") -> bool:
        """Process a decision response"""
        try:
            # Update request status
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE decision_requests 
                    SET status = ?, assigned_supervisor = ?
                    WHERE id = ?
                """, (outcome.value, approver_id, request_id))
                
                # Insert response record
                response_id = f"resp_{uuid.uuid4()}"
                conn.execute("""
                    INSERT INTO decision_responses
                    (id, request_id, outcome, approver_id, timestamp, feedback)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    response_id,
                    request_id,
                    outcome.value,
                    approver_id,
                    time.time(),
                    feedback
                ))
            
            # Remove from active requests
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            
            # Trigger callbacks
            if outcome == DecisionOutcome.APPROVED:
                for callback in self.approval_callbacks:
                    try:
                        callback(request_id, approver_id, feedback)
                    except Exception as e:
                        logger.error(f"Approval callback error: {e}")
            elif outcome in [DecisionOutcome.REJECTED, DecisionOutcome.ESCALATED]:
                for callback in self.rejection_callbacks:
                    try:
                        callback(request_id, approver_id, feedback)
                    except Exception as e:
                        logger.error(f"Rejection callback error: {e}")
            
            logger.info(f"Decision {outcome.value} for request {request_id} by {approver_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing decision {request_id}: {e}")
            return False

    def provide_feedback(self, request_id: str, feedback_type: FeedbackType, 
                        rating: Optional[float] = None, comment: str = "", 
                        provided_by: str = "human") -> str:
        """Provide feedback on a decision request"""
        feedback_id = f"feedback_{uuid.uuid4()}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO feedback_records
                (id, decision_request_id, feedback_type, rating, comment, created_at, provided_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback_id,
                request_id,
                feedback_type.value,
                rating,
                comment,
                time.time(),
                provided_by
            ))
        
        # Trigger feedback callbacks
        for callback in self.feedback_callbacks:
            try:
                callback(request_id, feedback_type, rating, comment, provided_by)
            except Exception as e:
                logger.error(f"Feedback callback error: {e}")
        
        logger.info(f"Feedback provided for request {request_id}")
        return feedback_id

    def register_supervisor(self, profile: SupervisorProfile) -> str:
        """Register a human supervisor"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO supervisor_profiles
                (id, name, expertise_areas, permissions, availability, 
                 current_assignments, total_decisions, approval_rate, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.id,
                profile.name,
                json.dumps(profile.expertise_areas),
                json.dumps([p.value for p in profile.permissions]),
                profile.availability,
                profile.current_assignments,
                profile.total_decisions,
                profile.approval_rate,
                json.dumps(profile.metadata)
            ))
        
        self.supervisors[profile.id] = profile
        logger.info(f"Registered supervisor: {profile.name} ({profile.id})")
        return profile.id

    def get_supervisor_stats(self, supervisor_id: str) -> Dict[str, Any]:
        """Get statistics for a supervisor"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT current_assignments, total_decisions, approval_rate
                FROM supervisor_profiles 
                WHERE id = ?
            """, (supervisor_id,))
            row = cursor.fetchone()
            
            if row:
                # Count pending requests assigned to this supervisor
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM decision_requests 
                    WHERE assigned_supervisor = ? AND status = 'pending'
                """, (supervisor_id,))
                pending_count = cursor.fetchone()[0]
                
                return {
                    "current_assignments": row[0],
                    "total_decisions": row[1],
                    "approval_rate": row[2],
                    "pending_requests": pending_count
                }
        
        return {}

    def get_decision_analytics(self) -> Dict[str, Any]:
        """Get analytics about decision patterns"""
        with sqlite3.connect(self.db_path) as conn:
            # Count decisions by outcome
            cursor = conn.execute("""
                SELECT outcome, COUNT(*) 
                FROM decision_responses 
                GROUP BY outcome
            """)
            outcome_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Count requests by supervision level
            cursor = conn.execute("""
                SELECT requested_supervision, COUNT(*) 
                FROM decision_requests 
                GROUP BY requested_supervision
            """)
            supervision_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Average response time
            cursor = conn.execute("""
                SELECT AVG(timestamp - (SELECT created_at FROM decision_requests dr WHERE dr.id = dr.id))
                FROM decision_responses dr
                JOIN decision_requests r ON dr.request_id = r.id
            """)
            avg_response_time = cursor.fetchone()[0] or 0.0
            
            # Feedback distribution
            cursor = conn.execute("""
                SELECT feedback_type, COUNT(*) 
                FROM feedback_records 
                GROUP BY feedback_type
            """)
            feedback_counts = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "outcome_distribution": outcome_counts,
            "supervision_distribution": supervision_counts,
            "average_response_time": avg_response_time,
            "feedback_distribution": feedback_counts,
            "total_decisions": sum(outcome_counts.values()),
            "total_requests": sum(supervision_counts.values())
        }


class InteractiveSupervisor:
    """Interactive interface for human supervisors"""

    def __init__(self, decision_system: DecisionApprovalSystem):
        self.decision_system = decision_system
        self.current_supervisor_id: Optional[str] = None

    def authenticate_supervisor(self, supervisor_id: str) -> bool:
        """Authenticate a supervisor"""
        with sqlite3.connect(self.decision_system.db_path) as conn:
            cursor = conn.execute(
                "SELECT id FROM supervisor_profiles WHERE id = ?", (supervisor_id,)
            )
            if cursor.fetchone():
                self.current_supervisor_id = supervisor_id
                return True
        return False

    def display_decision_request(self, request: DecisionRequest) -> None:
        """Display a decision request in an interactive format"""
        console.print(Panel(
            f"[bold blue]Decision Request #{request.id}[/bold blue]\n"
            f"[bold]Type:[/bold] {request.request_type}\n"
            f"[bold]Description:[/bold] {request.description}\n"
            f"[bold]Requested By:[/bold] {request.requested_by}\n"
            f"[bold]Urgency:[/bold] {request.urgency.name}\n"
            f"[bold]Supervision Level:[/bold] {request.requested_supervision.value}\n"
            f"[bold]Created:[/bold] {datetime.fromtimestamp(request.created_at)}",
            title="Decision Request",
            border_style="blue"
        ))

        # Display context
        if request.context:
            console.print("\n[bold]Context:[/bold]")
            for key, value in request.context.items():
                console.print(f"  {key}: {value}")

    def get_decision_input(self) -> Tuple[DecisionOutcome, str]:
        """Get decision input from human supervisor"""
        console.print("\n[bold]Choose an action:[/bold]")
        console.print("1. Approve")
        console.print("2. Reject")
        console.print("3. Defer (need more time/information)")
        console.print("4. Escalate (to higher authority)")
        
        choice = Prompt.ask("Enter choice (1-4)", choices=["1", "2", "3", "4"], default="1")
        
        outcomes = {
            "1": DecisionOutcome.APPROVED,
            "2": DecisionOutcome.REJECTED,
            "3": DecisionOutcome.DEFERRED,
            "4": DecisionOutcome.ESCALATED
        }
        
        outcome = outcomes[choice]
        
        feedback = Prompt.ask("Enter feedback/comment (optional)", default="")
        
        return outcome, feedback

    async def process_pending_requests(self):
        """Process all pending requests interactively"""
        if not self.current_supervisor_id:
            console.print("[red]❌ Not authenticated as supervisor[/red]")
            return

        pending_requests = self.decision_system.get_pending_requests(self.current_supervisor_id)
        
        if not pending_requests:
            console.print("[green]✅ No pending requests to review[/green]")
            return

        console.print(f"[blue]📋 Found {len(pending_requests)} pending requests[/blue]")
        
        for request in pending_requests:
            self.display_decision_request(request)
            
            outcome, feedback = self.get_decision_input()
            
            # Process the decision
            success = False
            if outcome == DecisionOutcome.APPROVED:
                success = self.decision_system.approve_request(request.id, self.current_supervisor_id, feedback)
            elif outcome == DecisionOutcome.REJECTED:
                success = self.decision_system.reject_request(request.id, self.current_supervisor_id, feedback)
            elif outcome == DecisionOutcome.DEFERRED:
                success = self.decision_system.defer_request(request.id, self.current_supervisor_id, feedback)
            elif outcome == DecisionOutcome.ESCALATED:
                success = self.decision_system.escalate_request(request.id, self.current_supervisor_id, feedback)
            
            if success:
                console.print(f"[green]✅ Decision recorded for request {request.id}[/green]")
            else:
                console.print(f"[red]❌ Failed to record decision for request {request.id}[/red]")
            
            # Ask if user wants to continue
            if len(pending_requests) > 1:
                continue_review = Confirm.ask("Process next request?", default=True)
                if not continue_review:
                    break

    def display_supervisor_dashboard(self):
        """Display supervisor dashboard"""
        if not self.current_supervisor_id:
            console.print("[red]❌ Not authenticated as supervisor[/red]")
            return

        stats = self.decision_system.get_supervisor_stats(self.current_supervisor_id)
        
        console.print(Panel(
            f"[bold blue]Supervisor Dashboard[/bold blue]\n"
            f"Supervisor: {self.current_supervisor_id}\n"
            f"Current Assignments: {stats.get('current_assignments', 0)}\n"
            f"Total Decisions: {stats.get('total_decisions', 0)}\n"
            f"Approval Rate: {stats.get('approval_rate', 0):.1%}\n"
            f"Pending Requests: {stats.get('pending_requests', 0)}",
            title="Supervisor Status",
            border_style="blue"
        ))

        # Show pending requests
        pending = self.decision_system.get_pending_requests(self.current_supervisor_id)
        if pending:
            table = Table(title="Pending Requests")
            table.add_column("ID", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Urgency", style="red")
            table.add_column("Description", style="green")

            for req in pending[:10]:  # Show first 10
                table.add_row(
                    req.id,
                    req.request_type,
                    req.urgency.name,
                    req.description[:50] + "..." if len(req.description) > 50 else req.description
                )

            console.print(table)
        else:
            console.print("[green]✅ No pending requests[/green]")


class SupervisionPolicyEngine:
    """Engine for determining supervision requirements"""

    def __init__(self, decision_system: DecisionApprovalSystem):
        self.decision_system = decision_system
        self.policies: List[Dict[str, Any]] = []
        self._load_default_policies()

    def _load_default_policies(self):
        """Load default supervision policies"""
        self.policies = [
            {
                "name": "code_generation_policy",
                "description": "Requires approval for code generation in critical systems",
                "conditions": {
                    "action_type": "code_generation",
                    "target_system": ["production", "core", "security"]
                },
                "required_level": SupervisionLevel.APPROVAL,
                "urgency": UrgencyLevel.HIGH
            },
            {
                "name": "data_access_policy",
                "description": "Requires review for sensitive data access",
                "conditions": {
                    "action_type": "data_access",
                    "data_sensitivity": ["high", "critical"]
                },
                "required_level": SupervisionLevel.REVIEW,
                "urgency": UrgencyLevel.MEDIUM
            },
            {
                "name": "system_change_policy",
                "description": "Requires approval for system configuration changes",
                "conditions": {
                    "action_type": "system_change",
                    "change_scope": ["global", "critical"]
                },
                "required_level": SupervisionLevel.APPROVAL,
                "urgency": UrgencyLevel.HIGH
            }
        ]

    def evaluate_supervision_needed(self, action_context: Dict[str, Any]) -> Tuple[SupervisionLevel, UrgencyLevel, str]:
        """Evaluate what level of supervision is needed for an action"""
        for policy in self.policies:
            if self._policy_matches(policy, action_context):
                return (
                    SupervisionLevel(policy["required_level"]),
                    UrgencyLevel(policy["urgency"]),
                    policy["name"]
                )
        
        # Default: automatic for most actions
        return SupervisionLevel.AUTOMATIC, UrgencyLevel.LOW, "default_policy"

    def _policy_matches(self, policy: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if a policy applies to the given context"""
        conditions = policy["conditions"]
        
        for key, expected_values in conditions.items():
            actual_value = context.get(key)
            
            if isinstance(expected_values, list):
                if actual_value not in expected_values:
                    return False
            else:
                if actual_value != expected_values:
                    return False
        
        return True

    def submit_for_supervision(self, action_type: str, description: str, 
                              context: Dict[str, Any], requested_by: str) -> Optional[str]:
        """Submit an action for supervision based on policies"""
        level, urgency, policy_name = self.evaluate_supervision_needed(context)
        
        if level == SupervisionLevel.AUTOMATIC:
            # No supervision needed
            return None
        
        request = DecisionRequest(
            id=f"req_{uuid.uuid4()}",
            request_type=action_type,
            description=description,
            context=context,
            requested_supervision=level,
            urgency=urgency,
            created_at=time.time(),
            requested_by=requested_by,
            metadata={"policy_applied": policy_name}
        )
        
        return self.decision_system.submit_decision_request(request)


class SupervisionDashboard:
    """Management dashboard for supervision system"""

    def __init__(self, decision_system: DecisionApprovalSystem, 
                 policy_engine: SupervisionPolicyEngine):
        self.decision_system = decision_system
        self.policy_engine = policy_engine

    def display_system_dashboard(self):
        """Display overall supervision system dashboard"""
        analytics = self.decision_system.get_decision_analytics()
        
        console.print(Panel(
            f"[bold blue]Supervision System Dashboard[/bold blue]\n"
            f"Total Decisions: {analytics['total_decisions']}\n"
            f"Total Requests: {analytics['total_requests']}\n"
            f"Average Response Time: {analytics['average_response_time']:.2f}s\n"
            f"Active Supervisors: {len(self.decision_system.supervisors)}",
            title="System Overview",
            border_style="blue"
        ))

        # Outcome distribution
        if analytics['outcome_distribution']:
            table = Table(title="Decision Outcomes")
            table.add_column("Outcome", style="cyan")
            table.add_column("Count", style="magenta")
            table.add_column("Percentage", style="green")

            total = analytics['total_decisions']
            for outcome, count in analytics['outcome_distribution'].items():
                percentage = (count / total * 100) if total > 0 else 0
                table.add_row(outcome, str(count), f"{percentage:.1f}%")

            console.print(table)

        # Supervision level distribution
        if analytics['supervision_distribution']:
            table = Table(title="Supervision Levels")
            table.add_column("Level", style="cyan")
            table.add_column("Count", style="magenta")
            table.add_column("Percentage", style="green")

            total = analytics['total_requests']
            for level, count in analytics['supervision_distribution'].items():
                percentage = (count / total * 100) if total > 0 else 0
                table.add_row(level, str(count), f"{percentage:.1f}%")

            console.print(table)

    def display_policy_compliance(self):
        """Display policy compliance information"""
        console.print("\n[bold]Active Policies:[/bold]")
        for policy in self.policy_engine.policies:
            console.print(f"• {policy['name']}: {policy['description']}")


async def demo_supervision_system():
    """Demonstrate the human-in-the-loop supervision system"""
    console.print("[bold green]👨‍💼 Initializing Human-in-the-Loop Supervision System[/bold green]")
    
    # Initialize components
    decision_system = DecisionApprovalSystem()
    policy_engine = SupervisionPolicyEngine(decision_system)
    supervisor_interface = InteractiveSupervisor(decision_system)
    dashboard = SupervisionDashboard(decision_system, policy_engine)
    
    # Register a sample supervisor
    supervisor = SupervisorProfile(
        id="supervisor_001",
        name="AI Oversight Manager",
        expertise_areas=["code_review", "security", "compliance"],
        permissions=[SupervisionLevel.REVIEW, SupervisionLevel.APPROVAL],
        availability=True
    )
    decision_system.register_supervisor(supervisor)
    
    # Simulate some decision requests
    console.print("[blue]📋 Submitting sample decision requests...[/blue]")
    
    # Request 1: Code generation in production
    request1 = DecisionRequest(
        id="req_001",
        request_type="code_generation",
        description="Generate authentication module for production system",
        context={
            "action_type": "code_generation",
            "target_system": "production",
            "code_type": "authentication",
            "estimated_lines": 200
        },
        requested_supervision=SupervisionLevel.APPROVAL,
        urgency=UrgencyLevel.HIGH,
        created_at=time.time(),
        requested_by="ai_assistant_001"
    )
    decision_system.submit_decision_request(request1)
    
    # Request 2: Data access request
    request2 = DecisionRequest(
        id="req_002",
        request_type="data_access",
        description="Access customer payment information for analysis",
        context={
            "action_type": "data_access",
            "data_sensitivity": "high",
            "data_type": "payment_info",
            "purpose": "fraud_analysis"
        },
        requested_supervision=SupervisionLevel.REVIEW,
        urgency=UrgencyLevel.MEDIUM,
        created_at=time.time(),
        requested_by="data_analyst_001"
    )
    decision_system.submit_decision_request(request2)
    
    # Display dashboard
    console.print("\n[bold]📊 Supervision System Dashboard:[/bold]")
    dashboard.display_system_dashboard()
    
    # Authenticate supervisor
    supervisor_interface.authenticate_supervisor("supervisor_001")
    
    # Display supervisor dashboard
    console.print("\n[bold]👤 Supervisor Dashboard:[/bold]")
    supervisor_interface.display_supervisor_dashboard()
    
    # Note: In a real system, we would have interactive decision making
    # For this demo, we'll just show what would happen
    
    console.print("\n[green]✅ Supervision System Demo Completed[/green]")
    console.print("[yellow]💡 In a real implementation, supervisors would interactively review and approve requests[/yellow]")


if __name__ == "__main__":
    # Don't run by default to avoid external dependencies
    # asyncio.run(demo_supervision_system())
    pass