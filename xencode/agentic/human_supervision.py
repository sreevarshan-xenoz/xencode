"""
Human-in-the-Loop supervision system for multi-agent systems in Xencode
"""
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import json
import sqlite3
import threading
from pathlib import Path
from collections import defaultdict


class SupervisionLevel(Enum):
    """Levels of human supervision required."""
    AUTONOMOUS = "autonomous"  # No human oversight needed
    OVERSIGHT = "oversight"    # Human notified, can intervene
    APPROVAL_REQUIRED = "approval_required"  # Human approval required
    HUMAN_ONLY = "human_only"  # Only human can handle


class DecisionCategory(Enum):
    """Categories of decisions requiring supervision."""
    CRITICAL = "critical"        # Business-critical decisions
    FINANCIAL = "financial"      # Financial implications
    SECURITY = "security"        # Security-related decisions
    ETHICAL = "ethical"          # Ethical considerations
    CUSTOMER_IMPACT = "customer_impact"  # Customer-facing impacts
    DATA_PRIVACY = "data_privacy"  # Privacy/data handling
    OPERATIONAL = "operational"  # Operational changes


class ApprovalStatus(Enum):
    """Status of approval requests."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class FeedbackType(Enum):
    """Types of feedback that can be provided."""
    CORRECTION = "correction"          # Correcting agent behavior
    VALIDATION = "validation"          # Validating agent output
    GUIDANCE = "guidance"              # Providing guidance
    EVALUATION = "evaluation"          # Evaluating performance
    SUGGESTION = "suggestion"          # Suggesting improvements


@dataclass
class SupervisionRequest:
    """Represents a request for human supervision."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    task_description: str = ""
    decision_category: DecisionCategory = DecisionCategory.OPERATIONAL
    supervision_level: SupervisionLevel = SupervisionLevel.OVERSIGHT
    request_timestamp: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    priority: int = 1  # Higher number = higher priority
    context: Dict[str, Any] = field(default_factory=dict)
    required_action: str = ""  # What human needs to do
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver_id: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanFeedback:
    """Represents feedback provided by humans."""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    supervisor_id: str = ""
    agent_id: str = ""
    task_id: str = ""
    feedback_type: FeedbackType = FeedbackType.VALIDATION
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    rating: Optional[int] = None  # 1-5 scale
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalRule:
    """Rule defining when human approval is required."""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    decision_categories: Set[DecisionCategory] = field(default_factory=set)
    supervision_level: SupervisionLevel = SupervisionLevel.OVERSIGHT
    conditions: Dict[str, Any] = field(default_factory=dict)  # Conditions that trigger this rule
    priority: int = 1  # Higher number = higher priority
    enabled: bool = True
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class SupervisionEngine:
    """Main engine for managing human-in-the-loop supervision."""
    
    def __init__(self, db_path: str = "supervision.db"):
        self.db_path = db_path
        self.approval_rules: List[ApprovalRule] = []
        self.pending_requests: Dict[str, SupervisionRequest] = {}
        self.feedback_records: List[HumanFeedback] = []
        self.access_lock = threading.RLock()
        
        # Initialize database
        self._init_db()
        
        # Initialize default rules
        self._init_default_rules()
    
    def _init_db(self):
        """Initialize the supervision database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create supervision_requests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS supervision_requests (
                request_id TEXT PRIMARY KEY,
                agent_id TEXT,
                task_description TEXT,
                decision_category TEXT,
                supervision_level TEXT,
                request_timestamp TEXT,
                due_date TEXT,
                priority INTEGER,
                context TEXT,
                required_action TEXT,
                status TEXT,
                approver_id TEXT,
                approval_timestamp TEXT,
                feedback TEXT,
                metadata TEXT
            )
        ''')
        
        # Create human_feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS human_feedback (
                feedback_id TEXT PRIMARY KEY,
                supervisor_id TEXT,
                agent_id TEXT,
                task_id TEXT,
                feedback_type TEXT,
                content TEXT,
                timestamp TEXT,
                rating INTEGER,
                metadata TEXT
            )
        ''')
        
        # Create approval_rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS approval_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                decision_categories TEXT,
                supervision_level TEXT,
                conditions TEXT,
                priority INTEGER,
                enabled BOOLEAN,
                created_by TEXT,
                created_at TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_request_status ON supervision_requests(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_request_agent ON supervision_requests(agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_request_category ON supervision_requests(decision_category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_request_priority ON supervision_requests(priority)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_agent ON human_feedback(agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_supervisor ON human_feedback(supervisor_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_type ON human_feedback(feedback_type)')
        
        conn.commit()
        conn.close()
    
    def _init_default_rules(self):
        """Initialize default approval rules."""
        default_rules = [
            ApprovalRule(
                name="Critical Decision Rule",
                description="Requires approval for critical business decisions",
                decision_categories={DecisionCategory.CRITICAL},
                supervision_level=SupervisionLevel.APPROVAL_REQUIRED,
                conditions={"impact_level": "high"},
                priority=5,
                created_by="system"
            ),
            ApprovalRule(
                name="Financial Decision Rule",
                description="Requires approval for financial decisions",
                decision_categories={DecisionCategory.FINANCIAL},
                supervision_level=SupervisionLevel.APPROVAL_REQUIRED,
                conditions={"amount_threshold": 1000},
                priority=4,
                created_by="system"
            ),
            ApprovalRule(
                name="Security Decision Rule",
                description="Requires oversight for security-related decisions",
                decision_categories={DecisionCategory.SECURITY},
                supervision_level=SupervisionLevel.OVERSIGHT,
                conditions={"security_impact": "medium_to_high"},
                priority=4,
                created_by="system"
            ),
            ApprovalRule(
                name="Data Privacy Rule",
                description="Requires approval for data privacy decisions",
                decision_categories={DecisionCategory.DATA_PRIVACY},
                supervision_level=SupervisionLevel.APPROVAL_REQUIRED,
                conditions={"contains_pii": True},
                priority=5,
                created_by="system"
            )
        ]
        
        for rule in default_rules:
            self.add_approval_rule(rule)
    
    def add_approval_rule(self, rule: ApprovalRule):
        """Add a new approval rule."""
        with self.access_lock:
            self.approval_rules.append(rule)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO approval_rules
                (rule_id, name, description, decision_categories, supervision_level, conditions, priority, enabled, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id,
                rule.name,
                rule.description,
                json.dumps([cat.value for cat in rule.decision_categories]),
                rule.supervision_level.value,
                json.dumps(rule.conditions),
                rule.priority,
                rule.enabled,
                rule.created_by,
                rule.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
    
    def evaluate_supervision_needed(self, agent_id: str, task_description: str, 
                                  decision_category: DecisionCategory, 
                                  context: Dict[str, Any] = None) -> SupervisionLevel:
        """Evaluate if human supervision is needed for a task."""
        context = context or {}
        
        # Check all rules to see if any apply
        applicable_rules = []
        for rule in self.approval_rules:
            if not rule.enabled:
                continue
                
            # Check if decision category matches
            if decision_category in rule.decision_categories:
                # Check conditions
                conditions_met = True
                for key, value in rule.conditions.items():
                    if key in context:
                        if isinstance(value, (int, float)) and isinstance(context[key], (int, float)):
                            # Numeric comparison
                            if context[key] >= value:
                                applicable_rules.append(rule)
                        elif context[key] == value:
                            applicable_rules.append(rule)
                    else:
                        # If condition key doesn't exist in context, check if it's a default that should trigger
                        if value is True and key.startswith("contains_"):
                            # Special case for PII detection
                            if key == "contains_pii":
                                # Check if context contains PII indicators
                                text_context = " ".join(str(v) for v in context.values() if isinstance(v, str))
                                pii_indicators = ["social security", "credit card", "ssn", "card number"]
                                if any(indicator in text_context.lower() for indicator in pii_indicators):
                                    applicable_rules.append(rule)
        
        # Return the highest priority supervision level from applicable rules
        if applicable_rules:
            highest_priority_rule = max(applicable_rules, key=lambda r: r.priority)
            return highest_priority_rule.supervision_level
        
        # Default: no supervision needed for routine tasks
        return SupervisionLevel.AUTONOMOUS
    
    def create_supervision_request(self, agent_id: str, task_description: str, 
                                 decision_category: DecisionCategory, 
                                 supervision_level: SupervisionLevel,
                                 context: Dict[str, Any] = None,
                                 required_action: str = "Review and approve",
                                 priority: int = 1,
                                 due_date: Optional[datetime] = None) -> str:
        """Create a request for human supervision."""
        context = context or {}
        
        request = SupervisionRequest(
            agent_id=agent_id,
            task_description=task_description,
            decision_category=decision_category,
            supervision_level=supervision_level,
            context=context,
            required_action=required_action,
            priority=priority,
            due_date=due_date
        )
        
        with self.access_lock:
            # Store in memory
            self.pending_requests[request.request_id] = request
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO supervision_requests
                (request_id, agent_id, task_description, decision_category, supervision_level,
                 request_timestamp, due_date, priority, context, required_action, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                request.request_id,
                request.agent_id,
                request.task_description,
                request.decision_category.value,
                request.supervision_level.value,
                request.request_timestamp.isoformat(),
                request.due_date.isoformat() if request.due_date else None,
                request.priority,
                json.dumps(request.context),
                request.required_action,
                request.status.value,
                json.dumps(request.metadata)
            ))
            
            conn.commit()
            conn.close()
        
        return request.request_id
    
    def get_pending_requests(self, supervisor_id: Optional[str] = None, 
                           decision_category: Optional[DecisionCategory] = None,
                           priority_threshold: int = 0) -> List[SupervisionRequest]:
        """Get pending supervision requests."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM supervision_requests WHERE status = ?"
        params = [ApprovalStatus.PENDING.value]
        
        if supervisor_id:
            # In a real system, we'd have logic to assign requests to supervisors
            pass
        
        if decision_category:
            query += " AND decision_category = ?"
            params.append(decision_category.value)
        
        query += " AND priority >= ? ORDER BY priority DESC, request_timestamp ASC"
        params.append(priority_threshold)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        requests = []
        for row in rows:
            request = SupervisionRequest(
                request_id=row[0],
                agent_id=row[1],
                task_description=row[2],
                decision_category=DecisionCategory(row[3]),
                supervision_level=SupervisionLevel(row[4]),
                request_timestamp=datetime.fromisoformat(row[5]),
                due_date=datetime.fromisoformat(row[6]) if row[6] else None,
                priority=row[7],
                context=json.loads(row[8]) if row[8] else {},
                required_action=row[9],
                status=ApprovalStatus(row[10]),
                approver_id=row[11],
                approval_timestamp=datetime.fromisoformat(row[12]) if row[12] else None,
                feedback=row[13],
                metadata=json.loads(row[14]) if row[14] else {}
            )
            requests.append(request)
        
        return requests
    
    def approve_request(self, request_id: str, approver_id: str, feedback: str = "") -> bool:
        """Approve a supervision request."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE supervision_requests 
            SET status = ?, approver_id = ?, approval_timestamp = ?, feedback = ?
            WHERE request_id = ?
        ''', (
            ApprovalStatus.APPROVED.value,
            approver_id,
            datetime.now().isoformat(),
            feedback,
            request_id
        ))
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        if rows_affected > 0:
            # Update in-memory cache
            if request_id in self.pending_requests:
                request = self.pending_requests[request_id]
                request.status = ApprovalStatus.APPROVED
                request.approver_id = approver_id
                request.approval_timestamp = datetime.now()
                request.feedback = feedback
                del self.pending_requests[request_id]  # Remove from pending
            
            return True
        
        return False
    
    def reject_request(self, request_id: str, approver_id: str, reason: str = "") -> bool:
        """Reject a supervision request."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE supervision_requests 
            SET status = ?, approver_id = ?, approval_timestamp = ?, feedback = ?
            WHERE request_id = ?
        ''', (
            ApprovalStatus.REJECTED.value,
            approver_id,
            datetime.now().isoformat(),
            reason,
            request_id
        ))
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        if rows_affected > 0:
            # Update in-memory cache
            if request_id in self.pending_requests:
                request = self.pending_requests[request_id]
                request.status = ApprovalStatus.REJECTED
                request.approver_id = approver_id
                request.approval_timestamp = datetime.now()
                request.feedback = reason
                del self.pending_requests[request_id]  # Remove from pending
            
            return True
        
        return False
    
    def submit_feedback(self, supervisor_id: str, agent_id: str, task_id: str,
                       feedback_type: FeedbackType, content: str,
                       rating: Optional[int] = None,
                       metadata: Dict[str, Any] = None) -> str:
        """Submit feedback from a human supervisor."""
        metadata = metadata or {}
        
        feedback = HumanFeedback(
            supervisor_id=supervisor_id,
            agent_id=agent_id,
            task_id=task_id,
            feedback_type=feedback_type,
            content=content,
            rating=rating,
            metadata=metadata
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO human_feedback
            (feedback_id, supervisor_id, agent_id, task_id, feedback_type, content, timestamp, rating, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.feedback_id,
            feedback.supervisor_id,
            feedback.agent_id,
            feedback.task_id,
            feedback.feedback_type.value,
            feedback.content,
            feedback.timestamp.isoformat(),
            feedback.rating,
            json.dumps(feedback.metadata)
        ))
        
        conn.commit()
        conn.close()
        
        # Store in memory
        self.feedback_records.append(feedback)
        
        return feedback.feedback_id
    
    def get_feedback_for_agent(self, agent_id: str, feedback_type: Optional[FeedbackType] = None,
                              limit: int = 50) -> List[HumanFeedback]:
        """Get feedback for a specific agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM human_feedback WHERE agent_id = ?"
        params = [agent_id]
        
        if feedback_type:
            query += " AND feedback_type = ?"
            params.append(feedback_type.value)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        feedback_list = []
        for row in rows:
            feedback = HumanFeedback(
                feedback_id=row[0],
                supervisor_id=row[1],
                agent_id=row[2],
                task_id=row[3],
                feedback_type=FeedbackType(row[4]),
                content=row[5],
                timestamp=datetime.fromisoformat(row[6]),
                rating=row[7],
                metadata=json.loads(row[8]) if row[8] else {}
            )
            feedback_list.append(feedback)
        
        return feedback_list
    
    def get_agent_performance_score(self, agent_id: str) -> float:
        """Calculate an agent's performance score based on human feedback."""
        feedback_list = self.get_feedback_for_agent(agent_id)
        
        if not feedback_list:
            return 0.5  # Neutral score if no feedback
        
        # Calculate weighted score based on feedback type and ratings
        total_score = 0.0
        total_weight = 0.0
        
        for feedback in feedback_list:
            weight = 1.0  # Base weight
            
            # Adjust weight based on feedback type
            if feedback.feedback_type == FeedbackType.CORRECTION:
                weight = 0.8  # Corrections have moderate impact
            elif feedback.feedback_type == FeedbackType.VALIDATION:
                weight = 1.0  # Validations have normal impact
            elif feedback.feedback_type == FeedbackType.EVALUATION:
                weight = 1.2  # Evaluations have higher impact
            elif feedback.feedback_type == FeedbackType.GUIDANCE:
                weight = 0.9  # Guidance has moderate impact
            elif feedback.feedback_type == FeedbackType.SUGGESTION:
                weight = 0.7  # Suggestions have lower impact
            
            # Use rating if available, otherwise infer from content
            score = 0.5  # Default neutral
            if feedback.rating is not None:
                score = feedback.rating / 5.0  # Normalize 1-5 rating to 0-1
            else:
                # Infer score from content sentiment (simplified)
                positive_indicators = ["good", "well", "excellent", "great", "perfect", "correct"]
                negative_indicators = ["bad", "poor", "incorrect", "wrong", "needs improvement", "error"]
                
                content_lower = feedback.content.lower()
                pos_count = sum(1 for indicator in positive_indicators if indicator in content_lower)
                neg_count = sum(1 for indicator in negative_indicators if indicator in content_lower)
                
                if pos_count > neg_count:
                    score = 0.7
                elif neg_count > pos_count:
                    score = 0.3
                else:
                    score = 0.5
            
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.5


class HumanSupervisionInterface:
    """Interface for human supervisors to interact with the system."""
    
    def __init__(self, supervision_engine: SupervisionEngine):
        self.supervision_engine = supervision_engine
        self.current_user_id = "default_supervisor"
        self.access_lock = threading.RLock()
    
    def set_current_user(self, user_id: str):
        """Set the current supervisor user ID."""
        with self.access_lock:
            self.current_user_id = user_id
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for the supervisor."""
        with self.access_lock:
            # Get pending requests
            pending_requests = self.supervision_engine.get_pending_requests(
                priority_threshold=1
            )
            
            # Get recent feedback
            recent_feedback = self.supervision_engine.get_feedback_for_agent(
                agent_id="any",  # This would be filtered by the actual agent in a real system
                limit=10
            )
            
            # Get statistics
            total_requests = len(pending_requests)
            high_priority_requests = sum(1 for req in pending_requests if req.priority >= 3)
            
            return {
                'pending_requests': len(pending_requests),
                'high_priority_requests': high_priority_requests,
                'recent_feedback_count': len(recent_feedback),
                'pending_requests_list': [
                    {
                        'request_id': req.request_id,
                        'agent_id': req.agent_id,
                        'task_description': req.task_description[:50] + "..." if len(req.task_description) > 50 else req.task_description,
                        'decision_category': req.decision_category.value,
                        'supervision_level': req.supervision_level.value,
                        'priority': req.priority,
                        'request_timestamp': req.request_timestamp.isoformat(),
                        'due_date': req.due_date.isoformat() if req.due_date else None
                    } for req in pending_requests
                ],
                'recent_feedback': [
                    {
                        'feedback_id': fb.feedback_id,
                        'supervisor_id': fb.supervisor_id,
                        'agent_id': fb.agent_id,
                        'feedback_type': fb.feedback_type.value,
                        'content_preview': fb.content[:50] + "..." if len(fb.content) > 50 else fb.content,
                        'timestamp': fb.timestamp.isoformat(),
                        'rating': fb.rating
                    } for fb in recent_feedback[:5]
                ]
            }
    
    def approve_task(self, request_id: str, feedback: str = "") -> bool:
        """Approve a task request."""
        return self.supervision_engine.approve_request(request_id, self.current_user_id, feedback)
    
    def reject_task(self, request_id: str, reason: str = "") -> bool:
        """Reject a task request."""
        return self.supervision_engine.reject_request(request_id, self.current_user_id, reason)
    
    def submit_agent_feedback(self, agent_id: str, task_id: str, feedback_type: FeedbackType,
                           content: str, rating: Optional[int] = None) -> str:
        """Submit feedback about an agent's performance."""
        return self.supervision_engine.submit_feedback(
            self.current_user_id, agent_id, task_id, feedback_type, content, rating
        )
    
    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific agent."""
        performance_score = self.supervision_engine.get_agent_performance_score(agent_id)
        
        feedback_list = self.supervision_engine.get_feedback_for_agent(agent_id)
        
        # Categorize feedback
        feedback_by_type = defaultdict(list)
        for fb in feedback_list:
            feedback_by_type[fb.feedback_type.value].append(fb)
        
        return {
            'agent_id': agent_id,
            'performance_score': performance_score,
            'total_feedback_count': len(feedback_list),
            'feedback_breakdown': {k: len(v) for k, v in feedback_by_type.items()},
            'recent_feedback': [
                {
                    'feedback_type': fb.feedback_type.value,
                    'content': fb.content[:100] + "..." if len(fb.content) > 100 else fb.content,
                    'rating': fb.rating,
                    'timestamp': fb.timestamp.isoformat()
                } for fb in feedback_list[:5]
            ]
        }


class FeedbackIntegrationSystem:
    """System for integrating human feedback into agent learning."""
    
    def __init__(self, supervision_engine: SupervisionEngine):
        self.supervision_engine = supervision_engine
        self.feedback_handlers: Dict[str, callable] = {}
        self.access_lock = threading.RLock()
    
    def register_feedback_handler(self, agent_type: str, handler_func: callable):
        """Register a function to handle feedback for a specific agent type."""
        with self.access_lock:
            self.feedback_handlers[agent_type] = handler_func
    
    def process_feedback_for_agent(self, agent_id: str, feedback: HumanFeedback):
        """Process feedback for a specific agent."""
        # Determine agent type from ID (simplified)
        agent_type = agent_id.split('_')[0] if '_' in agent_id else 'general'
        
        with self.access_lock:
            if agent_type in self.feedback_handlers:
                handler = self.feedback_handlers[agent_type]
                handler(agent_id, feedback)
    
    def get_feedback_summary(self, agent_id: str = None) -> Dict[str, Any]:
        """Get a summary of feedback across the system."""
        conn = sqlite3.connect(self.supervision_engine.db_path)
        cursor = conn.cursor()
        
        # Get feedback counts by type
        cursor.execute('SELECT feedback_type, COUNT(*) FROM human_feedback GROUP BY feedback_type')
        type_counts = dict(cursor.fetchall())
        
        # Get average ratings
        cursor.execute('SELECT AVG(rating) FROM human_feedback WHERE rating IS NOT NULL')
        avg_rating_row = cursor.fetchone()
        avg_rating = avg_rating_row[0] if avg_rating_row[0] is not None else None
        
        # Get feedback volume over time (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        cursor.execute('SELECT DATE(timestamp), COUNT(*) FROM human_feedback WHERE timestamp > ? GROUP BY DATE(timestamp)', (thirty_days_ago,))
        daily_counts = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_feedback': sum(type_counts.values()),
            'feedback_by_type': type_counts,
            'average_rating': avg_rating,
            'daily_feedback_trend': daily_counts,
            'recent_feedback_samples': self.supervision_engine.get_feedback_for_agent(agent_id or "any", limit=5)
        }


# Helper functions for common operations
def create_supervision_request_for_task(agent_id: str, task_description: str, 
                                      decision_category: DecisionCategory,
                                      context: Dict[str, Any] = None) -> SupervisionRequest:
    """Create a supervision request for a specific task."""
    engine = SupervisionEngine()
    supervision_level = engine.evaluate_supervision_needed(
        agent_id, task_description, decision_category, context
    )
    
    request_id = engine.create_supervision_request(
        agent_id, task_description, decision_category, supervision_level, context
    )
    
    # Retrieve the created request
    conn = sqlite3.connect(engine.db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM supervision_requests WHERE request_id = ?', (request_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return SupervisionRequest(
            request_id=row[0],
            agent_id=row[1],
            task_description=row[2],
            decision_category=DecisionCategory(row[3]),
            supervision_level=SupervisionLevel(row[4]),
            request_timestamp=datetime.fromisoformat(row[5]),
            due_date=datetime.fromisoformat(row[6]) if row[6] else None,
            priority=row[7],
            context=json.loads(row[8]) if row[8] else {},
            required_action=row[9],
            status=ApprovalStatus(row[10]),
            approver_id=row[11],
            approval_timestamp=datetime.fromisoformat(row[12]) if row[12] else None,
            feedback=row[13],
            metadata=json.loads(row[14]) if row[14] else {}
        )
    
    return None


def submit_human_feedback(supervisor_id: str, agent_id: str, task_id: str,
                        feedback_type: FeedbackType, content: str) -> str:
    """Submit human feedback for an agent."""
    engine = SupervisionEngine()
    return engine.submit_feedback(supervisor_id, agent_id, task_id, feedback_type, content)