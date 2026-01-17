"""
Security and governance system for multi-agent systems in Xencode
"""
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import json
import sqlite3
import threading
import hashlib
import hmac
import secrets
from pathlib import Path
from collections import defaultdict
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


class Permission(Enum):
    """Permissions for agents and users."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    COMMUNICATE = "communicate"
    ACCESS_RESOURCES = "access_resources"
    MODIFY_SETTINGS = "modify_settings"


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class AuditEventType(Enum):
    """Types of audit events."""
    AGENT_LOGIN = "agent_login"
    AGENT_LOGOUT = "agent_logout"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    RESOURCE_ACCESS = "resource_access"
    TASK_EXECUTION = "task_execution"
    CONFIG_CHANGE = "config_change"
    SECURITY_VIOLATION = "security_violation"
    AUTHENTICATION_ATTEMPT = "authentication_attempt"
    AUTHORIZATION_CHECK = "authorization_check"


class ComplianceStatus(Enum):
    """Compliance status for audit records."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    EXEMPT = "exempt"


@dataclass
class AgentIdentity:
    """Represents an agent identity in the system."""
    agent_id: str = ""
    agent_name: str = ""
    agent_type: str = ""
    public_key: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_authenticated: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessControlRule:
    """Rule for controlling access to resources."""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject: str = ""  # Agent ID or role
    resource: str = ""  # Resource identifier
    permission: Permission = Permission.READ
    effect: str = "allow"  # "allow" or "deny"
    conditions: Dict[str, Any] = field(default_factory=dict)  # Time, IP, etc.
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


@dataclass
class AuditRecord:
    """Record of a security or governance event."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.AUTHENTICATION_ATTEMPT
    agent_id: str = ""
    resource_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: str = ""
    user_agent: str = ""
    action_taken: str = ""
    result: str = ""  # success, failure, denied
    details: Dict[str, Any] = field(default_factory=dict)
    compliance_status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    rules: List[AccessControlRule] = field(default_factory=list)
    enforcement_level: SecurityLevel = SecurityLevel.INTERNAL
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class IdentityManager:
    """Manages agent identities and authentication."""
    
    def __init__(self, db_path: str = "identity.db"):
        self.db_path = db_path
        self.agents: Dict[str, AgentIdentity] = {}
        self.access_lock = threading.RLock()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize the identity database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create agents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                agent_name TEXT,
                agent_type TEXT,
                public_key TEXT,
                created_at TEXT,
                last_authenticated TEXT,
                is_active BOOLEAN,
                metadata TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_active ON agents(is_active)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_type ON agents(agent_type)')
        
        conn.commit()
        conn.close()
    
    def register_agent(self, agent_identity: AgentIdentity) -> bool:
        """Register a new agent identity."""
        with self.access_lock:
            # Validate agent identity
            if not agent_identity.agent_id or not agent_identity.public_key:
                return False
            
            # Store in memory
            self.agents[agent_identity.agent_id] = agent_identity
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO agents
                (agent_id, agent_name, agent_type, public_key, created_at, last_authenticated, is_active, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                agent_identity.agent_id,
                agent_identity.agent_name,
                agent_identity.agent_type,
                agent_identity.public_key,
                agent_identity.created_at.isoformat(),
                agent_identity.last_authenticated.isoformat() if agent_identity.last_authenticated else None,
                agent_identity.is_active,
                json.dumps(agent_identity.metadata)
            ))
            
            conn.commit()
            conn.close()
        
        return True
    
    def authenticate_agent(self, agent_id: str, token: str) -> bool:
        """Authenticate an agent using a token."""
        with self.access_lock:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            if not agent.is_active:
                return False
            
            # In a real system, this would verify the JWT token
            # For now, we'll just update the last authenticated time
            agent.last_authenticated = datetime.now()
            
            # Update in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE agents 
                SET last_authenticated = ? 
                WHERE agent_id = ?
            ''', (
                agent.last_authenticated.isoformat(),
                agent_id
            ))
            
            conn.commit()
            conn.close()
            
            return True
    
    def get_agent_identity(self, agent_id: str) -> Optional[AgentIdentity]:
        """Get agent identity by ID."""
        with self.access_lock:
            if agent_id in self.agents:
                return self.agents[agent_id]
        
        # Try to load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM agents WHERE agent_id = ?', (agent_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            agent = AgentIdentity(
                agent_id=row[0],
                agent_name=row[1],
                agent_type=row[2],
                public_key=row[3],
                created_at=datetime.fromisoformat(row[4]),
                last_authenticated=datetime.fromisoformat(row[5]) if row[5] else None,
                is_active=bool(row[6]),
                metadata=json.loads(row[7]) if row[7] else {}
            )
            
            # Cache in memory
            with self.access_lock:
                self.agents[agent_id] = agent
            
            return agent
        
        return None
    
    def deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate an agent."""
        with self.access_lock:
            if agent_id in self.agents:
                self.agents[agent_id].is_active = False
                
                # Update in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('UPDATE agents SET is_active = ? WHERE agent_id = ?', (False, agent_id))
                
                conn.commit()
                conn.close()
                
                return True
        
        return False


class AccessControlManager:
    """Manages access control for agents and resources."""
    
    def __init__(self, db_path: str = "access_control.db"):
        self.db_path = db_path
        self.rules: Dict[str, AccessControlRule] = {}
        self.access_lock = threading.RLock()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize the access control database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create access_control_rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_control_rules (
                rule_id TEXT PRIMARY KEY,
                subject TEXT,
                resource TEXT,
                permission TEXT,
                effect TEXT,
                conditions TEXT,
                created_by TEXT,
                created_at TEXT,
                expires_at TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_subject ON access_control_rules(subject)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_resource ON access_control_rules(resource)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_permission ON access_control_rules(permission)')
        
        conn.commit()
        conn.close()
    
    def add_rule(self, rule: AccessControlRule) -> str:
        """Add an access control rule."""
        with self.access_lock:
            # Store in memory
            self.rules[rule.rule_id] = rule
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO access_control_rules
                (rule_id, subject, resource, permission, effect, conditions, created_by, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id,
                rule.subject,
                rule.resource,
                rule.permission.value,
                rule.effect,
                json.dumps(rule.conditions),
                rule.created_by,
                rule.created_at.isoformat(),
                rule.expires_at.isoformat() if rule.expires_at else None
            ))
            
            conn.commit()
            conn.close()
        
        return rule.rule_id
    
    def check_permission(self, agent_id: str, resource: str, permission: Permission) -> bool:
        """Check if an agent has permission to access a resource."""
        with self.access_lock:
            # Check for explicit deny rules first
            for rule in self.rules.values():
                if (rule.subject == agent_id and 
                    rule.resource == resource and 
                    rule.permission == permission and 
                    rule.effect == "deny"):
                    
                    # Check if rule is expired
                    if rule.expires_at and rule.expires_at < datetime.now():
                        continue
                    
                    # Check conditions
                    if self._evaluate_conditions(rule.conditions):
                        return False
            
            # Check for explicit allow rules
            for rule in self.rules.values():
                if (rule.subject == agent_id and 
                    rule.resource == resource and 
                    rule.permission == permission and 
                    rule.effect == "allow"):
                    
                    # Check if rule is expired
                    if rule.expires_at and rule.expires_at < datetime.now():
                        continue
                    
                    # Check conditions
                    if self._evaluate_conditions(rule.conditions):
                        return True
            
            # Default deny
            return False
    
    def _evaluate_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Evaluate rule conditions."""
        if not conditions:
            return True
        
        # For now, just return True - in a real system, this would evaluate
        # time-based, IP-based, or other conditions
        return True
    
    def get_agent_permissions(self, agent_id: str) -> List[Tuple[str, Permission]]:
        """Get all permissions for an agent."""
        permissions = []
        
        with self.access_lock:
            for rule in self.rules.values():
                if rule.subject == agent_id and rule.effect == "allow":
                    # Check if rule is expired
                    if rule.expires_at and rule.expires_at < datetime.now():
                        continue
                    
                    # Check conditions
                    if self._evaluate_conditions(rule.conditions):
                        permissions.append((rule.resource, rule.permission))
        
        return permissions
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an access control rule."""
        with self.access_lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                
                # Remove from database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM access_control_rules WHERE rule_id = ?', (rule_id,))
                
                conn.commit()
                conn.close()
                
                return True
        
        return False


class AuditLogger:
    """Logs security and governance events."""
    
    def __init__(self, db_path: str = "audit.db"):
        self.db_path = db_path
        self.audit_records: List[AuditRecord] = []
        self.access_lock = threading.RLock()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize the audit database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create audit_records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_records (
                record_id TEXT PRIMARY KEY,
                event_type TEXT,
                agent_id TEXT,
                resource_id TEXT,
                timestamp TEXT,
                ip_address TEXT,
                user_agent TEXT,
                action_taken TEXT,
                result TEXT,
                details TEXT,
                compliance_status TEXT,
                metadata TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_records(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_id ON audit_records(agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_records(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_compliance_status ON audit_records(compliance_status)')
        
        conn.commit()
        conn.close()
    
    def log_event(self, event_type: AuditEventType, agent_id: str, resource_id: str = "",
                  action_taken: str = "", result: str = "success", ip_address: str = "",
                  user_agent: str = "", details: Dict[str, Any] = None,
                  compliance_status: ComplianceStatus = ComplianceStatus.COMPLIANT,
                  metadata: Dict[str, Any] = None) -> str:
        """Log an audit event."""
        details = details or {}
        metadata = metadata or {}
        
        record = AuditRecord(
            event_type=event_type,
            agent_id=agent_id,
            resource_id=resource_id,
            action_taken=action_taken,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            compliance_status=compliance_status,
            metadata=metadata
        )
        
        with self.access_lock:
            self.audit_records.append(record)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_records
                (record_id, event_type, agent_id, resource_id, timestamp, ip_address, user_agent, 
                 action_taken, result, details, compliance_status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.record_id,
                record.event_type.value,
                record.agent_id,
                record.resource_id,
                record.timestamp.isoformat(),
                record.ip_address,
                record.user_agent,
                record.action_taken,
                record.result,
                json.dumps(record.details),
                record.compliance_status.value,
                json.dumps(record.metadata)
            ))
            
            conn.commit()
            conn.close()
        
        return record.record_id
    
    def get_audit_records(self, agent_id: str = "", event_type: Optional[AuditEventType] = None,
                         start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                         limit: int = 100) -> List[AuditRecord]:
        """Get audit records with optional filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM audit_records WHERE 1=1"
        params = []
        
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            record = AuditRecord(
                record_id=row[0],
                event_type=AuditEventType(row[1]),
                agent_id=row[2],
                resource_id=row[3],
                timestamp=datetime.fromisoformat(row[4]),
                ip_address=row[5],
                user_agent=row[6],
                action_taken=row[7],
                result=row[8],
                details=json.loads(row[9]) if row[9] else {},
                compliance_status=ComplianceStatus(row[10]),
                metadata=json.loads(row[11]) if row[11] else {}
            )
            records.append(record)
        
        return records
    
    def get_compliance_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get a compliance report for a time period."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total records
        cursor.execute('SELECT COUNT(*) FROM audit_records WHERE timestamp BETWEEN ? AND ?', 
                      (start_time.isoformat(), end_time.isoformat()))
        total_records = cursor.fetchone()[0]
        
        # Get compliance status breakdown
        cursor.execute('''
            SELECT compliance_status, COUNT(*) 
            FROM audit_records 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY compliance_status
        ''', (start_time.isoformat(), end_time.isoformat()))
        status_counts = dict(cursor.fetchall())
        
        # Get security violations
        cursor.execute('''
            SELECT COUNT(*) 
            FROM audit_records 
            WHERE event_type = ? AND timestamp BETWEEN ? AND ?
        ''', (AuditEventType.SECURITY_VIOLATION.value, start_time.isoformat(), end_time.isoformat()))
        security_violations = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_records': total_records,
            'status_breakdown': status_counts,
            'security_violations': security_violations,
            'compliance_rate': (status_counts.get(ComplianceStatus.COMPLIANT.value, 0) / total_records) * 100 if total_records > 0 else 0,
            'report_period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
        }


class PrivacyPreservationManager:
    """Manages privacy preservation for sensitive data."""
    
    def __init__(self, db_path: str = "privacy.db"):
        self.db_path = db_path
        self.encryption_keys: Dict[str, Fernet] = {}
        self.access_lock = threading.RLock()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize the privacy database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create privacy_preservation table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS privacy_preservation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT,
                original_data_hash TEXT,
                transformed_data TEXT,
                transformation_type TEXT,
                timestamp TEXT,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_encryption_key(self, key_id: str) -> str:
        """Generate a new encryption key."""
        key = Fernet.generate_key()
        fernet = Fernet(key)
        self.encryption_keys[key_id] = fernet
        return key.decode()
    
    def encrypt_data(self, data: str, key_id: str = "default") -> str:
        """Encrypt sensitive data."""
        if key_id not in self.encryption_keys:
            # Generate a default key if it doesn't exist
            self.generate_encryption_key(key_id)
        
        fernet = self.encryption_keys[key_id]
        encrypted_data = fernet.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str, key_id: str = "default") -> str:
        """Decrypt sensitive data."""
        if key_id not in self.encryption_keys:
            raise ValueError(f"Encryption key '{key_id}' not found")
        
        fernet = self.encryption_keys[key_id]
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
        return decrypted_bytes.decode()
    
    def anonymize_data(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Anonymize sensitive fields in data."""
        anonymized = data.copy()
        
        for field in sensitive_fields:
            if field in anonymized:
                original_value = anonymized[field]
                
                # Store original hash for reference (without storing original data)
                original_hash = hashlib.sha256(str(original_value).encode()).hexdigest()
                
                # Replace with anonymized version
                if isinstance(original_value, str):
                    anonymized[field] = f"ANONYMIZED_{original_hash[:8]}"
                elif isinstance(original_value, (int, float)):
                    anonymized[field] = hash(str(original_value)) % 10000
                else:
                    anonymized[field] = f"ANONYMIZED_{type(original_value).__name__}"
        
        return anonymized
    
    def pseudonymize_data(self, data: Dict[str, Any], sensitive_fields: List[str], 
                         pseudonym_map: Dict[str, str] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Pseudonymize sensitive fields in data."""
        if pseudonym_map is None:
            pseudonym_map = {}
        
        pseudonymized = data.copy()
        
        for field in sensitive_fields:
            if field in pseudonymized:
                original_value = str(pseudonymized[field])
                
                # Create or reuse pseudonym
                if original_value not in pseudonym_map:
                    pseudonym = f"PSEUDO_{hashlib.md5(original_value.encode()).hexdigest()[:8]}"
                    pseudonym_map[original_value] = pseudonym
                
                pseudonymized[field] = pseudonym_map[original_value]
        
        return pseudonymized, pseudonym_map
    
    def apply_differential_privacy(self, data: List[Dict[str, Any]], epsilon: float = 1.0) -> List[Dict[str, Any]]:
        """Apply differential privacy to data."""
        # This is a simplified implementation
        # In a real system, this would use proper differential privacy mechanisms
        import random
        
        privatized_data = []
        for record in data:
            privatized_record = record.copy()
            
            # Add noise to numeric values
            for key, value in privatized_record.items():
                if isinstance(value, (int, float)):
                    # Add Laplace noise
                    b = 1.0 / epsilon  # Scale parameter for Laplace distribution
                    noise = random.uniform(-b, b)  # Simplified noise addition
                    privatized_record[key] = value + noise
            
            privatized_data.append(privatized_record)
        
        return privatized_data


class SecurityGovernanceSystem:
    """Main system for security and governance."""
    
    def __init__(self, db_path_prefix: str = "security"):
        self.identity_manager = IdentityManager(f"{db_path_prefix}_identity.db")
        self.access_control_manager = AccessControlManager(f"{db_path_prefix}_access_control.db")
        self.audit_logger = AuditLogger(f"{db_path_prefix}_audit.db")
        self.privacy_manager = PrivacyPreservationManager(f"{db_path_prefix}_privacy.db")
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.access_lock = threading.RLock()
    
    def register_agent_identity(self, agent_identity: AgentIdentity) -> bool:
        """Register a new agent identity."""
        success = self.identity_manager.register_agent(agent_identity)
        
        if success:
            # Log the event
            self.audit_logger.log_event(
                event_type=AuditEventType.AGENT_LOGIN,
                agent_id=agent_identity.agent_id,
                action_taken="Agent registration",
                result="success",
                details={"agent_name": agent_identity.agent_name, "agent_type": agent_identity.agent_type}
            )
        
        return success
    
    def authenticate_agent(self, agent_id: str, token: str) -> bool:
        """Authenticate an agent."""
        success = self.identity_manager.authenticate_agent(agent_id, token)
        
        # Log the authentication attempt
        self.audit_logger.log_event(
            event_type=AuditEventType.AUTHENTICATION_ATTEMPT,
            agent_id=agent_id,
            action_taken="Authentication attempt",
            result="success" if success else "failure",
            details={"token_valid": success}
        )
        
        return success
    
    def authorize_agent_action(self, agent_id: str, resource: str, permission: Permission) -> bool:
        """Authorize an agent action."""
        authorized = self.access_control_manager.check_permission(agent_id, resource, permission)
        
        # Log the authorization check
        self.audit_logger.log_event(
            event_type=AuditEventType.AUTHORIZATION_CHECK,
            agent_id=agent_id,
            resource_id=resource,
            action_taken=f"Check {permission.value} permission",
            result="authorized" if authorized else "denied",
            details={"permission": permission.value, "resource": resource}
        )
        
        return authorized
    
    def log_agent_action(self, agent_id: str, action: str, resource: str = "", 
                        result: str = "success", details: Dict[str, Any] = None) -> str:
        """Log an agent action for audit purposes."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.TASK_EXECUTION,
            agent_id=agent_id,
            resource_id=resource,
            action_taken=action,
            result=result,
            details=details or {}
        )
    
    def add_access_control_rule(self, rule: AccessControlRule) -> str:
        """Add an access control rule."""
        rule_id = self.access_control_manager.add_rule(rule)
        
        # Log the policy change
        self.audit_logger.log_event(
            event_type=AuditEventType.CONFIG_CHANGE,
            agent_id=rule.created_by,
            action_taken="Added access control rule",
            result="success",
            details={
                "rule_id": rule_id,
                "subject": rule.subject,
                "resource": rule.resource,
                "permission": rule.permission.value,
                "effect": rule.effect
            }
        )
        
        return rule_id
    
    def encrypt_sensitive_data(self, data: str, key_id: str = "default") -> str:
        """Encrypt sensitive data."""
        return self.privacy_manager.encrypt_data(data, key_id)
    
    def decrypt_sensitive_data(self, encrypted_data: str, key_id: str = "default") -> str:
        """Decrypt sensitive data."""
        return self.privacy_manager.decrypt_data(encrypted_data, key_id)
    
    def anonymize_sensitive_data(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Anonymize sensitive data."""
        return self.privacy_manager.anonymize_data(data, sensitive_fields)
    
    def get_agent_permissions(self, agent_id: str) -> List[Tuple[str, Permission]]:
        """Get all permissions for an agent."""
        return self.access_control_manager.get_agent_permissions(agent_id)
    
    def get_audit_records(self, agent_id: str = "", event_type: Optional[AuditEventType] = None,
                         start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                         limit: int = 100) -> List[AuditRecord]:
        """Get audit records."""
        return self.audit_logger.get_audit_records(agent_id, event_type, start_time, end_time, limit)
    
    def get_compliance_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get compliance report."""
        return self.audit_logger.get_compliance_report(start_time, end_time)
    
    def apply_privacy_preservation(self, data: List[Dict[str, Any]], 
                                  sensitive_fields: List[str],
                                  method: str = "anonymize") -> List[Dict[str, Any]]:
        """Apply privacy preservation to data."""
        if method == "anonymize":
            return [self.privacy_manager.anonymize_data(record, sensitive_fields) for record in data]
        elif method == "pseudonymize":
            result = []
            pseudonym_map = {}
            for record in data:
                anon_record, pseudonym_map = self.privacy_manager.pseudonymize_data(record, sensitive_fields, pseudonym_map)
                result.append(anon_record)
            return result
        elif method == "differential_privacy":
            return self.privacy_manager.apply_differential_privacy(data)
        else:
            raise ValueError(f"Unknown privacy method: {method}")


# Helper functions for common operations
def create_agent_identity(agent_id: str, agent_name: str, agent_type: str, public_key: str) -> AgentIdentity:
    """Create an agent identity."""
    return AgentIdentity(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_type=agent_type,
        public_key=public_key
    )


def create_access_control_rule(subject: str, resource: str, permission: Permission, 
                              effect: str = "allow", created_by: str = "system") -> AccessControlRule:
    """Create an access control rule."""
    return AccessControlRule(
        subject=subject,
        resource=resource,
        permission=permission,
        effect=effect,
        created_by=created_by
    )


def check_security_compliance(agent_id: str, required_permissions: List[Permission], 
                             security_system: SecurityGovernanceSystem) -> Dict[str, Any]:
    """Check if an agent meets security compliance requirements."""
    results = {}
    for perm in required_permissions:
        is_authorized = security_system.authorize_agent_action(agent_id, "system", perm)
        results[perm.value] = is_authorized
    
    all_compliant = all(results.values())
    
    return {
        'agent_id': agent_id,
        'permissions_checked': [p.value for p in required_permissions],
        'authorization_results': results,
        'overall_compliance': all_compliant,
        'missing_permissions': [perm for perm, authorized in results.items() if not authorized]
    }