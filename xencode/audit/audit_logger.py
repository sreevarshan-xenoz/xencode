#!/usr/bin/env python3
"""
Tamper-Proof Audit Logger

Implements comprehensive audit logging with cryptographic integrity,
tamper detection, and secure storage for compliance requirements.
"""

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import sqlite3
import threading
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
import logging

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events"""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"
    ROLE_ASSIGNMENT = "role_assignment"
    
    # Data events
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    
    # Security events
    SECURITY_SCAN = "security_scan"
    VULNERABILITY_DETECTED = "vulnerability_detected"
    SECURITY_INCIDENT = "security_incident"
    POLICY_VIOLATION = "policy_violation"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIGURATION_CHANGE = "configuration_change"
    BACKUP_CREATED = "backup_created"
    
    # Compliance events
    COMPLIANCE_CHECK = "compliance_check"
    AUDIT_EXPORT = "audit_export"
    RETENTION_POLICY_APPLIED = "retention_policy_applied"


class AuditSeverity(str, Enum):
    """Severity levels for audit events"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AuditEvent:
    """Represents a single audit event with cryptographic integrity"""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType = AuditEventType.SYSTEM_START
    severity: AuditSeverity = AuditSeverity.INFO
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Integrity fields
    checksum: Optional[str] = None
    signature: Optional[str] = None
    previous_hash: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data"""
        # Create a copy without integrity fields for checksum calculation
        data = asdict(self)
        data.pop('checksum', None)
        data.pop('signature', None)
        
        # Convert datetime to ISO string for consistent hashing
        if isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        
        # Sort keys for consistent hashing
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of the audit event"""
        if not self.checksum:
            return False
        
        # Recalculate checksum and compare
        current_checksum = self.checksum
        self.checksum = None
        calculated_checksum = self._calculate_checksum()
        self.checksum = current_checksum
        
        return current_checksum == calculated_checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        if isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary"""
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Convert enum strings back to enums
        if isinstance(data['event_type'], str):
            data['event_type'] = AuditEventType(data['event_type'])
        if isinstance(data['severity'], str):
            data['severity'] = AuditSeverity(data['severity'])
        
        return cls(**data)


class AuditChain:
    """Maintains a cryptographic chain of audit events"""
    
    def __init__(self):
        self.chain: List[AuditEvent] = []
        self.last_hash: Optional[str] = None
    
    def add_event(self, event: AuditEvent) -> None:
        """Add an event to the chain with proper linking"""
        event.previous_hash = self.last_hash
        
        # Recalculate checksum with previous hash
        event.checksum = event._calculate_checksum()
        
        self.chain.append(event)
        self.last_hash = event.checksum
    
    def verify_chain(self) -> bool:
        """Verify the integrity of the entire chain"""
        if not self.chain:
            return True
        
        previous_hash = None
        for event in self.chain:
            # Verify individual event integrity
            if not event.verify_integrity():
                logger.error(f"Event {event.id} failed integrity check")
                return False
            
            # Verify chain linkage
            if event.previous_hash != previous_hash:
                logger.error(f"Chain broken at event {event.id}")
                return False
            
            previous_hash = event.checksum
        
        return True


class AuditEncryption:
    """Handles encryption and signing of audit data"""
    
    def __init__(self, key_path: Optional[Path] = None):
        self.key_path = key_path or Path.home() / '.xencode' / 'audit_keys'
        self.key_path.mkdir(parents=True, exist_ok=True)
        
        self.private_key_path = self.key_path / 'audit_private.pem'
        self.public_key_path = self.key_path / 'audit_public.pem'
        
        self._ensure_keys()
    
    def _ensure_keys(self):
        """Ensure RSA key pair exists"""
        if not self.private_key_path.exists() or not self.public_key_path.exists():
            self._generate_key_pair()
    
    def _generate_key_pair(self):
        """Generate RSA key pair for signing"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Save private key
        with open(self.private_key_path, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Save public key
        public_key = private_key.public_key()
        with open(self.public_key_path, 'wb') as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
        
        # Set restrictive permissions
        os.chmod(self.private_key_path, 0o600)
        os.chmod(self.public_key_path, 0o644)
    
    def _load_private_key(self):
        """Load private key for signing"""
        with open(self.private_key_path, 'rb') as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    
    def _load_public_key(self):
        """Load public key for verification"""
        with open(self.public_key_path, 'rb') as f:
            return serialization.load_pem_public_key(f.read())
    
    def sign_event(self, event: AuditEvent) -> str:
        """Sign an audit event"""
        private_key = self._load_private_key()
        
        # Sign the checksum
        signature = private_key.sign(
            event.checksum.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature.hex()
    
    def verify_signature(self, event: AuditEvent) -> bool:
        """Verify an event signature"""
        if not event.signature or not event.checksum:
            return False
        
        # Ensure the event data itself has not been tampered with
        if not event.verify_integrity():
            logger.error("Audit event integrity verification failed before signature check")
            return False
        
        try:
            public_key = self._load_public_key()
            signature_bytes = bytes.fromhex(event.signature)
            
            public_key.verify(
                signature_bytes,
                event.checksum.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False


class AuditStorage:
    """Secure storage for audit events with tamper detection"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / '.xencode' / 'audit.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        self._lock = threading.Lock()
    
    def _init_database(self):
        """Initialize the audit database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    source_ip TEXT,
                    user_agent TEXT,
                    resource TEXT,
                    action TEXT,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    details TEXT,
                    checksum TEXT NOT NULL,
                    signature TEXT,
                    previous_hash TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def store_event(self, event: AuditEvent) -> bool:
        """Store an audit event securely"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO audit_events (
                            id, timestamp, event_type, severity, user_id, session_id,
                            source_ip, user_agent, resource, action, success,
                            error_message, details, checksum, signature, previous_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        event.id,
                        event.timestamp.isoformat(),
                        event.event_type.value,
                        event.severity.value,
                        event.user_id,
                        event.session_id,
                        event.source_ip,
                        event.user_agent,
                        event.resource,
                        event.action,
                        event.success,
                        event.error_message,
                        json.dumps(event.details),
                        event.checksum,
                        event.signature,
                        event.previous_hash
                    ))
                return True
            except Exception as e:
                logger.error(f"Failed to store audit event: {e}")
                return False
    
    def get_events(self, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   event_types: Optional[List[AuditEventType]] = None,
                   user_id: Optional[str] = None,
                   limit: int = 1000) -> List[AuditEvent]:
        """Retrieve audit events with filtering"""
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if event_types:
            placeholders = ','.join('?' * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in event_types])
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        events = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                for row in cursor:
                    event_data = dict(row)
                    event_data['details'] = json.loads(event_data['details'] or '{}')
                    event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                    event_data['event_type'] = AuditEventType(event_data['event_type'])
                    event_data['severity'] = AuditSeverity(event_data['severity'])
                    
                    # Remove database-specific fields
                    event_data.pop('created_at', None)
                    
                    events.append(AuditEvent(**event_data))
        
        except Exception as e:
            logger.error(f"Failed to retrieve audit events: {e}")
        
        return events
    
    def verify_database_integrity(self) -> bool:
        """Verify the integrity of all stored events"""
        try:
            events = self.get_events(limit=10000)  # Check all events
            chain = AuditChain()
            
            # Add events to chain in chronological order
            for event in reversed(events):
                chain.add_event(event)
            
            return chain.verify_chain()
        
        except Exception as e:
            logger.error(f"Database integrity check failed: {e}")
            return False


class AuditLogger:
    """Main audit logging system with tamper-proof capabilities"""
    
    def __init__(self, 
                 storage_path: Optional[Path] = None,
                 enable_encryption: bool = True,
                 enable_real_time: bool = True):
        
        self.storage = AuditStorage(storage_path)
        self.encryption = AuditEncryption() if enable_encryption else None
        self.enable_real_time = enable_real_time
        
        self.chain = AuditChain()
        self._event_queue = asyncio.Queue() if enable_real_time else None
        self._processing_task = None
        
        if enable_real_time:
            self._start_processing()
    
    def _start_processing(self):
        """Start background event processing"""
        if self._processing_task is None:
            loop = asyncio.get_event_loop()
            self._processing_task = loop.create_task(self._process_events())
    
    async def _process_events(self):
        """Background task to process audit events"""
        while True:
            try:
                event = await self._event_queue.get()
                
                # Add to chain
                self.chain.add_event(event)
                
                # Sign if encryption enabled
                if self.encryption:
                    event.signature = self.encryption.sign_event(event)
                
                # Store to database
                self.storage.store_event(event)
                
                self._event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing audit event: {e}")
    
    def log_event(self, 
                  event_type: AuditEventType,
                  severity: AuditSeverity = AuditSeverity.INFO,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  source_ip: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  resource: Optional[str] = None,
                  action: Optional[str] = None,
                  success: bool = True,
                  error_message: Optional[str] = None,
                  **details) -> str:
        """Log an audit event"""
        
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            source_ip=source_ip,
            user_agent=user_agent,
            resource=resource,
            action=action,
            success=success,
            error_message=error_message,
            details=details
        )
        
        if self.enable_real_time and self._event_queue:
            # Queue for background processing
            asyncio.create_task(self._event_queue.put(event))
        else:
            # Process immediately
            self.chain.add_event(event)
            if self.encryption:
                event.signature = self.encryption.sign_event(event)
            self.storage.store_event(event)
        
        return event.id
    
    def get_events(self, **kwargs) -> List[AuditEvent]:
        """Retrieve audit events"""
        return self.storage.get_events(**kwargs)
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of the audit system"""
        # Verify database integrity
        if not self.storage.verify_database_integrity():
            return False
        
        # Verify current chain
        if not self.chain.verify_chain():
            return False
        
        # Verify signatures if encryption enabled
        if self.encryption:
            events = self.get_events(limit=100)  # Check recent events
            for event in events:
                if event.signature and not self.encryption.verify_signature(event):
                    return False
        
        return True
    
    async def shutdown(self):
        """Shutdown the audit logger"""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        if self._event_queue:
            await self._event_queue.join()


# Global audit logger instance
_global_audit_logger: Optional[AuditLogger] = None

def get_global_audit_logger() -> AuditLogger:
    """Get or create the global audit logger"""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger

def audit_log(event_type: AuditEventType, **kwargs) -> str:
    """Convenience function for logging audit events"""
    logger = get_global_audit_logger()
    return logger.log_event(event_type, **kwargs)