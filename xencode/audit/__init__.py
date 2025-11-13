#!/usr/bin/env python3
"""
Comprehensive Audit System

Provides tamper-proof audit logging, security event correlation,
and compliance reporting for enterprise users.
"""

from .audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditChain,
    AuditEncryption,
    AuditStorage,
)
from .security_correlator import SecurityEventCorrelator, SecurityIncident
from .compliance_reporter import ComplianceReporter, ComplianceReport

__all__ = [
    'AuditLogger',
    'AuditEvent', 
    'AuditEventType',
    'AuditSeverity',
    'AuditChain',
    'AuditEncryption',
    'SecurityEventCorrelator',
    'SecurityIncident',
    'ComplianceReporter',
    'ComplianceReport',
    'AuditStorage',
]

# Global audit logger instance
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

def audit_event(event_type: AuditEventType, **kwargs):
    """Decorator for automatic audit logging"""
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            logger = get_audit_logger()
            try:
                result = func(*args, **func_kwargs)
                logger.log_event(event_type, success=True, **kwargs)
                return result
            except Exception as e:
                logger.log_event(event_type, success=False, error=str(e), **kwargs)
                raise
        return wrapper
    return decorator