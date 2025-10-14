#!/usr/bin/env python3
"""
Audit Logger

Provides comprehensive audit logging for security and compliance.
Tracks user actions, permission changes, and system events.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from xencode.models.user import AuditLogEntry, User, ResourceType


class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, log_file_path: Optional[str] = None):
        self.log_file_path = Path(log_file_path) if log_file_path else Path("audit.log")
        
        # In-memory log for recent entries
        self.memory_log: List[AuditLogEntry] = []
        self.max_memory_entries = 1000
        
        # Event categories to track
        self.tracked_events = {
            'authentication': ['login', 'logout', 'token_refresh', 'password_change'],
            'authorization': ['permission_check', 'permission_grant', 'permission_revoke'],
            'user_management': ['user_create', 'user_update', 'user_delete', 'user_lock', 'user_unlock'],
            'resource_access': ['file_read', 'file_write', 'file_delete', 'project_access'],
            'system': ['system_start', 'system_stop', 'configuration_change'],
            'security': ['failed_login', 'suspicious_activity', 'security_violation']
        }
        
        # Risk levels for different actions
        self.risk_levels = {
            'login': 'low',
            'logout': 'low',
            'token_refresh': 'low',
            'password_change': 'medium',
            'permission_grant': 'high',
            'permission_revoke': 'high',
            'user_create': 'medium',
            'user_delete': 'high',
            'user_lock': 'medium',
            'file_delete': 'medium',
            'configuration_change': 'high',
            'failed_login': 'medium',
            'security_violation': 'critical'
        }
    
    async def log_event(self,
                       user: Optional[User],
                       action: str,
                       resource_type: Optional[ResourceType] = None,
                       resource_id: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       success: bool = True,
                       error_message: Optional[str] = None) -> AuditLogEntry:
        """Log an audit event"""
        
        entry = AuditLogEntry(
            user_id=user.id if user else None,
            username=user.username if user else None,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
        
        # Add to memory log
        self.memory_log.append(entry)
        
        # Maintain memory log size
        if len(self.memory_log) > self.max_memory_entries:
            self.memory_log = self.memory_log[-self.max_memory_entries:]
        
        # Write to file
        await self._write_to_file(entry)
        
        # Check for suspicious patterns
        await self._check_suspicious_activity(entry)
        
        return entry
    
    async def log_authentication(self,
                                user: Optional[User],
                                action: str,
                                success: bool,
                                ip_address: Optional[str] = None,
                                user_agent: Optional[str] = None,
                                details: Optional[Dict[str, Any]] = None) -> AuditLogEntry:
        """Log authentication event"""
        
        return await self.log_event(
            user=user,
            action=action,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )
    
    async def log_authorization(self,
                               user: User,
                               action: str,
                               resource_type: ResourceType,
                               resource_id: Optional[str] = None,
                               success: bool = True,
                               details: Optional[Dict[str, Any]] = None) -> AuditLogEntry:
        """Log authorization event"""
        
        return await self.log_event(
            user=user,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            success=success
        )
    
    async def log_user_management(self,
                                 admin_user: User,
                                 action: str,
                                 target_user: Optional[User] = None,
                                 details: Optional[Dict[str, Any]] = None) -> AuditLogEntry:
        """Log user management event"""
        
        event_details = details or {}
        if target_user:
            event_details.update({
                'target_user_id': target_user.id,
                'target_username': target_user.username
            })
        
        return await self.log_event(
            user=admin_user,
            action=action,
            resource_type=ResourceType.USER,
            resource_id=target_user.id if target_user else None,
            details=event_details
        )
    
    async def log_resource_access(self,
                                 user: User,
                                 action: str,
                                 resource_type: ResourceType,
                                 resource_id: str,
                                 success: bool = True,
                                 details: Optional[Dict[str, Any]] = None) -> AuditLogEntry:
        """Log resource access event"""
        
        return await self.log_event(
            user=user,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            success=success
        )
    
    async def log_security_event(self,
                                user: Optional[User],
                                action: str,
                                severity: str = 'medium',
                                ip_address: Optional[str] = None,
                                details: Optional[Dict[str, Any]] = None) -> AuditLogEntry:
        """Log security event"""
        
        event_details = details or {}
        event_details['severity'] = severity
        
        return await self.log_event(
            user=user,
            action=action,
            details=event_details,
            ip_address=ip_address,
            success=False  # Security events are typically failures
        )
    
    async def _write_to_file(self, entry: AuditLogEntry) -> None:
        """Write audit entry to file"""
        
        try:
            # Ensure log directory exists
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write entry as JSON line
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                json_line = json.dumps(entry.to_dict())
                f.write(json_line + '\n')
                
        except Exception as e:
            # Log to memory only if file write fails
            print(f"Failed to write audit log to file: {e}")
    
    async def _check_suspicious_activity(self, entry: AuditLogEntry) -> None:
        """Check for suspicious activity patterns"""
        
        if not entry.user_id:
            return
        
        # Check for multiple failed logins
        if entry.action == 'login' and not entry.success:
            recent_failures = await self.get_recent_events(
                user_id=entry.user_id,
                action='login',
                success=False,
                hours=1
            )
            
            if len(recent_failures) >= 5:
                await self.log_security_event(
                    user=None,  # System event
                    action='suspicious_activity',
                    severity='high',
                    ip_address=entry.ip_address,
                    details={
                        'type': 'multiple_failed_logins',
                        'user_id': entry.user_id,
                        'failure_count': len(recent_failures)
                    }
                )
        
        # Check for unusual access patterns
        if entry.action in ['file_read', 'file_write', 'file_delete']:
            recent_access = await self.get_recent_events(
                user_id=entry.user_id,
                action=entry.action,
                hours=1
            )
            
            if len(recent_access) >= 50:  # High volume access
                await self.log_security_event(
                    user=None,
                    action='suspicious_activity',
                    severity='medium',
                    details={
                        'type': 'high_volume_access',
                        'user_id': entry.user_id,
                        'action': entry.action,
                        'count': len(recent_access)
                    }
                )
        
        # Check for privilege escalation attempts
        if entry.action == 'permission_check' and not entry.success:
            details = entry.details or {}
            if details.get('permission_type') in ['admin', 'delete']:
                await self.log_security_event(
                    user=None,
                    action='suspicious_activity',
                    severity='medium',
                    details={
                        'type': 'privilege_escalation_attempt',
                        'user_id': entry.user_id,
                        'attempted_permission': details.get('permission_type')
                    }
                )
    
    async def get_recent_events(self,
                               user_id: Optional[str] = None,
                               action: Optional[str] = None,
                               success: Optional[bool] = None,
                               hours: int = 24) -> List[AuditLogEntry]:
        """Get recent audit events"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        events = []
        for entry in reversed(self.memory_log):  # Most recent first
            if entry.timestamp < cutoff_time:
                break
            
            # Apply filters
            if user_id and entry.user_id != user_id:
                continue
            
            if action and entry.action != action:
                continue
            
            if success is not None and entry.success != success:
                continue
            
            events.append(entry)
        
        return events
    
    async def get_events_by_category(self, 
                                    category: str,
                                    hours: int = 24) -> List[AuditLogEntry]:
        """Get events by category"""
        
        if category not in self.tracked_events:
            return []
        
        actions = self.tracked_events[category]
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        events = []
        for entry in reversed(self.memory_log):
            if entry.timestamp < cutoff_time:
                break
            
            if entry.action in actions:
                events.append(entry)
        
        return events
    
    async def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for the specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        summary = {
            'total_events': 0,
            'failed_events': 0,
            'unique_users': set(),
            'unique_ips': set(),
            'event_counts': {},
            'risk_distribution': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
            'suspicious_activities': 0
        }
        
        for entry in self.memory_log:
            if entry.timestamp < cutoff_time:
                continue
            
            summary['total_events'] += 1
            
            if not entry.success:
                summary['failed_events'] += 1
            
            if entry.user_id:
                summary['unique_users'].add(entry.user_id)
            
            if entry.ip_address:
                summary['unique_ips'].add(entry.ip_address)
            
            # Count events by action
            action = entry.action
            summary['event_counts'][action] = summary['event_counts'].get(action, 0) + 1
            
            # Count by risk level
            risk_level = self.risk_levels.get(action, 'low')
            summary['risk_distribution'][risk_level] += 1
            
            # Count suspicious activities
            if action == 'suspicious_activity':
                summary['suspicious_activities'] += 1
        
        # Convert sets to counts
        summary['unique_users'] = len(summary['unique_users'])
        summary['unique_ips'] = len(summary['unique_ips'])
        
        return summary
    
    async def export_logs(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         format: str = 'json') -> str:
        """Export audit logs in specified format"""
        
        # Filter logs by time range
        logs = self.memory_log
        
        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]
        
        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]
        
        if format == 'json':
            return json.dumps([log.to_dict() for log in logs], indent=2)
        elif format == 'csv':
            # Simple CSV export
            lines = ['timestamp,user_id,username,action,resource_type,resource_id,success,ip_address']
            for log in logs:
                line = f"{log.timestamp.isoformat()},{log.user_id or ''},{log.username or ''},{log.action},{log.resource_type.value if log.resource_type else ''},{log.resource_id or ''},{log.success},{log.ip_address or ''}"
                lines.append(line)
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audit logger statistics"""
        
        return {
            'memory_log_entries': len(self.memory_log),
            'max_memory_entries': self.max_memory_entries,
            'log_file_path': str(self.log_file_path),
            'tracked_event_categories': list(self.tracked_events.keys()),
            'total_tracked_events': sum(len(events) for events in self.tracked_events.values())
        }
    
    async def cleanup_old_logs(self, days: int = 90) -> int:
        """Clean up old log entries from memory"""
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        original_count = len(self.memory_log)
        self.memory_log = [log for log in self.memory_log if log.timestamp >= cutoff_time]
        
        return original_count - len(self.memory_log)


# Global audit logger instance
audit_logger = AuditLogger()