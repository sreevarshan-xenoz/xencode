#!/usr/bin/env python3
"""
Security Event Correlator

Analyzes audit events to detect security incidents, patterns,
and potential threats through correlation and analysis.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import statistics

from .audit_logger import AuditEvent, AuditEventType, AuditSeverity

logger = logging.getLogger(__name__)


class IncidentType(str, Enum):
    """Types of security incidents"""
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    POLICY_VIOLATION = "policy_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    MULTIPLE_FAILURES = "multiple_failures"
    UNUSUAL_TIMING = "unusual_timing"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"


class IncidentSeverity(str, Enum):
    """Severity levels for security incidents"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SecurityIncident:
    """Represents a detected security incident"""
    
    id: str
    incident_type: IncidentType
    severity: IncidentSeverity
    title: str
    description: str
    detected_at: datetime
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Related events and entities
    related_events: List[str] = field(default_factory=list)
    affected_users: Set[str] = field(default_factory=set)
    affected_resources: Set[str] = field(default_factory=set)
    source_ips: Set[str] = field(default_factory=set)
    
    # Analysis data
    confidence_score: float = 0.0
    risk_score: float = 0.0
    indicators: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Status tracking
    status: str = "open"
    assigned_to: Optional[str] = None
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class CorrelationRule:
    """Base class for correlation rules"""
    
    def __init__(self, name: str, description: str, severity: IncidentSeverity):
        self.name = name
        self.description = description
        self.severity = severity
        self.enabled = True
    
    def analyze(self, events: List[AuditEvent], context: Dict[str, Any]) -> List[SecurityIncident]:
        """Analyze events and return detected incidents"""
        raise NotImplementedError


class BruteForceRule(CorrelationRule):
    """Detects brute force attacks"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 time_window: int = 300,  # 5 minutes
                 severity: IncidentSeverity = IncidentSeverity.HIGH):
        super().__init__(
            "Brute Force Detection",
            "Detects multiple failed login attempts from same source",
            severity
        )
        self.failure_threshold = failure_threshold
        self.time_window = time_window
    
    def analyze(self, events: List[AuditEvent], context: Dict[str, Any]) -> List[SecurityIncident]:
        incidents = []
        
        # Group failed login attempts by source IP
        failed_attempts = defaultdict(list)
        
        for event in events:
            if (event.event_type == AuditEventType.LOGIN_FAILURE and 
                not event.success and 
                event.source_ip):
                failed_attempts[event.source_ip].append(event)
        
        # Check each IP for brute force pattern
        for source_ip, attempts in failed_attempts.items():
            if len(attempts) >= self.failure_threshold:
                # Check if attempts are within time window
                attempts.sort(key=lambda x: x.timestamp)
                
                for i in range(len(attempts) - self.failure_threshold + 1):
                    window_start = attempts[i].timestamp
                    window_end = attempts[i + self.failure_threshold - 1].timestamp
                    
                    if (window_end - window_start).total_seconds() <= self.time_window:
                        # Brute force detected
                        incident = SecurityIncident(
                            id=f"bf_{int(time.time())}_{hash(source_ip) % 10000}",
                            incident_type=IncidentType.BRUTE_FORCE_ATTACK,
                            severity=self.severity,
                            title=f"Brute Force Attack from {source_ip}",
                            description=f"Detected {len(attempts)} failed login attempts from {source_ip} within {self.time_window} seconds",
                            detected_at=datetime.now(timezone.utc),
                            start_time=window_start,
                            end_time=window_end,
                            related_events=[e.id for e in attempts[i:i+self.failure_threshold]],
                            source_ips={source_ip},
                            affected_users={e.user_id for e in attempts if e.user_id},
                            confidence_score=min(1.0, len(attempts) / (self.failure_threshold * 2)),
                            risk_score=0.8,
                            indicators=[
                                f"{len(attempts)} failed login attempts",
                                f"Time window: {(window_end - window_start).total_seconds()} seconds",
                                f"Source IP: {source_ip}"
                            ],
                            recommendations=[
                                f"Block or rate-limit IP address {source_ip}",
                                "Review authentication logs for affected accounts",
                                "Consider implementing account lockout policies",
                                "Monitor for successful logins from this IP"
                            ]
                        )
                        incidents.append(incident)
                        break  # Only report one incident per IP
        
        return incidents


class PrivilegeEscalationRule(CorrelationRule):
    """Detects privilege escalation attempts"""
    
    def __init__(self, severity: IncidentSeverity = IncidentSeverity.CRITICAL):
        super().__init__(
            "Privilege Escalation Detection",
            "Detects attempts to gain elevated privileges",
            severity
        )
    
    def analyze(self, events: List[AuditEvent], context: Dict[str, Any]) -> List[SecurityIncident]:
        incidents = []
        
        # Look for role assignment events followed by high-privilege actions
        role_changes = []
        privilege_actions = []
        
        for event in events:
            if event.event_type == AuditEventType.ROLE_ASSIGNMENT:
                role_changes.append(event)
            elif (event.event_type in [AuditEventType.CONFIGURATION_CHANGE, 
                                     AuditEventType.DATA_DELETION,
                                     AuditEventType.SYSTEM_START] and
                  event.success):
                privilege_actions.append(event)
        
        # Check for suspicious patterns
        for role_event in role_changes:
            # Look for privilege actions within 1 hour of role change
            time_threshold = role_event.timestamp + timedelta(hours=1)
            
            suspicious_actions = [
                action for action in privilege_actions
                if (action.user_id == role_event.user_id and
                    role_event.timestamp <= action.timestamp <= time_threshold)
            ]
            
            if suspicious_actions:
                incident = SecurityIncident(
                    id=f"pe_{int(time.time())}_{hash(role_event.user_id or 'unknown') % 10000}",
                    incident_type=IncidentType.PRIVILEGE_ESCALATION,
                    severity=self.severity,
                    title=f"Potential Privilege Escalation by {role_event.user_id}",
                    description=f"User {role_event.user_id} performed {len(suspicious_actions)} privileged actions shortly after role assignment",
                    detected_at=datetime.now(timezone.utc),
                    start_time=role_event.timestamp,
                    end_time=max(action.timestamp for action in suspicious_actions),
                    related_events=[role_event.id] + [a.id for a in suspicious_actions],
                    affected_users={role_event.user_id} if role_event.user_id else set(),
                    source_ips={role_event.source_ip} if role_event.source_ip else set(),
                    confidence_score=min(1.0, len(suspicious_actions) / 3),
                    risk_score=0.9,
                    indicators=[
                        f"Role assignment at {role_event.timestamp}",
                        f"{len(suspicious_actions)} privileged actions within 1 hour",
                        f"Actions: {', '.join(a.action or a.event_type.value for a in suspicious_actions)}"
                    ],
                    recommendations=[
                        f"Review role assignment for user {role_event.user_id}",
                        "Verify legitimacy of privileged actions",
                        "Consider implementing approval workflows for role changes",
                        "Monitor user activity for additional suspicious behavior"
                    ]
                )
                incidents.append(incident)
        
        return incidents


class DataExfiltrationRule(CorrelationRule):
    """Detects potential data exfiltration"""
    
    def __init__(self, 
                 export_threshold: int = 3,
                 time_window: int = 3600,  # 1 hour
                 severity: IncidentSeverity = IncidentSeverity.HIGH):
        super().__init__(
            "Data Exfiltration Detection",
            "Detects unusual data export patterns",
            severity
        )
        self.export_threshold = export_threshold
        self.time_window = time_window
    
    def analyze(self, events: List[AuditEvent], context: Dict[str, Any]) -> List[SecurityIncident]:
        incidents = []
        
        # Group data export events by user
        exports_by_user = defaultdict(list)
        
        for event in events:
            if event.event_type == AuditEventType.DATA_EXPORT and event.success:
                exports_by_user[event.user_id].append(event)
        
        # Check for unusual export patterns
        for user_id, exports in exports_by_user.items():
            if len(exports) >= self.export_threshold:
                exports.sort(key=lambda x: x.timestamp)
                
                # Check for exports within time window
                for i in range(len(exports) - self.export_threshold + 1):
                    window_start = exports[i].timestamp
                    window_end = exports[i + self.export_threshold - 1].timestamp
                    
                    if (window_end - window_start).total_seconds() <= self.time_window:
                        # Potential exfiltration detected
                        incident = SecurityIncident(
                            id=f"de_{int(time.time())}_{hash(user_id or 'unknown') % 10000}",
                            incident_type=IncidentType.DATA_EXFILTRATION,
                            severity=self.severity,
                            title=f"Potential Data Exfiltration by {user_id}",
                            description=f"User {user_id} exported {len(exports)} datasets within {self.time_window/3600:.1f} hours",
                            detected_at=datetime.now(timezone.utc),
                            start_time=window_start,
                            end_time=window_end,
                            related_events=[e.id for e in exports[i:i+self.export_threshold]],
                            affected_users={user_id} if user_id else set(),
                            affected_resources={e.resource for e in exports if e.resource},
                            source_ips={e.source_ip for e in exports if e.source_ip},
                            confidence_score=min(1.0, len(exports) / (self.export_threshold * 2)),
                            risk_score=0.7,
                            indicators=[
                                f"{len(exports)} data exports",
                                f"Time window: {(window_end - window_start).total_seconds()/3600:.1f} hours",
                                f"Resources: {', '.join({e.resource for e in exports if e.resource})}"
                            ],
                            recommendations=[
                                f"Review data export activities for user {user_id}",
                                "Verify business justification for exports",
                                "Check if exported data contains sensitive information",
                                "Consider implementing data loss prevention (DLP) controls"
                            ]
                        )
                        incidents.append(incident)
                        break
        
        return incidents


class AnomalousActivityRule(CorrelationRule):
    """Detects anomalous user behavior patterns"""
    
    def __init__(self, severity: IncidentSeverity = IncidentSeverity.MEDIUM):
        super().__init__(
            "Anomalous Activity Detection",
            "Detects unusual user behavior patterns",
            severity
        )
    
    def analyze(self, events: List[AuditEvent], context: Dict[str, Any]) -> List[SecurityIncident]:
        incidents = []
        
        # Analyze user activity patterns
        user_activities = defaultdict(list)
        
        for event in events:
            if event.user_id:
                user_activities[event.user_id].append(event)
        
        # Check for anomalies
        for user_id, activities in user_activities.items():
            anomalies = self._detect_anomalies(user_id, activities, context)
            
            if anomalies:
                incident = SecurityIncident(
                    id=f"aa_{int(time.time())}_{hash(user_id) % 10000}",
                    incident_type=IncidentType.ANOMALOUS_BEHAVIOR,
                    severity=self.severity,
                    title=f"Anomalous Activity by {user_id}",
                    description=f"Detected {len(anomalies)} anomalous behaviors for user {user_id}",
                    detected_at=datetime.now(timezone.utc),
                    start_time=min(a['timestamp'] for a in anomalies),
                    end_time=max(a['timestamp'] for a in anomalies),
                    related_events=[a['event_id'] for a in anomalies],
                    affected_users={user_id},
                    confidence_score=min(1.0, len(anomalies) / 3),
                    risk_score=0.5,
                    indicators=[a['description'] for a in anomalies],
                    recommendations=[
                        f"Review recent activity for user {user_id}",
                        "Verify user identity and account security",
                        "Check for compromised credentials",
                        "Monitor user behavior for additional anomalies"
                    ]
                )
                incidents.append(incident)
        
        return incidents
    
    def _detect_anomalies(self, user_id: str, activities: List[AuditEvent], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in user activities"""
        anomalies = []
        
        if len(activities) < 5:  # Need sufficient data
            return anomalies
        
        # Check for unusual timing patterns
        timestamps = [a.timestamp.hour for a in activities]
        if timestamps:
            avg_hour = statistics.mean(timestamps)
            std_hour = statistics.stdev(timestamps) if len(timestamps) > 1 else 0
            
            for activity in activities:
                hour_diff = abs(activity.timestamp.hour - avg_hour)
                if std_hour > 0 and hour_diff > (2 * std_hour + 3):  # Unusual time
                    anomalies.append({
                        'event_id': activity.id,
                        'timestamp': activity.timestamp,
                        'description': f"Activity at unusual time: {activity.timestamp.hour}:00 (avg: {avg_hour:.1f})"
                    })
        
        # Check for unusual IP addresses
        ip_addresses = [a.source_ip for a in activities if a.source_ip]
        if ip_addresses:
            ip_counts = defaultdict(int)
            for ip in ip_addresses:
                ip_counts[ip] += 1
            
            most_common_ip = max(ip_counts, key=ip_counts.get)
            
            for activity in activities:
                if (activity.source_ip and 
                    activity.source_ip != most_common_ip and 
                    ip_counts[activity.source_ip] == 1):  # Single use IP
                    anomalies.append({
                        'event_id': activity.id,
                        'timestamp': activity.timestamp,
                        'description': f"Activity from unusual IP: {activity.source_ip}"
                    })
        
        # Check for unusual resource access
        resources = [a.resource for a in activities if a.resource]
        if resources:
            resource_counts = defaultdict(int)
            for resource in resources:
                resource_counts[resource] += 1
            
            for activity in activities:
                if (activity.resource and 
                    resource_counts[activity.resource] == 1 and
                    activity.event_type in [AuditEventType.DATA_ACCESS, AuditEventType.DATA_MODIFICATION]):
                    anomalies.append({
                        'event_id': activity.id,
                        'timestamp': activity.timestamp,
                        'description': f"First-time access to resource: {activity.resource}"
                    })
        
        return anomalies


class SecurityEventCorrelator:
    """Main security event correlation engine"""
    
    def __init__(self):
        self.rules: List[CorrelationRule] = []
        self.incidents: List[SecurityIncident] = []
        self.context: Dict[str, Any] = {}
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default correlation rules"""
        self.rules = [
            BruteForceRule(),
            PrivilegeEscalationRule(),
            DataExfiltrationRule(),
            AnomalousActivityRule()
        ]
    
    def add_rule(self, rule: CorrelationRule):
        """Add a custom correlation rule"""
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str):
        """Remove a correlation rule by name"""
        self.rules = [r for r in self.rules if r.name != rule_name]
    
    def analyze_events(self, events: List[AuditEvent]) -> List[SecurityIncident]:
        """Analyze events and detect security incidents"""
        new_incidents = []
        
        # Apply each enabled rule
        for rule in self.rules:
            if rule.enabled:
                try:
                    rule_incidents = rule.analyze(events, self.context)
                    new_incidents.extend(rule_incidents)
                    
                    logger.info(f"Rule '{rule.name}' detected {len(rule_incidents)} incidents")
                    
                except Exception as e:
                    logger.error(f"Error in correlation rule '{rule.name}': {e}")
        
        # Add to incident list
        self.incidents.extend(new_incidents)
        
        # Log detected incidents
        for incident in new_incidents:
            logger.warning(f"Security incident detected: {incident.title} (Severity: {incident.severity.value})")
        
        return new_incidents
    
    def get_incidents(self, 
                     severity: Optional[IncidentSeverity] = None,
                     incident_type: Optional[IncidentType] = None,
                     status: Optional[str] = None,
                     limit: int = 100) -> List[SecurityIncident]:
        """Get incidents with filtering"""
        
        filtered_incidents = self.incidents
        
        if severity:
            filtered_incidents = [i for i in filtered_incidents if i.severity == severity]
        
        if incident_type:
            filtered_incidents = [i for i in filtered_incidents if i.incident_type == incident_type]
        
        if status:
            filtered_incidents = [i for i in filtered_incidents if i.status == status]
        
        # Sort by detection time (newest first)
        filtered_incidents.sort(key=lambda x: x.detected_at, reverse=True)
        
        return filtered_incidents[:limit]
    
    def update_incident_status(self, incident_id: str, status: str, resolution: Optional[str] = None):
        """Update incident status"""
        for incident in self.incidents:
            if incident.id == incident_id:
                incident.status = status
                if resolution:
                    incident.resolution = resolution
                if status in ['resolved', 'closed']:
                    incident.resolved_at = datetime.now(timezone.utc)
                break
    
    def get_incident_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected incidents"""
        if not self.incidents:
            return {}
        
        # Count by severity
        severity_counts = defaultdict(int)
        for incident in self.incidents:
            severity_counts[incident.severity.value] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for incident in self.incidents:
            type_counts[incident.incident_type.value] += 1
        
        # Count by status
        status_counts = defaultdict(int)
        for incident in self.incidents:
            status_counts[incident.status] += 1
        
        # Recent incidents (last 24 hours)
        recent_threshold = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_incidents = [i for i in self.incidents if i.detected_at >= recent_threshold]
        
        return {
            'total_incidents': len(self.incidents),
            'recent_incidents': len(recent_incidents),
            'severity_distribution': dict(severity_counts),
            'type_distribution': dict(type_counts),
            'status_distribution': dict(status_counts),
            'average_risk_score': statistics.mean([i.risk_score for i in self.incidents]),
            'rules_enabled': len([r for r in self.rules if r.enabled]),
            'rules_total': len(self.rules)
        }
    
    def update_context(self, context_updates: Dict[str, Any]):
        """Update correlation context"""
        self.context.update(context_updates)


# Global correlator instance
_global_correlator: Optional[SecurityEventCorrelator] = None

def get_global_correlator() -> SecurityEventCorrelator:
    """Get or create the global security correlator"""
    global _global_correlator
    if _global_correlator is None:
        _global_correlator = SecurityEventCorrelator()
    return _global_correlator