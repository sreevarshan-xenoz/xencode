#!/usr/bin/env python3
"""
Compliance Reporter

Generates compliance reports for various regulatory frameworks
including GDPR, SOX, HIPAA, PCI-DSS, and custom compliance requirements.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
import logging

from .audit_logger import AuditEvent, AuditEventType, AuditSeverity
from .security_correlator import SecurityIncident, IncidentSeverity

logger = logging.getLogger(__name__)


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"  # General Data Protection Regulation
    SOX = "sox"    # Sarbanes-Oxley Act
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    ISO_27001 = "iso_27001"  # Information Security Management
    NIST = "nist"  # NIST Cybersecurity Framework
    CUSTOM = "custom"  # Custom compliance requirements


class ComplianceStatus(str, Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNKNOWN = "unknown"


@dataclass
class ComplianceRequirement:
    """Represents a single compliance requirement"""
    
    id: str
    framework: ComplianceFramework
    title: str
    description: str
    category: str
    mandatory: bool = True
    
    # Audit criteria
    required_events: List[AuditEventType] = field(default_factory=list)
    prohibited_events: List[AuditEventType] = field(default_factory=list)
    time_window: Optional[int] = None  # seconds
    frequency_requirement: Optional[str] = None
    
    # Validation rules
    validation_rules: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceCheck:
    """Result of a compliance requirement check"""
    
    requirement_id: str
    status: ComplianceStatus
    score: float  # 0.0 to 1.0
    evidence_count: int
    missing_evidence: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_checked: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    
    id: str
    framework: ComplianceFramework
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    
    # Overall compliance
    overall_status: ComplianceStatus
    compliance_score: float  # 0.0 to 1.0
    
    # Requirement results
    total_requirements: int
    compliant_requirements: int
    non_compliant_requirements: int
    partially_compliant_requirements: int
    
    # Detailed results
    requirement_checks: List[ComplianceCheck] = field(default_factory=list)
    
    # Incidents and violations
    security_incidents: List[SecurityIncident] = field(default_factory=list)
    compliance_violations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recommendations and actions
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    
    # Audit trail
    audited_events: int = 0
    audit_period_coverage: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class GDPRRequirements:
    """GDPR compliance requirements"""
    
    @staticmethod
    def get_requirements() -> List[ComplianceRequirement]:
        return [
            ComplianceRequirement(
                id="gdpr_art_30",
                framework=ComplianceFramework.GDPR,
                title="Records of Processing Activities",
                description="Maintain records of all data processing activities",
                category="Documentation",
                required_events=[
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.DATA_MODIFICATION,
                    AuditEventType.DATA_DELETION
                ],
                validation_rules=[
                    "All data processing activities must be logged",
                    "Records must include purpose, categories, and retention periods"
                ],
                evidence_requirements=[
                    "Audit logs of data processing",
                    "Data processing register",
                    "Purpose documentation"
                ]
            ),
            ComplianceRequirement(
                id="gdpr_art_32",
                framework=ComplianceFramework.GDPR,
                title="Security of Processing",
                description="Implement appropriate technical and organizational measures",
                category="Security",
                required_events=[
                    AuditEventType.SECURITY_SCAN,
                    AuditEventType.ACCESS_GRANTED,
                    AuditEventType.ACCESS_DENIED
                ],
                validation_rules=[
                    "Regular security assessments must be conducted",
                    "Access controls must be implemented and monitored"
                ],
                evidence_requirements=[
                    "Security scan reports",
                    "Access control logs",
                    "Incident response records"
                ]
            ),
            ComplianceRequirement(
                id="gdpr_art_33",
                framework=ComplianceFramework.GDPR,
                title="Notification of Personal Data Breach",
                description="Report data breaches within 72 hours",
                category="Incident Response",
                required_events=[
                    AuditEventType.SECURITY_INCIDENT
                ],
                time_window=259200,  # 72 hours
                validation_rules=[
                    "Data breaches must be detected and reported within 72 hours",
                    "Breach notifications must include required information"
                ],
                evidence_requirements=[
                    "Incident detection logs",
                    "Breach notification records",
                    "Timeline documentation"
                ]
            ),
            ComplianceRequirement(
                id="gdpr_art_25",
                framework=ComplianceFramework.GDPR,
                title="Data Protection by Design and by Default",
                description="Implement privacy by design principles",
                category="Privacy",
                required_events=[
                    AuditEventType.CONFIGURATION_CHANGE,
                    AuditEventType.SYSTEM_START
                ],
                validation_rules=[
                    "Privacy controls must be implemented by default",
                    "Data minimization principles must be applied"
                ],
                evidence_requirements=[
                    "System configuration records",
                    "Privacy impact assessments",
                    "Data minimization documentation"
                ]
            )
        ]


class SOXRequirements:
    """SOX compliance requirements"""
    
    @staticmethod
    def get_requirements() -> List[ComplianceRequirement]:
        return [
            ComplianceRequirement(
                id="sox_302",
                framework=ComplianceFramework.SOX,
                title="Corporate Responsibility for Financial Reports",
                description="Maintain accurate financial reporting controls",
                category="Financial Controls",
                required_events=[
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.DATA_MODIFICATION,
                    AuditEventType.CONFIGURATION_CHANGE
                ],
                validation_rules=[
                    "All financial data access must be logged and monitored",
                    "Changes to financial systems must be authorized and documented"
                ],
                evidence_requirements=[
                    "Financial data access logs",
                    "Change management records",
                    "Authorization documentation"
                ]
            ),
            ComplianceRequirement(
                id="sox_404",
                framework=ComplianceFramework.SOX,
                title="Management Assessment of Internal Controls",
                description="Assess effectiveness of internal controls over financial reporting",
                category="Internal Controls",
                required_events=[
                    AuditEventType.COMPLIANCE_CHECK,
                    AuditEventType.AUDIT_EXPORT
                ],
                frequency_requirement="quarterly",
                validation_rules=[
                    "Internal controls must be assessed quarterly",
                    "Control deficiencies must be documented and remediated"
                ],
                evidence_requirements=[
                    "Control assessment reports",
                    "Deficiency documentation",
                    "Remediation evidence"
                ]
            )
        ]


class ComplianceEngine:
    """Core compliance checking engine"""
    
    def __init__(self):
        self.requirements: Dict[ComplianceFramework, List[ComplianceRequirement]] = {}
        self._load_default_requirements()
    
    def _load_default_requirements(self):
        """Load default compliance requirements"""
        self.requirements[ComplianceFramework.GDPR] = GDPRRequirements.get_requirements()
        self.requirements[ComplianceFramework.SOX] = SOXRequirements.get_requirements()
    
    def add_requirement(self, requirement: ComplianceRequirement):
        """Add a custom compliance requirement"""
        if requirement.framework not in self.requirements:
            self.requirements[requirement.framework] = []
        self.requirements[requirement.framework].append(requirement)
    
    def check_requirement(self, 
                         requirement: ComplianceRequirement,
                         events: List[AuditEvent],
                         incidents: List[SecurityIncident]) -> ComplianceCheck:
        """Check compliance for a single requirement"""
        
        evidence_count = 0
        missing_evidence = []
        violations = []
        recommendations = []
        
        # Check for required events
        if requirement.required_events:
            required_event_types = set(requirement.required_events)
            found_event_types = set(event.event_type for event in events)
            
            missing_events = required_event_types - found_event_types
            if missing_events:
                missing_evidence.extend([f"Missing {event.value} events" for event in missing_events])
            else:
                evidence_count += len(requirement.required_events)
        
        # Check for prohibited events
        if requirement.prohibited_events:
            prohibited_event_types = set(requirement.prohibited_events)
            found_prohibited = prohibited_event_types.intersection(
                set(event.event_type for event in events)
            )
            if found_prohibited:
                violations.extend([f"Found prohibited {event.value} events" for event in found_prohibited])
        
        # Check time window requirements
        if requirement.time_window:
            recent_events = [
                event for event in events
                if (datetime.now(timezone.utc) - event.timestamp).total_seconds() <= requirement.time_window
            ]
            if not recent_events and requirement.required_events:
                violations.append(f"No required events found within {requirement.time_window} seconds")
        
        # Check frequency requirements
        if requirement.frequency_requirement:
            frequency_met = self._check_frequency_requirement(
                requirement.frequency_requirement, events
            )
            if not frequency_met:
                violations.append(f"Frequency requirement not met: {requirement.frequency_requirement}")
        
        # Check for related security incidents
        related_incidents = [
            incident for incident in incidents
            if any(req_event.value in incident.indicators for req_event in requirement.required_events)
        ]
        
        if related_incidents:
            high_severity_incidents = [
                i for i in related_incidents 
                if i.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]
            ]
            if high_severity_incidents:
                violations.append(f"High-severity security incidents detected: {len(high_severity_incidents)}")
        
        # Calculate compliance score
        total_checks = len(requirement.required_events) + len(requirement.validation_rules)
        if total_checks == 0:
            score = 1.0 if not violations else 0.0
        else:
            score = max(0.0, (evidence_count - len(violations)) / total_checks)
        
        # Determine status
        if score >= 0.9 and not violations:
            status = ComplianceStatus.COMPLIANT
        elif score >= 0.5:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        # Generate recommendations
        if violations:
            recommendations.append("Address identified violations immediately")
        if missing_evidence:
            recommendations.append("Implement missing audit controls")
        if score < 0.8:
            recommendations.append("Improve compliance monitoring and controls")
        
        return ComplianceCheck(
            requirement_id=requirement.id,
            status=status,
            score=score,
            evidence_count=evidence_count,
            missing_evidence=missing_evidence,
            violations=violations,
            recommendations=recommendations
        )
    
    def _check_frequency_requirement(self, frequency: str, events: List[AuditEvent]) -> bool:
        """Check if frequency requirement is met"""
        if frequency == "daily":
            threshold = datetime.now(timezone.utc) - timedelta(days=1)
        elif frequency == "weekly":
            threshold = datetime.now(timezone.utc) - timedelta(weeks=1)
        elif frequency == "monthly":
            threshold = datetime.now(timezone.utc) - timedelta(days=30)
        elif frequency == "quarterly":
            threshold = datetime.now(timezone.utc) - timedelta(days=90)
        else:
            return True  # Unknown frequency, assume met
        
        recent_events = [event for event in events if event.timestamp >= threshold]
        return len(recent_events) > 0


class ComplianceReporter:
    """Generates comprehensive compliance reports"""
    
    def __init__(self):
        self.engine = ComplianceEngine()
    
    def generate_report(self,
                       framework: ComplianceFramework,
                       events: List[AuditEvent],
                       incidents: List[SecurityIncident],
                       period_start: Optional[datetime] = None,
                       period_end: Optional[datetime] = None) -> ComplianceReport:
        """Generate a compliance report for the specified framework"""
        
        if period_end is None:
            period_end = datetime.now(timezone.utc)
        if period_start is None:
            period_start = period_end - timedelta(days=30)  # Default 30-day period
        
        # Filter events to reporting period
        period_events = [
            event for event in events
            if period_start <= event.timestamp <= period_end
        ]
        
        # Filter incidents to reporting period
        period_incidents = [
            incident for incident in incidents
            if period_start <= incident.detected_at <= period_end
        ]
        
        # Get requirements for framework
        requirements = self.engine.requirements.get(framework, [])
        
        # Check each requirement
        requirement_checks = []
        compliant_count = 0
        non_compliant_count = 0
        partially_compliant_count = 0
        
        for requirement in requirements:
            check = self.engine.check_requirement(requirement, period_events, period_incidents)
            requirement_checks.append(check)
            
            if check.status == ComplianceStatus.COMPLIANT:
                compliant_count += 1
            elif check.status == ComplianceStatus.NON_COMPLIANT:
                non_compliant_count += 1
            elif check.status == ComplianceStatus.PARTIALLY_COMPLIANT:
                partially_compliant_count += 1
        
        # Calculate overall compliance
        if not requirements:
            overall_status = ComplianceStatus.UNKNOWN
            compliance_score = 0.0
        else:
            compliance_score = sum(check.score for check in requirement_checks) / len(requirements)
            
            if compliance_score >= 0.9:
                overall_status = ComplianceStatus.COMPLIANT
            elif compliance_score >= 0.7:
                overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
            else:
                overall_status = ComplianceStatus.NON_COMPLIANT
        
        # Identify critical issues
        critical_issues = []
        all_recommendations = []
        
        for check in requirement_checks:
            if check.status == ComplianceStatus.NON_COMPLIANT:
                critical_issues.extend(check.violations)
            all_recommendations.extend(check.recommendations)
        
        # Add high-severity incidents as critical issues
        critical_incidents = [
            i for i in period_incidents
            if i.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]
        ]
        critical_issues.extend([f"Security incident: {i.title}" for i in critical_incidents])
        
        # Generate action items
        action_items = []
        for check in requirement_checks:
            if check.status != ComplianceStatus.COMPLIANT:
                action_items.append({
                    'requirement_id': check.requirement_id,
                    'priority': 'high' if check.status == ComplianceStatus.NON_COMPLIANT else 'medium',
                    'description': f"Address compliance issues for {check.requirement_id}",
                    'violations': check.violations,
                    'recommendations': check.recommendations
                })
        
        # Calculate audit coverage
        total_possible_events = len(period_events) if period_events else 1
        audited_events = len([e for e in period_events if e.event_type in [
            AuditEventType.DATA_ACCESS, AuditEventType.DATA_MODIFICATION,
            AuditEventType.SECURITY_SCAN, AuditEventType.COMPLIANCE_CHECK
        ]])
        audit_coverage = audited_events / total_possible_events
        
        return ComplianceReport(
            id=f"compliance_{framework.value}_{int(time.time())}",
            framework=framework,
            generated_at=datetime.now(timezone.utc),
            period_start=period_start,
            period_end=period_end,
            overall_status=overall_status,
            compliance_score=compliance_score,
            total_requirements=len(requirements),
            compliant_requirements=compliant_count,
            non_compliant_requirements=non_compliant_count,
            partially_compliant_requirements=partially_compliant_count,
            requirement_checks=requirement_checks,
            security_incidents=period_incidents,
            compliance_violations=[
                {'requirement': check.requirement_id, 'violations': check.violations}
                for check in requirement_checks if check.violations
            ],
            critical_issues=critical_issues,
            recommendations=list(set(all_recommendations)),  # Remove duplicates
            action_items=action_items,
            audited_events=audited_events,
            audit_period_coverage=audit_coverage
        )
    
    def export_report(self, report: ComplianceReport, format: str = 'json') -> str:
        """Export compliance report in specified format"""
        
        if format.lower() == 'json':
            return self._export_json(report)
        elif format.lower() == 'html':
            return self._export_html(report)
        elif format.lower() == 'csv':
            return self._export_csv(report)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, report: ComplianceReport) -> str:
        """Export report as JSON"""
        
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        report_dict = {
            'id': report.id,
            'framework': report.framework.value,
            'generated_at': report.generated_at.isoformat(),
            'period_start': report.period_start.isoformat(),
            'period_end': report.period_end.isoformat(),
            'overall_status': report.overall_status.value,
            'compliance_score': report.compliance_score,
            'summary': {
                'total_requirements': report.total_requirements,
                'compliant_requirements': report.compliant_requirements,
                'non_compliant_requirements': report.non_compliant_requirements,
                'partially_compliant_requirements': report.partially_compliant_requirements
            },
            'requirement_checks': [
                {
                    'requirement_id': check.requirement_id,
                    'status': check.status.value,
                    'score': check.score,
                    'evidence_count': check.evidence_count,
                    'missing_evidence': check.missing_evidence,
                    'violations': check.violations,
                    'recommendations': check.recommendations,
                    'last_checked': check.last_checked.isoformat()
                }
                for check in report.requirement_checks
            ],
            'security_incidents': [
                {
                    'id': incident.id,
                    'type': incident.incident_type.value,
                    'severity': incident.severity.value,
                    'title': incident.title,
                    'detected_at': incident.detected_at.isoformat()
                }
                for incident in report.security_incidents
            ],
            'critical_issues': report.critical_issues,
            'recommendations': report.recommendations,
            'action_items': report.action_items,
            'audit_metrics': {
                'audited_events': report.audited_events,
                'audit_period_coverage': report.audit_period_coverage
            }
        }
        
        return json.dumps(report_dict, indent=2, default=serialize_datetime)
    
    def _export_html(self, report: ComplianceReport) -> str:
        """Export report as HTML"""
        
        status_colors = {
            ComplianceStatus.COMPLIANT: '#28a745',
            ComplianceStatus.PARTIALLY_COMPLIANT: '#ffc107',
            ComplianceStatus.NON_COMPLIANT: '#dc3545',
            ComplianceStatus.UNKNOWN: '#6c757d'
        }
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compliance Report - {report.framework.value.upper()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .status {{ color: {status_colors[report.overall_status]}; font-weight: bold; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .section {{ margin: 20px 0; }}
                .requirement {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .compliant {{ border-left: 5px solid #28a745; }}
                .non-compliant {{ border-left: 5px solid #dc3545; }}
                .partially-compliant {{ border-left: 5px solid #ffc107; }}
                .critical {{ background-color: #f8d7da; padding: 10px; border-radius: 5px; margin: 5px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Compliance Report: {report.framework.value.upper()}</h1>
                <p><strong>Report ID:</strong> {report.id}</p>
                <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p><strong>Period:</strong> {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}</p>
                <p><strong>Overall Status:</strong> <span class="status">{report.overall_status.value.replace('_', ' ').title()}</span></p>
                <p><strong>Compliance Score:</strong> <span class="score">{report.compliance_score:.1%}</span></p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Requirements</td><td>{report.total_requirements}</td></tr>
                    <tr><td>Compliant</td><td>{report.compliant_requirements}</td></tr>
                    <tr><td>Partially Compliant</td><td>{report.partially_compliant_requirements}</td></tr>
                    <tr><td>Non-Compliant</td><td>{report.non_compliant_requirements}</td></tr>
                    <tr><td>Audited Events</td><td>{report.audited_events}</td></tr>
                    <tr><td>Audit Coverage</td><td>{report.audit_period_coverage:.1%}</td></tr>
                </table>
            </div>
        """
        
        if report.critical_issues:
            html += """
            <div class="section">
                <h2>Critical Issues</h2>
            """
            for issue in report.critical_issues:
                html += f'<div class="critical">{issue}</div>'
            html += "</div>"
        
        html += """
            <div class="section">
                <h2>Requirement Details</h2>
        """
        
        for check in report.requirement_checks:
            status_class = check.status.value.replace('_', '-')
            html += f"""
                <div class="requirement {status_class}">
                    <h3>{check.requirement_id}</h3>
                    <p><strong>Status:</strong> {check.status.value.replace('_', ' ').title()}</p>
                    <p><strong>Score:</strong> {check.score:.1%}</p>
                    <p><strong>Evidence Count:</strong> {check.evidence_count}</p>
            """
            
            if check.violations:
                html += "<p><strong>Violations:</strong></p><ul>"
                for violation in check.violations:
                    html += f"<li>{violation}</li>"
                html += "</ul>"
            
            if check.recommendations:
                html += "<p><strong>Recommendations:</strong></p><ul>"
                for rec in check.recommendations:
                    html += f"<li>{rec}</li>"
                html += "</ul>"
            
            html += "</div>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _export_csv(self, report: ComplianceReport) -> str:
        """Export report as CSV"""
        
        lines = [
            "Requirement ID,Status,Score,Evidence Count,Violations,Recommendations"
        ]
        
        for check in report.requirement_checks:
            violations = '; '.join(check.violations)
            recommendations = '; '.join(check.recommendations)
            
            lines.append(f'"{check.requirement_id}","{check.status.value}",{check.score},{check.evidence_count},"{violations}","{recommendations}"')
        
        return '\n'.join(lines)


# Global compliance reporter instance
_global_reporter: Optional[ComplianceReporter] = None

def get_global_compliance_reporter() -> ComplianceReporter:
    """Get or create the global compliance reporter"""
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = ComplianceReporter()
    return _global_reporter