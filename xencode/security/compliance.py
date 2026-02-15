"""
Automated Compliance Framework
Implements ComplianceManager for regulatory requirements, GDPR, HIPAA, SOX compliance checking,
automated audit trail generation, and compliance reporting and alerts.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import secrets
import hashlib
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
import inspect
from functools import wraps


logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Regulatory compliance standards."""
    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOX = "sox"  # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    CCPA = "ccpa"  # California Consumer Privacy Act
    ISO_27001 = "iso_27001"  # Information Security Management
    SOC_2 = "soc_2"  # Service Organization Control 2


class ComplianceRequirement(Enum):
    """Specific compliance requirements."""
    # GDPR Requirements
    LAWFUL_PROCESSING = "gdpr_lawful_processing"
    CONSENT_MANAGEMENT = "gdpr_consent_management"
    RIGHT_TO_ACCESS = "gdpr_right_to_access"
    RIGHT_TO_ERASURE = "gdpr_right_to_erasure"
    DATA_PORTABILITY = "gdpr_data_portability"
    PRIVACY_BY_DESIGN = "gdpr_privacy_by_design"
    DPIA_REQUIRED = "gdpr_dpia_required"
    
    # HIPAA Requirements
    PRIVACY_RULE = "hipaa_privacy_rule"
    SECURITY_RULE = "hipaa_security_rule"
    BREACH_NOTIFICATION = "hipaa_breach_notification"
    MINIMUM_NECESSARY = "hipaa_minimum_necessary"
    SANITIZED_DISCLOSURE = "hipaa_sanitized_disclosure"
    
    # SOX Requirements
    INTERNAL_CONTROLS = "sox_internal_controls"
    DOCUMENTATION_MAINTENANCE = "sox_documentation_maintenance"
    OFFICER_CERTIFICATION = "sox_officer_certification"
    AUDIT_REQUIREMENTS = "sox_audit_requirements"
    FINANCIAL_TRANSPARENCY = "sox_financial_transparency"


class ComplianceStatus(Enum):
    """Status of compliance checks."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    EXEMPT = "exempt"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ComplianceCheck:
    """A single compliance check."""
    check_id: str
    requirement: ComplianceRequirement
    standard: ComplianceStandard
    description: str
    check_function: str  # Name of the function to perform the check
    parameters: Dict[str, Any]
    last_run: Optional[datetime]
    status: ComplianceStatus
    findings: List[str]
    metadata: Dict[str, Any]


@dataclass
class ComplianceFinding:
    """A finding from a compliance check."""
    finding_id: str
    check_id: str
    severity: str  # low, medium, high, critical
    description: str
    details: Dict[str, Any]
    timestamp: datetime
    remediation_steps: List[str]
    status: str  # open, in_progress, resolved, waived


@dataclass
class AuditTrailEntry:
    """An entry in the audit trail."""
    entry_id: str
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    compliance_impact: List[ComplianceRequirement]
    metadata: Dict[str, Any]


@dataclass
class ComplianceReport:
    """A compliance report."""
    report_id: str
    standard: ComplianceStandard
    generated_at: datetime
    coverage_period_start: datetime
    coverage_period_end: datetime
    checks_performed: List[str]
    overall_status: ComplianceStatus
    findings_summary: Dict[str, int]  # severity -> count
    recommendations: List[str]
    metadata: Dict[str, Any]


class ComplianceRuleEngine:
    """Engine for evaluating compliance rules."""
    
    def __init__(self):
        self.rules: Dict[str, callable] = {}
        self.register_default_rules()
        
    def register_rule(self, rule_id: str, rule_function: callable):
        """Register a compliance rule function."""
        self.rules[rule_id] = rule_function
        
    def register_default_rules(self):
        """Register default compliance rules."""
        # GDPR rules
        self.register_rule("gdpr_consent_exists", self._check_gdpr_consent_exists)
        self.register_rule("gdpr_data_minimization", self._check_gdpr_data_minimization)
        self.register_rule("gdpr_right_to_erasure", self._check_gdpr_right_to_erasure)
        
        # HIPAA rules
        self.register_rule("hipaa_access_logs", self._check_hipaa_access_logs)
        self.register_rule("hipaa_encryption", self._check_hipaa_encryption)
        self.register_rule("hipaa_breach_procedures", self._check_hipaa_breach_procedures)
        
        # SOX rules
        self.register_rule("sox_financial_controls", self._check_sox_financial_controls)
        self.register_rule("sox_documentation", self._check_sox_documentation)
        
    def evaluate_rule(self, rule_id: str, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Evaluate a compliance rule against provided data.
        
        Returns:
            Tuple of (is_compliant, list_of_findings)
        """
        if rule_id not in self.rules:
            raise ValueError(f"Rule {rule_id} not found")
            
        rule_function = self.rules[rule_id]
        return rule_function(data)
        
    def _check_gdpr_consent_exists(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if GDPR consent exists for data processing."""
        user_id = data.get("user_id")
        processing_purpose = data.get("purpose", "")
        
        # In a real system, this would check a consent management database
        # For this demo, we'll simulate the check
        has_consent = data.get("has_consent", False)
        consent_purpose = data.get("consent_purpose", "")
        
        if not has_consent:
            return False, [f"No consent found for user {user_id}"]
        elif consent_purpose != processing_purpose:
            return False, [f"Consent purpose mismatch for user {user_id}: expected '{processing_purpose}', got '{consent_purpose}'"]
        else:
            return True, []
            
    def _check_gdpr_data_minimization(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if only necessary data is being processed (GDPR data minimization)."""
        requested_fields = data.get("requested_fields", [])
        necessary_fields = data.get("necessary_fields", [])
        
        unnecessary_fields = [field for field in requested_fields if field not in necessary_fields]
        
        if unnecessary_fields:
            return False, [f"Unnecessary fields accessed: {unnecessary_fields}"]
        else:
            return True, []
            
    def _check_gdpr_right_to_erasure(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if right to erasure is properly handled."""
        user_id = data.get("user_id")
        deletion_requested = data.get("deletion_requested", False)
        deletion_completed = data.get("deletion_completed", False)
        
        if deletion_requested and not deletion_completed:
            return False, [f"Deletion requested for user {user_id} but not completed"]
        else:
            return True, []
            
    def _check_hipaa_access_logs(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if HIPAA-compliant access logs are maintained."""
        access_logs = data.get("access_logs", [])
        required_fields = ["timestamp", "user_id", "action", "resource"]
        
        missing_logs = []
        for log in access_logs:
            missing_fields = [field for field in required_fields if field not in log]
            if missing_fields:
                missing_logs.append(f"Missing fields {missing_fields} in log entry")
                
        if missing_logs:
            return False, missing_logs
        else:
            return True, []
            
    def _check_hipaa_encryption(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if PHI is properly encrypted (HIPAA security rule)."""
        data_type = data.get("data_type", "")
        is_encrypted = data.get("is_encrypted", False)
        encryption_standard = data.get("encryption_standard", "")
        
        if data_type.lower() in ["phi", "protected_health_information"] and not is_encrypted:
            return False, ["PHI data is not encrypted"]
        elif data_type.lower() in ["phi", "protected_health_information"] and encryption_standard not in ["AES-256", "RSA-2048"]:
            return False, [f"Inadequate encryption standard for PHI: {encryption_standard}"]
        else:
            return True, []
            
    def _check_hipaa_breach_procedures(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if HIPAA breach notification procedures are in place."""
        breach_occurred = data.get("breach_occurred", False)
        notification_procedures_in_place = data.get("notification_procedures_in_place", False)
        breach_reported = data.get("breach_reported", False)
        
        if breach_occurred and not notification_procedures_in_place:
            return False, ["No breach notification procedures in place"]
        elif breach_occurred and not breach_reported:
            return False, ["Breach occurred but not reported"]
        else:
            return True, []
            
    def _check_sox_financial_controls(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check SOX financial controls."""
        transaction_amount = data.get("transaction_amount", 0)
        approval_threshold = data.get("approval_threshold", 10000)
        has_approval = data.get("has_approval", False)
        approver_role = data.get("approver_role", "")
        
        if transaction_amount > approval_threshold and not has_approval:
            return False, [f"Transaction ${transaction_amount} exceeds approval threshold (${approval_threshold}) without approval"]
        elif transaction_amount > approval_threshold and approver_role not in ["manager", "director", "officer"]:
            return False, [f"Transaction approved by role '{approver_role}' which may not have sufficient authority"]
        else:
            return True, []
            
    def _check_sox_documentation(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check SOX documentation requirements."""
        document_type = data.get("document_type", "")
        has_audit_trail = data.get("has_audit_trail", False)
        is_signed = data.get("is_signed", False)
        retention_period = data.get("retention_period_days", 0)
        
        if document_type in ["financial_statement", "internal_control_document"] and not has_audit_trail:
            return False, [f"Document {document_type} lacks required audit trail"]
        elif document_type in ["financial_statement", "internal_control_document"] and not is_signed:
            return False, [f"Document {document_type} is not properly signed"]
        elif retention_period < 7 * 365:  # 7 years for SOX
            return False, [f"Document retention period ({retention_period} days) is less than 7 years required by SOX"]
        else:
            return True, []


class AuditTrailManager:
    """Manages the compliance audit trail."""
    
    def __init__(self):
        self.audit_entries: List[AuditTrailEntry] = []
        self.retention_period = timedelta(days=365 * 7)  # 7 years for SOX compliance
        
    def log_action(
        self, 
        user_id: str, 
        action: str, 
        resource: str, 
        details: Dict[str, Any] = None,
        compliance_impact: List[ComplianceRequirement] = None
    ) -> str:
        """Log an action to the audit trail."""
        entry_id = f"audit_{secrets.token_hex(8)}"
        
        entry = AuditTrailEntry(
            entry_id=entry_id,
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            details=details or {},
            compliance_impact=compliance_impact or [],
            metadata={"logged_by": "compliance_framework"}
        )
        
        self.audit_entries.append(entry)
        
        logger.info(f"Audit entry logged: {entry_id} - {action} by {user_id}")
        return entry_id
        
    def get_entries_for_user(self, user_id: str) -> List[AuditTrailEntry]:
        """Get audit entries for a specific user."""
        return [entry for entry in self.audit_entries if entry.user_id == user_id]
        
    def get_entries_for_resource(self, resource: str) -> List[AuditTrailEntry]:
        """Get audit entries for a specific resource."""
        return [entry for entry in self.audit_entries if entry.resource == resource]
        
    def get_entries_by_compliance_requirement(self, requirement: ComplianceRequirement) -> List[AuditTrailEntry]:
        """Get audit entries related to a specific compliance requirement."""
        return [entry for entry in self.audit_entries if requirement in entry.compliance_impact]
        
    def cleanup_old_entries(self):
        """Remove audit entries older than the retention period."""
        cutoff_date = datetime.now() - self.retention_period
        old_entries = [entry for entry in self.audit_entries if entry.timestamp < cutoff_date]
        
        for entry in old_entries:
            self.audit_entries.remove(entry)
            
        logger.info(f"Cleaned up {len(old_entries)} old audit entries")
        
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate a compliance-focused audit report."""
        relevant_entries = [
            entry for entry in self.audit_entries
            if start_date <= entry.timestamp <= end_date
        ]
        
        # Count actions by type
        action_counts = {}
        for entry in relevant_entries:
            action_type = entry.action
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
            
        # Count entries by compliance requirement
        requirement_counts = {}
        for entry in relevant_entries:
            for req in entry.compliance_impact:
                req_str = req.value
                requirement_counts[req_str] = requirement_counts.get(req_str, 0) + 1
                
        return {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_entries": len(relevant_entries),
            "action_distribution": action_counts,
            "compliance_requirement_impact": requirement_counts,
            "users_involved": len(set(entry.user_id for entry in relevant_entries)),
            "resources_accessed": len(set(entry.resource for entry in relevant_entries))
        }


class ComplianceAlertManager:
    """Manages compliance alerts and notifications."""
    
    def __init__(self):
        self.alerts: List[ComplianceFinding] = []
        self.subscribers: Dict[str, List[str]] = {}  # requirement -> [email_addresses]
        self.severity_thresholds = {
            "low": 1,
            "medium": 2, 
            "high": 3,
            "critical": 4
        }
        
    def create_alert(self, finding: ComplianceFinding):
        """Create a compliance alert from a finding."""
        self.alerts.append(finding)
        
        # Notify subscribers if severity meets threshold
        if self.severity_thresholds.get(finding.severity, 0) >= 2:  # Medium+ severity
            self._notify_subscribers(finding)
            
        logger.warning(f"Compliance alert created: {finding.description}")
        
    def subscribe_to_requirement(self, email: str, requirement: ComplianceRequirement):
        """Subscribe an email to alerts for a specific requirement."""
        req_str = requirement.value
        if req_str not in self.subscribers:
            self.subscribers[req_str] = []
        if email not in self.subscribers[req_str]:
            self.subscribers[req_str].append(email)
            
    def get_unresolved_alerts(self) -> List[ComplianceFinding]:
        """Get all unresolved compliance alerts."""
        return [alert for alert in self.alerts if alert.status == "open"]
        
    def _notify_subscribers(self, finding: ComplianceFinding):
        """Notify subscribers about a compliance finding."""
        # In a real system, this would send emails or other notifications
        # For this demo, we'll just log the notification
        affected_requirements = [req.value for req in finding.compliance_impact] if hasattr(finding, 'compliance_impact') else []
        
        for req in affected_requirements:
            if req in self.subscribers:
                for email in self.subscribers[req]:
                    logger.info(f"Sent compliance alert to {email}: {finding.description}")
        

class ComplianceManager:
    """
    Compliance manager for regulatory requirements with GDPR, HIPAA, SOX compliance checking,
    automated audit trail generation, and compliance reporting.
    """
    
    def __init__(self):
        self.rule_engine = ComplianceRuleEngine()
        self.audit_trail = AuditTrailManager()
        self.alert_manager = ComplianceAlertManager()
        self.compliance_checks: Dict[str, ComplianceCheck] = {}
        self.compliance_standards = {
            ComplianceStandard.GDPR: [
                ComplianceRequirement.LAWFUL_PROCESSING,
                ComplianceRequirement.CONSENT_MANAGEMENT,
                ComplianceRequirement.RIGHT_TO_ACCESS,
                ComplianceRequirement.RIGHT_TO_ERASURE,
                ComplianceRequirement.DATA_PORTABILITY,
                ComplianceRequirement.PRIVACY_BY_DESIGN,
                ComplianceRequirement.DPIA_REQUIRED
            ],
            ComplianceStandard.HIPAA: [
                ComplianceRequirement.PRIVACY_RULE,
                ComplianceRequirement.SECURITY_RULE,
                ComplianceRequirement.BREACH_NOTIFICATION,
                ComplianceRequirement.MINIMUM_NECESSARY,
                ComplianceRequirement.SANITIZED_DISCLOSURE
            ],
            ComplianceStandard.SOX: [
                ComplianceRequirement.INTERNAL_CONTROLS,
                ComplianceRequirement.DOCUMENTATION_MAINTENANCE,
                ComplianceRequirement.OFFICER_CERTIFICATION,
                ComplianceRequirement.AUDIT_REQUIREMENTS,
                ComplianceRequirement.FINANCIAL_TRANSPARENCY
            ]
        }
        
    def register_compliance_check(
        self, 
        requirement: ComplianceRequirement, 
        standard: ComplianceStandard,
        description: str,
        check_function: str,
        parameters: Dict[str, Any] = None
    ) -> str:
        """Register a compliance check."""
        check_id = f"check_{secrets.token_hex(8)}"
        
        check = ComplianceCheck(
            check_id=check_id,
            requirement=requirement,
            standard=standard,
            description=description,
            check_function=check_function,
            parameters=parameters or {},
            last_run=None,
            status=ComplianceStatus.NOT_APPLICABLE,
            findings=[],
            metadata={"registered_at": datetime.now().isoformat()}
        )
        
        self.compliance_checks[check_id] = check
        
        logger.info(f"Registered compliance check: {check_id} for {requirement.value}")
        return check_id
        
    def run_compliance_check(self, check_id: str, data: Dict[str, Any]) -> ComplianceStatus:
        """Run a specific compliance check."""
        if check_id not in self.compliance_checks:
            raise ValueError(f"Compliance check {check_id} not found")
            
        check = self.compliance_checks[check_id]
        
        # Evaluate the rule
        is_compliant, findings = self.rule_engine.evaluate_rule(check.check_function, data)
        
        # Update check status
        new_status = ComplianceStatus.COMPLIANT if is_compliant else ComplianceStatus.NON_COMPLIANT
        check.status = new_status
        check.findings = findings
        check.last_run = datetime.now()
        
        # Create alerts for non-compliant findings
        if not is_compliant:
            for finding_desc in findings:
                severity = "high" if "critical" in finding_desc.lower() else "medium"
                
                finding = ComplianceFinding(
                    finding_id=f"finding_{secrets.token_hex(8)}",
                    check_id=check_id,
                    severity=severity,
                    description=finding_desc,
                    details={"check_data": data, "rule_evaluated": check.check_function},
                    timestamp=datetime.now(),
                    remediation_steps=["Review the flagged issue", "Take corrective action", "Re-run compliance check"],
                    status="open"
                )
                
                self.alert_manager.create_alert(finding)
        
        logger.info(f"Compliance check {check_id} completed with status: {new_status.value}")
        return new_status
        
    def run_standard_compliance_check(
        self, 
        standard: ComplianceStandard, 
        data: Dict[str, Any]
    ) -> Dict[str, ComplianceStatus]:
        """Run all compliance checks for a specific standard."""
        results = {}
        
        if standard not in self.compliance_standards:
            raise ValueError(f"Unknown compliance standard: {standard}")
            
        requirements = self.compliance_standards[standard]
        
        for requirement in requirements:
            # Find the appropriate check for this requirement
            check_id = self._find_check_for_requirement(requirement)
            if check_id:
                status = self.run_compliance_check(check_id, data)
                results[requirement.value] = status
            else:
                # If no specific check exists, try to run a generic check
                # based on the rule engine
                rule_id = self._get_default_rule_for_requirement(requirement)
                if rule_id:
                    is_compliant, findings = self.rule_engine.evaluate_rule(rule_id, data)
                    status = ComplianceStatus.COMPLIANT if is_compliant else ComplianceStatus.NON_COMPLIANT
                    results[requirement.value] = status
                    
                    if not is_compliant:
                        # Create a generic finding
                        finding = ComplianceFinding(
                            finding_id=f"finding_{secrets.token_hex(8)}",
                            check_id="generic_check",
                            severity="medium",
                            description=f"Non-compliance with {requirement.value}: {findings[0] if findings else 'General compliance issue'}",
                            details={"check_data": data, "requirement": requirement.value},
                            timestamp=datetime.now(),
                            remediation_steps=["Review compliance requirement", "Address identified issues"],
                            status="open"
                        )
                        self.alert_manager.create_alert(finding)
        
        return results
        
    def _find_check_for_requirement(self, requirement: ComplianceRequirement) -> Optional[str]:
        """Find a registered check for a specific requirement."""
        for check_id, check in self.compliance_checks.items():
            if check.requirement == requirement:
                return check_id
        return None
        
    def _get_default_rule_for_requirement(self, requirement: ComplianceRequirement) -> Optional[str]:
        """Get the default rule ID for a requirement."""
        # Map requirements to default rules
        requirement_to_rule = {
            ComplianceRequirement.LAWFUL_PROCESSING: "gdpr_consent_exists",
            ComplianceRequirement.CONSENT_MANAGEMENT: "gdpr_consent_exists",
            ComplianceRequirement.DATA_MINIMIZATION: "gdpr_data_minimization",  # Note: This isn't in the enum but used in rule
            ComplianceRequirement.RIGHT_TO_ERASURE: "gdpr_right_to_erasure",
            ComplianceRequirement.PRIVACY_RULE: "hipaa_access_logs",
            ComplianceRequirement.SECURITY_RULE: "hipaa_encryption",
            ComplianceRequirement.BREACH_NOTIFICATION: "hipaa_breach_procedures",
            ComplianceRequirement.INTERNAL_CONTROLS: "sox_financial_controls",
            ComplianceRequirement.DOCUMENTATION_MAINTENANCE: "sox_documentation"
        }
        
        # Handle special case for data minimization
        if "DATA_MINIMIZATION" in requirement.value:
            return "gdpr_data_minimization"
            
        return requirement_to_rule.get(requirement)
        
    def generate_compliance_report(
        self, 
        standard: ComplianceStandard, 
        start_date: datetime = None,
        end_date: datetime = None
    ) -> ComplianceReport:
        """Generate a compliance report."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
            
        report_id = f"report_{secrets.token_hex(8)}"
        
        # Run compliance checks for the standard
        check_results = self.run_standard_compliance_check(standard, {})
        
        # Count findings by severity
        findings_summary = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for finding in self.alert_manager.alerts:
            if finding.status == "open":
                findings_summary[finding.severity] = findings_summary.get(finding.severity, 0) + 1
                
        # Determine overall status
        non_compliant_count = sum(1 for status in check_results.values() if status == ComplianceStatus.NON_COMPLIANT)
        total_checks = len(check_results)
        
        if non_compliant_count == 0:
            overall_status = ComplianceStatus.COMPLIANT
        elif non_compliant_count == total_checks:
            overall_status = ComplianceStatus.NON_COMPLIANT
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT  # Partial non-compliance is still non-compliance
            
        report = ComplianceReport(
            report_id=report_id,
            standard=standard,
            generated_at=datetime.now(),
            coverage_period_start=start_date,
            coverage_period_end=end_date,
            checks_performed=list(check_results.keys()),
            overall_status=overall_status,
            findings_summary=findings_summary,
            recommendations=[
                "Address all high and critical severity findings immediately",
                "Review and strengthen compliance controls",
                "Schedule regular compliance audits"
            ],
            metadata={
                "checks_run": len(check_results),
                "non_compliant_checks": non_compliant_count
            }
        )
        
        logger.info(f"Generated compliance report: {report_id} for {standard.value}")
        return report
        
    def log_compliance_action(
        self, 
        user_id: str, 
        action: str, 
        resource: str, 
        details: Dict[str, Any] = None,
        compliance_requirements: List[ComplianceRequirement] = None
    ) -> str:
        """Log a compliance-relevant action to the audit trail."""
        return self.audit_trail.log_action(
            user_id, action, resource, details, compliance_requirements
        )
        
    def get_compliance_status(self, standard: ComplianceStandard) -> Dict[str, Any]:
        """Get the current compliance status for a standard."""
        # This would typically check the most recent compliance assessments
        # For this implementation, we'll return a summary based on alerts
        
        all_findings = self.alert_manager.get_unresolved_alerts()
        standard_findings = [f for f in all_findings if f.check_id.startswith(standard.value)]
        
        # Count by severity
        severity_counts = {}
        for finding in standard_findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            
        return {
            "standard": standard.value,
            "last_assessment": max((f.timestamp for f in all_findings), default=None),
            "open_findings_count": len(standard_findings),
            "findings_by_severity": severity_counts,
            "compliance_status": ComplianceStatus.NON_COMPLIANT if standard_findings else ComplianceStatus.COMPLIANT
        }
        
    def subscribe_to_alerts(self, email: str, requirement: ComplianceRequirement):
        """Subscribe an email to compliance alerts for a requirement."""
        self.alert_manager.subscribe_to_requirement(email, requirement)
        
    def get_compliance_metrics(self) -> Dict[str, Any]:
        """Get overall compliance metrics."""
        all_findings = self.alert_manager.get_unresolved_alerts()
        
        # Count by severity
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for finding in all_findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            
        # Count by standard
        standard_counts = {}
        for finding in all_findings:
            # Extract standard from check_id or other metadata
            # This is a simplification - in reality, you'd have better categorization
            standard_counts["unknown"] = standard_counts.get("unknown", 0) + 1
            
        return {
            "total_open_findings": len(all_findings),
            "findings_by_severity": severity_counts,
            "findings_by_standard": standard_counts,
            "total_audit_entries": len(self.audit_trail.audit_entries),
            "active_subscribers": len(set(email for emails in self.alert_manager.subscribers.values() for email in emails))
        }


# Decorator for automatically logging compliance-relevant actions
def compliance_log(
    action: str, 
    resource: str, 
    requirements: List[ComplianceRequirement] = None
):
    """
    Decorator to automatically log compliance-relevant actions.
    
    Args:
        action: Description of the action being performed
        resource: Resource being acted upon
        requirements: List of compliance requirements this action relates to
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get the compliance manager from the instance or as a parameter
            compliance_mgr = None
            
            # Try to get from instance (first arg is usually self)
            if args and hasattr(args[0], 'compliance_manager'):
                compliance_mgr = args[0].compliance_manager
            elif 'compliance_manager' in kwargs:
                compliance_mgr = kwargs['compliance_manager']
            
            result = await func(*args, **kwargs)
            
            # Log the action if we have a compliance manager
            if compliance_mgr and hasattr(compliance_mgr, 'log_compliance_action'):
                # Extract user info from args/kwargs - this is application-specific
                user_id = getattr(args[0], 'user_id', 'unknown') if args else 'unknown'
                
                try:
                    compliance_mgr.log_compliance_action(
                        user_id=user_id,
                        action=action,
                        resource=resource,
                        details={"function": func.__name__, "args": str(args[1:]) if len(args) > 1 else [], "kwargs": str(kwargs)},
                        compliance_requirements=requirements
                    )
                except Exception as e:
                    logger.error(f"Failed to log compliance action: {e}")
                    
            return result
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get the compliance manager from the instance or as a parameter
            compliance_mgr = None
            
            # Try to get from instance (first arg is usually self)
            if args and hasattr(args[0], 'compliance_manager'):
                compliance_mgr = args[0].compliance_manager
            elif 'compliance_manager' in kwargs:
                compliance_mgr = kwargs['compliance_manager']
            
            result = func(*args, **kwargs)
            
            # Log the action if we have a compliance manager
            if compliance_mgr and hasattr(compliance_mgr, 'log_compliance_action'):
                # Extract user info from args/kwargs - this is application-specific
                user_id = getattr(args[0], 'user_id', 'unknown') if args else 'unknown'
                
                try:
                    compliance_mgr.log_compliance_action(
                        user_id=user_id,
                        action=action,
                        resource=resource,
                        details={"function": func.__name__, "args": str(args[1:]) if len(args) > 1 else [], "kwargs": str(kwargs)},
                        compliance_requirements=requirements
                    )
                except Exception as e:
                    logger.error(f"Failed to log compliance action: {e}")
                    
            return result
            
        # Return the appropriate wrapper based on whether the original function is async
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Convenience function for easy use
def create_compliance_manager() -> ComplianceManager:
    """
    Convenience function to create a compliance manager.
    
    Returns:
        ComplianceManager instance
    """
    return ComplianceManager()