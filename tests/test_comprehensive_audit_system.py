#!/usr/bin/env python3
"""
Tests for Comprehensive Audit System

Tests the tamper-proof audit logging, security event correlation,
and compliance reporting functionality.
"""

import asyncio
import json
import tempfile
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from xencode.audit.audit_logger import (
    AuditLogger, AuditEvent, AuditEventType, AuditSeverity,
    AuditChain, AuditEncryption, AuditStorage
)
from xencode.audit.security_correlator import (
    SecurityEventCorrelator, SecurityIncident, IncidentType, IncidentSeverity,
    BruteForceRule, PrivilegeEscalationRule, DataExfiltrationRule
)
from xencode.audit.compliance_reporter import (
    ComplianceReporter, ComplianceFramework, ComplianceStatus,
    ComplianceRequirement, GDPRRequirements
)


class TestAuditEvent:
    """Test audit event functionality"""
    
    def test_event_creation(self):
        """Test creating an audit event"""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS,
            severity=AuditSeverity.INFO,
            user_id="test_user",
            source_ip="192.168.1.1",
            success=True
        )
        
        assert event.event_type == AuditEventType.LOGIN_SUCCESS
        assert event.severity == AuditSeverity.INFO
        assert event.user_id == "test_user"
        assert event.source_ip == "192.168.1.1"
        assert event.success is True
        assert event.checksum is not None
        assert len(event.checksum) == 64  # SHA-256 hex
    
    def test_event_integrity_verification(self):
        """Test event integrity verification"""
        event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            user_id="test_user"
        )
        
        # Event should verify correctly
        assert event.verify_integrity() is True
        
        # Tamper with event
        event.user_id = "tampered_user"
        
        # Should fail verification
        assert event.verify_integrity() is False
    
    def test_event_serialization(self):
        """Test event to/from dict conversion"""
        original_event = AuditEvent(
            event_type=AuditEventType.SECURITY_SCAN,
            severity=AuditSeverity.HIGH,
            details={"scan_type": "vulnerability", "findings": 5}
        )
        
        # Convert to dict and back
        event_dict = original_event.to_dict()
        restored_event = AuditEvent.from_dict(event_dict)
        
        assert restored_event.event_type == original_event.event_type
        assert restored_event.severity == original_event.severity
        assert restored_event.details == original_event.details
        assert restored_event.checksum == original_event.checksum


class TestAuditChain:
    """Test audit chain functionality"""
    
    def test_chain_creation_and_linking(self):
        """Test creating and linking audit events in chain"""
        chain = AuditChain()
        
        # Add first event
        event1 = AuditEvent(event_type=AuditEventType.SYSTEM_START)
        chain.add_event(event1)
        
        assert len(chain.chain) == 1
        assert event1.previous_hash is None
        assert chain.last_hash == event1.checksum
        
        # Add second event
        event2 = AuditEvent(event_type=AuditEventType.LOGIN_SUCCESS)
        chain.add_event(event2)
        
        assert len(chain.chain) == 2
        assert event2.previous_hash == event1.checksum
        assert chain.last_hash == event2.checksum
    
    def test_chain_integrity_verification(self):
        """Test chain integrity verification"""
        chain = AuditChain()
        
        # Add multiple events
        for i in range(5):
            event = AuditEvent(
                event_type=AuditEventType.DATA_ACCESS,
                user_id=f"user_{i}"
            )
            chain.add_event(event)
        
        # Chain should verify correctly
        assert chain.verify_chain() is True
        
        # Tamper with middle event
        chain.chain[2].user_id = "tampered_user"
        
        # Chain should fail verification
        assert chain.verify_chain() is False


class TestAuditEncryption:
    """Test audit encryption and signing"""
    
    def test_key_generation(self):
        """Test RSA key pair generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_path = Path(temp_dir) / "test_keys"
            encryption = AuditEncryption(key_path)
            
            assert encryption.private_key_path.exists()
            assert encryption.public_key_path.exists()
    
    def test_event_signing_and_verification(self):
        """Test event signing and signature verification"""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_path = Path(temp_dir) / "test_keys"
            encryption = AuditEncryption(key_path)
            
            event = AuditEvent(
                event_type=AuditEventType.SECURITY_INCIDENT,
                severity=AuditSeverity.CRITICAL
            )
            
            # Sign event
            signature = encryption.sign_event(event)
            event.signature = signature
            
            assert signature is not None
            assert len(signature) > 0
            
            # Verify signature
            assert encryption.verify_signature(event) is True
            
            # Tamper with event
            event.severity = AuditSeverity.LOW
            
            # Signature should fail verification
            assert encryption.verify_signature(event) is False


class TestAuditStorage:
    """Test audit storage functionality"""
    
    def test_database_initialization(self):
        """Test database initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_audit.db"
            storage = AuditStorage(db_path)
            
            assert db_path.exists()
    
    def test_event_storage_and_retrieval(self):
        """Test storing and retrieving audit events"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_audit.db"
            storage = AuditStorage(db_path)
            
            # Create test events
            events = []
            for i in range(5):
                event = AuditEvent(
                    event_type=AuditEventType.DATA_ACCESS,
                    user_id=f"user_{i}",
                    resource=f"resource_{i}"
                )
                events.append(event)
                
                # Store event
                assert storage.store_event(event) is True
            
            # Retrieve all events
            retrieved_events = storage.get_events()
            assert len(retrieved_events) == 5
            
            # Test filtering by user
            user_events = storage.get_events(user_id="user_1")
            assert len(user_events) == 1
            assert user_events[0].user_id == "user_1"
            
            # Test filtering by event type
            access_events = storage.get_events(event_types=[AuditEventType.DATA_ACCESS])
            assert len(access_events) == 5
    
    def test_database_integrity_verification(self):
        """Test database integrity verification"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_audit.db"
            storage = AuditStorage(db_path)
            
            # Add events in order
            for i in range(3):
                event = AuditEvent(
                    event_type=AuditEventType.LOGIN_SUCCESS,
                    user_id=f"user_{i}"
                )
                storage.store_event(event)
            
            # Verify integrity
            assert storage.verify_database_integrity() is True


class TestAuditLogger:
    """Test main audit logger functionality"""
    
    @pytest.mark.asyncio
    async def test_logger_initialization(self):
        """Test audit logger initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "test_audit.db"
            logger = AuditLogger(storage_path, enable_real_time=False)
            
            assert logger.storage is not None
            assert logger.encryption is not None
            assert logger.chain is not None
    
    def test_event_logging(self):
        """Test logging audit events"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "test_audit.db"
            logger = AuditLogger(storage_path, enable_real_time=False)
            
            # Log an event
            event_id = logger.log_event(
                AuditEventType.LOGIN_SUCCESS,
                severity=AuditSeverity.INFO,
                user_id="test_user",
                source_ip="192.168.1.1"
            )
            
            assert event_id is not None
            
            # Retrieve events
            events = logger.get_events()
            assert len(events) == 1
            assert events[0].event_type == AuditEventType.LOGIN_SUCCESS
            assert events[0].user_id == "test_user"
    
    def test_integrity_verification(self):
        """Test audit system integrity verification"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "test_audit.db"
            logger = AuditLogger(storage_path, enable_real_time=False)
            
            # Log multiple events
            for i in range(5):
                logger.log_event(
                    AuditEventType.DATA_ACCESS,
                    user_id=f"user_{i}"
                )
            
            # Verify integrity
            assert logger.verify_integrity() is True


class TestSecurityEventCorrelator:
    """Test security event correlation"""
    
    def test_correlator_initialization(self):
        """Test correlator initialization"""
        correlator = SecurityEventCorrelator()
        
        assert len(correlator.rules) > 0
        assert all(rule.enabled for rule in correlator.rules)
    
    def test_brute_force_detection(self):
        """Test brute force attack detection"""
        correlator = SecurityEventCorrelator()
        
        # Create failed login events
        events = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(6):  # Above threshold
            event = AuditEvent(
                event_type=AuditEventType.LOGIN_FAILURE,
                timestamp=base_time + timedelta(seconds=i * 30),
                source_ip="192.168.1.100",
                user_id=f"user_{i % 2}",  # Alternate users
                success=False
            )
            events.append(event)
        
        # Analyze events
        incidents = correlator.analyze_events(events)
        
        # Should detect brute force
        brute_force_incidents = [
            i for i in incidents 
            if i.incident_type == IncidentType.BRUTE_FORCE_ATTACK
        ]
        assert len(brute_force_incidents) > 0
        
        incident = brute_force_incidents[0]
        assert "192.168.1.100" in incident.source_ips
        assert incident.severity == IncidentSeverity.HIGH
    
    def test_privilege_escalation_detection(self):
        """Test privilege escalation detection"""
        correlator = SecurityEventCorrelator()
        
        base_time = datetime.now(timezone.utc)
        events = []
        
        # Role assignment event
        role_event = AuditEvent(
            event_type=AuditEventType.ROLE_ASSIGNMENT,
            timestamp=base_time,
            user_id="test_user",
            action="assign_admin_role"
        )
        events.append(role_event)
        
        # Privileged actions shortly after
        for i in range(3):
            priv_event = AuditEvent(
                event_type=AuditEventType.CONFIGURATION_CHANGE,
                timestamp=base_time + timedelta(minutes=i * 10),
                user_id="test_user",
                action="modify_system_config",
                success=True
            )
            events.append(priv_event)
        
        # Analyze events
        incidents = correlator.analyze_events(events)
        
        # Should detect privilege escalation
        escalation_incidents = [
            i for i in incidents 
            if i.incident_type == IncidentType.PRIVILEGE_ESCALATION
        ]
        assert len(escalation_incidents) > 0
        
        incident = escalation_incidents[0]
        assert "test_user" in incident.affected_users
        assert incident.severity == IncidentSeverity.CRITICAL
    
    def test_data_exfiltration_detection(self):
        """Test data exfiltration detection"""
        correlator = SecurityEventCorrelator()
        
        base_time = datetime.now(timezone.utc)
        events = []
        
        # Multiple data exports
        for i in range(4):  # Above threshold
            export_event = AuditEvent(
                event_type=AuditEventType.DATA_EXPORT,
                timestamp=base_time + timedelta(minutes=i * 10),
                user_id="suspicious_user",
                resource=f"database_table_{i}",
                success=True
            )
            events.append(export_event)
        
        # Analyze events
        incidents = correlator.analyze_events(events)
        
        # Should detect data exfiltration
        exfiltration_incidents = [
            i for i in incidents 
            if i.incident_type == IncidentType.DATA_EXFILTRATION
        ]
        assert len(exfiltration_incidents) > 0
        
        incident = exfiltration_incidents[0]
        assert "suspicious_user" in incident.affected_users
        assert incident.severity == IncidentSeverity.HIGH
    
    def test_incident_management(self):
        """Test incident management functionality"""
        correlator = SecurityEventCorrelator()
        
        # Create a test incident
        incident = SecurityIncident(
            id="test_incident",
            incident_type=IncidentType.SUSPICIOUS_ACTIVITY,
            severity=IncidentSeverity.MEDIUM,
            title="Test Incident",
            description="Test incident for management",
            detected_at=datetime.now(timezone.utc),
            start_time=datetime.now(timezone.utc)
        )
        
        correlator.incidents.append(incident)
        
        # Test filtering
        medium_incidents = correlator.get_incidents(severity=IncidentSeverity.MEDIUM)
        assert len(medium_incidents) == 1
        assert medium_incidents[0].id == "test_incident"
        
        # Test status update
        correlator.update_incident_status("test_incident", "resolved", "False positive")
        assert incident.status == "resolved"
        assert incident.resolution == "False positive"
        assert incident.resolved_at is not None
    
    def test_incident_statistics(self):
        """Test incident statistics generation"""
        correlator = SecurityEventCorrelator()
        
        # Add test incidents
        incidents = [
            SecurityIncident(
                id=f"incident_{i}",
                incident_type=IncidentType.BRUTE_FORCE_ATTACK,
                severity=IncidentSeverity.HIGH if i % 2 == 0 else IncidentSeverity.MEDIUM,
                title=f"Test Incident {i}",
                description="Test incident",
                detected_at=datetime.now(timezone.utc),
                start_time=datetime.now(timezone.utc),
                risk_score=0.7
            )
            for i in range(5)
        ]
        
        correlator.incidents.extend(incidents)
        
        stats = correlator.get_incident_statistics()
        
        assert stats['total_incidents'] == 5
        assert stats['severity_distribution']['high'] == 3
        assert stats['severity_distribution']['medium'] == 2
        assert stats['type_distribution']['brute_force_attack'] == 5
        assert 'average_risk_score' in stats


class TestComplianceReporter:
    """Test compliance reporting functionality"""
    
    def test_reporter_initialization(self):
        """Test compliance reporter initialization"""
        reporter = ComplianceReporter()
        
        assert reporter.engine is not None
        assert ComplianceFramework.GDPR in reporter.engine.requirements
        assert len(reporter.engine.requirements[ComplianceFramework.GDPR]) > 0
    
    def test_gdpr_requirements_loading(self):
        """Test GDPR requirements loading"""
        requirements = GDPRRequirements.get_requirements()
        
        assert len(requirements) > 0
        
        # Check specific requirements
        art_30 = next((r for r in requirements if r.id == "gdpr_art_30"), None)
        assert art_30 is not None
        assert art_30.framework == ComplianceFramework.GDPR
        assert AuditEventType.DATA_ACCESS in art_30.required_events
    
    def test_compliance_check(self):
        """Test individual compliance requirement checking"""
        reporter = ComplianceReporter()
        
        # Create test requirement
        requirement = ComplianceRequirement(
            id="test_req",
            framework=ComplianceFramework.CUSTOM,
            title="Test Requirement",
            description="Test requirement for checking",
            category="Testing",
            required_events=[AuditEventType.DATA_ACCESS, AuditEventType.SECURITY_SCAN]
        )
        
        # Create matching events
        events = [
            AuditEvent(event_type=AuditEventType.DATA_ACCESS, user_id="test_user"),
            AuditEvent(event_type=AuditEventType.SECURITY_SCAN, user_id="test_user")
        ]
        
        # Check compliance
        check = reporter.engine.check_requirement(requirement, events, [])
        
        assert check.requirement_id == "test_req"
        assert check.status == ComplianceStatus.COMPLIANT
        assert check.score > 0.8
        assert check.evidence_count > 0
    
    def test_gdpr_compliance_report_generation(self):
        """Test GDPR compliance report generation"""
        reporter = ComplianceReporter()
        
        # Create test events covering GDPR requirements
        base_time = datetime.now(timezone.utc)
        events = [
            AuditEvent(
                event_type=AuditEventType.DATA_ACCESS,
                timestamp=base_time - timedelta(days=1),
                user_id="test_user",
                resource="customer_data"
            ),
            AuditEvent(
                event_type=AuditEventType.SECURITY_SCAN,
                timestamp=base_time - timedelta(days=2),
                user_id="security_team"
            ),
            AuditEvent(
                event_type=AuditEventType.CONFIGURATION_CHANGE,
                timestamp=base_time - timedelta(days=3),
                user_id="admin_user",
                action="update_privacy_settings"
            )
        ]
        
        # Generate report
        report = reporter.generate_report(
            ComplianceFramework.GDPR,
            events,
            [],
            period_start=base_time - timedelta(days=7),
            period_end=base_time
        )
        
        assert report.framework == ComplianceFramework.GDPR
        assert report.total_requirements > 0
        assert report.compliance_score >= 0.0
        assert report.overall_status in [
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.PARTIALLY_COMPLIANT,
            ComplianceStatus.NON_COMPLIANT
        ]
        assert len(report.requirement_checks) == report.total_requirements
    
    def test_report_export_json(self):
        """Test JSON report export"""
        reporter = ComplianceReporter()
        
        # Create minimal report
        events = [
            AuditEvent(event_type=AuditEventType.DATA_ACCESS, user_id="test_user")
        ]
        
        report = reporter.generate_report(ComplianceFramework.GDPR, events, [])
        
        # Export as JSON
        json_report = reporter.export_report(report, 'json')
        
        assert json_report is not None
        assert len(json_report) > 0
        
        # Verify it's valid JSON
        parsed = json.loads(json_report)
        assert parsed['framework'] == 'gdpr'
        assert 'compliance_score' in parsed
        assert 'requirement_checks' in parsed
    
    def test_report_export_html(self):
        """Test HTML report export"""
        reporter = ComplianceReporter()
        
        # Create minimal report
        events = [
            AuditEvent(event_type=AuditEventType.DATA_ACCESS, user_id="test_user")
        ]
        
        report = reporter.generate_report(ComplianceFramework.GDPR, events, [])
        
        # Export as HTML
        html_report = reporter.export_report(report, 'html')
        
        assert html_report is not None
        assert len(html_report) > 0
        assert '<html>' in html_report
        assert 'Compliance Report' in html_report
        assert 'GDPR' in html_report
    
    def test_report_export_csv(self):
        """Test CSV report export"""
        reporter = ComplianceReporter()
        
        # Create minimal report
        events = [
            AuditEvent(event_type=AuditEventType.DATA_ACCESS, user_id="test_user")
        ]
        
        report = reporter.generate_report(ComplianceFramework.GDPR, events, [])
        
        # Export as CSV
        csv_report = reporter.export_report(report, 'csv')
        
        assert csv_report is not None
        assert len(csv_report) > 0
        assert 'Requirement ID,Status,Score' in csv_report
        assert 'gdpr_' in csv_report  # Should contain GDPR requirement IDs


class TestIntegration:
    """Integration tests for the complete audit system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_audit_workflow(self):
        """Test complete audit workflow from logging to compliance reporting"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "integration_audit.db"
            
            # Initialize components
            audit_logger = AuditLogger(storage_path, enable_real_time=False)
            correlator = SecurityEventCorrelator()
            reporter = ComplianceReporter()
            
            # Simulate user activity with potential security issues
            base_time = datetime.now(timezone.utc)
            
            # Normal activity
            audit_logger.log_event(
                AuditEventType.LOGIN_SUCCESS,
                user_id="normal_user",
                source_ip="192.168.1.10"
            )
            
            audit_logger.log_event(
                AuditEventType.DATA_ACCESS,
                user_id="normal_user",
                resource="customer_records"
            )
            
            # Suspicious activity - multiple failed logins
            for i in range(6):
                audit_logger.log_event(
                    AuditEventType.LOGIN_FAILURE,
                    user_id="attacker",
                    source_ip="10.0.0.100",
                    success=False
                )
            
            # Data export activity
            for i in range(4):
                audit_logger.log_event(
                    AuditEventType.DATA_EXPORT,
                    user_id="insider_threat",
                    resource=f"sensitive_data_{i}"
                )
            
            # Security scan (compliance requirement)
            audit_logger.log_event(
                AuditEventType.SECURITY_SCAN,
                user_id="security_team",
                details={"scan_type": "vulnerability", "findings": 2}
            )
            
            # Get all logged events
            events = audit_logger.get_events()
            assert len(events) >= 12  # Should have all logged events
            
            # Verify audit integrity
            assert audit_logger.verify_integrity() is True
            
            # Analyze for security incidents
            incidents = correlator.analyze_events(events)
            
            # Should detect brute force and data exfiltration
            assert len(incidents) >= 2
            
            incident_types = {incident.incident_type for incident in incidents}
            assert IncidentType.BRUTE_FORCE_ATTACK in incident_types
            assert IncidentType.DATA_EXFILTRATION in incident_types
            
            # Generate compliance report
            report = reporter.generate_report(
                ComplianceFramework.GDPR,
                events,
                incidents
            )
            
            # Verify report
            assert report.framework == ComplianceFramework.GDPR
            assert report.audited_events > 0
            assert len(report.security_incidents) >= 2
            
            # Should have some compliance issues due to security incidents
            assert report.overall_status in [
                ComplianceStatus.PARTIALLY_COMPLIANT,
                ComplianceStatus.NON_COMPLIANT
            ]
            
            # Export report
            json_report = reporter.export_report(report, 'json')
            assert len(json_report) > 1000  # Should be substantial
            
            # Verify JSON structure
            parsed_report = json.loads(json_report)
            assert parsed_report['framework'] == 'gdpr'
            assert len(parsed_report['security_incidents']) >= 2
            assert 'requirement_checks' in parsed_report
    
    def test_performance_with_large_dataset(self):
        """Test system performance with larger datasets"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "performance_audit.db"
            
            # Initialize components
            audit_logger = AuditLogger(storage_path, enable_real_time=False)
            correlator = SecurityEventCorrelator()
            
            # Generate large number of events
            import time
            start_time = time.time()
            
            for i in range(1000):
                audit_logger.log_event(
                    AuditEventType.DATA_ACCESS,
                    user_id=f"user_{i % 100}",
                    resource=f"resource_{i % 50}",
                    source_ip=f"192.168.1.{i % 255}"
                )
            
            logging_time = time.time() - start_time
            
            # Should complete within reasonable time
            assert logging_time < 30  # 30 seconds for 1000 events
            
            # Retrieve events
            start_time = time.time()
            events = audit_logger.get_events(limit=1000)
            retrieval_time = time.time() - start_time
            
            assert len(events) == 1000
            assert retrieval_time < 5  # 5 seconds for retrieval
            
            # Analyze for incidents
            start_time = time.time()
            incidents = correlator.analyze_events(events)
            analysis_time = time.time() - start_time
            
            assert analysis_time < 10  # 10 seconds for analysis
            
            # Verify integrity
            start_time = time.time()
            integrity_ok = audit_logger.verify_integrity()
            integrity_time = time.time() - start_time
            
            assert integrity_ok is True
            assert integrity_time < 15  # 15 seconds for integrity check


if __name__ == '__main__':
    pytest.main([__file__])