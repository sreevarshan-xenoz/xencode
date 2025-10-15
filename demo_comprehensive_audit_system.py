#!/usr/bin/env python3
"""
Demo: Comprehensive Audit System

Demonstrates the tamper-proof audit logging, security event correlation,
and compliance reporting capabilities of the enhanced audit system.
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.tree import Tree

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from xencode.audit.audit_logger import (
    AuditLogger, AuditEvent, AuditEventType, AuditSeverity
)
from xencode.audit.security_correlator import (
    SecurityEventCorrelator, IncidentType, IncidentSeverity
)
from xencode.audit.compliance_reporter import (
    ComplianceReporter, ComplianceFramework, ComplianceStatus
)


async def demo_comprehensive_audit_system():
    """Demonstrate comprehensive audit system capabilities"""
    
    console = Console()
    console.print("🛡️ [bold cyan]Comprehensive Audit System Demo[/bold cyan]\n")
    
    # Initialize components
    console.print("🔧 [bold yellow]Initializing Audit System Components[/bold yellow]")
    
    audit_logger = AuditLogger(enable_real_time=False, enable_encryption=True)
    correlator = SecurityEventCorrelator()
    reporter = ComplianceReporter()
    
    console.print("✅ Audit Logger initialized with tamper-proof capabilities")
    console.print("✅ Security Event Correlator loaded with detection rules")
    console.print("✅ Compliance Reporter configured for GDPR and SOX\n")
    
    # Demo 1: Tamper-Proof Audit Logging
    console.print("📝 [bold green]Demo 1: Tamper-Proof Audit Logging[/bold green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating audit events...", total=None)
        
        # Simulate various user activities
        activities = [
            (AuditEventType.SYSTEM_START, "system", "System startup"),
            (AuditEventType.LOGIN_SUCCESS, "alice", "User login"),
            (AuditEventType.DATA_ACCESS, "alice", "Accessing customer records"),
            (AuditEventType.CONFIGURATION_CHANGE, "admin", "Updated security settings"),
            (AuditEventType.SECURITY_SCAN, "security_team", "Vulnerability scan"),
            (AuditEventType.DATA_EXPORT, "bob", "Exported sales data"),
            (AuditEventType.PERMISSION_CHANGE, "admin", "Modified user permissions"),
            (AuditEventType.LOGOUT, "alice", "User logout"),
        ]
        
        event_ids = []
        for event_type, user_id, description in activities:
            event_id = audit_logger.log_event(
                event_type,
                severity=AuditSeverity.INFO,
                user_id=user_id,
                source_ip=f"192.168.1.{hash(user_id) % 255}",
                action=description,
                details={"demo": True, "timestamp": time.time()}
            )
            event_ids.append(event_id)
        
        progress.update(task, completed=True)
    
    # Display audit events
    events = audit_logger.get_events(limit=10)
    
    events_table = Table(title="Recent Audit Events")
    events_table.add_column("Event Type", style="cyan")
    events_table.add_column("User", style="green")
    events_table.add_column("Timestamp", style="yellow")
    events_table.add_column("Integrity", style="magenta")
    
    for event in events[:8]:  # Show first 8 events
        integrity_status = "✅ Valid" if event.verify_integrity() else "❌ Tampered"
        events_table.add_row(
            event.event_type.value,
            event.user_id or "system",
            event.timestamp.strftime("%H:%M:%S"),
            integrity_status
        )
    
    console.print(events_table)
    console.print()
    
    # Verify audit chain integrity
    integrity_ok = audit_logger.verify_integrity()
    integrity_panel = Panel(
        f"🔒 Cryptographic Chain Integrity: {'✅ VERIFIED' if integrity_ok else '❌ COMPROMISED'}",
        border_style="green" if integrity_ok else "red"
    )
    console.print(integrity_panel)
    console.print()
    
    # Demo 2: Security Event Correlation
    console.print("🚨 [bold red]Demo 2: Security Event Correlation[/bold red]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Simulating security incidents...", total=None)
        
        # Simulate brute force attack
        attacker_ip = "10.0.0.100"
        for i in range(7):  # Above threshold
            audit_logger.log_event(
                AuditEventType.LOGIN_FAILURE,
                severity=AuditSeverity.WARNING,
                user_id=f"target_user_{i % 3}",
                source_ip=attacker_ip,
                success=False,
                error_message="Invalid credentials",
                details={"attempt": i + 1, "attack_type": "brute_force"}
            )
        
        # Simulate privilege escalation
        audit_logger.log_event(
            AuditEventType.ROLE_ASSIGNMENT,
            severity=AuditSeverity.INFO,
            user_id="suspicious_user",
            action="assign_admin_role",
            details={"role": "administrator", "assigned_by": "hr_system"}
        )
        
        # Suspicious privileged actions
        for action in ["delete_audit_logs", "modify_security_policy", "create_backdoor_user"]:
            audit_logger.log_event(
                AuditEventType.CONFIGURATION_CHANGE,
                severity=AuditSeverity.HIGH,
                user_id="suspicious_user",
                action=action,
                success=True,
                details={"privilege_level": "admin", "suspicious": True}
            )
        
        # Simulate data exfiltration
        for i in range(5):  # Above threshold
            audit_logger.log_event(
                AuditEventType.DATA_EXPORT,
                severity=AuditSeverity.MEDIUM,
                user_id="insider_threat",
                resource=f"sensitive_database_table_{i}",
                success=True,
                details={"export_size_mb": 150 + i * 50, "destination": "external_drive"}
            )
        
        progress.update(task, completed=True)
    
    # Analyze for security incidents
    all_events = audit_logger.get_events(limit=50)
    incidents = correlator.analyze_events(all_events)
    
    console.print(f"🔍 Detected {len(incidents)} security incidents:")
    
    incidents_table = Table()
    incidents_table.add_column("Incident Type", style="red")
    incidents_table.add_column("Severity", style="yellow")
    incidents_table.add_column("Title", style="cyan")
    incidents_table.add_column("Risk Score", style="magenta")
    
    for incident in incidents:
        severity_color = {
            IncidentSeverity.CRITICAL: "[red]CRITICAL[/red]",
            IncidentSeverity.HIGH: "[orange1]HIGH[/orange1]",
            IncidentSeverity.MEDIUM: "[yellow]MEDIUM[/yellow]",
            IncidentSeverity.LOW: "[green]LOW[/green]"
        }.get(incident.severity, incident.severity.value)
        
        incidents_table.add_row(
            incident.incident_type.value.replace('_', ' ').title(),
            severity_color,
            incident.title,
            f"{incident.risk_score:.1f}"
        )
    
    console.print(incidents_table)
    console.print()
    
    # Display incident details
    if incidents:
        console.print("📋 [bold blue]Incident Analysis Details[/bold blue]")
        
        for i, incident in enumerate(incidents[:2], 1):  # Show first 2 incidents
            incident_panel = Panel(
                f"**Type:** {incident.incident_type.value}\n"
                f"**Severity:** {incident.severity.value}\n"
                f"**Confidence:** {incident.confidence_score:.1%}\n"
                f"**Affected Users:** {', '.join(incident.affected_users) if incident.affected_users else 'None'}\n"
                f"**Source IPs:** {', '.join(incident.source_ips) if incident.source_ips else 'None'}\n"
                f"**Indicators:** {len(incident.indicators)} detected\n"
                f"**Recommendations:** {len(incident.recommendations)} actions",
                title=f"🚨 Incident {i}: {incident.title}",
                border_style="red" if incident.severity == IncidentSeverity.CRITICAL else "yellow"
            )
            console.print(incident_panel)
        
        console.print()
    
    # Demo 3: Compliance Reporting
    console.print("📊 [bold purple]Demo 3: Compliance Reporting[/bold purple]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating compliance reports...", total=None)
        
        # Generate GDPR compliance report
        gdpr_report = reporter.generate_report(
            ComplianceFramework.GDPR,
            all_events,
            incidents,
            period_start=datetime.now(timezone.utc) - timedelta(days=30),
            period_end=datetime.now(timezone.utc)
        )
        
        progress.update(task, completed=True)
    
    # Display compliance summary
    compliance_table = Table(title="GDPR Compliance Report Summary")
    compliance_table.add_column("Metric", style="cyan")
    compliance_table.add_column("Value", style="green")
    compliance_table.add_column("Status", style="yellow")
    
    status_color = {
        ComplianceStatus.COMPLIANT: "[green]✅ COMPLIANT[/green]",
        ComplianceStatus.PARTIALLY_COMPLIANT: "[yellow]⚠️ PARTIALLY COMPLIANT[/yellow]",
        ComplianceStatus.NON_COMPLIANT: "[red]❌ NON-COMPLIANT[/red]",
        ComplianceStatus.UNKNOWN: "[gray]❓ UNKNOWN[/gray]"
    }.get(gdpr_report.overall_status, gdpr_report.overall_status.value)
    
    compliance_table.add_row("Overall Status", gdpr_report.overall_status.value.replace('_', ' ').title(), status_color)
    compliance_table.add_row("Compliance Score", f"{gdpr_report.compliance_score:.1%}", "")
    compliance_table.add_row("Total Requirements", str(gdpr_report.total_requirements), "")
    compliance_table.add_row("Compliant", str(gdpr_report.compliant_requirements), "[green]✅[/green]")
    compliance_table.add_row("Non-Compliant", str(gdpr_report.non_compliant_requirements), "[red]❌[/red]")
    compliance_table.add_row("Audited Events", str(gdpr_report.audited_events), "")
    compliance_table.add_row("Security Incidents", str(len(gdpr_report.security_incidents)), "[red]🚨[/red]" if gdpr_report.security_incidents else "[green]✅[/green]")
    
    console.print(compliance_table)
    console.print()
    
    # Display requirement details
    console.print("📋 [bold blue]Compliance Requirement Details[/bold blue]")
    
    requirements_table = Table()
    requirements_table.add_column("Requirement ID", style="cyan")
    requirements_table.add_column("Status", style="yellow")
    requirements_table.add_column("Score", style="green")
    requirements_table.add_column("Issues", style="red")
    
    for check in gdpr_report.requirement_checks[:5]:  # Show first 5
        status_emoji = {
            ComplianceStatus.COMPLIANT: "✅",
            ComplianceStatus.PARTIALLY_COMPLIANT: "⚠️",
            ComplianceStatus.NON_COMPLIANT: "❌",
            ComplianceStatus.UNKNOWN: "❓"
        }.get(check.status, "❓")
        
        requirements_table.add_row(
            check.requirement_id,
            f"{status_emoji} {check.status.value.replace('_', ' ').title()}",
            f"{check.score:.1%}",
            str(len(check.violations))
        )
    
    console.print(requirements_table)
    console.print()
    
    # Demo 4: Report Export
    console.print("📤 [bold green]Demo 4: Report Export Capabilities[/bold green]")
    
    # Export in different formats
    json_report = reporter.export_report(gdpr_report, 'json')
    html_report = reporter.export_report(gdpr_report, 'html')
    csv_report = reporter.export_report(gdpr_report, 'csv')
    
    export_table = Table(title="Report Export Formats")
    export_table.add_column("Format", style="cyan")
    export_table.add_column("Size", style="green")
    export_table.add_column("Features", style="yellow")
    
    export_table.add_row(
        "JSON", 
        f"{len(json_report):,} chars",
        "Machine-readable, API integration"
    )
    export_table.add_row(
        "HTML", 
        f"{len(html_report):,} chars",
        "Human-readable, web display"
    )
    export_table.add_row(
        "CSV", 
        f"{len(csv_report):,} chars",
        "Spreadsheet import, data analysis"
    )
    
    console.print(export_table)
    console.print()
    
    # Show sample JSON structure
    console.print("📄 [bold cyan]Sample JSON Report Structure[/bold cyan]")
    
    import json
    json_sample = json.loads(json_report)
    
    # Create a tree view of the JSON structure
    tree = Tree("📊 GDPR Compliance Report")
    tree.add(f"🆔 ID: {json_sample['id']}")
    tree.add(f"🏛️ Framework: {json_sample['framework'].upper()}")
    tree.add(f"📅 Generated: {json_sample['generated_at'][:19]}")
    tree.add(f"📈 Score: {json_sample['compliance_score']:.1%}")
    
    summary_branch = tree.add("📋 Summary")
    summary_branch.add(f"Total Requirements: {json_sample['summary']['total_requirements']}")
    summary_branch.add(f"Compliant: {json_sample['summary']['compliant_requirements']}")
    summary_branch.add(f"Non-Compliant: {json_sample['summary']['non_compliant_requirements']}")
    
    checks_branch = tree.add(f"✅ Requirement Checks ({len(json_sample['requirement_checks'])})")
    for check in json_sample['requirement_checks'][:3]:
        checks_branch.add(f"{check['requirement_id']}: {check['status']} ({check['score']:.1%})")
    
    if json_sample['security_incidents']:
        incidents_branch = tree.add(f"🚨 Security Incidents ({len(json_sample['security_incidents'])})")
        for incident in json_sample['security_incidents'][:3]:
            incidents_branch.add(f"{incident['type']}: {incident['severity']}")
    
    console.print(tree)
    console.print()
    
    # Demo 5: System Statistics
    console.print("📊 [bold magenta]Demo 5: System Statistics[/bold magenta]")
    
    # Get correlator statistics
    correlator_stats = correlator.get_incident_statistics()
    
    stats_table = Table(title="Audit System Statistics")
    stats_table.add_column("Component", style="cyan")
    stats_table.add_column("Metric", style="yellow")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Audit Logger", "Total Events Logged", str(len(all_events)))
    stats_table.add_row("Audit Logger", "Chain Integrity", "✅ Verified" if integrity_ok else "❌ Compromised")
    stats_table.add_row("Security Correlator", "Total Incidents", str(correlator_stats.get('total_incidents', 0)))
    stats_table.add_row("Security Correlator", "Active Rules", str(correlator_stats.get('rules_enabled', 0)))
    stats_table.add_row("Security Correlator", "Average Risk Score", f"{correlator_stats.get('average_risk_score', 0):.2f}")
    stats_table.add_row("Compliance Reporter", "Frameworks Supported", "GDPR, SOX, HIPAA, PCI-DSS")
    stats_table.add_row("Compliance Reporter", "Export Formats", "JSON, HTML, CSV")
    
    console.print(stats_table)
    console.print()
    
    # Demo 6: Security Recommendations
    console.print("💡 [bold yellow]Demo 6: Security Recommendations[/bold yellow]")
    
    recommendations = [
        "🔒 Implement account lockout policies to prevent brute force attacks",
        "👥 Establish approval workflows for privilege escalation",
        "📊 Set up real-time monitoring for data export activities",
        "🔍 Regular security audits and penetration testing",
        "📋 Automated compliance checking and reporting",
        "🚨 Incident response procedures and escalation paths",
        "🔐 Multi-factor authentication for administrative accounts",
        "📝 Comprehensive audit logging for all system activities",
        "🛡️ Network segmentation and access controls",
        "📚 Security awareness training for all users"
    ]
    
    for i, recommendation in enumerate(recommendations, 1):
        console.print(f"   {i:2d}. {recommendation}")
    
    console.print()
    
    # Summary
    console.print("🎯 [bold green]Demo Summary[/bold green]")
    
    summary_panel = Panel(
        f"✅ **Tamper-Proof Logging**: {len(all_events)} events with cryptographic integrity\n"
        f"🚨 **Security Correlation**: {len(incidents)} incidents detected across {len(correlator.rules)} rules\n"
        f"📊 **Compliance Reporting**: GDPR compliance at {gdpr_report.compliance_score:.1%}\n"
        f"📤 **Multi-Format Export**: JSON, HTML, and CSV reports generated\n"
        f"🔒 **Enterprise Ready**: Tamper detection, audit trails, and regulatory compliance\n\n"
        f"The comprehensive audit system provides enterprise-grade security monitoring,\n"
        f"incident detection, and compliance reporting with cryptographic integrity.",
        title="🛡️ Comprehensive Audit System Capabilities",
        border_style="green"
    )
    
    console.print(summary_panel)
    console.print()
    
    console.print("🎉 [bold cyan]Comprehensive Audit System Demo Complete![/bold cyan]")
    console.print("The system successfully demonstrated tamper-proof logging, security correlation,")
    console.print("and compliance reporting capabilities for enterprise security requirements.")


if __name__ == "__main__":
    asyncio.run(demo_comprehensive_audit_system())