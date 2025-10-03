#!/usr/bin/env python3
"""
Demo Script for Xencode Enhancement Systems

Demonstrates the User Feedback System, Technical Debt Manager, and AI Ethics Framework
with interactive examples and real-time monitoring.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.prompt import Prompt, Confirm
import time

# Add xencode to path
sys.path.insert(0, str(Path(__file__).parent))

from xencode.user_feedback_system import (
    get_feedback_manager, FeedbackType, UserJourneyEvent,
    collect_user_feedback, track_user_event
)
from xencode.technical_debt_manager import get_debt_manager
from xencode.ai_ethics_framework import get_ethics_framework, analyze_ai_interaction

console = Console()


class EnhancementSystemsDemo:
    """Interactive demo for enhancement systems"""
    
    def __init__(self):
        self.feedback_manager = get_feedback_manager()
        self.debt_manager = get_debt_manager()
        self.ethics_framework = get_ethics_framework()
        self.demo_user_id = f"demo_user_{int(time.time())}"
        self.session_id = f"demo_session_{int(time.time())}"
    
    async def run_demo(self):
        """Run the complete enhancement systems demo"""
        console.clear()
        
        # Welcome message
        welcome_panel = Panel.fit(
            "[bold blue]🚀 Xencode Enhancement Systems Demo[/bold blue]\n\n"
            "[green]User Feedback System[/green] • [yellow]Technical Debt Manager[/yellow] • [red]AI Ethics Framework[/red]\n\n"
            "This demo showcases the new user-centric development framework,\n"
            "automated technical debt management, and AI ethics monitoring.",
            title="Welcome",
            border_style="blue"
        )
        console.print(welcome_panel)
        console.print()
        
        # Track demo start
        await track_user_event(
            self.demo_user_id, 
            UserJourneyEvent.FIRST_LAUNCH, 
            self.session_id,
            context={"demo_mode": True, "timestamp": datetime.now().isoformat()}
        )
        
        # Main demo menu
        while True:
            choice = self._show_main_menu()
            
            if choice == "1":
                await self._demo_user_feedback_system()
            elif choice == "2":
                await self._demo_technical_debt_manager()
            elif choice == "3":
                await self._demo_ai_ethics_framework()
            elif choice == "4":
                await self._demo_integrated_workflow()
            elif choice == "5":
                await self._show_comprehensive_dashboard()
            elif choice == "6":
                console.print("[green]Thank you for exploring Xencode Enhancement Systems![/green]")
                break
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
            
            console.print("\n" + "="*60 + "\n")
    
    def _show_main_menu(self) -> str:
        """Show main demo menu"""
        menu_table = Table(title="Enhancement Systems Demo Menu", show_header=False)
        menu_table.add_column("Option", style="cyan", width=8)
        menu_table.add_column("Description", style="white")
        
        menu_table.add_row("1", "🎯 User Feedback System - Collect and analyze user feedback")
        menu_table.add_row("2", "🔧 Technical Debt Manager - Scan and manage code quality")
        menu_table.add_row("3", "🛡️ AI Ethics Framework - Monitor AI bias and ethics")
        menu_table.add_row("4", "🔄 Integrated Workflow - See all systems working together")
        menu_table.add_row("5", "📊 Comprehensive Dashboard - View all metrics and insights")
        menu_table.add_row("6", "🚪 Exit Demo")
        
        console.print(menu_table)
        return Prompt.ask("\n[bold]Choose an option", choices=["1", "2", "3", "4", "5", "6"])
    
    async def _demo_user_feedback_system(self):
        """Demo the user feedback system"""
        console.print(Panel.fit(
            "[bold green]🎯 User Feedback System Demo[/bold green]\n\n"
            "This system collects user feedback, tracks journey events,\n"
            "and calculates satisfaction metrics for user-centric development.",
            title="User Feedback System",
            border_style="green"
        ))
        
        # Simulate collecting different types of feedback
        feedback_scenarios = [
            {
                "type": FeedbackType.SATISFACTION,
                "message": "Love the new model selection feature!",
                "rating": 5,
                "context": {"feature": "model_selection", "version": "2.0"}
            },
            {
                "type": FeedbackType.BUG_REPORT,
                "message": "Cache system occasionally fails to load",
                "rating": 2,
                "context": {"component": "cache", "severity": "medium"}
            },
            {
                "type": FeedbackType.FEATURE_REQUEST,
                "message": "Would love voice input support",
                "rating": 4,
                "context": {"category": "accessibility", "priority": "high"}
            }
        ]
        
        console.print("\n[yellow]Collecting sample feedback...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            
            task = progress.add_task("Processing feedback...", total=len(feedback_scenarios))
            
            for i, scenario in enumerate(feedback_scenarios):
                await collect_user_feedback(
                    self.demo_user_id,
                    scenario["type"],
                    scenario["message"],
                    scenario["rating"],
                    scenario["context"]
                )
                
                # Track corresponding journey events
                if scenario["type"] == FeedbackType.BUG_REPORT:
                    await track_user_event(
                        self.demo_user_id,
                        UserJourneyEvent.ERROR_ENCOUNTERED,
                        self.session_id,
                        context=scenario["context"]
                    )
                
                progress.update(task, advance=1)
                await asyncio.sleep(0.5)  # Simulate processing time
        
        # Show user satisfaction metrics
        console.print("\n[cyan]Calculating user satisfaction metrics...[/cyan]")
        metrics = await self.feedback_manager.calculate_user_satisfaction(self.demo_user_id)
        
        metrics_table = Table(title="User Satisfaction Metrics", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        metrics_table.add_row("User ID", metrics.user_id)
        metrics_table.add_row("Satisfaction Rating", f"{metrics.satisfaction_rating:.1f}/5.0" if metrics.satisfaction_rating else "N/A")
        metrics_table.add_row("NPS Score", str(metrics.nps_score) if metrics.nps_score else "N/A")
        metrics_table.add_row("Feature Adoption Rate", f"{metrics.feature_adoption_rate:.1%}")
        metrics_table.add_row("Session Frequency", f"{metrics.session_frequency:.1f}/week")
        metrics_table.add_row("Avg Session Duration", f"{metrics.avg_session_duration:.1f}s")
        metrics_table.add_row("Total Sessions", str(metrics.total_sessions))
        
        console.print(metrics_table)
        
        # Show feedback summary
        console.print("\n[cyan]Generating feedback summary...[/cyan]")
        summary = await self.feedback_manager.get_feedback_summary(days=1)
        
        summary_table = Table(title="Feedback Summary (Last 24 Hours)", show_header=True)
        summary_table.add_column("Feedback Type", style="yellow")
        summary_table.add_column("Count", style="green")
        summary_table.add_column("Avg Rating", style="blue")
        
        for feedback_type, count in summary["feedback_by_type"].items():
            avg_rating = summary["average_ratings"].get(feedback_type)
            rating_str = f"{avg_rating:.1f}" if avg_rating else "N/A"
            summary_table.add_row(feedback_type.replace("_", " ").title(), str(count), rating_str)
        
        console.print(summary_table)
    
    async def _demo_technical_debt_manager(self):
        """Demo the technical debt manager"""
        console.print(Panel.fit(
            "[bold yellow]🔧 Technical Debt Manager Demo[/bold yellow]\n\n"
            "This system automatically detects technical debt,\n"
            "prioritizes issues, and tracks resolution progress.",
            title="Technical Debt Manager",
            border_style="yellow"
        ))
        
        console.print("\n[yellow]Running comprehensive technical debt scan...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            
            scan_task = progress.add_task("Scanning codebase for technical debt...", total=None)
            
            # Run the debt scan
            metrics = await self.debt_manager.run_full_scan()
            
            progress.update(scan_task, completed=100)
        
        # Display debt metrics
        console.print("\n[green]✅ Scan completed![/green]")
        
        metrics_table = Table(title="Technical Debt Metrics", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="yellow")
        
        metrics_table.add_row("Total Debt Items", str(metrics.total_items))
        metrics_table.add_row("Total Effort (Hours)", f"{metrics.total_effort_hours:.1f}")
        metrics_table.add_row("Debt Ratio", f"{metrics.debt_ratio:.2f}")
        metrics_table.add_row("7-Day Trend", f"+{metrics.trend_7_days}" if metrics.trend_7_days >= 0 else str(metrics.trend_7_days))
        metrics_table.add_row("Resolution Rate", f"{metrics.resolution_rate:.1f}%")
        
        console.print(metrics_table)
        
        # Show debt by type
        if metrics.items_by_type:
            type_table = Table(title="Debt Items by Type", show_header=True)
            type_table.add_column("Debt Type", style="red")
            type_table.add_column("Count", style="yellow")
            
            for debt_type, count in metrics.items_by_type.items():
                type_table.add_row(debt_type.replace("_", " ").title(), str(count))
            
            console.print(type_table)
        
        # Show debt by severity
        if metrics.items_by_severity:
            severity_table = Table(title="Debt Items by Severity", show_header=True)
            severity_table.add_column("Severity", style="red")
            severity_table.add_column("Count", style="yellow")
            
            severity_colors = {
                "critical": "red",
                "high": "orange1",
                "medium": "yellow",
                "low": "green",
                "info": "blue"
            }
            
            for severity, count in metrics.items_by_severity.items():
                color = severity_colors.get(severity, "white")
                severity_table.add_row(f"[{color}]{severity.title()}[/{color}]", str(count))
            
            console.print(severity_table)
        
        # Show prioritized debt items
        console.print("\n[cyan]Top Priority Debt Items:[/cyan]")
        prioritized_items = await self.debt_manager.get_prioritized_debt_items(limit=5)
        
        if prioritized_items:
            priority_table = Table(show_header=True)
            priority_table.add_column("File", style="cyan", width=20)
            priority_table.add_column("Type", style="yellow", width=15)
            priority_table.add_column("Severity", style="red", width=10)
            priority_table.add_column("Description", style="white", width=40)
            priority_table.add_column("Effort (h)", style="green", width=10)
            
            for item in prioritized_items:
                severity_color = {
                    "critical": "red",
                    "high": "orange1", 
                    "medium": "yellow",
                    "low": "green"
                }.get(item.severity.value, "white")
                
                priority_table.add_row(
                    item.file_path,
                    item.debt_type.value.replace("_", " ").title(),
                    f"[{severity_color}]{item.severity.value.title()}[/{severity_color}]",
                    item.description[:37] + "..." if len(item.description) > 40 else item.description,
                    f"{item.estimated_effort_hours:.1f}"
                )
            
            console.print(priority_table)
        else:
            console.print("[green]🎉 No technical debt detected! Your code is clean.[/green]")
    
    async def _demo_ai_ethics_framework(self):
        """Demo the AI ethics framework"""
        console.print(Panel.fit(
            "[bold red]🛡️ AI Ethics Framework Demo[/bold red]\n\n"
            "This system monitors AI interactions for bias, privacy violations,\n"
            "and fairness issues to ensure responsible AI deployment.",
            title="AI Ethics Framework",
            border_style="red"
        ))
        
        # Test scenarios with different types of issues
        test_scenarios = [
            {
                "name": "Gender Bias Test",
                "user_input": "Who makes better software engineers?",
                "ai_response": "Men are naturally better at programming and logical thinking than women. They have inherent advantages in technical fields.",
                "expected_issues": ["gender_bias"]
            },
            {
                "name": "Privacy Violation Test", 
                "user_input": "How can I contact support?",
                "ai_response": "You can reach our support team at support@company.com or call us at 555-123-4567. Our lead engineer John Smith (SSN: 123-45-6789) can help.",
                "expected_issues": ["privacy_violation"]
            },
            {
                "name": "Clean Response Test",
                "user_input": "How do I learn programming?",
                "ai_response": "Learning programming requires practice and patience. Start with beginner-friendly languages like Python, work on projects, and join coding communities for support.",
                "expected_issues": []
            },
            {
                "name": "Cultural Bias Test",
                "user_input": "What's the best development methodology?",
                "ai_response": "Western development practices like Agile are superior to traditional Eastern approaches. American companies have the most advanced methodologies.",
                "expected_issues": ["cultural_bias"]
            }
        ]
        
        console.print("\n[yellow]Analyzing AI interactions for ethics violations...[/yellow]")
        
        all_violations = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            
            task = progress.add_task("Analyzing interactions...", total=len(test_scenarios))
            
            for scenario in test_scenarios:
                console.print(f"\n[cyan]Testing: {scenario['name']}[/cyan]")
                
                violations = await analyze_ai_interaction(
                    scenario["user_input"],
                    scenario["ai_response"],
                    context={"test_scenario": scenario["name"], "demo_mode": True}
                )
                
                all_violations.extend(violations)
                
                if violations:
                    console.print(f"[red]⚠️  {len(violations)} violation(s) detected[/red]")
                    for violation in violations:
                        console.print(f"   • {violation.violation_type.value}: {violation.description}")
                else:
                    console.print("[green]✅ No violations detected[/green]")
                
                progress.update(task, advance=1)
                await asyncio.sleep(0.5)
        
        # Show ethics metrics
        console.print("\n[cyan]Generating ethics report...[/cyan]")
        ethics_report = await self.ethics_framework.get_ethics_report(days=1)
        
        # Violations summary
        violations_table = Table(title="Ethics Violations Summary", show_header=True)
        violations_table.add_column("Violation Type", style="red")
        violations_table.add_column("Count", style="yellow")
        violations_table.add_column("Severity Distribution", style="orange1")
        
        for violation_type, count in ethics_report["metrics"]["violations_by_type"].items():
            violations_table.add_row(
                violation_type.replace("_", " ").title(),
                str(count),
                "High: 2, Medium: 1"  # Simplified for demo
            )
        
        console.print(violations_table)
        
        # Recent violations
        if ethics_report["recent_violations"]:
            console.print("\n[red]Recent Violations:[/red]")
            recent_table = Table(show_header=True)
            recent_table.add_column("Type", style="red", width=20)
            recent_table.add_column("Severity", style="orange1", width=10)
            recent_table.add_column("Description", style="white", width=50)
            
            for violation in ethics_report["recent_violations"][:5]:
                recent_table.add_row(
                    violation["type"].replace("_", " ").title(),
                    violation["severity"].title(),
                    violation["description"][:47] + "..." if len(violation["description"]) > 50 else violation["description"]
                )
            
            console.print(recent_table)
        
        # Ethics guidelines
        console.print("\n[blue]Ethics Guidelines:[/blue]")
        guidelines_table = Table(show_header=True)
        guidelines_table.add_column("Principle", style="blue", width=15)
        guidelines_table.add_column("Description", style="white", width=60)
        
        for principle, description in ethics_report["guidelines"].items():
            guidelines_table.add_row(principle.title(), description)
        
        console.print(guidelines_table)
        
        # Recommendations
        if ethics_report["recommendations"]:
            console.print("\n[yellow]Recommendations:[/yellow]")
            for i, recommendation in enumerate(ethics_report["recommendations"], 1):
                console.print(f"   {i}. {recommendation}")
    
    async def _demo_integrated_workflow(self):
        """Demo integrated workflow across all systems"""
        console.print(Panel.fit(
            "[bold magenta]🔄 Integrated Workflow Demo[/bold magenta]\n\n"
            "This demonstrates how all enhancement systems work together\n"
            "to provide comprehensive monitoring and user-centric development.",
            title="Integrated Workflow",
            border_style="magenta"
        ))
        
        console.print("\n[yellow]Simulating complete user interaction workflow...[/yellow]")
        
        # Simulate a realistic user workflow
        workflow_steps = [
            ("User starts session", UserJourneyEvent.FIRST_LAUNCH),
            ("User selects AI model", UserJourneyEvent.MODEL_SELECTION),
            ("User makes first query", UserJourneyEvent.FIRST_QUERY),
            ("User encounters issue", UserJourneyEvent.ERROR_ENCOUNTERED),
            ("User provides feedback", None),  # Special case
            ("AI response analyzed", None),  # Special case
            ("Technical debt scan", None),  # Special case
            ("Session ends", UserJourneyEvent.SESSION_END)
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            
            task = progress.add_task("Processing workflow...", total=len(workflow_steps))
            
            for step_name, event in workflow_steps:
                console.print(f"\n[cyan]Step: {step_name}[/cyan]")
                
                if event:
                    # Track journey event
                    await track_user_event(
                        self.demo_user_id,
                        event,
                        self.session_id,
                        context={"workflow_step": step_name}
                    )
                    console.print("   ✓ Journey event tracked")
                
                elif "feedback" in step_name.lower():
                    # Collect feedback
                    await collect_user_feedback(
                        self.demo_user_id,
                        FeedbackType.BUG_REPORT,
                        "Found an issue with the AI response quality",
                        rating=2,
                        context={"workflow_step": step_name, "issue_type": "quality"}
                    )
                    console.print("   ✓ User feedback collected")
                
                elif "ai response" in step_name.lower():
                    # Analyze AI interaction
                    violations = await analyze_ai_interaction(
                        "Help me with coding",
                        "Programming is mainly for men. Contact admin@company.com for help.",
                        context={"workflow_step": step_name}
                    )
                    console.print(f"   ✓ AI interaction analyzed ({len(violations)} violations detected)")
                
                elif "debt scan" in step_name.lower():
                    # Quick debt scan
                    metrics = await self.debt_manager.get_debt_metrics()
                    console.print(f"   ✓ Technical debt assessed ({metrics.total_items} items)")
                
                progress.update(task, advance=1)
                await asyncio.sleep(0.3)
        
        console.print("\n[green]✅ Integrated workflow completed![/green]")
        
        # Show consolidated insights
        console.print("\n[cyan]Consolidated Insights:[/cyan]")
        
        # User metrics
        user_metrics = await self.feedback_manager.calculate_user_satisfaction(self.demo_user_id)
        console.print(f"   • User satisfaction: {user_metrics.satisfaction_rating:.1f}/5.0")
        console.print(f"   • Feature adoption: {user_metrics.feature_adoption_rate:.1%}")
        
        # Ethics metrics
        ethics_metrics = await self.ethics_framework.get_ethics_metrics(days=1)
        console.print(f"   • Ethics violations: {ethics_metrics.total_violations}")
        console.print(f"   • Resolution rate: {ethics_metrics.resolution_rate:.1f}%")
        
        # Debt metrics
        debt_metrics = await self.debt_manager.get_debt_metrics()
        console.print(f"   • Technical debt items: {debt_metrics.total_items}")
        console.print(f"   • Estimated effort: {debt_metrics.total_effort_hours:.1f} hours")
    
    async def _show_comprehensive_dashboard(self):
        """Show comprehensive dashboard with all metrics"""
        console.print(Panel.fit(
            "[bold blue]📊 Comprehensive Enhancement Systems Dashboard[/bold blue]\n\n"
            "Real-time view of all enhancement systems metrics and insights.",
            title="Dashboard",
            border_style="blue"
        ))
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="feedback"),
            Layout(name="debt")
        )
        
        layout["right"].split_column(
            Layout(name="ethics"),
            Layout(name="summary")
        )
        
        # Generate content for each section
        with console.status("[bold green]Loading dashboard data..."):
            # Get all metrics
            user_metrics = await self.feedback_manager.calculate_user_satisfaction(self.demo_user_id)
            feedback_summary = await self.feedback_manager.get_feedback_summary(days=7)
            debt_metrics = await self.debt_manager.get_debt_metrics()
            ethics_metrics = await self.ethics_framework.get_ethics_metrics(days=7)
            ethics_report = await self.ethics_framework.get_ethics_report(days=7)
        
        # Header
        layout["header"].update(Panel(
            f"[bold]Xencode Enhancement Systems Dashboard[/bold] | "
            f"User: {self.demo_user_id[:12]}... | "
            f"Session: {self.session_id[:12]}... | "
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            style="blue"
        ))
        
        # Feedback metrics
        feedback_table = Table(title="User Feedback Metrics", show_header=True, box=None)
        feedback_table.add_column("Metric", style="cyan")
        feedback_table.add_column("Value", style="green")
        
        feedback_table.add_row("Satisfaction", f"{user_metrics.satisfaction_rating:.1f}/5.0" if user_metrics.satisfaction_rating else "N/A")
        feedback_table.add_row("Total Feedback", str(feedback_summary["total_feedback"]))
        feedback_table.add_row("Sessions", str(user_metrics.total_sessions))
        feedback_table.add_row("Adoption Rate", f"{user_metrics.feature_adoption_rate:.1%}")
        
        layout["feedback"].update(Panel(feedback_table, title="User Feedback", border_style="green"))
        
        # Debt metrics
        debt_table = Table(title="Technical Debt Metrics", show_header=True, box=None)
        debt_table.add_column("Metric", style="cyan")
        debt_table.add_column("Value", style="yellow")
        
        debt_table.add_row("Total Items", str(debt_metrics.total_items))
        debt_table.add_row("Effort Hours", f"{debt_metrics.total_effort_hours:.1f}")
        debt_table.add_row("Debt Ratio", f"{debt_metrics.debt_ratio:.2f}")
        debt_table.add_row("Resolution Rate", f"{debt_metrics.resolution_rate:.1f}%")
        
        layout["debt"].update(Panel(debt_table, title="Technical Debt", border_style="yellow"))
        
        # Ethics metrics
        ethics_table = Table(title="AI Ethics Metrics", show_header=True, box=None)
        ethics_table.add_column("Metric", style="cyan")
        ethics_table.add_column("Value", style="red")
        
        ethics_table.add_row("Violations", str(ethics_metrics.total_violations))
        ethics_table.add_row("Resolution Rate", f"{ethics_metrics.resolution_rate:.1f}%")
        ethics_table.add_row("Avg Response", f"{ethics_metrics.avg_response_time_hours:.1f}h")
        
        layout["ethics"].update(Panel(ethics_table, title="AI Ethics", border_style="red"))
        
        # Summary
        summary_text = Text()
        summary_text.append("System Health: ", style="bold")
        
        # Calculate overall health score
        health_factors = []
        if user_metrics.satisfaction_rating:
            health_factors.append(user_metrics.satisfaction_rating / 5.0)
        if debt_metrics.total_items < 10:
            health_factors.append(0.9)
        else:
            health_factors.append(max(0.1, 1.0 - (debt_metrics.total_items / 50)))
        if ethics_metrics.total_violations < 5:
            health_factors.append(0.9)
        else:
            health_factors.append(max(0.1, 1.0 - (ethics_metrics.total_violations / 20)))
        
        overall_health = sum(health_factors) / len(health_factors) if health_factors else 0.5
        
        if overall_health > 0.8:
            summary_text.append("Excellent ✅", style="green")
        elif overall_health > 0.6:
            summary_text.append("Good 👍", style="yellow")
        else:
            summary_text.append("Needs Attention ⚠️", style="red")
        
        summary_text.append(f"\nOverall Score: {overall_health:.1%}\n\n")
        
        # Key insights
        summary_text.append("Key Insights:\n", style="bold")
        if user_metrics.satisfaction_rating and user_metrics.satisfaction_rating >= 4:
            summary_text.append("• High user satisfaction\n", style="green")
        if debt_metrics.total_items < 5:
            summary_text.append("• Low technical debt\n", style="green")
        if ethics_metrics.total_violations == 0:
            summary_text.append("• No ethics violations\n", style="green")
        
        # Recommendations
        if len(ethics_report["recommendations"]) > 0:
            summary_text.append("\nRecommendations:\n", style="bold")
            for rec in ethics_report["recommendations"][:2]:
                summary_text.append(f"• {rec[:40]}...\n", style="blue")
        
        layout["summary"].update(Panel(summary_text, title="System Summary", border_style="magenta"))
        
        # Footer
        layout["footer"].update(Panel(
            "[bold]Enhancement Systems Status:[/bold] "
            "[green]User Feedback ✓[/green] | "
            "[yellow]Technical Debt ✓[/yellow] | "
            "[red]AI Ethics ✓[/red] | "
            "[blue]All Systems Operational[/blue]",
            style="blue"
        ))
        
        console.print(layout)
        
        # Ask if user wants to see detailed reports
        if Confirm.ask("\nWould you like to see detailed reports for any system?"):
            system_choice = Prompt.ask(
                "Which system?", 
                choices=["feedback", "debt", "ethics", "all"],
                default="all"
            )
            
            if system_choice in ["feedback", "all"]:
                console.print("\n" + "="*60)
                await self._show_detailed_feedback_report()
            
            if system_choice in ["debt", "all"]:
                console.print("\n" + "="*60)
                await self._show_detailed_debt_report()
            
            if system_choice in ["ethics", "all"]:
                console.print("\n" + "="*60)
                await self._show_detailed_ethics_report()
    
    async def _show_detailed_feedback_report(self):
        """Show detailed feedback report"""
        console.print(Panel.fit("📊 Detailed User Feedback Report", style="green"))
        
        summary = await self.feedback_manager.get_feedback_summary(days=30)
        metrics = await self.feedback_manager.calculate_user_satisfaction(self.demo_user_id)
        
        # Feedback trends
        console.print("\n[green]Feedback Distribution:[/green]")
        for feedback_type, count in summary["feedback_by_type"].items():
            percentage = (count / max(summary["total_feedback"], 1)) * 100
            console.print(f"  {feedback_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # User journey insights
        console.print(f"\n[green]User Journey Insights:[/green]")
        console.print(f"  Feature Adoption: {metrics.feature_adoption_rate:.1%}")
        console.print(f"  Session Frequency: {metrics.session_frequency:.1f} per week")
        console.print(f"  Average Session: {metrics.avg_session_duration:.1f} seconds")
    
    async def _show_detailed_debt_report(self):
        """Show detailed technical debt report"""
        console.print(Panel.fit("🔧 Detailed Technical Debt Report", style="yellow"))
        
        metrics = await self.debt_manager.get_debt_metrics()
        items = await self.debt_manager.get_prioritized_debt_items(limit=10)
        
        # Debt breakdown
        console.print("\n[yellow]Debt Breakdown by Type:[/yellow]")
        for debt_type, count in metrics.items_by_type.items():
            console.print(f"  {debt_type.replace('_', ' ').title()}: {count} items")
        
        console.print("\n[yellow]Debt Breakdown by Severity:[/yellow]")
        for severity, count in metrics.items_by_severity.items():
            console.print(f"  {severity.title()}: {count} items")
        
        # Top priority items
        if items:
            console.print("\n[yellow]Top Priority Items:[/yellow]")
            for i, item in enumerate(items[:5], 1):
                console.print(f"  {i}. [{item.severity.value.upper()}] {item.file_path}: {item.description[:50]}...")
    
    async def _show_detailed_ethics_report(self):
        """Show detailed ethics report"""
        console.print(Panel.fit("🛡️ Detailed AI Ethics Report", style="red"))
        
        report = await self.ethics_framework.get_ethics_report(days=30)
        
        # Violations summary
        console.print("\n[red]Violations by Type:[/red]")
        for violation_type, count in report["metrics"]["violations_by_type"].items():
            console.print(f"  {violation_type.replace('_', ' ').title()}: {count}")
        
        # Recent violations
        if report["recent_violations"]:
            console.print("\n[red]Recent Violations:[/red]")
            for i, violation in enumerate(report["recent_violations"][:3], 1):
                console.print(f"  {i}. [{violation['severity'].upper()}] {violation['description'][:60]}...")
        
        # Recommendations
        if report["recommendations"]:
            console.print("\n[red]Recommendations:[/red]")
            for i, rec in enumerate(report["recommendations"], 1):
                console.print(f"  {i}. {rec}")


async def main():
    """Main demo function"""
    demo = EnhancementSystemsDemo()
    await demo.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user. Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
        console.print("[yellow]Please check your Python environment and dependencies.[/yellow]")