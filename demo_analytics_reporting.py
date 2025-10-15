#!/usr/bin/env python3
"""
Demo script for Analytics Reporting System

This script demonstrates the comprehensive analytics reporting capabilities including:
- Multi-format report generation (JSON, CSV, HTML, PDF, Markdown)
- Scheduled reporting and automated delivery
- Analytics API for external integrations
- Custom report templates and filtering
"""

import asyncio
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xencode'))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

async def main():
    """Main demo function"""
    console = Console()
    
    console.print("🚀 [bold cyan]Xencode Analytics Reporting System Demo[/bold cyan]\n")
    console.print("This demo showcases comprehensive reporting capabilities with multiple formats and scheduling.\n")
    
    try:
        # Import the reporting system
        from analytics_reporting_system import (
            AnalyticsReportingSystem, ReportConfig, DeliveryConfig,
            ReportFormat, ReportType, DeliveryMethod
        )
        
        console.print("✅ Analytics Reporting System loaded successfully")
        
        # Create reporting system
        reporting_system = AnalyticsReportingSystem()
        
        # Start the system
        console.print("🔄 Starting reporting system...")
        await reporting_system.start()
        console.print("✅ Reporting system started\n")
        
        # Demo 1: Generate reports in multiple formats
        await demo_multi_format_reports(console, reporting_system)
        
        # Demo 2: Scheduled reporting
        await demo_scheduled_reporting(console, reporting_system)
        
        # Demo 3: Custom filters and configurations
        await demo_custom_reports(console, reporting_system)
        
        # Demo 4: Report management
        await demo_report_management(console, reporting_system)
        
        # Demo 5: Analytics API (if available)
        await demo_analytics_api(console)
        
        # Cleanup
        console.print("\n🛑 Stopping reporting system...")
        await reporting_system.stop()
        console.print("✅ Demo completed successfully!")
        
    except ImportError as e:
        console.print(f"❌ [red]Import error: {e}[/red]")
        console.print("Please ensure the analytics reporting system is available.")
    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()

async def demo_multi_format_reports(console: Console, reporting_system):
    """Demo multi-format report generation"""
    console.print("📊 [bold yellow]Demo 1: Multi-Format Report Generation[/bold yellow]\n")
    
    # Create output directory
    output_dir = Path("demo_reports")
    output_dir.mkdir(exist_ok=True)
    
    # Test different report formats
    formats_to_test = [
        (ReportFormat.JSON, "json"),
        (ReportFormat.CSV, "csv"),
        (ReportFormat.HTML, "html"),
        (ReportFormat.MARKDOWN, "md")
    ]
    
    # Check if PDF is available
    try:
        from reportlab.lib.pagesizes import letter
        formats_to_test.append((ReportFormat.PDF, "pdf"))
        console.print("📄 PDF generation available")
    except ImportError:
        console.print("⚠️ PDF generation not available (install reportlab for PDF support)")
    
    console.print(f"🔄 Generating reports in {len(formats_to_test)} formats...\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for report_format, extension in formats_to_test:
            task = progress.add_task(f"Generating {report_format.value.upper()} report...", total=1)
            
            # Create report configuration
            config = ReportConfig(
                report_type=ReportType.SUMMARY,
                format=report_format,
                title=f"Xencode Analytics Report - {report_format.value.upper()}",
                description=f"Comprehensive analytics report in {report_format.value} format",
                time_period_hours=72,  # 3 days
                include_charts=True,
                include_recommendations=True
            )
            
            # Generate report
            report = await reporting_system.generate_report(config)
            
            # Save to file
            file_path = output_dir / f"analytics_report.{extension}"
            await reporting_system.save_report(report, file_path)
            
            progress.update(task, completed=1)
            console.print(f"   ✅ {report_format.value.upper()}: {file_path}")
    
    console.print(f"\n📁 All reports saved to: {output_dir.absolute()}\n")

async def demo_scheduled_reporting(console: Console, reporting_system):
    """Demo scheduled reporting functionality"""
    console.print("📅 [bold yellow]Demo 2: Scheduled Reporting[/bold yellow]\n")
    
    # Schedule different types of reports
    schedules = [
        {
            "name": "Daily Summary",
            "schedule": "daily",
            "format": ReportFormat.HTML,
            "destination": "reports/daily_summary.html"
        },
        {
            "name": "Weekly Analysis", 
            "schedule": "weekly",
            "format": ReportFormat.PDF,
            "destination": "reports/weekly_analysis.pdf"
        },
        {
            "name": "Hourly Monitoring",
            "schedule": "every_6_hours",
            "format": ReportFormat.JSON,
            "destination": "reports/monitoring.json"
        }
    ]
    
    scheduled_ids = []
    
    for schedule_info in schedules:
        # Create report config
        report_config = ReportConfig(
            report_type=ReportType.SUMMARY,
            format=schedule_info["format"],
            title=schedule_info["name"],
            description=f"Automated {schedule_info['name'].lower()} report",
            time_period_hours=24
        )
        
        # Create delivery config
        delivery_config = DeliveryConfig(
            method=DeliveryMethod.FILE,
            destination=schedule_info["destination"],
            schedule=schedule_info["schedule"]
        )
        
        # Schedule the report
        schedule_id = reporting_system.schedule_report(report_config, delivery_config)
        scheduled_ids.append((schedule_id, schedule_info["name"]))
        
        console.print(f"✅ Scheduled: {schedule_info['name']} ({schedule_info['schedule']})")
    
    # Show scheduled reports
    console.print("\n📋 Active Scheduled Reports:")
    scheduled_reports = reporting_system.get_scheduled_reports()
    
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Schedule ID", style="cyan")
    table.add_column("Report Title", style="green")
    table.add_column("Format", style="yellow")
    table.add_column("Schedule", style="blue")
    table.add_column("Status", style="magenta")
    
    for schedule_id, info in scheduled_reports.items():
        table.add_row(
            schedule_id[:8] + "...",
            info["report_config"].title,
            info["report_config"].format.value.upper(),
            info["delivery_config"].schedule,
            "Active"
        )
    
    console.print(table)
    console.print()

async def demo_custom_reports(console: Console, reporting_system):
    """Demo custom report configurations and filters"""
    console.print("🎛️ [bold yellow]Demo 3: Custom Reports and Filters[/bold yellow]\n")
    
    # Custom report with filters
    custom_configs = [
        {
            "name": "High-Confidence Patterns",
            "filters": {"min_confidence": 0.8},
            "description": "Only patterns with >80% confidence"
        },
        {
            "name": "Significant Cost Savings",
            "filters": {"min_savings": 100.0},
            "description": "Only optimizations with >$100 savings"
        },
        {
            "name": "Combined Filters",
            "filters": {"min_confidence": 0.7, "min_savings": 50.0},
            "description": "High confidence patterns and meaningful savings"
        }
    ]
    
    for custom_config in custom_configs:
        console.print(f"🔍 Generating: {custom_config['name']}")
        console.print(f"   📝 {custom_config['description']}")
        
        config = ReportConfig(
            report_type=ReportType.DETAILED,
            format=ReportFormat.JSON,
            title=custom_config["name"],
            description=custom_config["description"],
            time_period_hours=168,  # 1 week
            custom_filters=custom_config["filters"],
            include_recommendations=True
        )
        
        report = await reporting_system.generate_report(config)
        
        # Parse and show filtered results
        import json
        report_data = json.loads(report.content)
        analytics_data = report_data.get("analytics_data", {})
        
        patterns_count = len(analytics_data.get("usage_patterns", []))
        optimizations_count = len(analytics_data.get("cost_optimizations", []))
        
        console.print(f"   📊 Results: {patterns_count} patterns, {optimizations_count} optimizations")
        console.print()

async def demo_report_management(console: Console, reporting_system):
    """Demo report management and history"""
    console.print("📚 [bold yellow]Demo 4: Report Management[/bold yellow]\n")
    
    # Show report history
    history = reporting_system.get_report_history()
    
    console.print(f"📋 Generated Reports History ({len(history)} reports):")
    
    if history:
        history_table = Table(box=box.SIMPLE, show_header=True)
        history_table.add_column("Report ID", style="cyan")
        history_table.add_column("Title", style="green")
        history_table.add_column("Format", style="yellow")
        history_table.add_column("Generated", style="blue")
        history_table.add_column("Size", style="magenta")
        
        for report in history[-10:]:  # Show last 10 reports
            size = len(report.content)
            size_str = f"{size} chars" if isinstance(report.content, str) else f"{size} bytes"
            
            history_table.add_row(
                report.report_id[:12] + "...",
                report.config.title[:30] + ("..." if len(report.config.title) > 30 else ""),
                report.format.value.upper(),
                report.generated_at.strftime("%H:%M:%S"),
                size_str
            )
        
        console.print(history_table)
    else:
        console.print("   No reports in history")
    
    # Show system statistics
    console.print("\n📊 System Statistics:")
    stats_table = Table(box=box.SIMPLE, show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    total_reports = len(history)
    scheduled_reports = len(reporting_system.get_scheduled_reports())
    
    # Calculate format distribution
    format_counts = {}
    for report in history:
        format_name = report.format.value
        format_counts[format_name] = format_counts.get(format_name, 0) + 1
    
    stats_table.add_row("Total Reports Generated", str(total_reports))
    stats_table.add_row("Active Scheduled Reports", str(scheduled_reports))
    
    for format_name, count in format_counts.items():
        stats_table.add_row(f"{format_name.upper()} Reports", str(count))
    
    console.print(stats_table)
    console.print()

async def demo_analytics_api(console: Console):
    """Demo Analytics API functionality"""
    console.print("🌐 [bold yellow]Demo 5: Analytics API[/bold yellow]\n")
    
    try:
        from analytics_api import AnalyticsAPI
        
        console.print("✅ Analytics API available")
        console.print("🔧 API Features:")
        console.print("   📊 GET /analytics/summary - Get analytics summary")
        console.print("   🔍 GET /analytics/patterns - Get usage patterns")
        console.print("   💰 GET /analytics/optimizations - Get cost optimizations")
        console.print("   📈 GET /analytics/trends/{metric} - Get trend analysis")
        console.print("   📄 POST /reports/generate - Generate reports")
        console.print("   📅 POST /reports/schedule - Schedule reports")
        console.print("   📋 GET /reports/scheduled - List scheduled reports")
        
        console.print("\n🚀 To start the API server, run:")
        console.print("   [dim]python -c \"from xencode.analytics_api import run_analytics_api_demo; import asyncio; asyncio.run(run_analytics_api_demo())\"[/dim]")
        
        console.print("\n📖 API Documentation:")
        console.print("   Swagger UI: http://localhost:8000/docs")
        console.print("   ReDoc: http://localhost:8000/redoc")
        
        console.print("\n🔑 Authentication:")
        console.print("   Use 'demo-token' as Bearer token for testing")
        
        # Show example API usage
        console.print("\n💡 Example API Calls:")
        
        examples = [
            {
                "method": "GET",
                "endpoint": "/analytics/summary?hours=24",
                "description": "Get 24-hour analytics summary"
            },
            {
                "method": "GET", 
                "endpoint": "/analytics/patterns?min_confidence=0.8",
                "description": "Get high-confidence usage patterns"
            },
            {
                "method": "POST",
                "endpoint": "/reports/generate",
                "description": "Generate custom report",
                "body": {
                    "report_type": "summary",
                    "format": "html",
                    "title": "Custom Report",
                    "time_period_hours": 48
                }
            }
        ]
        
        for example in examples:
            console.print(f"   {example['method']} {example['endpoint']}")
            console.print(f"      {example['description']}")
            if 'body' in example:
                console.print(f"      Body: {example['body']}")
        
    except ImportError:
        console.print("⚠️ Analytics API not available (requires FastAPI)")
        console.print("   Install with: pip install fastapi uvicorn")
    
    console.print()

async def show_sample_report_content(console: Console):
    """Show sample report content"""
    console.print("📄 [bold yellow]Sample Report Content Preview[/bold yellow]\n")
    
    # Show what a typical report contains
    sample_sections = [
        "📊 Executive Summary",
        "🔍 Usage Patterns Analysis", 
        "💰 Cost Optimization Opportunities",
        "📈 Performance Trends",
        "👥 User Behavior Insights",
        "💡 Key Recommendations",
        "📋 Detailed Analytics Data"
    ]
    
    console.print("📋 Typical Report Sections:")
    for section in sample_sections:
        console.print(f"   {section}")
    
    console.print("\n📊 Sample Analytics Data:")
    sample_data = Table(box=box.SIMPLE, show_header=True)
    sample_data.add_column("Metric", style="cyan")
    sample_data.add_column("Value", style="green")
    sample_data.add_column("Trend", style="yellow")
    
    sample_data.add_row("Usage Patterns Detected", "5", "↗️ Increasing")
    sample_data.add_row("Users Analyzed", "127", "→ Stable")
    sample_data.add_row("Cost Optimizations", "8", "↗️ More opportunities")
    sample_data.add_row("Potential Monthly Savings", "$342.50", "↗️ Growing")
    sample_data.add_row("System Efficiency", "87%", "↗️ Improving")
    
    console.print(sample_data)
    console.print()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")