#!/usr/bin/env python3
"""
Demo script for the performance monitoring dashboard

This script demonstrates the live performance monitoring dashboard
with real-time metrics and alerts.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xencode'))

from standalone_performance_dashboard import (
    StandalonePerformanceDashboard, PerformanceAlert
)
from rich.console import Console

async def main():
    """Main demo function"""
    console = Console()
    
    console.print("🚀 [bold cyan]Xencode Performance Monitoring Dashboard Demo[/bold cyan]\n")
    console.print("This demo shows real-time system monitoring with alerts and visualizations.\n")
    
    # Create dashboard
    dashboard = StandalonePerformanceDashboard()
    
    # Add alert callback for console notifications
    def alert_callback(alert: PerformanceAlert):
        severity_colors = {
            "warning": "yellow",
            "critical": "red",
            "emergency": "bold red"
        }
        color = severity_colors.get(alert.severity.value, "white")
        console.print(f"🚨 [{color}]ALERT[/{color}]: {alert.title} - {alert.description}")
    
    dashboard.add_alert_callback(alert_callback)
    
    # Generate some initial sample data to make the demo more interesting
    console.print("📊 Generating sample metrics data for demonstration...")
    dashboard.generate_sample_data()
    console.print("✅ Sample data generated\n")
    
    console.print("🔄 Starting live dashboard...")
    console.print("   📈 Real-time system metrics")
    console.print("   🚨 Automated alert generation")
    console.print("   💻 Resource utilization monitoring")
    console.print("   📊 Performance trend analysis")
    console.print("\n[dim]Press Ctrl+C to stop the dashboard[/dim]\n")
    
    try:
        # Start the live dashboard with 1-second refresh rate
        await dashboard.start_live_dashboard(refresh_interval=1.0)
    except KeyboardInterrupt:
        console.print("\n👋 [green]Dashboard stopped by user[/green]")
    except Exception as e:
        console.print(f"\n❌ [red]Error running dashboard: {e}[/red]")
    finally:
        await dashboard.stop()
        
        # Show final summary
        console.print("\n📋 [bold]Dashboard Session Summary[/bold]")
        final_alerts = dashboard.performance_monitor.get_active_alerts()
        console.print(f"   🚨 Active alerts: {len(final_alerts)}")
        
        if final_alerts:
            console.print("   📝 Recent alerts:")
            for i, alert in enumerate(final_alerts[:5]):  # Show last 5 alerts
                console.print(f"      {i+1}. {alert.severity.value.upper()}: {alert.metric_type.value} at {alert.current_value:.1f}")
        
        console.print("\n✨ [green]Thank you for trying the Xencode Performance Dashboard![/green]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")