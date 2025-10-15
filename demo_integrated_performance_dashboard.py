#!/usr/bin/env python3
"""
Demo script for the integrated performance monitoring dashboard

This script demonstrates the performance monitoring dashboard integrated
with the existing Xencode analytics infrastructure.
"""

import asyncio
import sys
import os
sys.path.insert(0, '.')

from rich.console import Console

async def main():
    """Main demo function"""
    console = Console()
    
    console.print("🚀 [bold cyan]Xencode Integrated Performance Monitoring Dashboard Demo[/bold cyan]\n")
    console.print("This demo shows the performance dashboard integrated with Xencode analytics.\n")
    
    try:
        # Try to import the integrated dashboard
        from xencode.performance_monitoring_dashboard import PerformanceMonitoringDashboard
        from xencode.monitoring.metrics_collector import PrometheusMetricsCollector
        
        console.print("✅ Integrated dashboard components loaded")
        
        # Create Prometheus collector
        prometheus_collector = PrometheusMetricsCollector(
            metrics_port=8001,  # Use different port to avoid conflicts
            collect_system_metrics=True
        )
        
        # Create integrated dashboard
        dashboard = PerformanceMonitoringDashboard(prometheus_collector)
        
        console.print("✅ Dashboard created with Prometheus integration")
        
        # Start the dashboard
        await dashboard.start()
        
        console.print("✅ Dashboard started with analytics integration")
        
        # Show dashboard status
        status = dashboard.get_dashboard_status()
        console.print(f"📊 Dashboard Status:")
        console.print(f"   Running: {status['running']}")
        console.print(f"   Active Alerts: {status['active_alerts']}")
        console.print(f"   Monitoring Available: {status['monitoring_available']}")
        console.print(f"   Prometheus: {status['prometheus_status']['running'] if status['prometheus_status'] else 'Not available'}")
        
        if 'analytics_integration' in status:
            console.print(f"   Analytics Integration: {'✅ Connected' if status['analytics_integration'] else '❌ Not available'}")
        
        console.print("\n🔄 Starting live dashboard (Press Ctrl+C to stop)...")
        
        try:
            await dashboard.start_live_dashboard(refresh_interval=2.0)
        except KeyboardInterrupt:
            console.print("\n👋 [green]Dashboard stopped by user[/green]")
        
        await dashboard.stop()
        
    except ImportError as e:
        console.print(f"❌ [red]Import error: {e}[/red]")
        console.print("\n🔄 Falling back to standalone dashboard...")
        
        # Fallback to standalone dashboard
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xencode'))
        from standalone_performance_dashboard import StandalonePerformanceDashboard
        
        dashboard = StandalonePerformanceDashboard()
        dashboard.generate_sample_data()
        
        console.print("✅ Standalone dashboard created with sample data")
        console.print("\n🔄 Starting standalone dashboard (Press Ctrl+C to stop)...")
        
        try:
            await dashboard.start_live_dashboard(refresh_interval=2.0)
        except KeyboardInterrupt:
            console.print("\n👋 [green]Dashboard stopped by user[/green]")
        
        await dashboard.stop()
    
    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")