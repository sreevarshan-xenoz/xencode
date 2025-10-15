#!/usr/bin/env python3
"""
Demo: Performance Monitoring and Optimization System

This demo showcases the comprehensive performance monitoring system with
automated alerting and optimization capabilities for all Xencode components.
"""

import asyncio
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from xencode.monitoring.performance_optimizer import (
    PerformanceMonitoringSystem,
    PerformanceAlert,
    PerformanceThreshold,
    AlertSeverity,
    MetricType,
    get_performance_monitoring_system,
    initialize_performance_monitoring
)


async def demo_performance_monitoring():
    """Demonstrate performance monitoring and optimization capabilities"""
    
    console = Console()
    console.print("📊 [bold cyan]Performance Monitoring & Optimization Demo[/bold cyan]\n")
    
    # Initialize the monitoring system
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing performance monitoring system...", total=None)
        
        monitoring_system = get_performance_monitoring_system()
        
        progress.update(task, completed=True)
    
    console.print("✅ Performance monitoring system initialized\n")
    
    # Demo 1: System Status Overview
    console.print("🖥️ [bold yellow]System Status Overview[/bold yellow]")
    
    status = monitoring_system.get_system_status()
    
    # Create status table
    status_table = Table(title="Current System Status")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Details", style="yellow")
    
    status_table.add_row(
        "Monitoring", 
        "🟢 Active" if status["monitoring_active"] else "🔴 Inactive",
        "Real-time metrics collection"
    )
    status_table.add_row(
        "CPU Usage", 
        f"{status['current_metrics']['cpu_usage']:.1f}%",
        "Current processor utilization"
    )
    status_table.add_row(
        "Memory Usage", 
        f"{status['current_metrics']['memory_usage']:.1f}%",
        "Current memory utilization"
    )
    status_table.add_row(
        "Active Alerts", 
        str(status["active_alerts"]),
        "Number of active performance alerts"
    )
    status_table.add_row(
        "Auto Optimization", 
        "🟢 Enabled" if status["auto_optimization"] else "🔴 Disabled",
        "Automated performance optimization"
    )
    
    console.print(status_table)
    console.print()
    
    # Demo 2: Performance Thresholds Configuration
    console.print("⚙️ [bold yellow]Performance Thresholds Configuration[/bold yellow]")
    
    thresholds_table = Table(title="Performance Alert Thresholds")
    thresholds_table.add_column("Metric", style="cyan")
    thresholds_table.add_column("Warning", style="yellow")
    thresholds_table.add_column("Critical", style="red")
    thresholds_table.add_column("Emergency", style="bold red")
    thresholds_table.add_column("Duration", style="blue")
    
    for metric_type, threshold in monitoring_system.alert_manager.thresholds.items():
        emergency_val = f"{threshold.emergency_threshold:.1f}%" if threshold.emergency_threshold else "N/A"
        
        if metric_type == MetricType.CACHE_HIT_RATE:
            # For cache hit rate, lower is worse
            thresholds_table.add_row(
                metric_type.value.replace('_', ' ').title(),
                f"< {threshold.warning_threshold:.1f}%",
                f"< {threshold.critical_threshold:.1f}%",
                f"< {emergency_val}" if emergency_val != "N/A" else "N/A",
                f"{threshold.duration_seconds}s"
            )
        else:
            # For other metrics, higher is worse
            thresholds_table.add_row(
                metric_type.value.replace('_', ' ').title(),
                f"> {threshold.warning_threshold:.1f}%",
                f"> {threshold.critical_threshold:.1f}%",
                f"> {emergency_val}" if emergency_val != "N/A" else "N/A",
                f"{threshold.duration_seconds}s"
            )
    
    console.print(thresholds_table)
    console.print()
    
    # Demo 3: Simulate Performance Issues and Alerts
    console.print("🚨 [bold yellow]Performance Alert Simulation[/bold yellow]")
    
    # Add alert callback to capture alerts
    alerts_received = []
    
    def alert_callback(alert: PerformanceAlert):
        alerts_received.append(alert)
        console.print(f"  🔔 Alert: {alert.title} - {alert.description}")
    
    monitoring_system.add_alert_callback(alert_callback)
    
    # Simulate high resource usage
    console.print("  📈 Simulating high resource usage scenarios...")
    
    # High CPU usage
    high_cpu_metrics = {
        MetricType.CPU_USAGE: 92.0,  # Exceeds critical threshold
        MetricType.MEMORY_USAGE: 65.0,
        MetricType.CACHE_HIT_RATE: 85.0
    }
    
    await monitoring_system.alert_manager.check_thresholds(high_cpu_metrics)
    
    # High memory usage
    high_memory_metrics = {
        MetricType.CPU_USAGE: 45.0,
        MetricType.MEMORY_USAGE: 93.0,  # Exceeds critical threshold
        MetricType.CACHE_HIT_RATE: 75.0
    }
    
    await monitoring_system.alert_manager.check_thresholds(high_memory_metrics)
    
    # Low cache hit rate
    low_cache_metrics = {
        MetricType.CPU_USAGE: 35.0,
        MetricType.MEMORY_USAGE: 55.0,
        MetricType.CACHE_HIT_RATE: 45.0  # Below critical threshold
    }
    
    await monitoring_system.alert_manager.check_thresholds(low_cache_metrics)
    
    console.print(f"  ✅ Generated {len(alerts_received)} performance alerts")
    console.print()
    
    # Demo 4: Active Alerts Display
    console.print("📋 [bold yellow]Active Performance Alerts[/bold yellow]")
    
    active_alerts = monitoring_system.alert_manager.get_active_alerts()
    
    if active_alerts:
        alerts_table = Table(title="Active Performance Alerts")
        alerts_table.add_column("Severity", style="red")
        alerts_table.add_column("Metric", style="cyan")
        alerts_table.add_column("Current Value", style="yellow")
        alerts_table.add_column("Threshold", style="blue")
        alerts_table.add_column("Time", style="green")
        
        for alert in active_alerts:
            severity_color = {
                AlertSeverity.WARNING: "[yellow]WARNING[/yellow]",
                AlertSeverity.CRITICAL: "[red]CRITICAL[/red]",
                AlertSeverity.EMERGENCY: "[bold red]EMERGENCY[/bold red]"
            }.get(alert.severity, alert.severity.value)
            
            alerts_table.add_row(
                severity_color,
                alert.metric_type.value.replace('_', ' ').title(),
                f"{alert.current_value:.1f}%",
                f"{alert.threshold_value:.1f}%",
                alert.timestamp.strftime("%H:%M:%S")
            )
        
        console.print(alerts_table)
    else:
        console.print("  ✅ No active alerts - system is healthy")
    
    console.print()
    
    # Demo 5: Automated Optimization
    console.print("🔧 [bold yellow]Automated Performance Optimization[/bold yellow]")
    
    console.print("  ⚡ Running automated optimization analysis...")
    
    # Test optimization with high memory usage
    optimization_metrics = {
        MetricType.MEMORY_USAGE: 88.0,  # Triggers memory cleanup
        MetricType.CACHE_HIT_RATE: 55.0,  # Triggers cache warming
        MetricType.CPU_USAGE: 92.0  # Triggers process throttling
    }
    
    optimization_actions = await monitoring_system.optimizer.analyze_and_optimize(optimization_metrics)
    
    if optimization_actions:
        console.print(f"  ✅ Executed {len(optimization_actions)} optimization actions:")
        
        for action in optimization_actions:
            status_icon = "✅" if action.executed else "❌"
            console.print(f"    {status_icon} {action.description}")
            if action.result:
                console.print(f"      Result: {action.result}")
    else:
        console.print("  ℹ️ No optimization actions needed at this time")
    
    console.print()
    
    # Demo 6: Performance Trends Analysis
    console.print("📈 [bold yellow]Performance Trends Analysis[/bold yellow]")
    
    # Simulate some historical data
    current_time = time.time()
    
    # Add trending data to metrics collector
    for i in range(20):
        timestamp = current_time - (i * 30)  # 30-second intervals
        
        # Simulate increasing CPU trend
        cpu_value = 50 + (i * 2)  # Gradually increasing
        monitoring_system.metrics_collector.metrics_history[MetricType.CPU_USAGE].append((timestamp, cpu_value))
        
        # Simulate decreasing cache hit rate
        cache_value = 90 - (i * 1.5)  # Gradually decreasing
        monitoring_system.metrics_collector.metrics_history[MetricType.CACHE_HIT_RATE].append((timestamp, cache_value))
        
        # Simulate stable memory usage
        memory_value = 65 + (i % 3 - 1)  # Small fluctuations around 65%
        monitoring_system.metrics_collector.metrics_history[MetricType.MEMORY_USAGE].append((timestamp, memory_value))
    
    # Calculate trends
    cpu_trend = monitoring_system.metrics_collector.calculate_trend(MetricType.CPU_USAGE, 10)
    memory_trend = monitoring_system.metrics_collector.calculate_trend(MetricType.MEMORY_USAGE, 10)
    cache_trend = monitoring_system.metrics_collector.calculate_trend(MetricType.CACHE_HIT_RATE, 10)
    
    trends_table = Table(title="Performance Trends (Last 10 Minutes)")
    trends_table.add_column("Metric", style="cyan")
    trends_table.add_column("Trend", style="yellow")
    trends_table.add_column("Indicator", style="green")
    trends_table.add_column("Recommendation", style="blue")
    
    trend_indicators = {
        "increasing": "📈 Increasing",
        "decreasing": "📉 Decreasing", 
        "stable": "➡️ Stable",
        None: "❓ Insufficient Data"
    }
    
    trend_recommendations = {
        (MetricType.CPU_USAGE, "increasing"): "Monitor for potential bottlenecks",
        (MetricType.CPU_USAGE, "stable"): "CPU usage is stable",
        (MetricType.MEMORY_USAGE, "increasing"): "Consider memory optimization",
        (MetricType.MEMORY_USAGE, "stable"): "Memory usage is stable",
        (MetricType.CACHE_HIT_RATE, "decreasing"): "Investigate cache performance",
        (MetricType.CACHE_HIT_RATE, "stable"): "Cache performance is stable"
    }
    
    trends_table.add_row(
        "CPU Usage",
        trend_indicators.get(cpu_trend, "❓ Unknown"),
        cpu_trend or "Unknown",
        trend_recommendations.get((MetricType.CPU_USAGE, cpu_trend), "Monitor closely")
    )
    
    trends_table.add_row(
        "Memory Usage",
        trend_indicators.get(memory_trend, "❓ Unknown"),
        memory_trend or "Unknown",
        trend_recommendations.get((MetricType.MEMORY_USAGE, memory_trend), "Monitor closely")
    )
    
    trends_table.add_row(
        "Cache Hit Rate",
        trend_indicators.get(cache_trend, "❓ Unknown"),
        cache_trend or "Unknown",
        trend_recommendations.get((MetricType.CACHE_HIT_RATE, cache_trend), "Monitor closely")
    )
    
    console.print(trends_table)
    console.print()
    
    # Demo 7: Performance Report Generation
    console.print("📊 [bold yellow]Comprehensive Performance Report[/bold yellow]")
    
    report = monitoring_system.get_performance_report()
    
    report_panel = Panel(
        f"""
🕐 Report Generated: {report['timestamp']}

🏥 System Health: {report['system_health'].upper()}

📈 Performance Trends:
  • CPU Usage: {report['trends']['cpu_usage'] or 'Unknown'}
  • Memory Usage: {report['trends']['memory_usage'] or 'Unknown'}
  • Cache Hit Rate: {report['trends']['cache_hit_rate'] or 'Unknown'}

🚨 Alert Summary:
  • Active Alerts: {report['alerts']['active']}
  • Warning: {report['alerts']['summary']['warning']}
  • Critical: {report['alerts']['summary']['critical']}
  • Emergency: {report['alerts']['summary']['emergency']}

🔧 Optimization Summary:
  • Total Actions Executed: {report['optimizations']['total_executed']}
  • Recent Actions (24h): {report['optimizations']['recent']}
        """.strip(),
        title="📋 Performance Analysis Report",
        border_style="blue"
    )
    
    console.print(report_panel)
    console.print()
    
    # Demo 8: Custom Threshold Configuration
    console.print("⚙️ [bold yellow]Custom Threshold Configuration[/bold yellow]")
    
    console.print("  🔧 Configuring custom performance thresholds...")
    
    # Create custom threshold for response time
    custom_threshold = PerformanceThreshold(
        MetricType.RESPONSE_TIME,
        warning_threshold=500.0,  # 500ms
        critical_threshold=2000.0,  # 2 seconds
        emergency_threshold=5000.0,  # 5 seconds
        duration_seconds=30,
        enabled=True
    )
    
    monitoring_system.alert_manager.update_threshold(MetricType.RESPONSE_TIME, custom_threshold)
    
    console.print("  ✅ Updated response time thresholds:")
    console.print(f"    • Warning: > 500ms")
    console.print(f"    • Critical: > 2000ms")
    console.print(f"    • Emergency: > 5000ms")
    console.print()
    
    # Demo 9: Real-time Monitoring Simulation
    console.print("📡 [bold yellow]Real-time Monitoring Simulation[/bold yellow]")
    
    console.print("  🔄 Simulating 10 seconds of real-time monitoring...")
    
    # Create a simple real-time display
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="metrics", size=10),
        Layout(name="alerts", size=8)
    )
    
    def create_metrics_display():
        """Create real-time metrics display"""
        current_status = monitoring_system.get_system_status()
        
        metrics_table = Table(title="Real-time System Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Current Value", style="yellow")
        metrics_table.add_column("Status", style="green")
        
        cpu_usage = current_status['current_metrics']['cpu_usage']
        memory_usage = current_status['current_metrics']['memory_usage']
        
        cpu_status = "🟢 Normal" if cpu_usage < 70 else "🟡 Warning" if cpu_usage < 85 else "🔴 Critical"
        memory_status = "🟢 Normal" if memory_usage < 75 else "🟡 Warning" if memory_usage < 90 else "🔴 Critical"
        
        metrics_table.add_row("CPU Usage", f"{cpu_usage:.1f}%", cpu_status)
        metrics_table.add_row("Memory Usage", f"{memory_usage:.1f}%", memory_status)
        metrics_table.add_row("Active Alerts", str(current_status['active_alerts']), "📊 Monitoring")
        
        return metrics_table
    
    def create_alerts_display():
        """Create alerts display"""
        active_alerts = monitoring_system.alert_manager.get_active_alerts()
        
        if active_alerts:
            alerts_text = "\n".join([
                f"🚨 {alert.severity.value.upper()}: {alert.title}"
                for alert in active_alerts[-3:]  # Show last 3 alerts
            ])
        else:
            alerts_text = "✅ No active alerts - System is healthy"
        
        return Panel(alerts_text, title="Active Alerts", border_style="red")
    
    # Simulate real-time updates
    with Live(layout, refresh_per_second=2, console=console) as live:
        layout["header"].update(Panel("📊 Real-time Performance Monitoring", style="bold cyan"))
        
        for i in range(10):  # 10 iterations = ~5 seconds
            layout["metrics"].update(create_metrics_display())
            layout["alerts"].update(create_alerts_display())
            
            await asyncio.sleep(0.5)
    
    console.print("\n  ✅ Real-time monitoring simulation completed")
    console.print()
    
    # Demo Summary
    console.print("📋 [bold green]Demo Summary[/bold green]")
    
    summary_panel = Panel(
        """
✅ Performance Monitoring: Real-time metrics collection and analysis
✅ Intelligent Alerting: Configurable thresholds with severity levels
✅ Automated Optimization: Smart performance improvements
✅ Trend Analysis: Historical data analysis and predictions
✅ Comprehensive Reporting: Detailed system health reports
✅ Custom Configuration: Flexible threshold and rule management
✅ Real-time Dashboard: Live monitoring capabilities

The performance monitoring system provides:
• 🚀 Proactive performance issue detection
• 🧠 Intelligent automated optimization
• 📊 Comprehensive system health insights
• 🔔 Real-time alerting and notifications
• 📈 Historical trend analysis
• ⚙️ Flexible configuration options
        """.strip(),
        title="🎉 Performance Monitoring System Features",
        border_style="green"
    )
    
    console.print(summary_panel)
    
    console.print("\n🎊 [bold cyan]Performance Monitoring Demo Complete![/bold cyan]")
    console.print("The system provides comprehensive performance monitoring with")
    console.print("intelligent alerting and automated optimization for optimal")
    console.print("system performance and reliability.")


async def main():
    """Main demo function"""
    try:
        await demo_performance_monitoring()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())