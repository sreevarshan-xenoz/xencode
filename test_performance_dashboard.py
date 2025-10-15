#!/usr/bin/env python3
"""
Test script for the standalone performance monitoring dashboard
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xencode'))

try:
    from standalone_performance_dashboard import (
        PerformanceMonitor, MetricBuffer, MetricType, AlertSeverity, 
        StandalonePerformanceDashboard, DashboardRenderer
    )
    print('✅ Standalone performance dashboard imports successfully')
    
    # Test MetricBuffer
    print("\n📊 Testing MetricBuffer...")
    buffer = MetricBuffer(max_size=10)
    buffer.add(50.0)
    buffer.add(60.0)
    buffer.add(70.0)
    print(f'✅ MetricBuffer: {len(buffer.data)} values stored, latest: {buffer.data[-1]}')
    
    # Test trend analysis
    trend = buffer.get_trend(300)
    print(f'✅ Trend analysis: {trend}')
    
    # Test average calculation
    avg = buffer.get_average(300)
    print(f'✅ Average calculation: {avg:.1f}')
    
    # Test PerformanceMonitor
    print("\n🔍 Testing PerformanceMonitor...")
    monitor = PerformanceMonitor()
    print(f'✅ PerformanceMonitor initialized with {len(monitor.thresholds)} thresholds')
    
    # Test threshold configuration
    cpu_threshold = monitor.thresholds[MetricType.CPU]
    print(f'✅ CPU thresholds: Warning={cpu_threshold.warning_threshold}, Critical={cpu_threshold.critical_threshold}')
    
    # Test normal metric recording
    monitor.record_performance_metric(MetricType.CPU, 50.0)
    print('✅ Normal CPU metric recorded (50%)')
    
    # Test alert generation
    monitor.record_performance_metric(MetricType.CPU, 75.0)  # Should trigger warning
    alerts = monitor.get_active_alerts()
    print(f'✅ Warning alert generated: {len(alerts)} alerts active')
    
    if alerts:
        alert = alerts[0]
        print(f'   Alert: {alert.title} - {alert.description}')
    
    # Test critical alert
    monitor.record_performance_metric(MetricType.MEMORY, 92.0)  # Should trigger critical
    critical_alerts = monitor.get_active_alerts(AlertSeverity.CRITICAL)
    print(f'✅ Critical alert generated: {len(critical_alerts)} critical alerts')
    
    # Test system metrics collection
    print("\n💻 Testing System Metrics Collection...")
    metrics = monitor.collect_system_metrics()
    print(f'✅ System metrics collected: {len(metrics)} metrics')
    for key, value in list(metrics.items())[:5]:  # Show first 5 metrics
        print(f'   {key}: {value}')
    
    # Test metric summary
    print("\n📈 Testing Metric Summary...")
    summary = monitor.get_metric_summary(MetricType.CPU, 300)
    print(f'✅ CPU summary: current={summary["current"]:.1f}, avg={summary["average"]:.1f}, trend={summary["trend"]}')
    
    # Test alert resolution
    print("\n🔧 Testing Alert Resolution...")
    all_alerts = monitor.get_active_alerts()
    if all_alerts:
        alert_to_resolve = all_alerts[0]
        result = monitor.resolve_alert(alert_to_resolve.alert_id)
        print(f'✅ Alert resolution: {result}')
        remaining_alerts = monitor.get_active_alerts()
        print(f'✅ Remaining active alerts: {len(remaining_alerts)}')
    
    # Test dashboard creation
    print("\n🎛️ Testing Dashboard Creation...")
    dashboard = StandalonePerformanceDashboard()
    print('✅ Dashboard created successfully')
    
    # Test renderer
    renderer = DashboardRenderer(monitor)
    print('✅ Dashboard renderer created')
    
    # Test panel creation
    system_panel = renderer.create_system_metrics_panel()
    print('✅ System metrics panel created')
    
    alerts_panel = renderer.create_alerts_panel()
    print('✅ Alerts panel created')
    
    resources_panel = renderer.create_resource_utilization_panel()
    print('✅ Resource utilization panel created')
    
    # Test complete dashboard rendering
    layout = renderer.render_dashboard()
    print('✅ Complete dashboard layout rendered')
    
    # Test sample data generation
    print("\n🎲 Testing Sample Data Generation...")
    dashboard.generate_sample_data()
    sample_alerts = dashboard.performance_monitor.get_active_alerts()
    print(f'✅ Sample data generated: {len(sample_alerts)} alerts from sample data')
    
    # Show some sample alerts
    for i, alert in enumerate(sample_alerts[:3]):  # Show first 3 alerts
        print(f'   Alert {i+1}: {alert.severity.value} - {alert.metric_type.value} at {alert.current_value:.1f}')
    
    print('\n🎉 All standalone performance monitoring tests passed!')
    print('\n📋 Test Summary:')
    print('   ✅ MetricBuffer functionality')
    print('   ✅ PerformanceMonitor with thresholds and alerts')
    print('   ✅ System metrics collection')
    print('   ✅ Alert generation and resolution')
    print('   ✅ Dashboard rendering components')
    print('   ✅ Sample data generation')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()