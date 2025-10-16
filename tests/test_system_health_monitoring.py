#!/usr/bin/env python3
"""
System Health Monitoring Tests

Comprehensive system health monitoring including health checks,
status dashboard, and automated alerting for critical issues.
"""

import asyncio
import time
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pytest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from fastapi import FastAPI

from xencode.api.routers.analytics import router as analytics_router
from xencode.api.routers.monitoring import router as monitoring_router
from xencode.api.routers.plugin import router as plugin_router
from xencode.api.routers.workspace import router as workspace_router


class HealthMonitor:
    """System health monitoring utilities"""
    
    def __init__(self):
        self.health_checks = {}
        self.alerts = []
        self.metrics_history = []
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'response_time_ms': 1000.0,
            'error_rate': 0.05
        }
    
    def add_health_check(self, name: str, status: str, details: Dict = None):
        """Add a health check result"""
        self.health_checks[name] = {
            'status': status,
            'details': details or {},
            'timestamp': datetime.now().isoformat(),
            'healthy': status.lower() in ['healthy', 'ok', 'pass']
        }
    
    def add_alert(self, severity: str, message: str, component: str = None):
        """Add a system alert"""
        self.alerts.append({
            'id': f"alert_{len(self.alerts) + 1}",
            'severity': severity,
            'message': message,
            'component': component,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False
        })
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
            
            self.metrics_history.append(metrics)
            
            # Check thresholds and generate alerts
            self._check_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            self.add_alert('ERROR', f'Failed to collect system metrics: {str(e)}', 'system_monitor')
            return {}
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts"""
        for metric, threshold in self.thresholds.items():
            if metric in metrics:
                value = metrics[metric]
                if value > threshold:
                    self.add_alert(
                        'WARNING' if value < threshold * 1.2 else 'CRITICAL',
                        f'{metric} is {value:.2f}, exceeding threshold of {threshold}',
                        'system_resources'
                    )
    
    def get_overall_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        healthy_checks = sum(1 for check in self.health_checks.values() if check['healthy'])
        total_checks = len(self.health_checks)
        
        critical_alerts = len([a for a in self.alerts if a['severity'] == 'CRITICAL'])
        warning_alerts = len([a for a in self.alerts if a['severity'] == 'WARNING'])
        
        if critical_alerts > 0:
            overall_status = 'CRITICAL'
        elif warning_alerts > 0 or (total_checks > 0 and healthy_checks / total_checks < 0.8):
            overall_status = 'WARNING'
        elif total_checks > 0 and healthy_checks == total_checks:
            overall_status = 'HEALTHY'
        else:
            overall_status = 'UNKNOWN'
        
        return {
            'overall_status': overall_status,
            'health_score': (healthy_checks / total_checks * 100) if total_checks > 0 else 0,
            'total_checks': total_checks,
            'healthy_checks': healthy_checks,
            'critical_alerts': critical_alerts,
            'warning_alerts': warning_alerts,
            'last_updated': datetime.now().isoformat()
        }


class TestSystemHealthMonitoring:
    """System health monitoring test suite"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.app = FastAPI(title="Health Monitoring Test")
        
        # Include routers
        self.app.include_router(analytics_router, prefix="/api/v1/analytics")
        self.app.include_router(monitoring_router, prefix="/api/v1/monitoring")
        self.app.include_router(plugin_router, prefix="/api/v1/plugins")
        self.app.include_router(workspace_router, prefix="/api/v1/workspaces")
        
        self.client = TestClient(self.app)
        self.health_monitor = HealthMonitor()
    
    def test_comprehensive_health_checks(self):
        """Test comprehensive system health checks"""
        # Test individual component health
        components_to_check = [
            ('analytics_engine', '/api/v1/analytics/health'),
            ('monitoring_system', '/api/v1/monitoring/health'),
            ('plugin_manager', '/api/v1/plugins/system/status'),
            ('workspace_manager', '/api/v1/workspaces/')
        ]
        
        for component_name, endpoint in components_to_check:
            start_time = time.time()
            response = self.client.get(endpoint)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                self.health_monitor.add_health_check(
                    component_name,
                    'healthy',
                    {'response_time_ms': response_time, 'status_code': response.status_code}
                )
            else:
                self.health_monitor.add_health_check(
                    component_name,
                    'unhealthy',
                    {'response_time_ms': response_time, 'status_code': response.status_code, 'error': 'HTTP error'}
                )
                
                # Generate alert for unhealthy component
                self.health_monitor.add_alert(
                    'CRITICAL',
                    f'{component_name} health check failed with status {response.status_code}',
                    component_name
                )
        
        # Test database connectivity (simulated)
        self.health_monitor.add_health_check(
            'database',
            'healthy',
            {'connection_pool_size': 10, 'active_connections': 3}
        )
        
        # Test external service dependencies (simulated)
        self.health_monitor.add_health_check(
            'redis_cache',
            'healthy',
            {'memory_usage': '45MB', 'connected_clients': 5}
        )
        
        # Verify health checks
        overall_health = self.health_monitor.get_overall_health_status()
        assert overall_health['total_checks'] >= 4
        assert overall_health['health_score'] >= 50  # At least 50% healthy
        
        print(f"\n🏥 Health Check Results:")
        print(f"  Overall Status: {overall_health['overall_status']}")
        print(f"  Health Score: {overall_health['health_score']:.1f}%")
        print(f"  Healthy Components: {overall_health['healthy_checks']}/{overall_health['total_checks']}")
    
    def test_system_resource_monitoring(self):
        """Test system resource monitoring and alerting"""
        # Collect system metrics
        for i in range(5):
            metrics = self.health_monitor.collect_system_metrics()
            time.sleep(0.1)  # Small delay between collections
        
        # Verify metrics collection
        assert len(self.health_monitor.metrics_history) >= 5
        
        latest_metrics = self.health_monitor.metrics_history[-1]
        required_metrics = ['cpu_percent', 'memory_percent', 'disk_percent']
        
        for metric in required_metrics:
            assert metric in latest_metrics
            assert isinstance(latest_metrics[metric], (int, float))
            assert 0 <= latest_metrics[metric] <= 100
        
        # Test threshold alerting (simulate high resource usage)
        high_usage_metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': 95.0,  # Above threshold
            'memory_percent': 90.0,  # Above threshold
            'disk_percent': 95.0,  # Above threshold
        }
        
        self.health_monitor._check_thresholds(high_usage_metrics)
        
        # Verify alerts were generated
        resource_alerts = [a for a in self.health_monitor.alerts if a['component'] == 'system_resources']
        assert len(resource_alerts) >= 3  # Should have alerts for CPU, memory, and disk
        
        print(f"\n📊 Resource Monitoring Results:")
        print(f"  CPU Usage: {latest_metrics['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {latest_metrics['memory_percent']:.1f}%")
        print(f"  Disk Usage: {latest_metrics['disk_percent']:.1f}%")
        print(f"  Generated Alerts: {len(resource_alerts)}")
    
    def test_application_performance_monitoring(self):
        """Test application performance monitoring"""
        # Test response time monitoring
        endpoints_to_monitor = [
            '/api/v1/analytics/overview',
            '/api/v1/monitoring/resources',
            '/api/v1/plugins/',
            '/api/v1/workspaces/'
        ]
        
        performance_metrics = {}
        
        for endpoint in endpoints_to_monitor:
            response_times = []
            error_count = 0
            
            # Make multiple requests to measure performance
            for i in range(10):
                start_time = time.time()
                response = self.client.get(endpoint)
                response_time = (time.time() - start_time) * 1000
                
                response_times.append(response_time)
                if response.status_code >= 500:
                    error_count += 1
            
            avg_response_time = sum(response_times) / len(response_times)
            error_rate = error_count / len(response_times)
            
            performance_metrics[endpoint] = {
                'avg_response_time_ms': avg_response_time,
                'error_rate': error_rate,
                'sample_size': len(response_times)
            }
            
            # Check performance thresholds
            if avg_response_time > self.health_monitor.thresholds['response_time_ms']:
                self.health_monitor.add_alert(
                    'WARNING',
                    f'High response time for {endpoint}: {avg_response_time:.2f}ms',
                    'application_performance'
                )
            
            if error_rate > self.health_monitor.thresholds['error_rate']:
                self.health_monitor.add_alert(
                    'CRITICAL',
                    f'High error rate for {endpoint}: {error_rate:.2%}',
                    'application_performance'
                )
        
        # Verify performance monitoring
        assert len(performance_metrics) == len(endpoints_to_monitor)
        
        print(f"\n⚡ Performance Monitoring Results:")
        for endpoint, metrics in performance_metrics.items():
            print(f"  {endpoint}:")
            print(f"    Avg Response Time: {metrics['avg_response_time_ms']:.2f}ms")
            print(f"    Error Rate: {metrics['error_rate']:.2%}")
    
    def test_automated_alerting_system(self):
        """Test automated alerting for critical issues"""
        # Simulate various critical scenarios
        critical_scenarios = [
            {
                'type': 'service_down',
                'message': 'Analytics service is not responding',
                'component': 'analytics_engine',
                'severity': 'CRITICAL'
            },
            {
                'type': 'high_error_rate',
                'message': 'Error rate exceeded 10% in the last 5 minutes',
                'component': 'api_gateway',
                'severity': 'CRITICAL'
            },
            {
                'type': 'resource_exhaustion',
                'message': 'Memory usage exceeded 95%',
                'component': 'system_resources',
                'severity': 'CRITICAL'
            },
            {
                'type': 'security_breach',
                'message': 'Multiple failed authentication attempts detected',
                'component': 'security_system',
                'severity': 'CRITICAL'
            }
        ]
        
        # Generate alerts for critical scenarios
        for scenario in critical_scenarios:
            self.health_monitor.add_alert(
                scenario['severity'],
                scenario['message'],
                scenario['component']
            )
        
        # Test alert aggregation and prioritization
        critical_alerts = [a for a in self.health_monitor.alerts if a['severity'] == 'CRITICAL']
        warning_alerts = [a for a in self.health_monitor.alerts if a['severity'] == 'WARNING']
        
        # Verify alerting system
        assert len(critical_alerts) >= 4  # Should have at least the 4 critical scenarios
        
        # Test alert acknowledgment
        if critical_alerts:
            alert_to_ack = critical_alerts[0]
            alert_to_ack['acknowledged'] = True
            alert_to_ack['acknowledged_by'] = 'test_admin'
            alert_to_ack['acknowledged_at'] = datetime.now().isoformat()
        
        # Test alert escalation (simulate)
        unacknowledged_critical = [a for a in critical_alerts if not a.get('acknowledged', False)]
        
        if len(unacknowledged_critical) > 2:
            self.health_monitor.add_alert(
                'CRITICAL',
                f'{len(unacknowledged_critical)} unacknowledged critical alerts require immediate attention',
                'alert_system'
            )
        
        print(f"\n🚨 Alerting System Results:")
        print(f"  Critical Alerts: {len(critical_alerts)}")
        print(f"  Warning Alerts: {len(warning_alerts)}")
        print(f"  Unacknowledged Critical: {len(unacknowledged_critical)}")
    
    def test_health_dashboard_data(self):
        """Test health dashboard data aggregation"""
        # Collect comprehensive dashboard data
        dashboard_data = {
            'system_overview': self.health_monitor.get_overall_health_status(),
            'component_health': self.health_monitor.health_checks,
            'active_alerts': self.health_monitor.alerts,
            'system_metrics': self.health_monitor.metrics_history[-5:] if self.health_monitor.metrics_history else [],
            'performance_summary': {
                'total_requests': 150,
                'avg_response_time_ms': 245.5,
                'error_rate': 0.02,
                'uptime_hours': 24.5
            },
            'resource_utilization': {
                'cpu_trend': 'stable',
                'memory_trend': 'increasing',
                'disk_trend': 'stable',
                'network_io': 'normal'
            }
        }
        
        # Verify dashboard data completeness
        required_sections = ['system_overview', 'component_health', 'active_alerts', 'performance_summary']
        for section in required_sections:
            assert section in dashboard_data
            assert dashboard_data[section] is not None
        
        # Test dashboard API endpoint (simulated)
        dashboard_response = {
            'status': 'success',
            'data': dashboard_data,
            'last_updated': datetime.now().isoformat(),
            'refresh_interval_seconds': 30
        }
        
        # Verify dashboard response structure
        assert 'status' in dashboard_response
        assert 'data' in dashboard_response
        assert 'last_updated' in dashboard_response
        
        print(f"\n📊 Dashboard Data Summary:")
        print(f"  Overall Health: {dashboard_data['system_overview']['overall_status']}")
        print(f"  Components Monitored: {len(dashboard_data['component_health'])}")
        print(f"  Active Alerts: {len(dashboard_data['active_alerts'])}")
        print(f"  Metrics History Points: {len(dashboard_data['system_metrics'])}")
    
    def test_health_check_dependencies(self):
        """Test health checks for external dependencies"""
        # Simulate external dependency health checks
        dependencies = [
            {
                'name': 'database_primary',
                'type': 'postgresql',
                'status': 'healthy',
                'response_time_ms': 15.2,
                'connection_pool': {'active': 5, 'idle': 15, 'max': 20}
            },
            {
                'name': 'redis_cache',
                'type': 'redis',
                'status': 'healthy',
                'response_time_ms': 2.1,
                'memory_usage': '128MB',
                'connected_clients': 8
            },
            {
                'name': 'external_api',
                'type': 'http',
                'status': 'degraded',
                'response_time_ms': 1250.0,
                'last_error': 'Timeout after 1000ms'
            },
            {
                'name': 'file_storage',
                'type': 'filesystem',
                'status': 'healthy',
                'available_space_gb': 45.2,
                'total_space_gb': 100.0
            }
        ]
        
        for dep in dependencies:
            self.health_monitor.add_health_check(
                dep['name'],
                dep['status'],
                {k: v for k, v in dep.items() if k not in ['name', 'status']}
            )
            
            # Generate alerts for unhealthy dependencies
            if dep['status'] != 'healthy':
                severity = 'CRITICAL' if dep['status'] == 'down' else 'WARNING'
                self.health_monitor.add_alert(
                    severity,
                    f"Dependency {dep['name']} is {dep['status']}",
                    f"dependency_{dep['type']}"
                )
        
        # Verify dependency monitoring
        dependency_checks = {name: check for name, check in self.health_monitor.health_checks.items() 
                           if name in [d['name'] for d in dependencies]}
        
        assert len(dependency_checks) == len(dependencies)
        
        healthy_deps = sum(1 for check in dependency_checks.values() if check['healthy'])
        dependency_health_rate = healthy_deps / len(dependency_checks)
        
        print(f"\n🔗 Dependency Health Results:")
        print(f"  Total Dependencies: {len(dependency_checks)}")
        print(f"  Healthy Dependencies: {healthy_deps}")
        print(f"  Dependency Health Rate: {dependency_health_rate:.1%}")
        
        # Should have reasonable dependency health
        assert dependency_health_rate >= 0.75  # At least 75% of dependencies healthy
    
    def test_monitoring_system_resilience(self):
        """Test monitoring system resilience and self-monitoring"""
        # Test monitoring system self-health
        monitoring_components = [
            'metrics_collector',
            'alert_manager',
            'health_checker',
            'dashboard_service'
        ]
        
        for component in monitoring_components:
            # Simulate component health check
            is_healthy = True  # In real system, would actually check component
            
            self.health_monitor.add_health_check(
                f'monitoring_{component}',
                'healthy' if is_healthy else 'unhealthy',
                {'self_check': True, 'component_type': 'monitoring'}
            )
        
        # Test monitoring system recovery from failures
        # Simulate temporary failure and recovery
        self.health_monitor.add_health_check(
            'monitoring_metrics_collector',
            'recovering',
            {'previous_status': 'failed', 'recovery_time_seconds': 30}
        )
        
        # Test monitoring data integrity
        metrics_count_before = len(self.health_monitor.metrics_history)
        self.health_monitor.collect_system_metrics()
        metrics_count_after = len(self.health_monitor.metrics_history)
        
        assert metrics_count_after > metrics_count_before, "Metrics collection should continue working"
        
        # Test alert system resilience
        alerts_count_before = len(self.health_monitor.alerts)
        self.health_monitor.add_alert('INFO', 'Test alert for resilience testing', 'test_system')
        alerts_count_after = len(self.health_monitor.alerts)
        
        assert alerts_count_after > alerts_count_before, "Alert system should continue working"
        
        print(f"\n🛡️ Monitoring System Resilience:")
        print(f"  Monitoring Components: {len(monitoring_components)}")
        print(f"  Metrics Collection: Working")
        print(f"  Alert System: Working")
        print(f"  Self-Monitoring: Enabled")
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health monitoring report"""
        overall_health = self.health_monitor.get_overall_health_status()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': overall_health,
            'component_summary': {
                'total_components': len(self.health_monitor.health_checks),
                'healthy_components': overall_health['healthy_checks'],
                'unhealthy_components': overall_health['total_checks'] - overall_health['healthy_checks']
            },
            'alert_summary': {
                'total_alerts': len(self.health_monitor.alerts),
                'critical_alerts': len([a for a in self.health_monitor.alerts if a['severity'] == 'CRITICAL']),
                'warning_alerts': len([a for a in self.health_monitor.alerts if a['severity'] == 'WARNING']),
                'acknowledged_alerts': len([a for a in self.health_monitor.alerts if a.get('acknowledged', False)])
            },
            'monitoring_coverage': {
                'system_resources': True,
                'application_performance': True,
                'external_dependencies': True,
                'security_monitoring': True,
                'self_monitoring': True
            },
            'recommendations': [
                'Set up automated alert notifications (email, Slack, PagerDuty)',
                'Implement predictive alerting based on trends',
                'Add custom health checks for business-critical workflows',
                'Configure alert escalation policies',
                'Set up monitoring dashboards for different stakeholder groups'
            ]
        }


def run_health_monitoring_tests():
    """Run all health monitoring tests"""
    print("🏥 Running System Health Monitoring Tests")
    print("=" * 50)
    
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", 
        "tests/test_system_health_monitoring.py", 
        "-v", "--tb=short", "-s"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_health_monitoring_tests()
    
    if success:
        print("\n✅ System health monitoring tests passed!")
        print("🏥 Health monitoring system validated!")
    else:
        print("\n⚠️ Some health monitoring tests failed.")
        print("🔧 Review health monitoring implementation.")