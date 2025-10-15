#!/usr/bin/env python3
"""
FastAPI Server Demo

Demonstrates the Xencode FastAPI application with all endpoints and features.
"""

import asyncio
import time
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich import box

console = Console()


def test_api_endpoints(base_url: str = "http://localhost:8000"):
    """Test various API endpoints"""
    
    console.print(f"\n[bold blue]🌐 Testing API Endpoints at {base_url}[/bold blue]")
    
    endpoints_to_test = [
        ("GET", "/health", "Basic health check"),
        ("GET", "/health/detailed", "Detailed health check"),
        ("GET", "/metrics", "Application metrics"),
        ("GET", "/info", "Application information"),
        ("GET", "/api/v1/monitoring/health", "Monitoring system health"),
        ("GET", "/api/v1/monitoring/resources", "Resource usage"),
        ("GET", "/api/v1/monitoring/statistics", "Monitoring statistics"),
        ("GET", "/api/v1/analytics/status", "Analytics system status"),
        ("GET", "/api/v1/analytics/metrics/usage", "Usage metrics"),
        ("GET", "/api/v1/analytics/insights", "Analytics insights"),
        ("GET", "/api/v1/analytics/dashboard/data", "Dashboard data"),
    ]
    
    results = []
    
    for method, endpoint, description in endpoints_to_test:
        try:
            url = f"{base_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json={}, timeout=10)
            else:
                continue
            
            status = "✅ SUCCESS" if response.status_code < 400 else "❌ FAILED"
            results.append({
                "endpoint": endpoint,
                "description": description,
                "status_code": response.status_code,
                "status": status,
                "response_time": response.elapsed.total_seconds()
            })
            
            console.print(f"  {status} {endpoint} ({response.status_code}) - {description}")
            
        except requests.exceptions.RequestException as e:
            results.append({
                "endpoint": endpoint,
                "description": description,
                "status_code": "ERROR",
                "status": "❌ ERROR",
                "response_time": 0
            })
            console.print(f"  ❌ ERROR {endpoint} - {str(e)}")
        
        time.sleep(0.1)  # Small delay between requests
    
    return results


def test_monitoring_endpoints(base_url: str = "http://localhost:8000"):
    """Test monitoring-specific endpoints"""
    
    console.print(f"\n[bold blue]📊 Testing Monitoring Endpoints[/bold blue]")
    
    monitoring_tests = [
        {
            "name": "Resource Usage",
            "method": "GET",
            "endpoint": "/api/v1/monitoring/resources",
            "expected_fields": ["resource_type", "current_usage", "timestamp"]
        },
        {
            "name": "System Health",
            "method": "GET", 
            "endpoint": "/api/v1/monitoring/health",
            "expected_fields": ["status", "timestamp", "components"]
        },
        {
            "name": "Performance Metrics",
            "method": "GET",
            "endpoint": "/api/v1/monitoring/metrics",
            "expected_fields": ["timestamp", "cpu_usage", "memory_usage"]
        },
        {
            "name": "Active Alerts",
            "method": "GET",
            "endpoint": "/api/v1/monitoring/alerts",
            "expected_fields": ["alerts", "total_count"]
        }
    ]
    
    for test in monitoring_tests:
        try:
            url = f"{base_url}{test['endpoint']}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for expected fields
                missing_fields = []
                if isinstance(data, dict):
                    for field in test["expected_fields"]:
                        if field not in data and not any(field in item for item in data.values() if isinstance(item, dict)):
                            missing_fields.append(field)
                elif isinstance(data, list) and data:
                    for field in test["expected_fields"]:
                        if field not in data[0]:
                            missing_fields.append(field)
                
                if missing_fields:
                    console.print(f"  ⚠️  {test['name']}: Missing fields {missing_fields}")
                else:
                    console.print(f"  ✅ {test['name']}: All expected fields present")
            else:
                console.print(f"  ❌ {test['name']}: HTTP {response.status_code}")
                
        except Exception as e:
            console.print(f"  ❌ {test['name']}: {str(e)}")


def test_analytics_endpoints(base_url: str = "http://localhost:8000"):
    """Test analytics-specific endpoints"""
    
    console.print(f"\n[bold blue]📈 Testing Analytics Endpoints[/bold blue]")
    
    # Test report generation
    try:
        report_data = {
            "report_type": "summary",
            "format": "json",
            "title": "Test Report",
            "time_period_hours": 24
        }
        
        response = requests.post(
            f"{base_url}/api/v1/analytics/reports",
            json=report_data,
            timeout=30
        )
        
        if response.status_code == 200:
            report_info = response.json()
            console.print(f"  ✅ Report Generation: Created report {report_info.get('report_id', 'unknown')}")
            
            # Test report download
            if "report_id" in report_info:
                download_url = f"{base_url}/api/v1/analytics/reports/{report_info['report_id']}/download"
                download_response = requests.get(download_url, timeout=10)
                
                if download_response.status_code == 200:
                    console.print(f"  ✅ Report Download: Successfully downloaded report")
                else:
                    console.print(f"  ❌ Report Download: HTTP {download_response.status_code}")
        else:
            console.print(f"  ❌ Report Generation: HTTP {response.status_code}")
            
    except Exception as e:
        console.print(f"  ❌ Report Generation: {str(e)}")
    
    # Test analytics query
    try:
        query_data = {
            "query_type": "usage_summary",
            "time_range_hours": 24,
            "filters": {}
        }
        
        response = requests.post(
            f"{base_url}/api/v1/analytics/query",
            json=query_data,
            timeout=10
        )
        
        if response.status_code == 200:
            query_result = response.json()
            console.print(f"  ✅ Analytics Query: Executed query {query_result.get('query_id', 'unknown')}")
        else:
            console.print(f"  ❌ Analytics Query: HTTP {response.status_code}")
            
    except Exception as e:
        console.print(f"  ❌ Analytics Query: {str(e)}")


def test_resource_management_integration(base_url: str = "http://localhost:8000"):
    """Test resource management integration"""
    
    console.print(f"\n[bold blue]🔧 Testing Resource Management Integration[/bold blue]")
    
    try:
        # Test cleanup trigger
        response = requests.post(
            f"{base_url}/api/v1/monitoring/cleanup?priority=low",
            timeout=30
        )
        
        if response.status_code == 200:
            cleanup_result = response.json()
            console.print(f"  ✅ Cleanup Trigger: Executed {cleanup_result.get('tasks_executed', 0)} tasks")
        else:
            console.print(f"  ❌ Cleanup Trigger: HTTP {response.status_code}")
    
    except Exception as e:
        console.print(f"  ❌ Cleanup Trigger: {str(e)}")
    
    try:
        # Test memory snapshot
        response = requests.post(
            f"{base_url}/api/v1/monitoring/memory/snapshot?label=api_test",
            timeout=10
        )
        
        if response.status_code == 200:
            console.print(f"  ✅ Memory Snapshot: Successfully taken")
        else:
            console.print(f"  ❌ Memory Snapshot: HTTP {response.status_code}")
    
    except Exception as e:
        console.print(f"  ❌ Memory Snapshot: {str(e)}")
    
    try:
        # Test memory analysis
        response = requests.get(
            f"{base_url}/api/v1/monitoring/memory/analysis",
            timeout=10
        )
        
        if response.status_code == 200:
            analysis = response.json()
            console.print(f"  ✅ Memory Analysis: Retrieved analysis data")
        else:
            console.print(f"  ❌ Memory Analysis: HTTP {response.status_code}")
    
    except Exception as e:
        console.print(f"  ❌ Memory Analysis: {str(e)}")


def display_api_summary(results: list):
    """Display a summary of API test results"""
    
    console.print(f"\n[bold blue]📋 API Test Summary[/bold blue]")
    
    table = Table(title="Endpoint Test Results", box=box.ROUNDED)
    table.add_column("Endpoint", style="cyan", width=40)
    table.add_column("Status", style="white", width=12)
    table.add_column("Code", style="yellow", width=8)
    table.add_column("Time (s)", style="green", width=10)
    table.add_column("Description", style="dim", width=30)
    
    success_count = 0
    total_count = len(results)
    
    for result in results:
        status_style = "green" if "SUCCESS" in result["status"] else "red"
        
        table.add_row(
            result["endpoint"],
            f"[{status_style}]{result['status']}[/{status_style}]",
            str(result["status_code"]),
            f"{result['response_time']:.3f}" if isinstance(result['response_time'], float) else "N/A",
            result["description"]
        )
        
        if "SUCCESS" in result["status"]:
            success_count += 1
    
    console.print(table)
    
    # Summary statistics
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    summary_panel = Panel(
        f"[bold green]✅ Successful: {success_count}/{total_count}[/bold green]\n"
        f"[bold blue]📊 Success Rate: {success_rate:.1f}%[/bold blue]\n"
        f"[bold yellow]⚡ Average Response Time: {sum(r['response_time'] for r in results if isinstance(r['response_time'], float)) / len([r for r in results if isinstance(r['response_time'], float)]):.3f}s[/bold yellow]",
        title="Test Summary",
        border_style="green" if success_rate > 80 else "yellow" if success_rate > 60 else "red"
    )
    
    console.print(summary_panel)


def main():
    """Main demo function"""
    console.print(Panel.fit(
        "[bold cyan]Xencode FastAPI Server Demo[/bold cyan]\n"
        "Testing comprehensive REST API endpoints and functionality",
        border_style="blue"
    ))
    
    # Check if server is running
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            console.print(f"[red]❌ Server not responding properly at {base_url}[/red]")
            console.print("[yellow]💡 Start the server with: python xencode/api/main.py[/yellow]")
            return
    except requests.exceptions.RequestException:
        console.print(f"[red]❌ Cannot connect to server at {base_url}[/red]")
        console.print("[yellow]💡 Start the server with: python xencode/api/main.py[/yellow]")
        return
    
    console.print(f"[green]✅ Server is running at {base_url}[/green]")
    
    # Run tests
    results = test_api_endpoints(base_url)
    test_monitoring_endpoints(base_url)
    test_analytics_endpoints(base_url)
    test_resource_management_integration(base_url)
    
    # Display summary
    display_api_summary(results)
    
    console.print("\n[bold green]✅ FastAPI Server Demo Completed![/bold green]")
    console.print(f"[dim]🌐 API Documentation: {base_url}/docs[/dim]")
    console.print(f"[dim]📚 ReDoc Documentation: {base_url}/redoc[/dim]")


if __name__ == "__main__":
    main()