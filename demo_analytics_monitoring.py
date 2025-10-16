#!/usr/bin/env python3
"""
Demo: Analytics and Monitoring API

Demonstrates the analytics and monitoring endpoints including metrics collection,
dashboard data, system health monitoring, and comprehensive reporting.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the analytics and monitoring routers
from xencode.api.routers.analytics import router as analytics_router
from xencode.api.routers.monitoring import router as monitoring_router


def create_demo_app() -> FastAPI:
    """Create demo FastAPI application with analytics and monitoring"""
    
    app = FastAPI(
        title="Xencode Analytics & Monitoring Demo",
        description="Demo of analytics and monitoring with comprehensive observability",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["Analytics"])
    app.include_router(monitoring_router, prefix="/api/v1/monitoring", tags=["Monitoring"])
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Xencode Analytics & Monitoring Demo",
            "version": "1.0.0",
            "endpoints": {
                "analytics": "/api/v1/analytics",
                "monitoring": "/api/v1/monitoring",
                "docs": "/docs",
                "redoc": "/redoc"
            }
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "analytics_engine": "available",
                "monitoring_system": "available",
                "metrics_collector": "available",
                "performance_optimizer": "available"
            }
        }
    
    return app


async def demo_analytics_monitoring():
    """Demo analytics and monitoring operations"""
    
    print("📊 Xencode Analytics & Monitoring Demo")
    print("=" * 50)
    
    print("\n📈 Analytics Endpoints:")
    print("  GET    /api/v1/analytics/overview                - Analytics overview")
    print("  POST   /api/v1/analytics/metrics                - Record metrics")
    print("  GET    /api/v1/analytics/metrics                - Get metrics data")
    print("  POST   /api/v1/analytics/events                 - Record events")
    print("  GET    /api/v1/analytics/events                 - Get events data")
    print("  POST   /api/v1/analytics/reports                - Generate reports")
    print("  GET    /api/v1/analytics/reports/{id}           - Get report status")
    print("  GET    /api/v1/analytics/reports/{id}/download  - Download report")
    print("  GET    /api/v1/analytics/dashboard              - Dashboard data")
    print("  GET    /api/v1/analytics/health                 - Analytics health")
    print("  GET    /api/v1/analytics/insights               - AI insights")
    
    print("\n🔍 Monitoring Endpoints:")
    print("  GET    /api/v1/monitoring/health                - System health")
    print("  GET    /api/v1/monitoring/resources/{type}      - Resource usage")
    print("  GET    /api/v1/monitoring/resources             - All resources")
    print("  GET    /api/v1/monitoring/performance           - Performance metrics")
    print("  POST   /api/v1/monitoring/cleanup               - Trigger cleanup")
    print("  GET    /api/v1/monitoring/alerts                - System alerts")
    print("  POST   /api/v1/monitoring/alerts/{id}/acknowledge - Acknowledge alert")
    print("  GET    /api/v1/monitoring/processes             - Running processes")
    print("  GET    /api/v1/monitoring/network               - Network statistics")
    print("  GET    /api/v1/monitoring/disk                  - Disk statistics")
    print("  GET    /api/v1/monitoring/dashboard             - Monitoring dashboard")
    print("  POST   /api/v1/monitoring/config                - Update config")
    
    print("\n🔧 Key Features:")
    print("  ✅ Real-time metrics collection and aggregation")
    print("  ✅ Event tracking and analytics")
    print("  ✅ Comprehensive system health monitoring")
    print("  ✅ Resource usage tracking (CPU, Memory, Disk, Network)")
    print("  ✅ Performance metrics and optimization")
    print("  ✅ Alert system with severity levels")
    print("  ✅ Process monitoring and management")
    print("  ✅ Dashboard data for visualization")
    print("  ✅ Report generation and export")
    print("  ✅ AI-powered insights and recommendations")
    
    print("\n📊 Example Metric Recording:")
    metric_request = {
        "name": "response_time_ms",
        "value": 45.2,
        "metric_type": "gauge",
        "labels": {
            "endpoint": "/api/v1/plugins",
            "method": "GET",
            "status": "200"
        },
        "timestamp": datetime.now().isoformat()
    }
    print(json.dumps(metric_request, indent=2))
    
    print("\n📝 Example Event Recording:")
    event_request = {
        "event_type": "plugin_execution",
        "event_data": {
            "plugin_id": "file-operations",
            "method": "ls_dir",
            "execution_time_ms": 125,
            "success": True
        },
        "user_id": "user123",
        "session_id": "session456",
        "timestamp": datetime.now().isoformat()
    }
    print(json.dumps(event_request, indent=2))
    
    print("\n📋 Example Report Request:")
    report_request = {
        "report_type": "performance",
        "format": "json",
        "time_range": "1d",
        "filters": {
            "include_raw_data": False,
            "group_by": "hour"
        },
        "email_delivery": "admin@example.com"
    }
    print(json.dumps(report_request, indent=2))
    
    print("\n🎛️ Example Dashboard Request:")
    dashboard_request = {
        "dashboard_type": "overview",
        "time_range": "1h",
        "refresh_interval": 30
    }
    print(json.dumps(dashboard_request, indent=2))
    
    print("\n🚨 Example System Alert:")
    alert_example = {
        "id": "alert_memory_high",
        "severity": "high",
        "title": "High Memory Usage",
        "description": "Memory usage is at 87.3%",
        "resource_type": "memory",
        "threshold_value": 80.0,
        "current_value": 87.3,
        "created_at": datetime.now().isoformat(),
        "acknowledged": False,
        "resolved": False
    }
    print(json.dumps(alert_example, indent=2))
    
    print("\n🧹 Example Cleanup Request:")
    cleanup_request = {
        "resource_types": ["memory", "cache", "disk"],
        "priority": "normal",
        "force": False,
        "dry_run": False
    }
    print(json.dumps(cleanup_request, indent=2))
    
    print("\n📈 Example Performance Metrics:")
    performance_metrics = {
        "timestamp": datetime.now().isoformat(),
        "response_time_ms": 45.2,
        "throughput_rps": 125.8,
        "error_rate_percent": 0.02,
        "cache_hit_rate_percent": 94.5,
        "active_connections": 156,
        "queue_length": 3,
        "memory_usage_mb": 512.8,
        "cpu_usage_percent": 23.4
    }
    print(json.dumps(performance_metrics, indent=2))
    
    print("\n🔍 Example Resource Usage:")
    resource_usage = {
        "resource_type": "memory",
        "current_usage": 6.8,
        "peak_usage": 8.0,
        "average_usage": 5.4,
        "unit": "GB",
        "timestamp": datetime.now().isoformat(),
        "limit_soft": 6.4,
        "limit_hard": 8.0,
        "utilization_percent": 85.0,
        "trend": "increasing"
    }
    print(json.dumps(resource_usage, indent=2))
    
    print("\n🎯 Example AI Insights:")
    insights_example = {
        "insight_type": "performance",
        "recommendations": [
            {
                "title": "Optimize Database Queries",
                "description": "Query response time increased by 15% in the last hour",
                "priority": "medium",
                "impact": "performance",
                "estimated_improvement": "20% faster queries"
            },
            {
                "title": "Scale Memory Resources",
                "description": "Memory usage consistently above 80% threshold",
                "priority": "high",
                "impact": "stability",
                "estimated_improvement": "Prevent OOM errors"
            }
        ],
        "trends": {
            "user_growth": 0.12,
            "performance_trend": -0.05,
            "error_trend": 0.02,
            "resource_trend": 0.08
        },
        "generated_at": datetime.now().isoformat()
    }
    print(json.dumps(insights_example, indent=2))
    
    print("\n🎯 To test the API:")
    print("  1. Run: python demo_analytics_monitoring.py")
    print("  2. Open: http://localhost:8000/docs")
    print("  3. Try the analytics and monitoring endpoints")
    print("  4. Record metrics and events")
    print("  5. Generate reports and view dashboards")
    print("  6. Monitor system health and resources")


def main():
    """Main demo function"""
    
    # Run the demo
    asyncio.run(demo_analytics_monitoring())
    
    print("\n🚀 Starting FastAPI server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("📊 Analytics Overview: http://localhost:8000/api/v1/analytics/overview")
    print("🔍 System Health: http://localhost:8000/api/v1/monitoring/health")
    print("📈 Performance Metrics: http://localhost:8000/api/v1/monitoring/performance")
    print("⏹️  Press Ctrl+C to stop")
    
    # Create and run the app
    app = create_demo_app()
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )


if __name__ == "__main__":
    main()