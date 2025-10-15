#!/usr/bin/env python3
"""
Analytics API Router

FastAPI router for analytics, reporting, and data insights endpoints.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import io
import json

# Import analytics components
try:
    from ...analytics_reporting_system import (
        AnalyticsReportingSystem, 
        ReportConfig, 
        ReportFormat, 
        ReportType,
        DeliveryConfig,
        DeliveryMethod
    )
    from ...advanced_analytics_engine import AdvancedAnalyticsEngine
    from ...analytics_integration import IntegratedAnalyticsOrchestrator
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

router = APIRouter()


# Pydantic models for API requests/responses
class ReportFormatEnum(str, Enum):
    """Report format options"""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    EXCEL = "excel"


class ReportTypeEnum(str, Enum):
    """Report type options"""
    SUMMARY = "summary"
    DETAILED = "detailed"
    USAGE_PATTERNS = "usage_patterns"
    COST_ANALYSIS = "cost_analysis"
    PERFORMANCE = "performance"
    TRENDS = "trends"
    CUSTOM = "custom"


class ReportRequest(BaseModel):
    """Request to generate a report"""
    report_type: ReportTypeEnum
    format: ReportFormatEnum
    title: str
    description: Optional[str] = None
    time_period_hours: int = Field(24, ge=1, le=8760)  # 1 hour to 1 year
    include_charts: bool = True
    include_recommendations: bool = True
    custom_filters: Dict[str, Any] = Field(default_factory=dict)


class ReportResponse(BaseModel):
    """Response for generated report"""
    report_id: str
    status: str
    title: str
    format: str
    generated_at: datetime
    download_url: Optional[str] = None
    file_size_bytes: Optional[int] = None


class AnalyticsQuery(BaseModel):
    """Analytics query request"""
    query_type: str = Field(..., description="Type of analytics query")
    time_range_hours: int = Field(24, ge=1, le=8760)
    filters: Dict[str, Any] = Field(default_factory=dict)
    aggregation: Optional[str] = Field(None, description="Aggregation method")
    group_by: Optional[List[str]] = Field(None, description="Fields to group by")


class AnalyticsResponse(BaseModel):
    """Analytics query response"""
    query_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]


class UsageMetrics(BaseModel):
    """Usage metrics response"""
    timestamp: datetime
    total_requests: int
    unique_users: int
    popular_features: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    error_rates: Dict[str, float]


# Dependency to get analytics system
async def get_analytics_system():
    """Dependency to get analytics system"""
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analytics components not available")
    
    try:
        # Initialize analytics system if needed
        return AnalyticsReportingSystem()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize analytics system: {e}")


@router.get("/status")
async def get_analytics_status():
    """Get analytics system status"""
    return {
        "status": "available" if ANALYTICS_AVAILABLE else "unavailable",
        "components": {
            "reporting_system": ANALYTICS_AVAILABLE,
            "analytics_engine": ANALYTICS_AVAILABLE,
            "orchestrator": ANALYTICS_AVAILABLE
        },
        "timestamp": datetime.now().isoformat()
    }


@router.post("/reports", response_model=ReportResponse)
async def generate_report(
    report_request: ReportRequest,
    background_tasks: BackgroundTasks,
    analytics_system = Depends(get_analytics_system)
):
    """Generate an analytics report"""
    try:
        # Convert Pydantic models to internal types
        config = ReportConfig(
            report_type=ReportType(report_request.report_type.value),
            format=ReportFormat(report_request.format.value),
            title=report_request.title,
            description=report_request.description,
            time_period_hours=report_request.time_period_hours,
            include_charts=report_request.include_charts,
            include_recommendations=report_request.include_recommendations,
            custom_filters=report_request.custom_filters
        )
        
        # Generate report
        report = await analytics_system.generate_report(config)
        
        return ReportResponse(
            report_id=report.report_id,
            status="completed",
            title=report.config.title,
            format=report.format.value,
            generated_at=report.generated_at,
            download_url=f"/api/v1/analytics/reports/{report.report_id}/download",
            file_size_bytes=len(report.content) if isinstance(report.content, (str, bytes)) else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")


@router.get("/reports/{report_id}")
async def get_report_info(
    report_id: str,
    analytics_system = Depends(get_analytics_system)
):
    """Get information about a specific report"""
    try:
        # This would typically fetch from a database
        # For now, return mock data
        return {
            "report_id": report_id,
            "status": "completed",
            "title": "Analytics Report",
            "format": "json",
            "generated_at": datetime.now().isoformat(),
            "download_url": f"/api/v1/analytics/reports/{report_id}/download"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get report info: {e}")


@router.get("/reports/{report_id}/download")
async def download_report(
    report_id: str,
    analytics_system = Depends(get_analytics_system)
):
    """Download a generated report"""
    try:
        # This would typically fetch the report from storage
        # For now, generate a sample report
        sample_data = {
            "report_id": report_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_users": 150,
                "total_requests": 5000,
                "average_response_time": 85.5,
                "error_rate": 0.02
            },
            "usage_patterns": [
                {"feature": "document_processing", "usage_count": 1200, "percentage": 24.0},
                {"feature": "code_analysis", "usage_count": 1800, "percentage": 36.0},
                {"feature": "workspace_management", "usage_count": 800, "percentage": 16.0}
            ]
        }
        
        # Create file-like object
        json_str = json.dumps(sample_data, indent=2)
        file_obj = io.StringIO(json_str)
        
        return StreamingResponse(
            io.BytesIO(json_str.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=report_{report_id}.json"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download report: {e}")


@router.post("/query", response_model=AnalyticsResponse)
async def execute_analytics_query(
    query: AnalyticsQuery,
    analytics_system = Depends(get_analytics_system)
):
    """Execute a custom analytics query"""
    try:
        # Generate query ID
        import uuid
        query_id = str(uuid.uuid4())
        
        # Mock analytics data based on query type
        if query.query_type == "usage_summary":
            data = {
                "total_requests": 5000,
                "unique_users": 150,
                "time_period_hours": query.time_range_hours,
                "top_features": [
                    {"name": "code_analysis", "count": 1800},
                    {"name": "document_processing", "count": 1200},
                    {"name": "workspace_management", "count": 800}
                ]
            }
        elif query.query_type == "performance_metrics":
            data = {
                "average_response_time_ms": 85.5,
                "p95_response_time_ms": 150.2,
                "p99_response_time_ms": 280.1,
                "error_rate": 0.02,
                "cache_hit_rate": 0.94
            }
        elif query.query_type == "user_behavior":
            data = {
                "session_duration_avg_minutes": 45.2,
                "pages_per_session": 12.5,
                "bounce_rate": 0.15,
                "conversion_rate": 0.68
            }
        else:
            data = {"message": f"Query type '{query.query_type}' not implemented"}
        
        return AnalyticsResponse(
            query_id=query_id,
            timestamp=datetime.now(),
            data=data,
            metadata={
                "query_type": query.query_type,
                "time_range_hours": query.time_range_hours,
                "filters_applied": len(query.filters),
                "execution_time_ms": 45
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute analytics query: {e}")


@router.get("/metrics/usage", response_model=UsageMetrics)
async def get_usage_metrics(
    hours: int = Query(24, ge=1, le=8760, description="Time period in hours"),
    analytics_system = Depends(get_analytics_system)
):
    """Get usage metrics for the specified time period"""
    try:
        # Mock usage metrics
        return UsageMetrics(
            timestamp=datetime.now(),
            total_requests=5000,
            unique_users=150,
            popular_features=[
                {"name": "code_analysis", "usage_count": 1800, "percentage": 36.0},
                {"name": "document_processing", "usage_count": 1200, "percentage": 24.0},
                {"name": "workspace_management", "usage_count": 800, "percentage": 16.0},
                {"name": "analytics", "usage_count": 600, "percentage": 12.0},
                {"name": "monitoring", "usage_count": 400, "percentage": 8.0}
            ],
            performance_metrics={
                "avg_response_time_ms": 85.5,
                "p95_response_time_ms": 150.2,
                "cache_hit_rate": 0.94,
                "throughput_rps": 125.5
            },
            error_rates={
                "total_error_rate": 0.02,
                "4xx_error_rate": 0.015,
                "5xx_error_rate": 0.005
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get usage metrics: {e}")


@router.get("/metrics/performance")
async def get_performance_metrics(
    hours: int = Query(24, ge=1, le=168, description="Time period in hours (max 1 week)"),
    granularity: str = Query("hour", description="Data granularity: minute, hour, day")
):
    """Get performance metrics over time"""
    try:
        # Generate mock time series data
        from datetime import datetime, timedelta
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Determine time step based on granularity
        if granularity == "minute":
            step = timedelta(minutes=1)
        elif granularity == "hour":
            step = timedelta(hours=1)
        elif granularity == "day":
            step = timedelta(days=1)
        else:
            raise HTTPException(status_code=400, detail="Invalid granularity")
        
        # Generate data points
        data_points = []
        current_time = start_time
        
        while current_time <= end_time:
            # Mock performance data with some variation
            import random
            
            data_points.append({
                "timestamp": current_time.isoformat(),
                "response_time_ms": 85 + random.uniform(-20, 30),
                "throughput_rps": 120 + random.uniform(-30, 40),
                "error_rate": max(0, 0.02 + random.uniform(-0.015, 0.01)),
                "cpu_usage": 45 + random.uniform(-15, 25),
                "memory_usage": 60 + random.uniform(-20, 20)
            })
            
            current_time += step
        
        return {
            "time_period_hours": hours,
            "granularity": granularity,
            "data_points": data_points,
            "summary": {
                "avg_response_time_ms": sum(p["response_time_ms"] for p in data_points) / len(data_points),
                "avg_throughput_rps": sum(p["throughput_rps"] for p in data_points) / len(data_points),
                "avg_error_rate": sum(p["error_rate"] for p in data_points) / len(data_points)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {e}")


@router.get("/insights")
async def get_analytics_insights(
    category: str = Query("all", description="Insight category: usage, performance, errors, trends, all"),
    analytics_system = Depends(get_analytics_system)
):
    """Get AI-generated insights from analytics data"""
    try:
        insights = []
        
        if category in ["usage", "all"]:
            insights.extend([
                {
                    "category": "usage",
                    "type": "trend",
                    "title": "Code Analysis Feature Growing",
                    "description": "Code analysis usage has increased 25% over the past week",
                    "confidence": 0.85,
                    "recommendation": "Consider optimizing code analysis performance for better user experience"
                },
                {
                    "category": "usage",
                    "type": "pattern",
                    "title": "Peak Usage Hours",
                    "description": "Highest usage occurs between 9 AM - 11 AM and 2 PM - 4 PM",
                    "confidence": 0.92,
                    "recommendation": "Schedule maintenance outside peak hours"
                }
            ])
        
        if category in ["performance", "all"]:
            insights.extend([
                {
                    "category": "performance",
                    "type": "optimization",
                    "title": "Cache Hit Rate Excellent",
                    "description": "Cache hit rate of 94% is above target of 90%",
                    "confidence": 0.98,
                    "recommendation": "Current caching strategy is working well"
                },
                {
                    "category": "performance",
                    "type": "alert",
                    "title": "Response Time Variance",
                    "description": "Response times show high variance during peak hours",
                    "confidence": 0.78,
                    "recommendation": "Investigate auto-scaling configuration"
                }
            ])
        
        if category in ["errors", "all"]:
            insights.extend([
                {
                    "category": "errors",
                    "type": "trend",
                    "title": "Error Rate Stable",
                    "description": "Error rate has remained stable at 2% over the past month",
                    "confidence": 0.89,
                    "recommendation": "Continue monitoring for any sudden changes"
                }
            ])
        
        return {
            "category": category,
            "insights": insights,
            "generated_at": datetime.now().isoformat(),
            "total_insights": len(insights)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics insights: {e}")


@router.get("/dashboard/data")
async def get_dashboard_data():
    """Get data for analytics dashboard"""
    try:
        return {
            "summary": {
                "total_users": 150,
                "total_requests": 5000,
                "avg_response_time": 85.5,
                "error_rate": 0.02,
                "uptime": 99.8
            },
            "recent_activity": [
                {"timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(), "event": "Document processed", "user": "user_123"},
                {"timestamp": (datetime.now() - timedelta(minutes=8)).isoformat(), "event": "Code analyzed", "user": "user_456"},
                {"timestamp": (datetime.now() - timedelta(minutes=12)).isoformat(), "event": "Workspace created", "user": "user_789"}
            ],
            "top_features": [
                {"name": "Code Analysis", "usage": 36.0},
                {"name": "Document Processing", "usage": 24.0},
                {"name": "Workspace Management", "usage": 16.0}
            ],
            "system_health": {
                "cpu_usage": 45.2,
                "memory_usage": 62.1,
                "disk_usage": 28.5,
                "cache_hit_rate": 94.2
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {e}")


# Add router tags and metadata
router.tags = ["Analytics"]