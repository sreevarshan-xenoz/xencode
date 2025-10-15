#!/usr/bin/env python3
"""
Analytics API

RESTful API for external integrations with Xencode analytics system.
Provides endpoints for accessing analytics data, generating reports,
and managing scheduled reporting.

Key Features:
- RESTful API endpoints for analytics data access
- Report generation and download endpoints
- Scheduled reporting management
- Authentication and rate limiting
- Data filtering and aggregation
- WebSocket support for real-time updates
- OpenAPI/Swagger documentation
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import asdict

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Depends, Query, Path as PathParam, BackgroundTasks
    from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Mock classes for development without FastAPI
    class FastAPI:
        def __init__(self, *args, **kwargs): pass
        def get(self, *args, **kwargs): return lambda f: f
        def post(self, *args, **kwargs): return lambda f: f
        def put(self, *args, **kwargs): return lambda f: f
        def delete(self, *args, **kwargs): return lambda f: f
    
    class BaseModel:
        pass
    
    def Field(*args, **kwargs):
        return None
    
    class HTTPException(Exception):
        def __init__(self, status_code, detail): pass

# Import analytics components
try:
    from .analytics_reporting_system import (
        AnalyticsReportingSystem, ReportConfig, DeliveryConfig,
        ReportFormat, ReportType, DeliveryMethod
    )
    from .advanced_analytics_engine import AdvancedAnalyticsEngine
    from .analytics_integration import IntegratedAnalyticsOrchestrator
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    # Mock classes
    class AnalyticsReportingSystem:
        def __init__(self, *args, **kwargs): pass
        async def generate_report(self, *args, **kwargs): return None
    
    class AdvancedAnalyticsEngine:
        def __init__(self, *args, **kwargs): pass
    
    class IntegratedAnalyticsOrchestrator:
        def __init__(self, *args, **kwargs): pass


# Pydantic models for API
class AnalyticsSummaryResponse(BaseModel):
    """Response model for analytics summary"""
    patterns_detected: int = Field(..., description="Number of usage patterns detected")
    users_analyzed: int = Field(..., description="Number of users analyzed")
    optimizations_found: int = Field(..., description="Number of cost optimizations found")
    total_potential_savings: float = Field(..., description="Total potential cost savings")
    analysis_period_hours: int = Field(..., description="Analysis time period in hours")
    generated_at: str = Field(..., description="Analysis generation timestamp")


class UsagePatternResponse(BaseModel):
    """Response model for usage patterns"""
    pattern_id: str = Field(..., description="Unique pattern identifier")
    pattern_type: str = Field(..., description="Type of usage pattern")
    description: str = Field(..., description="Pattern description")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    frequency: float = Field(..., ge=0, le=1, description="Pattern frequency (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional pattern metadata")


class CostOptimizationResponse(BaseModel):
    """Response model for cost optimizations"""
    optimization_id: str = Field(..., description="Unique optimization identifier")
    optimization_type: str = Field(..., description="Type of cost optimization")
    title: str = Field(..., description="Optimization title")
    description: str = Field(..., description="Optimization description")
    potential_savings: float = Field(..., ge=0, description="Potential cost savings")
    implementation_effort: str = Field(..., description="Implementation effort level")
    impact_score: float = Field(..., ge=0, le=1, description="Impact score (0-1)")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")


class ReportGenerationRequest(BaseModel):
    """Request model for report generation"""
    report_type: str = Field(..., description="Type of report to generate")
    format: str = Field(..., description="Output format (json, csv, html, pdf, markdown)")
    title: str = Field(..., description="Report title")
    description: Optional[str] = Field(None, description="Report description")
    time_period_hours: int = Field(24, ge=1, le=8760, description="Analysis time period in hours")
    include_charts: bool = Field(True, description="Include charts in report")
    include_recommendations: bool = Field(True, description="Include recommendations")
    custom_filters: Dict[str, Any] = Field(default_factory=dict, description="Custom data filters")


class ScheduledReportRequest(BaseModel):
    """Request model for scheduled reports"""
    report_config: ReportGenerationRequest = Field(..., description="Report configuration")
    delivery_method: str = Field(..., description="Delivery method (file, email, webhook)")
    destination: str = Field(..., description="Delivery destination")
    schedule: str = Field(..., description="Schedule pattern (daily, weekly, monthly, etc.)")
    subject: Optional[str] = Field(None, description="Email subject (for email delivery)")


class TrendAnalysisResponse(BaseModel):
    """Response model for trend analysis"""
    metric_name: str = Field(..., description="Name of the analyzed metric")
    trend_direction: str = Field(..., description="Trend direction")
    trend_strength: float = Field(..., ge=0, le=1, description="Trend strength (0-1)")
    seasonality_detected: bool = Field(..., description="Whether seasonality was detected")
    anomalies_count: int = Field(..., ge=0, description="Number of anomalies detected")
    predictions_count: int = Field(..., ge=0, description="Number of predictions generated")


class SystemHealthResponse(BaseModel):
    """Response model for system health"""
    status: str = Field(..., description="Overall system status")
    components: Dict[str, bool] = Field(..., description="Component availability status")
    active_alerts: int = Field(..., ge=0, description="Number of active alerts")
    last_analysis: Optional[str] = Field(None, description="Last analysis timestamp")
    uptime_seconds: float = Field(..., ge=0, description="System uptime in seconds")


# Authentication (simplified for demo)
security = HTTPBearer() if FASTAPI_AVAILABLE else None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get current authenticated user (simplified implementation)"""
    # In production, this would validate JWT tokens, API keys, etc.
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Simple token validation (replace with proper authentication)
    if credentials.credentials == "demo-token":
        return "demo-user"
    
    raise HTTPException(status_code=401, detail="Invalid authentication credentials")


# Rate limiting (simplified)
class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""
        now = time.time()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window_seconds
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True


# Global rate limiter
rate_limiter = RateLimiter()

async def check_rate_limit(user: str = Depends(get_current_user)) -> str:
    """Check rate limit for user"""
    if not rate_limiter.is_allowed(user):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return user


class AnalyticsAPI:
    """Main Analytics API class"""
    
    def __init__(self):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for Analytics API. Install with: pip install fastapi uvicorn")
        
        self.app = FastAPI(
            title="Xencode Analytics API",
            description="RESTful API for Xencode analytics system",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize analytics components
        self._initialize_analytics()
        
        # Setup routes
        self._setup_routes()
        
        # System state
        self.start_time = time.time()
    
    def _initialize_analytics(self):
        """Initialize analytics components"""
        if ANALYTICS_AVAILABLE:
            try:
                self.analytics_engine = AdvancedAnalyticsEngine()
                self.reporting_system = AnalyticsReportingSystem(self.analytics_engine)
                self.orchestrator = IntegratedAnalyticsOrchestrator()
            except Exception as e:
                print(f"Warning: Could not initialize analytics components: {e}")
                self.analytics_engine = None
                self.reporting_system = None
                self.orchestrator = None
        else:
            self.analytics_engine = None
            self.reporting_system = None
            self.orchestrator = None
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", tags=["General"])
        async def root():
            """API root endpoint"""
            return {
                "message": "Xencode Analytics API",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/health"
            }
        
        @self.app.get("/health", response_model=SystemHealthResponse, tags=["General"])
        async def health_check():
            """System health check endpoint"""
            uptime = time.time() - self.start_time
            
            components = {
                "analytics_engine": self.analytics_engine is not None,
                "reporting_system": self.reporting_system is not None,
                "orchestrator": self.orchestrator is not None
            }
            
            # Determine overall status
            if all(components.values()):
                status = "healthy"
            elif any(components.values()):
                status = "degraded"
            else:
                status = "unhealthy"
            
            return SystemHealthResponse(
                status=status,
                components=components,
                active_alerts=0,  # Would get from actual monitoring
                last_analysis=datetime.now().isoformat(),
                uptime_seconds=uptime
            )
        
        @self.app.get("/analytics/summary", response_model=AnalyticsSummaryResponse, tags=["Analytics"])
        async def get_analytics_summary(
            hours: int = Query(24, ge=1, le=8760, description="Analysis time period in hours"),
            user: str = Depends(check_rate_limit)
        ):
            """Get analytics summary for specified time period"""
            
            if not self.analytics_engine:
                raise HTTPException(status_code=503, detail="Analytics engine not available")
            
            try:
                # Generate sample data if needed
                self.analytics_engine.generate_sample_data(days=max(1, hours // 24))
                
                # Run analysis
                results = await self.analytics_engine.run_comprehensive_analysis(hours)
                summary = results.get("summary", {})
                
                return AnalyticsSummaryResponse(
                    patterns_detected=summary.get("patterns_detected", 0),
                    users_analyzed=summary.get("users_analyzed", 0),
                    optimizations_found=summary.get("optimizations_found", 0),
                    total_potential_savings=summary.get("total_potential_savings", 0.0),
                    analysis_period_hours=hours,
                    generated_at=results.get("analysis_timestamp", datetime.now().isoformat())
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        @self.app.get("/analytics/patterns", response_model=List[UsagePatternResponse], tags=["Analytics"])
        async def get_usage_patterns(
            hours: int = Query(24, ge=1, le=8760, description="Analysis time period in hours"),
            min_confidence: float = Query(0.0, ge=0, le=1, description="Minimum confidence threshold"),
            user: str = Depends(check_rate_limit)
        ):
            """Get detected usage patterns"""
            
            if not self.analytics_engine:
                raise HTTPException(status_code=503, detail="Analytics engine not available")
            
            try:
                self.analytics_engine.generate_sample_data(days=max(1, hours // 24))
                results = await self.analytics_engine.run_comprehensive_analysis(hours)
                patterns = results.get("usage_patterns", [])
                
                # Filter by confidence
                filtered_patterns = [
                    p for p in patterns 
                    if p.get("confidence", 0) >= min_confidence
                ]
                
                return [
                    UsagePatternResponse(
                        pattern_id=pattern.get("pattern_id", f"pattern_{i}"),
                        pattern_type=pattern.get("type", "unknown"),
                        description=pattern.get("description", ""),
                        confidence=pattern.get("confidence", 0.0),
                        frequency=pattern.get("frequency", 0.0),
                        metadata=pattern.get("metadata", {})
                    )
                    for i, pattern in enumerate(filtered_patterns)
                ]
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")
        
        @self.app.get("/analytics/optimizations", response_model=List[CostOptimizationResponse], tags=["Analytics"])
        async def get_cost_optimizations(
            hours: int = Query(24, ge=1, le=8760, description="Analysis time period in hours"),
            min_savings: float = Query(0.0, ge=0, description="Minimum savings threshold"),
            user: str = Depends(check_rate_limit)
        ):
            """Get cost optimization recommendations"""
            
            if not self.analytics_engine:
                raise HTTPException(status_code=503, detail="Analytics engine not available")
            
            try:
                self.analytics_engine.generate_sample_data(days=max(1, hours // 24))
                results = await self.analytics_engine.run_comprehensive_analysis(hours)
                optimizations = results.get("cost_optimizations", [])
                
                # Filter by minimum savings
                filtered_opts = [
                    opt for opt in optimizations 
                    if opt.get("potential_savings", 0) >= min_savings
                ]
                
                return [
                    CostOptimizationResponse(
                        optimization_id=opt.get("optimization_id", f"opt_{i}"),
                        optimization_type=opt.get("type", "unknown"),
                        title=opt.get("title", ""),
                        description=opt.get("description", ""),
                        potential_savings=opt.get("potential_savings", 0.0),
                        implementation_effort=opt.get("implementation_effort", "unknown"),
                        impact_score=opt.get("impact_score", 0.0),
                        recommended_actions=opt.get("recommended_actions", [])
                    )
                    for i, opt in enumerate(filtered_opts)
                ]
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Optimization analysis failed: {str(e)}")
        
        @self.app.get("/analytics/trends/{metric_name}", response_model=TrendAnalysisResponse, tags=["Analytics"])
        async def get_trend_analysis(
            metric_name: str = PathParam(..., description="Name of metric to analyze"),
            hours: int = Query(168, ge=24, le=8760, description="Analysis time period in hours"),
            user: str = Depends(check_rate_limit)
        ):
            """Get trend analysis for specific metric"""
            
            if not self.analytics_engine:
                raise HTTPException(status_code=503, detail="Analytics engine not available")
            
            try:
                # Generate sample data
                self.analytics_engine.generate_sample_data(days=max(1, hours // 24))
                
                # Analyze trends
                trend_analysis = self.analytics_engine.trend_analyzer.analyze_trends(metric_name, hours)
                
                return TrendAnalysisResponse(
                    metric_name=trend_analysis.metric_name,
                    trend_direction=trend_analysis.trend_direction,
                    trend_strength=trend_analysis.trend_strength,
                    seasonality_detected=trend_analysis.seasonality_detected,
                    anomalies_count=len(trend_analysis.anomalies_detected),
                    predictions_count=len(trend_analysis.predicted_values)
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")
        
        @self.app.post("/reports/generate", tags=["Reports"])
        async def generate_report(
            request: ReportGenerationRequest,
            background_tasks: BackgroundTasks,
            user: str = Depends(check_rate_limit)
        ):
            """Generate analytics report"""
            
            if not self.reporting_system:
                raise HTTPException(status_code=503, detail="Reporting system not available")
            
            try:
                # Convert request to ReportConfig
                from .analytics_reporting_system import ReportConfig, ReportFormat, ReportType
                
                config = ReportConfig(
                    report_type=ReportType(request.report_type),
                    format=ReportFormat(request.format),
                    title=request.title,
                    description=request.description,
                    time_period_hours=request.time_period_hours,
                    include_charts=request.include_charts,
                    include_recommendations=request.include_recommendations,
                    custom_filters=request.custom_filters
                )
                
                # Generate report
                report = await self.reporting_system.generate_report(config)
                
                return {
                    "report_id": report.report_id,
                    "status": "generated",
                    "format": report.format.value,
                    "generated_at": report.generated_at.isoformat(),
                    "download_url": f"/reports/{report.report_id}/download"
                }
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
        
        @self.app.get("/reports/{report_id}/download", tags=["Reports"])
        async def download_report(
            report_id: str = PathParam(..., description="Report ID"),
            user: str = Depends(check_rate_limit)
        ):
            """Download generated report"""
            
            if not self.reporting_system:
                raise HTTPException(status_code=503, detail="Reporting system not available")
            
            # Get report from history
            reports = self.reporting_system.get_report_history()
            report = next((r for r in reports if r.report_id == report_id), None)
            
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            
            # Determine content type
            content_type_map = {
                "json": "application/json",
                "csv": "text/csv",
                "html": "text/html",
                "pdf": "application/pdf",
                "markdown": "text/markdown"
            }
            
            content_type = content_type_map.get(report.format.value, "application/octet-stream")
            filename = f"{report_id}.{report.format.value}"
            
            if isinstance(report.content, bytes):
                return StreamingResponse(
                    io.BytesIO(report.content),
                    media_type=content_type,
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )
            else:
                return StreamingResponse(
                    io.StringIO(report.content),
                    media_type=content_type,
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )
        
        @self.app.post("/reports/schedule", tags=["Reports"])
        async def schedule_report(
            request: ScheduledReportRequest,
            user: str = Depends(check_rate_limit)
        ):
            """Schedule recurring report generation"""
            
            if not self.reporting_system:
                raise HTTPException(status_code=503, detail="Reporting system not available")
            
            try:
                from .analytics_reporting_system import ReportConfig, DeliveryConfig, ReportFormat, ReportType, DeliveryMethod
                
                # Convert request to configs
                report_config = ReportConfig(
                    report_type=ReportType(request.report_config.report_type),
                    format=ReportFormat(request.report_config.format),
                    title=request.report_config.title,
                    description=request.report_config.description,
                    time_period_hours=request.report_config.time_period_hours,
                    include_charts=request.report_config.include_charts,
                    include_recommendations=request.report_config.include_recommendations,
                    custom_filters=request.report_config.custom_filters
                )
                
                delivery_config = DeliveryConfig(
                    method=DeliveryMethod(request.delivery_method),
                    destination=request.destination,
                    schedule=request.schedule,
                    subject=request.subject
                )
                
                # Schedule report
                schedule_id = self.reporting_system.schedule_report(report_config, delivery_config)
                
                return {
                    "schedule_id": schedule_id,
                    "status": "scheduled",
                    "schedule": request.schedule,
                    "next_run": "calculated_based_on_schedule"  # Would calculate actual next run time
                }
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Scheduling failed: {str(e)}")
        
        @self.app.get("/reports/scheduled", tags=["Reports"])
        async def get_scheduled_reports(user: str = Depends(check_rate_limit)):
            """Get list of scheduled reports"""
            
            if not self.reporting_system:
                raise HTTPException(status_code=503, detail="Reporting system not available")
            
            scheduled_reports = self.reporting_system.get_scheduled_reports()
            
            return {
                "scheduled_reports": [
                    {
                        "schedule_id": schedule_id,
                        "report_title": info["report_config"].title,
                        "format": info["report_config"].format.value,
                        "schedule": info["delivery_config"].schedule,
                        "created_at": info["created_at"].isoformat(),
                        "last_run": info["last_run"].isoformat() if info["last_run"] else None,
                        "run_count": info["run_count"]
                    }
                    for schedule_id, info in scheduled_reports.items()
                ]
            }
        
        @self.app.delete("/reports/scheduled/{schedule_id}", tags=["Reports"])
        async def cancel_scheduled_report(
            schedule_id: str = PathParam(..., description="Schedule ID"),
            user: str = Depends(check_rate_limit)
        ):
            """Cancel scheduled report"""
            
            if not self.reporting_system:
                raise HTTPException(status_code=503, detail="Reporting system not available")
            
            scheduled_reports = self.reporting_system.get_scheduled_reports()
            
            if schedule_id not in scheduled_reports:
                raise HTTPException(status_code=404, detail="Scheduled report not found")
            
            # Remove from scheduler
            del self.reporting_system.scheduler.scheduled_reports[schedule_id]
            
            return {"status": "cancelled", "schedule_id": schedule_id}
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the API server"""
        if self.reporting_system:
            await self.reporting_system.start()
        
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop_server(self):
        """Stop the API server"""
        if self.reporting_system:
            await self.reporting_system.stop()


# Demo function
async def run_analytics_api_demo():
    """Run analytics API demo"""
    from rich.console import Console
    
    console = Console()
    console.print("🚀 [bold cyan]Analytics API Demo[/bold cyan]\n")
    
    try:
        # Create API instance
        api = AnalyticsAPI()
        
        console.print("🔧 Analytics API initialized successfully")
        console.print("📊 Available endpoints:")
        console.print("   GET  /health - System health check")
        console.print("   GET  /analytics/summary - Analytics summary")
        console.print("   GET  /analytics/patterns - Usage patterns")
        console.print("   GET  /analytics/optimizations - Cost optimizations")
        console.print("   GET  /analytics/trends/{metric} - Trend analysis")
        console.print("   POST /reports/generate - Generate report")
        console.print("   GET  /reports/{id}/download - Download report")
        console.print("   POST /reports/schedule - Schedule report")
        console.print("   GET  /reports/scheduled - List scheduled reports")
        
        console.print("\n🌐 Starting API server on http://localhost:8000")
        console.print("📖 API documentation available at http://localhost:8000/docs")
        console.print("🔍 Alternative docs at http://localhost:8000/redoc")
        console.print("\n💡 Use 'demo-token' as Bearer token for authentication")
        console.print("\n[dim]Press Ctrl+C to stop the server[/dim]")
        
        # Start server
        await api.start_server(host="localhost", port=8000)
        
    except KeyboardInterrupt:
        console.print("\n👋 API server stopped")
    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        asyncio.run(run_analytics_api_demo())
    else:
        print("FastAPI is required to run the Analytics API. Install with: pip install fastapi uvicorn")