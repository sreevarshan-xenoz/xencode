#!/usr/bin/env python3
"""
Analytics Integration Module

Integrates the advanced analytics engine with the existing Xencode analytics
infrastructure, providing seamless data flow and unified reporting.

Key Features:
- Integration with existing analytics infrastructure
- Unified data collection and analysis
- Cross-component analytics correlation
- Enhanced reporting with advanced insights
- Real-time analytics pipeline
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

# Import existing analytics components
try:
    from .analytics.analytics_infrastructure import AnalyticsInfrastructure
    from .advanced_analytics_dashboard import AnalyticsDashboard
    from .performance_monitoring_dashboard import PerformanceMonitoringDashboard
    from .advanced_analytics_engine import AdvancedAnalyticsEngine
    ANALYTICS_COMPONENTS_AVAILABLE = True
except ImportError:
    ANALYTICS_COMPONENTS_AVAILABLE = False
    # Mock classes for standalone operation
    class AnalyticsInfrastructure:
        def __init__(self, *args, **kwargs): pass
        def get_system_status(self): return {}
        def track_system_event(self, *args, **kwargs): pass
    
    class AnalyticsDashboard:
        def __init__(self, *args, **kwargs): pass
    
    class PerformanceMonitoringDashboard:
        def __init__(self, *args, **kwargs): pass
    
    class AdvancedAnalyticsEngine:
        def __init__(self, *args, **kwargs): pass


@dataclass
class IntegratedAnalyticsConfig:
    """Configuration for integrated analytics system"""
    enable_advanced_analytics: bool = True
    enable_performance_monitoring: bool = True
    enable_cost_optimization: bool = True
    enable_ml_trends: bool = True
    analytics_storage_path: Optional[Path] = None
    sync_interval_seconds: int = 300  # 5 minutes
    retention_days: int = 90


@dataclass
class AnalyticsInsight:
    """Unified analytics insight"""
    insight_id: str
    insight_type: str
    title: str
    description: str
    severity: str  # info, warning, critical
    confidence: float
    timestamp: datetime
    source_component: str
    metadata: Dict[str, Any]
    recommendations: List[str]


class AnalyticsDataBridge:
    """Bridges data between different analytics components"""
    
    def __init__(self, config: IntegratedAnalyticsConfig):
        self.config = config
        self.data_mappings = self._initialize_data_mappings()
    
    def _initialize_data_mappings(self) -> Dict[str, Any]:
        """Initialize data mapping configurations"""
        return {
            "performance_to_usage": {
                "cpu_usage": "system_performance",
                "memory_usage": "resource_utilization",
                "response_time": "user_experience"
            },
            "cost_to_performance": {
                "model_cost": "efficiency_metric",
                "usage_cost": "resource_cost"
            },
            "usage_to_trends": {
                "request_count": "usage_volume",
                "error_rate": "quality_metric"
            }
        }
    
    def sync_performance_data(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync performance monitoring data to analytics engine"""
        synced_data = {}
        
        for metric, value in performance_data.items():
            if metric in self.data_mappings["performance_to_usage"]:
                mapped_metric = self.data_mappings["performance_to_usage"][metric]
                synced_data[mapped_metric] = {
                    "value": value,
                    "timestamp": time.time(),
                    "source": "performance_monitor"
                }
        
        return synced_data
    
    def sync_usage_data(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync usage analytics data to performance monitor"""
        synced_data = {}
        
        for metric, value in usage_data.items():
            if metric in self.data_mappings["usage_to_trends"]:
                mapped_metric = self.data_mappings["usage_to_trends"][metric]
                synced_data[mapped_metric] = {
                    "value": value,
                    "timestamp": time.time(),
                    "source": "usage_analyzer"
                }
        
        return synced_data
    
    def correlate_insights(self, insights_by_component: Dict[str, List[Any]]) -> List[AnalyticsInsight]:
        """Correlate insights from different components"""
        correlated_insights = []
        
        # Cross-reference performance and cost insights
        performance_insights = insights_by_component.get("performance", [])
        cost_insights = insights_by_component.get("cost", [])
        
        for perf_insight in performance_insights:
            for cost_insight in cost_insights:
                correlation = self._calculate_insight_correlation(perf_insight, cost_insight)
                
                if correlation > 0.7:  # High correlation
                    correlated_insight = AnalyticsInsight(
                        insight_id=f"corr_{int(time.time())}",
                        insight_type="correlation",
                        title=f"Performance-Cost Correlation Detected",
                        description=f"Performance issue correlates with cost optimization opportunity",
                        severity="warning",
                        confidence=correlation,
                        timestamp=datetime.now(),
                        source_component="analytics_integration",
                        metadata={
                            "performance_insight": perf_insight,
                            "cost_insight": cost_insight,
                            "correlation_score": correlation
                        },
                        recommendations=[
                            "Address performance issue to improve cost efficiency",
                            "Consider resource optimization strategies",
                            "Monitor correlation trends over time"
                        ]
                    )
                    correlated_insights.append(correlated_insight)
        
        return correlated_insights
    
    def _calculate_insight_correlation(self, insight1: Any, insight2: Any) -> float:
        """Calculate correlation between two insights"""
        # Simple correlation based on timing and severity
        time_correlation = 0.5  # Base correlation
        
        # Check if insights are related by timing (within 1 hour)
        if hasattr(insight1, 'timestamp') and hasattr(insight2, 'timestamp'):
            time_diff = abs((insight1.timestamp - insight2.timestamp).total_seconds())
            if time_diff < 3600:  # Within 1 hour
                time_correlation += 0.3
        
        # Check severity alignment
        if hasattr(insight1, 'severity') and hasattr(insight2, 'severity'):
            if insight1.severity == insight2.severity:
                time_correlation += 0.2
        
        return min(time_correlation, 1.0)


class IntegratedAnalyticsOrchestrator:
    """Orchestrates all analytics components in a unified system"""
    
    def __init__(self, config: Optional[IntegratedAnalyticsConfig] = None):
        self.config = config or IntegratedAnalyticsConfig()
        
        # Initialize storage path
        if not self.config.analytics_storage_path:
            self.config.analytics_storage_path = Path.home() / ".xencode" / "integrated_analytics"
        self.config.analytics_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Data bridge for cross-component communication
        self.data_bridge = AnalyticsDataBridge(self.config)
        
        # State management
        self._running = False
        self._sync_task = None
        
        # Unified insights storage
        self.unified_insights: List[AnalyticsInsight] = []
    
    def _initialize_components(self) -> None:
        """Initialize all analytics components"""
        
        # Core analytics infrastructure
        if ANALYTICS_COMPONENTS_AVAILABLE:
            try:
                self.analytics_infrastructure = AnalyticsInfrastructure()
            except Exception:
                self.analytics_infrastructure = None
        else:
            self.analytics_infrastructure = None
        
        # Advanced analytics engine
        if self.config.enable_advanced_analytics:
            try:
                analytics_db_path = self.config.analytics_storage_path / "advanced_analytics.db"
                self.advanced_engine = AdvancedAnalyticsEngine(analytics_db_path)
            except Exception:
                self.advanced_engine = None
        else:
            self.advanced_engine = None
        
        # Performance monitoring dashboard
        if self.config.enable_performance_monitoring:
            try:
                self.performance_dashboard = PerformanceMonitoringDashboard()
            except Exception:
                self.performance_dashboard = None
        else:
            self.performance_dashboard = None
        
        # Analytics dashboard
        try:
            dashboard_db_path = self.config.analytics_storage_path / "dashboard_metrics.db"
            self.analytics_dashboard = AnalyticsDashboard(dashboard_db_path)
        except Exception:
            self.analytics_dashboard = None
    
    async def start(self) -> None:
        """Start the integrated analytics system"""
        if self._running:
            return
        
        self._running = True
        
        print("🚀 Starting Integrated Analytics System...")
        
        # Start core analytics infrastructure
        if self.analytics_infrastructure:
            try:
                await self.analytics_infrastructure.start()
                print("   ✅ Analytics infrastructure started")
            except Exception as e:
                print(f"   ⚠️ Analytics infrastructure failed to start: {e}")
        
        # Start performance monitoring
        if self.performance_dashboard:
            try:
                await self.performance_dashboard.start()
                print("   ✅ Performance monitoring started")
            except Exception as e:
                print(f"   ⚠️ Performance monitoring failed to start: {e}")
        
        # Start data synchronization
        self._sync_task = asyncio.create_task(self._sync_loop())
        
        print("✅ Integrated Analytics System started")
        print(f"   📊 Advanced Analytics: {'Enabled' if self.advanced_engine else 'Disabled'}")
        print(f"   📈 Performance Monitoring: {'Enabled' if self.performance_dashboard else 'Disabled'}")
        print(f"   💾 Storage: {self.config.analytics_storage_path}")
    
    async def stop(self) -> None:
        """Stop the integrated analytics system"""
        self._running = False
        
        print("🛑 Stopping Integrated Analytics System...")
        
        # Cancel sync task
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        # Stop components
        if self.analytics_infrastructure:
            try:
                await self.analytics_infrastructure.stop()
            except Exception as e:
                print(f"   ⚠️ Error stopping analytics infrastructure: {e}")
        
        if self.performance_dashboard:
            try:
                await self.performance_dashboard.stop()
            except Exception as e:
                print(f"   ⚠️ Error stopping performance dashboard: {e}")
        
        print("✅ Integrated Analytics System stopped")
    
    async def _sync_loop(self) -> None:
        """Background synchronization loop"""
        while self._running:
            try:
                await self._sync_analytics_data()
                await asyncio.sleep(self.config.sync_interval_seconds)
            except Exception as e:
                print(f"Error in analytics sync loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _sync_analytics_data(self) -> None:
        """Synchronize data between analytics components"""
        try:
            # Collect data from all components
            component_data = {}
            
            # Get performance data
            if self.performance_dashboard:
                try:
                    perf_status = self.performance_dashboard.get_dashboard_status()
                    component_data["performance"] = perf_status
                except Exception as e:
                    print(f"Error getting performance data: {e}")
            
            # Get analytics infrastructure data
            if self.analytics_infrastructure:
                try:
                    analytics_status = self.analytics_infrastructure.get_system_status()
                    component_data["analytics"] = analytics_status
                except Exception as e:
                    print(f"Error getting analytics data: {e}")
            
            # Sync data between components using the bridge
            if component_data:
                self._process_cross_component_data(component_data)
            
        except Exception as e:
            print(f"Error syncing analytics data: {e}")
    
    def _process_cross_component_data(self, component_data: Dict[str, Any]) -> None:
        """Process and correlate data from different components"""
        
        # Extract insights from each component
        insights_by_component = {}
        
        # Process performance insights
        if "performance" in component_data:
            perf_data = component_data["performance"]
            perf_insights = self._extract_performance_insights(perf_data)
            insights_by_component["performance"] = perf_insights
        
        # Process analytics insights
        if "analytics" in component_data:
            analytics_data = component_data["analytics"]
            analytics_insights = self._extract_analytics_insights(analytics_data)
            insights_by_component["analytics"] = analytics_insights
        
        # Correlate insights across components
        if len(insights_by_component) > 1:
            correlated_insights = self.data_bridge.correlate_insights(insights_by_component)
            self.unified_insights.extend(correlated_insights)
            
            # Keep only recent insights (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.unified_insights = [
                insight for insight in self.unified_insights 
                if insight.timestamp > cutoff_time
            ]
    
    def _extract_performance_insights(self, perf_data: Dict[str, Any]) -> List[Any]:
        """Extract insights from performance data"""
        insights = []
        
        # Check for performance issues
        if perf_data.get("active_alerts", 0) > 0:
            insights.append({
                "type": "performance_alert",
                "severity": "warning",
                "timestamp": datetime.now(),
                "description": f"{perf_data['active_alerts']} active performance alerts"
            })
        
        return insights
    
    def _extract_analytics_insights(self, analytics_data: Dict[str, Any]) -> List[Any]:
        """Extract insights from analytics data"""
        insights = []
        
        # Check analytics infrastructure status
        if analytics_data.get("analytics_infrastructure", {}).get("running", False):
            insights.append({
                "type": "analytics_healthy",
                "severity": "info",
                "timestamp": datetime.now(),
                "description": "Analytics infrastructure is running normally"
            })
        
        return insights
    
    async def run_comprehensive_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Run comprehensive analysis across all components"""
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_period_hours": hours,
            "component_results": {},
            "unified_insights": [],
            "cross_component_correlations": [],
            "summary": {}
        }
        
        # Run advanced analytics if available
        if self.advanced_engine:
            try:
                advanced_results = await self.advanced_engine.run_comprehensive_analysis(hours)
                results["component_results"]["advanced_analytics"] = advanced_results
            except Exception as e:
                results["component_results"]["advanced_analytics"] = {"error": str(e)}
        
        # Get performance monitoring data
        if self.performance_dashboard:
            try:
                perf_status = self.performance_dashboard.get_dashboard_status()
                results["component_results"]["performance_monitoring"] = perf_status
            except Exception as e:
                results["component_results"]["performance_monitoring"] = {"error": str(e)}
        
        # Get analytics infrastructure status
        if self.analytics_infrastructure:
            try:
                analytics_status = self.analytics_infrastructure.get_system_status()
                results["component_results"]["analytics_infrastructure"] = analytics_status
            except Exception as e:
                results["component_results"]["analytics_infrastructure"] = {"error": str(e)}
        
        # Add unified insights
        results["unified_insights"] = [
            {
                "insight_id": insight.insight_id,
                "type": insight.insight_type,
                "title": insight.title,
                "description": insight.description,
                "severity": insight.severity,
                "confidence": insight.confidence,
                "timestamp": insight.timestamp.isoformat(),
                "source": insight.source_component,
                "recommendations": insight.recommendations
            }
            for insight in self.unified_insights
        ]
        
        # Generate summary
        results["summary"] = {
            "components_analyzed": len(results["component_results"]),
            "unified_insights_count": len(results["unified_insights"]),
            "system_health": self._assess_system_health(results),
            "key_recommendations": self._generate_key_recommendations(results)
        }
        
        return results
    
    def _assess_system_health(self, results: Dict[str, Any]) -> str:
        """Assess overall system health based on analysis results"""
        
        # Count critical issues
        critical_issues = 0
        warning_issues = 0
        
        for insight in self.unified_insights:
            if insight.severity == "critical":
                critical_issues += 1
            elif insight.severity == "warning":
                warning_issues += 1
        
        # Check component health
        component_errors = sum(
            1 for component_result in results["component_results"].values()
            if isinstance(component_result, dict) and "error" in component_result
        )
        
        if critical_issues > 0 or component_errors > 1:
            return "critical"
        elif warning_issues > 2 or component_errors > 0:
            return "warning"
        else:
            return "healthy"
    
    def _generate_key_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate key recommendations based on analysis"""
        recommendations = []
        
        # Collect recommendations from unified insights
        for insight in self.unified_insights:
            recommendations.extend(insight.recommendations)
        
        # Add component-specific recommendations
        advanced_results = results["component_results"].get("advanced_analytics", {})
        if "cost_optimizations" in advanced_results:
            cost_opts = advanced_results["cost_optimizations"]
            if cost_opts:
                total_savings = sum(opt.get("potential_savings", 0) for opt in cost_opts)
                if total_savings > 10:
                    recommendations.append(f"Implement cost optimizations for ${total_savings:.2f} potential savings")
        
        # Remove duplicates and limit to top 5
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:5]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "integrated_analytics": {
                "running": self._running,
                "config": {
                    "advanced_analytics": self.config.enable_advanced_analytics,
                    "performance_monitoring": self.config.enable_performance_monitoring,
                    "cost_optimization": self.config.enable_cost_optimization,
                    "ml_trends": self.config.enable_ml_trends
                },
                "storage_path": str(self.config.analytics_storage_path)
            },
            "components": {
                "analytics_infrastructure": self.analytics_infrastructure is not None,
                "advanced_engine": self.advanced_engine is not None,
                "performance_dashboard": self.performance_dashboard is not None,
                "analytics_dashboard": self.analytics_dashboard is not None
            },
            "unified_insights": len(self.unified_insights),
            "last_sync": datetime.now().isoformat() if self._running else None
        }


# Global integrated analytics orchestrator
integrated_analytics = IntegratedAnalyticsOrchestrator()


# Demo function
async def run_integrated_analytics_demo():
    """Run integrated analytics demo"""
    from rich.console import Console
    
    console = Console()
    console.print("🚀 [bold cyan]Integrated Analytics System Demo[/bold cyan]\n")
    
    # Create integrated analytics system
    config = IntegratedAnalyticsConfig(
        enable_advanced_analytics=True,
        enable_performance_monitoring=True,
        enable_cost_optimization=True,
        sync_interval_seconds=10  # Fast sync for demo
    )
    
    orchestrator = IntegratedAnalyticsOrchestrator(config)
    
    try:
        # Start the system
        console.print("🔄 Starting integrated analytics system...")
        await orchestrator.start()
        
        # Generate some sample data
        if orchestrator.advanced_engine:
            console.print("📊 Generating sample data...")
            orchestrator.advanced_engine.generate_sample_data(days=3)
        
        # Run comprehensive analysis
        console.print("🔍 Running comprehensive analysis...")
        results = await orchestrator.run_comprehensive_analysis(hours=72)
        
        # Display results
        console.print("📈 [bold green]Integrated Analysis Results:[/bold green]\n")
        
        summary = results.get("summary", {})
        console.print(f"   📊 Components analyzed: {summary.get('components_analyzed', 0)}")
        console.print(f"   💡 Unified insights: {summary.get('unified_insights_count', 0)}")
        console.print(f"   🏥 System health: {summary.get('system_health', 'unknown')}")
        
        recommendations = summary.get('key_recommendations', [])
        if recommendations:
            console.print("\n💡 [bold yellow]Key Recommendations:[/bold yellow]")
            for i, rec in enumerate(recommendations[:3]):
                console.print(f"   {i+1}. {rec}")
        
        # Show component results
        component_results = results.get("component_results", {})
        console.print(f"\n🔧 [bold blue]Component Results:[/bold blue]")
        for component, result in component_results.items():
            if isinstance(result, dict) and "error" not in result:
                console.print(f"   ✅ {component}: Operational")
            else:
                console.print(f"   ⚠️ {component}: Issues detected")
        
        console.print("\n✨ [green]Integrated analytics demo complete![/green]")
        
    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(run_integrated_analytics_demo())