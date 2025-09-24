#!/usr/bin/env python3
"""
Advanced Analytics Dashboard

Real-time performance metrics, usage analysis, and cost optimization for Xencode.
Provides comprehensive insights into system performance, model usage patterns,
and resource consumption with beautiful visualizations.

Key Features:
- Real-time performance monitoring with live charts
- Usage pattern analysis and insights generation
- Cost tracking and optimization recommendations
- Model performance comparison and benchmarking
- Interactive dashboard with Rich UI components
- Historical data analysis and trend prediction
- Export capabilities for reports and analytics
"""

import asyncio
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import sqlite3
from collections import defaultdict, deque
import statistics

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich.tree import Tree
from rich.align import Align
from rich.layout import Layout
from rich import box
import yaml


@dataclass
class MetricPoint:
    """Single metric measurement point"""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    response_time: float = 0.0
    tokens_per_second: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    request_count: int = 0
    error_count: int = 0
    cache_hit_rate: float = 0.0
    model_accuracy: float = 0.0


@dataclass
class UsageStats:
    """Usage statistics container"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    total_tokens_processed: int = 0
    unique_users: int = 0
    most_used_model: str = ""
    peak_usage_time: str = ""


@dataclass
class CostMetrics:
    """Cost tracking metrics"""
    total_cost: float = 0.0
    cost_per_request: float = 0.0
    cost_per_token: float = 0.0
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_trends: List[float] = field(default_factory=list)
    optimization_savings: float = 0.0


class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("metrics.db")
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.console = Console()
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    timestamp REAL,
                    metric_name TEXT,
                    value REAL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_events (
                    timestamp REAL,
                    event_type TEXT,
                    user_id TEXT,
                    model TEXT,
                    tokens INTEGER,
                    cost REAL,
                    success BOOLEAN
                )
            """)
    
    def record_metric(self, name: str, value: float, metadata: Dict[str, Any] = None) -> None:
        """Record a metric point"""
        metadata = metadata or {}
        point = MetricPoint(time.time(), value, metadata)
        self.metrics_buffer[name].append(point)
        
        # Persist to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO metrics VALUES (?, ?, ?, ?)",
                (point.timestamp, name, value, json.dumps(metadata))
            )
    
    def record_usage_event(self, event_type: str, user_id: str, model: str, 
                          tokens: int, cost: float, success: bool) -> None:
        """Record a usage event"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO usage_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), event_type, user_id, model, tokens, cost, success)
            )
    
    def get_recent_metrics(self, name: str, minutes: int = 60) -> List[MetricPoint]:
        """Get recent metrics for a given time window"""
        cutoff_time = time.time() - (minutes * 60)
        return [point for point in self.metrics_buffer[name] if point.timestamp >= cutoff_time]
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculate current performance metrics"""
        recent_response_times = [p.value for p in self.get_recent_metrics("response_time", 10)]
        recent_tokens = [p.value for p in self.get_recent_metrics("tokens_per_second", 10)]
        recent_memory = [p.value for p in self.get_recent_metrics("memory_usage", 5)]
        recent_cpu = [p.value for p in self.get_recent_metrics("cpu_usage", 5)]
        
        return PerformanceMetrics(
            response_time=statistics.mean(recent_response_times) if recent_response_times else 0.0,
            tokens_per_second=statistics.mean(recent_tokens) if recent_tokens else 0.0,
            memory_usage=statistics.mean(recent_memory) if recent_memory else 0.0,
            cpu_usage=statistics.mean(recent_cpu) if recent_cpu else 0.0,
            request_count=len(self.get_recent_metrics("request", 60)),
            error_count=len(self.get_recent_metrics("error", 60)),
            cache_hit_rate=self._calculate_cache_hit_rate(),
            model_accuracy=self._calculate_model_accuracy()
        )
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        cache_hits = len(self.get_recent_metrics("cache_hit", 60))
        cache_misses = len(self.get_recent_metrics("cache_miss", 60))
        total = cache_hits + cache_misses
        return (cache_hits / total * 100) if total > 0 else 0.0
    
    def _calculate_model_accuracy(self) -> float:
        """Calculate model accuracy from recent feedback"""
        accuracy_points = self.get_recent_metrics("model_accuracy", 60)
        return statistics.mean([p.value for p in accuracy_points]) if accuracy_points else 95.0


class AnalyticsEngine:
    """Advanced analytics and insights engine"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
        self.console = Console()
    
    def analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyze usage patterns and generate insights"""
        with sqlite3.connect(self.collector.db_path) as conn:
            cursor = conn.cursor()
            
            # Get usage statistics
            cursor.execute("""
                SELECT COUNT(*), AVG(tokens), SUM(cost), model, 
                       COUNT(DISTINCT user_id), AVG(CASE WHEN success THEN 1 ELSE 0 END)
                FROM usage_events 
                WHERE timestamp > ? 
                GROUP BY model
            """, (time.time() - 86400,))  # Last 24 hours
            
            model_stats = cursor.fetchall()
            
            # Peak usage analysis
            cursor.execute("""
                SELECT strftime('%H', datetime(timestamp, 'unixepoch')) as hour, 
                       COUNT(*) as count
                FROM usage_events 
                WHERE timestamp > ?
                GROUP BY hour
                ORDER BY count DESC
                LIMIT 1
            """, (time.time() - 86400,))
            
            peak_hour = cursor.fetchone()
            
            return {
                "model_statistics": model_stats,
                "peak_usage_hour": peak_hour[0] if peak_hour else "N/A",
                "usage_trends": self._analyze_usage_trends(),
                "user_behavior": self._analyze_user_behavior(),
                "performance_insights": self._generate_performance_insights()
            }
    
    def _analyze_usage_trends(self) -> Dict[str, Any]:
        """Analyze usage trends over time"""
        with sqlite3.connect(self.collector.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date(timestamp, 'unixepoch') as day, COUNT(*) as requests
                FROM usage_events 
                WHERE timestamp > ?
                GROUP BY day
                ORDER BY day
            """, (time.time() - 604800,))  # Last week
            
            daily_usage = cursor.fetchall()
            
            if len(daily_usage) >= 2:
                recent_avg = statistics.mean([row[1] for row in daily_usage[-3:]])
                earlier_avg = statistics.mean([row[1] for row in daily_usage[:-3]]) if len(daily_usage) > 3 else recent_avg
                trend = "increasing" if recent_avg > earlier_avg else "decreasing"
            else:
                trend = "stable"
            
            return {
                "daily_usage": daily_usage,
                "trend": trend,
                "growth_rate": self._calculate_growth_rate(daily_usage)
            }
    
    def _analyze_user_behavior(self) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        with sqlite3.connect(self.collector.db_path) as conn:
            cursor = conn.cursor()
            
            # User session analysis
            cursor.execute("""
                SELECT user_id, COUNT(*) as requests, AVG(tokens) as avg_tokens,
                       MAX(timestamp) - MIN(timestamp) as session_length
                FROM usage_events 
                WHERE timestamp > ?
                GROUP BY user_id
            """, (time.time() - 86400,))
            
            user_sessions = cursor.fetchall()
            
            return {
                "active_users": len(user_sessions),
                "avg_requests_per_user": statistics.mean([s[1] for s in user_sessions]) if user_sessions else 0,
                "avg_session_length": statistics.mean([s[3] for s in user_sessions]) if user_sessions else 0,
                "power_users": [s for s in user_sessions if s[1] > 50]  # More than 50 requests
            }
    
    def _generate_performance_insights(self) -> List[str]:
        """Generate performance insights and recommendations"""
        insights = []
        metrics = self.collector.get_performance_metrics()
        
        if metrics.response_time > 2.0:
            insights.append("⚠️ High response times detected - consider optimizing model selection")
        
        if metrics.cache_hit_rate < 70:
            insights.append("💡 Low cache hit rate - review caching strategies")
        
        if metrics.error_count > metrics.request_count * 0.05:
            insights.append("🚨 High error rate - investigate system stability")
        
        if metrics.memory_usage > 80:
            insights.append("📊 High memory usage - consider resource optimization")
        
        if not insights:
            insights.append("✅ System performing optimally")
        
        return insights
    
    def _calculate_growth_rate(self, daily_usage: List[Tuple]) -> float:
        """Calculate growth rate from daily usage data"""
        if len(daily_usage) < 2:
            return 0.0
        
        first_week = daily_usage[:len(daily_usage)//2]
        second_week = daily_usage[len(daily_usage)//2:]
        
        if not first_week or not second_week:
            return 0.0
        
        first_avg = statistics.mean([row[1] for row in first_week])
        second_avg = statistics.mean([row[1] for row in second_week])
        
        return ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0.0


class CostOptimizer:
    """Cost tracking and optimization recommendations"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
        self.model_costs = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "claude-3": 0.015,
            "gemini-pro": 0.001,
            "local": 0.0
        }
    
    def calculate_cost_metrics(self) -> CostMetrics:
        """Calculate comprehensive cost metrics"""
        with sqlite3.connect(self.collector.db_path) as conn:
            cursor = conn.cursor()
            
            # Total costs last 24 hours
            cursor.execute("""
                SELECT SUM(cost), COUNT(*), SUM(tokens), model
                FROM usage_events 
                WHERE timestamp > ?
                GROUP BY model
            """, (time.time() - 86400,))
            
            model_costs = cursor.fetchall()
            
            total_cost = sum(row[0] or 0 for row in model_costs)
            total_requests = sum(row[1] for row in model_costs)
            total_tokens = sum(row[2] for row in model_costs)
            
            cost_by_model = {row[3]: row[0] or 0 for row in model_costs}
            
            return CostMetrics(
                total_cost=total_cost,
                cost_per_request=total_cost / total_requests if total_requests > 0 else 0,
                cost_per_token=total_cost / total_tokens if total_tokens > 0 else 0,
                cost_by_model=cost_by_model,
                optimization_savings=self._calculate_potential_savings(model_costs)
            )
    
    def _calculate_potential_savings(self, model_costs: List[Tuple]) -> float:
        """Calculate potential savings from optimization"""
        total_savings = 0.0
        
        for cost, requests, tokens, model in model_costs:
            if model in ["gpt-4", "claude-3"]:  # Expensive models
                # Calculate savings if 30% moved to cheaper alternatives
                cheaper_cost = tokens * 0.002 * 0.3  # 30% to GPT-3.5
                current_cost = cost or 0
                savings = max(0, current_cost * 0.3 - cheaper_cost)
                total_savings += savings
        
        return total_savings


class DashboardRenderer:
    """Renders the analytics dashboard with Rich UI"""
    
    def __init__(self, analytics_engine: AnalyticsEngine, cost_optimizer: CostOptimizer):
        self.analytics = analytics_engine
        self.cost_optimizer = cost_optimizer
        self.console = Console()
    
    def create_performance_panel(self) -> Panel:
        """Create performance metrics panel"""
        metrics = self.analytics.collector.get_performance_metrics()
        
        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        # Add metrics with status indicators
        table.add_row(
            "Response Time", 
            f"{metrics.response_time:.2f}s",
            "🟢" if metrics.response_time < 2.0 else "🟡" if metrics.response_time < 5.0 else "🔴"
        )
        table.add_row(
            "Tokens/Second", 
            f"{metrics.tokens_per_second:.1f}",
            "🟢" if metrics.tokens_per_second > 10 else "🟡"
        )
        table.add_row(
            "Cache Hit Rate", 
            f"{metrics.cache_hit_rate:.1f}%",
            "🟢" if metrics.cache_hit_rate > 80 else "🟡" if metrics.cache_hit_rate > 60 else "🔴"
        )
        table.add_row("Memory Usage", f"{metrics.memory_usage:.1f}%", "🟢" if metrics.memory_usage < 80 else "🔴")
        table.add_row("Requests (1h)", str(metrics.request_count), "🟢")
        table.add_row("Errors (1h)", str(metrics.error_count), "🟢" if metrics.error_count == 0 else "🔴")
        
        return Panel(table, title="📊 Performance Metrics", border_style="green")
    
    def create_usage_panel(self) -> Panel:
        """Create usage statistics panel"""
        usage_data = self.analytics.analyze_usage_patterns()
        
        table = Table(box=box.SIMPLE)
        table.add_column("Model", style="cyan")
        table.add_column("Requests", style="green")
        table.add_column("Avg Tokens", style="yellow")
        table.add_column("Success Rate", style="blue")
        
        for stats in usage_data["model_statistics"]:
            requests, avg_tokens, total_cost, model, unique_users, success_rate = stats
            table.add_row(
                model,
                str(requests),
                f"{avg_tokens:.0f}",
                f"{success_rate*100:.1f}%"
            )
        
        return Panel(table, title="📈 Usage Statistics (24h)", border_style="blue")
    
    def create_cost_panel(self) -> Panel:
        """Create cost analysis panel"""
        cost_metrics = self.cost_optimizer.calculate_cost_metrics()
        
        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Cost (24h)", f"${cost_metrics.total_cost:.4f}")
        table.add_row("Cost per Request", f"${cost_metrics.cost_per_request:.6f}")
        table.add_row("Cost per Token", f"${cost_metrics.cost_per_token:.8f}")
        table.add_row("Potential Savings", f"${cost_metrics.optimization_savings:.4f}")
        
        # Add cost by model
        for model, cost in cost_metrics.cost_by_model.items():
            table.add_row(f"  {model}", f"${cost:.4f}")
        
        return Panel(table, title="💰 Cost Analysis", border_style="yellow")
    
    def create_insights_panel(self) -> Panel:
        """Create insights and recommendations panel"""
        insights = self.analytics._generate_performance_insights()
        
        text = Text()
        for insight in insights:
            text.append(f"{insight}\n", style="white")
        
        return Panel(text, title="💡 Insights & Recommendations", border_style="magenta")
    
    def render_dashboard(self) -> Layout:
        """Render the complete dashboard"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        layout["header"].update(Panel(
            Align.center(Text("🚀 Xencode Analytics Dashboard", style="bold cyan")),
            style="white on blue"
        ))
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="performance"),
            Layout(name="cost")
        )
        
        layout["right"].split_column(
            Layout(name="usage"),
            Layout(name="insights")
        )
        
        layout["performance"].update(self.create_performance_panel())
        layout["usage"].update(self.create_usage_panel())
        layout["cost"].update(self.create_cost_panel())
        layout["insights"].update(self.create_insights_panel())
        
        layout["footer"].update(Panel(
            Align.center(Text(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")),
            style="white on dark_blue"
        ))
        
        return layout


class AnalyticsDashboard:
    """Main analytics dashboard orchestrator"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.metrics_collector = MetricsCollector(db_path)
        self.analytics_engine = AnalyticsEngine(self.metrics_collector)
        self.cost_optimizer = CostOptimizer(self.metrics_collector)
        self.renderer = DashboardRenderer(self.analytics_engine, self.cost_optimizer)
        self.console = Console()
        self.is_running = False
    
    async def start_live_dashboard(self, refresh_interval: float = 2.0) -> None:
        """Start live dashboard with auto-refresh"""
        self.is_running = True
        
        with Live(self.renderer.render_dashboard(), refresh_per_second=1/refresh_interval, console=self.console) as live:
            while self.is_running:
                live.update(self.renderer.render_dashboard())
                await asyncio.sleep(refresh_interval)
    
    def stop_dashboard(self) -> None:
        """Stop the live dashboard"""
        self.is_running = False
    
    def generate_sample_data(self) -> None:
        """Generate sample data for demonstration"""
        import random
        
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3", "gemini-pro", "local"]
        users = [f"user_{i}" for i in range(10)]
        
        # Generate sample metrics
        for i in range(100):
            # Performance metrics
            self.metrics_collector.record_metric("response_time", random.uniform(0.5, 3.0))
            self.metrics_collector.record_metric("tokens_per_second", random.uniform(5, 25))
            self.metrics_collector.record_metric("memory_usage", random.uniform(30, 85))
            self.metrics_collector.record_metric("cpu_usage", random.uniform(20, 70))
            
            # Cache metrics
            if random.random() > 0.3:
                self.metrics_collector.record_metric("cache_hit", 1)
            else:
                self.metrics_collector.record_metric("cache_miss", 1)
            
            # Usage events
            model = random.choice(models)
            user = random.choice(users)
            tokens = random.randint(50, 500)
            cost = tokens * self.cost_optimizer.model_costs.get(model, 0.001) / 1000
            success = random.random() > 0.05  # 95% success rate
            
            self.metrics_collector.record_usage_event(
                "chat_completion", user, model, tokens, cost, success
            )
            
            if not success:
                self.metrics_collector.record_metric("error", 1)
            else:
                self.metrics_collector.record_metric("request", 1)
    
    def export_report(self, format: str = "json") -> str:
        """Export analytics report"""
        usage_data = self.analytics_engine.analyze_usage_patterns()
        performance_metrics = self.metrics_collector.get_performance_metrics()
        cost_metrics = self.cost_optimizer.calculate_cost_metrics()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "performance": {
                "response_time": performance_metrics.response_time,
                "tokens_per_second": performance_metrics.tokens_per_second,
                "cache_hit_rate": performance_metrics.cache_hit_rate,
                "error_rate": performance_metrics.error_count / max(performance_metrics.request_count, 1)
            },
            "usage": usage_data,
            "costs": {
                "total_cost": cost_metrics.total_cost,
                "cost_per_request": cost_metrics.cost_per_request,
                "potential_savings": cost_metrics.optimization_savings
            },
            "insights": self.analytics_engine._generate_performance_insights()
        }
        
        if format == "json":
            return json.dumps(report, indent=2)
        elif format == "yaml":
            return yaml.dump(report, default_flow_style=False)
        else:
            return str(report)


# Demo and Testing Functions
async def run_dashboard_demo():
    """Run an interactive dashboard demonstration"""
    console = Console()
    console.print("🚀 Starting Xencode Analytics Dashboard Demo...\n")
    
    dashboard = AnalyticsDashboard()
    
    # Generate sample data
    console.print("📊 Generating sample metrics data...")
    dashboard.generate_sample_data()
    console.print("✅ Sample data generated\n")
    
    console.print("🔄 Starting live dashboard (Press Ctrl+C to stop)...")
    
    try:
        await dashboard.start_live_dashboard(refresh_interval=1.0)
    except KeyboardInterrupt:
        dashboard.stop_dashboard()
        console.print("\n👋 Dashboard stopped")
    
    # Export sample report
    console.print("\n📋 Generating analytics report...")
    report = dashboard.export_report("json")
    
    report_path = Path("analytics_report.json")
    report_path.write_text(report)
    console.print(f"✅ Report saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(run_dashboard_demo())