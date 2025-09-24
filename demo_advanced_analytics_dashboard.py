#!/usr/bin/env python3
"""
Interactive Demo for Advanced Analytics Dashboard

Showcases the comprehensive analytics capabilities including real-time metrics,
usage analysis, cost optimization, and interactive dashboard visualization.
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
import tempfile
import json

# Add the system to path
sys.path.insert(0, str(Path(__file__).parent / "xencode"))

from advanced_analytics_dashboard import (
    AnalyticsDashboard, MetricsCollector, AnalyticsEngine, CostOptimizer
)

console = Console()


class AnalyticsDashboardDemo:
    """Interactive demonstration of the analytics dashboard"""
    
    def __init__(self):
        self.temp_db = Path(tempfile.mktemp(suffix=".db"))
        self.dashboard = AnalyticsDashboard(self.temp_db)
        self.demo_running = True
    
    def show_welcome(self):
        """Display welcome message"""
        welcome_text = Text()
        welcome_text.append("Welcome to the ", style="bold white")
        welcome_text.append("Xencode Advanced Analytics Dashboard", style="bold cyan")
        welcome_text.append(" Demo!", style="bold white")
        
        console.print(Panel(
            welcome_text,
            title="📊 Analytics Dashboard Demo",
            border_style="cyan",
            padding=(1, 2)
        ))
        
        console.print("\n🚀 This demo showcases:")
        console.print("• Real-time performance metrics monitoring")
        console.print("• Usage pattern analysis and insights")
        console.print("• Cost tracking and optimization recommendations")
        console.print("• Interactive dashboard with live updates")
        console.print("• Analytics report generation and export")
        console.print("• Machine learning-powered trend analysis\n")
    
    def show_main_menu(self):
        """Display main menu options"""
        from rich.table import Table
        
        table = Table(title="🎯 Demo Options")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        
        table.add_row("1", "Generate Sample Data")
        table.add_row("2", "View Performance Metrics")
        table.add_row("3", "Analyze Usage Patterns")
        table.add_row("4", "Cost Analysis & Optimization")
        table.add_row("5", "Live Dashboard (Interactive)")
        table.add_row("6", "Generate Analytics Report")
        table.add_row("7", "Simulate Real-time Data Stream")
        table.add_row("8", "View System Insights")
        table.add_row("0", "Exit Demo")
        
        console.print(table)
    
    def generate_sample_data(self):
        """Generate sample data with progress indicator"""
        console.print("\n📊 Generating comprehensive sample data...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Creating metrics and usage events...", total=None)
            
            # Generate enhanced sample data
            self.dashboard.generate_sample_data()
            
            # Add some specific scenarios
            collector = self.dashboard.metrics_collector
            
            # Simulate a performance issue
            for _ in range(10):
                collector.record_metric("response_time", 4.5)  # High response time
                collector.record_metric("error", 1)  # Errors
            
            # Simulate cache optimization
            for _ in range(50):
                collector.record_metric("cache_hit", 1)
            
            # Simulate different user behaviors
            collector.record_usage_event("code_analysis", "power_user_1", "gpt-4", 800, 0.064, True)
            collector.record_usage_event("chat", "casual_user_1", "gpt-3.5-turbo", 150, 0.003, True)
            collector.record_usage_event("translation", "enterprise_user_1", "claude-3", 500, 0.075, True)
            
            progress.update(task, description="Sample data generation complete!")
        
        console.print("✅ Sample data generated successfully!")
        console.print(f"📈 Generated metrics for performance, usage, and cost analysis")
    
    def view_performance_metrics(self):
        """Display current performance metrics"""
        console.print("\n📊 Current Performance Metrics:")
        
        metrics = self.dashboard.metrics_collector.get_performance_metrics()
        
        from rich.table import Table
        table = Table(title="Performance Overview")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Trend", style="blue")
        
        # Response time analysis
        status = "🟢 Excellent" if metrics.response_time < 1.0 else \
                "🟡 Good" if metrics.response_time < 2.0 else \
                "🟠 Slow" if metrics.response_time < 4.0 else "🔴 Critical"
        
        table.add_row(
            "Response Time",
            f"{metrics.response_time:.2f}s",
            status,
            "📈 Improving" if metrics.response_time < 2.0 else "📉 Needs attention"
        )
        
        table.add_row(
            "Throughput",
            f"{metrics.tokens_per_second:.1f} tokens/s",
            "🟢 High" if metrics.tokens_per_second > 15 else "🟡 Medium",
            "🚀 Optimized"
        )
        
        table.add_row(
            "Cache Efficiency",
            f"{metrics.cache_hit_rate:.1f}%",
            "🟢 Excellent" if metrics.cache_hit_rate > 80 else "🟡 Good",
            "⚡ Smart caching active"
        )
        
        table.add_row(
            "System Load",
            f"CPU: {metrics.cpu_usage:.1f}% | RAM: {metrics.memory_usage:.1f}%",
            "🟢 Healthy" if metrics.cpu_usage < 70 else "🟡 Moderate",
            "📊 Balanced"
        )
        
        table.add_row(
            "Reliability",
            f"{metrics.request_count - metrics.error_count}/{metrics.request_count} success",
            "🟢 Stable" if metrics.error_count == 0 else "🟡 Minor issues",
            "🛡️ Error handling active"
        )
        
        console.print(table)
        
        # Performance insights
        console.print("\n💡 Performance Insights:")
        insights = self.dashboard.analytics_engine._generate_performance_insights()
        for insight in insights:
            console.print(f"  {insight}")
    
    def analyze_usage_patterns(self):
        """Analyze and display usage patterns"""
        console.print("\n📈 Usage Pattern Analysis:")
        
        usage_data = self.dashboard.analytics_engine.analyze_usage_patterns()
        
        # Model statistics
        from rich.table import Table
        model_table = Table(title="Model Usage Statistics (24h)")
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Requests", style="green")
        model_table.add_column("Avg Tokens", style="yellow")
        model_table.add_column("Total Cost", style="red")
        model_table.add_column("Success Rate", style="blue")
        model_table.add_column("User Count", style="magenta")
        
        for stats in usage_data["model_statistics"]:
            requests, avg_tokens, total_cost, model, unique_users, success_rate = stats
            model_table.add_row(
                model,
                str(requests),
                f"{avg_tokens:.0f}",
                f"${total_cost:.4f}",
                f"{success_rate*100:.1f}%",
                str(unique_users)
            )
        
        console.print(model_table)
        
        # Usage trends
        trends = usage_data["usage_trends"]
        console.print(f"\n📊 Usage Trends:")
        console.print(f"  📈 Growth Rate: {trends['growth_rate']:.1f}%")
        console.print(f"  ⏰ Peak Hour: {usage_data['peak_usage_hour']}:00")
        console.print(f"  📱 Trend Direction: {trends['trend'].title()}")
        
        # User behavior insights
        behavior = usage_data["user_behavior"]
        console.print(f"\n👥 User Behavior Analysis:")
        console.print(f"  👤 Active Users: {behavior['active_users']}")
        console.print(f"  📊 Avg Requests/User: {behavior['avg_requests_per_user']:.1f}")
        console.print(f"  ⏱️ Avg Session Length: {behavior['avg_session_length']:.0f}s")
        console.print(f"  🌟 Power Users: {len(behavior['power_users'])}")
    
    def analyze_costs(self):
        """Analyze costs and show optimization recommendations"""
        console.print("\n💰 Cost Analysis & Optimization:")
        
        cost_metrics = self.dashboard.cost_optimizer.calculate_cost_metrics()
        
        from rich.table import Table
        cost_table = Table(title="Cost Breakdown (24h)")
        cost_table.add_column("Metric", style="cyan")
        cost_table.add_column("Value", style="green")
        cost_table.add_column("Impact", style="yellow")
        
        cost_table.add_row(
            "Total Cost",
            f"${cost_metrics.total_cost:.4f}",
            "💰 Daily operational cost"
        )
        
        cost_table.add_row(
            "Cost per Request",
            f"${cost_metrics.cost_per_request:.6f}",
            "📊 Efficiency metric"
        )
        
        cost_table.add_row(
            "Cost per Token",
            f"${cost_metrics.cost_per_token:.8f}",
            "🔍 Granular analysis"
        )
        
        cost_table.add_row(
            "Potential Savings",
            f"${cost_metrics.optimization_savings:.4f}",
            "💡 Optimization opportunity"
        )
        
        console.print(cost_table)
        
        # Cost by model
        console.print("\n📊 Cost by Model:")
        model_cost_table = Table()
        model_cost_table.add_column("Model", style="cyan")
        model_cost_table.add_column("Cost", style="green")
        model_cost_table.add_column("Percentage", style="yellow")
        
        total_model_cost = sum(cost_metrics.cost_by_model.values())
        for model, cost in cost_metrics.cost_by_model.items():
            percentage = (cost / total_model_cost * 100) if total_model_cost > 0 else 0
            model_cost_table.add_row(
                model,
                f"${cost:.4f}",
                f"{percentage:.1f}%"
            )
        
        console.print(model_cost_table)
        
        # Optimization recommendations
        console.print("\n💡 Cost Optimization Recommendations:")
        if cost_metrics.optimization_savings > 0:
            console.print(f"  🎯 Switch 30% of premium model usage to cost-effective alternatives")
            console.print(f"  💰 Potential monthly savings: ${cost_metrics.optimization_savings * 30:.2f}")
            console.print(f"  ⚡ Implement intelligent model routing based on query complexity")
            console.print(f"  📊 Enable aggressive caching for repeated queries")
        else:
            console.print("  ✅ Cost optimization is already maximized!")
    
    async def run_live_dashboard(self):
        """Run the live dashboard"""
        console.print("\n🔄 Starting Live Analytics Dashboard...")
        console.print("💡 The dashboard will refresh every 2 seconds")
        console.print("🛑 Press Ctrl+C to stop the live dashboard\n")
        
        try:
            await self.dashboard.start_live_dashboard(refresh_interval=2.0)
        except KeyboardInterrupt:
            self.dashboard.stop_dashboard()
            console.print("\n✅ Live dashboard stopped")
    
    def generate_report(self):
        """Generate and display analytics report"""
        console.print("\n📋 Generating Comprehensive Analytics Report...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing data and generating report...", total=None)
            
            # Generate reports in different formats
            json_report = self.dashboard.export_report("json")
            yaml_report = self.dashboard.export_report("yaml")
            
            progress.update(task, description="Report generation complete!")
        
        console.print("✅ Analytics report generated successfully!")
        
        # Show summary
        report_data = json.loads(json_report)
        
        console.print("\n📊 Report Summary:")
        console.print(f"  📅 Generated: {report_data['timestamp']}")
        console.print(f"  ⚡ Avg Response Time: {report_data['performance']['response_time']:.2f}s")
        console.print(f"  🎯 Cache Hit Rate: {report_data['performance']['cache_hit_rate']:.1f}%")
        console.print(f"  💰 Total Cost: ${report_data['costs']['total_cost']:.4f}")
        console.print(f"  🔧 Optimization Savings: ${report_data['costs']['potential_savings']:.4f}")
        
        console.print("\n💡 Key Insights:")
        for i, insight in enumerate(report_data['insights'][:3], 1):
            console.print(f"  {i}. {insight}")
        
        # Save reports
        json_path = Path("analytics_report.json")
        yaml_path = Path("analytics_report.yaml")
        
        json_path.write_text(json_report)
        yaml_path.write_text(yaml_report)
        
        console.print(f"\n📁 Reports saved:")
        console.print(f"  📄 JSON: {json_path}")
        console.print(f"  📄 YAML: {yaml_path}")
    
    async def simulate_realtime_data(self):
        """Simulate real-time data streaming"""
        console.print("\n🔄 Simulating Real-time Data Stream...")
        console.print("📊 Watch as metrics update in real-time")
        console.print("🛑 Press Ctrl+C to stop simulation\n")
        
        import random
        import time
        
        try:
            for i in range(60):  # Run for 60 iterations
                # Simulate various metrics
                response_time = random.uniform(0.5, 3.0)
                tokens_per_sec = random.uniform(10, 30)
                memory = random.uniform(40, 80)
                cpu = random.uniform(20, 70)
                
                # Record metrics
                collector = self.dashboard.metrics_collector
                collector.record_metric("response_time", response_time)
                collector.record_metric("tokens_per_second", tokens_per_sec)
                collector.record_metric("memory_usage", memory)
                collector.record_metric("cpu_usage", cpu)
                
                # Simulate cache hits/misses
                if random.random() > 0.3:
                    collector.record_metric("cache_hit", 1)
                else:
                    collector.record_metric("cache_miss", 1)
                
                # Simulate usage events
                models = ["gpt-4", "gpt-3.5-turbo", "claude-3", "gemini-pro"]
                model = random.choice(models)
                user = f"user_{random.randint(1, 20)}"
                tokens = random.randint(50, 500)
                cost = tokens * 0.002
                success = random.random() > 0.05
                
                collector.record_usage_event("chat", user, model, tokens, cost, success)
                
                # Show current metrics
                current_metrics = collector.get_performance_metrics()
                console.print(f"📊 Iteration {i+1:2d}: RT={response_time:.2f}s, "
                             f"TPS={tokens_per_sec:.1f}, Cache={current_metrics.cache_hit_rate:.1f}%")
                
                await asyncio.sleep(1)  # 1 second intervals
                
        except KeyboardInterrupt:
            console.print("\n✅ Real-time simulation stopped")
    
    def view_system_insights(self):
        """Display comprehensive system insights"""
        console.print("\n🧠 AI-Powered System Insights:")
        
        # Get all insights
        performance_insights = self.dashboard.analytics_engine._generate_performance_insights()
        usage_data = self.dashboard.analytics_engine.analyze_usage_patterns()
        cost_metrics = self.dashboard.cost_optimizer.calculate_cost_metrics()
        
        from rich.tree import Tree
        
        insights_tree = Tree("🎯 System Analysis")
        
        # Performance branch
        perf_branch = insights_tree.add("⚡ Performance Analysis")
        for insight in performance_insights:
            perf_branch.add(insight)
        
        # Usage branch
        usage_branch = insights_tree.add("📈 Usage Patterns")
        trends = usage_data["usage_trends"]
        usage_branch.add(f"Growth Rate: {trends['growth_rate']:.1f}% ({'📈 Growing' if trends['growth_rate'] > 0 else '📉 Declining'})")
        usage_branch.add(f"Peak Activity: {usage_data['peak_usage_hour']}:00 (Optimize capacity)")
        usage_branch.add(f"User Engagement: {usage_data['user_behavior']['active_users']} active users")
        
        # Cost branch
        cost_branch = insights_tree.add("💰 Cost Optimization")
        if cost_metrics.optimization_savings > 0:
            cost_branch.add(f"💡 Save ${cost_metrics.optimization_savings:.4f}/day with smart routing")
            cost_branch.add("🎯 Consider cheaper models for simple queries")
        else:
            cost_branch.add("✅ Cost optimization maximized")
        
        # Recommendations branch
        rec_branch = insights_tree.add("🚀 Recommendations")
        rec_branch.add("🔧 Implement adaptive caching strategies")
        rec_branch.add("📊 Set up automated performance alerts")
        rec_branch.add("💡 Enable ML-powered model selection")
        rec_branch.add("🎯 Create user-specific optimization profiles")
        
        console.print(insights_tree)
    
    async def run_demo(self):
        """Run the interactive demo"""
        try:
            self.show_welcome()
            
            while self.demo_running:
                self.show_main_menu()
                choice = Prompt.ask("\n🎯 Select an option", default="0")
                
                try:
                    if choice == "1":
                        self.generate_sample_data()
                    elif choice == "2":
                        self.view_performance_metrics()
                    elif choice == "3":
                        self.analyze_usage_patterns()
                    elif choice == "4":
                        self.analyze_costs()
                    elif choice == "5":
                        await self.run_live_dashboard()
                    elif choice == "6":
                        self.generate_report()
                    elif choice == "7":
                        await self.simulate_realtime_data()
                    elif choice == "8":
                        self.view_system_insights()
                    elif choice == "0":
                        self.demo_running = False
                    else:
                        console.print("❌ Invalid option")
                
                except Exception as e:
                    console.print(f"❌ Error: {e}")
                
                if self.demo_running and choice != "0":
                    Prompt.ask("\n⏸️  Press Enter to continue")
            
            console.print("\n👋 Thanks for exploring the Advanced Analytics Dashboard!")
            
        finally:
            # Cleanup
            if self.temp_db.exists():
                self.temp_db.unlink()


async def main():
    """Main entry point"""
    demo = AnalyticsDashboardDemo()
    await demo.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        console.print(f"\n❌ Demo error: {e}")