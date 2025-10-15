#!/usr/bin/env python3
"""
Demo script for Advanced Analytics Features

This script demonstrates the advanced analytics capabilities including:
- Usage pattern analysis and user behavior insights
- Cost tracking and optimization recommendations
- ML-powered trend analysis and anomaly detection
- Comprehensive analytics reporting
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xencode'))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.tree import Tree
from rich import box

async def main():
    """Main demo function"""
    console = Console()
    
    console.print("🚀 [bold cyan]Xencode Advanced Analytics Features Demo[/bold cyan]\n")
    console.print("This demo showcases ML-powered analytics, cost optimization, and usage insights.\n")
    
    try:
        # Import the advanced analytics engine
        from advanced_analytics_engine import AdvancedAnalyticsEngine
        
        console.print("✅ Advanced Analytics Engine loaded successfully")
        
        # Create analytics engine
        engine = AdvancedAnalyticsEngine()
        
        console.print("📊 Generating comprehensive sample data (7 days)...")
        engine.generate_sample_data(days=7)
        console.print("✅ Sample data generated with realistic patterns\n")
        
        # Run comprehensive analysis
        console.print("🔍 Running comprehensive analytics analysis...")
        results = await engine.run_comprehensive_analysis(hours=168)  # 1 week
        console.print("✅ Analysis complete!\n")
        
        # Display results in organized panels
        display_analysis_results(console, results)
        
        # Show detailed insights
        console.print("\n" + "="*80)
        console.print("🔬 [bold yellow]Detailed Analytics Insights[/bold yellow]")
        console.print("="*80 + "\n")
        
        await show_detailed_insights(console, engine)
        
    except ImportError as e:
        console.print(f"❌ [red]Import error: {e}[/red]")
        console.print("Please ensure the advanced analytics engine is available.")
    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()

def display_analysis_results(console: Console, results: dict):
    """Display analysis results in organized panels"""
    
    # Summary Panel
    summary = results.get("summary", {})
    summary_table = Table(box=box.SIMPLE, show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("📋 Patterns Detected", str(summary.get('patterns_detected', 0)))
    summary_table.add_row("👥 Users Analyzed", str(summary.get('users_analyzed', 0)))
    summary_table.add_row("💡 Optimizations Found", str(summary.get('optimizations_found', 0)))
    summary_table.add_row("💰 Potential Savings", f"${summary.get('total_potential_savings', 0):.2f}")
    summary_table.add_row("🎯 High-Impact Optimizations", str(summary.get('high_impact_optimizations', 0)))
    summary_table.add_row("📈 Trends Analyzed", str(summary.get('trends_analyzed', 0)))
    
    summary_panel = Panel(summary_table, title="📊 Analysis Summary", border_style="green")
    
    # Usage Patterns Panel
    patterns = results.get("usage_patterns", [])
    if patterns:
        patterns_tree = Tree("🔍 Detected Usage Patterns")
        for i, pattern in enumerate(patterns[:5]):  # Show top 5
            pattern_node = patterns_tree.add(f"[bold]{pattern['type']}[/bold] (Confidence: {pattern['confidence']:.1%})")
            pattern_node.add(f"[dim]{pattern['description']}[/dim]")
    else:
        patterns_tree = Text("No usage patterns detected", style="dim")
    
    patterns_panel = Panel(patterns_tree, title="🔍 Usage Patterns", border_style="blue")
    
    # Cost Optimizations Panel
    optimizations = results.get("cost_optimizations", [])
    if optimizations:
        opt_table = Table(box=box.SIMPLE, show_header=True)
        opt_table.add_column("Optimization", style="cyan")
        opt_table.add_column("Savings", style="green")
        opt_table.add_column("Effort", style="yellow")
        
        for opt in optimizations[:5]:  # Show top 5
            opt_table.add_row(
                opt['title'][:40] + "..." if len(opt['title']) > 40 else opt['title'],
                f"${opt['potential_savings']:.2f}",
                opt['implementation_effort']
            )
    else:
        opt_table = Text("No cost optimizations found", style="dim")
    
    opt_panel = Panel(opt_table, title="💡 Cost Optimizations", border_style="yellow")
    
    # ROI Projections Panel
    roi = results.get("roi_projections", {})
    if roi:
        roi_table = Table(box=box.SIMPLE, show_header=False)
        roi_table.add_column("Metric", style="cyan")
        roi_table.add_column("Value", style="green")
        
        roi_table.add_row("Monthly Savings", f"${roi.get('potential_monthly_savings', 0):.2f}")
        roi_table.add_row("Annual Savings", f"${roi.get('potential_annual_savings', 0):.2f}")
        roi_table.add_row("Implementation Cost", f"${roi.get('implementation_cost', 0):.2f}")
        roi_table.add_row("ROI Percentage", f"{roi.get('roi_percentage', 0):.1f}%")
        roi_table.add_row("Payback Period", f"{roi.get('payback_period_months', 0):.1f} months")
    else:
        roi_table = Text("No ROI data available", style="dim")
    
    roi_panel = Panel(roi_table, title="💰 ROI Projections", border_style="magenta")
    
    # Display panels in columns
    console.print(summary_panel)
    console.print()
    
    columns = Columns([patterns_panel, opt_panel], equal=True)
    console.print(columns)
    console.print()
    
    console.print(roi_panel)

async def show_detailed_insights(console: Console, engine):
    """Show detailed analytics insights"""
    
    # Usage Pattern Analysis
    console.print("🔍 [bold blue]Usage Pattern Analysis[/bold blue]")
    patterns = engine.usage_analyzer.analyze_usage_patterns(hours=168)
    
    if patterns:
        for i, pattern in enumerate(patterns[:3]):
            console.print(f"\n   {i+1}. [bold]{pattern.pattern_type.replace('_', ' ').title()}[/bold]")
            console.print(f"      📝 {pattern.description}")
            console.print(f"      🎯 Confidence: {pattern.confidence:.1%}")
            console.print(f"      📊 Frequency: {pattern.frequency:.1%}")
            
            if pattern.metadata:
                console.print(f"      📋 Details: {pattern.metadata}")
    else:
        console.print("   No significant usage patterns detected")
    
    # User Behavior Profiles
    console.print("\n👥 [bold green]User Behavior Profiles[/bold green]")
    profiles = engine.usage_analyzer.generate_user_profiles(hours=168)
    
    if profiles:
        behavior_clusters = {}
        for profile in profiles:
            cluster = profile.behavior_cluster
            if cluster not in behavior_clusters:
                behavior_clusters[cluster] = []
            behavior_clusters[cluster].append(profile)
        
        for cluster, cluster_profiles in behavior_clusters.items():
            console.print(f"\n   📊 [bold]{cluster.replace('_', ' ').title()}[/bold] ({len(cluster_profiles)} users)")
            
            # Show example profile
            example = cluster_profiles[0]
            console.print(f"      🔄 Usage Frequency: {example.usage_frequency}")
            console.print(f"      🤖 Preferred Models: {', '.join(example.preferred_models[:3])}")
            console.print(f"      💰 Cost Efficiency: {example.cost_efficiency_score:.1%}")
            
            if example.recommendations:
                console.print(f"      💡 Recommendations:")
                for rec in example.recommendations[:2]:
                    console.print(f"         • {rec}")
    else:
        console.print("   No user profiles available")
    
    # Cost Optimization Analysis
    console.print("\n💰 [bold yellow]Cost Optimization Analysis[/bold yellow]")
    optimizations = engine.cost_optimizer.analyze_cost_optimization_opportunities(hours=168)
    
    if optimizations:
        # Group by optimization type
        opt_types = {}
        for opt in optimizations:
            opt_type = opt.optimization_type
            if opt_type not in opt_types:
                opt_types[opt_type] = []
            opt_types[opt_type].append(opt)
        
        for opt_type, type_opts in opt_types.items():
            total_savings = sum(opt.potential_savings for opt in type_opts)
            console.print(f"\n   💡 [bold]{opt_type.replace('_', ' ').title()}[/bold]")
            console.print(f"      💰 Total Potential Savings: ${total_savings:.2f}")
            console.print(f"      📊 Opportunities: {len(type_opts)}")
            
            # Show top opportunity
            top_opt = max(type_opts, key=lambda x: x.potential_savings)
            console.print(f"      🎯 Top Opportunity: {top_opt.title}")
            console.print(f"         💵 Savings: ${top_opt.potential_savings:.2f}")
            console.print(f"         🔧 Effort: {top_opt.implementation_effort}")
    else:
        console.print("   No cost optimization opportunities found")
    
    # Trend Analysis
    console.print("\n📈 [bold magenta]ML-Powered Trend Analysis[/bold magenta]")
    
    key_metrics = ["cpu_usage", "memory_usage", "response_time"]
    for metric in key_metrics:
        try:
            trend_analysis = engine.trend_analyzer.analyze_trends(metric, hours=168)
            
            console.print(f"\n   📊 [bold]{metric.replace('_', ' ').title()}[/bold]")
            console.print(f"      📈 Trend: {trend_analysis.trend_direction}")
            console.print(f"      💪 Strength: {trend_analysis.trend_strength:.1%}")
            console.print(f"      🔄 Seasonality: {'Yes' if trend_analysis.seasonality_detected else 'No'}")
            console.print(f"      ⚠️ Anomalies: {len(trend_analysis.anomalies_detected)}")
            console.print(f"      🔮 Predictions: {len(trend_analysis.predicted_values)} data points")
            
            if trend_analysis.anomalies_detected:
                console.print(f"      🚨 Recent Anomalies: {len([a for a in trend_analysis.anomalies_detected if (engine.trend_analyzer._get_current_time() - a.timestamp()).total_seconds() < 86400])}")
                
        except Exception as e:
            console.print(f"   ❌ Error analyzing {metric}: {str(e)}")
    
    # Performance Insights
    console.print("\n🎯 [bold red]Key Performance Insights[/bold red]")
    
    # Generate insights based on analysis
    insights = []
    
    if patterns:
        pattern_types = [p.pattern_type for p in patterns]
        if "model_dominance" in pattern_types:
            insights.append("🤖 Model usage is concentrated - consider diversification")
        if "user_segmentation" in pattern_types:
            insights.append("👥 Clear user segments detected - tailor experiences accordingly")
    
    if optimizations:
        total_savings = sum(opt.potential_savings for opt in optimizations)
        if total_savings > 100:
            insights.append(f"💰 Significant cost savings available: ${total_savings:.2f}")
        
        high_impact = [opt for opt in optimizations if opt.impact_score > 0.5]
        if high_impact:
            insights.append(f"🎯 {len(high_impact)} high-impact optimizations identified")
    
    if not insights:
        insights.append("✅ System is operating efficiently with no major issues detected")
    
    for insight in insights:
        console.print(f"   {insight}")
    
    console.print("\n✨ [green]Advanced analytics analysis complete![/green]")
    console.print("\n💡 [dim]Tip: Run this analysis regularly to maintain optimal performance and costs.[/dim]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")