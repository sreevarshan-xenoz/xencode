#!/usr/bin/env python3
"""
Resource Management System Demo

Demonstrates the comprehensive resource management capabilities including:
- Memory tracking and optimization
- Garbage collection management
- Resource pooling for expensive operations
- Temporary file management
- Automated cleanup operations
"""

import asyncio
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from xencode.monitoring.resource_manager import (
    get_resource_manager, 
    ResourceType, 
    CleanupPriority,
    ResourceLimit
)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich import box

console = Console()


def create_memory_intensive_objects():
    """Create memory-intensive objects to demonstrate cleanup"""
    # Create large lists to consume memory
    large_data = []
    for i in range(1000):
        large_data.append([j for j in range(1000)])
    
    # Create some circular references to test GC
    class Node:
        def __init__(self, value):
            self.value = value
            self.children = []
            self.parent = None
    
    # Create circular reference structure
    root = Node("root")
    for i in range(100):
        child = Node(f"child_{i}")
        child.parent = root
        root.children.append(child)
        
        # Create grandchildren with circular refs
        for j in range(10):
            grandchild = Node(f"grandchild_{i}_{j}")
            grandchild.parent = child
            child.children.append(grandchild)
    
    return large_data, root


async def demonstrate_memory_tracking():
    """Demonstrate memory tracking capabilities"""
    console.print("\n[bold blue]🧠 Memory Tracking Demonstration[/bold blue]")
    
    resource_manager = await get_resource_manager()
    
    # Take initial snapshot
    console.print("📸 Taking initial memory snapshot...")
    resource_manager.memory_tracker.take_snapshot("initial")
    
    # Create memory-intensive objects
    console.print("🏗️  Creating memory-intensive objects...")
    large_data, circular_refs = create_memory_intensive_objects()
    
    # Take snapshot after allocation
    resource_manager.memory_tracker.take_snapshot("after_allocation")
    
    # Analyze memory growth
    analysis = resource_manager.memory_tracker.analyze_memory_growth()
    
    table = Table(title="Memory Growth Analysis", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Growth", f"{analysis.get('total_growth_mb', 0):.2f} MB")
    table.add_row("Memory Leaks Detected", str(analysis.get('memory_leaks_detected', False)))
    table.add_row("Top Growing Files", str(len(analysis.get('top_growing_files', []))))
    
    console.print(table)
    
    # Show current memory usage
    memory_usage = resource_manager.memory_tracker.get_current_memory_usage()
    
    usage_table = Table(title="Current Memory Usage", box=box.ROUNDED)
    usage_table.add_column("Metric", style="cyan")
    usage_table.add_column("Value", style="yellow")
    
    for key, value in memory_usage.items():
        if isinstance(value, float):
            usage_table.add_row(key.replace('_', ' ').title(), f"{value:.2f}")
        else:
            usage_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(usage_table)
    
    return large_data, circular_refs


async def demonstrate_resource_pooling():
    """Demonstrate resource pooling for expensive operations"""
    console.print("\n[bold blue]🏊 Resource Pooling Demonstration[/bold blue]")
    
    resource_manager = await get_resource_manager()
    
    # Create a resource pool for expensive database connections (simulated)
    class MockDatabaseConnection:
        def __init__(self):
            self.connection_id = time.time()
            time.sleep(0.1)  # Simulate expensive connection setup
        
        def query(self, sql):
            return f"Result for: {sql} (conn: {self.connection_id})"
        
        def close(self):
            pass  # Cleanup logic
    
    # Create resource pool
    db_pool = resource_manager.create_resource_pool(
        "database_connections",
        MockDatabaseConnection,
        max_size=5,
        cleanup_function=lambda conn: conn.close()
    )
    
    console.print("🔧 Created database connection pool (max_size=5)")
    
    # Demonstrate pool usage
    connections = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Acquire connections
        task = progress.add_task("Acquiring connections...", total=8)
        
        for i in range(8):
            conn = db_pool.acquire()
            connections.append(conn)
            result = conn.query(f"SELECT * FROM table_{i}")
            console.print(f"  📊 Query {i+1}: {result[:50]}...")
            progress.advance(task)
            time.sleep(0.1)
        
        # Show pool statistics
        stats = db_pool.get_stats()
        
        stats_table = Table(title="Resource Pool Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        for key, value in stats.items():
            stats_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(stats_table)
        
        # Release connections
        release_task = progress.add_task("Releasing connections...", total=len(connections))
        
        for conn in connections:
            db_pool.release(conn)
            progress.advance(release_task)
            time.sleep(0.05)
    
    console.print("✅ All connections released back to pool")


async def demonstrate_temp_file_management():
    """Demonstrate temporary file management"""
    console.print("\n[bold blue]📁 Temporary File Management Demonstration[/bold blue]")
    
    resource_manager = await get_resource_manager()
    temp_manager = resource_manager.temp_file_manager
    
    # Create several temporary files
    temp_files = []
    
    console.print("📝 Creating temporary files...")
    for i in range(5):
        file_id, file_path = temp_manager.create_temp_file(
            prefix=f"demo_{i}_",
            suffix=".txt"
        )
        
        # Write some data to the file
        with open(file_path, 'w') as f:
            f.write(f"This is temporary file {i}\n" * 100)
        
        temp_files.append((file_id, file_path))
        console.print(f"  📄 Created: {file_path.name}")
    
    # Show temporary file usage
    usage = temp_manager.get_temp_usage()
    
    usage_table = Table(title="Temporary File Usage", box=box.ROUNDED)
    usage_table.add_column("Metric", style="cyan")
    usage_table.add_column("Value", style="yellow")
    
    usage_table.add_row("Total Files", str(usage["total_files"]))
    usage_table.add_row("Total Size", f"{usage['total_size_mb']:.2f} MB")
    usage_table.add_row("Base Directory", usage["base_directory"])
    
    console.print(usage_table)
    
    # Cleanup some files manually
    console.print("\n🧹 Cleaning up temporary files...")
    for file_id, file_path in temp_files[:2]:
        if temp_manager.cleanup_temp_file(file_id):
            console.print(f"  ✅ Cleaned: {file_path.name}")
    
    # Show updated usage
    updated_usage = temp_manager.get_temp_usage()
    console.print(f"📊 Files remaining: {updated_usage['total_files']}")


async def demonstrate_automated_cleanup():
    """Demonstrate automated cleanup operations"""
    console.print("\n[bold blue]🤖 Automated Cleanup Demonstration[/bold blue]")
    
    resource_manager = await get_resource_manager()
    
    # Show current resource usage
    usage = await resource_manager.get_resource_usage()
    
    usage_table = Table(title="Resource Usage Before Cleanup", box=box.ROUNDED)
    usage_table.add_column("Resource Type", style="cyan")
    usage_table.add_column("Current Usage", style="yellow")
    usage_table.add_column("Limit", style="red")
    usage_table.add_column("Status", style="green")
    
    for resource_type, resource_usage in usage.items():
        limit_str = "N/A"
        status = "✅ OK"
        
        if resource_usage.limit:
            limit_str = f"{resource_usage.limit.soft_limit}/{resource_usage.limit.hard_limit} {resource_usage.limit.unit}"
            
            if resource_usage.current_usage >= resource_usage.limit.hard_limit:
                status = "🔴 CRITICAL"
            elif resource_usage.current_usage >= resource_usage.limit.soft_limit:
                status = "🟡 WARNING"
        
        usage_table.add_row(
            resource_type.value.replace('_', ' ').title(),
            f"{resource_usage.current_usage:.1f}",
            limit_str,
            status
        )
    
    console.print(usage_table)
    
    # Trigger cleanup operations
    console.print("\n🧹 Triggering cleanup operations...")
    
    cleanup_results = await resource_manager.trigger_cleanup(CleanupPriority.LOW)
    
    results_table = Table(title="Cleanup Results", box=box.ROUNDED)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Tasks Executed", str(cleanup_results["tasks_executed"]))
    results_table.add_row("Tasks Successful", str(cleanup_results["tasks_successful"]))
    results_table.add_row("Memory Freed", f"{cleanup_results['memory_freed_mb']:.2f} MB")
    results_table.add_row("Errors", str(len(cleanup_results["errors"])))
    
    console.print(results_table)
    
    if cleanup_results["errors"]:
        console.print("\n[red]Errors encountered:[/red]")
        for error in cleanup_results["errors"]:
            console.print(f"  ❌ {error}")


async def demonstrate_resource_limits():
    """Demonstrate resource limit monitoring"""
    console.print("\n[bold blue]⚠️  Resource Limit Monitoring Demonstration[/bold blue]")
    
    resource_manager = await get_resource_manager()
    
    # Set a low memory limit for demonstration
    resource_manager.resource_limits[ResourceType.MEMORY] = ResourceLimit(
        ResourceType.MEMORY,
        soft_limit=1.0,   # Very low limit for demo
        hard_limit=2.0,
        unit="percentage"
    )
    
    console.print("🔧 Set low memory limits for demonstration")
    console.print("   Soft limit: 1.0%, Hard limit: 2.0%")
    
    # Check for violations
    violations = await resource_manager.check_resource_limits()
    
    if violations:
        violations_table = Table(title="Resource Limit Violations", box=box.ROUNDED)
        violations_table.add_column("Resource", style="cyan")
        violations_table.add_column("Current", style="yellow")
        violations_table.add_column("Limit", style="red")
        violations_table.add_column("Severity", style="magenta")
        
        for violation in violations:
            limit = violation.limit
            severity = "🔴 CRITICAL" if violation.current_usage >= limit.hard_limit else "🟡 WARNING"
            
            violations_table.add_row(
                violation.resource_type.value.replace('_', ' ').title(),
                f"{violation.current_usage:.1f} {limit.unit}",
                f"{limit.soft_limit}/{limit.hard_limit} {limit.unit}",
                severity
            )
        
        console.print(violations_table)
    else:
        console.print("✅ No resource limit violations detected")
    
    # Reset to reasonable limits
    resource_manager.resource_limits[ResourceType.MEMORY] = ResourceLimit(
        ResourceType.MEMORY,
        soft_limit=75.0,
        hard_limit=90.0,
        unit="percentage"
    )
    console.print("🔧 Reset memory limits to normal values")


async def show_comprehensive_statistics():
    """Show comprehensive resource management statistics"""
    console.print("\n[bold blue]📊 Comprehensive Statistics[/bold blue]")
    
    resource_manager = await get_resource_manager()
    stats = resource_manager.get_statistics()
    
    # Cleanup statistics
    cleanup_table = Table(title="Cleanup Statistics", box=box.ROUNDED)
    cleanup_table.add_column("Metric", style="cyan")
    cleanup_table.add_column("Value", style="green")
    
    cleanup_stats = stats["cleanup_stats"]
    cleanup_table.add_row("Total Cleanups", str(cleanup_stats["total_cleanups"]))
    cleanup_table.add_row("Memory Freed", f"{cleanup_stats['memory_freed_mb']:.2f} MB")
    cleanup_table.add_row("Files Cleaned", str(cleanup_stats["files_cleaned"]))
    cleanup_table.add_row("Last Cleanup", str(cleanup_stats["last_cleanup"]))
    
    console.print(cleanup_table)
    
    # Resource pools statistics
    if stats["resource_pools"]:
        pools_table = Table(title="Resource Pools Statistics", box=box.ROUNDED)
        pools_table.add_column("Pool Name", style="cyan")
        pools_table.add_column("Pool Size", style="yellow")
        pools_table.add_column("In Use", style="red")
        pools_table.add_column("Reuse Ratio", style="green")
        
        for pool_name, pool_stats in stats["resource_pools"].items():
            pools_table.add_row(
                pool_name,
                str(pool_stats["pool_size"]),
                str(pool_stats["in_use"]),
                f"{pool_stats['reuse_ratio']:.2%}"
            )
        
        console.print(pools_table)
    
    # Memory analysis
    memory_analysis = stats["memory_analysis"]
    if not memory_analysis.get("error"):
        memory_table = Table(title="Memory Analysis", box=box.ROUNDED)
        memory_table.add_column("Metric", style="cyan")
        memory_table.add_column("Value", style="yellow")
        
        memory_table.add_row("Total Growth", f"{memory_analysis.get('total_growth_mb', 0):.2f} MB")
        memory_table.add_row("Memory Leaks Detected", str(memory_analysis.get('memory_leaks_detected', False)))
        memory_table.add_row("Top Growing Files", str(len(memory_analysis.get('top_growing_files', []))))
        
        console.print(memory_table)


async def main():
    """Main demonstration function"""
    console.print(Panel.fit(
        "[bold cyan]Xencode Resource Management System Demo[/bold cyan]\n"
        "Comprehensive resource monitoring, optimization, and cleanup",
        border_style="blue"
    ))
    
    try:
        # Initialize resource management
        console.print("\n[yellow]🚀 Initializing Resource Management System...[/yellow]")
        resource_manager = await get_resource_manager()
        
        # Run demonstrations
        large_data, circular_refs = await demonstrate_memory_tracking()
        await demonstrate_resource_pooling()
        await demonstrate_temp_file_management()
        await demonstrate_resource_limits()
        await demonstrate_automated_cleanup()
        await show_comprehensive_statistics()
        
        # Cleanup demo objects
        console.print("\n[yellow]🧹 Cleaning up demo objects...[/yellow]")
        del large_data, circular_refs
        
        # Final cleanup
        final_cleanup = await resource_manager.trigger_cleanup(CleanupPriority.HIGH)
        console.print(f"✅ Final cleanup completed: {final_cleanup['memory_freed_mb']:.2f} MB freed")
        
        console.print("\n[bold green]✅ Resource Management Demo Completed Successfully![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo failed: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            resource_manager = await get_resource_manager()
            await resource_manager.stop()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())