#!/usr/bin/env python3
"""
Demo: Multi-Modal Cache System

This demo showcases the enhanced caching system for multi-modal processing
including document processing cache, code analysis cache, and workspace cache
with intelligent invalidation and warming strategies.
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from xencode.cache.multimodal_cache import (
    MultiModalCacheSystem,
    DocumentProcessingCache,
    CodeAnalysisCache,
    WorkspaceCache,
    CacheType,
    get_multimodal_cache,
    initialize_multimodal_cache
)
from xencode.models.document import ProcessedDocument, DocumentType
from xencode.models.code_analysis import CodeAnalysisResult, AnalysisIssue, AnalysisType, SeverityLevel, CodeLocation
from xencode.advanced_cache_system import HybridCacheManager


async def demo_multimodal_cache():
    """Demonstrate multi-modal cache system capabilities"""
    
    console = Console()
    console.print("🚀 [bold cyan]Multi-Modal Cache System Demo[/bold cyan]\n")
    
    # Initialize the cache system
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing multi-modal cache system...", total=None)
        
        # Create base cache system
        base_cache = HybridCacheManager(memory_cache_mb=128, disk_cache_mb=512)
        multimodal_cache = MultiModalCacheSystem(base_cache)
        
        progress.update(task, completed=True)
    
    console.print("✅ Multi-modal cache system initialized\n")
    
    # Demo 1: Document Processing Cache
    console.print("📄 [bold yellow]Document Processing Cache Demo[/bold yellow]")
    
    # Create sample documents
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a sample document for testing the cache system. " * 10)
        sample_doc_path = f.name
    
    try:
        # Simulate document processing
        processed_doc = ProcessedDocument(
            id="doc_001",
            original_filename="sample.txt",
            document_type=DocumentType.PDF,
            extracted_text="This is extracted text from the document...",
            metadata={"pages": 5, "language": "en"},
            processing_time_ms=1500,
            confidence_score=0.95
        )
        
        # Cache the processed document
        console.print("  📝 Caching processed document...")
        success = await multimodal_cache.document_cache.cache_processed_document(
            sample_doc_path, DocumentType.PDF, processed_doc
        )
        console.print(f"  ✅ Document cached: {success}")
        
        # Retrieve from cache
        console.print("  🔍 Retrieving from cache...")
        cached_doc = await multimodal_cache.document_cache.get_processed_document(
            sample_doc_path, DocumentType.PDF
        )
        
        if cached_doc:
            console.print(f"  ✅ Cache hit! Retrieved document: {cached_doc.id}")
            console.print(f"     Processing time: {cached_doc.processing_time_ms}ms")
            console.print(f"     Confidence: {cached_doc.confidence_score}")
        else:
            console.print("  ❌ Cache miss")
        
        # Test cache invalidation
        console.print("  🗑️  Testing cache invalidation...")
        invalidated = await multimodal_cache.document_cache.invalidate_document_cache(sample_doc_path)
        console.print(f"  ✅ Invalidated {invalidated} cache entries")
        
    finally:
        # Cleanup
        Path(sample_doc_path).unlink(missing_ok=True)
    
    console.print()
    
    # Demo 2: Code Analysis Cache
    console.print("🔍 [bold yellow]Code Analysis Cache Demo[/bold yellow]")
    
    # Sample code for analysis
    sample_code = '''
def fibonacci(n):
    """Calculate fibonacci number recursively"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# This could be optimized with memoization
result = fibonacci(10)
print(f"Fibonacci(10) = {result}")
'''
    
    # Simulate code analysis result
    analysis_result = CodeAnalysisResult(
        language="python",
        syntax_errors=[],
        suggestions=[
            AnalysisIssue(
                analysis_type=AnalysisType.PERFORMANCE,
                severity=SeverityLevel.WARNING,
                message="Recursive fibonacci is inefficient",
                description="Consider using memoization or iterative approach",
                location=CodeLocation(line=4, column=5),
                code_snippet="return fibonacci(n-1) + fibonacci(n-2)",
                rule_id="performance_recursion",
                rule_name="Inefficient Recursion",
                confidence=0.8
            )
        ],
        complexity_score=15,
        security_issues=[],
        performance_hints=[]
    )
    
    # Cache the analysis result
    console.print("  🔍 Caching code analysis result...")
    success = await multimodal_cache.code_cache.cache_analysis_result(
        sample_code, "python", analysis_result
    )
    console.print(f"  ✅ Analysis result cached: {success}")
    
    # Retrieve from cache
    console.print("  📊 Retrieving analysis from cache...")
    cached_analysis = await multimodal_cache.code_cache.get_analysis_result(
        sample_code, "python"
    )
    
    if cached_analysis:
        console.print(f"  ✅ Cache hit! Retrieved analysis for {cached_analysis.language}")
        console.print(f"     Complexity score: {cached_analysis.complexity_score}")
        console.print(f"     Suggestions: {len(cached_analysis.suggestions)}")
    else:
        console.print("  ❌ Cache miss")
    
    console.print()
    
    # Demo 3: Workspace Cache
    console.print("🏢 [bold yellow]Workspace Cache Demo[/bold yellow]")
    
    # Sample workspace data
    workspace_data = {
        "files": [
            {"name": "main.py", "size": 1024, "modified": "2024-10-15T10:30:00Z"},
            {"name": "utils.py", "size": 512, "modified": "2024-10-15T09:15:00Z"},
            {"name": "config.json", "size": 256, "modified": "2024-10-14T16:45:00Z"}
        ],
        "settings": {
            "theme": "dark",
            "auto_save": True,
            "tab_size": 4
        },
        "recent_activity": [
            {"action": "file_opened", "file": "main.py", "timestamp": "2024-10-15T10:30:00Z"},
            {"action": "file_saved", "file": "utils.py", "timestamp": "2024-10-15T10:25:00Z"}
        ]
    }
    
    workspace_id = "workspace_demo_001"
    
    # Cache workspace data
    console.print("  💾 Caching workspace data...")
    success = await multimodal_cache.workspace_cache.cache_workspace_data(
        workspace_id, "files", workspace_data["files"], version=1
    )
    console.print(f"  ✅ Workspace files cached: {success}")
    
    success = await multimodal_cache.workspace_cache.cache_workspace_data(
        workspace_id, "settings", workspace_data["settings"], version=1
    )
    console.print(f"  ✅ Workspace settings cached: {success}")
    
    # Retrieve from cache
    console.print("  📂 Retrieving workspace data from cache...")
    cached_files = await multimodal_cache.workspace_cache.get_workspace_data(
        workspace_id, "files"
    )
    cached_settings = await multimodal_cache.workspace_cache.get_workspace_data(
        workspace_id, "settings"
    )
    
    if cached_files and cached_settings:
        console.print(f"  ✅ Cache hit! Retrieved workspace data")
        console.print(f"     Files: {len(cached_files)} items")
        console.print(f"     Settings: {len(cached_settings)} items")
    else:
        console.print("  ❌ Cache miss")
    
    # Test workspace invalidation
    console.print("  🔄 Testing workspace cache invalidation...")
    invalidated = await multimodal_cache.workspace_cache.invalidate_workspace_cache(workspace_id)
    console.print(f"  ✅ Invalidated {invalidated} workspace cache entries")
    
    console.print()
    
    # Demo 4: Cache Warming
    console.print("🔥 [bold yellow]Cache Warming Demo[/bold yellow]")
    
    # Add warming tasks
    console.print("  📋 Adding cache warming tasks...")
    
    # Create temporary files for warming demo
    temp_files = []
    for i, ext in enumerate(['.pdf', '.docx', '.html']):
        with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
            f.write(f"Sample document {i+1} content for warming demo")
            temp_files.append(f.name)
    
    try:
        # Add document warming tasks
        await multimodal_cache.warming_manager.warm_frequently_accessed_documents(temp_files)
        console.print(f"  ✅ Added {len(temp_files)} document warming tasks")
        
        # Add workspace warming tasks
        await multimodal_cache.warming_manager.warm_workspace_data(
            workspace_id, ["files", "settings", "history"]
        )
        console.print(f"  ✅ Added workspace warming tasks")
        
        # Show warming queue
        queue_size = len(multimodal_cache.warming_manager.warming_queue)
        console.print(f"  📊 Warming queue size: {queue_size} tasks")
        
        # Process some warming tasks
        console.print("  ⚡ Processing warming tasks...")
        await multimodal_cache.warming_manager.process_warming_queue(max_tasks=3)
        
        remaining_tasks = len(multimodal_cache.warming_manager.warming_queue)
        console.print(f"  ✅ Processed warming tasks. Remaining: {remaining_tasks}")
        
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            Path(temp_file).unlink(missing_ok=True)
    
    console.print()
    
    # Demo 5: Cache Statistics and Monitoring
    console.print("📊 [bold yellow]Cache Statistics and Monitoring[/bold yellow]")
    
    # Get comprehensive statistics
    stats = await multimodal_cache.get_cache_statistics()
    
    # Create statistics table
    stats_table = Table(title="Multi-Modal Cache Statistics")
    stats_table.add_column("Category", style="cyan")
    stats_table.add_column("Metric", style="yellow")
    stats_table.add_column("Value", style="green")
    
    # Base cache stats
    base_stats = stats.get('base_cache', {})
    stats_table.add_row("Base Cache", "Memory Usage", f"{base_stats.get('memory_usage_mb', 0):.2f} MB")
    stats_table.add_row("Base Cache", "Disk Usage", f"{base_stats.get('disk_usage_mb', 0):.2f} MB")
    stats_table.add_row("Base Cache", "Hit Rate", f"{base_stats.get('hit_rate', 0):.1f}%")
    
    # Multimodal stats
    multimodal_stats = stats.get('multimodal', {})
    stats_table.add_row("Documents", "Cache Hits", str(multimodal_stats.get('document_hits', 0)))
    stats_table.add_row("Documents", "Cache Misses", str(multimodal_stats.get('document_misses', 0)))
    stats_table.add_row("Code Analysis", "Cache Hits", str(multimodal_stats.get('code_hits', 0)))
    stats_table.add_row("Code Analysis", "Cache Misses", str(multimodal_stats.get('code_misses', 0)))
    stats_table.add_row("Workspace", "Cache Hits", str(multimodal_stats.get('workspace_hits', 0)))
    stats_table.add_row("Workspace", "Cache Misses", str(multimodal_stats.get('workspace_misses', 0)))
    stats_table.add_row("System", "Invalidations", str(multimodal_stats.get('invalidations', 0)))
    stats_table.add_row("System", "Warming Tasks", str(multimodal_stats.get('warming_tasks', 0)))
    
    console.print(stats_table)
    console.print()
    
    # Demo 6: Change Event Handling
    console.print("🔄 [bold yellow]Change Event Handling Demo[/bold yellow]")
    
    # Simulate file change event
    console.print("  📝 Simulating file change event...")
    file_change_event = {
        'type': 'file_changed',
        'file_path': '/path/to/changed/file.py'
    }
    
    await multimodal_cache.handle_change_event(file_change_event)
    console.print("  ✅ File change event processed")
    
    # Simulate workspace change event
    console.print("  🏢 Simulating workspace change event...")
    workspace_change_event = {
        'type': 'workspace_changed',
        'workspace_id': workspace_id,
        'change': None  # No specific change object
    }
    
    await multimodal_cache.handle_change_event(workspace_change_event)
    console.print("  ✅ Workspace change event processed")
    
    console.print()
    
    # Demo 7: Cache Optimization
    console.print("⚡ [bold yellow]Cache Optimization Demo[/bold yellow]")
    
    console.print("  🔧 Running cache optimization...")
    await multimodal_cache.optimize_cache()
    console.print("  ✅ Cache optimization completed")
    
    # Get updated statistics
    updated_stats = await multimodal_cache.get_cache_statistics()
    console.print("  📊 Updated cache statistics available")
    
    console.print()
    
    # Demo Summary
    console.print("📋 [bold green]Demo Summary[/bold green]")
    
    summary_panel = Panel(
        """
✅ Document Processing Cache: Intelligent caching with file change detection
✅ Code Analysis Cache: Fast retrieval of expensive analysis results  
✅ Workspace Cache: Collaborative data caching with CRDT support
✅ Cache Warming: Proactive loading of frequently accessed data
✅ Statistics Monitoring: Comprehensive metrics and performance tracking
✅ Change Event Handling: Smart invalidation based on file/workspace changes
✅ Cache Optimization: Automatic cleanup and memory management

The multi-modal cache system provides:
• 🚀 Significant performance improvements for repeated operations
• 🧠 Intelligent invalidation strategies to maintain data consistency
• 📊 Comprehensive monitoring and analytics
• 🔥 Proactive cache warming for optimal user experience
• ⚡ Automatic optimization and resource management
        """.strip(),
        title="🎉 Multi-Modal Cache System Features",
        border_style="green"
    )
    
    console.print(summary_panel)
    
    console.print("\n🎊 [bold cyan]Multi-Modal Cache System Demo Complete![/bold cyan]")
    console.print("The enhanced caching system provides intelligent, high-performance")
    console.print("caching for all types of multi-modal content with smart invalidation")
    console.print("and warming strategies for optimal user experience.")


if __name__ == "__main__":
    asyncio.run(demo_multimodal_cache())