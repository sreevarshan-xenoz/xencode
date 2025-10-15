#!/usr/bin/env python3
"""
File Operations Plugin Demo

Demonstrates the comprehensive file system operations plugin including:
- Directory listing and navigation
- File reading (text and binary)
- Content search and pattern matching
- Directory tree visualization
- Secure file mutations with RBAC
"""

import asyncio
import tempfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from xencode.plugins.file_operations import FileOperationsPlugin, PluginContext

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from rich import box

console = Console()


def create_demo_workspace():
    """Create a demo workspace with sample files"""
    temp_dir = Path(tempfile.mkdtemp(prefix="xencode_file_ops_"))
    
    # Create directory structure
    (temp_dir / "src").mkdir()
    (temp_dir / "tests").mkdir()
    (temp_dir / "docs").mkdir()
    (temp_dir / "config").mkdir()
    
    # Create Python files
    (temp_dir / "src" / "main.py").write_text("""#!/usr/bin/env python3
\"\"\"
Main application module
\"\"\"

import os
import sys
from typing import Dict, Any

def main() -> int:
    print("Hello, Xencode File Operations!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
""")
    
    (temp_dir / "src" / "utils.py").write_text("""#!/usr/bin/env python3
\"\"\"
Utility functions
\"\"\"

import json
from pathlib import Path

def load_config(config_path: Path) -> Dict[str, Any]:
    \"\"\"Load configuration from JSON file\"\"\"
    with open(config_path) as f:
        return json.load(f)

def save_config(config: Dict[str, Any], config_path: Path) -> None:
    \"\"\"Save configuration to JSON file\"\"\"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
""")
    
    # Create test files
    (temp_dir / "tests" / "test_main.py").write_text("""#!/usr/bin/env python3
\"\"\"
Tests for main module
\"\"\"

import pytest
from src.main import main

def test_main_returns_zero():
    \"\"\"Test that main function returns 0\"\"\"
    assert main() == 0

def test_main_prints_message(capsys):
    \"\"\"Test that main function prints expected message\"\"\"
    main()
    captured = capsys.readouterr()
    assert "Hello, Xencode" in captured.out
""")
    
    # Create config files
    (temp_dir / "config" / "settings.json").write_text("""{
  "app_name": "Xencode File Operations Demo",
  "version": "1.0.0",
  "debug": true,
  "features": {
    "file_operations": true,
    "search": true,
    "mutations": true
  }
}""")
    
    # Create documentation
    (temp_dir / "docs" / "README.md").write_text("""# Xencode File Operations Demo

This is a demonstration of the Xencode file operations plugin.

## Features

- Directory listing and navigation
- File reading (text and binary)
- Content search and pattern matching
- Directory tree visualization
- Secure file mutations with RBAC

## Usage

Run the demo script to see all features in action.
""")
    
    # Create a binary file
    (temp_dir / "data.bin").write_bytes(b"\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07")
    
    # Create a file with intentional lint errors for testing
    (temp_dir / "bad_code.py").write_text("""import os,sys
def bad_function( ):
    x=1+2
    print( "hello world" )
    return x
""")
    
    return temp_dir


async def demonstrate_directory_listing(plugin: FileOperationsPlugin, workspace: Path):
    """Demonstrate directory listing capabilities"""
    console.print("\\n[bold blue]📁 Directory Listing Demonstration[/bold blue]")
    
    # List root directory
    result = await plugin.ls_dir(workspace)
    
    table = Table(title="Workspace Contents", box=box.ROUNDED)
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Name", style="white", width=30)
    table.add_column("Count", style="green", width=8)
    
    table.add_row("📁 Directories", ", ".join(result["dirs"]), str(len(result["dirs"])))
    table.add_row("📄 Files", ", ".join(result["files"]), str(len(result["files"])))
    
    console.print(table)
    
    # List subdirectory
    src_result = await plugin.ls_dir(workspace / "src")
    console.print(f"\\n📂 Contents of 'src' directory: {', '.join(src_result['files'])}")


async def demonstrate_file_reading(plugin: FileOperationsPlugin, workspace: Path):
    """Demonstrate file reading capabilities"""
    console.print("\\n[bold blue]📖 File Reading Demonstration[/bold blue]")
    
    # Read text file
    main_py_content = await plugin.read_file(workspace / "src" / "main.py")
    console.print("\\n[yellow]📄 Contents of main.py (first 200 chars):[/yellow]")
    console.print(f"[dim]{main_py_content[:200]}...[/dim]")
    
    # Read JSON config
    config_content = await plugin.read_file(workspace / "config" / "settings.json")
    console.print("\\n[yellow]⚙️  Contents of settings.json:[/yellow]")
    console.print(f"[dim]{config_content}[/dim]")
    
    # Read binary file
    binary_content = await plugin.read_file(workspace / "data.bin", binary=True)
    console.print(f"\\n[yellow]🔢 Binary file content (8 bytes):[/yellow] {binary_content.hex()}")


async def demonstrate_search_capabilities(plugin: FileOperationsPlugin, workspace: Path):
    """Demonstrate search capabilities"""
    console.print("\\n[bold blue]🔍 Search Capabilities Demonstration[/bold blue]")
    
    # Search for Python files
    py_files = await plugin.search_pathnames_only("*.py", workspace)
    console.print(f"\\n🐍 Found {len(py_files)} Python files:")
    for py_file in py_files:
        console.print(f"  • {py_file.relative_to(workspace)}")
    
    # Search for files containing "import"
    console.print("\\n📦 Files containing 'import' keyword:")
    import_count = 0
    async for file_path, line in plugin.search_for_files("import", recursive=True, path=workspace):
        if import_count < 3:  # Limit output
            console.print(f"  • {file_path.relative_to(workspace)}: {line[:50]}...")
        import_count += 1
    
    console.print(f"   Total matches: {import_count}")
    
    # Search within specific file
    main_py_path = workspace / "src" / "main.py"
    def_matches = await plugin.search_in_file(main_py_path, "def ")
    console.print(f"\\n🔧 Function definitions in main.py: {len(def_matches)} found")
    for line_num, line in def_matches:
        console.print(f"  Line {line_num}: {line.strip()}")


async def demonstrate_tree_visualization(plugin: FileOperationsPlugin, workspace: Path):
    """Demonstrate directory tree visualization"""
    console.print("\\n[bold blue]🌳 Directory Tree Visualization[/bold blue]")
    
    tree_output = await plugin.get_dir_tree(workspace, depth=3)
    
    console.print("\\n[yellow]📊 Workspace Structure:[/yellow]")
    console.print(f"[dim]{tree_output}[/dim]")


async def demonstrate_lint_detection(plugin: FileOperationsPlugin, workspace: Path):
    """Demonstrate lint error detection"""
    console.print("\\n[bold blue]🔍 Lint Error Detection Demonstration[/bold blue]")
    
    try:
        bad_code_path = workspace / "bad_code.py"
        lint_errors = await plugin.read_lint_errors(bad_code_path)
        
        if lint_errors:
            table = Table(title="Lint Errors Found", box=box.ROUNDED)
            table.add_column("Line", style="red", width=6)
            table.add_column("Column", style="yellow", width=8)
            table.add_column("Code", style="cyan", width=8)
            table.add_column("Message", style="white", width=50)
            
            for error in lint_errors[:5]:  # Show first 5 errors
                table.add_row(
                    str(error.get("line", "?")),
                    str(error.get("column", "?")),
                    error.get("code", ""),
                    error.get("msg", "")[:50]
                )
            
            console.print(table)
        else:
            console.print("✅ No lint errors found (or linter not available)")
            
    except Exception as e:
        console.print(f"⚠️  Lint detection failed: {e}")


async def demonstrate_file_mutations(plugin: FileOperationsPlugin, workspace: Path):
    """Demonstrate secure file mutation capabilities"""
    console.print("\\n[bold blue]✏️  File Mutation Demonstration[/bold blue]")
    
    try:
        # Create a new file
        new_file_path = workspace / "demo_created.txt"
        await plugin.create_file_or_folder(
            new_file_path, 
            content="This file was created by the Xencode file operations plugin!\\n"
        )
        console.print(f"✅ Created new file: {new_file_path.name}")
        
        # Create a new directory
        new_dir_path = workspace / "demo_directory"
        await plugin.create_file_or_folder(new_dir_path, is_directory=True)
        console.print(f"✅ Created new directory: {new_dir_path.name}")
        
        # Edit an existing file
        edit_file_path = workspace / "demo_created.txt"
        await plugin.edit_file(
            edit_file_path,
            search="created by",
            replace="modified by"
        )
        console.print(f"✅ Edited file: {edit_file_path.name}")
        
        # Read the modified content
        modified_content = await plugin.read_file(edit_file_path)
        console.print(f"📄 Modified content: {modified_content.strip()}")
        
        # Clean up - delete the created files
        await plugin.delete_file_or_folder(new_file_path)
        await plugin.delete_file_or_folder(new_dir_path)
        console.print("🧹 Cleaned up demo files")
        
    except Exception as e:
        console.print(f"❌ File mutation failed: {e}")


async def demonstrate_rbac_security(workspace: Path):
    """Demonstrate RBAC security features"""
    console.print("\\n[bold blue]🔒 RBAC Security Demonstration[/bold blue]")
    
    # Create plugin with limited permissions
    limited_context = PluginContext()
    limited_context.workspace_root = workspace
    limited_context.permissions = ["file:read"]  # Only read permission
    
    limited_plugin = FileOperationsPlugin(limited_context)
    
    try:
        # This should work (read permission)
        await limited_plugin.ls_dir(workspace)
        console.print("✅ Directory listing allowed with read permission")
        
        # This should fail (no write permission)
        try:
            await limited_plugin.create_file_or_folder(
                workspace / "forbidden.txt",
                content="This should not work"
            )
            console.print("❌ Security breach: file creation should have been blocked!")
        except PermissionError as e:
            console.print(f"✅ Security working: {e}")
            
    except Exception as e:
        console.print(f"❌ RBAC test failed: {e}")


async def show_performance_metrics(plugin: FileOperationsPlugin):
    """Show performance metrics and caching"""
    console.print("\\n[bold blue]⚡ Performance Metrics[/bold blue]")
    
    # The plugin has internal caching - demonstrate cache hits
    workspace = plugin.context.workspace_root
    
    import time
    
    # First call (cache miss)
    start_time = time.time()
    await plugin.ls_dir(workspace)
    first_call_time = time.time() - start_time
    
    # Second call (cache hit)
    start_time = time.time()
    await plugin.ls_dir(workspace)
    second_call_time = time.time() - start_time
    
    table = Table(title="Performance Metrics", box=box.ROUNDED)
    table.add_column("Operation", style="cyan")
    table.add_column("Time (ms)", style="green")
    table.add_column("Status", style="yellow")
    
    table.add_row("First ls_dir call", f"{first_call_time*1000:.2f}", "Cache miss")
    table.add_row("Second ls_dir call", f"{second_call_time*1000:.2f}", "Cache hit")
    table.add_row("Speed improvement", f"{(first_call_time/second_call_time):.1f}x", "Faster")
    
    console.print(table)


async def main():
    """Main demo function"""
    console.print(Panel.fit(
        "[bold cyan]Xencode File Operations Plugin Demo[/bold cyan]\\n"
        "Comprehensive file system operations with security and performance",
        border_style="blue"
    ))
    
    # Create demo workspace
    console.print("\\n[yellow]🏗️  Setting up demo workspace...[/yellow]")
    workspace = create_demo_workspace()
    console.print(f"📁 Demo workspace created at: {workspace}")
    
    # Create plugin context with full permissions
    context = PluginContext()
    context.workspace_root = workspace
    context.user_id = "demo_user"
    context.workspace_id = "demo_workspace"
    context.permissions = [
        "file:read", "file:write", "file:search", 
        "file:view", "file:lint", "file:delete"
    ]
    
    # Initialize plugin
    plugin = FileOperationsPlugin(context)
    
    try:
        # Run demonstrations
        await demonstrate_directory_listing(plugin, workspace)
        await demonstrate_file_reading(plugin, workspace)
        await demonstrate_search_capabilities(plugin, workspace)
        await demonstrate_tree_visualization(plugin, workspace)
        await demonstrate_lint_detection(plugin, workspace)
        await demonstrate_file_mutations(plugin, workspace)
        await demonstrate_rbac_security(workspace)
        await show_performance_metrics(plugin)
        
        console.print("\\n[bold green]✅ File Operations Plugin Demo Completed Successfully![/bold green]")
        
    except Exception as e:
        console.print(f"\\n[red]❌ Demo failed: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(workspace)
            console.print(f"\\n[dim]🧹 Cleaned up demo workspace: {workspace}[/dim]")
        except Exception as e:
            console.print(f"\\n[dim]⚠️  Failed to cleanup workspace: {e}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())