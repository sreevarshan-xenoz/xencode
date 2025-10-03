#!/usr/bin/env python3
"""
Enhanced UI Components for Xencode Warp Terminal

Rich UI components for rendering different types of command output
with optimized layouts and interactive features.
"""

import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.layout import Layout
from rich.columns import Columns
from rich.align import Align
from rich.progress import Progress, BarColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich import box


class OutputRenderer:
    """Renders different types of command output with appropriate formatting"""
    
    def __init__(self):
        self.console = Console()
    
    def render_output(self, output_data: Dict[str, Any], metadata: Dict[str, Any]) -> Union[Text, Table, Tree, Panel]:
        """Render output based on its type"""
        output_type = output_data.get("type", "text")
        data = output_data.get("data", "")
        
        renderers = {
            "text": self._render_text,
            "json": self._render_json,
            "git_status": self._render_git_status,
            "git_log": self._render_git_log,
            "file_list": self._render_file_list,
            "process_list": self._render_process_list,
            "docker_images": self._render_docker_images,
            "npm_list": self._render_npm_list,
            "pip_list": self._render_pip_list,
            "error": self._render_error,
        }
        
        renderer = renderers.get(output_type, self._render_text)
        return renderer(data, metadata)
    
    def _render_text(self, data: Any, metadata: Dict[str, Any]) -> Text:
        """Render plain text output"""
        text = str(data)
        
        # Apply syntax highlighting for common patterns
        if self._looks_like_code(text):
            return self._render_code(text)
        elif self._looks_like_log(text):
            return self._render_log(text)
        else:
            return Text(text, style="white")
    
    def _render_json(self, data: Any, metadata: Dict[str, Any]) -> Panel:
        """Render JSON data with syntax highlighting"""
        try:
            if isinstance(data, str):
                json_data = json.loads(data)
            else:
                json_data = data
            
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            
            return Panel(
                syntax,
                title="[bold green]JSON Output[/bold green]",
                border_style="green"
            )
        except (json.JSONDecodeError, TypeError):
            return Text(str(data), style="red")
    
    def _render_git_status(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Panel:
        """Render git status with colored sections"""
        content = []
        
        # Branch info
        branch = data.get("branch", "unknown")
        content.append(Text(f"On branch {branch}", style="bold blue"))
        content.append(Text())  # Empty line
        
        # Staged files
        staged = data.get("staged", [])
        if staged:
            content.append(Text("Changes to be committed:", style="bold green"))
            for file in staged:
                content.append(Text(f"  new file:   {file}", style="green"))
            content.append(Text())
        
        # Modified files
        modified = data.get("modified", [])
        if modified:
            content.append(Text("Changes not staged for commit:", style="bold yellow"))
            for file in modified:
                content.append(Text(f"  modified:   {file}", style="yellow"))
            content.append(Text())
        
        # Untracked files
        untracked = data.get("untracked", [])
        if untracked:
            content.append(Text("Untracked files:", style="bold red"))
            for file in untracked:
                content.append(Text(f"  {file}", style="red"))
        
        if not staged and not modified and not untracked:
            content.append(Text("Working tree clean", style="green"))
        
        return Panel(
            Group(*content),
            title="[bold]Git Status[/bold]",
            border_style="blue"
        )
    
    def _render_git_log(self, data: List[str], metadata: Dict[str, Any]) -> Panel:
        """Render git log as a formatted list"""
        if not data:
            return Text("No commits found", style="dim")
        
        content = []
        for i, commit in enumerate(data):
            # Parse commit line (hash + message)
            parts = commit.split(' ', 1)
            if len(parts) == 2:
                hash_part, message = parts
                content.append(Text.assemble(
                    (f"{hash_part[:7]}", "yellow"),
                    (" ", ""),
                    (message, "white")
                ))
            else:
                content.append(Text(commit, style="white"))
        
        return Panel(
            Group(*content),
            title="[bold]Git Log[/bold]",
            border_style="yellow"
        )
    
    def _render_file_list(self, data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Table:
        """Render file listing as a table"""
        if not data:
            return Text("No files found", style="dim")
        
        table = Table(box=box.SIMPLE_HEAD)
        
        # Determine columns based on available data
        sample_file = data[0] if data else {}
        
        if "permissions" in sample_file:
            # Full ls -la format
            table.add_column("Permissions", style="cyan", width=11)
            table.add_column("Links", style="dim", width=5)
            table.add_column("Owner", style="blue", width=10)
            table.add_column("Group", style="blue", width=10)
            table.add_column("Size", style="green", width=8)
            table.add_column("Date", style="yellow", width=12)
            table.add_column("Name", style="white")
            
            for file_info in data:
                # Color-code file types
                name = file_info.get("name", "")
                name_style = self._get_file_style(name, file_info.get("permissions", ""))
                
                table.add_row(
                    file_info.get("permissions", ""),
                    file_info.get("links", ""),
                    file_info.get("owner", ""),
                    file_info.get("group", ""),
                    file_info.get("size", ""),
                    f"{file_info.get('month', '')} {file_info.get('day', '')} {file_info.get('time', '')}",
                    Text(name, style=name_style)
                )
        else:
            # Simple format
            table.add_column("Files", style="white")
            for file_info in data:
                name = file_info.get("name", "")
                name_style = self._get_file_style(name)
                table.add_row(Text(name, style=name_style))
        
        return table
    
    def _render_process_list(self, data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Table:
        """Render process list as a table"""
        if not data:
            return Text("No processes found", style="dim")
        
        table = Table(box=box.SIMPLE_HEAD, show_lines=False)
        
        # Get column names from first process
        if data:
            columns = list(data[0].keys())
            
            # Add columns with appropriate styling
            for col in columns:
                if col.upper() in ["PID", "PPID"]:
                    table.add_column(col.upper(), style="cyan", width=8)
                elif col.upper() in ["USER", "OWNER"]:
                    table.add_column(col.upper(), style="blue", width=10)
                elif col.upper() in ["CPU", "%CPU", "MEM", "%MEM"]:
                    table.add_column(col.upper(), style="yellow", width=6)
                elif col.upper() in ["STAT", "STATUS"]:
                    table.add_column(col.upper(), style="green", width=6)
                elif col.upper() in ["COMMAND", "CMD"]:
                    table.add_column(col.upper(), style="white")
                else:
                    table.add_column(col.upper(), style="dim", width=8)
            
            # Add rows
            for process in data[:20]:  # Limit to first 20 processes
                row = []
                for col in columns:
                    value = process.get(col, "")
                    
                    # Special formatting for certain columns
                    if col.upper() in ["CPU", "%CPU", "MEM", "%MEM"]:
                        try:
                            float_val = float(value)
                            if float_val > 50:
                                row.append(Text(value, style="red bold"))
                            elif float_val > 20:
                                row.append(Text(value, style="yellow"))
                            else:
                                row.append(Text(value, style="green"))
                        except ValueError:
                            row.append(Text(value, style="dim"))
                    elif col.upper() == "COMMAND":
                        # Truncate long commands
                        cmd = str(value)
                        if len(cmd) > 50:
                            cmd = cmd[:47] + "..."
                        row.append(Text(cmd, style="white"))
                    else:
                        row.append(str(value))
                
                table.add_row(*row)
        
        return table
    
    def _render_docker_images(self, data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Table:
        """Render Docker images as a table"""
        if not data:
            return Text("No Docker images found", style="dim")
        
        table = Table(box=box.SIMPLE_HEAD)
        table.add_column("Repository", style="cyan")
        table.add_column("Tag", style="yellow")
        table.add_column("Image ID", style="blue")
        table.add_column("Created", style="green")
        table.add_column("Size", style="magenta")
        
        for image in data:
            table.add_row(
                image.get("REPOSITORY", ""),
                image.get("TAG", ""),
                image.get("IMAGE ID", "")[:12] + "..." if len(image.get("IMAGE ID", "")) > 12 else image.get("IMAGE ID", ""),
                image.get("CREATED", ""),
                image.get("SIZE", "")
            )
        
        return table
    
    def _render_npm_list(self, data: Any, metadata: Dict[str, Any]) -> Union[Tree, Text]:
        """Render npm package list"""
        if isinstance(data, dict):
            # JSON format from npm list --json
            return self._render_npm_tree(data)
        else:
            # Plain text format
            return Text(str(data), style="white")
    
    def _render_npm_tree(self, data: Dict[str, Any]) -> Tree:
        """Render npm dependencies as a tree"""
        tree = Tree("📦 Dependencies")
        
        dependencies = data.get("dependencies", {})
        for name, info in dependencies.items():
            version = info.get("version", "unknown")
            node = tree.add(f"[cyan]{name}[/cyan] [dim]@{version}[/dim]")
            
            # Add nested dependencies
            nested_deps = info.get("dependencies", {})
            for nested_name, nested_info in list(nested_deps.items())[:5]:  # Limit depth
                nested_version = nested_info.get("version", "unknown")
                node.add(f"[yellow]{nested_name}[/yellow] [dim]@{nested_version}[/dim]")
        
        return tree
    
    def _render_pip_list(self, data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Table:
        """Render pip package list as a table"""
        if not data:
            return Text("No packages found", style="dim")
        
        table = Table(box=box.SIMPLE_HEAD)
        table.add_column("Package", style="cyan")
        table.add_column("Version", style="green")
        
        for package in data:
            table.add_row(
                package.get("name", ""),
                package.get("version", "")
            )
        
        return table
    
    def _render_error(self, data: Any, metadata: Dict[str, Any]) -> Panel:
        """Render error output"""
        error_text = str(data)
        
        return Panel(
            Text(error_text, style="red"),
            title="[bold red]Error[/bold red]",
            border_style="red"
        )
    
    def _render_code(self, text: str) -> Panel:
        """Render text that looks like code"""
        # Try to detect language
        language = "text"
        if "def " in text or "import " in text:
            language = "python"
        elif "function " in text or "const " in text:
            language = "javascript"
        elif "#include" in text or "int main" in text:
            language = "c"
        
        syntax = Syntax(text, language, theme="monokai", line_numbers=False)
        return Panel(syntax, border_style="blue")
    
    def _render_log(self, text: str) -> Text:
        """Render text that looks like log output"""
        lines = text.split('\n')
        result = Text()
        
        for line in lines:
            line_lower = line.lower()
            if "error" in line_lower or "fail" in line_lower:
                result.append(line + '\n', style="red")
            elif "warn" in line_lower:
                result.append(line + '\n', style="yellow")
            elif "info" in line_lower or "success" in line_lower:
                result.append(line + '\n', style="green")
            else:
                result.append(line + '\n', style="white")
        
        return result
    
    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like code"""
        code_indicators = [
            "def ", "function ", "class ", "import ", "from ",
            "#include", "int main", "public class", "<?php"
        ]
        return any(indicator in text for indicator in code_indicators)
    
    def _looks_like_log(self, text: str) -> bool:
        """Check if text looks like log output"""
        log_indicators = [
            "[ERROR]", "[WARN]", "[INFO]", "[DEBUG]",
            "ERROR:", "WARNING:", "INFO:", "FATAL:",
            "error:", "warning:", "info:"
        ]
        return any(indicator in text for indicator in log_indicators)
    
    def _get_file_style(self, filename: str, permissions: str = "") -> str:
        """Get appropriate style for file based on type"""
        if permissions.startswith('d'):
            return "blue bold"  # Directory
        elif permissions and permissions[3] == 'x':
            return "green bold"  # Executable
        elif filename.startswith('.'):
            return "dim"  # Hidden file
        elif any(filename.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c']):
            return "cyan"  # Code file
        elif any(filename.endswith(ext) for ext in ['.txt', '.md', '.rst', '.doc']):
            return "yellow"  # Document
        elif any(filename.endswith(ext) for ext in ['.jpg', '.png', '.gif', '.svg']):
            return "magenta"  # Image
        else:
            return "white"  # Default


class WarpLayoutManager:
    """Manages complex layouts for the Warp terminal"""
    
    def __init__(self):
        self.console = Console()
        self.output_renderer = OutputRenderer()
    
    def create_command_block_panel(self, block, expanded: bool = False) -> Panel:
        """Create a comprehensive panel for a command block"""
        # Command header
        command_text = Text(f"$ {block.command}", style="bold cyan")
        
        # Metadata
        exit_code = block.metadata.get('exit_code', '?')
        duration = block.metadata.get('duration_ms', '?')
        timestamp = block.timestamp.strftime("%H:%M:%S") if hasattr(block, 'timestamp') else "unknown"
        
        # Color-code exit code
        if exit_code == 0:
            exit_style = "green"
        elif exit_code == -1:
            exit_style = "yellow"
        else:
            exit_style = "red"
        
        metadata_text = Text.assemble(
            ("Exit: ", "dim"),
            (str(exit_code), exit_style),
            (" | Duration: ", "dim"),
            (f"{duration}ms", "yellow"),
            (" | Time: ", "dim"),
            (timestamp, "blue"),
            (" | Tags: ", "dim"),
            (", ".join(block.tags), "magenta")
        )
        
        # Render output
        if expanded:
            output_content = self.output_renderer.render_output(block.output_data, block.metadata)
        else:
            # Preview mode
            output_data = block.output_data.get("data", "")
            if isinstance(output_data, str) and len(output_data) > 200:
                preview = output_data[:200] + "..."
                output_content = Text(preview, style="white")
                output_content.append("\n[dim]... (press 'e' to expand)[/dim]", style="dim")
            else:
                output_content = self.output_renderer.render_output(block.output_data, block.metadata)
        
        # Combine all elements
        content = Group(
            command_text,
            Text(),  # Empty line
            output_content,
            Text(),  # Empty line
            metadata_text
        )
        
        # Panel styling based on exit code
        if exit_code == 0:
            border_style = "green"
            title_style = "bold green"
        elif exit_code == -1:
            border_style = "yellow"
            title_style = "bold yellow"
        else:
            border_style = "red"
            title_style = "bold red"
        
        return Panel(
            content,
            title=f"[{title_style}]Command Block[/{title_style}] [dim]{block.id}[/dim]",
            border_style=border_style,
            box=box.ROUNDED
        )
    
    def create_sidebar_panel(self, blocks: List, ai_suggestions: List[str] = None) -> Panel:
        """Create sidebar with recent commands and suggestions"""
        content = []
        
        # Recent commands
        if blocks:
            content.append(Text("Recent Commands", style="bold blue"))
            recent_tree = Tree("📝")
            
            for block in list(blocks)[-5:]:  # Last 5 commands
                exit_code = block.metadata.get('exit_code', '?')
                status_icon = "✅" if exit_code == 0 else "❌" if exit_code != '?' else "⏳"
                recent_tree.add(f"{status_icon} {block.command}")
            
            content.append(recent_tree)
            content.append(Text())  # Empty line
        
        # AI suggestions
        if ai_suggestions:
            content.append(Text("AI Suggestions", style="bold green"))
            suggestions_tree = Tree("🤖")
            
            for suggestion in ai_suggestions[:5]:  # Top 5 suggestions
                suggestions_tree.add(f"💡 {suggestion}")
            
            content.append(suggestions_tree)
            content.append(Text())  # Empty line
        
        # System info
        content.append(Text("System Info", style="bold yellow"))
        system_tree = Tree("💻")
        system_tree.add(f"Blocks: {len(blocks)}")
        system_tree.add(f"Session: {datetime.now().strftime('%H:%M:%S')}")
        
        content.append(system_tree)
        
        return Panel(
            Group(*content),
            title="[bold]Sidebar[/bold]",
            border_style="blue",
            box=box.ROUNDED
        )
    
    def create_full_layout(self, blocks: List, expanded_block_id: str = None, 
                          ai_suggestions: List[str] = None) -> Layout:
        """Create the full terminal layout"""
        layout = Layout()
        
        # Split into header, main, and footer
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Split main into terminal and sidebar
        layout["main"].split_row(
            Layout(name="terminal", ratio=3),
            Layout(name="sidebar", ratio=1)
        )
        
        # Header
        header_text = Text("Xencode Warp Terminal", style="bold blue")
        layout["header"].update(Panel(
            Align.center(header_text),
            box=box.ROUNDED,
            style="blue"
        ))
        
        # Terminal area with command blocks
        if blocks:
            terminal_blocks = []
            for block in list(blocks)[-3:]:  # Show last 3 blocks
                expanded = (expanded_block_id == block.id)
                panel = self.create_command_block_panel(block, expanded)
                terminal_blocks.append(panel)
            
            terminal_content = Group(*terminal_blocks)
        else:
            terminal_content = Panel(
                Align.center(Text("No commands yet\nType a command to get started!", style="dim")),
                box=box.ROUNDED
            )
        
        layout["terminal"].update(terminal_content)
        
        # Sidebar
        sidebar_panel = self.create_sidebar_panel(blocks, ai_suggestions)
        layout["sidebar"].update(sidebar_panel)
        
        # Footer
        footer_text = Text("Press 'p' for palette | 'e' to expand | Ctrl+C to exit", style="dim")
        layout["footer"].update(Panel(
            Align.center(footer_text),
            box=box.ROUNDED,
            style="dim"
        ))
        
        return layout


# Example usage
if __name__ == "__main__":
    from xencode.warp_terminal import WarpTerminal, example_ai_suggester
    
    # Create terminal and run some sample commands
    terminal = WarpTerminal(ai_suggester=example_ai_suggester)
    
    sample_commands = [
        "echo 'Hello World'",
        "ls -la",
        "git status",
        "ps aux | head -5"
    ]
    
    for cmd in sample_commands:
        terminal.run_command(cmd)
    
    # Test UI components
    layout_manager = WarpLayoutManager()
    
    console = Console()
    console.print("[bold blue]Testing Enhanced UI Components[/bold blue]")
    
    # Create and display layout
    layout = layout_manager.create_full_layout(
        list(terminal.command_blocks),
        ai_suggestions=["git add .", "git commit", "docker ps"]
    )
    
    console.print(layout)