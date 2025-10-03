#!/usr/bin/env python3
"""
Xencode Warp-Style Terminal: Core Implementation

A Warp-like terminal experience with structured command blocks, AI suggestions,
and performance optimizations for large outputs.
"""

import asyncio
import json
import time
import threading
import subprocess
from typing import Dict, List, Optional, Any, Callable, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import deque
import uuid
import logging

from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# Try to import wgpu for GPU acceleration (optional)
try:
    import wgpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CommandBlock:
    """Structured representation of a command and its output"""
    id: str
    command: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommandBlock':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class StreamingOutputParser:
    """Parse command output in chunks to handle large outputs efficiently"""
    
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size
        self.parsers = {
            "git": self._parse_git_output,
            "ls": self._parse_ls_output,
            "ps": self._parse_ps_output,
            "docker": self._parse_docker_output,
            "npm": self._parse_npm_output,
            "pip": self._parse_pip_output,
        }
    
    def parse_streaming(self, command: str, output_stream: Iterator[str]) -> Iterator[Dict[str, Any]]:
        """Parse output in chunks for large outputs"""
        buffer = ""
        output_type = self._detect_output_type(command)
        
        for chunk in output_stream:
            buffer += chunk
            
            # Process buffer in chunks
            while len(buffer) >= self.chunk_size:
                process_chunk = buffer[:self.chunk_size]
                buffer = buffer[self.chunk_size:]
                
                # Parse the chunk based on output type
                if output_type == "json":
                    # Try to parse complete JSON objects
                    try:
                        json_data = json.loads(process_chunk)
                        yield {"type": "json", "data": json_data, "partial": False}
                    except json.JSONDecodeError:
                        # If not complete JSON, yield as partial
                        yield {"type": "text", "data": process_chunk, "partial": True}
                else:
                    # For other types, just yield the chunk
                    yield {"type": output_type, "data": process_chunk, "partial": True}
        
        # Process remaining buffer
        if buffer:
            if output_type == "json":
                try:
                    json_data = json.loads(buffer)
                    yield {"type": "json", "data": json_data, "partial": False}
                except json.JSONDecodeError:
                    yield {"type": "text", "data": buffer, "partial": True}
            else:
                yield {"type": output_type, "data": buffer, "partial": False}
    
    def parse_output(self, command: str, output: str, exit_code: int) -> Dict[str, Any]:
        """Parse command output into structured data"""
        # Determine command type
        cmd_parts = command.split()
        cmd_type = cmd_parts[0] if cmd_parts else "unknown"
        
        # Try to use a specific parser
        if cmd_type in self.parsers:
            return self.parsers[cmd_type](command, output, exit_code)
        
        # Try to detect JSON output
        try:
            json_data = json.loads(output)
            return {"type": "json", "data": json_data}
        except json.JSONDecodeError:
            pass
        
        # Default: return as text
        return {"type": "text", "data": output}
    
    def _detect_output_type(self, command: str) -> str:
        """Detect the output type based on command"""
        cmd_parts = command.split()
        if cmd_parts:
            cmd_type = cmd_parts[0]
            if cmd_type in self.parsers:
                return cmd_type
        return "text"
    
    def _parse_git_output(self, command: str, output: str, exit_code: int) -> Dict[str, Any]:
        """Parse git command output"""
        if "status" in command:
            # Parse git status output
            lines = output.strip().split('\n')
            modified = []
            untracked = []
            staged = []
            
            for line in lines:
                if line.startswith('M '):
                    modified.append(line[2:].strip())
                elif line.startswith('?? '):
                    untracked.append(line[3:].strip())
                elif line.startswith('A '):
                    staged.append(line[2:].strip())
            
            return {
                "type": "git_status",
                "data": {
                    "modified": modified,
                    "untracked": untracked,
                    "staged": staged,
                    "branch": self._extract_git_branch(output)
                }
            }
        elif "log" in command:
            # Parse git log output
            lines = output.strip().split('\n')
            commits = []
            for line in lines:
                if line:
                    commits.append(line.strip())
            return {"type": "git_log", "data": commits}
        
        # Default for other git commands
        return {"type": "git", "data": output}
    
    def _parse_ls_output(self, command: str, output: str, exit_code: int) -> Dict[str, Any]:
        """Parse ls command output"""
        lines = output.strip().split('\n')
        files = []
        
        for line in lines:
            if not line.strip():
                continue
                
            parts = line.split()
            if len(parts) >= 9:  # Full ls -la format
                files.append({
                    "permissions": parts[0],
                    "links": parts[1],
                    "owner": parts[2],
                    "group": parts[3],
                    "size": parts[4],
                    "month": parts[5],
                    "day": parts[6],
                    "time": parts[7],
                    "name": " ".join(parts[8:])
                })
            else:  # Simple ls format
                files.append({"name": line.strip()})
        
        return {"type": "file_list", "data": files}
    
    def _parse_ps_output(self, command: str, output: str, exit_code: int) -> Dict[str, Any]:
        """Parse ps command output"""
        lines = output.strip().split('\n')
        if not lines:
            return {"type": "process_list", "data": []}
        
        # Assume first line is header
        headers = lines[0].split()
        processes = []
        
        for line in lines[1:]:
            parts = line.split(None, len(headers)-1)
            if len(parts) == len(headers):
                process = {headers[i]: parts[i] for i in range(len(headers))}
                processes.append(process)
        
        return {"type": "process_list", "data": processes}
    
    def _parse_docker_output(self, command: str, output: str, exit_code: int) -> Dict[str, Any]:
        """Parse docker command output"""
        if "ps" in command:
            return self._parse_ps_output(command, output, exit_code)
        elif "images" in command:
            lines = output.strip().split('\n')
            if not lines:
                return {"type": "docker_images", "data": []}
            
            headers = lines[0].split()
            images = []
            for line in lines[1:]:
                parts = line.split(None, len(headers)-1)
                if len(parts) == len(headers):
                    image = {headers[i]: parts[i] for i in range(len(headers))}
                    images.append(image)
            
            return {"type": "docker_images", "data": images}
        
        return {"type": "docker", "data": output}
    
    def _parse_npm_output(self, command: str, output: str, exit_code: int) -> Dict[str, Any]:
        """Parse npm command output"""
        if "list" in command:
            try:
                # Try to parse as JSON first
                json_data = json.loads(output)
                return {"type": "npm_list", "data": json_data}
            except json.JSONDecodeError:
                pass
        
        return {"type": "npm", "data": output}
    
    def _parse_pip_output(self, command: str, output: str, exit_code: int) -> Dict[str, Any]:
        """Parse pip command output"""
        if "list" in command:
            lines = output.strip().split('\n')
            packages = []
            for line in lines[2:]:  # Skip header lines
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        packages.append({"name": parts[0], "version": parts[1]})
            
            return {"type": "pip_list", "data": packages}
        
        return {"type": "pip", "data": output}
    
    def _extract_git_branch(self, output: str) -> str:
        """Extract branch name from git status output"""
        for line in output.split('\n'):
            if line.startswith("On branch "):
                return line[11:]
        return "unknown"


class LazyCommandBlock:
    """Command block with lazy rendering for large outputs"""
    
    def __init__(self, id: str, command: str, output_data: Dict[str, Any], 
                 metadata: Dict[str, Any], tags: List[str]):
        self.id = id
        self.command = command
        self.output_data = output_data
        self.metadata = metadata
        self.tags = tags
        self.timestamp = time.time()
        self._rendered_cache = None
        self._is_expanded = False
    
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Rich console rendering with lazy evaluation"""
        if self._rendered_cache is None or self._is_expanded:
            self._rendered_cache = self._create_panel()
        yield self._rendered_cache
    
    def _create_panel(self) -> Panel:
        """Create the panel for this command block"""
        # Command
        command_text = Text(f"$ {self.command}", style="bold cyan")
        
        # Output based on type and expansion state
        if self._is_expanded:
            output_text = self._format_full_output()
        else:
            output_text = self._format_preview_output()
        
        # Metadata
        metadata_text = Text(
            f"Exit: {self.metadata.get('exit_code', '?')} | "
            f"Duration: {self.metadata.get('duration_ms', '?')}ms | "
            f"Tags: {', '.join(self.tags)}",
            style="dim"
        )
        
        # Combine all elements
        content = f"{command_text}\n\n{output_text}\n\n{metadata_text}"
        
        return Panel(
            content,
            title=f"[bold]Command[/bold] [dim]{self.id}[/dim]",
            box=box.ROUNDED
        )
    
    def _format_preview_output(self) -> Text:
        """Format a preview of the output"""
        output_type = self.output_data.get("type", "text")
        output_data = self.output_data.get("data", "")
        
        if output_type == "json":
            # For JSON, show a preview of the structure
            if isinstance(output_data, dict):
                keys = list(output_data.keys())[:5]
                preview = f"{{ {', '.join(f'{k}: ...' for k in keys)} }}"
                if len(output_data) > 5:
                    preview += f" (+{len(output_data)-5} more)"
                return Text(preview, style="green")
            elif isinstance(output_data, list):
                preview = f"[{len(output_data)} items]"
                return Text(preview, style="green")
        
        # For other types, show first few lines
        if isinstance(output_data, str):
            lines = output_data.split('\n')
            preview = '\n'.join(lines[:3])
            if len(lines) > 3:
                preview += f"\n... (+{len(lines)-3} more lines)"
            return Text(preview, style="white")
        
        return Text(
            str(output_data)[:100] + "..." if len(str(output_data)) > 100 else str(output_data), 
            style="white"
        )
    
    def _format_full_output(self) -> Text:
        """Format the full output"""
        output_type = self.output_data.get("type", "text")
        output_data = self.output_data.get("data", "")
        
        if output_type == "json":
            return Text(json.dumps(output_data, indent=2), style="white")
        else:
            return Text(str(output_data), style="white")
    
    def toggle_expansion(self):
        """Toggle between preview and full output"""
        self._is_expanded = not self._is_expanded
        self._rendered_cache = None  # Invalidate cache


class GPUAcceleratedRenderer:
    """GPU-accelerated terminal rendering (simplified for POC)"""
    
    def __init__(self):
        self.console = Console()
        self.use_gpu = GPU_AVAILABLE
        if self.use_gpu:
            self._init_gpu()
    
    def _init_gpu(self):
        """Initialize GPU context (simplified for POC)"""
        # In a full implementation, this would set up wgpu context
        # For POC, we'll just flag that GPU is available
        logger.info("GPU acceleration available")
    
    def render_block(self, block: CommandBlock) -> Panel:
        """Render a command block with GPU acceleration if available"""
        # Create a rich panel with the command and output
        command_text = Text(f"$ {block.command}", style="bold cyan")
        
        # Format output based on type
        if block.output_data.get("type") == "json":
            output_text = Text(json.dumps(block.output_data.get("data", {}), indent=2))
        else:
            output_text = Text(str(block.output_data.get("data", "")))
        
        # Add metadata
        metadata_text = Text(
            f"Exit: {block.metadata.get('exit_code', '?')} | "
            f"Duration: {block.metadata.get('duration_ms', '?')}ms | "
            f"Tags: {', '.join(block.tags)}",
            style="dim"
        )
        
        content = f"{command_text}\n\n{output_text}\n\n{metadata_text}"
        
        return Panel(
            content, 
            title=f"[bold]Command Block[/bold] [dim]{block.id}[/dim]"
        )
    
    def render_live(self, blocks: List[CommandBlock]) -> Layout:
        """Render multiple blocks in a live layout"""
        layout = Layout()
        
        # Create a table of blocks
        table = Table(box=box.ROUNDED)
        table.add_column("Command", style="cyan")
        table.add_column("Output", style="white")
        table.add_column("Metadata", style="dim")
        
        for block in blocks[-5:]:  # Show last 5 blocks
            output_preview = str(block.output_data.get("data", ""))[:50] + "..." \
                if len(str(block.output_data.get("data", ""))) > 50 \
                else str(block.output_data.get("data", ""))
            
            metadata = f"Exit: {block.metadata.get('exit_code', '?')} | " \
                      f"{block.metadata.get('duration_ms', '?')}ms"
            
            table.add_row(
                block.command,
                output_preview,
                metadata
            )
        
        layout.update(Panel(table, title="[bold]Command History[/bold]"))
        return layout


class WarpTerminal:
    """Main Warp-like terminal implementation"""
    
    def __init__(self, ai_suggester: Optional[Callable] = None, max_blocks: int = 20):
        self.command_blocks: deque = deque(maxlen=max_blocks)  # Limit memory usage
        self.renderer = GPUAcceleratedRenderer()
        self.output_parser = StreamingOutputParser()
        self.ai_suggester = ai_suggester
        self.console = Console()
        self.running = False
        
        # AI suggestions caching
        self._ai_suggestions_cache = None
        self._ai_suggestions_cache_time = 0
        self._ai_suggestions_cache_ttl = 30  # Cache for 30 seconds
    
    def run_command(self, command: str) -> CommandBlock:
        """Execute a command and create a structured block"""
        start_time = time.time()
        
        # Execute the command
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30  # Add timeout to prevent hanging
            )
            exit_code = result.returncode
            output = result.stdout
            error = result.stderr
        except subprocess.TimeoutExpired:
            exit_code = -1
            output = ""
            error = "Command timed out after 30 seconds"
        except Exception as e:
            exit_code = -1
            output = ""
            error = str(e)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Parse the output
        parsed_output = self.output_parser.parse_output(command, output, exit_code)
        
        # Create tags based on command and output
        tags = self._generate_tags(command, parsed_output)
        
        # Create metadata
        metadata = {
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "error": error if error else None
        }
        
        # Create and store the command block
        block = CommandBlock(
            id=f"cmd_{len(self.command_blocks)+1}",
            command=command,
            input_data={},
            output_data=parsed_output,
            metadata=metadata,
            timestamp=datetime.now(),
            tags=tags
        )
        
        self.command_blocks.append(block)
        return block
    
    def run_command_streaming(self, command: str) -> LazyCommandBlock:
        """Execute a command with streaming output for large results"""
        start_time = time.time()
        
        # Create a block immediately with empty output
        block = LazyCommandBlock(
            id=f"cmd_{len(self.command_blocks)+1}",
            command=command,
            output_data={"type": "text", "data": "", "partial": True},
            metadata={"exit_code": None, "duration_ms": None},
            tags=[command.split()[0] if command.split() else "unknown"]
        )
        
        # Add to blocks immediately
        self.command_blocks.append(block)
        
        # Execute command in a separate thread to avoid blocking
        def execute_command():
            try:
                # Start the process
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1  # Line buffered
                )
                
                # Stream output
                output_lines = []
                output_type = self.output_parser._detect_output_type(command)
                
                # Process stdout line by line
                for line in process.stdout:
                    output_lines.append(line)
                    
                    # Update block with partial output
                    if output_type == "json":
                        # For JSON, wait until we have complete output
                        continue
                    else:
                        # For other types, update with new content
                        block.output_data = {
                            "type": output_type,
                            "data": "".join(output_lines),
                            "partial": True
                        }
                
                # Wait for process to complete
                process.wait()
                exit_code = process.returncode
                
                # Get any error output
                error_output = process.stderr.read()
                
                # Finalize output
                duration_ms = int((time.time() - start_time) * 1000)
                
                if output_type == "json":
                    # Parse JSON output
                    full_output = "".join(output_lines)
                    try:
                        json_data = json.loads(full_output)
                        block.output_data = {"type": "json", "data": json_data, "partial": False}
                    except json.JSONDecodeError:
                        block.output_data = {"type": "text", "data": full_output, "partial": False}
                else:
                    block.output_data = {
                        "type": output_type,
                        "data": "".join(output_lines),
                        "partial": False
                    }
                
                # Update metadata
                block.metadata = {
                    "exit_code": exit_code,
                    "duration_ms": duration_ms,
                    "error": error_output if error_output else None
                }
                
                # Update tags
                block.tags = self._generate_tags(command, block.output_data)
                
            except Exception as e:
                # Handle errors
                duration_ms = int((time.time() - start_time) * 1000)
                block.output_data = {"type": "error", "data": str(e), "partial": False}
                block.metadata = {
                    "exit_code": -1,
                    "duration_ms": duration_ms,
                    "error": str(e)
                }
                block.tags = ["error"]
        
        # Start execution in background
        thread = threading.Thread(target=execute_command)
        thread.daemon = True
        thread.start()
        
        return block
    
    def get_ai_suggestions_async(self) -> List[str]:
        """Get AI suggestions asynchronously with advanced caching and context awareness"""
        current_time = time.time()
        
        # Check if we have cached suggestions that are still valid
        if (self._ai_suggestions_cache is not None and 
            current_time - self._ai_suggestions_cache_time < self._ai_suggestions_cache_ttl):
            return self._ai_suggestions_cache
        
        # Start background task to get suggestions
        def get_suggestions():
            try:
                # Get context from recent commands
                recent_commands = [block.command for block in list(self.command_blocks)[-10:]]
                
                # Try advanced AI integration first
                try:
                    from .warp_ai_integration import get_warp_ai_integration
                    from pathlib import Path
                    import asyncio
                    
                    ai_integration = get_warp_ai_integration()
                    
                    # Run async AI suggestions in a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    suggestions = loop.run_until_complete(
                        ai_integration.get_smart_suggestions(recent_commands, Path.cwd())
                    )
                    
                    loop.close()
                    
                    self._ai_suggestions_cache = suggestions
                    self._ai_suggestions_cache_time = time.time()
                    
                except ImportError:
                    # Fallback to simple AI suggester
                    if self.ai_suggester:
                        suggestions = self.ai_suggester(recent_commands)
                        self._ai_suggestions_cache = suggestions
                        self._ai_suggestions_cache_time = time.time()
                    
            except Exception as e:
                logger.warning(f"AI suggestions failed: {e}")
                # Fallback to simple suggestions
                if self.ai_suggester:
                    try:
                        recent_commands = [block.command for block in list(self.command_blocks)[-5:]]
                        suggestions = self.ai_suggester(recent_commands)
                        self._ai_suggestions_cache = suggestions
                        self._ai_suggestions_cache_time = time.time()
                    except Exception:
                        self._ai_suggestions_cache = []
                else:
                    self._ai_suggestions_cache = []
        
        # Start in background thread
        thread = threading.Thread(target=get_suggestions)
        thread.daemon = True
        thread.start()
        
        # Return cached suggestions or empty list
        return self._ai_suggestions_cache if self._ai_suggestions_cache is not None else []
    
    def _generate_tags(self, command: str, output_data: Dict[str, Any]) -> List[str]:
        """Generate tags for a command block"""
        tags = []
        
        # Add command type as a tag
        cmd_parts = command.split()
        if cmd_parts:
            tags.append(cmd_parts[0])
        
        # Add output type as a tag
        output_type = output_data.get("type", "unknown")
        tags.append(output_type)
        
        # Add specific tags based on content
        if output_type == "git_status":
            if output_data.get("data", {}).get("modified"):
                tags.append("has_changes")
            if output_data.get("data", {}).get("untracked"):
                tags.append("has_untracked")
        
        return tags
    
    def start_interactive_session(self):
        """Start an interactive terminal session"""
        self.running = True
        self.console.print("[bold green]Xencode Warp Terminal Started[/bold green]")
        self.console.print("Press Ctrl+C to exit, or type 'palette' to open command palette")
        
        with Live(self.renderer.render_live(list(self.command_blocks)), refresh_per_second=4) as live:
            while self.running:
                try:
                    # Get user input
                    command = Prompt.ask("[bold cyan]$[/bold cyan]")
                    
                    if command.lower() == "exit":
                        self.running = False
                    elif command.lower() == "palette":
                        # Show command palette (simplified for now)
                        self._show_simple_palette()
                    elif command.strip():
                        # Execute the command
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            transient=True,
                        ) as progress:
                            task = progress.add_task(f"Running: {command}...", total=None)
                            block = self.run_command(command)
                        
                        live.update(self.renderer.render_live(list(self.command_blocks)))
                        
                except KeyboardInterrupt:
                    self.running = False
                except EOFError:
                    self.running = False
        
        self.console.print("[bold green]Session ended[/bold green]")
    
    def _show_simple_palette(self):
        """Show a simple command palette (will be enhanced in Week 2)"""
        # Get recent commands for suggestions
        recent_commands = [block.command for block in list(self.command_blocks)[-10:]]
        
        # Add AI suggestions if available
        ai_suggestions = self.get_ai_suggestions_async()
        if ai_suggestions:
            recent_commands.extend(ai_suggestions)
        
        # Show command palette
        self.console.print("[bold]Command Palette[/bold]")
        for i, cmd in enumerate(recent_commands):
            self.console.print(f"{i+1}. {cmd}")
        
        # Get user selection
        choice = Prompt.ask("Select command (number) or type new command")
        
        try:
            # Try to parse as a number
            idx = int(choice) - 1
            if 0 <= idx < len(recent_commands):
                selected_command = recent_commands[idx]
                # Execute the selected command
                block = self.run_command(selected_command)
        except ValueError:
            # Not a number, execute as is
            if choice.strip():
                block = self.run_command(choice)


# Example AI suggester function
def example_ai_suggester(recent_commands: List[str]) -> List[str]:
    """Example AI suggester function"""
    suggestions = []
    
    # Simple rule-based suggestions based on recent commands
    if any("git" in cmd for cmd in recent_commands):
        suggestions.extend(["git status", "git add .", "git commit -m 'Update'", "git push"])
    
    if any("ls" in cmd for cmd in recent_commands):
        suggestions.extend(["ls -la", "ls -lh", "tree"])
    
    if any("docker" in cmd for cmd in recent_commands):
        suggestions.extend(["docker ps", "docker images", "docker logs"])
    
    if any("npm" in cmd for cmd in recent_commands):
        suggestions.extend(["npm install", "npm run build", "npm test"])
    
    # Remove duplicates and limit to 5 suggestions
    return list(dict.fromkeys(suggestions))[:5]


# Main entry point
if __name__ == "__main__":
    terminal = WarpTerminal(ai_suggester=example_ai_suggester)
    terminal.start_interactive_session()