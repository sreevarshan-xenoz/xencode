#!/usr/bin/env python3
"""
Xencode Warp Terminal - Main Integration

Brings together all components: robust terminal, session management,
logging, and debugging for a complete Warp-like experience.
"""

import asyncio
import signal
import sys
import threading
import time
from typing import Optional, Callable, List
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from .warp_terminal import WarpTerminal
from .warp_ui_components import WarpLayoutManager
from .enhanced_command_palette import WarpTerminalWithPalette
from .system.robustness_manager import RobustWarpTerminal
from .system.session_manager import get_session_manager, cleanup_on_exit
from .system.debugging_manager import get_debugging_manager, debug_function, trace_function


class IntegratedWarpTerminal:
    """Fully integrated Warp terminal with all enhanced features"""
    
    def __init__(self, ai_suggester: Optional[Callable] = None):
        # Use the robust terminal as the base
        self.terminal = RobustWarpTerminal(ai_suggester=ai_suggester)
        
        # UI components
        self.layout_manager = WarpLayoutManager()
        self.console = Console()
        
        # Enhanced palette
        self.palette_terminal = WarpTerminalWithPalette(self.terminal)
        
        # Managers
        self.session_manager = get_session_manager()
        self.debugging_manager = get_debugging_manager()
        
        # Runtime state
        self.running = False
        self.expanded_block_id = None
        
        # Register cleanup
        import atexit
        atexit.register(cleanup_on_exit)
    
    @trace_function
    @debug_function
    def start_interactive_session(self):
        """Start the full-featured interactive session"""
        self.running = True
        self.console.print("[bold green]Xencode Warp Terminal - Enhanced Edition[/bold green]")
        self.console.print("Commands: Ctrl+C to exit | Ctrl+P for palette | 'help' for commands")
        
        # Start resource monitoring
        self.debugging_manager.start_resource_monitor("warp_terminal", interval=10.0)
        self.debugging_manager.start_performance_monitor("warp_terminal", interval=5.0)
        
        # Create initial layout
        layout = self.layout_manager.create_full_layout(
            list(self.terminal.command_blocks),
            expanded_block_id=self.expanded_block_id,
            ai_suggestions=self.terminal.get_ai_suggestions_async()
        )
        
        with Live(layout, refresh_per_second=4, screen=True) as live:
            while self.running:
                try:
                    # Get user input
                    command = Prompt.ask("[bold cyan]$[/bold cyan]", default="")
                    
                    if command.lower() in ["exit", "quit"]:
                        self.running = False
                    elif command.lower() in ["palette", "p"]:
                        # Show enhanced command palette
                        selected_command = self.palette_terminal.show_enhanced_palette()
                        if selected_command:
                            block = self.terminal.run_command(selected_command)
                    elif command.lower() == "help":
                        self._show_help()
                    elif command.lower() == "expand":
                        self._toggle_expand_last_block()
                    elif command.lower() == "sessions":
                        self._show_sessions()
                    elif command.lower() == "diagnostics":
                        self._run_diagnostics()
                    elif command.strip():
                        # Execute the command with robust handling
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            transient=True,
                        ) as progress:
                            task = progress.add_task(f"Running: {command}...", total=None)
                            
                            # Use the robust command execution
                            block = self.terminal.run_command(command)
                        
                        # Update the live display
                        layout = self.layout_manager.create_full_layout(
                            list(self.terminal.command_blocks),
                            expanded_block_id=self.expanded_block_id,
                            ai_suggestions=self.terminal.get_ai_suggestions_async()
                        )
                        live.update(layout)
                    
                    # Periodic health check
                    if time.time() - self.terminal.last_health_check > 30:
                        health = self.terminal.health_check()
                        if not health["healthy"]:
                            self.console.print(f"[yellow]Health warning: {health}[/yellow]")
                
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Interrupted. Press Ctrl+C again to force exit.[/yellow]")
                    try:
                        # Give a chance to exit cleanly
                        time.sleep(0.5)
                    except KeyboardInterrupt:
                        self.running = False
                except EOFError:
                    self.running = False
        
        # Cleanup
        self.terminal.shutdown()
        self.debugging_manager.stop_resource_monitor("warp_terminal")
        self.debugging_manager.stop_performance_monitor("warp_terminal")
        
        self.console.print("[bold green]Session ended[/bold green]")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
[b]Xencode Warp Terminal Commands:[/b]
- [cyan]Ctrl+P[/cyan]: Open enhanced command palette
- [cyan]expand[/cyan]: Toggle expansion of last command block
- [cyan]sessions[/cyan]: Show recent sessions
- [cyan]diagnostics[/cyan]: Run system diagnostics
- [cyan]help[/cyan]: Show this help
- [cyan]exit/quit[/cyan]: Exit the terminal

[b]Keyboard Shortcuts:[/b]
- [cyan]↑/↓[/cyan]: Navigate command history
- [cyan]Tab[/cyan]: Auto-complete from suggestions
- [cyan]Ctrl+C[/cyan]: Interrupt or exit
        """
        self.console.print(help_text)
    
    def _toggle_expand_last_block(self):
        """Toggle expansion of the last command block"""
        if self.terminal.command_blocks:
            last_block = self.terminal.command_blocks[-1]
            if self.expanded_block_id == last_block.id:
                self.expanded_block_id = None
            else:
                self.expanded_block_id = last_block.id
            self.console.print(f"Toggled expansion for block {last_block.id}")
    
    def _show_sessions(self):
        """Show recent sessions"""
        recent_sessions = self.session_manager.get_recent_sessions(limit=5)
        
        if not recent_sessions:
            self.console.print("No recent sessions found")
            return
        
        self.console.print("[bold]Recent Sessions:[/bold]")
        for session in recent_sessions:
            status = "🟢" if session.is_active else "🔴"
            self.console.print(
                f"{status} {session.id} - "
                f"Commands: {session.command_count}, "
                f"Started: {session.start_time.strftime('%H:%M:%S')}"
            )
    
    @debug_function
    def _run_diagnostics(self):
        """Run system diagnostics"""
        self.console.print("[bold]Running Diagnostics...[/bold]")
        
        # Run diagnostics
        diagnostics = self.debugging_manager.run_diagnostics()
        
        # Show key metrics
        perf_stats = diagnostics.get("performance_stats", {})
        self.console.print(f"CPU Usage: {perf_stats.get('cpu_percent', 'N/A')}%")
        self.console.print(f"Memory Usage: {perf_stats.get('memory_percent', 'N/A')}%")
        self.console.print(f"Disk Usage: {perf_stats.get('disk_usage_percent', 'N/A')}%")
        
        # Show recent failures
        failure_summary = self.terminal.recovery_manager.get_failure_summary()
        self.console.print(f"Recent Failures: {failure_summary.get('total_failures', 0)}")
        
        # Show profile data
        profile_data = diagnostics.get("profile_data", {})
        if profile_data:
            self.console.print("\n[b]Performance Profiles:[/b]")
            for func_name, stats in list(profile_data.items())[:5]:  # Show top 5
                self.console.print(
                    f"  {func_name}: {stats['call_count']} calls, "
                    f"avg {stats['avg_time']*1000:.2f}ms"
                )


def main():
    """Main entry point for the integrated Warp terminal"""
    def example_ai_suggester(recent_commands: List[str]) -> List[str]:
        """Example AI suggester with more sophisticated logic"""
        suggestions = []
        
        # Context-aware suggestions
        if any("git" in cmd for cmd in recent_commands[-3:]):  # Last 3 commands
            suggestions.extend(["git status", "git add .", "git commit -m 'Update'", "git push"])
        
        if any("docker" in cmd for cmd in recent_commands[-3:]):
            suggestions.extend(["docker ps", "docker images", "docker logs --tail 10"])
        
        if any("python" in cmd for cmd in recent_commands[-3:]):
            suggestions.extend(["python -m pip list", "python -m pytest", "python -m black ."])
        
        if any("npm" in cmd for cmd in recent_commands[-3:]):
            suggestions.extend(["npm run build", "npm test", "npm audit"])
        
        # Add some general suggestions
        suggestions.extend(["ls -la", "pwd", "history"])
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(suggestions))[:8]  # Return up to 8 unique suggestions
    
    # Create and run the integrated terminal
    terminal = IntegratedWarpTerminal(ai_suggester=example_ai_suggester)
    
    try:
        terminal.start_interactive_session()
    except Exception as e:
        print(f"Terminal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()