#!/usr/bin/env python3
"""
Enhanced Xencode Warp Terminal Demo

Showcases Week 2 enhancements:
- Enhanced command palette with keyboard navigation
- Rich UI components for different output types
- Improved layouts and visual design
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.live import Live
import time

# Add xencode to path
sys.path.insert(0, str(Path(__file__).parent))

from xencode.warp_terminal import WarpTerminal, example_ai_suggester
from xencode.enhanced_command_palette import EnhancedCommandPalette, WarpTerminalWithPalette
from xencode.warp_ui_components import WarpLayoutManager, OutputRenderer

console = Console()


class EnhancedWarpDemo:
    """Demo for enhanced Warp terminal features"""
    
    def __init__(self):
        self.terminal = WarpTerminal(ai_suggester=example_ai_suggester)
        self.enhanced_terminal = WarpTerminalWithPalette(self.terminal)
        self.layout_manager = WarpLayoutManager()
        self.output_renderer = OutputRenderer()
    
    def run_demo(self):
        """Run the enhanced demo"""
        console.clear()
        
        # Welcome message
        welcome_panel = Panel.fit(
            "[bold blue]🚀 Enhanced Xencode Warp Terminal Demo[/bold blue]\n\n"
            "[green]Week 2 Features:[/green]\n"
            "• Enhanced Command Palette with Fuzzy Search\n"
            "• Rich UI Components for Different Output Types\n"
            "• Improved Layouts and Visual Design\n"
            "• Keyboard Navigation and Shortcuts\n\n"
            "Experience the next level of terminal interaction!",
            title="Enhanced Warp Terminal",
            border_style="blue"
        )
        console.print(welcome_panel)
        console.print()
        
        # Main demo menu
        while True:
            choice = self._show_main_menu()
            
            if choice == "1":
                self._demo_enhanced_palette()
            elif choice == "2":
                self._demo_rich_output_rendering()
            elif choice == "3":
                self._demo_improved_layouts()
            elif choice == "4":
                self._demo_interactive_features()
            elif choice == "5":
                self._start_enhanced_session()
            elif choice == "6":
                console.print("[green]Thank you for trying Enhanced Warp Terminal![/green]")
                break
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
            
            console.print("\n" + "="*60 + "\n")
    
    def _show_main_menu(self) -> str:
        """Show enhanced demo menu"""
        menu_table = Table(title="Enhanced Warp Terminal Demo", show_header=False)
        menu_table.add_column("Option", style="cyan", width=8)
        menu_table.add_column("Description", style="white")
        
        menu_table.add_row("1", "🎯 Enhanced Command Palette - Fuzzy search & AI suggestions")
        menu_table.add_row("2", "🎨 Rich Output Rendering - Structured display for different types")
        menu_table.add_row("3", "📐 Improved Layouts - Sidebar, panels, and organized views")
        menu_table.add_row("4", "⚡ Interactive Features - Keyboard shortcuts and navigation")
        menu_table.add_row("5", "💻 Enhanced Session - Full experience with all features")
        menu_table.add_row("6", "🚪 Exit Demo")
        
        console.print(menu_table)
        return Prompt.ask("\n[bold]Choose an option", choices=["1", "2", "3", "4", "5", "6"])
    
    def _demo_enhanced_palette(self):
        """Demo the enhanced command palette"""
        console.print(Panel.fit(
            "[bold green]🎯 Enhanced Command Palette Demo[/bold green]\n\n"
            "Features:\n"
            "• Fuzzy search with intelligent matching\n"
            "• AI-powered command suggestions\n"
            "• Keyboard navigation (↑↓ arrows, Enter, Tab)\n"
            "• Command history with frequency tracking\n"
            "• Built-in command templates",
            title="Enhanced Command Palette",
            border_style="green"
        ))
        
        # Build some command history first
        console.print("\n[yellow]Building command history for demo...[/yellow]")
        demo_commands = [
            "git status",
            "ls -la",
            "docker ps",
            "npm install",
            "python -m pip list",
            "git log --oneline -5"
        ]
        
        for cmd in demo_commands:
            console.print(f"  Executing: {cmd}")
            self.terminal.run_command(cmd)
        
        console.print("\n[cyan]Now let's test the enhanced command palette![/cyan]")
        
        if Confirm.ask("Open enhanced command palette?"):
            try:
                # Show the enhanced palette
                selected_command = self.enhanced_terminal.show_enhanced_palette()
                
                if selected_command:
                    console.print(f"\n[green]Selected command:[/green] {selected_command}")
                    
                    if Confirm.ask("Execute this command?"):
                        console.print(f"[cyan]Executing:[/cyan] {selected_command}")
                        block = self.terminal.run_command(selected_command)
                        
                        # Show the result with enhanced rendering
                        panel = self.layout_manager.create_command_block_panel(block, expanded=True)
                        console.print(panel)
                else:
                    console.print("[yellow]No command selected[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]Palette error:[/red] {e}")
                console.print("[yellow]Note: Enhanced palette requires prompt_toolkit for full functionality[/yellow]")
    
    def _demo_rich_output_rendering(self):
        """Demo rich output rendering for different types"""
        console.print(Panel.fit(
            "[bold yellow]🎨 Rich Output Rendering Demo[/bold yellow]\n\n"
            "See how different command outputs are rendered with:\n"
            "• Syntax highlighting for code and JSON\n"
            "• Structured tables for lists and processes\n"
            "• Color-coded git status and file listings\n"
            "• Interactive trees for hierarchical data",
            title="Rich Output Rendering",
            border_style="yellow"
        ))
        
        rendering_demos = [
            ("echo '{\"name\": \"test\", \"version\": \"1.0.0\", \"dependencies\": [\"react\", \"typescript\"]}'", "JSON Output"),
            ("ls -la", "File Listing"),
            ("git status", "Git Status"),
            ("ps aux | head -10", "Process List"),
            ("echo 'def hello_world():\n    print(\"Hello, World!\")\n    return True'", "Code Detection")
        ]
        
        for cmd, description in rendering_demos:
            console.print(f"\n[cyan]Demo:[/cyan] {description}")
            console.print(f"[dim]Command: {cmd}[/dim]")
            
            try:
                # Execute command
                block = self.terminal.run_command(cmd)
                
                # Show enhanced rendering
                panel = self.layout_manager.create_command_block_panel(block, expanded=True)
                console.print(panel)
                
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
            
            if not Confirm.ask("Continue to next rendering demo?", default=True):
                break
    
    def _demo_improved_layouts(self):
        """Demo improved layouts and visual design"""
        console.print(Panel.fit(
            "[bold magenta]📐 Improved Layouts Demo[/bold magenta]\n\n"
            "Enhanced visual design features:\n"
            "• Organized sidebar with recent commands and AI suggestions\n"
            "• Color-coded command blocks based on exit status\n"
            "• Expandable/collapsible output sections\n"
            "• Professional panel styling and borders",
            title="Improved Layouts",
            border_style="magenta"
        ))
        
        # Execute some commands to populate the layout
        console.print("\n[yellow]Executing commands to demonstrate layout...[/yellow]")
        
        layout_commands = [
            "echo 'Success command'",
            "ls /nonexistent 2>/dev/null || echo 'Failed command'",
            "date",
            "whoami"
        ]
        
        for cmd in layout_commands:
            console.print(f"  Running: {cmd}")
            self.terminal.run_command(cmd)
        
        # Show the full layout
        console.print("\n[cyan]Full Layout with Sidebar:[/cyan]")
        
        ai_suggestions = self.terminal.get_ai_suggestions_async()
        layout = self.layout_manager.create_full_layout(
            list(self.terminal.command_blocks),
            ai_suggestions=ai_suggestions or ["git add .", "docker ps", "npm test"]
        )
        
        console.print(layout)
        
        # Demo expandable blocks
        if self.terminal.command_blocks:
            last_block = list(self.terminal.command_blocks)[-1]
            console.print(f"\n[cyan]Expanded view of last command:[/cyan]")
            
            expanded_panel = self.layout_manager.create_command_block_panel(last_block, expanded=True)
            console.print(expanded_panel)
    
    def _demo_interactive_features(self):
        """Demo interactive features and keyboard shortcuts"""
        console.print(Panel.fit(
            "[bold red]⚡ Interactive Features Demo[/bold red]\n\n"
            "Interactive capabilities:\n"
            "• Live updating terminal display\n"
            "• Real-time command execution feedback\n"
            "• Keyboard shortcuts for common actions\n"
            "• Responsive UI that adapts to content",
            title="Interactive Features",
            border_style="red"
        ))
        
        console.print("\n[cyan]Live Terminal Demo:[/cyan]")
        console.print("[dim]Watch the terminal update in real-time as commands execute[/dim]")
        
        if Confirm.ask("Start live demo?"):
            # Create a live updating display
            with Live(
                self.layout_manager.create_full_layout(list(self.terminal.command_blocks)),
                refresh_per_second=2
            ) as live:
                
                interactive_commands = [
                    "echo 'Live update 1'",
                    "sleep 1 && echo 'Live update 2'",
                    "date",
                    "echo 'Live demo complete!'"
                ]
                
                for i, cmd in enumerate(interactive_commands):
                    console.print(f"\n[yellow]Executing ({i+1}/{len(interactive_commands)}):[/yellow] {cmd}")
                    
                    # Execute command
                    block = self.terminal.run_command(cmd)
                    
                    # Update live display
                    ai_suggestions = ["git status", "ls -la", "docker ps"]
                    updated_layout = self.layout_manager.create_full_layout(
                        list(self.terminal.command_blocks),
                        ai_suggestions=ai_suggestions
                    )
                    live.update(updated_layout)
                    
                    # Brief pause to show the update
                    time.sleep(1.5)
                
                console.print("\n[green]Live demo completed![/green]")
    
    def _start_enhanced_session(self):
        """Start full enhanced interactive session"""
        console.print(Panel.fit(
            "[bold cyan]💻 Enhanced Interactive Session[/bold cyan]\n\n"
            "Full Warp terminal experience with all enhancements:\n"
            "• Type commands normally\n"
            "• Use 'palette' or 'p' for enhanced command palette\n"
            "• All output will be rendered with rich formatting\n"
            "• Type 'exit' to return to demo menu",
            title="Enhanced Interactive Session",
            border_style="cyan"
        ))
        
        if Confirm.ask("Start enhanced interactive session?"):
            console.print("\n[green]Starting Enhanced Warp Terminal...[/green]")
            console.print("[dim]Tips:[/dim]")
            console.print("[dim]  • Type 'palette' for AI-powered command suggestions[/dim]")
            console.print("[dim]  • Try commands like 'git status', 'ls -la', 'ps aux'[/dim]")
            console.print("[dim]  • All output will be beautifully formatted![/dim]\n")
            
            try:
                # Start the enhanced session
                self.enhanced_terminal.start_interactive_session_with_palette()
            except KeyboardInterrupt:
                console.print("\n[yellow]Session interrupted.[/yellow]")
            
            console.print("[green]Returned to demo menu.[/green]")
        else:
            console.print("[yellow]Enhanced session skipped.[/yellow]")


def main():
    """Main demo function"""
    try:
        demo = EnhancedWarpDemo()
        demo.run_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user. Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
        console.print("[yellow]Please check your Python environment and dependencies.[/yellow]")


if __name__ == "__main__":
    main()