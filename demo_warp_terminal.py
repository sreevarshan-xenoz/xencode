#!/usr/bin/env python3
"""
Xencode Warp Terminal Demo

Interactive demo showcasing the core Warp-like terminal features:
- Structured command blocks
- AI-powered suggestions  
- Performance optimizations
- Rich terminal UI
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

# Add xencode to path
sys.path.insert(0, str(Path(__file__).parent))

from xencode.warp_terminal import WarpTerminal, example_ai_suggester
from xencode.warp_testing_harness import CommandTestingHarness, run_comprehensive_test

console = Console()


class WarpTerminalDemo:
    """Interactive demo for the Warp-style terminal"""
    
    def __init__(self):
        self.terminal = WarpTerminal(ai_suggester=example_ai_suggester)
        self.harness = CommandTestingHarness()
    
    def run_demo(self):
        """Run the complete Warp terminal demo"""
        console.clear()
        
        # Welcome message
        welcome_panel = Panel.fit(
            "[bold blue]🚀 Xencode Warp Terminal Demo[/bold blue]\n\n"
            "[green]Structured Command Blocks[/green] • [yellow]AI Suggestions[/yellow] • [red]Performance Optimized[/red]\n\n"
            "Experience the next-generation terminal with Warp-like features\n"
            "built on top of Xencode's AI capabilities.",
            title="Welcome to Warp Terminal",
            border_style="blue"
        )
        console.print(welcome_panel)
        console.print()
        
        # Main demo menu
        while True:
            choice = self._show_main_menu()
            
            if choice == "1":
                self._demo_basic_commands()
            elif choice == "2":
                self._demo_structured_output()
            elif choice == "3":
                self._demo_ai_suggestions()
            elif choice == "4":
                self._demo_performance_features()
            elif choice == "5":
                self._run_test_suite()
            elif choice == "6":
                self._start_interactive_session()
            elif choice == "7":
                console.print("[green]Thank you for trying Xencode Warp Terminal![/green]")
                break
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
            
            console.print("\n" + "="*60 + "\n")
    
    def _show_main_menu(self) -> str:
        """Show main demo menu"""
        menu_table = Table(title="Warp Terminal Demo Menu", show_header=False)
        menu_table.add_column("Option", style="cyan", width=8)
        menu_table.add_column("Description", style="white")
        
        menu_table.add_row("1", "🎯 Basic Commands - Execute and view structured output")
        menu_table.add_row("2", "📊 Structured Output - See parsing for git, ls, ps commands")
        menu_table.add_row("3", "🤖 AI Suggestions - Experience context-aware command suggestions")
        menu_table.add_row("4", "⚡ Performance Features - Lazy rendering and streaming output")
        menu_table.add_row("5", "🧪 Test Suite - Run comprehensive tests and benchmarks")
        menu_table.add_row("6", "💻 Interactive Session - Full terminal experience")
        menu_table.add_row("7", "🚪 Exit Demo")
        
        console.print(menu_table)
        return Prompt.ask("\n[bold]Choose an option", choices=["1", "2", "3", "4", "5", "6", "7"])
    
    def _demo_basic_commands(self):
        """Demo basic command execution"""
        console.print(Panel.fit(
            "[bold green]🎯 Basic Commands Demo[/bold green]\n\n"
            "Execute basic commands and see how they're structured into blocks\n"
            "with metadata, tags, and formatted output.",
            title="Basic Commands",
            border_style="green"
        ))
        
        demo_commands = [
            "echo 'Hello from Warp Terminal!'",
            "date",
            "whoami",
            "pwd"
        ]
        
        for cmd in demo_commands:
            console.print(f"\n[cyan]Executing:[/cyan] {cmd}")
            block = self.terminal.run_command(cmd)
            
            # Display the structured block
            panel = self.terminal.renderer.render_block(block)
            console.print(panel)
            
            if not Confirm.ask("Continue to next command?", default=True):
                break
    
    def _demo_structured_output(self):
        """Demo structured output parsing"""
        console.print(Panel.fit(
            "[bold yellow]📊 Structured Output Demo[/bold yellow]\n\n"
            "See how different command outputs are parsed into structured data\n"
            "for better visualization and processing.",
            title="Structured Output",
            border_style="yellow"
        ))
        
        structured_commands = [
            ("ls -la", "File listing with detailed information"),
            ("git status", "Git repository status with changes"),
            ("ps aux | head -10", "Process list with system information"),
            ("echo '{\"name\": \"test\", \"value\": 42}'", "JSON output parsing")
        ]
        
        for cmd, description in structured_commands:
            console.print(f"\n[cyan]Command:[/cyan] {cmd}")
            console.print(f"[dim]{description}[/dim]")
            
            try:
                block = self.terminal.run_command(cmd)
                
                # Show the parsed structure
                output_type = block.output_data.get("type", "text")
                console.print(f"[green]Parsed as:[/green] {output_type}")
                
                # Display formatted output
                panel = self.terminal.renderer.render_block(block)
                console.print(panel)
                
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
            
            if not Confirm.ask("Continue to next command?", default=True):
                break
    
    def _demo_ai_suggestions(self):
        """Demo AI-powered command suggestions"""
        console.print(Panel.fit(
            "[bold magenta]🤖 AI Suggestions Demo[/bold magenta]\n\n"
            "Experience context-aware command suggestions based on your\n"
            "command history and current working context.",
            title="AI Suggestions",
            border_style="magenta"
        ))
        
        # Execute some commands to build context
        context_commands = ["git status", "ls -la", "pwd"]
        
        console.print("[yellow]Building command context...[/yellow]")
        for cmd in context_commands:
            console.print(f"  Executing: {cmd}")
            self.terminal.run_command(cmd)
        
        # Get AI suggestions
        console.print("\n[cyan]Getting AI suggestions based on context...[/cyan]")
        suggestions = self.terminal.get_ai_suggestions_async()
        
        if suggestions:
            console.print("\n[green]AI Suggested Commands:[/green]")
            for i, suggestion in enumerate(suggestions, 1):
                console.print(f"  {i}. {suggestion}")
            
            # Let user try a suggestion
            if Confirm.ask("\nWould you like to execute one of these suggestions?"):
                choice = Prompt.ask(
                    "Which suggestion? (number)", 
                    choices=[str(i) for i in range(1, len(suggestions) + 1)]
                )
                
                selected_cmd = suggestions[int(choice) - 1]
                console.print(f"\n[cyan]Executing suggested command:[/cyan] {selected_cmd}")
                block = self.terminal.run_command(selected_cmd)
                panel = self.terminal.renderer.render_block(block)
                console.print(panel)
        else:
            console.print("[yellow]No AI suggestions available yet. Try running more commands![/yellow]")
    
    def _demo_performance_features(self):
        """Demo performance optimization features"""
        console.print(Panel.fit(
            "[bold red]⚡ Performance Features Demo[/bold red]\n\n"
            "See lazy rendering, streaming output, and memory management\n"
            "in action with performance-intensive commands.",
            title="Performance Features",
            border_style="red"
        ))
        
        performance_commands = [
            ("find /usr -name '*.so' | head -20", "Large output with streaming"),
            ("ps aux", "Process list with structured parsing"),
            ("ls -laR /etc | head -50", "Recursive listing with lazy rendering")
        ]
        
        for cmd, description in performance_commands:
            console.print(f"\n[cyan]Performance Test:[/cyan] {cmd}")
            console.print(f"[dim]{description}[/dim]")
            
            # Use streaming command execution
            console.print("[yellow]Executing with streaming output...[/yellow]")
            start_time = time.time()
            
            try:
                import time
                block = self.terminal.run_command_streaming(cmd)
                
                # Wait a moment for streaming to start
                time.sleep(1)
                
                # Show performance info
                duration = time.time() - start_time
                console.print(f"[green]Started in {duration*1000:.1f}ms (streaming)[/green]")
                
                # Wait for completion
                timeout = 10
                elapsed = 0
                while block.metadata.get('exit_code') is None and elapsed < timeout:
                    time.sleep(0.1)
                    elapsed += 0.1
                
                if block.metadata.get('exit_code') is not None:
                    total_duration = block.metadata.get('duration_ms', 0)
                    output_size = len(str(block.output_data.get('data', '')))
                    console.print(f"[green]Completed in {total_duration}ms, output: {output_size} bytes[/green]")
                else:
                    console.print("[yellow]Command still running (streaming in background)[/yellow]")
                
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
            
            if not Confirm.ask("Continue to next performance test?", default=True):
                break
    
    def _run_test_suite(self):
        """Run the comprehensive test suite"""
        console.print(Panel.fit(
            "[bold blue]🧪 Test Suite Demo[/bold blue]\n\n"
            "Run comprehensive tests to validate performance, reliability,\n"
            "and parsing accuracy of the Warp terminal implementation.",
            title="Test Suite",
            border_style="blue"
        ))
        
        if Confirm.ask("This will run 25+ test commands. Continue?"):
            console.print("\n[yellow]Running comprehensive test suite...[/yellow]")
            
            try:
                # Run the test suite
                stress_results, parser_results, benchmark_results = run_comprehensive_test()
                
                # Show summary
                total_tests = len(stress_results)
                passed_tests = sum(1 for r in stress_results if r.success)
                
                console.print(f"\n[green]Test Summary:[/green]")
                console.print(f"  Total Tests: {total_tests}")
                console.print(f"  Passed: {passed_tests}")
                console.print(f"  Success Rate: {passed_tests/total_tests*100:.1f}%")
                
            except Exception as e:
                console.print(f"[red]Test suite failed:[/red] {e}")
        else:
            console.print("[yellow]Test suite skipped.[/yellow]")
    
    def _start_interactive_session(self):
        """Start full interactive terminal session"""
        console.print(Panel.fit(
            "[bold cyan]💻 Interactive Session[/bold cyan]\n\n"
            "Starting full Warp terminal experience.\n"
            "Type 'exit' to return to demo menu, 'palette' for command palette.",
            title="Interactive Session",
            border_style="cyan"
        ))
        
        if Confirm.ask("Start interactive terminal session?"):
            console.print("\n[green]Starting Warp Terminal...[/green]")
            console.print("[dim]Tip: Type 'palette' to see AI suggestions![/dim]\n")
            
            try:
                self.terminal.start_interactive_session()
            except KeyboardInterrupt:
                console.print("\n[yellow]Session interrupted.[/yellow]")
            
            console.print("[green]Returned to demo menu.[/green]")
        else:
            console.print("[yellow]Interactive session skipped.[/yellow]")


def main():
    """Main demo function"""
    try:
        demo = WarpTerminalDemo()
        demo.run_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user. Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
        console.print("[yellow]Please check your Python environment and dependencies.[/yellow]")


if __name__ == "__main__":
    main()