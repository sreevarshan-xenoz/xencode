#!/usr/bin/env python3
"""
Enhanced Command Palette for Xencode Warp Terminal

Implements fuzzy search, keyboard navigation, and AI-powered suggestions
with a professional prompt_toolkit interface.
"""

import asyncio
from typing import List, Callable, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time
import threading

from rich.console import Console
from rich.text import Text
from rich.panel import Panel

# Try to import prompt_toolkit for enhanced UI
try:
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout.containers import HSplit, VSplit, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.layout import Layout
    from prompt_toolkit.widgets import Box
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.formatted_text import HTML, FormattedText
    from prompt_toolkit.styles import Style
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False


@dataclass
class CommandSuggestion:
    """A command suggestion with metadata"""
    command: str
    description: str
    source: str  # "history", "ai", "builtin"
    frequency: int = 0
    last_used: Optional[float] = None


class FuzzyMatcher:
    """Fuzzy matching algorithm for command suggestions"""
    
    @staticmethod
    def score_match(query: str, text: str) -> float:
        """Calculate fuzzy match score (0.0 to 1.0)"""
        if not query:
            return 1.0
        
        query = query.lower()
        text = text.lower()
        
        # Exact match gets highest score
        if query == text:
            return 1.0
        
        # Substring match
        if query in text:
            # Prefer matches at the beginning
            index = text.find(query)
            position_bonus = 1.0 - (index / len(text)) * 0.3
            length_bonus = len(query) / len(text)
            return 0.8 * position_bonus * length_bonus
        
        # Character-by-character fuzzy matching
        score = 0.0
        query_idx = 0
        consecutive_matches = 0
        
        for char in text:
            if query_idx < len(query) and char == query[query_idx]:
                score += 1.0 + consecutive_matches * 0.1  # Bonus for consecutive matches
                consecutive_matches += 1
                query_idx += 1
            else:
                consecutive_matches = 0
        
        # Normalize by query length
        if query_idx == len(query):
            return (score / len(query)) * 0.6  # Max 0.6 for fuzzy matches
        
        return 0.0
    
    @staticmethod
    def filter_and_sort(query: str, suggestions: List[CommandSuggestion], limit: int = 10) -> List[Tuple[CommandSuggestion, float]]:
        """Filter and sort suggestions by fuzzy match score"""
        scored_suggestions = []
        
        for suggestion in suggestions:
            score = FuzzyMatcher.score_match(query, suggestion.command)
            if score > 0.1:  # Minimum threshold
                scored_suggestions.append((suggestion, score))
        
        # Sort by score (descending) and frequency (descending)
        scored_suggestions.sort(key=lambda x: (x[1], x[0].frequency), reverse=True)
        
        return scored_suggestions[:limit]


class EnhancedCommandPalette:
    """Enhanced command palette with keyboard navigation and fuzzy search"""
    
    def __init__(self, command_history: List[str], ai_suggester: Optional[Callable] = None):
        self.command_history = command_history
        self.ai_suggester = ai_suggester
        self.console = Console()
        
        # Convert history to suggestions
        self.suggestions = self._build_suggestions()
        
        # UI state
        self.selected_index = 0
        self.filtered_suggestions = []
        self.query = ""
        self.result = None
        
        # Callbacks
        self.on_select = None
        self.on_close = None
        
        # AI suggestions cache
        self._ai_suggestions_cache = []
        self._ai_suggestions_loading = False
        
        if PROMPT_TOOLKIT_AVAILABLE:
            self._setup_prompt_toolkit_ui()
        else:
            self.console.print("[yellow]Warning: prompt_toolkit not available, using fallback UI[/yellow]")
    
    def _build_suggestions(self) -> List[CommandSuggestion]:
        """Build suggestions from command history"""
        suggestions = []
        command_counts = {}
        
        # Count command frequency
        for cmd in self.command_history:
            command_counts[cmd] = command_counts.get(cmd, 0) + 1
        
        # Create suggestions from history
        for cmd, count in command_counts.items():
            suggestions.append(CommandSuggestion(
                command=cmd,
                description=f"Used {count} time{'s' if count != 1 else ''}",
                source="history",
                frequency=count,
                last_used=time.time()
            ))
        
        # Add built-in suggestions
        builtin_commands = [
            ("git status", "Check repository status"),
            ("git add .", "Stage all changes"),
            ("git commit -m", "Commit with message"),
            ("git push", "Push to remote"),
            ("ls -la", "List files with details"),
            ("ps aux", "List all processes"),
            ("docker ps", "List running containers"),
            ("npm install", "Install dependencies"),
            ("python -m pip install", "Install Python package"),
        ]
        
        for cmd, desc in builtin_commands:
            if cmd not in command_counts:  # Don't duplicate history items
                suggestions.append(CommandSuggestion(
                    command=cmd,
                    description=desc,
                    source="builtin",
                    frequency=0
                ))
        
        return suggestions
    
    def _setup_prompt_toolkit_ui(self):
        """Set up the prompt_toolkit UI"""
        if not PROMPT_TOOLKIT_AVAILABLE:
            return
        
        # Key bindings
        kb = KeyBindings()
        
        @kb.add('escape')
        @kb.add('c-c')
        def _(event):
            """Close palette"""
            self.result = None
            event.app.exit()
        
        @kb.add('up')
        def _(event):
            """Move selection up"""
            if self.selected_index > 0:
                self.selected_index -= 1
                self._update_ui()
        
        @kb.add('down')
        def _(event):
            """Move selection down"""
            if self.selected_index < len(self.filtered_suggestions) - 1:
                self.selected_index += 1
                self._update_ui()
        
        @kb.add('enter')
        def _(event):
            """Select current item"""
            if 0 <= self.selected_index < len(self.filtered_suggestions):
                suggestion, _ = self.filtered_suggestions[self.selected_index]
                self.result = suggestion.command
                event.app.exit()
        
        @kb.add('tab')
        def _(event):
            """Auto-complete with selected item"""
            if 0 <= self.selected_index < len(self.filtered_suggestions):
                suggestion, _ = self.filtered_suggestions[self.selected_index]
                # Update query buffer with the selected command
                self.query_buffer.text = suggestion.command
                self.query = suggestion.command
                self._filter_suggestions()
                self._update_ui()
        
        @kb.add('c-n')
        def _(event):
            """Create new command from query"""
            self.result = self.query
            event.app.exit()
        
        # Create UI components
        self.query_buffer = Buffer()
        
        def on_query_changed(buffer):
            """Handle query text changes"""
            self.query = buffer.text
            self._filter_suggestions()
            self.selected_index = 0
            self._update_ui()
        
        self.query_buffer.on_text_changed = on_query_changed
        
        # Command list control
        self.command_list_control = FormattedTextControl(
            text=self._get_command_list_text,
            focusable=True
        )
        
        # Status control
        self.status_control = FormattedTextControl(
            text=self._get_status_text,
            focusable=False
        )
        
        # Create layout
        self.layout = Layout(
            HSplit([
                # Header
                Window(
                    content=FormattedTextControl(
                        text=HTML('<b>Command Palette</b> - Type to search, ↑↓ to navigate, Enter to select')
                    ),
                    height=1,
                    style="class:header"
                ),
                # Query input
                Window(
                    content=BufferControl(buffer=self.query_buffer),
                    height=1,
                    style="class:input"
                ),
                # Separator
                Window(height=1, content=FormattedTextControl(text="─" * 80)),
                # Command list
                Window(
                    content=self.command_list_control,
                    style="class:list"
                ),
                # Status bar
                Window(
                    content=self.status_control,
                    height=1,
                    style="class:status"
                ),
            ])
        )
        
        # Style
        self.style = Style.from_dict({
            'header': 'bold',
            'input': 'cyan',
            'list': '',
            'status': 'dim',
            'selected': 'reverse bold',
            'ai': 'green',
            'history': 'blue',
            'builtin': 'yellow',
        })
        
        # Create application
        self.application = Application(
            layout=self.layout,
            key_bindings=kb,
            style=self.style,
            full_screen=False,
            mouse_support=True
        )
        
        # Initial filter
        self._filter_suggestions()
    
    def _filter_suggestions(self):
        """Filter suggestions based on current query"""
        # Get AI suggestions in background if not already loading
        if not self._ai_suggestions_loading and self.ai_suggester:
            self._load_ai_suggestions_async()
        
        # Combine all suggestions
        all_suggestions = self.suggestions + self._ai_suggestions_cache
        
        # Filter and sort by fuzzy match
        self.filtered_suggestions = FuzzyMatcher.filter_and_sort(
            self.query, all_suggestions, limit=15
        )
    
    def _load_ai_suggestions_async(self):
        """Load AI suggestions in background with advanced AI integration"""
        if self._ai_suggestions_loading:
            return
        
        self._ai_suggestions_loading = True
        
        def load_suggestions():
            try:
                ai_commands = []
                
                # Try advanced AI integration first
                if WARP_AI_AVAILABLE:
                    try:
                        import asyncio
                        from pathlib import Path
                        
                        ai_integration = get_warp_ai_integration()
                        
                        # Run async AI suggestions
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        ai_commands = loop.run_until_complete(
                            ai_integration.get_smart_suggestions(
                                self.command_history[-10:],  # More context
                                Path.cwd()
                            )
                        )
                        
                        loop.close()
                        
                    except Exception as e:
                        self.console.print(f"[yellow]Advanced AI failed, using fallback: {e}[/yellow]")
                        # Fallback to simple AI suggester
                        if self.ai_suggester:
                            ai_commands = self.ai_suggester(self.command_history[-5:])
                else:
                    # Use simple AI suggester
                    if self.ai_suggester:
                        ai_commands = self.ai_suggester(self.command_history[-5:])
                
                # Convert to CommandSuggestion objects
                ai_suggestions = []
                for cmd in ai_commands:
                    ai_suggestions.append(CommandSuggestion(
                        command=cmd,
                        description="AI suggested" if not WARP_AI_AVAILABLE else "Smart AI suggestion",
                        source="ai",
                        frequency=0
                    ))
                
                self._ai_suggestions_cache = ai_suggestions
                
                # Update UI if we have the application
                if hasattr(self, 'application') and self.application.is_running:
                    self._filter_suggestions()
                    self._update_ui()
                    
            except Exception as e:
                self.console.print(f"[red]AI suggestions failed: {e}[/red]")
            finally:
                self._ai_suggestions_loading = False
        
        # Start in background thread
        thread = threading.Thread(target=load_suggestions)
        thread.daemon = True
        thread.start()
    
    def _get_command_list_text(self):
        """Get formatted text for command list"""
        if not self.filtered_suggestions:
            return FormattedText([('', 'No matching commands found')])
        
        result = []
        
        for i, (suggestion, score) in enumerate(self.filtered_suggestions):
            # Style based on source
            if suggestion.source == "ai":
                style = "class:ai"
                prefix = "🤖 "
            elif suggestion.source == "history":
                style = "class:history"
                prefix = "📝 "
            else:
                style = "class:builtin"
                prefix = "⚙️  "
            
            # Highlight selected item
            if i == self.selected_index:
                style = "class:selected"
                prefix = "→ " + prefix
            else:
                prefix = "  " + prefix
            
            # Format the line
            line = f"{prefix}{suggestion.command}"
            if suggestion.description:
                line += f" - {suggestion.description}"
            
            result.append((style, line + '\n'))
        
        return FormattedText(result)
    
    def _get_status_text(self):
        """Get status bar text"""
        status_parts = []
        
        if self.filtered_suggestions:
            status_parts.append(f"{len(self.filtered_suggestions)} matches")
        
        if self._ai_suggestions_loading:
            status_parts.append("Loading AI suggestions...")
        elif self._ai_suggestions_cache:
            status_parts.append(f"{len(self._ai_suggestions_cache)} AI suggestions")
        
        status_text = " | ".join(status_parts) if status_parts else "Ready"
        
        return FormattedText([
            ('class:status', f"Status: {status_text} | Ctrl+C: Exit | Tab: Complete | Ctrl+N: New")
        ])
    
    def _update_ui(self):
        """Update the UI"""
        if hasattr(self, 'application') and self.application.is_running:
            self.application.invalidate()
    
    def show(self) -> Optional[str]:
        """Show the command palette and return selected command"""
        if not PROMPT_TOOLKIT_AVAILABLE:
            return self._show_fallback_palette()
        
        try:
            # Load AI suggestions in background
            self._load_ai_suggestions_async()
            
            # Run the application
            self.application.run()
            
            return self.result
            
        except Exception as e:
            self.console.print(f"[red]Command palette error: {e}[/red]")
            return self._show_fallback_palette()
    
    def _show_fallback_palette(self) -> Optional[str]:
        """Fallback palette for when prompt_toolkit is not available"""
        self.console.print("[bold]Command Palette (Fallback Mode)[/bold]")
        
        # Filter suggestions
        self._filter_suggestions()
        
        if not self.filtered_suggestions:
            self.console.print("No matching commands found")
            return None
        
        # Show suggestions
        for i, (suggestion, score) in enumerate(self.filtered_suggestions[:10]):
            source_icon = {"ai": "🤖", "history": "📝", "builtin": "⚙️"}.get(suggestion.source, "")
            self.console.print(f"{i+1}. {source_icon} {suggestion.command} - {suggestion.description}")
        
        # Get user selection
        from rich.prompt import Prompt
        choice = Prompt.ask("Select command (number) or type new command")
        
        try:
            # Try to parse as a number
            idx = int(choice) - 1
            if 0 <= idx < len(self.filtered_suggestions):
                suggestion, _ = self.filtered_suggestions[idx]
                return suggestion.command
        except ValueError:
            # Not a number, return as is
            return choice if choice.strip() else None
        
        return None


# Integration with WarpTerminal
class WarpTerminalWithPalette:
    """Warp terminal with enhanced command palette"""
    
    def __init__(self, terminal):
        self.terminal = terminal
        self.console = Console()
    
    def show_enhanced_palette(self) -> Optional[str]:
        """Show enhanced command palette"""
        # Get command history
        command_history = [block.command for block in self.terminal.command_blocks]
        
        # Create and show palette
        palette = EnhancedCommandPalette(
            command_history=command_history,
            ai_suggester=self.terminal.ai_suggester
        )
        
        return palette.show()
    
    def start_interactive_session_with_palette(self):
        """Start interactive session with enhanced palette"""
        self.terminal.running = True
        self.console.print("[bold green]Xencode Warp Terminal with Enhanced Palette[/bold green]")
        self.console.print("Press Ctrl+P for command palette, Ctrl+C to exit")
        
        from rich.live import Live
        
        with Live(self.terminal.renderer.render_live(list(self.terminal.command_blocks)), refresh_per_second=4) as live:
            while self.terminal.running:
                try:
                    from rich.prompt import Prompt
                    
                    # Get user input
                    command = Prompt.ask("[bold cyan]$[/bold cyan]")
                    
                    if command.lower() == "exit":
                        self.terminal.running = False
                    elif command.lower() in ["palette", "p"]:
                        # Show enhanced command palette
                        selected_command = self.show_enhanced_palette()
                        if selected_command:
                            # Execute the selected command
                            block = self.terminal.run_command(selected_command)
                            live.update(self.terminal.renderer.render_live(list(self.terminal.command_blocks)))
                    elif command.strip():
                        # Execute the command
                        from rich.progress import Progress, SpinnerColumn, TextColumn
                        
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            transient=True,
                        ) as progress:
                            task = progress.add_task(f"Running: {command}...", total=None)
                            block = self.terminal.run_command(command)
                        
                        live.update(self.terminal.renderer.render_live(list(self.terminal.command_blocks)))
                        
                except KeyboardInterrupt:
                    self.terminal.running = False
                except EOFError:
                    self.terminal.running = False
        
        self.console.print("[bold green]Session ended[/bold green]")


# Example usage
if __name__ == "__main__":
    from xencode.warp_terminal import WarpTerminal, example_ai_suggester
try:
    from xencode.warp_ai_integration import get_warp_ai_integration
    WARP_AI_AVAILABLE = True
except ImportError:
    WARP_AI_AVAILABLE = False
    
    # Create terminal
    terminal = WarpTerminal(ai_suggester=example_ai_suggester)
    
    # Add some sample commands to history
    sample_commands = ["git status", "ls -la", "docker ps", "npm install"]
    for cmd in sample_commands:
        terminal.run_command(cmd)
    
    # Test enhanced palette
    enhanced_terminal = WarpTerminalWithPalette(terminal)
    
    console = Console()
    console.print("[bold blue]Testing Enhanced Command Palette[/bold blue]")
    
    selected = enhanced_terminal.show_enhanced_palette()
    if selected:
        console.print(f"[green]Selected command:[/green] {selected}")
    else:
        console.print("[yellow]No command selected[/yellow]")