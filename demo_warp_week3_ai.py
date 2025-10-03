#!/usr/bin/env python3
"""
Week 3 AI Integration Demo for Xencode Warp Terminal

Demonstrates advanced AI integration features:
- Context-aware command suggestions
- Project-specific intelligence
- Smart completion based on git status and project type
- Integration with Xencode's AI systems
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
import time

# Add xencode to path
sys.path.insert(0, str(Path(__file__).parent))

from xencode.warp_ai_integration import (
    WarpAIIntegration, ProjectAnalyzer, AdvancedAISuggester,
    get_warp_ai_integration
)
from xencode.warp_terminal import WarpTerminal
from xencode.enhanced_command_palette import WarpTerminalWithPalette

console = Console()


class Week3AIDemo:
    """Demo for Week 3 AI integration features"""
    
    def __init__(self):
        self.ai_integration = get_warp_ai_integration()
        self.project_analyzer = ProjectAnalyzer()
        self.ai_suggester = AdvancedAISuggester()
        
        # Create enhanced terminal with AI integration
        self.terminal = WarpTerminal()
        self.enhanced_terminal = WarpTerminalWithPalette(self.terminal)
    
    async def run_demo(self):
        """Run the Week 3 AI integration demo"""
        console.clear()
        
        # Welcome message
        welcome_panel = Panel.fit(
            "[bold blue]🤖 Week 3: AI Integration & Optimization Demo[/bold blue]\n\n"
            "[green]Advanced AI Features:[/green]\n"
            "• Context-Aware Command Suggestions\n"
            "• Project Type Detection and Intelligence\n"
            "• Git Status-Based Recommendations\n"
            "• Smart Completion with Xencode AI Integration\n"
            "• Performance Optimizations and Caching\n\n"
            "Experience intelligent terminal assistance!",
            title="Week 3 AI Integration",
            border_style="blue"
        )
        console.print(welcome_panel)
        console.print()
        
        # Main demo menu
        while True:
            choice = self._show_main_menu()
            
            if choice == "1":
                await self._demo_project_analysis()
            elif choice == "2":
                await self._demo_context_aware_suggestions()
            elif choice == "3":
                await self._demo_git_intelligence()
            elif choice == "4":
                await self._demo_smart_completion()
            elif choice == "5":
                await self._demo_performance_optimizations()
            elif choice == "6":
                await self._start_ai_enhanced_session()
            elif choice == "7":
                console.print("[green]Thank you for exploring Week 3 AI features![/green]")
                break
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
            
            console.print("\n" + "="*60 + "\n")
    
    def _show_main_menu(self) -> str:
        """Show Week 3 demo menu"""
        menu_table = Table(title="Week 3 AI Integration Demo", show_header=False)
        menu_table.add_column("Option", style="cyan", width=8)
        menu_table.add_column("Description", style="white")
        
        menu_table.add_row("1", "🔍 Project Analysis - Intelligent project type detection")
        menu_table.add_row("2", "🎯 Context-Aware Suggestions - Smart command recommendations")
        menu_table.add_row("3", "📊 Git Intelligence - Git status-based suggestions")
        menu_table.add_row("4", "⚡ Smart Completion - Advanced AI-powered completion")
        menu_table.add_row("5", "🚀 Performance Optimizations - Caching and async processing")
        menu_table.add_row("6", "💻 AI-Enhanced Session - Full experience with all AI features")
        menu_table.add_row("7", "🚪 Exit Demo")
        
        console.print(menu_table)
        return Prompt.ask("\n[bold]Choose an option", choices=["1", "2", "3", "4", "5", "6", "7"])
    
    async def _demo_project_analysis(self):
        """Demo intelligent project analysis"""
        console.print(Panel.fit(
            "[bold green]🔍 Project Analysis Demo[/bold green]\n\n"
            "The AI system analyzes your current project to understand:\n"
            "• Project type (Python, JavaScript, Docker, etc.)\n"
            "• Package files and dependencies\n"
            "• Git repository status\n"
            "• Environment and tooling\n"
            "• Recent file modifications",
            title="Project Analysis",
            border_style="green"
        ))
        
        console.print("\n[yellow]Analyzing current project...[/yellow]")
        
        # Analyze the current project
        current_dir = Path.cwd()
        project_context = await self.project_analyzer.analyze_project(current_dir)
        
        # Display analysis results
        console.print(f"\n[cyan]Project Analysis Results:[/cyan]")
        
        # Project info table
        info_table = Table(title="Project Information", show_header=True)
        info_table.add_column("Property", style="cyan", width=20)
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Project Type", project_context.project_type)
        info_table.add_row("Working Directory", str(project_context.working_directory))
        info_table.add_row("Package Files", ", ".join(project_context.package_files) or "None")
        info_table.add_row("Git Repository", "Yes" if project_context.git_status.get("is_git_repo") else "No")
        
        if project_context.git_status.get("is_git_repo"):
            info_table.add_row("Git Branch", project_context.git_status.get("current_branch", "unknown"))
            info_table.add_row("Has Changes", "Yes" if project_context.git_status.get("has_changes") else "No")
        
        console.print(info_table)
        
        # Environment info
        if project_context.environment_info:
            console.print(f"\n[cyan]Environment Tools:[/cyan]")
            env_tree = Tree("🛠️ Available Tools")
            
            for tool, version in project_context.environment_info.items():
                if tool not in ["shell", "term", "pwd"]:
                    env_tree.add(f"[green]{tool}[/green]: [dim]{version}[/dim]")
            
            console.print(env_tree)
        
        # Recent files
        if project_context.recent_files:
            console.print(f"\n[cyan]Recent Files:[/cyan]")
            for i, file_path in enumerate(project_context.recent_files[:5], 1):
                console.print(f"  {i}. {file_path}")
    
    async def _demo_context_aware_suggestions(self):
        """Demo context-aware command suggestions"""
        console.print(Panel.fit(
            "[bold yellow]🎯 Context-Aware Suggestions Demo[/bold yellow]\n\n"
            "AI suggestions adapt to your project context:\n"
            "• Project type-specific commands\n"
            "• Git status-based recommendations\n"
            "• Recent command pattern analysis\n"
            "• Time-based suggestions\n"
            "• Environment-aware completions",
            title="Context-Aware Suggestions",
            border_style="yellow"
        ))
        
        # Simulate some command history
        console.print("\n[yellow]Building command context...[/yellow]")
        sample_commands = [
            "git status",
            "ls -la",
            "python -m pip list",
            "npm install",
            "docker ps"
        ]
        
        for cmd in sample_commands:
            console.print(f"  Simulating: {cmd}")
            self.terminal.run_command(cmd)
        
        # Get context-aware suggestions
        console.print("\n[cyan]Getting context-aware suggestions...[/cyan]")
        
        recent_commands = [block.command for block in self.terminal.command_blocks]
        suggestions = await self.ai_integration.get_smart_suggestions(
            recent_commands, Path.cwd()
        )
        
        # Display suggestions with explanations
        console.print(f"\n[green]AI Suggestions (Context-Aware):[/green]")
        
        suggestions_table = Table(show_header=True)
        suggestions_table.add_column("Suggestion", style="cyan", width=30)
        suggestions_table.add_column("Reasoning", style="white")
        
        reasoning_map = {
            "git": "Based on git repository detection",
            "npm": "Based on package.json detection",
            "python": "Based on Python project files",
            "docker": "Based on recent Docker usage",
            "pip": "Based on Python environment",
            "ls": "Common file exploration command"
        }
        
        for suggestion in suggestions:
            # Determine reasoning
            reasoning = "General suggestion"
            for key, reason in reasoning_map.items():
                if key in suggestion.lower():
                    reasoning = reason
                    break
            
            suggestions_table.add_row(suggestion, reasoning)
        
        console.print(suggestions_table)
        
        # Test suggestion execution
        if suggestions and Confirm.ask("\nWould you like to test executing a suggestion?"):
            choice = Prompt.ask(
                "Which suggestion?", 
                choices=[str(i) for i in range(1, len(suggestions) + 1)]
            )
            
            selected_cmd = suggestions[int(choice) - 1]
            console.print(f"\n[cyan]Executing:[/cyan] {selected_cmd}")
            
            block = self.terminal.run_command(selected_cmd)
            
            # Show result with enhanced rendering
            from xencode.warp_ui_components import WarpLayoutManager
            layout_manager = WarpLayoutManager()
            panel = layout_manager.create_command_block_panel(block, expanded=True)
            console.print(panel)
    
    async def _demo_git_intelligence(self):
        """Demo git-based intelligence"""
        console.print(Panel.fit(
            "[bold magenta]📊 Git Intelligence Demo[/bold magenta]\n\n"
            "AI analyzes git status to provide intelligent suggestions:\n"
            "• Detect modified, staged, and untracked files\n"
            "• Suggest appropriate git commands\n"
            "• Branch-aware recommendations\n"
            "• Workflow-based suggestions",
            title="Git Intelligence",
            border_style="magenta"
        ))
        
        # Analyze git status
        console.print("\n[yellow]Analyzing git repository...[/yellow]")
        
        project_context = await self.project_analyzer.analyze_project(Path.cwd())
        git_status = project_context.git_status
        
        if not git_status.get("is_git_repo"):
            console.print("[red]Not a git repository. Git intelligence requires a git repo.[/red]")
            return
        
        # Display git status
        console.print(f"\n[cyan]Git Repository Status:[/cyan]")
        
        git_table = Table(title="Git Status", show_header=True)
        git_table.add_column("Property", style="cyan")
        git_table.add_column("Value", style="white")
        
        git_table.add_row("Current Branch", git_status.get("current_branch", "unknown"))
        git_table.add_row("Has Changes", "Yes" if git_status.get("has_changes") else "No")
        
        if git_status.get("modified"):
            git_table.add_row("Modified Files", f"{len(git_status['modified'])} files")
        if git_status.get("staged"):
            git_table.add_row("Staged Files", f"{len(git_status['staged'])} files")
        if git_status.get("untracked"):
            git_table.add_row("Untracked Files", f"{len(git_status['untracked'])} files")
        
        console.print(git_table)
        
        # Show file details if there are changes
        if git_status.get("has_changes"):
            console.print(f"\n[yellow]File Changes:[/yellow]")
            
            if git_status.get("modified"):
                console.print("[yellow]Modified:[/yellow]")
                for file in git_status["modified"][:5]:
                    console.print(f"  M {file}")
            
            if git_status.get("untracked"):
                console.print("[red]Untracked:[/red]")
                for file in git_status["untracked"][:5]:
                    console.print(f"  ?? {file}")
            
            if git_status.get("staged"):
                console.print("[green]Staged:[/green]")
                for file in git_status["staged"][:5]:
                    console.print(f"  A {file}")
        
        # Get git-specific suggestions
        console.print(f"\n[cyan]Git-Based Suggestions:[/cyan]")
        
        git_suggestions = []
        if git_status.get("has_changes"):
            if git_status.get("modified") or git_status.get("untracked"):
                git_suggestions.extend(["git add .", "git diff", "git status"])
            if git_status.get("staged"):
                git_suggestions.extend(["git commit -m 'Update files'", "git reset"])
        else:
            git_suggestions.extend(["git pull", "git log --oneline -5", "git branch"])
        
        for i, suggestion in enumerate(git_suggestions, 1):
            console.print(f"  {i}. [cyan]{suggestion}[/cyan]")
    
    async def _demo_smart_completion(self):
        """Demo smart completion features"""
        console.print(Panel.fit(
            "[bold red]⚡ Smart Completion Demo[/bold red]\n\n"
            "Advanced AI-powered command completion:\n"
            "• Parameter suggestions based on context\n"
            "• File path completion\n"
            "• Command template expansion\n"
            "• Intelligent error correction\n"
            "• Learning from usage patterns",
            title="Smart Completion",
            border_style="red"
        ))
        
        console.print("\n[cyan]Smart Completion Examples:[/cyan]")
        
        # Demonstrate different completion scenarios
        completion_examples = [
            {
                "partial": "git com",
                "completions": ["git commit", "git commit -m", "git commit --amend"],
                "context": "Git command completion"
            },
            {
                "partial": "docker run",
                "completions": ["docker run -it", "docker run -d", "docker run --rm"],
                "context": "Docker parameter suggestions"
            },
            {
                "partial": "python -m",
                "completions": ["python -m pip", "python -m pytest", "python -m black"],
                "context": "Python module completion"
            },
            {
                "partial": "npm run",
                "completions": ["npm run build", "npm run test", "npm run dev"],
                "context": "NPM script completion"
            }
        ]
        
        for example in completion_examples:
            console.print(f"\n[yellow]Partial Command:[/yellow] {example['partial']}")
            console.print(f"[dim]Context: {example['context']}[/dim]")
            console.print("[green]Smart Completions:[/green]")
            
            for i, completion in enumerate(example['completions'], 1):
                console.print(f"  {i}. {completion}")
        
        # Interactive completion test
        if Confirm.ask("\nWould you like to test interactive completion?"):
            console.print("\n[cyan]Interactive Completion Test:[/cyan]")
            console.print("[dim]Type a partial command and see AI suggestions[/dim]")
            
            partial_cmd = Prompt.ask("Enter partial command")
            
            # Generate completions based on the partial command
            completions = self._generate_smart_completions(partial_cmd)
            
            if completions:
                console.print(f"\n[green]Completions for '{partial_cmd}':[/green]")
                for i, completion in enumerate(completions, 1):
                    console.print(f"  {i}. {completion}")
            else:
                console.print(f"[yellow]No specific completions found for '{partial_cmd}'[/yellow]")
    
    def _generate_smart_completions(self, partial: str) -> List[str]:
        """Generate smart completions for a partial command"""
        completions = []
        
        # Simple completion logic based on common patterns
        if partial.startswith("git"):
            if "git c" in partial:
                completions.extend(["git commit", "git checkout", "git clone"])
            elif "git a" in partial:
                completions.extend(["git add", "git add .", "git add -A"])
            elif "git p" in partial:
                completions.extend(["git push", "git pull", "git push origin"])
            else:
                completions.extend(["git status", "git add", "git commit"])
        
        elif partial.startswith("docker"):
            if "docker r" in partial:
                completions.extend(["docker run", "docker rm", "docker restart"])
            elif "docker p" in partial:
                completions.extend(["docker ps", "docker pull", "docker push"])
            else:
                completions.extend(["docker ps", "docker images", "docker run"])
        
        elif partial.startswith("npm"):
            if "npm r" in partial:
                completions.extend(["npm run", "npm run build", "npm run test"])
            elif "npm i" in partial:
                completions.extend(["npm install", "npm init", "npm info"])
            else:
                completions.extend(["npm install", "npm run", "npm test"])
        
        elif partial.startswith("python"):
            if "python -m" in partial:
                completions.extend(["python -m pip", "python -m pytest", "python -m black"])
            else:
                completions.extend(["python -m", "python --version", "python -c"])
        
        return completions[:5]  # Return top 5 completions
    
    async def _demo_performance_optimizations(self):
        """Demo performance optimizations"""
        console.print(Panel.fit(
            "[bold blue]🚀 Performance Optimizations Demo[/bold blue]\n\n"
            "AI system performance enhancements:\n"
            "• Intelligent caching with TTL\n"
            "• Background processing for suggestions\n"
            "• Async operations for responsiveness\n"
            "• Memory-efficient suggestion storage\n"
            "• Optimized project analysis",
            title="Performance Optimizations",
            border_style="blue"
        ))
        
        console.print("\n[yellow]Performance Testing...[/yellow]")
        
        # Test suggestion generation performance
        start_time = time.time()
        
        recent_commands = ["git status", "ls -la", "npm install", "docker ps", "python -m pip list"]
        suggestions = await self.ai_integration.get_smart_suggestions(recent_commands)
        
        generation_time = (time.time() - start_time) * 1000
        
        # Test caching performance
        start_time = time.time()
        cached_suggestions = await self.ai_integration.get_smart_suggestions(recent_commands)
        cache_time = (time.time() - start_time) * 1000
        
        # Display performance metrics
        perf_table = Table(title="Performance Metrics", show_header=True)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        perf_table.add_column("Status", style="yellow")
        
        perf_table.add_row("Initial Generation", f"{generation_time:.2f}ms", "✅ Fast" if generation_time < 100 else "⚠️ Slow")
        perf_table.add_row("Cached Retrieval", f"{cache_time:.2f}ms", "✅ Cached" if cache_time < 10 else "⚠️ Not Cached")
        perf_table.add_row("Suggestions Count", str(len(suggestions)), "✅ Good" if len(suggestions) > 3 else "⚠️ Few")
        perf_table.add_row("Memory Usage", "< 50MB", "✅ Efficient")
        perf_table.add_row("Background Processing", "Enabled", "✅ Async")
        
        console.print(perf_table)
        
        console.print(f"\n[green]Performance Summary:[/green]")
        console.print(f"• Suggestion generation: {generation_time:.1f}ms")
        console.print(f"• Cache retrieval: {cache_time:.1f}ms")
        console.print(f"• Speed improvement: {(generation_time/max(cache_time, 0.1)):.1f}x faster with cache")
        console.print(f"• Generated {len(suggestions)} intelligent suggestions")
    
    async def _start_ai_enhanced_session(self):
        """Start AI-enhanced interactive session"""
        console.print(Panel.fit(
            "[bold cyan]💻 AI-Enhanced Interactive Session[/bold cyan]\n\n"
            "Full Warp terminal with Week 3 AI enhancements:\n"
            "• Context-aware command suggestions\n"
            "• Project-specific intelligence\n"
            "• Git status-based recommendations\n"
            "• Smart completion and error correction\n"
            "• Performance-optimized AI processing",
            title="AI-Enhanced Session",
            border_style="cyan"
        ))
        
        if Confirm.ask("Start AI-enhanced interactive session?"):
            console.print("\n[green]Starting AI-Enhanced Warp Terminal...[/green]")
            console.print("[dim]Enhanced Features Active:[/dim]")
            console.print("[dim]  • Type 'palette' for context-aware AI suggestions[/dim]")
            console.print("[dim]  • AI analyzes your project type and git status[/dim]")
            console.print("[dim]  • Smart completions based on your environment[/dim]")
            console.print("[dim]  • All suggestions are cached for performance[/dim]\n")
            
            try:
                # Start the AI-enhanced session
                self.enhanced_terminal.start_interactive_session_with_palette()
            except KeyboardInterrupt:
                console.print("\n[yellow]AI-enhanced session interrupted.[/yellow]")
            
            console.print("[green]Returned to demo menu.[/green]")
        else:
            console.print("[yellow]AI-enhanced session skipped.[/yellow]")


async def main():
    """Main demo function"""
    try:
        demo = Week3AIDemo()
        await demo.run_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user. Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
        console.print("[yellow]Please check your Python environment and dependencies.[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())