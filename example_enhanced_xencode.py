#!/usr/bin/env python3
"""
Enhanced Xencode Example

Demonstrates how to use Xencode with the new enhancement systems:
- User Feedback System for user-centric development
- Technical Debt Manager for code quality monitoring  
- AI Ethics Framework for responsible AI deployment
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Add xencode to path
sys.path.insert(0, str(Path(__file__).parent))

from xencode.enhancement_integration import (
    get_enhancement_integration,
    track_model_selection,
    track_query_response,
    collect_response_feedback,
    report_system_error,
    get_system_insights
)

console = Console()


class EnhancedXencode:
    """Enhanced Xencode with integrated monitoring systems"""
    
    def __init__(self, user_id: str = None):
        self.integration = get_enhancement_integration(user_id)
        self.console = Console()
        
    async def initialize(self):
        """Initialize the enhanced Xencode system"""
        self.console.print(Panel.fit(
            "[bold blue]🚀 Enhanced Xencode Initializing[/bold blue]\n\n"
            "[green]✓ User Feedback System[/green]\n"
            "[yellow]✓ Technical Debt Manager[/yellow]\n"
            "[red]✓ AI Ethics Framework[/red]\n"
            "[cyan]✓ Integration Layer[/cyan]",
            title="Enhancement Systems",
            border_style="blue"
        ))
        
        # Track initialization
        await self.integration.track_feature_usage("system_initialization")
    
    async def select_model(self, model_name: str):
        """Select AI model with tracking"""
        self.console.print(f"[cyan]Selecting model: {model_name}[/cyan]")
        
        # Track model selection
        await track_model_selection(model_name, "user_choice")
        
        self.console.print("[green]✓ Model selected and tracked[/green]")
    
    async def process_query(self, user_query: str) -> str:
        """Process user query with comprehensive monitoring"""
        self.console.print(f"[yellow]Processing query: {user_query[:50]}...[/yellow]")
        
        # Simulate AI response (in real implementation, this would call the actual AI)
        ai_response = self._simulate_ai_response(user_query)
        
        # Track query and analyze response
        violations = await track_query_response(user_query, ai_response, response_time_ms=150)
        
        # Show ethics analysis results
        if violations:
            self.console.print(f"[red]⚠️  {len(violations)} ethics violation(s) detected[/red]")
            for violation in violations:
                self.console.print(f"   • {violation.violation_type.value}: {violation.description}")
        else:
            self.console.print("[green]✓ No ethics violations detected[/green]")
        
        return ai_response
    
    def _simulate_ai_response(self, query: str) -> str:
        """Simulate AI response (replace with actual AI integration)"""
        # Simulate different types of responses for demo
        if "bias" in query.lower():
            return "Men are naturally better at programming than women. Contact admin@company.com for more info."
        elif "help" in query.lower():
            return "I'd be happy to help you with your programming questions. What specific topic would you like to explore?"
        else:
            return f"Here's a helpful response to your query about: {query[:30]}..."
    
    async def collect_feedback(self, rating: int, message: str = ""):
        """Collect user feedback on the interaction"""
        await collect_response_feedback(rating, message)
        self.console.print(f"[green]✓ Feedback collected: {rating}/5 stars[/green]")
    
    async def handle_error(self, error_message: str):
        """Handle and report system errors"""
        await report_system_error(error_message, {"component": "ai_processing"})
        self.console.print(f"[red]✗ Error reported: {error_message}[/red]")
    
    async def show_insights(self):
        """Show system insights and metrics"""
        insights = await get_system_insights()
        
        self.console.print("\n[bold cyan]System Insights:[/bold cyan]")
        for key, value in insights.items():
            if key != "error":
                self.console.print(f"  {key.replace('_', ' ').title()}: {value}")
    
    async def run_maintenance(self):
        """Run background maintenance tasks"""
        self.console.print("[yellow]Running background maintenance...[/yellow]")
        await self.integration.run_background_maintenance()
        self.console.print("[green]✓ Maintenance completed[/green]")
    
    async def shutdown(self):
        """Shutdown the system gracefully"""
        await self.integration.end_session()
        self.console.print("[blue]Session ended gracefully[/blue]")


async def demo_enhanced_xencode():
    """Demo the enhanced Xencode system"""
    console.print(Panel.fit(
        "[bold green]Enhanced Xencode Demo[/bold green]\n\n"
        "This demonstrates Xencode with integrated enhancement systems\n"
        "for user feedback, technical debt management, and AI ethics.",
        title="Demo",
        border_style="green"
    ))
    
    # Initialize enhanced Xencode
    xencode = EnhancedXencode("demo_user_enhanced")
    await xencode.initialize()
    
    # Demo workflow
    console.print("\n[bold]Demo Workflow:[/bold]")
    
    # 1. Model selection
    console.print("\n1. Model Selection:")
    await xencode.select_model("llama3.1:8b")
    
    # 2. Process queries with different characteristics
    console.print("\n2. Processing Queries:")
    
    # Normal query
    console.print("\n   Normal Query:")
    response1 = await xencode.process_query("How do I learn Python programming?")
    console.print(f"   Response: {response1[:60]}...")
    
    # Query that might trigger bias detection
    console.print("\n   Potentially Biased Query:")
    response2 = await xencode.process_query("Tell me about bias in AI systems")
    console.print(f"   Response: {response2[:60]}...")
    
    # 3. Collect feedback
    console.print("\n3. Collecting Feedback:")
    await xencode.collect_feedback(4, "Good responses, but detected some issues")
    
    # 4. Simulate an error
    console.print("\n4. Error Handling:")
    await xencode.handle_error("Model temporarily unavailable")
    
    # 5. Show insights
    console.print("\n5. System Insights:")
    await xencode.show_insights()
    
    # 6. Run maintenance
    console.print("\n6. Background Maintenance:")
    await xencode.run_maintenance()
    
    # 7. Shutdown
    console.print("\n7. Graceful Shutdown:")
    await xencode.shutdown()
    
    console.print("\n[bold green]✅ Enhanced Xencode demo completed![/bold green]")


async def interactive_demo():
    """Interactive demo allowing user input"""
    console.print(Panel.fit(
        "[bold blue]Interactive Enhanced Xencode[/bold blue]\n\n"
        "Try asking questions and see the enhancement systems in action!\n"
        "Type 'quit' to exit, 'insights' to see metrics, 'feedback <rating>' to rate.",
        title="Interactive Demo",
        border_style="blue"
    ))
    
    xencode = EnhancedXencode("interactive_user")
    await xencode.initialize()
    
    # Select a model
    await xencode.select_model("llama3.1:8b")
    
    console.print("\n[green]Ready! Ask me anything...[/green]")
    
    try:
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'insights':
                await xencode.show_insights()
            elif user_input.lower().startswith('feedback '):
                try:
                    rating = int(user_input.split()[1])
                    await xencode.collect_feedback(rating, "User provided rating")
                except (IndexError, ValueError):
                    console.print("[red]Usage: feedback <1-5>[/red]")
            elif user_input:
                response = await xencode.process_query(user_input)
                console.print(f"\n[cyan]Response:[/cyan] {response}")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    
    await xencode.shutdown()
    console.print("[green]Goodbye![/green]")


async def main():
    """Main function"""
    console.print("[bold]Choose demo mode:[/bold]")
    console.print("1. Automated Demo")
    console.print("2. Interactive Demo")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        await demo_enhanced_xencode()
    elif choice == "2":
        await interactive_demo()
    else:
        console.print("[red]Invalid choice[/red]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted. Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")