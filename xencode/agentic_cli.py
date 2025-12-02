import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from xencode.agentic.manager import LangChainManager

app = typer.Typer(help="Xencode Agentic CLI")
console = Console()


@app.command()
def start(
    model: str = typer.Option("qwen3:4b", help="Model to use for the agent"),
    base_url: str = typer.Option("http://localhost:11434", help="Ollama base URL"),
):
    """Start an interactive agentic session."""
    console.print(Panel.fit(f"Starting Agentic Session with {model}", style="bold blue"))
    
    try:
        manager = LangChainManager(model_name=model, base_url=base_url)
        console.print("[green]Agent initialized successfully![/green]")
        console.print("Type 'exit' or 'quit' to end the session.\n")

        while True:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            
            if user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not user_input.strip():
                continue

            with console.status("[bold green]Agent is thinking...[/bold green]"):
                response = manager.run_agent(user_input)
            
            console.print(Panel(response, title="Agent", border_style="green"))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    app()
