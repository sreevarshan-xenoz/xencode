#!/usr/bin/env python3

import sys
import subprocess
import requests
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()
DEFAULT_MODEL = "qwen:4b"

def list_models():
    try:
        output = subprocess.check_output(["ollama", "list"], text=True)
        console.print(Panel(output, title="📦 Installed Models", style="cyan"))
    except Exception as e:
        console.print(f"[red]❌ Error listing models: {e}[/red]")

def update_model(model):
    console.print(f"[yellow]🔄 Pulling latest model: {model}[/yellow]")
    try:
        subprocess.run(["ollama", "pull", model], check=True, capture_output=True)
        console.print(f"[green]✅ Successfully pulled {model}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to pull {model}: {e.stderr.decode()}[/red]")

def run_query(model, prompt):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    
    try:
        r = requests.post(url, json=payload)
        r.raise_for_status()
        return r.json()["response"]
    except requests.exceptions.RequestException as e:
        console.print(f"[red]❌ API Error: {e}[/red]")
        sys.exit(1)