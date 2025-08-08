#!/usr/bin/env python3

import sys
import subprocess
import requests
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()
DEFAULT_MODEL = "qwen3:4b"

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

def extract_thinking_and_answer(text):
    """Extract thinking section and answer from Qwen response"""
    thinking = ""
    answer = text
    
    # Look for thinking section in Qwen format

    if "<think>" in text and "</think>" in text:
        try:
            thinking = text.split("<think>")[1].split("</think>")[0].strip()
            answer = text.split("</think>")[1].strip()
        except IndexError:
            # If parsing fails, treat entire text as answer
            pass
    elif "🧠 Thinking:" in text:
        # Handle alternative thinking format
        try:
            thinking = text.split("🧠 Thinking:")[1].split("\n\n")[0].strip()
            answer = text.split("🧠 Thinking:")[1].split("\n\n", 1)[1].strip()
        except IndexError:
            pass
    
    return thinking, answer

def format_output(text):
    """Format output in Claude Code style"""
    thinking, answer = extract_thinking_and_answer(text)
    
    # 🧠 Thinking Section
    if thinking:
        console.print("[bold yellow]🧠 Thinking...[/bold yellow]")
        console.print(f"[dim]{thinking}[/dim]\n")
    
    # 📄 Answer Section
    if answer.strip():
        console.print("[bold green]📄 Answer[/bold green]")
        
        # Handle markdown in answer
        if answer.startswith("```") and "```" in answer[3:]:
            # Extract and format code blocks
            parts = answer.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Text
                    if part.strip():
                        console.print(Markdown(part.strip()))
                else:  # Code
                    if part.strip():
                        lang = part.split('\n')[0] if '\n' in part else ""
                        code_content = part[part.find('\n')+1:] if '\n' in part else part
                        console.print(Syntax(code_content, lang or "plaintext", theme="monokai"))
        else:
            console.print(Markdown(answer.strip()))
        
        console.print()

def display_chat_banner(model, online_status, is_update=False):
    """Display the chat mode banner with model and online status"""
    online_text = "Yes" if online_status == "true" else "No"
    banner_text = f"""=== Xencode Chat Mode ===
Model: {model} | Online: {online_text}
Type 'exit', 'quit', or press Ctrl+C/Ctrl+D to exit."""
    
    if is_update:
        # Clear previous lines and redisplay banner for connectivity updates
        console.print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
    
    console.print(Panel(banner_text, style="cyan", title="🤖 Xencode AI"))
    console.print()

def display_prompt():
    """Display the chat prompt"""
    console.print("[bold blue][You] >[/bold blue] ", end="")

def update_online_status():
    """Check internet connectivity with lightweight ping"""
    try:
        # Use a lightweight ping check without blocking user interaction
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "1", "8.8.8.8"], 
            capture_output=True, 
            timeout=2
        )
        return "true" if result.returncode == 0 else "false"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return "false"

def main():
    args = sys.argv[1:]
    online = "false"
    chat_mode = False
    
    # Parse online flag
    if "--online=true" in args:
        online = "true"
        args = [arg for arg in args if arg != "--online=true"]
    elif "--online=false" in args:
        online = "false"
        args = [arg for arg in args if arg != "--online=false"]
    
    # Parse chat mode flag
    if "--chat-mode" in args:
        chat_mode = True
        args = [arg for arg in args if arg != "--chat-mode"]
    
    # Validate chat mode vs inline mode conflicts
    if chat_mode and args and not any(flag in args for flag in ["--list-models", "--update", "-m"]):
        console.print("[red]❌ Error: Chat mode cannot be used with inline prompts. Use chat mode without arguments or inline mode with prompts.[/red]")
        return
    
    # Handle --list-models
    if "--list-models" in args:
        list_models()
        return
    
    # Handle --update
    if "--update" in args:
        args = [arg for arg in args if arg != "--update"]
        if online == "true":
            model = DEFAULT_MODEL
            if "-m" in args:
                idx = args.index("-m")
                if idx + 1 < len(args):
                    model = args[idx + 1]
                    args.pop(idx)
                    args.pop(idx)
            update_model(model)
        else:
            console.print("[red]❌ No internet. Cannot update model.[/red]")
        return
    
    # Handle model specification
    model = DEFAULT_MODEL
    if "-m" in args:
        idx = args.index("-m")
        if idx + 1 < len(args):
            model = args[idx + 1]
            args.pop(idx)
            args.pop(idx)
    
    # Handle chat mode or inline prompt
    if chat_mode:
        # Chat mode will be implemented in subsequent tasks
        console.print("[yellow]🚧 Chat mode not yet implemented[/yellow]")
        return
    else:
        # Handle inline prompt
        if not args:
            console.print("[red]⚠ Please provide a prompt[/red]")
            return
        
        prompt = " ".join(args)
        
        try:
            response = run_query(model, prompt)
            format_output(response)
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

if __name__ == "__main__":
    main()