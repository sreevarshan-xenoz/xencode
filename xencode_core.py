#!/usr/bin/env python3

import sys
import subprocess
import requests
import time
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live

# Try to import prompt_toolkit for enhanced input handling
try:
    from prompt_toolkit import prompt
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

console = Console()
DEFAULT_MODEL = "qwen3:4b"

# Claude-style streaming timing configuration
THINKING_STREAM_DELAY = 0.045  # 40-60ms per token
ANSWER_STREAM_DELAY = 0.030    # 20-40ms per token
THINKING_TO_ANSWER_PAUSE = 0.5 # 500ms pause between sections
THINKING_LINE_PAUSE = 0.125    # 100-150ms between thinking lines

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

def stream_thinking_section(thinking_text):
    """Stream thinking section with dim yellow italic styling and breathing pauses"""
    if not thinking_text:
        return
    
    console.print("🧠 Thinking...", style="dim italic yellow")
    
    lines = thinking_text.split('\n')
    for line in lines:
        if line.strip():  # Only process non-empty lines
            # Stream each character with timing
            for char in line:
                console.print(char, end='', style="dim italic yellow")
                sys.stdout.flush()
                time.sleep(THINKING_STREAM_DELAY)
            console.print()  # New line after each line
            time.sleep(THINKING_LINE_PAUSE)  # Breathing pause between lines

def stream_answer_section(answer_text):
    """Stream answer section with bold green styling and markdown support"""
    if not answer_text.strip():
        return
    
    console.print("\n📄 Answer", style="bold green")
    
    # Handle markdown in answer with streaming
    if answer_text.startswith("```") and "```" in answer_text[3:]:
        # For code blocks, stream the entire formatted block at once for readability
        parts = answer_text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Text
                if part.strip():
                    # Stream text character by character
                    for char in part.strip():
                        console.print(char, end='')
                        sys.stdout.flush()
                        time.sleep(ANSWER_STREAM_DELAY)
                    console.print()
            else:  # Code
                if part.strip():
                    lang = part.split('\n')[0] if '\n' in part else ""
                    code_content = part[part.find('\n')+1:] if '\n' in part else part
                    # Display code block immediately for readability
                    console.print(Syntax(code_content, lang or "plaintext", theme="monokai"))
    else:
        # Stream regular text character by character
        for char in answer_text.strip():
            console.print(char, end='')
            sys.stdout.flush()
            time.sleep(ANSWER_STREAM_DELAY)
        console.print()

def stream_claude_response(thinking_text, answer_text):
    """Stream complete response with exact Claude timing and formatting"""
    # Stream thinking section with breathing pauses
    if thinking_text:
        stream_thinking_section(thinking_text)
        # 0.5s pause between thinking and answer sections
        time.sleep(THINKING_TO_ANSWER_PAUSE)
    
    # Stream answer section
    stream_answer_section(answer_text)

def format_output(text, streaming=False):
    """Format output in Claude Code style with optional streaming"""
    thinking, answer = extract_thinking_and_answer(text)
    
    if streaming:
        # Use Claude-style streaming
        stream_claude_response(thinking, answer)
    else:
        # Use existing non-streaming format for backward compatibility
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
    """Display Claude-style centered banner with exact formatting"""
    if is_update:
        # Clear previous lines and redisplay banner for connectivity updates
        console.print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
    
    # Claude-style centered banner with exact format
    banner_lines = [
        "╔══════════════════════════════════════════╗",
        "║ Xencode AI (Claude-Code Style | Qwen) ║",
        "╚══════════════════════════════════════════╝"
    ]
    
    for line in banner_lines:
        console.print(line, style="cyan", justify="center")
    
    console.print("Offline-First | Hyprland Ready | Arch Optimized", style="dim", justify="center")
    console.print()
    
    # Dynamic status line with appropriate emojis
    if online_status == "true":
        status_line = "🌐 Online Mode - using local+internet models"
    else:
        status_line = "📡 Offline Mode - local models only"
    
    console.print(status_line, style="bold", justify="center")
    console.print()

def display_prompt():
    """Display the chat prompt in bold white"""
    console.print("[bold white][You] >[/bold white] ", end="")

def get_multiline_input():
    """Get user input with multiline support using prompt_toolkit if available"""
    if PROMPT_TOOLKIT_AVAILABLE:
        try:
            # Create key bindings for multiline input
            bindings = KeyBindings()
            
            @bindings.add(Keys.ControlM)  # Enter key
            def _(event):
                """Handle Enter key - submit input"""
                event.app.exit(result=event.app.current_buffer.text)
            
            @bindings.add('s-enter')  # Shift+Enter
            def _(event):
                """Handle Shift+Enter - add new line"""
                event.current_buffer.insert_text('\n')
            
            # Use prompt_toolkit for enhanced input with bold white prompt
            user_input = prompt(
                "",  # Empty prompt since we display our own
                multiline=True,
                key_bindings=bindings,
                wrap_lines=True,
                mouse_support=True
            )
            
            # Automatic newline trimming for submitted input
            return user_input.strip()
            
        except Exception:
            # Fall back to basic input if prompt_toolkit fails
            return input().strip()
    else:
        # Graceful fallback to basic input() if prompt_toolkit is not available
        return input().strip()

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

def handle_chat_exit():
    """Display goodbye message using existing Rich formatting"""
    console.print()
    console.print(Panel(
        "[bold green]👋 Thanks for using Xencode! Goodbye![/bold green]",
        style="cyan",
        title="🤖 Xencode AI"
    ))
    console.print()

def is_exit_command(user_input):
    """Check if user input is an exit command (exit, quit, q)"""
    exit_commands = ['exit', 'quit', 'q']
    return user_input.lower().strip() in exit_commands

def chat_mode(model, online):
    """Interactive chat loop that displays banner and prompts for input"""
    # Display initial banner
    display_chat_banner(model, online)
    
    # Track current online status for dynamic updates
    current_online = online
    
    while True:
        try:
            # Display prompt
            display_prompt()
            
            # Get user input with multiline support
            user_input = get_multiline_input()
            
            # Handle empty input gracefully without API calls
            if not user_input:
                console.print("[dim]Please enter a message or type 'exit' to quit.[/dim]")
                continue
            
            # Check for exit commands
            if is_exit_command(user_input):
                handle_chat_exit()
                break
            
            # Show thinking indicator with maximum 200ms latency
            console.print("[bold yellow]🧠 [Thinking...][/bold yellow]")
            
            # Check connectivity before API call
            new_online = update_online_status()
            if new_online != current_online:
                current_online = new_online
                display_chat_banner(model, current_online, is_update=True)
                display_prompt()
                console.print(user_input)  # Re-display user input after banner update
                console.print("[bold yellow]🧠 [Thinking...][/bold yellow]")
            
            # Process the query using existing functions with streaming
            try:
                response = run_query(model, user_input)
                format_output(response, streaming=True)
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
            
            console.print()  # Add spacing between interactions
            
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and Ctrl+D signal handling
            handle_chat_exit()
            break
        except Exception as e:
            console.print(f"[red]❌ Unexpected error: {e}[/red]")
            continue

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
        chat_mode(model, online)
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