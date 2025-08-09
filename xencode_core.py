#!/usr/bin/env python3

import sys
import subprocess
import requests
import json
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

console = Console(force_terminal=True, legacy_windows=False, color_system="auto")
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
    except FileNotFoundError:
        # Ollama not installed error
        error_panel = Panel(
            "❌ Ollama not found\n\nPlease install Ollama:\n• Visit: https://ollama.ai\n• Or use your package manager",
            title="Missing Dependency",
            style="red",
            border_style="red"
        )
        console.print(error_panel)
    except Exception as e:
        # Generic error panel
        error_panel = Panel(
            f"❌ Error listing models: {str(e)}\n\nPlease check your Ollama installation.",
            title="Model List Error",
            style="red",
            border_style="red"
        )
        console.print(error_panel)

def update_model(model):
    console.print(f"[yellow]🔄 Pulling latest model: {model}[/yellow]")
    try:
        subprocess.run(["ollama", "pull", model], check=True, capture_output=True)
        # Success panel
        success_panel = Panel(
            f"✅ Successfully pulled {model}\n\nThe model is now ready to use.",
            title="Model Updated",
            style="green",
            border_style="green"
        )
        console.print(success_panel)
    except FileNotFoundError:
        # Ollama not installed error
        error_panel = Panel(
            "❌ Ollama not found\n\nPlease install Ollama:\n• Visit: https://ollama.ai\n• Or use your package manager",
            title="Missing Dependency",
            style="red",
            border_style="red"
        )
        console.print(error_panel)
    except subprocess.CalledProcessError as e:
        # Model pull failed - could be missing model
        stderr_text = e.stderr.decode() if e.stderr else "Unknown error"
        if "not found" in stderr_text.lower():
            warning_panel = Panel(
                f"⚠️ Model '{model}' not found\n\nTry running: ./xencode.sh --list-models\nto see available models, or check the Ollama model library.",
                title="Model Not Found",
                style="yellow",
                border_style="yellow"
            )
            console.print(warning_panel)
        else:
            error_panel = Panel(
                f"❌ Failed to pull {model}\n\nError: {stderr_text}\n\nPlease check your internet connection and try again.",
                title="Update Failed",
                style="red",
                border_style="red"
            )
            console.print(error_panel)

def run_query(model, prompt):
    """Non-streaming query for backward compatibility"""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    
    try:
        r = requests.post(url, json=payload)
        r.raise_for_status()
        return r.json()["response"]
    except requests.exceptions.ConnectionError:
        # Claude-style connection error panel
        error_panel = Panel(
            "❌ Cannot connect to Ollama service\n\nPlease check:\n• Is Ollama running? Try: systemctl start ollama\n• Is the service accessible at localhost:11434?",
            title="Connection Error",
            style="red",
            border_style="red"
        )
        console.print(error_panel)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        # Generic API error panel
        error_panel = Panel(
            f"❌ API Error: {str(e)}\n\nPlease check your Ollama installation and try again.",
            title="API Error",
            style="red",
            border_style="red"
        )
        console.print(error_panel)
        sys.exit(1)

def run_streaming_query(model, prompt):
    """Real-time streaming query with Claude-style display"""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}
    
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        # Initialize streaming state
        full_response = ""
        in_thinking = False
        thinking_displayed = False
        answer_started = False
        displayed_chars = 0  # Track how many characters we've displayed
        
        # Show initial thinking indicator
        console.print("🧠 Thinking...", style="dim italic yellow")
        
        # Process streaming response
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'response' in chunk and chunk['response']:
                        token = chunk['response']
                        full_response += token
                        
                        # Check if we're entering thinking section
                        if not in_thinking and '<think>' in full_response:
                            in_thinking = True
                            # Start displaying from after <think>
                            think_start = full_response.find('<think>') + 7
                            displayed_chars = think_start
                            
                        # Check if we're exiting thinking section
                        elif in_thinking and '</think>' in full_response:
                            # Display remaining thinking content up to </think>
                            think_end = full_response.find('</think>')
                            if displayed_chars < think_end:
                                remaining_thinking = full_response[displayed_chars:think_end]
                                for char in remaining_thinking:
                                    print(f"\033[2;3;33m{char}\033[0m", end='', flush=True)
                                    time.sleep(THINKING_STREAM_DELAY)
                            
                            # Transition to answer
                            in_thinking = False
                            thinking_displayed = True
                            print()  # New line after thinking
                            time.sleep(THINKING_TO_ANSWER_PAUSE)
                            console.print("📄 Answer", style="bold green")
                            answer_started = True
                            
                            # Set displayed_chars to start of answer
                            displayed_chars = full_response.find('</think>') + 8
                            
                        # Stream new content based on current state
                        if in_thinking and not thinking_displayed:
                            # Stream thinking content
                            new_content = full_response[displayed_chars:]
                            for char in new_content:
                                print(f"\033[2;3;33m{char}\033[0m", end='', flush=True)
                                time.sleep(THINKING_STREAM_DELAY)
                            displayed_chars = len(full_response)
                            
                        elif answer_started:
                            # Stream answer content
                            new_content = full_response[displayed_chars:]
                            for char in new_content:
                                print(char, end='', flush=True)
                                time.sleep(ANSWER_STREAM_DELAY)
                            displayed_chars = len(full_response)
                            
                        elif not in_thinking and not answer_started and '<think>' not in full_response:
                            # No thinking section, start answer immediately
                            console.print("📄 Answer", style="bold green")
                            answer_started = True
                            new_content = full_response[displayed_chars:]
                            for char in new_content:
                                print(char, end='', flush=True)
                                time.sleep(ANSWER_STREAM_DELAY)
                            displayed_chars = len(full_response)
                            
                except json.JSONDecodeError:
                    continue
        
        print()  # Final newline
        return full_response
        
    except requests.exceptions.ConnectionError:
        # Claude-style connection error panel
        error_panel = Panel(
            "❌ Cannot connect to Ollama service\n\nPlease check:\n• Is Ollama running? Try: systemctl start ollama\n• Is the service accessible at localhost:11434?",
            title="Connection Error",
            style="red",
            border_style="red"
        )
        console.print(error_panel)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        # Generic API error panel
        error_panel = Panel(
            f"❌ API Error: {str(e)}\n\nPlease check your Ollama installation and try again.",
            title="API Error",
            style="red",
            border_style="red"
        )
        console.print(error_panel)
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
            # Stream each character with timing using direct stdout for immediate display
            for char in line:
                # Use ANSI codes for styling with direct stdout for immediate flushing
                print(f"\033[2;3;33m{char}\033[0m", end='', flush=True)
                time.sleep(THINKING_STREAM_DELAY)
            print()  # New line after each line
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
                    # Stream text character by character using direct stdout
                    for char in part.strip():
                        print(char, end='', flush=True)
                        time.sleep(ANSWER_STREAM_DELAY)
                    print()
            else:  # Code
                if part.strip():
                    lang = part.split('\n')[0] if '\n' in part else ""
                    code_content = part[part.find('\n')+1:] if '\n' in part else part
                    # Display code block immediately for readability
                    console.print(Syntax(code_content, lang or "plaintext", theme="monokai"))
    else:
        # Stream regular text character by character using direct stdout
        for char in answer_text.strip():
            print(char, end='', flush=True)
            time.sleep(ANSWER_STREAM_DELAY)
        print()

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
        "║ Xencode AI (Claude-Code Style | Qwen)    ║",
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
            
            # Process the query using real-time streaming
            try:
                response = run_streaming_query(model, user_input)
                # No need for format_output since streaming is handled in run_streaming_query
            except Exception as e:
                # Claude-style error panel for chat mode
                error_panel = Panel(
                    f"❌ Error processing your request\n\n{str(e)}\n\nPlease try again or check your setup.",
                    title="Processing Error",
                    style="red",
                    border_style="red"
                )
                console.print(error_panel)
            
            console.print()  # Add spacing between interactions
            
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and Ctrl+D signal handling
            handle_chat_exit()
            break
        except Exception as e:
            # Generic error panel for unexpected errors in chat mode
            error_panel = Panel(
                f"❌ Unexpected error: {str(e)}\n\nThe chat session will continue. Please try again.",
                title="Unexpected Error",
                style="red",
                border_style="red"
            )
            console.print(error_panel)
            continue

def main():
    args = sys.argv[1:]
    online = "false"
    chat_mode_enabled = False
    
    # Parse online flag
    if "--online=true" in args:
        online = "true"
        args = [arg for arg in args if arg != "--online=true"]
    elif "--online=false" in args:
        online = "false"
        args = [arg for arg in args if arg != "--online=false"]
    
    # Parse chat mode flag
    if "--chat-mode" in args:
        chat_mode_enabled = True
        args = [arg for arg in args if arg != "--chat-mode"]
    
    # Validate chat mode vs inline mode conflicts
    if chat_mode_enabled and args and not any(flag in args for flag in ["--list-models", "--update", "-m"]):
        error_panel = Panel(
            "❌ Invalid usage\n\nChat mode cannot be used with inline prompts.\n\nUse:\n• Chat mode: ./xencode.sh\n• Inline mode: ./xencode.sh \"your prompt\"",
            title="Usage Error",
            style="red",
            border_style="red"
        )
        console.print(error_panel)
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
            warning_panel = Panel(
                "⚠️ No internet connection\n\nCannot update models while offline.\nPlease check your connection and try again.",
                title="Offline Mode",
                style="yellow",
                border_style="yellow"
            )
            console.print(warning_panel)
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
    if chat_mode_enabled:
        chat_mode(model, online)
        return
    else:
        # Handle inline prompt
        if not args:
            warning_panel = Panel(
                "⚠️ No prompt provided\n\nUsage:\n• Inline mode: ./xencode.sh \"your prompt\"\n• Chat mode: ./xencode.sh",
                title="Missing Prompt",
                style="yellow",
                border_style="yellow"
            )
            console.print(warning_panel)
            return
        
        prompt = " ".join(args)
        
        try:
            response = run_query(model, prompt)
            format_output(response)
        except Exception as e:
            # Generic error panel for inline mode
            error_panel = Panel(
                f"❌ Unexpected error: {str(e)}\n\nPlease check your setup and try again.",
                title="Error",
                style="red",
                border_style="red"
            )
            console.print(error_panel)

if __name__ == "__main__":
    main()