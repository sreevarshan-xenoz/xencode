#!/usr/bin/env python3

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

# Import from new modular structure
from xencode.core import (
    create_file,
    read_file,
    write_file,
    delete_file,
    ModelManager,
    get_smart_default_model,
    get_available_models,
    list_models,
    update_model,
    ConversationMemory,
    ResponseCache
)

# Try to import prompt_toolkit for enhanced input handling
try:
    from prompt_toolkit import prompt

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

# Import Enhanced CLI System with graceful fallback
try:
    from enhanced_cli_system import EnhancedXencodeCLI

    ENHANCED_CLI_AVAILABLE = True
except ImportError:
    ENHANCED_CLI_AVAILABLE = False
    EnhancedXencodeCLI = None

# Import enhanced systems for chat commands
try:
    from xencode.multi_model_system import MultiModelManager
except ImportError:
    MultiModelManager = None

try:
    from xencode.smart_context_system import SmartContextManager
except ImportError:
    SmartContextManager = None

try:
    from xencode.code_analysis_system import CodeAnalyzer
except ImportError:
    CodeAnalyzer = None

# Import project context detection
try:
    from xencode.project_context import get_project_context
    PROJECT_CONTEXT_AVAILABLE = True
except ImportError:
    PROJECT_CONTEXT_AVAILABLE = False

# Import System Checker
try:
    from xencode.system_checker import SystemChecker
    SYSTEM_CHECKER_AVAILABLE = True
    system_checker = SystemChecker()
    # Print system status on startup (optional, or can be behind a flag)
    # system_checker.print_status()
except ImportError:
    SYSTEM_CHECKER_AVAILABLE = False
    system_checker = None

# Import Intelligent Model Selector for First Run Setup
try:
    from xencode.intelligent_model_selector import FirstRunSetup
    FIRST_RUN_SETUP_AVAILABLE = True
except ImportError:
    FIRST_RUN_SETUP_AVAILABLE = False

# Import Ollama Fallback Manager for auto-start and installation
try:
    from xencode.ollama_fallback import OllamaFallbackManager
    OLLAMA_FALLBACK_AVAILABLE = True
except ImportError:
    OLLAMA_FALLBACK_AVAILABLE = False
    OllamaFallbackManager = None


# Suppress Rich color encoding warnings and other terminal warnings
os.environ.setdefault('FORCE_COLOR', '1')
os.environ.setdefault('TERM', 'xterm-256color')
os.environ.setdefault('COLORTERM', 'truecolor')

console = Console(
    force_terminal=True, legacy_windows=False, color_system="256", stderr=False
)

# Smart default model selection - will be updated based on available models
DEFAULT_MODEL = None  # Will be set dynamically

# Enhanced Claude-style streaming timing configuration
THINKING_STREAM_DELAY = 0.045  # 40-60ms per token
ANSWER_STREAM_DELAY = 0.030  # 20-40ms per token
THINKING_TO_ANSWER_PAUSE = 0.5  # 500ms pause between sections
THINKING_LINE_PAUSE = 0.125  # 100-150ms between thinking lines

# Performance and caching configuration
CACHE_ENABLED = True
CACHE_DIR = Path.home() / ".xencode" / "cache"
MAX_CACHE_SIZE = 100  # Maximum cached responses
RESPONSE_TIMEOUT = 30  # API response timeout in seconds

# Conversation memory configuration
MEMORY_ENABLED = True
MAX_MEMORY_ITEMS = 50
MEMORY_FILE = Path.home() / ".xencode" / "conversation_memory.json"


# Initialize global instances using the new modular components
memory = ConversationMemory()
cache = ResponseCache()
model_manager = ModelManager()


def get_available_models() -> List[str]:
    """Get available models with enhanced error handling and caching"""
    try:
        # Use the model manager for better performance
        model_manager.refresh_models()
        return model_manager.available_models
    except Exception:
        return []



try:
    from xencode.tui.utils.model_checker import ModelChecker
    MODEL_CHECKER_AVAILABLE = True
except ImportError:
    MODEL_CHECKER_AVAILABLE = False
    ModelChecker = None

def get_smart_default_model() -> Optional[str]:
    """Intelligently select the best available model"""
    available = []
    
    if MODEL_CHECKER_AVAILABLE:
        available = ModelChecker.get_available_models()
    else:
        # Minimal fallback if checker not available
        try:
            from xencode.tui.utils.model_checker import ModelChecker as MC
            available = MC.get_available_models()
        except ImportError:
            pass
            
    if not available:
        # Try one last direct check if list is empty or checker failed
        try:
            import subprocess
            output = subprocess.check_output(["ollama", "list"], text=True)
            if "NAME" in output:
                lines = output.strip().split('\n')
                available = [line.split()[0] for line in lines[1:] if line.strip()]
        except Exception:
            pass
    
    if not available:
        # No models found at all
        return None
    
    # Filter out embedding models
    chat_models = [m for m in available if "embed" not in m]
    if not chat_models:
        # Only embedding models? Use first avail as fallback, though unlikely to work for chat
        return available[0]
        
    # Preferred models in order of preference
    preferred_models = [
        "qwen2.5:7b",
        "qwen2.5:3b", 
        "qwen3:4b",
        "llama3.1:8b",
        "llama3.2:3b",
        "mistral:7b",
        "phi3:mini",
        "gemma2:2b",
    ]
    
    # Check for preferred models
    for preferred in preferred_models:
        for available_model in chat_models:
            if preferred in available_model.lower():
                return available_model
    
    # If no preferred model found, return the first available chat model
    return chat_models[0]



def list_models() -> None:
    """Enhanced model listing with health status and performance metrics"""
    try:
        # Refresh models and health status
        model_manager.refresh_models()

        if not model_manager.available_models:
            console.print(
                Panel(
                    "❌ No models found\n\nPlease install models with:\n• ollama pull qwen3:4b\n• ollama pull llama2\n• ollama pull mistral",
                    title="No Models Available",
                    style="red",
                    border_style="red",
                )
            )
            return

        # Create enhanced model table
        table = Table(
            title="📦 Installed Models", show_header=True, header_style="bold cyan"
        )
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Response Time", style="yellow")
        table.add_column("Last Check", style="dim")

        for model in model_manager.available_models:
            # Actively check model health
            try:
                is_healthy = model_manager.check_model_health(model)
                health = model_manager.model_health.get(model, {})
                status = health.get('status', 'unknown')

                # Color code status
                if status == 'healthy':
                    status_style = "✅ Healthy"
                    response_time = f"{health.get('response_time', 0):.3f}s"
                elif status == 'error':
                    status_style = "❌ Error"
                    response_time = "N/A"
                elif status == 'unavailable':
                    status_style = "⚠️ Unavailable"
                    response_time = "N/A"
                else:
                    status_style = "❓ Unknown"
                    response_time = "N/A"
            except Exception:
                status_style = "❓ Check Failed"
                response_time = "N/A"

            last_check = health.get('last_check', 0)
            if last_check:
                last_check_str = datetime.fromtimestamp(last_check).strftime("%H:%M:%S")
            else:
                last_check_str = "Never"

            table.add_row(model, status_style, response_time, last_check_str)

        console.print(table)

        # Show current model
        if model_manager.current_model:
            console.print(
                f"\n🎯 Current Model: [bold cyan]{model_manager.current_model}[/bold cyan]"
            )

        # Show recommendations
        if len(model_manager.available_models) == 1:
            console.print(
                "\n💡 Tip: Install more models for variety:\n• ollama pull llama2:7b\n• ollama pull mistral:7b"
            )

    except FileNotFoundError:
        # Ollama not installed error
        error_panel = Panel(
            "❌ Ollama not found\n\nPlease install Ollama:\n• Visit: https://ollama.ai\n• Or use your package manager",
            title="Missing Dependency",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
    except Exception as e:
        # Generic error panel
        error_panel = Panel(
            f"❌ Error listing models: {str(e)}\n\nPlease check your Ollama installation.",
            title="Model List Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)


def update_model(model: str) -> None:
    """Enhanced model update with progress tracking and validation"""
    console.print(f"[yellow]🔄 Pulling latest model: {model}[/yellow]")

    try:
        # Check if model exists first
        available_models = get_available_models()
        if model not in available_models:
            console.print(
                f"[yellow]📥 Model {model} not found locally, pulling from Ollama library...[/yellow]"
            )

        # Show progress spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Pulling {model}...", total=None)

            # Run the pull command
            result = subprocess.run(
                ["ollama", "pull", model], check=True, capture_output=True, text=True
            )

            progress.update(task, description=f"Validating {model}...")

            # Validate the model after pull
            if model_manager.check_model_health(model):
                progress.update(task, description=f"✅ {model} ready!")
                time.sleep(0.5)  # Show success briefly

                # Success panel with enhanced info
                success_panel = Panel(
                    f"✅ Successfully pulled and validated {model}\n\n"
                    f"📊 Model Status: [green]Healthy[/green]\n"
                    f"⚡ Response Time: [yellow]{model_manager.model_health[model]['response_time']:.3f}s[/yellow]\n"
                    f"🎯 Ready to use!",
                    title="Model Updated Successfully",
                    style="green",
                    border_style="green",
                )
                console.print(success_panel)

                # Refresh model list
                model_manager.refresh_models()
            else:
                raise Exception(f"Model {model} failed health check after pull")

    except FileNotFoundError:
        # Ollama not installed error
        error_panel = Panel(
            "❌ Ollama not found\n\nPlease install Ollama:\n• Visit: https://ollama.ai\n• Or use your package manager",
            title="Missing Dependency",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
    except subprocess.CalledProcessError as e:
        # Model pull failed - could be missing model
        stderr_text = e.stderr.decode() if e.stderr else "Unknown error"
        if "not found" in stderr_text.lower():
            warning_panel = Panel(
                f"⚠️ Model '{model}' not found in Ollama library\n\n"
                f"💡 Try these popular models:\n"
                f"• ollama pull qwen3:4b (fast, efficient)\n"
                f"• ollama pull llama2:7b (balanced)\n"
                f"• ollama pull mistral:7b (code-focused)\n"
                f"• ollama pull codellama:7b (programming)",
                title="Model Not Found",
                style="yellow",
                border_style="yellow",
            )
            console.print(warning_panel)
        else:
            error_panel = Panel(
                f"❌ Failed to pull {model}\n\nError: {stderr_text}\n\n"
                f"🔧 Troubleshooting:\n"
                f"• Check internet connection\n"
                f"• Verify Ollama service is running\n"
                f"• Try: systemctl restart ollama",
                title="Update Failed",
                style="red",
                border_style="red",
            )
            console.print(error_panel)
    except Exception as e:
        error_panel = Panel(
            f"❌ Unexpected error updating {model}\n\nError: {str(e)}\n\n"
            f"🔧 Please check your setup and try again.",
            title="Update Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)


def run_query(model: str, prompt: str) -> str:
    """Enhanced non-streaming query with caching, conversation memory, and project context"""
    # Check cache first
    cached_response = cache.get(prompt, model)
    if cached_response:
        console.print("[dim]💾 Using cached response[/dim]")
        return cached_response

    # Add user message to memory
    memory.add_message("user", prompt, model)

    url = "http://localhost:11434/api/generate"

    # Build context-aware prompt with project context
    context = memory.get_context(max_messages=5)
    context_prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in context]) if context else ""
    
    # Add project context if available and relevant
    project_info = ""
    if PROJECT_CONTEXT_AVAILABLE:
        try:
            project_ctx = get_project_context()
            if project_ctx.should_include_context(prompt):
                project_info = project_ctx.get_context_prompt()
        except Exception:
            pass  # Silently fail if project context detection fails
    
    # Build final prompt
    if project_info and context_prompt:
        enhanced_prompt = f"{project_info}{context_prompt}\n\nuser: {prompt}"
    elif project_info:
        enhanced_prompt = f"{project_info}user: {prompt}"
    elif context_prompt:
        enhanced_prompt = f"{context_prompt}\n\nuser: {prompt}"
    else:
        enhanced_prompt = prompt

    payload = {"model": model, "prompt": enhanced_prompt, "stream": False}

    try:
        # Show progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🤖 Processing...", total=None)

            r = requests.post(url, json=payload, timeout=RESPONSE_TIMEOUT)
            r.raise_for_status()

            response = r.json()["response"]

            # Cache the response
            cache.set(prompt, model, response)

            # Add AI response to memory
            memory.add_message("assistant", response, model)

            return response

    except requests.exceptions.ConnectionError:
        # Claude-style connection error panel
        error_panel = Panel(
            "❌ Cannot connect to Ollama service\n\nPlease check:\n• Is Ollama running? Try: systemctl start ollama\n• Is the service accessible at localhost:11434?",
            title="Connection Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        sys.exit(1)
    except requests.exceptions.Timeout:
        error_panel = Panel(
            f"⏰ Request timed out after {RESPONSE_TIMEOUT}s\n\n"
            f"🔧 Try:\n• Using a smaller model\n• Checking system resources\n• Restarting Ollama service",
            title="Request Timeout",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        # Generic API error panel
        error_panel = Panel(
            f"❌ API Error: {str(e)}\n\nPlease check your Ollama installation and try again.",
            title="API Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        sys.exit(1)


def run_streaming_query(model, prompt):
    """Enhanced REAL-TIME streaming query with conversation memory and context awareness"""
    # Add user message to memory
    memory.add_message("user", prompt, model)

    url = "http://localhost:11434/api/generate"

    # Build context-aware prompt with project context
    context = memory.get_context(max_messages=5)
    context_prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in context]) if context else ""
    
    # Add project context if available and relevant
    project_info = ""
    if PROJECT_CONTEXT_AVAILABLE:
        try:
            project_ctx = get_project_context()
            if project_ctx.should_include_context(prompt):
                project_info = project_ctx.get_context_prompt()
        except Exception:
            pass  # Silently fail if project context detection fails
    
    # Build final prompt
    if project_info and context_prompt:
        enhanced_prompt = f"{project_info}{context_prompt}\n\nuser: {prompt}"
    elif project_info:
        enhanced_prompt = f"{project_info}user: {prompt}"
    elif context_prompt:
        enhanced_prompt = f"{context_prompt}\n\nuser: {prompt}"
    else:
        enhanced_prompt = prompt

    payload = {"model": model, "prompt": enhanced_prompt, "stream": True}

    try:
        response = requests.post(
            url, json=payload, stream=True, timeout=RESPONSE_TIMEOUT
        )
        response.raise_for_status()

        # Stream tokens IMMEDIATELY as they arrive (no buffering)
        full_response = ""
        in_thinking = False
        thinking_shown = False
        answer_shown = False

        # Stream tokens in real-time
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))

                    # Check if streaming is complete
                    if chunk.get('done', False):
                        break

                    if 'response' in chunk and chunk['response']:
                        token = chunk['response']
                        full_response += token

                        # Detect thinking section markers
                        if '<think>' in token:
                            in_thinking = True
                            if not thinking_shown:
                                console.print("\n[bold cyan]Xencode[/bold cyan] [dim]›[/dim] [dim italic]thinking...[/dim italic]")
                                thinking_shown = True
                            continue
                        elif '</think>' in token:
                            in_thinking = False
                            if not answer_shown:
                                console.print("\n\n[bold cyan]Xencode[/bold cyan] [dim]›[/dim]")
                                answer_shown = True
                            continue

                        # Stream token immediately (no buffering!)
                        if in_thinking:
                            console.print(token, style="dim italic yellow", end="", highlight=False)
                        else:
                            if not answer_shown and not in_thinking:
                                console.print("\n[bold cyan]Xencode[/bold cyan] [dim]›[/dim]")
                                answer_shown = True
                            console.print(token, end="", highlight=False)

                        sys.stdout.flush()  # Force immediate output

                except json.JSONDecodeError:
                    continue

        console.print()  # New line at end

        # Add AI response to memory
        memory.add_message("assistant", full_response, model)

        return full_response

    except requests.exceptions.ConnectionError:
        # Claude-style connection error panel
        error_panel = Panel(
            "❌ Cannot connect to Ollama service\n\nPlease check:\n• Is Ollama running? Try: systemctl start ollama\n• Is the service accessible at localhost:11434?",
            title="Connection Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        sys.exit(1)
    except requests.exceptions.Timeout:
        error_panel = Panel(
            f"⏰ Request timed out after {RESPONSE_TIMEOUT}s\n\n"
            f"🔧 Try:\n• Using a smaller model\n• Checking system resources\n• Restarting Ollama service",
            title="Request Timeout",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        # Generic API error panel
        error_panel = Panel(
            f"❌ API Error: {str(e)}\n\nPlease check your Ollama installation and try again.",
            title="API Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        sys.exit(1)


def extract_thinking_and_answer(text: str) -> Tuple[str, str]:
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
            # Stream each character with timing using Rich console for consistent output
            for char in line:
                # Use Rich console for immediate display
                console.print(char, style="dim italic yellow", end="", highlight=False)
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
                    # Stream text character by character using Rich console
                    for char in part.strip():
                        console.print(char, end="", highlight=False)
                        time.sleep(ANSWER_STREAM_DELAY)
                    console.print()
            else:  # Code
                if part.strip():
                    lang = part.split('\n')[0] if '\n' in part else ""
                    code_content = part[part.find('\n') + 1 :] if '\n' in part else part
                    # Display code block immediately for readability
                    console.print(
                        Syntax(code_content, lang or "plaintext", theme="monokai")
                    )
    else:
        # Stream regular text character by character using Rich console
        for char in answer_text.strip():
            console.print(char, end="", highlight=False)
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
                            code_content = (
                                part[part.find('\n') + 1 :] if '\n' in part else part
                            )
                            console.print(
                                Syntax(
                                    code_content, lang or "plaintext", theme="monokai"
                                )
                            )
            else:
                console.print(Markdown(answer.strip()))

            console.print()


def display_chat_banner(model, online_status, is_update=False):
    """Display immersive full-screen banner (like Gemini/Crush/Claude CLI)"""
    if is_update:
        # Clear screen and redisplay banner for connectivity updates
        console.clear()

    # Immersive banner with modern design
    console.print()
    console.print("╔═══════════════════════════════════════════════════════════════╗", style="bold cyan", justify="center")
    console.print("║                                                               ║", style="bold cyan", justify="center")
    console.print(f"║                    🤖 XENCODE AI ASSISTANT                    ║", style="bold cyan", justify="center")
    console.print("║                                                               ║", style="bold cyan", justify="center")
    console.print("╚═══════════════════════════════════════════════════════════════╝", style="bold cyan", justify="center")
    console.print()
    
    # Model and status info
    console.print(f"Model: {model}", style="bold white", justify="center")
    
    # Dynamic status line with appropriate emojis
    if online_status == "true":
        status_line = "🌐 Online Mode"
        status_style = "bold green"
    else:
        status_line = "📡 Offline Mode"
        status_style = "bold yellow"
    
    console.print(status_line, style=status_style, justify="center")
    console.print()
    console.print("─" * 80, style="dim", justify="center")
    console.print()


def display_prompt():
    """Display the chat prompt with immersive styling (like Gemini/Crush/Claude CLI)"""
    console.print("\n[bold cyan]You[/bold cyan] [dim]›[/dim] ", end="")


def get_multiline_input():
    """Get user input with multiline support using prompt_toolkit if available"""
    if PROMPT_TOOLKIT_AVAILABLE:
        try:
            # Use prompt_toolkit with default key bindings for reliability
            # Enter submits, Ctrl+N adds new line
            user_input = prompt(
                "",  # Empty prompt since we display our own
                multiline=True,
                wrap_lines=True,
                mouse_support=False,  # Disable mouse support to avoid conflicts
            )

            # Clean and return input
            if user_input:
                return user_input.strip()
            return ""

        except Exception as e:
            # Fall back to basic input if prompt_toolkit fails
            console.print(
                f"[dim]Warning: prompt_toolkit failed, using basic input: {e}[/dim]"
            )
            return input().strip()
    else:
        # Graceful fallback to basic input() if prompt_toolkit is not available
        return input().strip()


def update_online_status():
    """Check internet connectivity with lightweight ping"""
    try:
        # Use a lightweight ping check without blocking user interaction
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "1", "8.8.8.8"], capture_output=True, timeout=2
        )
        return "true" if result.returncode == 0 else "false"
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return "false"


def handle_chat_exit():
    """Display goodbye message with immersive styling"""
    console.print()
    console.print()
    console.print("─" * 80, style="dim", justify="center")
    console.print()
    console.print("👋 Thanks for using Xencode!", style="bold cyan", justify="center")
    console.print("Your AI assistant that respects your privacy", style="dim", justify="center")
    console.print()
    console.print("─" * 80, style="dim", justify="center")
    console.print()


def handle_chat_command(command, current_model, current_online):
    """Handle enhanced chat commands with rich formatting"""
    cmd_parts = command.split()
    cmd = cmd_parts[0].lower()

    # Try enhanced chat commands first if available
    if ENHANCED_CLI_AVAILABLE:
        try:
            from enhanced_chat_commands import EnhancedChatCommands
            from enhanced_cli_system import FeatureDetector

            # Initialize enhanced chat commands if not already done
            if not hasattr(handle_chat_command, '_enhanced_commands'):
                detector = FeatureDetector()
                features = detector.detect_features()

                # Get enhanced systems if available
                enhanced_systems = {}
                if features.multi_model and MultiModelManager:
                    enhanced_systems['multi_model'] = MultiModelManager()
                if features.smart_context and SmartContextManager:
                    enhanced_systems['smart_context'] = SmartContextManager()
                if features.code_analysis and CodeAnalyzer:
                    enhanced_systems['code_analyzer'] = CodeAnalyzer()

                handle_chat_command._enhanced_commands = EnhancedChatCommands(
                    features, enhanced_systems
                )

            # Handle enhanced commands
            enhanced_commands = handle_chat_command._enhanced_commands

            # Check if this is an enhanced command
            enhanced_command_names = ["analyze", "model", "models", "context", "smart"]

            if cmd[1:] in enhanced_command_names:  # Remove leading /
                args = " ".join(cmd_parts[1:]) if len(cmd_parts) > 1 else ""
                response, new_model = enhanced_commands.handle_chat_command(
                    cmd[1:], args, current_model
                )

                console.print(Panel(response, style="cyan"))

                if new_model:
                    return (
                        "MODEL_SWITCH",
                        new_model,
                        f"Enhanced model switch to {new_model}",
                    )

                return True

            # Handle smart mode model suggestion for regular queries
            if not command.startswith("/") and enhanced_commands.smart_mode_enabled:
                suggested_model, reasoning = enhanced_commands.suggest_model_for_query(
                    command, current_model
                )

                if suggested_model != current_model:
                    console.print(f"[dim cyan]🤖 {reasoning}[/dim cyan]")
                    return "MODEL_SWITCH", suggested_model, reasoning

        except Exception as e:
            # Enhanced commands failed - fall back to legacy
            console.print(
                f"[dim yellow]⚠️ Enhanced commands unavailable: {e}[/dim yellow]"
            )

    # Legacy chat commands
    if cmd == "/help":
        show_help_panel()
        return True
    elif cmd == "/clear":
        memory.start_session()  # Start fresh session
        console.print(
            Panel("🧹 Conversation cleared. New session started.", style="green")
        )
        return True
    elif cmd == "/memory":
        show_memory_info()
        return True
    elif cmd == "/sessions":
        show_sessions_list()
        return True
    elif cmd == "/switch":
        if len(cmd_parts) > 1:
            session_id = cmd_parts[1]
            if memory.switch_session(session_id):
                console.print(
                    Panel(f"✅ Switched to session: {session_id}", style="green")
                )
            else:
                console.print(Panel(f"❌ Session not found: {session_id}", style="red"))
        else:
            console.print(Panel("❌ Usage: /switch <session_id>", style="red"))
        return True
    elif cmd == "/cache":
        show_cache_info()
        return True
    elif cmd == "/status":
        show_system_status(current_model, current_online)
        return True
    elif cmd == "/export":
        export_conversation()
        return True
    elif cmd == "/model":
        if len(cmd_parts) > 1:
            new_model = cmd_parts[1]
            success, message = model_manager.switch_model(new_model)
            if success:
                # Update the current model in the calling function
                # We'll need to return the new model
                return "MODEL_SWITCH", new_model, message
            else:
                console.print(Panel(f"❌ Model switch failed: {message}", style="red"))
                return True
        else:
            console.print(Panel("❌ Usage: /model <model_name>", style="red"))
            return True
    elif cmd == "/theme":
        if len(cmd_parts) > 1:
            change_theme(cmd_parts[1])
        else:
            show_available_themes()
        return True
    elif cmd == "/project":
        show_project_context()
        return True
    elif cmd == "/models":
        show_available_models_interactive()
        return True

    return False


def show_help_panel():
    """Display comprehensive help panel with all commands"""
    help_text = """
🎯 **Chat Commands:**
• /help - Show this help
• /clear - Clear current conversation
• /memory - Show memory usage
• /sessions - List all sessions
• /switch <id> - Switch to session
• /cache - Show cache info
• /status - System status
• /export - Export conversation
• /project - Show project context
• /theme <name> - Change theme

🔧 **Model Commands:**
• /models - Show available models with health status
• /model <name> - Switch to a different model
• /update <name> - Download/update a model

💬 **Regular Input:**
• Type your message and press Enter
• Use Shift+Enter for multiline
• Type 'exit' or 'quit' to end
"""

    # Add enhanced commands if available
    if ENHANCED_CLI_AVAILABLE:
        try:
            from enhanced_cli_system import FeatureDetector

            detector = FeatureDetector()
            features = detector.detect_features()

            if features.enhanced_features_available:
                enhanced_text = "\n\n🚀 **Enhanced Commands:**\n"

                if features.code_analysis:
                    enhanced_text += "• /analyze [path] - Analyze code quality\n"

                if features.multi_model:
                    enhanced_text += "• /models - List models with capabilities\n"

                if features.smart_context:
                    enhanced_text += "• /context - Show project context\n"
                    enhanced_text += "• /context clear - Clear context cache\n"
                    enhanced_text += "• /context refresh - Refresh context\n"

                if features.multi_model and features.smart_context:
                    enhanced_text += "• /smart on|off - Toggle smart model selection\n"

                enhanced_text += f"\n🎚️ Feature Level: {features.feature_level.upper()}"

                help_text += enhanced_text
        except Exception:
            # Enhanced commands not available
            pass

    help_panel = Panel(help_text, title="📚 Xencode Help", style="cyan")
    console.print(help_panel)


def show_memory_info():
    """Display memory usage and statistics"""
    context = memory.get_context()
    sessions = memory.list_sessions()

    memory_text = f"""
🧠 **Memory Information:**
• Current Session: {memory.current_session}
• Messages in Context: {len(context)}
• Total Sessions: {len(sessions)}
• Memory Limit: {MAX_MEMORY_ITEMS} messages

📊 **Current Context:**
"""

    if context:
        for i, msg in enumerate(context[-5:], 1):  # Show last 5 messages
            role = msg['role'].capitalize()
            content_preview = (
                msg['content'][:50] + "..."
                if len(msg['content']) > 50
                else msg['content']
            )
            memory_text += f"• {i}. {role}: {content_preview}\n"
    else:
        memory_text += "• No messages in context\n"

    memory_panel = Panel(memory_text, title="🧠 Memory Status", style="blue")
    console.print(memory_panel)


def show_sessions_list():
    """Display list of all conversation sessions"""
    sessions = memory.list_sessions()

    if not sessions:
        console.print(Panel("❌ No sessions found", style="red"))
        return

    table = Table(
        title="💬 Conversation Sessions", show_header=True, header_style="bold cyan"
    )
    table.add_column("Session ID", style="cyan")
    table.add_column("Messages", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Created", style="dim")
    table.add_column("Last Updated", style="dim")

    for session_id in sessions:
        session_data = memory.conversations.get(session_id, {})
        messages_count = len(session_data.get('messages', []))
        model = session_data.get('model', 'Unknown')
        created = session_data.get('created', 'Unknown')
        last_updated = session_data.get('last_updated', 'Unknown')

        # Format timestamps
        try:
            created_dt = datetime.fromisoformat(created)
            created_str = created_dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, AttributeError):
            created_str = created if isinstance(created, str) else "Unknown"

        try:
            updated_dt = datetime.fromisoformat(last_updated)
            updated_str = updated_dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, AttributeError):
            updated_str = last_updated if isinstance(last_updated, str) else "Unknown"

        # Highlight current session
        if session_id == memory.current_session:
            session_id = f"🎯 {session_id}"

        table.add_row(session_id, str(messages_count), model, created_str, updated_str)

    console.print(table)


def show_cache_info():
    """Display cache information and statistics"""
    try:
        cache_files = list(CACHE_DIR.glob("*.json"))
        cache_size = len(cache_files)

        cache_text = f"""
💾 **Cache Information:**
• Cache Directory: {CACHE_DIR}
• Cached Responses: {cache_size}
• Max Cache Size: {MAX_CACHE_SIZE}
• Cache Status: {'Enabled' if CACHE_ENABLED else 'Disabled'}
"""

        if cache_size > 0:
            cache_text += "\n📊 **Recent Cache Entries:**\n"
            # Show recent cache entries
            recent_files = sorted(
                cache_files, key=lambda x: x.stat().st_mtime, reverse=True
            )[:5]
            for i, cache_file in enumerate(recent_files, 1):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        prompt_preview = (
                            data.get('prompt', '')[:40] + "..."
                            if len(data.get('prompt', '')) > 40
                            else data.get('prompt', '')
                        )
                        cache_text += f"• {i}. {prompt_preview}\n"
                except (OSError, json.JSONDecodeError, KeyError) as e:
                    cache_text += f"• {i}. [Error reading cache: {type(e).__name__}]\n"

        cache_panel = Panel(cache_text, title="💾 Cache Status", style="magenta")
        console.print(cache_panel)

    except Exception as e:
        console.print(Panel(f"❌ Error reading cache: {str(e)}", style="red"))


def show_system_status(current_model, current_online):
    """Display comprehensive system status"""
    # Check Ollama service
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_status = "✅ Running" if response.status_code == 200 else "❌ Error"
    except (requests.RequestException, OSError):
        ollama_status = "❌ Not accessible"

    # Check model health actively
    try:
        if model_manager.check_model_health(current_model):
            model_health = model_manager.model_health.get(current_model, {})
            response_time = f"{model_health.get('response_time', 0):.3f}s"
            model_status_display = f"✅ Healthy ({response_time})"
        else:
            model_health = model_manager.model_health.get(current_model, {})
            model_status_display = (
                f"❌ {model_health.get('status', 'Unavailable').capitalize()}"
            )
    except Exception:
        model_status_display = "❌ Check Failed"

    status_text = f"""
🖥️ **System Status:**
• Ollama Service: {ollama_status}
• Current Model: {current_model}
• Model Status: {model_status_display}
• Internet: {'🌐 Online' if current_online == 'true' else '🔌 Offline'}
• Memory Usage: {len(memory.get_context())} messages
• Cache Status: {'✅ Enabled' if CACHE_ENABLED else '❌ Disabled'}

📊 **Performance:**
• Available Models: {len(model_manager.available_models)}
• Cache Size: {len(list(CACHE_DIR.glob('*.json'))) if CACHE_DIR.exists() else 0}
• Session Count: {len(memory.conversations)}
"""

    status_panel = Panel(status_text, title="📊 System Status", style="green")
    console.print(status_panel)


def export_conversation():
    """Export current conversation to file"""
    try:
        context = memory.get_context()
        if not context:
            console.print(Panel("❌ No conversation to export", style="red"))
            return

        # Create export directory
        export_dir = Path.home() / ".xencode" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.md"
        export_path = export_dir / filename

        # Export as markdown
        with open(export_path, 'w') as f:
            f.write("# Xencode Conversation Export\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Session:** {memory.current_session}\n\n")

            for msg in context:
                role = msg['role'].capitalize()
                content = msg['content']
                timestamp = msg['timestamp']

                f.write(f"## {role}\n\n")
                f.write(f"*{timestamp}*\n\n")
                f.write(f"{content}\n\n")
                f.write("---\n\n")

        console.print(
            Panel(f"✅ Conversation exported to:\n{export_path}", style="green")
        )

    except Exception as e:
        console.print(Panel(f"❌ Export failed: {str(e)}", style="red"))


def change_theme(theme_name):
    """Change the visual theme (placeholder for future implementation)"""
    console.print(
        Panel(
            f"🎨 Theme '{theme_name}' not implemented yet.\n\nAvailable themes:\n• default\n• dark\n• light\n• colorful",
            style="yellow",
        )
    )


def show_available_themes():
    """Show available themes"""
    themes_text = """
🎨 **Available Themes:**
• default - Standard Xencode theme
• dark - Dark mode (coming soon)
• light - Light mode (coming soon)
• colorful - Enhanced colors (coming soon)

💡 Use: /theme <name> to change
"""

    themes_panel = Panel(themes_text, title="🎨 Themes", style="magenta")
    console.print(themes_panel)


def show_project_context():
    """Show current project context"""
    if not PROJECT_CONTEXT_AVAILABLE:
        console.print(Panel(
            "❌ Project context detection not available\n\n"
            "This feature requires the project_context module.",
            title="Feature Not Available",
            style="red"
        ))
        return
    
    try:
        project_ctx = get_project_context()
        context = project_ctx.detect_project()
        
        context_text = f"""
📁 **Project Information:**
• Type: {context['type']}
• Directory: {project_ctx.cwd}

🌿 **Git Status:**
• Branch: {context['git']['branch']}
• Has Changes: {'Yes' if context['git']['has_changes'] else 'No'}

📝 **Modified Files:**
{chr(10).join(f"• {f}" for f in context['files'][:10]) if context['files'] else '• No modified files'}

📦 **Dependencies:**
{chr(10).join(f"• {d}" for d in context['dependencies'][:10]) if context['dependencies'] else '• No dependencies detected'}

💡 **Tip:** Project context is automatically included in code-related queries.
"""
        
        context_panel = Panel(context_text, title="📊 Project Context", style="blue")
        console.print(context_panel)
        
    except Exception as e:
        console.print(Panel(
            f"❌ Failed to detect project context\n\nError: {str(e)}",
            title="Context Detection Error",
            style="red"
        ))


def show_available_models_interactive():
    """Show available models with health status and allow switching"""
    models = get_available_models()
    
    if not models:
        console.print(Panel(
            "❌ No models found\n\n"
            "Install a model with: ollama pull qwen2.5:3b",
            title="No Models Available",
            style="red"
        ))
        return
    
    # Create table with model info
    table = Table(title="🤖 Available Models", show_header=True, header_style="bold cyan")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Model", style="white")
    table.add_column("Status", style="green")
    table.add_column("Response Time", style="yellow")
    table.add_column("Current", style="magenta")
    
    current_model = model_manager.current_model
    
    for i, model in enumerate(models, 1):
        # Check health
        is_healthy = model_manager.check_model_health(model)
        health = model_manager.model_health.get(model, {})
        
        if health.get('status') == 'healthy':
            status = "✅ Healthy"
            response_time = f"{health.get('response_time', 0):.2f}s"
        else:
            status = "❌ Error"
            response_time = "N/A"
        
        is_current = "⭐" if model == current_model else ""
        
        table.add_row(str(i), model, status, response_time, is_current)
    
    console.print(table)
    console.print()
    console.print("[dim]💡 Tip: Use /model <name> to switch models[/dim]")
    console.print(f"[dim]Current model: [bold]{current_model}[/bold][/dim]")


def is_exit_command(user_input):
    """Check if user input is an exit command (exit, quit, q)"""
    exit_commands = ['exit', 'quit', 'q']
    return user_input.lower().strip() in exit_commands


def create_file(path, content):
    try:
        p = os.path.abspath(path)
        with open(p, 'w') as f:
            f.write(content)
        console.print(Panel(f"✅ {p}", title="Created", style="green"))
    except (OSError, IOError, PermissionError) as e:
        console.print(Panel(f"❌ Failed: {type(e).__name__}", style="red"))


def read_file(path):
    try:
        with open(os.path.abspath(path), 'r') as f:
            console.print(Panel(f.read(), title=path, style="cyan"))
    except (OSError, IOError, FileNotFoundError) as e:
        console.print(Panel(f"❌ Error: {type(e).__name__}", style="red"))


def write_file(path, content):
    create_file(path, content)


def delete_file(path):
    try:
        os.remove(os.path.abspath(path))
        console.print(Panel(f"✅ {path}", title="Deleted", style="green"))
    except (OSError, FileNotFoundError, PermissionError) as e:
        console.print(Panel(f"❌ Failed: {type(e).__name__}", style="red"))


def chat_mode(model, online):
    """Enhanced interactive chat loop with immersive full-screen experience (like Gemini/Crush/Claude CLI)"""
    # Clear screen for immersive experience
    console.clear()
    
    # Start a new conversation session
    session_id = memory.start_session()

    # Display initial banner
    display_chat_banner(model, online)

    # Track current online status for dynamic updates
    current_online = online
    current_model = model

    # Show session info
    console.print(f"[dim]💬 Session: {session_id}[/dim]")
    console.print(f"[dim]🧠 Memory: {len(memory.get_context())} messages[/dim]")
    console.print()  # Extra spacing for clean look

    while True:
        try:
            # Display prompt
            display_prompt()

            # Get user input with multiline support
            user_input = get_multiline_input()

            # Handle empty input gracefully without API calls
            if not user_input:
                console.print(
                    "[dim]Please enter a message or type 'exit' to quit.[/dim]"
                )
                continue

            # Enhanced command system
            if user_input.startswith("/"):
                command_result = handle_chat_command(
                    user_input, current_model, current_online
                )
                if command_result:
                    # Handle special return values
                    if (
                        isinstance(command_result, tuple)
                        and command_result[0] == "MODEL_SWITCH"
                    ):
                        _, new_model, message = command_result
                        current_model = new_model
                        console.print(
                            Panel(
                                f"✅ Model switched to [bold]{current_model}[/bold]\n{message}",
                                style="green",
                            )
                        )
                    continue
                else:
                    # If command not handled, treat as regular input
                    pass

            # Check for exit commands
            if is_exit_command(user_input):
                handle_chat_exit()
                break

            # Show thinking indicator with immersive styling
            console.print("\n[bold cyan]Xencode[/bold cyan] [dim]›[/dim] [dim italic]processing...[/dim italic]")

            # Check connectivity before API call
            new_online = update_online_status()
            if new_online != current_online:
                current_online = new_online
                display_chat_banner(current_model, current_online, is_update=True)
                # Don't re-display prompt or input to avoid confusion
                console.print("[dim]🌐 Connection status updated[/dim]")
                console.print("[bold yellow]🧠 [Thinking...][/bold yellow]")

            # Process the query using real-time streaming
            try:
                response = run_streaming_query(current_model, user_input)
                # No need for format_output since streaming is handled in run_streaming_query
            except Exception as e:
                # Claude-style error panel for chat mode
                error_panel = Panel(
                    f"❌ Error processing your request\n\n{str(e)}\n\nPlease try again or check your setup.",
                    title="Processing Error",
                    style="red",
                    border_style="red",
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
                border_style="red",
            )
            console.print(error_panel)
            continue


def check_ollama_health():
    """Check if Ollama is running and accessible, with fallback auto-start"""
    # First, quick check if already running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True, "Ollama is running"
    except (requests.ConnectionError, requests.Timeout):
        pass
    except Exception:
        pass
    
    # Ollama not running - try fallback manager
    if OLLAMA_FALLBACK_AVAILABLE and OllamaFallbackManager:
        try:
            fallback = OllamaFallbackManager(auto_start=True, auto_install=False)
            
            if fallback.try_start_ollama():
                return True, "Ollama started successfully"
            
            # If not installed, return failure (installation needs explicit user action)
            if not fallback.is_ollama_installed():
                return False, (
                    "Ollama is not installed.\n"
                    "Run 'xencode' with first-time setup or install manually from: https://ollama.ai"
                )
        except Exception as e:
            console.print(f"[dim yellow]Fallback manager error: {e}[/dim yellow]")
    
    # Fallback to original error messages
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True, "Ollama is running"
        else:
            return False, f"Ollama returned status {response.status_code}"
    except requests.ConnectionError:
        return False, "Ollama is not running. Start it with: ollama serve"
    except requests.Timeout:
        return False, "Ollama is not responding (timeout)"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def is_first_run():
    """Check if this is the first run"""
    config_file = Path.home() / ".xencode" / "config.json"
    return not config_file.exists()


def run_first_time_setup():
    """Run interactive setup for first-time users"""
    # Try advanced setup wizard first
    if FIRST_RUN_SETUP_AVAILABLE:
        try:
            setup = FirstRunSetup()
            model = setup.run_setup()
            # If setup completed (model selected) or user skipped (model=None but wizard ran),
            # check if config exists to determine if we're done.
            # actually wizard saves config if successful.
            config_file = Path.home() / ".xencode" / "model_config.json"
            if config_file.exists() or model:
               return
        except Exception as e:
            console.print(f"[dim yellow]Advanced setup not available: {e}[/dim yellow]")

    console.print(Panel(
        "👋 Welcome to Xencode!\n\n"
        "Let's get you set up in 30 seconds...",
        title="First Run Setup",
        style="blue"
    ))
    
    # Use Ollama Fallback Manager for intelligent setup
    if OLLAMA_FALLBACK_AVAILABLE and OllamaFallbackManager:
        fallback = OllamaFallbackManager(auto_start=True, auto_install=False)
        success, message = fallback.ensure_ollama_available()
        
        if not success:
            console.print(Panel(
                f"❌ {message}",
                title="Ollama Setup Failed",
                style="red"
            ))
            sys.exit(1)
        else:
            console.print(f"[green]✅ {message}[/green]")
    else:
        # Legacy fallback: manual checks
        # Check Ollama installation
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True, timeout=5)
        except (FileNotFoundError, subprocess.CalledProcessError):
            console.print(Panel(
                "❌ Ollama is not installed\n\n"
                "Install Ollama:\n"
                "• Visit: https://ollama.ai\n"
                "• Or: curl https://ollama.ai/install.sh | sh",
                title="Ollama Not Found",
                style="red"
            ))
            sys.exit(1)
        
        # Check if Ollama is running
        is_healthy, message = check_ollama_health()
        if not is_healthy:
            console.print(Panel(
                f"❌ {message}\n\n"
                "Start Ollama:\n"
                "• Run: ollama serve\n"
                "• Or: systemctl start ollama",
                title="Ollama Not Running",
                style="red"
            ))
            sys.exit(1)
    
    # Check for models
    models = get_available_models()
    if not models:
        console.print(Panel(
            "⚠️ No models installed\n\n"
            "Would you like to install a recommended model?",
            title="No Models Found",
            style="yellow"
        ))
        
        # Recommend best model based on system
        recommended_model = "qwen2.5:3b"  # Small, fast, efficient
        
        if PROMPT_TOOLKIT_AVAILABLE:
            from prompt_toolkit import prompt as pt_prompt
            choice = pt_prompt(f"Install recommended model ({recommended_model})? [y/N]: ").lower()
            install_model = choice in ['y', 'yes']
        else:
            choice = input(f"Install recommended model ({recommended_model})? [y/N]: ").lower()
            install_model = choice in ['y', 'yes']
        
        if install_model:
            update_model(recommended_model)
            models = get_available_models()  # Refresh after install
        else:
            console.print("[yellow]You can install models later with: ollama pull <model>[/yellow]")
            sys.exit(0)
    
    # Save config with smart model selection
    config_dir = Path.home() / ".xencode"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.json"
    
    # Use smart model selection
    best_model = get_smart_default_model()
    
    config = {
        "default_model": best_model,
        "setup_completed": True,
        "setup_date": datetime.now().isoformat()
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    console.print(f"[green]✅ Setup complete! Using model: {best_model}[/green]\n")


def main():
    # Check for first run
    if is_first_run():
        run_first_time_setup()
    
    # Smart model selection - use best available model
    global DEFAULT_MODEL
    DEFAULT_MODEL = get_smart_default_model()
    
    args = sys.argv[1:]
    # Read online status from environment variable (set by xencode.sh)
    online = "true" if os.environ.get('XENCODE_ONLINE', 'false').lower() == 'true' else "false"
    
    # Check if we should force chat mode in current terminal (like Gemini/Crush/Claude CLI)
    force_chat = os.environ.get('XENCODE_FORCE_CHAT', 'false').lower() == 'true'
    chat_mode_enabled = force_chat or (len(args) == 0)

    # Initialize Enhanced CLI System if available
    enhanced_cli = None
    if ENHANCED_CLI_AVAILABLE and EnhancedXencodeCLI:
        try:
            enhanced_cli = EnhancedXencodeCLI()
        except Exception as e:
            # Enhanced CLI failed to initialize - continue with legacy mode
            console.print(
                f"[dim yellow]⚠️ Enhanced features unavailable: {e}[/dim yellow]"
            )

    # Try enhanced CLI processing first
    if enhanced_cli:
        try:
            # Create enhanced parser and parse arguments
            parser = enhanced_cli.create_parser()

            # Handle special case where no args means chat mode (legacy behavior)
            if not args:
                chat_mode_enabled = True
            else:
                # Parse enhanced arguments
                try:
                    parsed_args = parser.parse_args(args)

                    # Process enhanced commands
                    enhanced_result = enhanced_cli.process_enhanced_args(parsed_args)

                    if enhanced_result is not None:
                        # Enhanced command was processed
                        console.print(enhanced_result)
                        return

                    # Check if this should be handled as legacy command
                    if parsed_args.query:
                        # Legacy inline query mode
                        prompt = parsed_args.query
                        model = DEFAULT_MODEL

                        try:
                            response = run_query(model, prompt)
                            # For inline mode, only show the answer (no thinking section)
                            thinking, answer = extract_thinking_and_answer(response)
                            if answer.strip():
                                console.print(Markdown(answer.strip()))
                            else:
                                # Fallback to full response if no thinking tags
                                console.print(Markdown(response.strip()))
                        except Exception as e:
                            # Generic error panel for inline mode
                            error_panel = Panel(
                                f"❌ Unexpected error: {str(e)}\n\nPlease check your setup and try again.",
                                title="Error",
                                style="red",
                                border_style="red",
                            )
                            console.print(error_panel)
                        return

                except SystemExit:
                    # argparse called sys.exit (help or error)
                    return
                except Exception:
                    # Enhanced parsing failed - fall back to legacy parsing
                    pass
        except Exception:
            # Enhanced CLI processing failed - fall back to legacy
            pass

    # Legacy argument parsing (backward compatibility)
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

    # Auto-enable chat mode if no arguments (legacy behavior)
    if not args:
        chat_mode_enabled = True

    # Validate chat mode vs inline mode conflicts
    if (
        chat_mode_enabled
        and args
        and not any(flag in args for flag in ["--list-models", "--update", "-m"])
    ):
        error_panel = Panel(
            "❌ Invalid usage\n\nChat mode cannot be used with inline prompts.\n\nUse:\n• Chat mode: ./xencode.sh\n• Inline mode: ./xencode.sh \"your prompt\"",
            title="Usage Error",
            style="red",
            border_style="red",
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
                border_style="yellow",
            )
            console.print(warning_panel)
        return

    # Handle --status
    if "--status" in args:
        show_system_status(DEFAULT_MODEL, online)
        return

    # Handle --memory
    if "--memory" in args:
        show_memory_info()
        return

    # Handle --sessions
    if "--sessions" in args:
        show_sessions_list()
        return

    # Handle --cache
    if "--cache" in args:
        show_cache_info()
        return

    # Handle --export
    if "--export" in args:
        export_conversation()
        return

    # File operations
    if args and args[0] == 'file':
        if len(args) < 2:
            console.print(Panel("❌ No operation specified", style="red"))
            return
        op = args[1]
        if op == 'create' and len(args) >= 3:
            create_file(args[2], ' '.join(args[3:]))
        elif op == 'read' and len(args) >= 3:
            read_file(args[2])
        elif op == 'write' and len(args) >= 3:
            write_file(args[2], ' '.join(args[3:]))
        elif op == 'delete' and len(args) >= 3:
            delete_file(args[2])
        else:
            console.print(Panel(f"❌ Invalid operation: {op}", style="red"))
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
        # Health check before starting chat mode
        is_healthy, health_message = check_ollama_health()
        if not is_healthy:
            console.print(Panel(
                f"❌ {health_message}\n\n"
                "Please start Ollama:\n"
                "• Run: ollama serve\n"
                "• Or: systemctl start ollama",
                title="Ollama Not Available",
                style="red"
            ))
            return
        
        chat_mode(model, online)
        return
    else:
        # Handle inline prompt
        if not args:
            warning_panel = Panel(
                "⚠️ No prompt provided\n\nUsage:\n• Inline mode: ./xencode.sh \"your prompt\"\n• Chat mode: ./xencode.sh",
                title="Missing Prompt",
                style="yellow",
                border_style="yellow",
            )
            console.print(warning_panel)
            return

        prompt = " ".join(args)

        try:
            response = run_query(model, prompt)
            # For inline mode, only show the answer (no thinking section)
            thinking, answer = extract_thinking_and_answer(response)
            if answer.strip():
                console.print(Markdown(answer.strip()))
            else:
                # Fallback to full response if no thinking tags
                console.print(Markdown(response.strip()))
        except Exception as e:
            # Generic error panel for inline mode
            error_panel = Panel(
                f"❌ Unexpected error: {str(e)}\n\nPlease check your setup and try again.",
                title="Error",
                style="red",
                border_style="red",
            )
            console.print(error_panel)


if __name__ == "__main__":
    main()
