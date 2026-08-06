#!/usr/bin/env python3

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

# Import from new modular structure
from xencode.core import (
    ConversationMemory,
    ModelManager,
    ResponseCache,
    create_file,
    delete_file,
    get_available_models,
    get_smart_default_model,
    list_models,
    read_file,
    update_model,
    write_file,
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

# On Windows, force UTF-8 encoding to handle Unicode characters properly
if sys.platform.startswith('win'):
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.environ.setdefault('PYTHONUTF8', '1')

# Initialize Rich console with proper encoding handling for Windows
try:
    import sys

    # On Windows, handle encoding issues with Rich console
    if sys.platform.startswith('win'):
        # Force UTF-8 encoding for Rich console on Windows and disable problematic features
        console = Console(
            force_terminal=True,
            force_interactive=True,
            color_system="windows",
            legacy_windows=False,  # Important: Disable legacy Windows console
            encoding="utf-8",
            stderr=False,
            record=True  # Enable recording to handle encoding issues
        )
    else:
        console = Console(
            force_terminal=True,
            legacy_windows=False,
            color_system="256",
            stderr=False
        )
except Exception:
    # Fallback to basic console if there are issues
    console = Console(
        force_terminal=True,
        legacy_windows=False,
        color_system="256",
        stderr=False
    )

# Smart default model selection - will be updated based on available models
DEFAULT_MODEL = get_smart_default_model()  # Will be set dynamically

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


try:
    from xencode.tui.utils.model_checker import ModelChecker
    MODEL_CHECKER_AVAILABLE = True
except ImportError:
    MODEL_CHECKER_AVAILABLE = False
    ModelChecker = None


# ─── Local Overrides (override imports from xencode.core with Rich UI) ───

def get_available_models() -> List[str]:
    """Get available models with enhanced error handling and caching"""
    try:
        model_manager.refresh_models()
        return model_manager.available_models
    except Exception:
        return []


def get_smart_default_model() -> Optional[str]:  # noqa: C901 - intelligent fallback chain
    """Intelligently select the best available model"""
    available = []
    if MODEL_CHECKER_AVAILABLE:
        available = ModelChecker.get_available_models()
    if not available:
        try:
            from xencode.tui.utils.model_checker import ModelChecker as MC
            available = MC.get_available_models()
        except ImportError:
            pass
    if not available:
        try:
            output = subprocess.check_output(["ollama", "list"], text=True)
            if "NAME" in output:
                lines = output.strip().split('\n')
                available = [line.split()[0] for line in lines[1:] if line.strip()]
        except Exception:
            pass
    if not available:
        return None
    chat_models = [m for m in available if "embed" not in m]
    if not chat_models:
        return available[0]
    preferred_models = [
        "qwen2.5:7b", "qwen2.5:3b", "qwen3:4b",
        "llama3.1:8b", "llama3.2:3b", "mistral:7b", "phi3:mini", "gemma2:2b",
    ]
    for preferred in preferred_models:
        for available_model in chat_models:
            if preferred in available_model.lower():
                return available_model
    return chat_models[0]


def list_models() -> None:  # noqa: C901 - Rich UI with health checks
    """Enhanced model listing with health status and performance metrics"""
    try:
        model_manager.refresh_models()
        if not model_manager.available_models:
            console.print(Panel(
                "❌ No models found\n\nPlease install models with:\n• ollama pull qwen3:4b\n• ollama pull llama2\n• ollama pull mistral",
                title="No Models Available", style="red", border_style="red",
            ))
            return
        table = Table(title="📦 Installed Models", show_header=True, header_style="bold cyan")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Response Time", style="yellow")
        table.add_column("Last Check", style="dim")
        for model in model_manager.available_models:
            try:
                model_manager.check_model_health(model)
                health = model_manager.model_health.get(model, {})
                status = health.get('status', 'unknown')
                if status == 'healthy':
                    status_style = "OK Healthy"
                    response_time = f"{health.get('response_time', 0):.3f}s"
                elif status == 'error':
                    status_style = "ERROR Error"
                    response_time = "N/A"
                elif status == 'unavailable':
                    status_style = "WARNING Unavailable"
                    response_time = "N/A"
                else:
                    status_style = "❓ Unknown"
                    response_time = "N/A"
            except Exception:
                status_style = "❓ Check Failed"
                response_time = "N/A"
            last_check = health.get('last_check', 0) if 'health' in dir() else 0
            if isinstance(last_check, (int, float)) and last_check:
                last_check_str = datetime.fromtimestamp(last_check).strftime("%H:%M:%S")
            else:
                last_check_str = "Never"
            table.add_row(model, status_style, response_time, last_check_str)
        console.print(table)
        if model_manager.current_model:
            console.print(f"\n🎯 Current Model: [bold cyan]{model_manager.current_model}[/bold cyan]")
        if len(model_manager.available_models) == 1:
            console.print("\n💡 Tip: Install more models for variety:\n• ollama pull llama2:7b\n• ollama pull mistral:7b")
    except FileNotFoundError:
        console.print(Panel(
            "❌ Ollama not found\n\nPlease install Ollama:\n• Visit: https://ollama.ai\n• Or use your package manager",
            title="Missing Dependency", style="red", border_style="red",
        ))
    except Exception as e:
        console.print(Panel(
            f"❌ Error listing models: {str(e)}\n\nPlease check your Ollama installation.",
            title="Model List Error", style="red", border_style="red",
        ))


# ─── Helper Functions for Query Processing ─────────────────────────────

CLOUD_PROVIDERS = {
    "openai": {"config_key": "openai_api_key"},
    "google_gemini": {"config_key": "google_gemini_api_key"},
    "openrouter": {"config_key": "openrouter_api_key"},
    "qwen": {"config_key": None},
}


def _detect_cloud_provider(model: str) -> Optional[str]:
    """Detect if model name indicates a cloud provider. Returns provider name or None."""
    for prefix in CLOUD_PROVIDERS:
        if model.startswith(f"{prefix}:"):
            return prefix
    return None


def _handle_cloud_query(prompt: str, model: str, provider: str) -> Optional[str]:
    """Execute query against a cloud provider. Returns response or None on failure."""
    from xencode.model_providers import get_model_provider_manager
    from xencode.smart_config_manager import get_config

    provider_manager = get_model_provider_manager()
    config = get_config()
    provider_config = CLOUD_PROVIDERS.get(provider)

    if not provider_manager.providers and provider_config:
        config_key = provider_config["config_key"]
        if config_key:
            api_key = getattr(config.api_keys, config_key, None)
            if not api_key:
                return None
            provider_manager.configure_provider(provider, api_key)
        else:
            provider_manager.configure_provider(provider, "")

        asyncio.run(provider_manager.initialize_providers())

    model_name = model.replace(f"{provider}:", "")

    async def _call_provider():
        return await provider_manager.generate_with_provider(
            prompt, provider, model_name, max_tokens=2048, temperature=0.7
        )

    return asyncio.run(_call_provider())


def _build_context_prompt(prompt: str) -> str:
    """Build context-aware prompt with conversation memory and project context."""
    context = memory.get_context(max_messages=5)
    context_prompt = "\n\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in context]
    ) if context else ""

    project_info = ""
    if PROJECT_CONTEXT_AVAILABLE:
        try:
            project_ctx = get_project_context()
            if project_ctx.should_include_context(prompt):
                project_info = project_ctx.get_context_prompt()
        except Exception:
            pass

    if project_info and context_prompt:
        return f"{project_info}{context_prompt}\n\nuser: {prompt}"
    if project_info:
        return f"{project_info}user: {prompt}"
    if context_prompt:
        return f"{context_prompt}\n\nuser: {prompt}"
    return prompt


def _show_connection_error(title="Connection Error"):
    """Show a styled connection error panel."""
    error_panel = Panel(
        "❌ Cannot connect to Ollama service\n\nPlease check:\n• Is Ollama running? Try: systemctl start ollama\n• Is the service accessible at localhost:11434?",
        title=title,
        style="red",
        border_style="red",
    )
    console.print(error_panel)
    sys.exit(1)


def _show_timeout_error():
    """Show a styled timeout error panel."""
    error_panel = Panel(
        f"⏰ Request timed out after {RESPONSE_TIMEOUT}s\n\n"
        f"🔧 Try:\n• Using a smaller model\n• Checking system resources\n• Restarting Ollama service",
        title="Request Timeout",
        style="red",
        border_style="red",
    )
    console.print(error_panel)
    sys.exit(1)


def _show_api_error(error):
    """Show a styled API error panel."""
    error_panel = Panel(
        f"❌ API Error: {str(error)}\n\nPlease check your Ollama installation and try again.",
        title="API Error",
        style="red",
        border_style="red",
    )
    console.print(error_panel)
    sys.exit(1)


def _make_ollama_request(model: str, prompt: str) -> str:
    """Send a non-streaming request to Ollama and return the response."""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("🤖 Processing...", total=None)

        r = _post_ollama_request(url, payload)
        response = r.json()["response"]

        cache.set(prompt, model, response)
        memory.add_message("assistant", response, model)

        return response


def _post_ollama_request(url: str, payload: dict):
    """Send POST to Ollama with error handling."""
    try:
        r = requests.post(url, json=payload, timeout=RESPONSE_TIMEOUT)
        r.raise_for_status()
        return r
    except requests.exceptions.ConnectionError:
        _show_connection_error()
    except requests.exceptions.Timeout:
        _show_timeout_error()
    except requests.exceptions.RequestException as e:
        _show_api_error(e)



def run_query(model: str, prompt: str) -> str:
    """Enhanced non-streaming query with caching, conversation memory, and project context"""
    cached_response = cache.get(prompt, model)
    if cached_response:
        console.print("[dim]💾 Using cached response[/dim]")
        return cached_response

    memory.add_message("user", prompt, model)

    provider = _detect_cloud_provider(model)
    if provider:
        response = _handle_cloud_query(prompt, model, provider)
        if response:
            cache.set(prompt, model, response)
            memory.add_message("assistant", response, model)
            return response

    enhanced_prompt = _build_context_prompt(prompt)
    return _make_ollama_request(model, enhanced_prompt)


def run_streaming_query(model, prompt):
    """Enhanced REAL-TIME streaming query with conversation memory and context awareness"""
    memory.add_message("user", prompt, model)

    enhanced_prompt = _build_context_prompt(prompt)

    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": enhanced_prompt, "stream": True}

    try:
        response = requests.post(url, json=payload, stream=True, timeout=RESPONSE_TIMEOUT)
        response.raise_for_status()

        full_response = ""
        in_thinking = False
        thinking_shown = False
        answer_shown = False

        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if chunk.get('done', False):
                        break
                    if 'response' in chunk and chunk['response']:
                        token = chunk['response']
                        full_response += token
                        thinking_shown, answer_shown, in_thinking = _stream_token(
                            token, in_thinking, thinking_shown, answer_shown
                        )
                        sys.stdout.flush()
                except json.JSONDecodeError:
                    continue

        console.print()
        memory.add_message("assistant", full_response, model)
        return full_response

    except requests.exceptions.ConnectionError:
        _show_connection_error()
    except requests.exceptions.Timeout:
        _show_timeout_error()
    except requests.exceptions.RequestException as e:
        _show_api_error(e)


def _stream_token(token, in_thinking, thinking_shown, answer_shown):
    """Stream a single token with styling, returns updated state."""
    if '<think>' in token:
        if not thinking_shown:
            console.print("\n[bold cyan]Xencode[/bold cyan] [dim]›[/dim] [dim italic]thinking...[/dim italic]")
            thinking_shown = True
        in_thinking = True
    elif '</think>' in token:
        in_thinking = False
        if not answer_shown:
            console.print("\n\n[bold cyan]Xencode[/bold cyan] [dim]›[/dim]")
            answer_shown = True
    else:
        if in_thinking:
            console.print(token, style="dim italic yellow", end="", highlight=False)
        else:
            if not answer_shown:
                console.print("\n[bold cyan]Xencode[/bold cyan] [dim]›[/dim]")
                answer_shown = True
            console.print(token, end="", highlight=False)

    return thinking_shown, answer_shown, in_thinking


def extract_thinking_and_answer(text: str) -> Tuple[str, str]:
    """Extract thinking section and answer from Qwen response"""
    thinking = ""
    answer = text

    if "<think>" in text and "</think>" in text:
        try:
            thinking = text.split("<think>")[1].split("</think>")[0].strip()
            answer = text.split("</think>")[1].strip()
        except IndexError:
            pass
    elif "🧠 Thinking:" in text:
        try:
            thinking = text.split("🧠 Thinking:")[1].split("\n\n")[0].strip()
            answer = text.split("🧠 Thinking:")[1].split("\n\n", 1)[1].strip()
        except IndexError:
            pass

    return thinking, answer


def stream_thinking_section(thinking_text: str):
    """Stream thinking section with dim yellow italic styling and breathing pauses"""
    if not thinking_text:
        return

    console.print("🧠 Thinking...", style="dim italic yellow")

    lines = thinking_text.split('\n')
    for line in lines:
        if line.strip():
            for char in line:
                console.print(char, style="dim italic yellow", end="", highlight=False)
                time.sleep(THINKING_STREAM_DELAY)
            console.print()
            time.sleep(THINKING_LINE_PAUSE)


def stream_answer_section(answer_text: str):
    """Stream answer section with bold green styling and markdown support"""
    if not answer_text.strip():
        return

    console.print("\n📄 Answer", style="bold green")

    if answer_text.startswith("```") and "```" in answer_text[3:]:
        parts = answer_text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part.strip():
                    for char in part.strip():
                        console.print(char, end="", highlight=False)
                        time.sleep(ANSWER_STREAM_DELAY)
                    console.print()
            else:
                if part.strip():
                    lang = part.split('\n')[0] if '\n' in part else ""
                    code_content = part[part.find('\n') + 1:] if '\n' in part else part
                    console.print(Syntax(code_content, lang or "plaintext", theme="monokai"))
    else:
        for char in answer_text.strip():
            console.print(char, end="", highlight=False)
            time.sleep(ANSWER_STREAM_DELAY)
        console.print()


def stream_claude_response(thinking_text: str, answer_text: str):
    """Stream complete response with exact Claude timing and formatting"""
    if thinking_text:
        stream_thinking_section(thinking_text)
        time.sleep(THINKING_TO_ANSWER_PAUSE)
    stream_answer_section(answer_text)


def format_output(text, streaming=False):
    """Format output in Claude Code style with optional streaming"""
    thinking, answer = extract_thinking_and_answer(text)

    if streaming:
        stream_claude_response(thinking, answer)
    else:
        if thinking:
            console.print("[bold yellow]🧠 Thinking...[/bold yellow]")
            console.print(f"[dim]{thinking}[/dim]\n")

        if answer.strip():
            console.print("[bold green]📄 Answer[/bold green]")
            if answer.startswith("```") and "```" in answer[3:]:
                parts = answer.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        if part.strip():
                            console.print(Markdown(part.strip()))
                    else:
                        if part.strip():
                            lang = part.split('\n')[0] if '\n' in part else ""
                            code_content = part[part.find('\n') + 1:] if '\n' in part else part
                            console.print(Syntax(code_content, lang or "plaintext", theme="monokai"))
            else:
                console.print(Markdown(answer.strip()))
            console.print()


def display_chat_banner(model, online_status, is_update=False):
    """Display immersive full-screen banner (like Gemini/Crush/Claude CLI)"""
    if is_update:
        console.clear()

    console.print()
    console.print("╔═══════════════════════════════════════════════════════════════╗", style="bold cyan", justify="center")
    console.print("║                                                               ║", style="bold cyan", justify="center")
    console.print("║                    🤖 XENCODE AI ASSISTANT                    ║", style="bold cyan", justify="center")
    console.print("║                                                               ║", style="bold cyan", justify="center")
    console.print("╚═══════════════════════════════════════════════════════════════╝", style="bold cyan", justify="center")
    console.print()
    console.print(f"Model: {model}", style="bold white", justify="center")

    if online_status == "true":
        console.print("🌐 Online Mode", style="bold green", justify="center")
    else:
        console.print("📡 Offline Mode", style="bold yellow", justify="center")

    console.print()
    console.print("─" * 80, style="dim", justify="center")
    console.print()


def display_prompt():
    """Display the chat prompt with immersive styling"""
    console.print("\n[bold cyan]You[/bold cyan] [dim]›[/dim] ", end="")


def get_multiline_input():
    """Get user input with multiline support using prompt_toolkit if available"""
    if PROMPT_TOOLKIT_AVAILABLE:
        try:
            user_input = prompt("", multiline=True, wrap_lines=True, mouse_support=False)
            return user_input.strip() if user_input else ""
        except Exception as e:
            console.print(f"[dim]Warning: prompt_toolkit failed, using basic input: {e}[/dim]")
            return input().strip()
    return input().strip()


def update_online_status():
    """Check internet connectivity with a cross-platform socket probe"""
    try:
        with socket.create_connection(("8.8.8.8", 53), timeout=2):
            return "true"
    except OSError:
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


# ─── Legacy Command Handlers ─────────────────────────────────────────

def _cmd_help():
    show_help_panel()
    return True


def _cmd_clear():
    memory.start_session()
    console.print(Panel("🧹 Conversation cleared. New session started.", style="green"))
    return True


def _cmd_memory():
    show_memory_info()
    return True


def _cmd_sessions():
    show_sessions_list()
    return True


def _cmd_switch(cmd_parts):
    if len(cmd_parts) > 1:
        session_id = cmd_parts[1]
        if memory.switch_session(session_id):
            console.print(Panel(f"OK Switched to session: {session_id}", style="green"))
        else:
            console.print(Panel(f"❌ Session not found: {session_id}", style="red"))
    else:
        console.print(Panel("❌ Usage: /switch <session_id>", style="red"))
    return True


def _cmd_cache():
    show_cache_info()
    return True


def _cmd_status(current_model, current_online):
    show_system_status(current_model, current_online)
    return True


def _cmd_export():
    export_conversation()
    return True


def _cmd_model(cmd_parts, current_model):
    if len(cmd_parts) > 1:
        new_model = cmd_parts[1]
        success, message = model_manager.switch_model(new_model)
        if success:
            return "MODEL_SWITCH", new_model, message
        console.print(Panel(f"❌ Model switch failed: {message}", style="red"))
        return True
    console.print(Panel("❌ Usage: /model <model_name>", style="red"))
    return True


def _cmd_theme(cmd_parts):
    if len(cmd_parts) > 1:
        change_theme(cmd_parts[1])
    else:
        show_available_themes()
    return True


def _cmd_project():
    show_project_context()
    return True


def _cmd_models():
    show_available_models_interactive()
    return True


LEGACY_COMMANDS = {
    "help": lambda parts, *_: _cmd_help(),
    "clear": lambda parts, *_: _cmd_clear(),
    "memory": lambda parts, *_: _cmd_memory(),
    "sessions": lambda parts, *_: _cmd_sessions(),
    "switch": lambda parts, cur_model, cur_online: _cmd_switch(parts),
    "cache": lambda parts, *_: _cmd_cache(),
    "status": lambda parts, cur_model, cur_online: _cmd_status(cur_model, cur_online),
    "export": lambda parts, *_: _cmd_export(),
    "model": lambda parts, cur_model, cur_online: _cmd_model(parts, cur_model),
    "theme": lambda parts, *_: _cmd_theme(parts),
    "project": lambda parts, *_: _cmd_project(),
    "models": lambda parts, *_: _cmd_models(),
}


def _init_enhanced_commands():
    """Initialize and return EnhancedChatCommands if available."""
    if not ENHANCED_CLI_AVAILABLE:
        return None
    try:
        from enhanced_chat_commands import EnhancedChatCommands
        from enhanced_cli_system import FeatureDetector

        detector = FeatureDetector()
        features = detector.detect_features()
        enhanced_systems = {}
        if features.multi_model and MultiModelManager:
            enhanced_systems['multi_model'] = MultiModelManager()
        if features.smart_context and SmartContextManager:
            enhanced_systems['smart_context'] = SmartContextManager()
        if features.code_analysis and CodeAnalyzer:
            enhanced_systems['code_analyzer'] = CodeAnalyzer()

        return EnhancedChatCommands(features, enhanced_systems)
    except Exception:
        return None


def handle_chat_command(command, current_model, current_online):
    """Handle enhanced chat commands with rich formatting"""
    cmd_parts = command.split()
    cmd = cmd_parts[0].lower()

    # Try enhanced chat commands first
    enhanced_commands = _init_enhanced_commands()
    if enhanced_commands:
        try:
            enhanced_command_names = ["analyze", "model", "models", "context", "smart"]
            if cmd[1:] in enhanced_command_names:
                args = " ".join(cmd_parts[1:]) if len(cmd_parts) > 1 else ""
                response, new_model = enhanced_commands.handle_chat_command(cmd[1:], args, current_model)
                console.print(Panel(response, style="cyan"))
                if new_model:
                    return "MODEL_SWITCH", new_model, f"Enhanced model switch to {new_model}"
                return True

            if not command.startswith("/") and enhanced_commands.smart_mode_enabled:
                suggested_model, reasoning = enhanced_commands.suggest_model_for_query(command, current_model)
                if suggested_model != current_model:
                    console.print(f"[dim cyan]🤖 {reasoning}[/dim cyan]")
                    return "MODEL_SWITCH", suggested_model, reasoning
        except Exception as e:
            console.print(f"[dim yellow]⚠️ Enhanced commands unavailable: {e}[/dim yellow]")

    # Legacy chat commands via dispatch table
    handler = LEGACY_COMMANDS.get(cmd[1:])
    if handler:
        return handler(cmd_parts, current_model, current_online)

    return False


# ─── Info / Status Display Functions ─────────────────────────────────

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
        for i, msg in enumerate(context[-5:], 1):
            role = msg['role'].capitalize()
            content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
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

    table = Table(title="💬 Conversation Sessions", show_header=True, header_style="bold cyan")
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
            recent_files = sorted(cache_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            for i, cache_file in enumerate(recent_files, 1):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        prompt_preview = data.get('prompt', '')[:40] + "..." if len(data.get('prompt', '')) > 40 else data.get('prompt', '')
                        cache_text += f"• {i}. {prompt_preview}\n"
                except (OSError, json.JSONDecodeError, KeyError) as e:
                    cache_text += f"• {i}. [Error reading cache: {type(e).__name__}]\n"

        cache_panel = Panel(cache_text, title="💾 Cache Status", style="magenta")
        console.print(cache_panel)

    except Exception as e:
        console.print(Panel(f"❌ Error reading cache: {str(e)}", style="red"))


def show_system_status(current_model, current_online):
    """Display comprehensive system status"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_status = "OK Running" if response.status_code == 200 else "ERROR Error"
    except (requests.RequestException, OSError):
        ollama_status = "ERROR Not accessible"

    try:
        if model_manager.check_model_health(current_model):
            model_health = model_manager.model_health.get(current_model, {})
            response_time = f"{model_health.get('response_time', 0):.3f}s"
            model_status_display = f"✅ Healthy ({response_time})"
        else:
            model_health = model_manager.model_health.get(current_model, {})
            model_status_display = f"ERROR {model_health.get('status', 'Unavailable').capitalize()}"
    except Exception:
        model_status_display = "ERROR Check Failed"

    status_text = f"""
🖥️ **System Status:**
• Ollama Service: {ollama_status}
• Current Model: {current_model}
• Model Status: {model_status_display}
• Internet: {'ONLINE' if current_online == 'true' else 'OFFLINE'}
• Memory Usage: {len(memory.get_context())} messages
• Cache Status: {'OK Enabled' if CACHE_ENABLED else 'ERROR Disabled'}

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

        export_dir = Path.home() / ".xencode" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.md"
        export_path = export_dir / filename

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

        console.print(Panel(f"✅ Conversation exported to:\n{export_path}", style="green"))

    except Exception as e:
        console.print(Panel(f"❌ Export failed: {str(e)}", style="red"))


def change_theme(theme_name):
    """Change the visual theme (placeholder for future implementation)"""
    console.print(Panel(
        f"🎨 Theme '{theme_name}' not implemented yet.\n\nAvailable themes:\n• default\n• dark\n• light\n• colorful",
        style="yellow",
    ))


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
            "❌ Project context detection not available\n\nThis feature requires the project_context module.",
            title="Feature Not Available", style="red"
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
        console.print(Panel(f"❌ Failed to detect project context\n\nError: {str(e)}", title="Context Detection Error", style="red"))


def show_available_models_interactive():
    """Show available models with health status and allow switching"""
    models = get_available_models()

    if not models:
        console.print(Panel(
            "❌ No models found\n\nInstall a model with: ollama pull qwen2.5:3b",
            title="No Models Available", style="red"
        ))
        return

    table = Table(title="🤖 Available Models", show_header=True, header_style="bold cyan")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Model", style="white")
    table.add_column("Status", style="green")
    table.add_column("Response Time", style="yellow")
    table.add_column("Current", style="magenta")

    current_model = model_manager.current_model
    for i, model in enumerate(models, 1):
        model_manager.check_model_health(model)
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


# ─── Chat Mode ────────────────────────────────────────────────────────

def chat_mode(model, online):  # noqa: C901 - interactive chat loop
    """Enhanced interactive chat loop with immersive full-screen experience"""
    console.clear()
    session_id = memory.start_session()
    display_chat_banner(model, online)

    current_online = online
    current_model = model

    console.print(f"[dim]💬 Session: {session_id}[/dim]")
    console.print(f"[dim]🧠 Memory: {len(memory.get_context())} messages[/dim]")
    console.print()

    while True:
        try:
            display_prompt()
            user_input = get_multiline_input()

            if not user_input:
                console.print("[dim]Please enter a message or type 'exit' to quit.[/dim]")
                continue

            if user_input.startswith("/"):
                command_result = handle_chat_command(user_input, current_model, current_online)
                if command_result:
                    if isinstance(command_result, tuple) and command_result[0] == "MODEL_SWITCH":
                        _, new_model, message = command_result
                        current_model = new_model
                        console.print(Panel(f"✅ Model switched to [bold]{current_model}[/bold]\n{message}", style="green"))
                    continue

            if is_exit_command(user_input):
                handle_chat_exit()
                break

            console.print("\n[bold cyan]Xencode[/bold cyan] [dim]›[/dim] [dim italic]processing...[/dim italic]")

            new_online = update_online_status()
            if new_online != current_online:
                current_online = new_online
                display_chat_banner(current_model, current_online, is_update=True)
                console.print("[dim]🌐 Connection status updated[/dim]")
                console.print("[bold yellow]🧠 [Thinking...][/bold yellow]")

            try:
                run_streaming_query(current_model, user_input)
            except Exception as e:
                error_panel = Panel(f"❌ Error processing your request\n\n{str(e)}\n\nPlease try again or check your setup.",
                                    title="Processing Error", style="red", border_style="red")
                console.print(error_panel)

            console.print()

        except (KeyboardInterrupt, EOFError):
            handle_chat_exit()
            break
        except Exception as e:
            error_panel = Panel(f"❌ Unexpected error: {str(e)}\n\nThe chat session will continue. Please try again.",
                                title="Unexpected Error", style="red", border_style="red")
            console.print(error_panel)
            continue


# ─── Ollama Health & First-Run Setup ─────────────────────────────────

def check_ollama_health():  # noqa: C901 - multi-stage health check chain
    """Check if Ollama is running and accessible, with fallback auto-start"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True, "Ollama is running"
    except (requests.ConnectionError, requests.Timeout):
        pass
    except Exception:
        pass

    if OLLAMA_FALLBACK_AVAILABLE and OllamaFallbackManager:
        try:
            fallback = OllamaFallbackManager(auto_start=True, auto_install=False)
            if fallback.try_start_ollama():
                return True, "Ollama started successfully"
            if not fallback.is_ollama_installed():
                return False, ("Ollama is not installed.\n"
                               "Run 'xencode' with first-time setup or install manually from: https://ollama.ai")
        except Exception as e:
            console.print(f"[dim yellow]Fallback manager error: {e}[/dim yellow]")

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True, "Ollama is running"
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


def run_first_time_setup():  # noqa: C901 - interactive setup wizard
    """Run interactive setup for first-time users"""
    if FIRST_RUN_SETUP_AVAILABLE:
        try:
            setup = FirstRunSetup()
            setup.run_setup()
            config_file = Path.home() / ".xencode" / "model_config.json"
            if config_file.exists():
                return
        except Exception as e:
            console.print(f"[dim yellow]Advanced setup not available: {e}[/dim yellow]")

    console.print(Panel("👋 Welcome to Xencode!\n\nLet's get you set up in 30 seconds...",
                        title="First Run Setup", style="blue"))

    if OLLAMA_FALLBACK_AVAILABLE and OllamaFallbackManager:
        fallback = OllamaFallbackManager(auto_start=True, auto_install=False)
        success, message = fallback.ensure_ollama_available()
        if not success:
            console.print(Panel(f"❌ {message}", title="Ollama Setup Failed", style="red"))
            sys.exit(1)
        console.print(f"[green]✅ {message}[/green]")
    else:
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True, timeout=5)
        except (FileNotFoundError, subprocess.CalledProcessError):
            console.print(Panel(
                "❌ Ollama is not installed\n\nInstall Ollama:\n• Visit: https://ollama.ai\n• Or: curl https://ollama.ai/install.sh | sh",
                title="Ollama Not Found", style="red"
            ))
            sys.exit(1)

        is_healthy, message = check_ollama_health()
        if not is_healthy:
            console.print(Panel(f"❌ {message}\n\nStart Ollama:\n• Run: ollama serve\n• Or: systemctl start ollama",
                                title="Ollama Not Running", style="red"))
            sys.exit(1)

    models = get_available_models()
    if not models:
        console.print(Panel("⚠️ No models installed\n\nWould you like to install a recommended model?",
                            title="No Models Found", style="yellow"))
        recommended_model = "qwen2.5:3b"

        if PROMPT_TOOLKIT_AVAILABLE:
            from prompt_toolkit import prompt as pt_prompt
            choice = pt_prompt(f"Install recommended model ({recommended_model})? [y/N]: ").lower()
        else:
            choice = input(f"Install recommended model ({recommended_model})? [y/N]: ").lower()

        if choice in ['y', 'yes']:
            update_model(recommended_model)
            models = get_available_models()

    config_dir = Path.home() / ".xencode"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.json"
    best_model = get_smart_default_model()
    config = {
        "default_model": best_model,
        "setup_completed": True,
        "setup_date": datetime.now().isoformat()
    }
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    console.print(f"[green]✅ Setup complete! Using model: {best_model}[/green]\n")


# ─── Legacy Argument Parsing ──────────────────────────────────────────

def _parse_legacy_args(args):
    """Parse legacy CLI arguments, returns (prompt, model, online, chat_mode)."""
    online = os.environ.get('XENCODE_ONLINE', 'false').lower() == 'true'
    chat_mode_enabled = os.environ.get('XENCODE_FORCE_CHAT', 'false').lower() == 'true'
    prompt = None

    if "--online=true" in args:
        online = True
        args = [arg for arg in args if arg != "--online=true"]
    elif "--online=false" in args:
        online = False
        args = [arg for arg in args if arg != "--online=false"]

    if "--chat-mode" in args:
        chat_mode_enabled = True
        args = [arg for arg in args if arg != "--chat-mode"]

    if not args:
        chat_mode_enabled = True

    if args and not chat_mode_enabled and not any(
        flag in args for flag in ["--list-models", "--update", "-m"]
    ):
        prompt = args[0] if args else None

    return prompt, online, chat_mode_enabled


def _run_enhanced_cli(enhanced_cli, args):
    """Run the enhanced CLI if available. Returns True if handled."""
    if not enhanced_cli:
        return False
    try:
        parser = enhanced_cli.create_parser()
        if not args:
            return False
        parsed_args = parser.parse_args(args)
        enhanced_result = enhanced_cli.process_enhanced_args(parsed_args)
        if enhanced_result is not None:
            console.print(enhanced_result)
            return True
        if parsed_args.query:
            _run_inline_query(parsed_args.query)
            return True
    except SystemExit:
        return True
    except Exception:
        pass
    return False


def _run_inline_query(prompt):
    """Run an inline query and print the answer."""
    model = DEFAULT_MODEL
    try:
        response = run_query(model, prompt)
        thinking, answer = extract_thinking_and_answer(response)
        if answer.strip():
            console.print(Markdown(answer.strip()))
        else:
            console.print(Markdown(response.strip()))
    except Exception as e:
        error_panel = Panel(
            f"❌ Unexpected error: {str(e)}\n\nPlease check your setup and try again.",
            title="Error", style="red", border_style="red",
        )
        console.print(error_panel)


def _handle_list_models():
    """Handle --list-models flag."""
    list_models()
    return True


def _is_flag_or_model_arg(arg):
    """Check if an argument is a flag or model argument."""
    return any(arg.startswith(flag) for flag in ["--", "-"])


# ─── Main Entry Point ─────────────────────────────────────────────────

def main():
    if is_first_run():
        run_first_time_setup()

    global DEFAULT_MODEL
    DEFAULT_MODEL = get_smart_default_model()

    args = sys.argv[1:]
    online = "true" if os.environ.get('XENCODE_ONLINE', 'false').lower() == 'true' else "false"

    # Initialize Enhanced CLI System if available
    enhanced_cli = None
    if ENHANCED_CLI_AVAILABLE and EnhancedXencodeCLI:
        try:
            enhanced_cli = EnhancedXencodeCLI()
        except Exception as e:
            console.print(f"[dim yellow]⚠️ Enhanced features unavailable: {e}[/dim yellow]")

    # Try enhanced CLI processing first
    if _run_enhanced_cli(enhanced_cli, args):
        return

    # Fall back to legacy argument parsing
    prompt, online_bool, chat_mode_enabled = _parse_legacy_args(args)

    if chat_mode_enabled and args and not any(
        flag in args for flag in ["--list-models", "--update", "-m"]
    ):
        error_panel = Panel(
            "❌ Invalid usage\n\nChat mode cannot be used with inline prompts.\n\n"
            "Use:\n• Chat mode: ./xencode.sh\n• Inline mode: ./xencode.sh \"your prompt\"",
            title="Usage Error", style="red", border_style="red",
        )
        console.print(error_panel)
        return

    if online_bool:
        online = "true"

    if prompt:
        _run_inline_query(prompt)
    else:
        chat_mode(DEFAULT_MODEL, online)


if __name__ == "__main__":
    main()
