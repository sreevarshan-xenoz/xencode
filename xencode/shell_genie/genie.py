import platform
import re
import shlex
import subprocess
from typing import Tuple

from rich.console import Console
from rich.prompt import Confirm

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    PromptTemplate = None

console = Console()

# Dangerous command patterns — block even if LLM suggests them
_SHELL_DANGEROUS = [
    r'\brm\s+-rf?\s+/', r'\bmkfs\b', r'\bdd\s+if=', r'\bchmod\s+777\b',
    r'\bshutdown\b', r'\breboot\b', r'\bhalt\b', r':\(\)\{',
    r'`\s*rm\b', r'>\s*/dev/sd', r'wget.*\|\s*(bash|sh)', r'curl.*\|\s*(bash|sh)',
]

def _shell_command_is_safe(command: str) -> tuple:
    """Check if a shell command is safe to execute."""
    for pattern in _SHELL_DANGEROUS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Dangerous pattern detected: {pattern}"
    return True, "OK"

def _safe_execute(command: str) -> subprocess.CompletedProcess:
    """Execute command with shell=False using shlex splitting."""
    parts = shlex.split(command, posix=platform.system() != "Windows")
    return subprocess.run(parts, shell=False, capture_output=True, text=True, check=True, timeout=30)

class ShellGenie:
    """
    Translates natural language to shell commands.
    """

    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.1)
        self.os_info = f"{platform.system()} {platform.release()}"
        self.shell_type = "PowerShell" if platform.system() == "Windows" else "Bash"

    def generate_command(self, instruction: str) -> Tuple[str, str]:
        """
        Generates a shell command from instruction.
        Returns (command, explanation)
        """
        template = """You are an expert command line assistant for {os_info} using {shell_type}.

Instruction: {instruction}

Return a JSON object with two keys:
1. "command": The exact command to execute. DANGEROUS COMMANDS (rm -rf /, format, etc.) MUST BE PREVENTED. output "SAFE_GUARD_TRIGGERED" if dangerous.
2. "explanation": A brief explanation of what the command does.

JSON Response:"""

        prompt = PromptTemplate.from_template(template)

        # We need a robust way to get JSON.
        # For now, let's just ask for raw text and parse, or assume the model is good at JSON.
        # Llama 3 is usually good.

        try:
            response = self.llm.invoke(prompt.format(
                os_info=self.os_info,
                shell_type=self.shell_type,
                instruction=instruction
            ))
            content = response.content.strip()

            # Simple parsing (robust enough for specific models, minimal dependency)
            import json
            # Find JSON start/end
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = content[start:end]
                data = json.loads(json_str)
                return data.get("command", ""), data.get("explanation", "")
            else:
                return "", "Failed to parse model response"

        except Exception as e:
            return "", f"Error generating command: {e}"

    def execute(self, command: str, auto_confirm: bool = False) -> bool:
        """Execute the command interactively with safety checks"""
        if not command or command == "SAFE_GUARD_TRIGGERED":
            console.print("[red]❌ Command generation safe-guarded or failed.[/red]")
            return False

        # Double-check: validate the generated command
        is_safe, reason = _shell_command_is_safe(command)
        if not is_safe:
            console.print(f"[red]❌ Command blocked: {reason}[/red]")
            return False

        console.print(f"\n[bold blue]Command:[/bold blue] [green]{command}[/green]")

        should_run = auto_confirm
        if not should_run:
            should_run = Confirm.ask("Execute this command?")

        if should_run:
            console.print("\n[dim]Output:[/dim]")
            try:
                result = _safe_execute(command)
                if result.stdout:
                    console.print(result.stdout)
                if result.stderr:
                    console.print(f"[yellow]{result.stderr}[/yellow]")
                return True
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Command failed with return code {e.returncode}[/red]")
                if e.stderr:
                    console.print(f"[dim]{e.stderr}[/dim]")
                return False
            except subprocess.TimeoutExpired:
                console.print("[red]Command timed out[/red]")
                return False
            except Exception as e:
                console.print(f"[red]Execution failed: {type(e).__name__}: {e}[/red]")
                return False
        else:
            console.print("[yellow]Cancelled.[/yellow]")
            return False
