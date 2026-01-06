import sys
import platform
import subprocess
from typing import Optional, Tuple
from rich.console import Console
from rich.prompt import Confirm
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

console = Console()

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
        """Execute the command interactively"""
        if not command or command == "SAFE_GUARD_TRIGGERED":
            console.print("[red]❌ Command generation safe-guarded or failed.[/red]")
            return False

        console.print(f"\n[bold blue]Command:[/bold blue] [green]{command}[/green]")
        
        should_run = auto_confirm
        if not should_run:
             should_run = Confirm.ask("Execute this command?")
             
        if should_run:
            console.print("\n[dim]Output:[/dim]")
            try:
                # Use shell=True for shell commands
                subprocess.run(command, shell=True, check=True)
                return True
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Command failed with return code {e.returncode}[/red]")
                return False
            except Exception as e:
                console.print(f"[red]Execution failed: {e}[/red]")
                return False
        else:
            console.print("[yellow]Cancelled.[/yellow]")
            return False
