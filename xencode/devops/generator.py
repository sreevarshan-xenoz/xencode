import os
from pathlib import Path
from typing import Dict, Optional, List
from rich.console import Console
from rich.prompt import Confirm
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

console = Console()

class DevOpsGenerator:
    """
    Generates infrastructure configuration files (Dockerfile, docker-compose.yml)
    based on project analysis.
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.2)
        
    def analyze_project(self, root_path: str = ".") -> Dict[str, str]:
        """Collects context from project files."""
        context = {}
        root = Path(root_path)
        
        # Check for Python
        if (root / "requirements.txt").exists():
            with open(root / "requirements.txt", "r") as f:
                context["requirements.txt"] = f.read()
        if (root / "pyproject.toml").exists():
            with open(root / "pyproject.toml", "r") as f:
                context["pyproject.toml"] = f.read()
                
        # Check for Node
        if (root / "package.json").exists():
            with open(root / "package.json", "r") as f:
                context["package.json"] = f.read()
                
        # Check for existing Dockerfile
        if (root / "Dockerfile").exists():
             context["existing_dockerfile"] = "Exists"
             
        return context

    def generate_dockerfile(self, context: Dict[str, str]) -> str:
        """Generates a Dockerfile based on context."""
        template = """You are a DevOps expert. Generate a production-ready Dockerfile for the following project structure.
        
Project Context:
{context_str}

Return ONLY the content of the Dockerfile. Do not include markdown formatting or explanations.
"""
        prompt = PromptTemplate.from_template(template)
        
        # Format context for prompt
        context_str = ""
        for filename, content in context.items():
            if filename == "existing_dockerfile": continue
            # Truncate large files
            display_content = content[:1000] + "..." if len(content) > 1000 else content
            context_str += f"--- {filename} ---\n{display_content}\n\n"
            
        if not context_str:
            return "# No dependency files found. Please create requirements.txt or package.json."

        response = self.llm.invoke(prompt.format(context_str=context_str))
        
        # Clean up code blocks if present
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)
            
        return content

    def generate_docker_compose(self, context: Dict[str, str]) -> str:
        """Generates a docker-compose.yml based on context."""
        template = """You are a DevOps expert. Generate a docker-compose.yml for this project.
Include services for the application and any likely databases (PostgreSQL/Redis) if implied by dependencies.

Project Context:
{context_str}

Return ONLY the content of the docker-compose.yml. Do not include markdown formatting.
"""
        prompt = PromptTemplate.from_template(template)
        
        context_str = ""
        for filename, content in context.items():
            if filename == "existing_dockerfile": continue
            display_content = content[:1000] + "..." if len(content) > 1000 else content
            context_str += f"--- {filename} ---\n{display_content}\n\n"

        response = self.llm.invoke(prompt.format(context_str=context_str))
        
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)
            
        return content

    def safe_write(self, filename: str, content: str) -> bool:
        """Writes file safely, asking for confirmation if it exists."""
        path = Path(filename)
        
        if path.exists():
            console.print(f"[yellow]⚠️  {filename} already exists.[/yellow]")
            if not Confirm.ask(f"Overwrite {filename}?", default=False):
                new_name = f"{filename}.generated"
                console.print(f"[blue]Saving as {new_name} instead.[/blue]")
                path = Path(new_name)
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            console.print(f"[green]✅ Saved to {path}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to write {filename}: {e}[/red]")
            return False
