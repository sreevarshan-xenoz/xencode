import time
import logging
from typing import Optional
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from rich.console import Console
from rich.panel import Panel
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Lazy import for RAG to avoid circular deps if needed
try:
    from xencode.rag.vector_store import VectorStore
except ImportError:
    VectorStore = None

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ShadowMode")

class ShadowMind:
    """
    The intelligence behind Shadow Mode. 
    Analyzes code context and proposes 'next steps' or completions.
    """
    def __init__(self, model_name: str = "qwen2.5:14b", base_url: str = "http://localhost:11434"):
        # Uses a smarter model for reasoning if available, or falls back
        self.llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.3)
        self.vector_store = VectorStore() if VectorStore else None

    def think(self, file_path: str, content: str) -> Optional[str]:
        """
        Analyzes the file content and produces a suggestion.
        """
        # 1. RAG Context Retrieval (Concept: What else allows this code to work?)
        context_block = ""
        if self.vector_store:
            try:
                # Search for concepts related to the file's content
                # Taking the last 10 lines as "active context" for search query
                lines = content.splitlines()
                query = "\n".join(lines[-10:]) if lines else "code context"
                results = self.vector_store.similarity_search(query, k=2)
                context_block = "\n".join([d.page_content[:500] for d in results])
            except Exception:
                pass # Fail silently on RAG

        # 2. Prompting
        template = """You are 'Shadow Mode', an AI pair programmer running in the background.
Your goal is to predict the user's next intent or spot a potential bug in the code they are writing.

Current File ({filename}):
{file_content}

Relevant Context from Codebase:
{context}

Analyze the LAST part of the file (where the user likely is).
If you see a clear next step (e.g., implementing a method defined in an interface, handling an error, completing a pattern), suggest it.
If the code looks complete and correct, return NOTHING (empty string).
Do NOT be chatty. output ONLY the code or specific comment suggestion.

Suggestion:"""
        
        prompt = PromptTemplate.from_template(template)
        
        try:
            response = self.llm.invoke(prompt.format(
                filename=Path(file_path).name,
                file_content=content, # In production, we'd limit this window
                context=context_block
            ))
            return response.content.strip()
        except Exception as e:
            logger.error(f"ShadowMind thought failed: {e}")
            return None


class ShadowWatcher(FileSystemEventHandler):
    """
    Watches for file changes and triggers ShadowMind.
    """
    def __init__(self, mind: ShadowMind, debounce_seconds: float = 2.0):
        self.mind = mind
        self.debounce_seconds = debounce_seconds
        self.last_trigger = 0.0
        
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Simple debounce
        now = time.time()
        if now - self.last_trigger < self.debounce_seconds:
            return
        self.last_trigger = now
        
        # Determine if text file
        path = Path(event.src_path)
        if path.suffix not in ['.py', '.js', '.ts', '.md', '.txt']:
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            console.print(f"[dim]👁️ Shadow saw change in {path.name}... thinking...[/dim]")
            suggestion = self.mind.think(str(path), content)
            
            if suggestion:
                console.print(Panel(
                    suggestion,
                    title=f"👻 Shadow Suggestion for {path.name}",
                    border_style="purple",
                    subtitle="Apply this if it helps!"
                ))
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")

def start_shadow_mode(path: str = ".", model: str = "qwen2.5:14b"):
    """Starts the persistent shadow mode watcher."""
    mind = ShadowMind(model_name=model)
    event_handler = ShadowWatcher(mind)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    
    console.print(Panel(
        f"[bold purple]👻 Shadow Mode Active[/bold purple]\n"
        f"Watching {path} for changes...\n"
        "I will suggest completions when I see patterns.",
        border_style="purple"
    ))
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
