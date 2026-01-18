"""
File operations module for Xencode
"""
import os
from pathlib import Path
from typing import Union

from rich.console import Console
from rich.panel import Panel

console = Console()

def create_file(path: Union[str, Path], content: str) -> None:
    """Create a file with the given content.

    Args:
        path: Path to the file to create
        content: Content to write to the file
    """
    try:
        abs_path = Path(path).resolve()
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        console.print(Panel(f"✅ {abs_path}", title="Created", style="green"))
    except (OSError, IOError, PermissionError) as e:
        console.print(Panel(f"❌ Failed: {type(e).__name__}", style="red"))


def read_file(path: Union[str, Path]) -> str:
    """Read the content of a file.

    Args:
        path: Path to the file to read

    Returns:
        Content of the file, or empty string if an error occurs
    """
    try:
        abs_path = Path(path).resolve()
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
        console.print(Panel(content, title=str(abs_path), style="cyan"))
        return content
    except (OSError, IOError, FileNotFoundError) as e:
        console.print(Panel(f"❌ Error: {type(e).__name__}", style="red"))
        return ""


def write_file(path: Union[str, Path], content: str) -> None:
    """Write content to a file.

    Args:
        path: Path to the file to write
        content: Content to write to the file
    """
    create_file(path, content)


def delete_file(path: Union[str, Path]) -> bool:
    """Delete a file.

    Args:
        path: Path to the file to delete

    Returns:
        True if the file was deleted successfully, False otherwise
    """
    try:
        abs_path = Path(path).resolve()
        os.remove(abs_path)
        console.print(Panel(f"✅ {abs_path}", title="Deleted", style="green"))
        return True
    except (OSError, FileNotFoundError, PermissionError) as e:
        console.print(Panel(f"❌ Failed: {type(e).__name__}", style="red"))
        return False