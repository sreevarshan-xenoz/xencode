import os
import re
import shlex
import subprocess
from typing import Type

try:
    from langchain.tools import BaseTool
except ImportError:
    try:
        from langchain_core.tools import BaseTool
    except ImportError:
        BaseTool = object

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object

    def Field(default=None, **kwargs):
        return default


# Allowed safe commands (allowlist approach)
ALLOWED_COMMANDS = frozenset({
    'ls', 'cat', 'head', 'tail', 'wc', 'grep', 'find', 'echo',
    'pwd', 'whoami', 'date', 'uname', 'df', 'du', 'free', 'top',
    'ps', 'ping', 'curl', 'wget', 'ssh', 'scp', 'rsync',
    'git', 'python', 'python3', 'pip', 'pip3', 'node', 'npm',
    'docker', 'make', 'cmake', 'gcc', 'g++', 'rustc', 'cargo',
    'mkdir', 'rm', 'cp', 'mv', 'chmod', 'chown', 'touch',
    'sed', 'awk', 'sort', 'uniq', 'cut', 'tr', 'xargs',
    'tree', 'file', 'stat', 'diff', 'patch',
})

# Dangerous patterns to block even in allowed commands
DANGEROUS_PATTERNS = [
    r'\brm\s+(-rf?|--no-preserve-root)\b',  # rm -rf
    r'\bmkfs\b',                             # format disk
    r'\bdd\s+if=',                           # raw disk write
    r'\bchmod\s+777\b',                      # overly permissive
    r';\s*(rm|mkfs|dd|shutdown|reboot|halt)\b',  # chained dangerous commands
    r'\|\s*(rm|mkfs|dd|shutdown|reboot|halt)\b', # piped dangerous commands
    r'`.*`',                                 # backtick substitution
    r'\$\(',                                 # command substitution
]


def _is_command_safe(command: str) -> tuple:
    """
    Validate a command against the allowlist and dangerous patterns.

    Returns:
        (is_safe: bool, reason: str)
    """
    # Check for dangerous patterns first
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Command contains dangerous pattern: {pattern}"

    # Parse command to get base executable
    try:
        parts = shlex.split(command, posix=os.name != 'nt')
    except ValueError:
        return False, "Command contains invalid quoting"

    if not parts:
        return False, "Empty command"

    # Allowlist check: base command must be in allowed set
    base = os.path.basename(parts[0])
    if base not in ALLOWED_COMMANDS:
        return False, f"Command '{base}' is not in the allowed command list"

    return True, "OK"


class ReadFileSchema(BaseModel):
    file_path: str = Field(description="The absolute path to the file to read")


class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Read the contents of a file from the local filesystem."
    args_schema: Type[BaseModel] = ReadFileSchema

    def _run(self, file_path: str) -> str:
        try:
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}"
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WriteFileSchema(BaseModel):
    file_path: str = Field(description="The absolute path to the file to write")
    content: str = Field(description="The content to write to the file")


class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "Write content to a file. Creates the file if it doesn't exist."
    args_schema: Type[BaseModel] = WriteFileSchema

    def _run(self, file_path: str, content: str) -> str:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class ExecuteCommandSchema(BaseModel):
    command: str = Field(description="The shell command to execute")


class ExecuteCommandTool(BaseTool):
    name: str = "execute_command"
    description: str = "Execute a shell command on the local system."
    args_schema: Type[BaseModel] = ExecuteCommandSchema

    def _run(self, command: str) -> str:
        # Security: validate command against allowlist
        is_safe, reason = _is_command_safe(command)
        if not is_safe:
            return f"Command blocked for security: {reason}"

        try:
            # Use shell=False with shlex.split for safety
            parts = shlex.split(command, posix=os.name != 'nt')
            result = subprocess.run(
                parts,
                shell=False,
                capture_output=True,
                text=True,
                timeout=60
            )
            output = result.stdout
            if result.stderr:
                output += f"\nStderr: {result.stderr}"
            return output
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except FileNotFoundError:
            return f"Error: Command not found: {command}"
        except Exception as e:
            return f"Error executing command: {type(e).__name__}: {e}"
