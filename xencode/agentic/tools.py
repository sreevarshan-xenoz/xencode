import os
import subprocess
from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


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
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60  # Safety timeout
            )
            output = result.stdout
            if result.stderr:
                output += f"\nStderr: {result.stderr}"
            return output
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error executing command: {str(e)}"
