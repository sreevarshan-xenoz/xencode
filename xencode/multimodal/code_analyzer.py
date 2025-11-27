"""Code repository analyzer."""

from pathlib import Path
from typing import Dict, Any, List
import ast
import re


class CodeAnalyzer:
    """Analyze code repositories and files."""

    LANGUAGE_EXTENSIONS = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.go': 'Go',
        '.rs': 'Rust',
        '.rb': 'Ruby',
        '.php': 'PHP',
        '.cs': 'C#',
        '.swift': 'Swift',
        '.kt': 'Kotlin'
    }

    def analyze_directory(self, directory_path: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Analyze a code directory/repository.
        
        Args:
            directory_path: Path to the directory to analyze.
            max_depth: Maximum depth to traverse.
            
        Returns:
            Dictionary containing analysis results.
        """
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")

        result = {
            "path": str(path),
            "total_files": 0,
            "languages": {},
            "files_by_language": {},
            "total_lines": 0,
            "error": None
        }

        try:
            for file_path in path.rglob('*'):
                if file_path.is_file() and self._should_analyze(file_path):
                    result["total_files"] += 1
                    
                    ext = file_path.suffix.lower()
                    language = self.LANGUAGE_EXTENSIONS.get(ext, 'Unknown')
                    
                    if language not in result["languages"]:
                        result["languages"][language] = 0
                        result["files_by_language"][language] = []
                    
                    result["languages"][language] += 1
                    result["files_by_language"][language].append(str(file_path.relative_to(path)))
                    
                    # Count lines
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = len(f.readlines())
                            result["total_lines"] += lines
                    except Exception:
                        pass

        except Exception as e:
            result["error"] = str(e)

        return result

    def analyze_python_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Python file using AST.
        
        Args:
            file_path: Path to the Python file.
            
        Returns:
            Dictionary containing Python-specific analysis.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        result = {
            "filename": path.name,
            "path": str(path),
            "classes": [],
            "functions": [],
            "imports": [],
            "lines_of_code": 0,
            "error": None
        }

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                result["lines_of_code"] = len(content.splitlines())

            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    result["classes"].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    result["functions"].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        result["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        result["imports"].append(node.module)

        except SyntaxError as e:
            result["error"] = f"Syntax error: {str(e)}"
        except Exception as e:
            result["error"] = str(e)

        return result

    def _should_analyze(self, path: Path) -> bool:
        """Check if a file should be analyzed."""
        # Skip hidden files and common directories to ignore
        if any(part.startswith('.') for part in path.parts):
            return False
        
        ignore_dirs = {'__pycache__', 'node_modules', 'venv', 'env', 'dist', 'build'}
        if any(ignore_dir in path.parts for ignore_dir in ignore_dirs):
            return False
        
        return path.suffix.lower() in self.LANGUAGE_EXTENSIONS
