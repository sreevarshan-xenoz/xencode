"""Advanced tools for the agentic system."""

import json
from pathlib import Path
from typing import Optional, Type, List, Dict, Any

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from git import Repo, InvalidGitRepositoryError
from duckduckgo_search import DDGS

# Import existing code analyzer
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from multimodal.code_analyzer import CodeAnalyzer


# ============================================================================
# Git Operations Tools
# ============================================================================

class GitStatusSchema(BaseModel):
    repo_path: str = Field(description="Path to the git repository", default=".")


class GitStatusTool(BaseTool):
    name: str = "git_status"
    description: str = "Get the current git status of a repository (changed files, untracked files, etc.)"
    args_schema: Type[BaseModel] = GitStatusSchema

    def _run(self, repo_path: str = ".") -> str:
        try:
            repo = Repo(repo_path)
            
            status_info = []
            status_info.append(f"Branch: {repo.active_branch.name}")
            
            # Changed files
            changed_files = [item.a_path for item in repo.index.diff(None)]
            if changed_files:
                status_info.append(f"\nModified files: {', '.join(changed_files)}")
            
            # Untracked files
            untracked = repo.untracked_files
            if untracked:
                status_info.append(f"\nUntracked files: {', '.join(untracked)}")
            
            # Staged files
            staged = [item.a_path for item in repo.index.diff("HEAD")]
            if staged:
                status_info.append(f"\nStaged files: {', '.join(staged)}")
            
            if not changed_files and not untracked and not staged:
                status_info.append("\nWorking tree clean")
            
            return "\n".join(status_info)
        except InvalidGitRepositoryError:
            return f"Error: {repo_path} is not a git repository"
        except Exception as e:
            return f"Error getting git status: {str(e)}"


class GitDiffSchema(BaseModel):
    repo_path: str = Field(description="Path to the git repository", default=".")
    file_path: Optional[str] = Field(description="Specific file to diff (optional)", default=None)


class GitDiffTool(BaseTool):
    name: str = "git_diff"
    description: str = "Show git diff for unstaged changes or a specific file"
    args_schema: Type[BaseModel] = GitDiffSchema

    def _run(self, repo_path: str = ".", file_path: Optional[str] = None) -> str:
        try:
            repo = Repo(repo_path)
            
            if file_path:
                diff = repo.git.diff(file_path)
            else:
                diff = repo.git.diff()
            
            if not diff:
                return "No changes to show"
            
            return diff
        except Exception as e:
            return f"Error getting git diff: {str(e)}"


class GitLogSchema(BaseModel):
    repo_path: str = Field(description="Path to the git repository", default=".")
    max_count: int = Field(description="Maximum number of commits to show", default=10)


class GitLogTool(BaseTool):
    name: str = "git_log"
    description: str = "Show recent git commit history"
    args_schema: Type[BaseModel] = GitLogSchema

    def _run(self, repo_path: str = ".", max_count: int = 10) -> str:
        try:
            repo = Repo(repo_path)
            commits = list(repo.iter_commits(max_count=max_count))
            
            log_entries = []
            for commit in commits:
                log_entries.append(
                    f"Commit: {commit.hexsha[:8]}\n"
                    f"Author: {commit.author.name}\n"
                    f"Date: {commit.committed_datetime}\n"
                    f"Message: {commit.message.strip()}\n"
                )
            
            return "\n".join(log_entries)
        except Exception as e:
            return f"Error getting git log: {str(e)}"


class GitCommitSchema(BaseModel):
    repo_path: str = Field(description="Path to the git repository", default=".")
    message: str = Field(description="Commit message")
    files: Optional[List[str]] = Field(description="Specific files to commit (optional, commits all if not specified)", default=None)


class GitCommitTool(BaseTool):
    name: str = "git_commit"
    description: str = "Stage and commit changes to git repository"
    args_schema: Type[BaseModel] = GitCommitSchema

    def _run(self, repo_path: str = ".", message: str = "", files: Optional[List[str]] = None) -> str:
        try:
            repo = Repo(repo_path)
            
            # Stage files
            if files:
                repo.index.add(files)
            else:
                repo.git.add(A=True)
            
            # Commit
            commit = repo.index.commit(message)
            
            return f"Successfully committed: {commit.hexsha[:8]} - {message}"
        except Exception as e:
            return f"Error committing changes: {str(e)}"


# ============================================================================
# Web Search Tool
# ============================================================================

class WebSearchSchema(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(description="Maximum number of results to return", default=5)


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web using DuckDuckGo and return relevant results"
    args_schema: Type[BaseModel] = WebSearchSchema

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            if not results:
                return "No results found"
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   URL: {result.get('href', 'No URL')}\n"
                    f"   {result.get('body', 'No description')}\n"
                )
            
            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error performing web search: {str(e)}"


# ============================================================================
# Code Analysis Tool
# ============================================================================

class CodeAnalysisSchema(BaseModel):
    path: str = Field(description="Path to file or directory to analyze")
    analysis_type: str = Field(
        description="Type of analysis: 'directory' for full directory analysis, 'python' for Python file analysis",
        default="directory"
    )


class CodeAnalysisTool(BaseTool):
    name: str = "code_analysis"
    description: str = "Analyze code files or directories to extract structure, functions, classes, etc."
    args_schema: Type[BaseModel] = CodeAnalysisSchema

    def _run(self, path: str, analysis_type: str = "directory") -> str:
        try:
            analyzer = CodeAnalyzer()
            
            if analysis_type == "python":
                result = analyzer.analyze_python_file(path)
            else:
                result = analyzer.analyze_directory(path)
            
            # Format the result as readable text
            if result.get("error"):
                return f"Error: {result['error']}"
            
            if analysis_type == "python":
                output = [
                    f"File: {result['filename']}",
                    f"Lines of code: {result['lines_of_code']}",
                    f"\nClasses: {', '.join(result['classes']) if result['classes'] else 'None'}",
                    f"Functions: {', '.join(result['functions']) if result['functions'] else 'None'}",
                    f"Imports: {', '.join(result['imports']) if result['imports'] else 'None'}"
                ]
            else:
                output = [
                    f"Directory: {result['path']}",
                    f"Total files: {result['total_files']}",
                    f"Total lines: {result['total_lines']}",
                    f"\nLanguages:"
                ]
                for lang, count in result['languages'].items():
                    output.append(f"  {lang}: {count} files")
            
            return "\n".join(output)
        except Exception as e:
            return f"Error analyzing code: {str(e)}"


# ============================================================================
# Tool Registry
# ============================================================================

class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all default tools."""
        # Import base tools
        from .tools import ReadFileTool, WriteFileTool, ExecuteCommandTool
        
        default_tools = [
            # Base tools
            ReadFileTool(),
            WriteFileTool(),
            ExecuteCommandTool(),
            # Git tools
            GitStatusTool(),
            GitDiffTool(),
            GitLogTool(),
            GitCommitTool(),
            # Search tool
            WebSearchTool(),
            # Analysis tool
            CodeAnalysisTool(),
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get tools by category (git, file, web, etc.)."""
        if category == "git":
            return [t for t in self._tools.values() if t.name.startswith("git_")]
        elif category == "file":
            return [t for t in self._tools.values() if "file" in t.name]
        elif category == "web":
            return [t for t in self._tools.values() if "web" in t.name or "search" in t.name]
        elif category == "code":
            return [t for t in self._tools.values() if "code" in t.name]
        else:
            return []
