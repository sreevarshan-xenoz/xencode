#!/usr/bin/env python3
"""
Project Context Detection for Xencode

Automatically detects project type and gathers relevant context.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


class ProjectContextManager:
    """Detect and manage project context"""

    def __init__(self):
        self.cwd = Path.cwd()
        self.context = None

    def detect_project(self) -> Dict:
        """Detect project type and gather context"""
        context = {
            "type": self._detect_type(),
            "files": self._get_relevant_files(),
            "git": self._get_git_info(),
            "dependencies": self._get_dependencies(),
        }

        self.context = context
        return context

    def _detect_type(self) -> str:
        """Detect project type from files"""
        if (self.cwd / "package.json").exists():
            return "javascript"
        elif (self.cwd / "requirements.txt").exists() or (
            self.cwd / "pyproject.toml"
        ).exists():
            return "python"
        elif (self.cwd / "Cargo.toml").exists():
            return "rust"
        elif (self.cwd / "go.mod").exists():
            return "go"
        elif (self.cwd / "pom.xml").exists():
            return "java"
        elif (self.cwd / "Gemfile").exists():
            return "ruby"
        elif (self.cwd / "composer.json").exists():
            return "php"
        return "unknown"

    def _get_relevant_files(self) -> List[str]:
        """Get list of recently modified files"""
        try:
            result = subprocess.run(
                ["git", "ls-files", "-m"],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                files = result.stdout.strip().split("\n")
                return [f for f in files if f][:10]
        except Exception:
            pass

        return []

    def _get_git_info(self) -> Dict:
        """Get git information"""
        try:
            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=2,
            )

            # Get status
            status_result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=2,
            )

            return {
                "branch": branch_result.stdout.strip()
                if branch_result.returncode == 0
                else "unknown",
                "status": status_result.stdout.strip()
                if status_result.returncode == 0
                else "",
                "has_changes": bool(status_result.stdout.strip()),
            }
        except Exception:
            return {"branch": "unknown", "status": "", "has_changes": False}

    def _get_dependencies(self) -> List[str]:
        """Get project dependencies"""
        deps = []

        # Python
        if (self.cwd / "requirements.txt").exists():
            try:
                with open(self.cwd / "requirements.txt") as f:
                    deps = [
                        line.strip().split("==")[0]
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ][:10]
            except Exception:
                pass

        # JavaScript
        elif (self.cwd / "package.json").exists():
            try:
                with open(self.cwd / "package.json") as f:
                    data = json.load(f)
                    deps = list(data.get("dependencies", {}).keys())[:10]
            except Exception:
                pass

        return deps

    def get_context_prompt(self) -> str:
        """Generate context prompt for AI"""
        if not self.context:
            self.detect_project()

        if self.context["type"] == "unknown":
            return ""

        prompt = "\n[Project Context]\n"
        prompt += f"Type: {self.context['type']}\n"

        if self.context["git"]["branch"] != "unknown":
            prompt += f"Git Branch: {self.context['git']['branch']}\n"

        if self.context["git"]["has_changes"]:
            prompt += "Status: Uncommitted changes\n"

        if self.context["files"]:
            prompt += f"Modified Files: {', '.join(self.context['files'][:5])}\n"

        if self.context["dependencies"]:
            prompt += f"Dependencies: {', '.join(self.context['dependencies'][:5])}\n"

        prompt += "[/Project Context]\n\n"

        return prompt

    def should_include_context(self, prompt: str) -> bool:
        """Determine if context should be included"""
        # Include context for code-related queries
        code_keywords = [
            "code",
            "function",
            "class",
            "bug",
            "error",
            "fix",
            "implement",
            "refactor",
            "test",
            "debug",
            "file",
            "project",
            "repository",
            "git",
        ]

        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in code_keywords)


# Global instance
_project_context = None


def get_project_context() -> ProjectContextManager:
    """Get or create global project context manager"""
    global _project_context
    if _project_context is None:
        _project_context = ProjectContextManager()
    return _project_context
