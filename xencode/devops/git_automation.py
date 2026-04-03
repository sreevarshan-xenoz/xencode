#!/usr/bin/env python3
"""
Deep Git Automation Engine

Safe stage/apply/commit with:
- Semantic commit generation
- Risk guardrails
- Change analysis
- Auto-commit message generation
"""

import asyncio
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from subprocess import PIPE, STDOUT

from rich.console import Console

console = Console()


class RiskLevel(Enum):
    """Risk levels for git operations"""
    SAFE = "safe"  # No risk
    LOW = "low"  # Minor changes
    MEDIUM = "medium"  # Review recommended
    HIGH = "high"  # Requires approval
    CRITICAL = "critical"  # Blocked without override


class ChangeType(Enum):
    """Types of changes"""
    NEW_FILE = "new_file"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    TYPE_CHANGED = "type_changed"


@dataclass
class GitChange:
    """Represents a single file change"""
    file_path: str
    change_type: ChangeType
    additions: int = 0
    deletions: int = 0
    diff: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "change_type": self.change_type.value,
            "additions": self.additions,
            "deletions": self.deletions,
            "risk_level": self.risk_level.value,
        }


@dataclass
class CommitResult:
    """Result of a commit operation"""
    success: bool
    commit_hash: Optional[str] = None
    message: str = ""
    files_changed: int = 0
    additions: int = 0
    deletions: int = 0
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "commit_hash": self.commit_hash,
            "message": self.message,
            "files_changed": self.files_changed,
            "additions": self.additions,
            "deletions": self.deletions,
            "error": self.error,
            "warnings": self.warnings,
        }


@dataclass
class GitStatus:
    """Git repository status"""
    branch: str
    ahead: int = 0
    behind: int = 0
    staged_changes: List[GitChange] = field(default_factory=list)
    unstaged_changes: List[GitChange] = field(default_factory=list)
    untracked_files: List[str] = field(default_factory=list)
    
    @property
    def is_clean(self) -> bool:
        """Check if working directory is clean"""
        return (
            len(self.staged_changes) == 0
            and len(self.unstaged_changes) == 0
            and len(self.untracked_files) == 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch": self.branch,
            "ahead": self.ahead,
            "behind": self.behind,
            "is_clean": self.is_clean,
            "staged_count": len(self.staged_changes),
            "unstaged_count": len(self.unstaged_changes),
            "untracked_count": len(self.untracked_files),
        }


class GitAutomation:
    """
    Git automation with safety guardrails
    
    Usage:
        git = GitAutomation(repo_path="/path/to/repo")
        status = await git.get_status()
        result = await git.semantic_commit("Added new feature")
    """
    
    # Risky file patterns
    RISKY_PATTERNS = [
        (r"\.env$", RiskLevel.HIGH),
        (r"credentials\.", RiskLevel.CRITICAL),
        (r"\.key$", RiskLevel.CRITICAL),
        (r"\.pem$", RiskLevel.CRITICAL),
        (r"secrets\.", RiskLevel.CRITICAL),
        (r"\.gitignore$", RiskLevel.MEDIUM),
        (r"\.gitattributes$", RiskLevel.MEDIUM),
        (r"package\.json$", RiskLevel.MEDIUM),
        (r"requirements\.txt$", RiskLevel.MEDIUM),
        (r"Cargo\.toml$", RiskLevel.MEDIUM),
    ]
    
    # Commit type prefixes for semantic commits
    COMMIT_TYPES = {
        "feat": "New features",
        "fix": "Bug fixes",
        "docs": "Documentation changes",
        "style": "Code style changes (formatting)",
        "refactor": "Code refactoring",
        "test": "Test additions/modifications",
        "chore": "Maintenance tasks",
        "perf": "Performance improvements",
        "ci": "CI/CD changes",
        "build": "Build system changes",
    }
    
    def __init__(
        self,
        repo_path: Optional[str] = None,
        auto_stage: bool = False,
        require_review: bool = True,
    ):
        """
        Initialize git automation
        
        Args:
            repo_path: Path to git repository (default: current directory)
            auto_stage: Automatically stage changes
            require_review: Require review before high-risk commits
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.auto_stage = auto_stage
        self.require_review = require_review
        self._git_executable = self._find_git()
    
    def _find_git(self) -> Optional[str]:
        """Find git executable"""
        import shutil
        return shutil.which("git")
    
    async def _run_git(
        self,
        *args: str,
        check: bool = True,
    ) -> Tuple[int, str, str]:
        """
        Run git command
        
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        if not self._git_executable:
            return -1, "", "Git not found"
        
        cmd = [self._git_executable, "-C", str(self.repo_path)] + list(args)
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=PIPE,
                stderr=STDOUT,
            )
            
            stdout, _ = await process.communicate()
            output = stdout.decode('utf-8', errors='replace')
            
            if check and process.returncode != 0:
                return process.returncode, "", output
            
            return process.returncode, output, ""
            
        except Exception as e:
            return -1, "", str(e)
    
    async def get_status(self) -> GitStatus:
        """Get repository status"""
        # Get current branch
        _, branch_output, _ = await self._run_git("branch", "--show-current")
        branch = branch_output.strip() or "HEAD"
        
        # Get remote tracking info
        _, remote_output, _ = await self._run_git(
            "rev-list", "--left-right", "--count", f"origin/{branch}...HEAD"
        )
        ahead, behind = 0, 0
        if remote_output.strip():
            parts = remote_output.strip().split()
            if len(parts) == 2:
                behind, ahead = map(int, parts)
        
        # Get staged changes
        _, staged_output, _ = await self._run_git(
            "diff", "--cached", "--name-status"
        )
        staged_changes = self._parse_changes(staged_output)
        
        # Get unstaged changes
        _, unstaged_output, _ = await self._run_git(
            "diff", "--name-status"
        )
        unstaged_changes = self._parse_changes(unstaged_output)
        
        # Get untracked files
        _, untracked_output, _ = await self._run_git(
            "ls-files", "--others", "--exclude-standard"
        )
        untracked_files = [
            f.strip() for f in untracked_output.strip().split('\n') if f.strip()
        ]
        
        return GitStatus(
            branch=branch,
            ahead=ahead,
            behind=behind,
            staged_changes=staged_changes,
            unstaged_changes=unstaged_changes,
            untracked_files=untracked_files,
        )
    
    def _parse_changes(self, output: str) -> List[GitChange]:
        """Parse git diff output into GitChange objects"""
        changes = []
        
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            status = parts[0]
            file_path = parts[1] if len(parts) > 1 else parts[0]
            
            # Determine change type
            if status.startswith('A'):
                change_type = ChangeType.NEW_FILE
            elif status.startswith('D'):
                change_type = ChangeType.DELETED
            elif status.startswith('R'):
                change_type = ChangeType.RENAMED
            elif status.startswith('T'):
                change_type = ChangeType.TYPE_CHANGED
            else:
                change_type = ChangeType.MODIFIED
            
            # Assess risk
            risk = self._assess_file_risk(file_path)
            
            changes.append(GitChange(
                file_path=file_path,
                change_type=change_type,
                risk_level=risk,
            ))
        
        return changes
    
    def _assess_file_risk(self, file_path: str) -> RiskLevel:
        """Assess risk level for a file"""
        for pattern, risk in self.RISKY_PATTERNS:
            if re.search(pattern, file_path, re.IGNORECASE):
                return risk
        
        # Check for deletion (higher risk)
        if file_path.startswith('D'):
            return RiskLevel.MEDIUM
        
        return RiskLevel.SAFE
    
    async def stage_all(self) -> int:
        """Stage all changes"""
        code, output, _ = await self._run_git("add", "-A")
        
        if code == 0:
            # Count staged files
            status = await self.get_status()
            return len(status.staged_changes)
        
        return 0
    
    async def stage_file(self, file_path: str) -> bool:
        """Stage specific file"""
        code, _, _ = await self._run_git("add", str(file_path))
        return code == 0
    
    async def unstage_all(self) -> bool:
        """Unstage all changes"""
        code, _, _ = await self._run_git("reset", "HEAD")
        return code == 0
    
    async def unstage_file(self, file_path: str) -> bool:
        """Unstage specific file"""
        code, _, _ = await self._run_git("reset", "HEAD", "--", str(file_path))
        return code == 0
    
    async def generate_commit_message(
        self,
        changes: Optional[List[GitChange]] = None,
        custom_context: Optional[str] = None,
    ) -> str:
        """
        Generate semantic commit message
        
        Args:
            changes: List of changes (default: staged changes)
            custom_context: Additional context for commit message
            
        Returns:
            Generated commit message
        """
        if changes is None:
            status = await self.get_status()
            changes = status.staged_changes
        
        if not changes:
            return "chore: no changes to commit"
        
        # Analyze changes
        added_files = [c for c in changes if c.change_type == ChangeType.NEW_FILE]
        deleted_files = [c for c in changes if c.change_type == ChangeType.DELETED]
        modified_files = [c for c in changes if c.change_type == ChangeType.MODIFIED]
        
        # Determine commit type
        if added_files and not deleted_files and not modified_files:
            commit_type = "feat"
            description = f"add {len(added_files)} new file(s)"
        elif deleted_files and not added_files and not modified_files:
            commit_type = "chore"
            description = f"remove {len(deleted_files)} file(s)"
        elif any("test" in c.file_path.lower() for c in changes):
            commit_type = "test"
            description = "add or update tests"
        elif any(".md" in c.file_path.lower() or ".rst" in c.file_path.lower() for c in changes):
            commit_type = "docs"
            description = "update documentation"
        else:
            commit_type = "fix" if len(changes) <= 2 else "refactor"
            description = f"update {len(changes)} file(s)"
        
        # Add context if provided
        if custom_context:
            description += f" - {custom_context}"
        
        # Build commit message
        message = f"{commit_type}: {description}"
        
        # Add file summary as body
        if len(changes) > 1:
            message += "\n\n"
            message += "Changed files:\n"
            for change in changes[:10]:  # Limit to 10 files
                message += f"- {change.file_path}\n"
            if len(changes) > 10:
                message += f"- ... and {len(changes) - 10} more\n"
        
        return message
    
    async def semantic_commit(
        self,
        message: Optional[str] = None,
        auto_generate: bool = True,
        context: Optional[str] = None,
        allow_risky: bool = False,
    ) -> CommitResult:
        """
        Create commit with semantic message
        
        Args:
            message: Custom commit message (auto-generated if None)
            auto_generate: Auto-generate message if None provided
            context: Context for message generation
            allow_risky: Allow committing risky files
            
        Returns:
            CommitResult with commit details
        """
        warnings = []
        
        # Check for risky files
        status = await self.get_status()
        
        for change in status.staged_changes:
            if change.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                if not allow_risky:
                    return CommitResult(
                        success=False,
                        error=f"Blocked: {change.file_path} is marked as {change.risk_level.value} risk",
                        warnings=warnings,
                    )
                else:
                    warnings.append(f"Committing high-risk file: {change.file_path}")
        
        # Auto-stage if enabled
        if self.auto_stage and not status.staged_changes:
            await self.stage_all()
            status = await self.get_status()
        
        if not status.staged_changes:
            return CommitResult(
                success=False,
                error="No changes to commit",
                warnings=warnings,
            )
        
        # Generate or use provided message
        if message is None and auto_generate:
            message = await self.generate_commit_message(context=context)
        elif message is None:
            return CommitResult(
                success=False,
                error="No commit message provided and auto-generate disabled",
                warnings=warnings,
            )
        
        # Create commit
        code, output, error = await self._run_git("commit", "-m", message)
        
        if code == 0:
            # Extract commit hash
            _, hash_output, _ = await self._run_git("rev-parse", "HEAD")
            commit_hash = hash_output.strip()[:7]
            
            # Get stats
            _, stats_output, _ = await self._run_git(
                "diff-tree", "--no-commit-id", "--stat", "-r", commit_hash
            )
            
            additions = deletions = 0
            for line in stats_output.split('\n'):
                if 'insertion' in line or 'deletion' in line:
                    match = re.search(r'(\d+) insertion', line)
                    if match:
                        additions = int(match.group(1))
                    match = re.search(r'(\d+) deletion', line)
                    if match:
                        deletions = int(match.group(1))
            
            return CommitResult(
                success=True,
                commit_hash=commit_hash,
                message=message,
                files_changed=len(status.staged_changes),
                additions=additions,
                deletions=deletions,
                warnings=warnings,
            )
        else:
            return CommitResult(
                success=False,
                error=error or output,
                warnings=warnings,
            )
    
    async def get_diff(self, file_path: Optional[str] = None) -> str:
        """Get diff for staged or specific file"""
        if file_path:
            _, output, _ = await self._run_git(
                "diff", "--cached", "--", str(file_path)
            )
        else:
            _, output, _ = await self._run_git("diff", "--cached")
        
        return output.strip()
    
    async def get_log(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent commit log"""
        _, output, _ = await self._run_git(
            "log", f"-{count}", "--format=%H|%s|%an|%ai",
        )
        
        commits = []
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split('|')
            if len(parts) >= 4:
                commits.append({
                    "hash": parts[0][:7],
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                })
        
        return commits
    
    async def create_branch(
        self,
        name: str,
        from_branch: Optional[str] = None,
    ) -> bool:
        """Create new branch"""
        args = ["checkout", "-b", name]
        if from_branch:
            args.append(from_branch)
        
        code, _, error = await self._run_git(*args)
        return code == 0
    
    async def switch_branch(self, branch: str) -> bool:
        """Switch to branch"""
        code, _, error = await self._run_git("checkout", branch)
        return code == 0
    
    async def get_current_branch(self) -> str:
        """Get current branch name"""
        _, output, _ = await self._run_git("branch", "--show-current")
        return output.strip() or "HEAD"


# Global instance
_git: Optional[GitAutomation] = None


def get_git_automation(
    repo_path: Optional[str] = None,
    auto_stage: bool = False,
) -> GitAutomation:
    """Get or create global git automation"""
    global _git
    if _git is None:
        _git = GitAutomation(repo_path=repo_path, auto_stage=auto_stage)
    return _git


# Convenience functions
async def git_status(repo_path: Optional[str] = None) -> GitStatus:
    """Get repository status"""
    git = get_git_automation(repo_path)
    return await git.get_status()


async def git_commit(
    message: Optional[str] = None,
    auto_generate: bool = True,
    repo_path: Optional[str] = None,
) -> CommitResult:
    """Create semantic commit"""
    git = get_git_automation(repo_path)
    return await git.semantic_commit(message=message, auto_generate=auto_generate)


if __name__ == "__main__":
    # Demo
    async def demo():
        console.print("[bold blue]Git Automation Demo[/bold blue]\n")
        
        git = GitAutomation()
        
        # Get status
        status = await git.get_status()
        console.print(f"[bold]Branch:[/bold] {status.branch}")
        console.print(f"[bold]Clean:[/bold] {status.is_clean}")
        console.print(f"[bold]Staged:[/bold] {len(status.staged_changes)}")
        console.print(f"[bold]Unstaged:[/bold] {len(status.unstaged_changes)}")
        console.print(f"[bold]Untracked:[/bold] {len(status.untracked_files)}")
        
        # Generate commit message
        if status.staged_changes:
            message = await git.generate_commit_message()
            console.print(f"\n[bold]Generated message:[/bold]\n{message}")
    
    asyncio.run(demo())
