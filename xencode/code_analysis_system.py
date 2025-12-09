#!/usr/bin/env python3
"""
Code Analysis System for Xencode
Provides intelligent code analysis, review, and suggestions
"""

import ast
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List


class IssueType(Enum):
    """Types of code issues"""

    SYNTAX_ERROR = "syntax_error"
    STYLE_ISSUE = "style_issue"
    POTENTIAL_BUG = "potential_bug"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"


class Severity(Enum):
    """Issue severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CodeIssue:
    """A code issue found during analysis"""

    type: IssueType
    severity: Severity
    message: str
    file_path: str
    line_number: int
    column: int = 0
    suggestion: str = ""
    code_snippet: str = ""


class CodeAnalyzer:
    """Advanced code analysis system"""

    def __init__(self):
        self.supported_extensions = {
            '.py': self.analyze_python,
            '.js': self.analyze_javascript,
            '.ts': self.analyze_typescript,
            '.jsx': self.analyze_javascript,
            '.tsx': self.analyze_typescript,
        }

    def analyze_file(self, file_path: Path) -> List[CodeIssue]:
        """Analyze a single file"""
        if not file_path.exists():
            return []

        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            return []

        try:
            analyzer_func = self.supported_extensions[extension]
            return analyzer_func(file_path)
        except Exception as e:
            return [
                CodeIssue(
                    type=IssueType.SYNTAX_ERROR,
                    severity=Severity.HIGH,
                    message=f"Failed to analyze file: {str(e)}",
                    file_path=str(file_path),
                    line_number=1,
                )
            ]

    def analyze_python(self, file_path: Path) -> List[CodeIssue]:
        """Analyze Python code"""
        issues = []

        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            # Parse AST for syntax errors
            try:
                tree = ast.parse(content)
                issues.extend(self.analyze_python_ast(tree, file_path, lines, content))
            except SyntaxError as e:
                issues.append(
                    CodeIssue(
                        type=IssueType.SYNTAX_ERROR,
                        severity=Severity.CRITICAL,
                        message=f"Syntax error: {e.msg}",
                        file_path=str(file_path),
                        line_number=e.lineno or 1,
                        column=e.offset or 0,
                        code_snippet=(
                            lines[e.lineno - 1]
                            if e.lineno and e.lineno <= len(lines)
                            else ""
                        ),
                    )
                )

            # Style and quality checks
            issues.extend(self.check_python_style(content, lines, file_path))

        except Exception as e:
            issues.append(
                CodeIssue(
                    type=IssueType.SYNTAX_ERROR,
                    severity=Severity.HIGH,
                    message=f"Error reading file: {str(e)}",
                    file_path=str(file_path),
                    line_number=1,
                )
            )

        return issues

    def analyze_python_ast(
        self, tree: ast.AST, file_path: Path, lines: List[str], content: str
    ) -> List[CodeIssue]:
        """Analyze Python AST for issues"""
        issues = []

        class IssueVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check for missing docstrings
                if not ast.get_docstring(node):
                    issues.append(
                        CodeIssue(
                            type=IssueType.DOCUMENTATION,
                            severity=Severity.LOW,
                            message=f"Function '{node.name}' missing docstring",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            suggestion="Add a docstring to document the function's purpose",
                        )
                    )

                # Check for too many arguments
                if len(node.args.args) > 5:
                    issues.append(
                        CodeIssue(
                            type=IssueType.MAINTAINABILITY,
                            severity=Severity.MEDIUM,
                            message=f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            suggestion="Consider using a class or reducing parameters",
                        )
                    )

                self.generic_visit(node)

            def visit_ClassDef(self, node):
                # Check for missing docstrings
                if not ast.get_docstring(node):
                    issues.append(
                        CodeIssue(
                            type=IssueType.DOCUMENTATION,
                            severity=Severity.LOW,
                            message=f"Class '{node.name}' missing docstring",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            suggestion="Add a docstring to document the class's purpose",
                        )
                    )

                self.generic_visit(node)

            def visit_Import(self, node):
                # Check for unused imports (basic check)
                for alias in node.names:
                    module_name = alias.name
                    if module_name not in content:
                        issues.append(
                            CodeIssue(
                                type=IssueType.STYLE_ISSUE,
                                severity=Severity.LOW,
                                message=f"Potentially unused import: {module_name}",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                suggestion="Remove unused imports to keep code clean",
                            )
                        )

                self.generic_visit(node)

            def visit_Try(self, node):
                # Check for bare except clauses
                for handler in node.handlers:
                    if handler.type is None:
                        issues.append(
                            CodeIssue(
                                type=IssueType.POTENTIAL_BUG,
                                severity=Severity.MEDIUM,
                                message="Bare except clause catches all exceptions",
                                file_path=str(file_path),
                                line_number=handler.lineno,
                                suggestion="Specify exception types or use 'except Exception:'",
                            )
                        )

                self.generic_visit(node)

        visitor = IssueVisitor()
        visitor.visit(tree)

        return issues

    def check_python_style(
        self, content: str, lines: List[str], file_path: Path
    ) -> List[CodeIssue]:
        """Check Python style issues"""
        issues = []

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:  # Black's default
                issues.append(
                    CodeIssue(
                        type=IssueType.STYLE_ISSUE,
                        severity=Severity.LOW,
                        message=f"Line too long ({len(line)} > 88 characters)",
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line,
                        suggestion="Break long lines for better readability",
                    )
                )

            # Check for trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                issues.append(
                    CodeIssue(
                        type=IssueType.STYLE_ISSUE,
                        severity=Severity.LOW,
                        message="Trailing whitespace",
                        file_path=str(file_path),
                        line_number=i,
                        suggestion="Remove trailing whitespace",
                    )
                )

            # Check for print statements (potential debug code)
            if re.search(r'\bprint\s*\(', line) and not line.strip().startswith('#'):
                issues.append(
                    CodeIssue(
                        type=IssueType.MAINTAINABILITY,
                        severity=Severity.LOW,
                        message="Print statement found (potential debug code)",
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion="Consider using logging instead of print",
                    )
                )

        return issues

    def analyze_javascript(self, file_path: Path) -> List[CodeIssue]:
        """Analyze JavaScript code"""
        issues = []

        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            # Basic JavaScript checks
            issues.extend(self.check_javascript_style(content, lines, file_path))

        except Exception as e:
            issues.append(
                CodeIssue(
                    type=IssueType.SYNTAX_ERROR,
                    severity=Severity.HIGH,
                    message=f"Error reading file: {str(e)}",
                    file_path=str(file_path),
                    line_number=1,
                )
            )

        return issues

    def analyze_typescript(self, file_path: Path) -> List[CodeIssue]:
        """Analyze TypeScript code"""
        # For now, use JavaScript analysis
        return self.analyze_javascript(file_path)

    def check_javascript_style(
        self, content: str, lines: List[str], file_path: Path
    ) -> List[CodeIssue]:
        """Check JavaScript style issues"""
        issues = []

        for i, line in enumerate(lines, 1):
            # Check for console.log (potential debug code)
            if 'console.log' in line and not line.strip().startswith('//'):
                issues.append(
                    CodeIssue(
                        type=IssueType.MAINTAINABILITY,
                        severity=Severity.LOW,
                        message="console.log found (potential debug code)",
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion="Remove console.log statements before production",
                    )
                )

            # Check for var usage (prefer let/const)
            if re.search(r'\bvar\s+', line):
                issues.append(
                    CodeIssue(
                        type=IssueType.STYLE_ISSUE,
                        severity=Severity.LOW,
                        message="Use 'let' or 'const' instead of 'var'",
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion="Replace 'var' with 'let' or 'const' for better scoping",
                    )
                )

        return issues

    def analyze_directory(
        self, directory: Path, recursive: bool = True
    ) -> Dict[str, List[CodeIssue]]:
        """Analyze all supported files in a directory"""
        results = {}

        if not directory.exists() or not directory.is_dir():
            return results

        # Get all files to analyze
        if recursive:
            files = [f for f in directory.rglob("*") if f.is_file()]
        else:
            files = [f for f in directory.iterdir() if f.is_file()]

        # Filter for supported files
        supported_files = [
            f for f in files if f.suffix.lower() in self.supported_extensions
        ]

        # Analyze each file
        for file_path in supported_files:
            issues = self.analyze_file(file_path)
            if issues:
                results[str(file_path)] = issues

        return results

    def generate_report(self, analysis_results: Dict[str, List[CodeIssue]]) -> str:
        """Generate a formatted analysis report"""
        if not analysis_results:
            return "✅ No issues found!"

        report_lines = []
        report_lines.append("📊 Code Analysis Report")
        report_lines.append("=" * 50)

        # Summary
        total_issues = sum(len(issues) for issues in analysis_results.values())
        total_files = len(analysis_results)

        report_lines.append("\n📈 Summary:")
        report_lines.append(f"  • Files analyzed: {total_files}")
        report_lines.append(f"  • Total issues: {total_issues}")

        # Count by severity
        severity_counts = dict.fromkeys(Severity, 0)
        for issues in analysis_results.values():
            for issue in issues:
                severity_counts[issue.severity] += 1

        report_lines.append("\n🚨 By Severity:")
        for severity, count in severity_counts.items():
            if count > 0:
                emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                report_lines.append(
                    f"  • {emoji.get(severity.value, '⚪')} {severity.value.title()}: {count}"
                )

        # Detailed issues
        report_lines.append("\n📋 Detailed Issues:")

        for file_path, issues in analysis_results.items():
            report_lines.append(f"\n📄 {Path(file_path).name}:")

            for issue in issues:
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                }
                emoji = severity_emoji.get(issue.severity.value, "⚪")

                report_lines.append(
                    f"  {emoji} Line {issue.line_number}: {issue.message}"
                )

                if issue.code_snippet:
                    report_lines.append(f"     Code: {issue.code_snippet}")

                if issue.suggestion:
                    report_lines.append(f"     💡 {issue.suggestion}")

        return "\n".join(report_lines)

    def get_raw_git_diff(self, staged: bool = True) -> str:
        """
        Get the raw git diff output.

        Args:
            staged: if True, check staged changes (git diff --cached),
                    otherwise check unstaged changes (git diff)

        Returns:
            Raw diff string or empty string if no changes
        """
        try:
            cmd = ["git", "diff", "--cached"] if staged else ["git", "diff"]
            diff_result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )

            if diff_result.returncode != 0:
                # If staged failed or empty, maybe fallback?
                # For now just return empty if failed
                return ""

            return diff_result.stdout.strip()
        except Exception:
            return ""

    def get_diff_from_ref(self, ref: str) -> str:
        """
        Get diff against a specific reference
        
        Args:
            ref: Git reference (branch, commit, tag)
            
        Returns:
            Raw diff string or empty string if no changes
        """
        try:
            cmd = ["git", "diff", ref]
            diff_result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )
            return diff_result.stdout.strip()
        except Exception:
            return ""

    def generate_commit_message(self) -> str:
        """
        Generate intelligent commit message based on git diff analysis

        Returns:
            Generated commit message based on changes
        """
        try:
            # Try staged changes first
            diff_content = self.get_raw_git_diff(staged=True)

            if not diff_content:
                # Try unstaged changes
                diff_content = self.get_raw_git_diff(staged=False)
                
                if not diff_content:
                    return "No changes detected for commit message generation"

            # Analyze the diff to determine change type and scope
            change_analysis = self.analyze_git_diff(diff_content)

            # Generate commit message based on analysis
            commit_message = self._generate_commit_message_from_analysis(
                change_analysis
            )

            return commit_message

        except Exception as e:
            return f"Error generating commit message: {e}"

    def analyze_git_diff(self, diff_content: str) -> Dict[str, Any]:
        """
        Analyze git diff to understand the nature of changes

        Args:
            diff_content: Raw git diff output

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'files_changed': [],
            'change_types': set(),
            'languages': set(),
            'additions': 0,
            'deletions': 0,
            'is_new_file': False,
            'is_deleted_file': False,
            'has_tests': False,
            'has_docs': False,
            'scope': 'unknown',
        }

        lines = diff_content.split('\n')
        current_file = None

        for line in lines:
            # Track file changes
            if line.startswith('diff --git'):
                # Extract file path
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[3][2:]  # Remove 'b/' prefix
                    current_file = file_path
                    analysis['files_changed'].append(file_path)

                    # Determine language/file type
                    if file_path.endswith('.py'):
                        analysis['languages'].add('Python')
                    elif file_path.endswith(('.js', '.jsx')):
                        analysis['languages'].add('JavaScript')
                    elif file_path.endswith(('.ts', '.tsx')):
                        analysis['languages'].add('TypeScript')
                    elif file_path.endswith('.md'):
                        analysis['has_docs'] = True
                    elif 'test' in file_path.lower() or file_path.endswith('_test.py'):
                        analysis['has_tests'] = True

            # Track new/deleted files
            elif line.startswith('new file mode'):
                analysis['is_new_file'] = True
                analysis['change_types'].add('add')
            elif line.startswith('deleted file mode'):
                analysis['is_deleted_file'] = True
                analysis['change_types'].add('delete')

            # Count additions/deletions
            elif line.startswith('+') and not line.startswith('+++'):
                analysis['additions'] += 1
            elif line.startswith('-') and not line.startswith('---'):
                analysis['deletions'] += 1

            # Detect change types based on content
            elif line.startswith('+') or line.startswith('-'):
                content = line[1:].strip()

                # Function/method changes
                if re.match(r'^\s*(def|function|class)\s+', content):
                    analysis['change_types'].add('function')

                # Import changes
                elif re.match(r'^\s*(import|from|require)', content):
                    analysis['change_types'].add('import')

                # Configuration changes
                elif current_file and any(
                    config in current_file
                    for config in ['.json', '.yml', '.yaml', '.toml', '.ini']
                ):
                    analysis['change_types'].add('config')

                # Documentation changes
                elif re.match(r'^\s*(#|//|\*|""")', content):
                    analysis['change_types'].add('docs')

        # Determine scope
        if len(analysis['files_changed']) == 1:
            analysis['scope'] = 'single_file'
        elif len(analysis['files_changed']) <= 5:
            analysis['scope'] = 'small'
        else:
            analysis['scope'] = 'large'

        return analysis

    def parse_diff_changes(self, diff_content: str) -> Dict[str, set]:
        """
        Parse diff to find changed lines per file
        
        Returns:
            Dictionary mapping file paths to sets of changed line numbers (1-based)
        """
        changes = {}
        current_file = None
        current_line = 0
        
        for line in diff_content.split('\n'):
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 4:
                    # diff --git a/path b/path -> use b/path (new)
                    b_path = parts[3]
                    if b_path.startswith('b/'):
                        current_file = b_path[2:]
                        changes[current_file] = set()
                    else:
                        current_file = None
            
            elif line.startswith('@@'):
                # @@ -1,5 +10,5 @@
                # Parse +start,len
                match = re.search(r'\+(\d+)(?:,(\d+))?', line)
                if match:
                    current_line = int(match.group(1))
            
            elif line.startswith('+') and not line.startswith('+++'):
                if current_file:
                    changes[current_file].add(current_line)
                current_line += 1
            
            elif line.startswith(' ') and current_file:
                current_line += 1
                
        return changes

    def analyze_diff_context(self, diff_content: str) -> List[CodeIssue]:
        """
        Analyze changes in a git diff, filtering issues to changed lines
        
        Args:
            diff_content: Raw git diff output
            
        Returns:
            List of CodeIssues found in the changed lines
        """
        changed_lines = self.parse_diff_changes(diff_content)
        all_issues = []
        
        for file_path, lines in changed_lines.items():
             path_obj = Path(file_path)
             if not path_obj.exists():
                 continue
                 
             # Run full analysis on file
             file_issues = self.analyze_file(path_obj)
             
             # Filter based on changed lines
             relevant_issues = [
                 issue for issue in file_issues 
                 if issue.line_number in lines
             ]
             all_issues.extend(relevant_issues)
             
        return all_issues

    def _generate_commit_message_from_analysis(self, analysis: Dict[str, Any]) -> str:
        """
        Generate commit message based on change analysis

        Args:
            analysis: Results from _analyze_git_diff

        Returns:
            Generated commit message
        """
        change_types = analysis['change_types']
        files_changed = analysis['files_changed']
        languages = analysis['languages']

        # Determine commit type prefix
        if analysis['is_new_file']:
            prefix = "feat"
            action = "Add"
        elif analysis['is_deleted_file']:
            prefix = "feat"
            action = "Remove"
        elif 'function' in change_types:
            prefix = (
                "feat" if analysis['additions'] > analysis['deletions'] else "refactor"
            )
            action = "Update" if prefix == "refactor" else "Add"
        elif 'config' in change_types:
            prefix = "config"
            action = "Update"
        elif 'docs' in change_types or analysis['has_docs']:
            prefix = "docs"
            action = "Update"
        elif analysis['has_tests']:
            prefix = "test"
            action = (
                "Add" if analysis['additions'] > analysis['deletions'] else "Update"
            )
        elif 'import' in change_types:
            prefix = "deps"
            action = "Update"
        else:
            prefix = "feat" if analysis['additions'] > analysis['deletions'] else "fix"
            action = "Update"

        # Generate description
        if len(files_changed) == 1:
            file_name = Path(files_changed[0]).name
            description = f"{action.lower()} {file_name}"
        elif analysis['scope'] == 'small':
            if languages:
                lang = list(languages)[0]
                description = f"{action.lower()} {lang} components"
            else:
                description = f"{action.lower()} {len(files_changed)} files"
        else:
            description = f"{action.lower()} multiple components"

        # Add language context if relevant
        if len(languages) == 1 and analysis['scope'] != 'single_file':
            lang = list(languages)[0]
            description = f"{action.lower()} {lang} {description.split(' ', 1)[1]}"

        # Generate final commit message
        commit_message = f"{prefix}: {description}"

        # Add body with more details if significant changes
        if analysis['additions'] + analysis['deletions'] > 20:
            body_lines = []

            if analysis['additions'] > 0:
                body_lines.append(f"- {analysis['additions']} additions")
            if analysis['deletions'] > 0:
                body_lines.append(f"- {analysis['deletions']} deletions")

            if change_types:
                types_str = ", ".join(sorted(change_types))
                body_lines.append(f"- Changes: {types_str}")

            if body_lines:
                commit_message += "\n\n" + "\n".join(body_lines)

        return commit_message

    def analyze_code_quality(self, path: str) -> str:
        """
        Analyze code quality for a given path and return a formatted report.
        (Wrapper for CLI compatibility)
        """
        path_obj = Path(path)
        if path_obj.is_file():
            issues = self.analyze_file(path_obj)
            results = {str(path_obj): issues} if issues else {}
        else:
            results = self.analyze_directory(path_obj, recursive=True)
            
        return self.generate_report(results)


# CLI interface for code analysis
def analyze_code_command(path: str = ".", recursive: bool = True) -> str:
    """Command-line interface for code analysis"""
    analyzer = CodeAnalyzer()
    path_obj = Path(path)

    if path_obj.is_file():
        # Analyze single file
        issues = analyzer.analyze_file(path_obj)
        results = {str(path_obj): issues} if issues else {}
    else:
        # Analyze directory
        results = analyzer.analyze_directory(path_obj, recursive)

    return analyzer.generate_report(results)


# Example usage
def demo_code_analysis():
    """Demonstrate code analysis"""
    analyzer = CodeAnalyzer()

    print("🔍 Code Analysis System Demo")
    print("=" * 40)

    # Analyze current directory
    current_dir = Path(".")
    results = analyzer.analyze_directory(current_dir, recursive=False)

    report = analyzer.generate_report(results)
    print(report)


if __name__ == "__main__":
    demo_code_analysis()
