#!/usr/bin/env python3
"""
Code Analysis System for Xencode
Provides intelligent code analysis, review, and suggestions
"""

import ast
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

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
            return [CodeIssue(
                type=IssueType.SYNTAX_ERROR,
                severity=Severity.HIGH,
                message=f"Failed to analyze file: {str(e)}",
                file_path=str(file_path),
                line_number=1
            )]
    
    def analyze_python(self, file_path: Path) -> List[CodeIssue]:
        """Analyze Python code"""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Parse AST for syntax errors
            try:
                tree = ast.parse(content)
                issues.extend(self.analyze_python_ast(tree, file_path, lines))
            except SyntaxError as e:
                issues.append(CodeIssue(
                    type=IssueType.SYNTAX_ERROR,
                    severity=Severity.CRITICAL,
                    message=f"Syntax error: {e.msg}",
                    file_path=str(file_path),
                    line_number=e.lineno or 1,
                    column=e.offset or 0,
                    code_snippet=lines[e.lineno - 1] if e.lineno and e.lineno <= len(lines) else ""
                ))
            
            # Style and quality checks
            issues.extend(self.check_python_style(content, lines, file_path))
            
        except Exception as e:
            issues.append(CodeIssue(
                type=IssueType.SYNTAX_ERROR,
                severity=Severity.HIGH,
                message=f"Error reading file: {str(e)}",
                file_path=str(file_path),
                line_number=1
            ))
        
        return issues
    
    def analyze_python_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[CodeIssue]:
        """Analyze Python AST for issues"""
        issues = []
        
        class IssueVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check for missing docstrings
                if not ast.get_docstring(node):
                    issues.append(CodeIssue(
                        type=IssueType.DOCUMENTATION,
                        severity=Severity.LOW,
                        message=f"Function '{node.name}' missing docstring",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        suggestion="Add a docstring to document the function's purpose"
                    ))
                
                # Check for too many arguments
                if len(node.args.args) > 5:
                    issues.append(CodeIssue(
                        type=IssueType.MAINTAINABILITY,
                        severity=Severity.MEDIUM,
                        message=f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        suggestion="Consider using a class or reducing parameters"
                    ))
                
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Check for missing docstrings
                if not ast.get_docstring(node):
                    issues.append(CodeIssue(
                        type=IssueType.DOCUMENTATION,
                        severity=Severity.LOW,
                        message=f"Class '{node.name}' missing docstring",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        suggestion="Add a docstring to document the class's purpose"
                    ))
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for unused imports (basic check)
                for alias in node.names:
                    module_name = alias.name
                    if module_name not in content:
                        issues.append(CodeIssue(
                            type=IssueType.STYLE_ISSUE,
                            severity=Severity.LOW,
                            message=f"Potentially unused import: {module_name}",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            suggestion="Remove unused imports to keep code clean"
                        ))
                
                self.generic_visit(node)
            
            def visit_Try(self, node):
                # Check for bare except clauses
                for handler in node.handlers:
                    if handler.type is None:
                        issues.append(CodeIssue(
                            type=IssueType.POTENTIAL_BUG,
                            severity=Severity.MEDIUM,
                            message="Bare except clause catches all exceptions",
                            file_path=str(file_path),
                            line_number=handler.lineno,
                            suggestion="Specify exception types or use 'except Exception:'"
                        ))
                
                self.generic_visit(node)
        
        visitor = IssueVisitor()
        visitor.visit(tree)
        
        return issues
    
    def check_python_style(self, content: str, lines: List[str], file_path: Path) -> List[CodeIssue]:
        """Check Python style issues"""
        issues = []
        
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:  # Black's default
                issues.append(CodeIssue(
                    type=IssueType.STYLE_ISSUE,
                    severity=Severity.LOW,
                    message=f"Line too long ({len(line)} > 88 characters)",
                    file_path=str(file_path),
                    line_number=i,
                    code_snippet=line,
                    suggestion="Break long lines for better readability"
                ))
            
            # Check for trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                issues.append(CodeIssue(
                    type=IssueType.STYLE_ISSUE,
                    severity=Severity.LOW,
                    message="Trailing whitespace",
                    file_path=str(file_path),
                    line_number=i,
                    suggestion="Remove trailing whitespace"
                ))
            
            # Check for print statements (potential debug code)
            if re.search(r'\bprint\s*\(', line) and not line.strip().startswith('#'):
                issues.append(CodeIssue(
                    type=IssueType.MAINTAINABILITY,
                    severity=Severity.LOW,
                    message="Print statement found (potential debug code)",
                    file_path=str(file_path),
                    line_number=i,
                    code_snippet=line.strip(),
                    suggestion="Consider using logging instead of print"
                ))
        
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
            issues.append(CodeIssue(
                type=IssueType.SYNTAX_ERROR,
                severity=Severity.HIGH,
                message=f"Error reading file: {str(e)}",
                file_path=str(file_path),
                line_number=1
            ))
        
        return issues
    
    def analyze_typescript(self, file_path: Path) -> List[CodeIssue]:
        """Analyze TypeScript code"""
        # For now, use JavaScript analysis
        return self.analyze_javascript(file_path)
    
    def check_javascript_style(self, content: str, lines: List[str], file_path: Path) -> List[CodeIssue]:
        """Check JavaScript style issues"""
        issues = []
        
        for i, line in enumerate(lines, 1):
            # Check for console.log (potential debug code)
            if 'console.log' in line and not line.strip().startswith('//'):
                issues.append(CodeIssue(
                    type=IssueType.MAINTAINABILITY,
                    severity=Severity.LOW,
                    message="console.log found (potential debug code)",
                    file_path=str(file_path),
                    line_number=i,
                    code_snippet=line.strip(),
                    suggestion="Remove console.log statements before production"
                ))
            
            # Check for var usage (prefer let/const)
            if re.search(r'\bvar\s+', line):
                issues.append(CodeIssue(
                    type=IssueType.STYLE_ISSUE,
                    severity=Severity.LOW,
                    message="Use 'let' or 'const' instead of 'var'",
                    file_path=str(file_path),
                    line_number=i,
                    code_snippet=line.strip(),
                    suggestion="Replace 'var' with 'let' or 'const' for better scoping"
                ))
        
        return issues
    
    def analyze_directory(self, directory: Path, recursive: bool = True) -> Dict[str, List[CodeIssue]]:
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
            f for f in files 
            if f.suffix.lower() in self.supported_extensions
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
        
        report_lines.append(f"\n📈 Summary:")
        report_lines.append(f"  • Files analyzed: {total_files}")
        report_lines.append(f"  • Total issues: {total_issues}")
        
        # Count by severity
        severity_counts = {severity: 0 for severity in Severity}
        for issues in analysis_results.values():
            for issue in issues:
                severity_counts[issue.severity] += 1
        
        report_lines.append(f"\n🚨 By Severity:")
        for severity, count in severity_counts.items():
            if count > 0:
                emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                report_lines.append(f"  • {emoji.get(severity.value, '⚪')} {severity.value.title()}: {count}")
        
        # Detailed issues
        report_lines.append(f"\n📋 Detailed Issues:")
        
        for file_path, issues in analysis_results.items():
            report_lines.append(f"\n📄 {Path(file_path).name}:")
            
            for issue in issues:
                severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                emoji = severity_emoji.get(issue.severity.value, "⚪")
                
                report_lines.append(f"  {emoji} Line {issue.line_number}: {issue.message}")
                
                if issue.code_snippet:
                    report_lines.append(f"     Code: {issue.code_snippet}")
                
                if issue.suggestion:
                    report_lines.append(f"     💡 {issue.suggestion}")
        
        return "\n".join(report_lines)

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