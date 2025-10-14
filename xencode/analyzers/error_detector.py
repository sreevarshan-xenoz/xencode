#!/usr/bin/env python3
"""
Error Detector

Detects common programming errors, bugs, and code quality issues.
Provides comprehensive error detection for multiple programming languages.
"""

import re
from typing import Dict, List, Optional, Tuple

from xencode.models.code_analysis import (
    AnalysisIssue,
    AnalysisType,
    CodeLocation,
    Language,
    SeverityLevel
)


class ErrorDetector:
    """Detects common programming errors and bugs"""
    
    def __init__(self):
        # Error patterns for different languages
        self.error_patterns = {
            Language.PYTHON: {
                'logic_errors': [
                    (r'if.*=.*:', 'Assignment in if condition (use == for comparison)'),
                    (r'while.*=.*:', 'Assignment in while condition (use == for comparison)'),
                    (r'elif.*=.*:', 'Assignment in elif condition (use == for comparison)'),
                    (r'is\s+True', 'Use "is True" only for singleton comparison'),
                    (r'is\s+False', 'Use "is False" only for singleton comparison'),
                    (r'==\s+None', 'Use "is None" instead of "== None"'),
                    (r'!=\s+None', 'Use "is not None" instead of "!= None"')
                ],
                'resource_leaks': [
                    (r'open\s*\([^)]*\)(?!\s*as\s+\w+)(?!\s*\.)', 'File opened without context manager'),
                    (r'\.close\(\)', 'Manual close() call - consider using context manager')
                ],
                'exception_handling': [
                    (r'except\s*:', 'Bare except clause catches all exceptions'),
                    (r'except\s+Exception\s*:', 'Catching Exception is too broad'),
                    (r'pass\s*$', 'Empty except block silently ignores errors'),
                    (r'raise\s+Exception\s*\(', 'Raising generic Exception - use specific exception type')
                ],
                'naming_issues': [
                    (r'def\s+[A-Z]', 'Function name should be lowercase with underscores'),
                    (r'class\s+[a-z]', 'Class name should use CapitalCase'),
                    (r'^\s*[A-Z][A-Z_]*\s*=', 'Constants should be ALL_CAPS')
                ],
                'performance_issues': [
                    (r'\+\s*=.*\[\]', 'List concatenation in loop - consider list comprehension'),
                    (r'for.*in\s+range\(len\(', 'Use enumerate() instead of range(len())'),
                    (r'\.keys\(\)\s*:', 'Iterating over dict.keys() - iterate over dict directly')
                ]
            },
            Language.JAVASCRIPT: {
                'logic_errors': [
                    (r'if\s*\(.*=.*\)', 'Assignment in if condition (use === for comparison)'),
                    (r'while\s*\(.*=.*\)', 'Assignment in while condition (use === for comparison)'),
                    (r'==\s*null', 'Use strict equality (=== null) or check for null/undefined'),
                    (r'!=\s*null', 'Use strict inequality (!== null) or check for null/undefined'),
                    (r'typeof.*==\s*["\']undefined["\']', 'Use === for typeof comparisons')
                ],
                'async_issues': [
                    (r'async\s+function.*(?!await)', 'Async function without await usage'),
                    (r'\.then\(.*\.catch\(', 'Promise chain without proper error handling'),
                    (r'new\s+Promise\s*\(.*(?!resolve|reject)', 'Promise constructor without resolve/reject')
                ],
                'scope_issues': [
                    (r'var\s+\w+.*for\s*\(', 'var in loop can cause closure issues'),
                    (r'function.*\{.*var.*\}', 'var declaration in function - consider let/const'),
                    (r'this\..*=.*function', 'Function assignment loses this context')
                ],
                'type_issues': [
                    (r'parseInt\([^,)]*\)', 'parseInt without radix parameter'),
                    (r'isNaN\(', 'isNaN has confusing behavior - use Number.isNaN()'),
                    (r'new\s+Array\(\d+\)', 'Array constructor with single number argument')
                ],
                'performance_issues': [
                    (r'document\.getElementById.*loop', 'DOM query in loop - cache the result'),
                    (r'\.innerHTML\s*\+=', 'innerHTML concatenation is inefficient'),
                    (r'for.*\.length', 'Cache array length in for loop')
                ]
            },
            Language.JAVA: {
                'logic_errors': [
                    (r'if\s*\(.*=.*\)', 'Assignment in if condition (use == for comparison)'),
                    (r'\.equals\(.*null', 'Calling equals on potentially null object'),
                    (r'==.*String', 'String comparison with == instead of .equals()'),
                    (r'catch.*Exception.*\{\s*\}', 'Empty catch block ignores exceptions')
                ],
                'resource_leaks': [
                    (r'new\s+FileInputStream.*(?!try)', 'FileInputStream not in try-with-resources'),
                    (r'new\s+BufferedReader.*(?!try)', 'BufferedReader not in try-with-resources'),
                    (r'\.close\(\)', 'Manual close() call - use try-with-resources')
                ],
                'concurrency_issues': [
                    (r'synchronized.*static', 'Synchronizing on static method/block'),
                    (r'volatile.*(?!final)', 'volatile without proper synchronization'),
                    (r'Thread\.sleep\(', 'Thread.sleep() can be interrupted')
                ]
            }
        }
        
        # Common anti-patterns across languages
        self.common_antipatterns = {
            'magic_numbers': r'\b\d{2,}\b(?!\s*[;,\)])',  # Numbers with 2+ digits not at end of statement
            'long_lines': r'.{120,}',  # Lines longer than 120 characters
            'deep_nesting': r'^\s{20,}',  # Lines with 20+ spaces (5+ levels of nesting)
            'commented_code': r'^\s*#.*[=\(\)\[\]{}]',  # Commented out code patterns
        }
    
    async def detect_errors(self, 
                           code: str, 
                           language: Language,
                           file_path: str = "") -> List[AnalysisIssue]:
        """Detect errors and issues in code"""
        
        issues = []
        lines = code.split('\n')
        
        # Language-specific error detection
        if language in self.error_patterns:
            patterns = self.error_patterns[language]
            
            for category, pattern_list in patterns.items():
                for pattern, description in pattern_list:
                    category_issues = await self._find_error_pattern(
                        lines, pattern, description, category, file_path
                    )
                    issues.extend(category_issues)
        
        # Common anti-pattern detection
        common_issues = await self._detect_common_antipatterns(lines, file_path)
        issues.extend(common_issues)
        
        # Language-specific additional checks
        if language == Language.PYTHON:
            issues.extend(await self._detect_python_specific_errors(lines, file_path))
        elif language == Language.JAVASCRIPT:
            issues.extend(await self._detect_javascript_specific_errors(lines, file_path))
        
        return issues
    
    async def _find_error_pattern(self, 
                                 lines: List[str], 
                                 pattern: str, 
                                 description: str,
                                 category: str,
                                 file_path: str) -> List[AnalysisIssue]:
        """Find error pattern in code lines"""
        
        issues = []
        regex = re.compile(pattern, re.IGNORECASE)
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments for most patterns
            line_stripped = line.strip()
            if (line_stripped.startswith('#') or 
                line_stripped.startswith('//') or 
                line_stripped.startswith('/*')):
                continue
            
            matches = regex.finditer(line)
            for match in matches:
                severity = self._get_severity_for_category(category)
                
                issue = AnalysisIssue(
                    analysis_type=AnalysisType.SYNTAX,
                    severity=severity,
                    message=description,
                    description=f"Potential error detected: {category}",
                    location=CodeLocation(
                        line=line_num,
                        column=match.start() + 1,
                        end_column=match.end() + 1
                    ),
                    code_snippet=line.strip(),
                    affected_code=match.group(0),
                    rule_id=f"error_{category}",
                    rule_name=category.replace('_', ' ').title(),
                    confidence=0.7
                )
                
                issues.append(issue)
        
        return issues
    
    async def _detect_common_antipatterns(self, 
                                         lines: List[str], 
                                         file_path: str) -> List[AnalysisIssue]:
        """Detect common anti-patterns"""
        
        issues = []
        
        for line_num, line in enumerate(lines, 1):
            # Check for magic numbers
            if re.search(self.common_antipatterns['magic_numbers'], line):
                # Skip if it's in a comment or string
                if not (line.strip().startswith('#') or line.strip().startswith('//')):
                    issues.append(AnalysisIssue(
                        analysis_type=AnalysisType.STYLE,
                        severity=SeverityLevel.INFO,
                        message="Magic number detected",
                        description="Consider using named constants for numeric literals",
                        location=CodeLocation(line=line_num, column=1),
                        code_snippet=line.strip(),
                        suggested_fix="Define a named constant for this value",
                        rule_id="magic_numbers",
                        confidence=0.6
                    ))
            
            # Check for long lines
            if len(line) > 120:
                issues.append(AnalysisIssue(
                    analysis_type=AnalysisType.STYLE,
                    severity=SeverityLevel.INFO,
                    message="Line too long",
                    description=f"Line has {len(line)} characters (recommended: ≤120)",
                    location=CodeLocation(line=line_num, column=121),
                    code_snippet=line[:50] + "..." if len(line) > 50 else line,
                    suggested_fix="Break line into multiple lines",
                    rule_id="long_lines",
                    confidence=1.0
                ))
            
            # Check for deep nesting
            if re.match(self.common_antipatterns['deep_nesting'], line):
                issues.append(AnalysisIssue(
                    analysis_type=AnalysisType.COMPLEXITY,
                    severity=SeverityLevel.WARNING,
                    message="Deep nesting detected",
                    description="Consider refactoring to reduce nesting levels",
                    location=CodeLocation(line=line_num, column=1),
                    code_snippet=line.strip(),
                    suggested_fix="Extract nested logic into separate functions",
                    rule_id="deep_nesting",
                    confidence=0.8
                ))
        
        return issues
    
    async def _detect_python_specific_errors(self, 
                                            lines: List[str], 
                                            file_path: str) -> List[AnalysisIssue]:
        """Detect Python-specific errors"""
        
        issues = []
        
        # Check for mutable default arguments
        for line_num, line in enumerate(lines, 1):
            if 'def ' in line and ('=[]' in line or '={}' in line):
                issues.append(AnalysisIssue(
                    analysis_type=AnalysisType.SYNTAX,
                    severity=SeverityLevel.ERROR,
                    message="Mutable default argument",
                    description="Mutable default arguments can cause unexpected behavior",
                    location=CodeLocation(line=line_num, column=line.find('=') + 1),
                    code_snippet=line.strip(),
                    suggested_fix="Use None as default and create mutable object inside function",
                    rule_id="mutable_default_argument",
                    confidence=0.9
                ))
        
        # Check for unused variables (basic check)
        variable_assignments = {}
        variable_usage = set()
        
        for line_num, line in enumerate(lines, 1):
            # Find variable assignments
            assignment_match = re.search(r'(\w+)\s*=', line)
            if assignment_match and not line.strip().startswith('#'):
                var_name = assignment_match.group(1)
                if not var_name.startswith('_'):  # Skip private variables
                    variable_assignments[var_name] = line_num
            
            # Find variable usage
            for var_name in variable_assignments:
                if var_name in line and f'{var_name} =' not in line:
                    variable_usage.add(var_name)
        
        # Report unused variables
        for var_name, line_num in variable_assignments.items():
            if var_name not in variable_usage:
                issues.append(AnalysisIssue(
                    analysis_type=AnalysisType.STYLE,
                    severity=SeverityLevel.WARNING,
                    message=f"Unused variable: {var_name}",
                    description="Variable is assigned but never used",
                    location=CodeLocation(line=line_num, column=1),
                    code_snippet=lines[line_num - 1].strip(),
                    suggested_fix="Remove unused variable or prefix with underscore",
                    rule_id="unused_variable",
                    confidence=0.7
                ))
        
        return issues
    
    async def _detect_javascript_specific_errors(self, 
                                                lines: List[str], 
                                                file_path: str) -> List[AnalysisIssue]:
        """Detect JavaScript-specific errors"""
        
        issues = []
        
        # Check for function hoisting issues
        function_declarations = {}
        function_calls = {}
        
        for line_num, line in enumerate(lines, 1):
            # Find function declarations
            func_match = re.search(r'function\s+(\w+)', line)
            if func_match:
                func_name = func_match.group(1)
                function_declarations[func_name] = line_num
            
            # Find function calls
            call_match = re.search(r'(\w+)\s*\(', line)
            if call_match:
                func_name = call_match.group(1)
                if func_name not in ['if', 'for', 'while', 'switch']:  # Skip control structures
                    if func_name not in function_calls:
                        function_calls[func_name] = []
                    function_calls[func_name].append(line_num)
        
        # Check for calls before declaration (potential hoisting issues)
        for func_name, call_lines in function_calls.items():
            if func_name in function_declarations:
                declaration_line = function_declarations[func_name]
                for call_line in call_lines:
                    if call_line < declaration_line:
                        issues.append(AnalysisIssue(
                            analysis_type=AnalysisType.STYLE,
                            severity=SeverityLevel.INFO,
                            message="Function called before declaration",
                            description="Relying on function hoisting can be confusing",
                            location=CodeLocation(line=call_line, column=1),
                            code_snippet=lines[call_line - 1].strip(),
                            suggested_fix="Declare function before calling it",
                            rule_id="function_hoisting",
                            confidence=0.6
                        ))
        
        return issues
    
    def _get_severity_for_category(self, category: str) -> SeverityLevel:
        """Get severity level for error category"""
        
        critical_errors = ['logic_errors', 'resource_leaks', 'exception_handling']
        warnings = ['performance_issues', 'async_issues', 'scope_issues', 'type_issues']
        info_issues = ['naming_issues', 'concurrency_issues']
        
        if category in critical_errors:
            return SeverityLevel.ERROR
        elif category in warnings:
            return SeverityLevel.WARNING
        else:
            return SeverityLevel.INFO
    
    def get_error_categories(self) -> List[str]:
        """Get list of error categories detected"""
        categories = set()
        for lang_patterns in self.error_patterns.values():
            categories.update(lang_patterns.keys())
        categories.update(['magic_numbers', 'long_lines', 'deep_nesting'])
        return sorted(list(categories))
    
    def is_language_supported(self, language: Language) -> bool:
        """Check if language is supported for error detection"""
        return language in self.error_patterns