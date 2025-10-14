#!/usr/bin/env python3
"""
Refactoring Suggestion Engine

Generates refactoring suggestions and code improvements with line-level precision.
Provides automated suggestions for common code quality improvements.
"""

import re
from typing import Dict, List, Optional, Tuple

from xencode.models.code_analysis import (
    AnalysisIssue,
    AnalysisType,
    CodeLocation,
    Language,
    RefactoringSuggestion,
    SeverityLevel
)


class RefactoringEngine:
    """Generates refactoring suggestions for code improvements"""
    
    def __init__(self):
        # Refactoring patterns for different languages
        self.refactoring_patterns = {
            Language.PYTHON: {
                'extract_method': [
                    (r'def\s+\w+.*:\s*\n((?:\s{4,}.*\n){10,})', 'Long method should be extracted into smaller methods'),
                ],
                'simplify_conditionals': [
                    (r'if\s+(.+):\s*\n\s+return\s+True\s*\n\s*else:\s*\n\s+return\s+False', 'Simplify boolean return'),
                    (r'if\s+(.+):\s*\n\s+(.+)\s*\n\s*else:\s*\n\s+\2', 'Duplicate code in if/else branches'),
                ],
                'improve_loops': [
                    (r'for\s+\w+\s+in\s+range\(len\((.+)\)\):', 'Use enumerate() instead of range(len())'),
                    (r'(\w+)\s*=\s*\[\]\s*\n.*for.*:\s*\n\s*\1\.append\(', 'Use list comprehension instead of loop'),
                ],
                'remove_dead_code': [
                    (r'def\s+(\w+).*:\s*\n(?:\s+.*\n)*?\s+pass\s*$', 'Remove empty function'),
                    (r'if\s+False\s*:', 'Remove unreachable code'),
                ],
                'improve_naming': [
                    (r'def\s+([a-z])(\w*)', 'Function name too short, consider more descriptive name'),
                    (r'(\w+)\s*=.*#.*TODO', 'Variable with TODO comment needs attention'),
                ]
            },
            Language.JAVASCRIPT: {
                'modernize_syntax': [
                    (r'var\s+(\w+)\s*=', 'Replace var with let or const'),
                    (r'function\s*\(([^)]*)\)\s*\{', 'Consider using arrow function'),
                ],
                'simplify_conditionals': [
                    (r'if\s*\((.+)\)\s*\{\s*return\s+true;?\s*\}\s*else\s*\{\s*return\s+false;?\s*\}', 'Simplify boolean return'),
                    (r'(.+)\s*\?\s*true\s*:\s*false', 'Unnecessary ternary operator'),
                ],
                'improve_async': [
                    (r'\.then\(.*\)\.catch\(', 'Consider using async/await instead of Promise chains'),
                    (r'new\s+Promise\s*\(\s*\(resolve,\s*reject\)\s*=>', 'Consider if Promise constructor is necessary'),
                ],
                'optimize_performance': [
                    (r'document\.getElementById\(.*\).*for', 'Cache DOM query outside loop'),
                    (r'\.innerHTML\s*\+=', 'Use DocumentFragment for multiple DOM updates'),
                ],
                'improve_error_handling': [
                    (r'catch\s*\(\w*\)\s*\{\s*\}', 'Empty catch block should handle or log error'),
                    (r'throw\s+new\s+Error\(\s*["\'].*["\']', 'Consider using specific error types'),
                ]
            }
        }
    
    async def generate_refactoring_suggestions(self, 
                                             code: str, 
                                             language: Language,
                                             file_path: str = "") -> Tuple[List[AnalysisIssue], List[RefactoringSuggestion]]:
        """Generate refactoring suggestions for code"""
        
        analysis_issues = []
        refactoring_suggestions = []
        
        if language not in self.refactoring_patterns:
            return analysis_issues, refactoring_suggestions
        
        lines = code.split('\n')
        patterns = self.refactoring_patterns[language]
        
        for category, pattern_list in patterns.items():
            for pattern, description in pattern_list:
                suggestions = await self._find_refactoring_opportunities(
                    code, lines, pattern, description, category, language, file_path
                )
                analysis_issues.extend([s[0] for s in suggestions])
                refactoring_suggestions.extend([s[1] for s in suggestions])
        
        return analysis_issues, refactoring_suggestions
    
    async def _find_refactoring_opportunities(self, 
                                            code: str,
                                            lines: List[str], 
                                            pattern: str, 
                                            description: str,
                                            category: str,
                                            language: Language,
                                            file_path: str) -> List[Tuple[AnalysisIssue, RefactoringSuggestion]]:
        """Find refactoring opportunities in code"""
        
        suggestions = []
        regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
        
        matches = regex.finditer(code)
        for match in matches:
            # Calculate line number
            line_num = code[:match.start()].count('\n') + 1
            
            # Generate refactoring suggestion
            before_code = match.group(0)
            after_code = await self._generate_refactored_code(
                before_code, category, language, match
            )
            
            if after_code and after_code != before_code:
                # Create analysis issue
                issue = AnalysisIssue(
                    analysis_type=AnalysisType.REFACTORING,
                    severity=SeverityLevel.INFO,
                    message=description,
                    description=f"Refactoring opportunity: {category}",
                    location=CodeLocation(
                        line=line_num,
                        column=1
                    ),
                    code_snippet=before_code.strip(),
                    suggested_fix=f"Refactor using {category.replace('_', ' ')} pattern",
                    rule_id=f"refactor_{category}",
                    rule_name=category.replace('_', ' ').title(),
                    confidence=0.7
                )
                
                # Create refactoring suggestion
                refactoring = RefactoringSuggestion(
                    refactoring_type=category,
                    description=description,
                    before_code=before_code.strip(),
                    after_code=after_code.strip(),
                    benefits=self._get_refactoring_benefits(category),
                    effort_level=self._get_effort_level(category),
                    confidence=0.7
                )
                
                suggestions.append((issue, refactoring))
        
        return suggestions
    
    async def _generate_refactored_code(self, 
                                       original_code: str, 
                                       category: str,
                                       language: Language,
                                       match: re.Match) -> str:
        """Generate refactored code based on category and language"""
        
        if language == Language.PYTHON:
            return await self._generate_python_refactoring(original_code, category, match)
        elif language == Language.JAVASCRIPT:
            return await self._generate_javascript_refactoring(original_code, category, match)
        
        return original_code
    
    async def _generate_python_refactoring(self, 
                                         original_code: str, 
                                         category: str,
                                         match: re.Match) -> str:
        """Generate Python-specific refactoring"""
        
        if category == 'simplify_conditionals':
            # if condition: return True else: return False -> return condition
            if 'return True' in original_code and 'return False' in original_code:
                condition = match.group(1) if match.groups() else 'condition'
                return f"return {condition}"
        
        elif category == 'improve_loops':
            if 'range(len(' in original_code:
                # for i in range(len(items)): -> for i, item in enumerate(items):
                list_name = match.group(1) if match.groups() else 'items'
                return f"for i, item in enumerate({list_name}):"
        
        elif category == 'modernize_syntax':
            if 'var ' in original_code:
                return original_code.replace('var ', 'let ')
        
        return original_code
    
    async def _generate_javascript_refactoring(self, 
                                             original_code: str, 
                                             category: str,
                                             match: re.Match) -> str:
        """Generate JavaScript-specific refactoring"""
        
        if category == 'modernize_syntax':
            if original_code.startswith('var '):
                # Determine if it should be let or const
                if '=' in original_code and not re.search(r'\w+\s*=.*=', original_code):
                    return original_code.replace('var ', 'const ', 1)
                else:
                    return original_code.replace('var ', 'let ', 1)
            
            elif original_code.startswith('function'):
                # function(params) { -> (params) => {
                params = match.group(1) if match.groups() else ''
                return f"({params}) => {{"
        
        elif category == 'simplify_conditionals':
            if 'return true' in original_code.lower() and 'return false' in original_code.lower():
                condition = match.group(1) if match.groups() else 'condition'
                return f"return {condition};"
            
            elif '? true : false' in original_code:
                # condition ? true : false -> condition
                condition = match.group(1) if match.groups() else 'condition'
                return condition
        
        elif category == 'improve_async':
            if '.then(' in original_code and '.catch(' in original_code:
                return "// Consider refactoring to async/await:\n// try {\n//   const result = await promise;\n// } catch (error) {\n//   // handle error\n// }"
        
        return original_code
    
    def _get_refactoring_benefits(self, category: str) -> List[str]:
        """Get benefits of refactoring for category"""
        
        benefits_map = {
            'extract_method': [
                'Improves code readability',
                'Reduces method complexity',
                'Enables code reuse',
                'Makes testing easier'
            ],
            'simplify_conditionals': [
                'Reduces code complexity',
                'Improves readability',
                'Eliminates redundant logic'
            ],
            'improve_loops': [
                'More Pythonic/idiomatic code',
                'Better performance',
                'Improved readability'
            ],
            'modernize_syntax': [
                'Uses modern language features',
                'Better scoping rules',
                'Improved maintainability'
            ],
            'improve_async': [
                'Better error handling',
                'More readable async code',
                'Easier debugging'
            ],
            'optimize_performance': [
                'Improved runtime performance',
                'Reduced DOM queries',
                'Better user experience'
            ],
            'remove_dead_code': [
                'Reduces codebase size',
                'Eliminates confusion',
                'Improves maintainability'
            ]
        }
        
        return benefits_map.get(category, ['Improves code quality'])
    
    def _get_effort_level(self, category: str) -> str:
        """Get effort level for refactoring category"""
        
        high_effort = ['extract_method', 'improve_async']
        medium_effort = ['improve_loops', 'optimize_performance']
        low_effort = ['simplify_conditionals', 'modernize_syntax', 'remove_dead_code']
        
        if category in high_effort:
            return 'high'
        elif category in medium_effort:
            return 'medium'
        else:
            return 'low'
    
    async def suggest_method_extraction(self, 
                                      code: str, 
                                      language: Language,
                                      min_lines: int = 10) -> List[RefactoringSuggestion]:
        """Suggest method extraction for long methods"""
        
        suggestions = []
        lines = code.split('\n')
        
        if language == Language.PYTHON:
            # Find long methods
            in_method = False
            method_start = 0
            method_lines = []
            
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    if in_method and len(method_lines) >= min_lines:
                        # Suggest extraction for previous method
                        method_code = '\n'.join(method_lines)
                        suggestion = RefactoringSuggestion(
                            refactoring_type='extract_method',
                            description=f'Method is {len(method_lines)} lines long, consider extracting smaller methods',
                            before_code=method_code,
                            after_code='# Extract logical blocks into separate methods',
                            benefits=['Improves readability', 'Reduces complexity', 'Enables reuse'],
                            effort_level='medium',
                            confidence=0.8
                        )
                        suggestions.append(suggestion)
                    
                    # Start new method
                    in_method = True
                    method_start = i
                    method_lines = [line]
                
                elif in_method:
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        # End of method
                        if len(method_lines) >= min_lines:
                            method_code = '\n'.join(method_lines)
                            suggestion = RefactoringSuggestion(
                                refactoring_type='extract_method',
                                description=f'Method is {len(method_lines)} lines long, consider extracting smaller methods',
                                before_code=method_code,
                                after_code='# Extract logical blocks into separate methods',
                                benefits=['Improves readability', 'Reduces complexity', 'Enables reuse'],
                                effort_level='medium',
                                confidence=0.8
                            )
                            suggestions.append(suggestion)
                        
                        in_method = False
                        method_lines = []
                    else:
                        method_lines.append(line)
        
        return suggestions
    
    def get_refactoring_categories(self) -> List[str]:
        """Get list of refactoring categories"""
        categories = set()
        for lang_patterns in self.refactoring_patterns.values():
            categories.update(lang_patterns.keys())
        return sorted(list(categories))
    
    def is_language_supported(self, language: Language) -> bool:
        """Check if language is supported for refactoring suggestions"""
        return language in self.refactoring_patterns