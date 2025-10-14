#!/usr/bin/env python3
"""
Security Analyzer

Detects security vulnerabilities and potential security issues in code.
Provides comprehensive security analysis for multiple programming languages.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from xencode.models.code_analysis import (
    AnalysisIssue,
    AnalysisType,
    CodeLocation,
    Language,
    SecurityIssue,
    SeverityLevel
)


class SecurityAnalyzer:
    """Analyzes code for security vulnerabilities"""
    
    def __init__(self):
        # Security patterns for different languages
        self.security_patterns = {
            Language.PYTHON: {
                'code_injection': [
                    (r'eval\s*\(', 'Use of eval() can execute arbitrary code'),
                    (r'exec\s*\(', 'Use of exec() can execute arbitrary code'),
                    (r'compile\s*\(', 'Dynamic code compilation can be dangerous'),
                    (r'__import__\s*\(', 'Dynamic imports can be exploited')
                ],
                'sql_injection': [
                    (r'cursor\.execute\s*\(\s*["\'].*%.*["\']', 'Potential SQL injection via string formatting'),
                    (r'\.execute\s*\(\s*f["\']', 'F-string in SQL query may allow injection'),
                    (r'\.execute\s*\(\s*["\'].*\+.*["\']', 'String concatenation in SQL query')
                ],
                'path_traversal': [
                    (r'open\s*\(\s*.*\+.*\)', 'Path concatenation without validation'),
                    (r'os\.path\.join\s*\(.*input', 'User input in file path'),
                    (r'\.\./', 'Potential path traversal sequence')
                ],
                'weak_crypto': [
                    (r'hashlib\.md5\s*\(', 'MD5 is cryptographically broken'),
                    (r'hashlib\.sha1\s*\(', 'SHA1 is cryptographically weak'),
                    (r'random\.random\s*\(', 'Use secrets module for cryptographic randomness')
                ],
                'hardcoded_secrets': [
                    (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password detected'),
                    (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key detected'),
                    (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret detected'),
                    (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token detected')
                ]
            },
            Language.JAVASCRIPT: {
                'code_injection': [
                    (r'eval\s*\(', 'Use of eval() can execute arbitrary code'),
                    (r'Function\s*\(', 'Function constructor can execute arbitrary code'),
                    (r'setTimeout\s*\(\s*["\']', 'setTimeout with string argument'),
                    (r'setInterval\s*\(\s*["\']', 'setInterval with string argument')
                ],
                'xss_vulnerabilities': [
                    (r'innerHTML\s*=', 'innerHTML assignment can lead to XSS'),
                    (r'outerHTML\s*=', 'outerHTML assignment can lead to XSS'),
                    (r'document\.write\s*\(', 'document.write can lead to XSS'),
                    (r'\.html\s*\(.*\+', 'Dynamic HTML content without sanitization')
                ],
                'prototype_pollution': [
                    (r'__proto__', 'Direct __proto__ manipulation'),
                    (r'constructor\.prototype', 'Prototype manipulation'),
                    (r'Object\.setPrototypeOf', 'Prototype modification')
                ],
                'weak_crypto': [
                    (r'Math\.random\s*\(', 'Math.random is not cryptographically secure'),
                    (r'btoa\s*\(', 'Base64 encoding is not encryption'),
                    (r'atob\s*\(', 'Base64 decoding is not decryption')
                ],
                'hardcoded_secrets': [
                    (r'password\s*[:=]\s*["\'][^"\']+["\']', 'Hardcoded password detected'),
                    (r'apiKey\s*[:=]\s*["\'][^"\']+["\']', 'Hardcoded API key detected'),
                    (r'secret\s*[:=]\s*["\'][^"\']+["\']', 'Hardcoded secret detected'),
                    (r'token\s*[:=]\s*["\'][^"\']+["\']', 'Hardcoded token detected')
                ]
            },
            Language.JAVA: {
                'code_injection': [
                    (r'Runtime\.getRuntime\(\)\.exec', 'Command execution can be dangerous'),
                    (r'ProcessBuilder', 'Process execution requires validation'),
                    (r'ScriptEngine\.eval', 'Script evaluation can execute arbitrary code')
                ],
                'sql_injection': [
                    (r'Statement\.execute.*\+', 'String concatenation in SQL query'),
                    (r'createStatement\(\)\.execute', 'Direct statement execution without parameters')
                ],
                'deserialization': [
                    (r'ObjectInputStream\.readObject', 'Deserialization can execute arbitrary code'),
                    (r'XMLDecoder\.readObject', 'XML deserialization vulnerability')
                ]
            }
        }
        
        # CWE (Common Weakness Enumeration) mappings
        self.cwe_mappings = {
            'code_injection': 'CWE-94',
            'sql_injection': 'CWE-89',
            'xss_vulnerabilities': 'CWE-79',
            'path_traversal': 'CWE-22',
            'weak_crypto': 'CWE-327',
            'hardcoded_secrets': 'CWE-798',
            'prototype_pollution': 'CWE-1321',
            'deserialization': 'CWE-502'
        }
    
    async def analyze_security(self, 
                              code: str, 
                              language: Language,
                              file_path: str = "") -> Tuple[List[AnalysisIssue], List[SecurityIssue]]:
        """Analyze code for security vulnerabilities"""
        
        analysis_issues = []
        security_issues = []
        
        if language not in self.security_patterns:
            return analysis_issues, security_issues
        
        lines = code.split('\n')
        patterns = self.security_patterns[language]
        
        for category, pattern_list in patterns.items():
            for pattern, description in pattern_list:
                issues = await self._find_security_pattern(
                    lines, pattern, description, category, file_path
                )
                analysis_issues.extend(issues)
                
                # Create corresponding SecurityIssue objects
                for issue in issues:
                    security_issue = SecurityIssue(
                        vulnerability_type=category,
                        cwe_id=self.cwe_mappings.get(category),
                        risk_level=issue.severity,
                        exploit_scenario=self._get_exploit_scenario(category),
                        mitigation=self._get_mitigation_advice(category, language)
                    )
                    security_issues.append(security_issue)
        
        return analysis_issues, security_issues
    
    async def _find_security_pattern(self, 
                                    lines: List[str], 
                                    pattern: str, 
                                    description: str,
                                    category: str,
                                    file_path: str) -> List[AnalysisIssue]:
        """Find security pattern in code lines"""
        
        issues = []
        regex = re.compile(pattern, re.IGNORECASE)
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            line_stripped = line.strip()
            if (line_stripped.startswith('#') or 
                line_stripped.startswith('//') or 
                line_stripped.startswith('/*')):
                continue
            
            matches = regex.finditer(line)
            for match in matches:
                severity = self._get_severity_for_category(category)
                
                issue = AnalysisIssue(
                    analysis_type=AnalysisType.SECURITY,
                    severity=severity,
                    message=description,
                    description=f"Security vulnerability detected: {category}",
                    location=CodeLocation(
                        line=line_num,
                        column=match.start() + 1,
                        end_column=match.end() + 1
                    ),
                    code_snippet=line.strip(),
                    affected_code=match.group(0),
                    rule_id=f"security_{category}",
                    rule_name=category.replace('_', ' ').title(),
                    confidence=0.8
                )
                
                issues.append(issue)
        
        return issues
    
    def _get_severity_for_category(self, category: str) -> SeverityLevel:
        """Get severity level for security category"""
        high_risk = ['code_injection', 'sql_injection', 'deserialization']
        medium_risk = ['xss_vulnerabilities', 'path_traversal', 'prototype_pollution']
        low_risk = ['weak_crypto', 'hardcoded_secrets']
        
        if category in high_risk:
            return SeverityLevel.ERROR
        elif category in medium_risk:
            return SeverityLevel.WARNING
        else:
            return SeverityLevel.INFO
    
    def _get_exploit_scenario(self, category: str) -> str:
        """Get exploit scenario description for category"""
        scenarios = {
            'code_injection': 'Attacker can execute arbitrary code by controlling input to eval/exec functions',
            'sql_injection': 'Attacker can manipulate database queries to access unauthorized data',
            'xss_vulnerabilities': 'Attacker can inject malicious scripts into web pages',
            'path_traversal': 'Attacker can access files outside intended directory',
            'weak_crypto': 'Attacker can break weak cryptographic algorithms',
            'hardcoded_secrets': 'Secrets exposed in source code can be extracted',
            'prototype_pollution': 'Attacker can modify object prototypes to affect application behavior',
            'deserialization': 'Attacker can execute code through malicious serialized objects'
        }
        return scenarios.get(category, 'Security vulnerability that could be exploited')
    
    def _get_mitigation_advice(self, category: str, language: Language) -> str:
        """Get mitigation advice for category and language"""
        
        mitigations = {
            'code_injection': {
                Language.PYTHON: 'Use ast.literal_eval() for safe evaluation, avoid eval/exec',
                Language.JAVASCRIPT: 'Avoid eval(), use JSON.parse() for data parsing',
                Language.JAVA: 'Validate and sanitize all inputs before execution'
            },
            'sql_injection': {
                Language.PYTHON: 'Use parameterized queries with cursor.execute(query, params)',
                Language.JAVASCRIPT: 'Use prepared statements or ORM with parameter binding',
                Language.JAVA: 'Use PreparedStatement with parameter placeholders'
            },
            'xss_vulnerabilities': {
                Language.JAVASCRIPT: 'Sanitize user input, use textContent instead of innerHTML'
            },
            'path_traversal': {
                Language.PYTHON: 'Validate file paths, use os.path.abspath() and check bounds'
            },
            'weak_crypto': {
                Language.PYTHON: 'Use hashlib.sha256() or stronger, secrets module for randomness',
                Language.JAVASCRIPT: 'Use Web Crypto API for secure random values'
            },
            'hardcoded_secrets': {
                Language.PYTHON: 'Use environment variables or secure configuration files',
                Language.JAVASCRIPT: 'Use environment variables, never commit secrets to code'
            }
        }
        
        category_mitigations = mitigations.get(category, {})
        return category_mitigations.get(language, 'Follow security best practices for this vulnerability type')
    
    async def analyze_dependencies(self, 
                                  code: str, 
                                  language: Language,
                                  file_path: str = "") -> List[AnalysisIssue]:
        """Analyze dependencies for known vulnerabilities"""
        
        issues = []
        lines = code.split('\n')
        
        # Known vulnerable packages (simplified list)
        vulnerable_packages = {
            Language.PYTHON: {
                'pickle': 'pickle module can execute arbitrary code during deserialization',
                'yaml.load': 'yaml.load() can execute arbitrary code, use yaml.safe_load()',
                'requests': 'Ensure requests version >= 2.20.0 for security fixes'
            },
            Language.JAVASCRIPT: {
                'lodash': 'Some lodash versions have prototype pollution vulnerabilities',
                'moment': 'moment.js is in maintenance mode, consider date-fns or dayjs',
                'jquery': 'Ensure jQuery version >= 3.5.0 for XSS fixes'
            }
        }
        
        if language not in vulnerable_packages:
            return issues
        
        packages = vulnerable_packages[language]
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for imports/requires
            for package, warning in packages.items():
                if language == Language.PYTHON:
                    if f'import {package}' in line or f'from {package}' in line:
                        issues.append(AnalysisIssue(
                            analysis_type=AnalysisType.SECURITY,
                            severity=SeverityLevel.WARNING,
                            message=f"Potentially vulnerable dependency: {package}",
                            description=warning,
                            location=CodeLocation(line=line_num, column=1),
                            code_snippet=line.strip(),
                            rule_id=f"vulnerable_dependency_{package}",
                            confidence=0.7
                        ))
                
                elif language == Language.JAVASCRIPT:
                    if f"require('{package}')" in line or f'require("{package}")' in line:
                        issues.append(AnalysisIssue(
                            analysis_type=AnalysisType.SECURITY,
                            severity=SeverityLevel.WARNING,
                            message=f"Potentially vulnerable dependency: {package}",
                            description=warning,
                            location=CodeLocation(line=line_num, column=1),
                            code_snippet=line.strip(),
                            rule_id=f"vulnerable_dependency_{package}",
                            confidence=0.7
                        ))
        
        return issues
    
    def get_security_categories(self) -> List[str]:
        """Get list of security categories analyzed"""
        categories = set()
        for lang_patterns in self.security_patterns.values():
            categories.update(lang_patterns.keys())
        return sorted(list(categories))
    
    def is_language_supported(self, language: Language) -> bool:
        """Check if language is supported for security analysis"""
        return language in self.security_patterns