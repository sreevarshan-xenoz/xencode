#!/usr/bin/env python3
"""
Enhanced Security Analyzer with Bandit Integration

Detects security vulnerabilities and potential security issues in code.
Provides comprehensive security analysis for multiple programming languages
with integrated Bandit scanning for Python code.
"""

import asyncio
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import logging

from xencode.models.code_analysis import (
    AnalysisIssue,
    AnalysisType,
    CodeLocation,
    Language,
    SecurityIssue,
    SeverityLevel
)

logger = logging.getLogger(__name__)


class BanditIntegration:
    """Integration with Bandit security scanner for Python code"""
    
    def __init__(self):
        self.bandit_available = self._check_bandit_availability()
        self.logger = logging.getLogger(__name__)
    
    def _check_bandit_availability(self) -> bool:
        """Check if Bandit is available in the system"""
        try:
            result = subprocess.run(['bandit', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    async def scan_python_code(self, code: str, file_path: str = "temp.py") -> List[Dict[str, Any]]:
        """Scan Python code using Bandit"""
        if not self.bandit_available:
            self.logger.warning("Bandit not available, skipping advanced Python security scan")
            return []
        
        try:
            # Create temporary file for scanning
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            # Run Bandit scan
            cmd = [
                'bandit',
                '-f', 'json',  # JSON output format
                '-ll',  # Low confidence level
                '-i',   # Don't exit on error
                temp_file_path
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            # Clean up temporary file
            Path(temp_file_path).unlink(missing_ok=True)
            
            if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                try:
                    bandit_results = json.loads(stdout.decode())
                    return bandit_results.get('results', [])
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse Bandit JSON output: {e}")
                    return []
            else:
                self.logger.error(f"Bandit scan failed: {stderr.decode()}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error running Bandit scan: {e}")
            return []
    
    def convert_bandit_to_analysis_issues(self, bandit_results: List[Dict[str, Any]], 
                                        file_path: str = "") -> List[AnalysisIssue]:
        """Convert Bandit results to AnalysisIssue objects"""
        issues = []
        
        for result in bandit_results:
            # Map Bandit severity to our severity levels
            severity_mapping = {
                'HIGH': SeverityLevel.ERROR,
                'MEDIUM': SeverityLevel.WARNING,
                'LOW': SeverityLevel.INFO
            }
            
            severity = severity_mapping.get(result.get('issue_severity', 'LOW'), SeverityLevel.INFO)
            
            # Map Bandit confidence to our confidence score
            confidence_mapping = {
                'HIGH': 0.9,
                'MEDIUM': 0.7,
                'LOW': 0.5
            }
            
            confidence = confidence_mapping.get(result.get('issue_confidence', 'LOW'), 0.5)
            
            issue = AnalysisIssue(
                analysis_type=AnalysisType.SECURITY,
                severity=severity,
                message=result.get('issue_text', 'Security issue detected'),
                description=f"Bandit {result.get('test_id', 'unknown')}: {result.get('issue_text', '')}",
                location=CodeLocation(
                    line=result.get('line_number', 1),
                    column=result.get('col_offset', 1) if result.get('col_offset') else 1,
                    end_column=result.get('col_offset', 1) + 10 if result.get('col_offset') else 11
                ),
                code_snippet=result.get('code', '').strip(),
                affected_code=result.get('code', '').strip(),
                rule_id=result.get('test_id', 'bandit_unknown'),
                rule_name=result.get('test_name', 'Bandit Security Check'),
                confidence=confidence
            )
            
            issues.append(issue)
        
        return issues


class VulnerabilityDatabase:
    """Database of known vulnerabilities and security patterns"""
    
    def __init__(self):
        self.cve_patterns = self._load_cve_patterns()
        self.owasp_top10 = self._load_owasp_patterns()
    
    def _load_cve_patterns(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Load CVE-based vulnerability patterns"""
        return {
            Language.PYTHON: [
                (r'pickle\.loads?\s*\(', 'CVE-2019-16935', 'Pickle deserialization vulnerability'),
                (r'yaml\.load\s*\((?!.*Loader=)', 'CVE-2017-18342', 'YAML unsafe loading'),
                (r'subprocess\.call\s*\(.*shell=True', 'CWE-78', 'Command injection via shell=True'),
                (r'os\.system\s*\(', 'CWE-78', 'Command injection via os.system'),
            ],
            Language.JAVASCRIPT: [
                (r'JSON\.parse\s*\(.*\+', 'CWE-94', 'JSON parsing with concatenation'),
                (r'new Function\s*\(', 'CWE-94', 'Dynamic function creation'),
                (r'window\.location\s*=.*\+', 'CWE-79', 'Open redirect vulnerability'),
            ]
        }
    
    def _load_owasp_patterns(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Load OWASP Top 10 vulnerability patterns"""
        return {
            'injection': [
                (r'\.execute\s*\(\s*["\'].*%.*["\']', 'A03:2021', 'SQL Injection'),
                (r'eval\s*\(', 'A03:2021', 'Code Injection'),
            ],
            'broken_authentication': [
                (r'password\s*==\s*["\'][^"\']*["\']', 'A07:2021', 'Hardcoded password'),
                (r'session\[.*\]\s*=.*without.*validation', 'A07:2021', 'Session fixation'),
            ],
            'sensitive_data_exposure': [
                (r'print\s*\(.*password', 'A02:2021', 'Password in logs'),
                (r'console\.log\s*\(.*token', 'A02:2021', 'Token in logs'),
            ],
            'security_misconfiguration': [
                (r'debug\s*=\s*True', 'A05:2021', 'Debug mode enabled'),
                (r'ssl_verify\s*=\s*False', 'A05:2021', 'SSL verification disabled'),
            ]
        }
    
    async def check_vulnerabilities(self, code: str, language: Language) -> List[AnalysisIssue]:
        """Check code against vulnerability database"""
        issues = []
        lines = code.split('\n')
        
        # Check CVE patterns
        if language in self.cve_patterns:
            for pattern, cve_id, description in self.cve_patterns[language]:
                regex = re.compile(pattern, re.IGNORECASE)
                for line_num, line in enumerate(lines, 1):
                    matches = regex.finditer(line)
                    for match in matches:
                        issue = AnalysisIssue(
                            analysis_type=AnalysisType.SECURITY,
                            severity=SeverityLevel.ERROR,
                            message=f"Known vulnerability: {cve_id}",
                            description=description,
                            location=CodeLocation(
                                line=line_num,
                                column=match.start() + 1,
                                end_column=match.end() + 1
                            ),
                            code_snippet=line.strip(),
                            affected_code=match.group(0),
                            rule_id=f"cve_{cve_id.lower().replace('-', '_')}",
                            rule_name=f"CVE Check: {cve_id}",
                            confidence=0.85
                        )
                        issues.append(issue)
        
        # Check OWASP patterns
        for category, patterns in self.owasp_top10.items():
            for pattern, owasp_id, description in patterns:
                regex = re.compile(pattern, re.IGNORECASE)
                for line_num, line in enumerate(lines, 1):
                    matches = regex.finditer(line)
                    for match in matches:
                        issue = AnalysisIssue(
                            analysis_type=AnalysisType.SECURITY,
                            severity=SeverityLevel.WARNING,
                            message=f"OWASP {owasp_id}: {category.replace('_', ' ').title()}",
                            description=description,
                            location=CodeLocation(
                                line=line_num,
                                column=match.start() + 1,
                                end_column=match.end() + 1
                            ),
                            code_snippet=line.strip(),
                            affected_code=match.group(0),
                            rule_id=f"owasp_{category}",
                            rule_name=f"OWASP {owasp_id}",
                            confidence=0.75
                        )
                        issues.append(issue)
        
        return issues


class SecurityReportGenerator:
    """Generates comprehensive security reports"""
    
    def __init__(self):
        self.report_templates = {
            'summary': self._generate_summary_template(),
            'detailed': self._generate_detailed_template(),
            'executive': self._generate_executive_template()
        }
    
    def _generate_summary_template(self) -> str:
        """Generate summary report template"""
        return """
# Security Analysis Summary

## Overview
- **Total Issues Found**: {total_issues}
- **Critical Issues**: {critical_count}
- **High Severity**: {high_count}
- **Medium Severity**: {medium_count}
- **Low Severity**: {low_count}

## Risk Assessment
- **Overall Risk Level**: {risk_level}
- **Compliance Score**: {compliance_score}%

## Top Security Concerns
{top_concerns}

## Recommendations
{recommendations}
"""
    
    def _generate_detailed_template(self) -> str:
        """Generate detailed report template"""
        return """
# Detailed Security Analysis Report

## Executive Summary
{executive_summary}

## Methodology
- Static code analysis using custom patterns
- Bandit integration for Python security scanning
- CVE database matching
- OWASP Top 10 vulnerability detection

## Findings by Category
{findings_by_category}

## Vulnerability Details
{vulnerability_details}

## Remediation Plan
{remediation_plan}

## Appendix
{appendix}
"""
    
    def _generate_executive_template(self) -> str:
        """Generate executive report template"""
        return """
# Executive Security Report

## Business Impact Summary
{business_impact}

## Risk Metrics
{risk_metrics}

## Compliance Status
{compliance_status}

## Investment Recommendations
{investment_recommendations}
"""
    
    async def generate_report(self, issues: List[AnalysisIssue], 
                            report_type: str = 'summary') -> str:
        """Generate security report from analysis issues"""
        
        if report_type not in self.report_templates:
            report_type = 'summary'
        
        # Calculate metrics
        total_issues = len(issues)
        critical_count = sum(1 for issue in issues if issue.severity == SeverityLevel.ERROR)
        high_count = sum(1 for issue in issues if issue.severity == SeverityLevel.WARNING)
        medium_count = sum(1 for issue in issues if issue.severity == SeverityLevel.INFO)
        low_count = total_issues - critical_count - high_count - medium_count
        
        # Calculate risk level
        if critical_count > 0:
            risk_level = "CRITICAL"
        elif high_count > 3:
            risk_level = "HIGH"
        elif high_count > 0 or medium_count > 5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calculate compliance score
        compliance_score = max(0, 100 - (critical_count * 25 + high_count * 10 + medium_count * 5))
        
        # Generate top concerns
        top_concerns = self._generate_top_concerns(issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues)
        
        template = self.report_templates[report_type]
        
        return template.format(
            total_issues=total_issues,
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            risk_level=risk_level,
            compliance_score=compliance_score,
            top_concerns=top_concerns,
            recommendations=recommendations,
            executive_summary=self._generate_executive_summary(issues),
            findings_by_category=self._generate_findings_by_category(issues),
            vulnerability_details=self._generate_vulnerability_details(issues),
            remediation_plan=self._generate_remediation_plan(issues),
            appendix=self._generate_appendix(issues),
            business_impact=self._generate_business_impact(issues),
            risk_metrics=self._generate_risk_metrics(issues),
            compliance_status=self._generate_compliance_status(issues),
            investment_recommendations=self._generate_investment_recommendations(issues)
        )
    
    def _generate_top_concerns(self, issues: List[AnalysisIssue]) -> str:
        """Generate top security concerns"""
        critical_issues = [issue for issue in issues if issue.severity == SeverityLevel.ERROR]
        
        if not critical_issues:
            return "No critical security issues found."
        
        concerns = []
        for i, issue in enumerate(critical_issues[:5], 1):
            concerns.append(f"{i}. {issue.message} (Line {issue.location.line})")
        
        return "\n".join(concerns)
    
    def _generate_recommendations(self, issues: List[AnalysisIssue]) -> str:
        """Generate security recommendations"""
        recommendations = [
            "1. Address all critical security issues immediately",
            "2. Implement secure coding practices",
            "3. Regular security code reviews",
            "4. Automated security testing in CI/CD pipeline"
        ]
        
        # Add specific recommendations based on issue types
        rule_counts = {}
        for issue in issues:
            rule_counts[issue.rule_id] = rule_counts.get(issue.rule_id, 0) + 1
        
        # Most common issues
        common_issues = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for rule_id, count in common_issues:
            if 'injection' in rule_id:
                recommendations.append("5. Implement input validation and parameterized queries")
            elif 'crypto' in rule_id:
                recommendations.append("6. Update cryptographic implementations")
            elif 'secrets' in rule_id:
                recommendations.append("7. Remove hardcoded secrets and use secure configuration")
        
        return "\n".join(recommendations)
    
    def _generate_executive_summary(self, issues: List[AnalysisIssue]) -> str:
        """Generate executive summary"""
        return f"Security analysis identified {len(issues)} potential vulnerabilities requiring attention."
    
    def _generate_findings_by_category(self, issues: List[AnalysisIssue]) -> str:
        """Generate findings by category"""
        categories = {}
        for issue in issues:
            category = issue.rule_name
            if category not in categories:
                categories[category] = []
            categories[category].append(issue)
        
        result = []
        for category, category_issues in categories.items():
            result.append(f"### {category}")
            result.append(f"Issues found: {len(category_issues)}")
            result.append("")
        
        return "\n".join(result)
    
    def _generate_vulnerability_details(self, issues: List[AnalysisIssue]) -> str:
        """Generate detailed vulnerability information"""
        details = []
        for i, issue in enumerate(issues[:10], 1):  # Limit to top 10
            details.append(f"## Vulnerability {i}")
            details.append(f"**Type**: {issue.rule_name}")
            details.append(f"**Severity**: {issue.severity.value}")
            details.append(f"**Location**: Line {issue.location.line}")
            details.append(f"**Description**: {issue.description}")
            details.append(f"**Code**: `{issue.code_snippet}`")
            details.append("")
        
        return "\n".join(details)
    
    def _generate_remediation_plan(self, issues: List[AnalysisIssue]) -> str:
        """Generate remediation plan"""
        return "Prioritize critical issues, then address high and medium severity vulnerabilities."
    
    def _generate_appendix(self, issues: List[AnalysisIssue]) -> str:
        """Generate report appendix"""
        return "Additional technical details and references available upon request."
    
    def _generate_business_impact(self, issues: List[AnalysisIssue]) -> str:
        """Generate business impact assessment"""
        critical_count = sum(1 for issue in issues if issue.severity == SeverityLevel.ERROR)
        if critical_count > 0:
            return "High business risk due to critical security vulnerabilities."
        return "Moderate business risk requiring attention to security issues."
    
    def _generate_risk_metrics(self, issues: List[AnalysisIssue]) -> str:
        """Generate risk metrics"""
        return f"Risk Score: {len(issues) * 10} (based on {len(issues)} issues found)"
    
    def _generate_compliance_status(self, issues: List[AnalysisIssue]) -> str:
        """Generate compliance status"""
        critical_count = sum(1 for issue in issues if issue.severity == SeverityLevel.ERROR)
        if critical_count == 0:
            return "Compliant with basic security standards"
        return "Non-compliant due to critical security issues"
    
    def _generate_investment_recommendations(self, issues: List[AnalysisIssue]) -> str:
        """Generate investment recommendations"""
        return "Invest in security training and automated security testing tools."


class SecurityAnalyzer:
    """Enhanced security analyzer with Bandit integration and comprehensive reporting"""
    
    def __init__(self):
        self.bandit_integration = BanditIntegration()
        self.vulnerability_db = VulnerabilityDatabase()
        self.report_generator = SecurityReportGenerator()
        
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
        """Comprehensive security analysis with multiple scanning methods"""
        
        analysis_issues = []
        security_issues = []
        
        # 1. Pattern-based analysis (existing functionality)
        if language in self.security_patterns:
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
        
        # 2. Bandit integration for Python code
        if language == Language.PYTHON:
            try:
                bandit_results = await self.bandit_integration.scan_python_code(code, file_path)
                bandit_issues = self.bandit_integration.convert_bandit_to_analysis_issues(
                    bandit_results, file_path
                )
                analysis_issues.extend(bandit_issues)
                
                # Create SecurityIssue objects for Bandit findings
                for issue in bandit_issues:
                    security_issue = SecurityIssue(
                        vulnerability_type=issue.rule_id,
                        cwe_id=self._extract_cwe_from_bandit(issue.rule_id),
                        risk_level=issue.severity,
                        exploit_scenario=f"Bandit detected: {issue.message}",
                        mitigation=self._get_bandit_mitigation(issue.rule_id)
                    )
                    security_issues.append(security_issue)
            except Exception as e:
                logger.warning(f"Bandit integration failed: {e}")
        
        # 3. Vulnerability database checks
        try:
            vuln_issues = await self.vulnerability_db.check_vulnerabilities(code, language)
            analysis_issues.extend(vuln_issues)
            
            # Create SecurityIssue objects for vulnerability findings
            for issue in vuln_issues:
                security_issue = SecurityIssue(
                    vulnerability_type=issue.rule_id,
                    cwe_id=self._extract_cwe_from_rule(issue.rule_id),
                    risk_level=issue.severity,
                    exploit_scenario=issue.description,
                    mitigation=self._get_vulnerability_mitigation(issue.rule_id)
                )
                security_issues.append(security_issue)
        except Exception as e:
            logger.warning(f"Vulnerability database check failed: {e}")
        
        # 4. Dependency analysis
        try:
            dep_issues = await self.analyze_dependencies(code, language, file_path)
            analysis_issues.extend(dep_issues)
        except Exception as e:
            logger.warning(f"Dependency analysis failed: {e}")
        
        return analysis_issues, security_issues
    
    def _extract_cwe_from_bandit(self, rule_id: str) -> Optional[str]:
        """Extract CWE ID from Bandit rule ID"""
        # Bandit rule ID to CWE mapping
        bandit_cwe_mapping = {
            'B101': 'CWE-78',   # assert_used
            'B102': 'CWE-78',   # exec_used
            'B103': 'CWE-78',   # set_bad_file_permissions
            'B104': 'CWE-200',  # hardcoded_bind_all_interfaces
            'B105': 'CWE-798',  # hardcoded_password_string
            'B106': 'CWE-798',  # hardcoded_password_funcarg
            'B107': 'CWE-798',  # hardcoded_password_default
            'B108': 'CWE-377',  # hardcoded_tmp_directory
            'B110': 'CWE-703',  # try_except_pass
            'B112': 'CWE-703',  # try_except_continue
            'B201': 'CWE-78',   # flask_debug_true
            'B301': 'CWE-502',  # pickle
            'B302': 'CWE-327',  # marshal
            'B303': 'CWE-327',  # md5
            'B304': 'CWE-327',  # des
            'B305': 'CWE-327',  # cipher
            'B306': 'CWE-327',  # mktemp_q
            'B307': 'CWE-94',   # eval
            'B308': 'CWE-327',  # mark_safe
            'B309': 'CWE-327',  # httpsconnection
            'B310': 'CWE-327',  # urllib_urlopen
            'B311': 'CWE-330',  # random
            'B312': 'CWE-327',  # telnetlib
            'B313': 'CWE-327',  # xml_bad_cElementTree
            'B314': 'CWE-327',  # xml_bad_ElementTree
            'B315': 'CWE-327',  # xml_bad_expatreader
            'B316': 'CWE-327',  # xml_bad_expatbuilder
            'B317': 'CWE-327',  # xml_bad_sax
            'B318': 'CWE-327',  # xml_bad_minidom
            'B319': 'CWE-327',  # xml_bad_pulldom
            'B320': 'CWE-327',  # xml_bad_etree
            'B321': 'CWE-327',  # ftplib
            'B322': 'CWE-295',  # input
            'B323': 'CWE-327',  # unverified_context
            'B324': 'CWE-327',  # hashlib_new_insecure_functions
            'B325': 'CWE-377',  # tempfile
            'B401': 'CWE-78',   # import_telnetlib
            'B402': 'CWE-78',   # import_ftplib
            'B403': 'CWE-78',   # import_pickle
            'B404': 'CWE-78',   # import_subprocess
            'B405': 'CWE-327',  # import_xml_etree
            'B406': 'CWE-327',  # import_xml_sax
            'B407': 'CWE-327',  # import_xml_expat
            'B408': 'CWE-327',  # import_xml_minidom
            'B409': 'CWE-327',  # import_xml_pulldom
            'B410': 'CWE-327',  # import_lxml
            'B411': 'CWE-327',  # import_xmlrpclib
            'B412': 'CWE-327',  # import_httpoxy
            'B501': 'CWE-295',  # request_with_no_cert_validation
            'B502': 'CWE-295',  # ssl_with_bad_version
            'B503': 'CWE-295',  # ssl_with_bad_defaults
            'B504': 'CWE-295',  # ssl_with_no_version
            'B505': 'CWE-327',  # weak_cryptographic_key
            'B506': 'CWE-522',  # yaml_load
            'B507': 'CWE-78',   # ssh_no_host_key_verification
            'B601': 'CWE-78',   # paramiko_calls
            'B602': 'CWE-78',   # subprocess_popen_with_shell_equals_true
            'B603': 'CWE-78',   # subprocess_without_shell_equals_true
            'B604': 'CWE-78',   # any_other_function_with_shell_equals_true
            'B605': 'CWE-78',   # start_process_with_a_shell
            'B606': 'CWE-78',   # start_process_with_no_shell
            'B607': 'CWE-78',   # start_process_with_partial_path
            'B608': 'CWE-89',   # hardcoded_sql_expressions
            'B609': 'CWE-78',   # linux_commands_wildcard_injection
            'B610': 'CWE-78',   # django_extra_used
            'B611': 'CWE-78',   # django_rawsql_used
            'B701': 'CWE-295',  # jinja2_autoescape_false
            'B702': 'CWE-295',  # use_of_mako_templates
            'B703': 'CWE-295'   # django_mark_safe
        }
        
        return bandit_cwe_mapping.get(rule_id)
    
    def _extract_cwe_from_rule(self, rule_id: str) -> Optional[str]:
        """Extract CWE ID from rule ID"""
        if 'cve_' in rule_id:
            # Extract CVE ID and map to CWE if possible
            return None  # Would need CVE to CWE mapping database
        elif 'owasp_' in rule_id:
            # Map OWASP categories to CWE
            owasp_cwe_mapping = {
                'owasp_injection': 'CWE-94',
                'owasp_broken_authentication': 'CWE-287',
                'owasp_sensitive_data_exposure': 'CWE-200',
                'owasp_security_misconfiguration': 'CWE-16'
            }
            return owasp_cwe_mapping.get(rule_id)
        
        return self.cwe_mappings.get(rule_id.replace('security_', ''))
    
    def _get_bandit_mitigation(self, rule_id: str) -> str:
        """Get mitigation advice for Bandit rule"""
        bandit_mitigations = {
            'B101': 'Remove assert statements from production code',
            'B102': 'Avoid using exec(), use safer alternatives',
            'B105': 'Use environment variables or secure configuration for passwords',
            'B301': 'Avoid pickle for untrusted data, use JSON instead',
            'B303': 'Use SHA-256 or stronger hash functions instead of MD5',
            'B307': 'Avoid eval(), use ast.literal_eval() for safe evaluation',
            'B311': 'Use secrets module for cryptographically secure random numbers',
            'B501': 'Enable SSL certificate validation',
            'B506': 'Use yaml.safe_load() instead of yaml.load()',
            'B602': 'Avoid shell=True in subprocess calls',
            'B608': 'Use parameterized queries to prevent SQL injection'
        }
        
        return bandit_mitigations.get(rule_id, 'Follow security best practices for this vulnerability')
    
    def _get_vulnerability_mitigation(self, rule_id: str) -> str:
        """Get mitigation advice for vulnerability rule"""
        if 'cve_' in rule_id:
            return 'Update to patched version or apply security fixes'
        elif 'owasp_' in rule_id:
            return 'Follow OWASP security guidelines for this vulnerability category'
        
        return 'Apply appropriate security controls for this vulnerability type'
    
    async def generate_security_report(self, 
                                     analysis_issues: List[AnalysisIssue],
                                     report_type: str = 'summary') -> str:
        """Generate comprehensive security report"""
        return await self.report_generator.generate_report(analysis_issues, report_type)
    
    async def run_comprehensive_scan(self, 
                                   code: str, 
                                   language: Language,
                                   file_path: str = "",
                                   generate_report: bool = True) -> Dict[str, Any]:
        """Run comprehensive security scan with optional reporting"""
        
        # Run security analysis
        analysis_issues, security_issues = await self.analyze_security(code, language, file_path)
        
        # Generate report if requested
        report = None
        if generate_report:
            report = await self.generate_security_report(analysis_issues, 'detailed')
        
        # Calculate security metrics
        metrics = self._calculate_security_metrics(analysis_issues)
        
        return {
            'analysis_issues': analysis_issues,
            'security_issues': security_issues,
            'metrics': metrics,
            'report': report,
            'scan_timestamp': asyncio.get_event_loop().time(),
            'bandit_available': self.bandit_integration.bandit_available
        }
    
    def _calculate_security_metrics(self, issues: List[AnalysisIssue]) -> Dict[str, Any]:
        """Calculate security metrics from analysis issues"""
        
        total_issues = len(issues)
        critical_count = sum(1 for issue in issues if issue.severity == SeverityLevel.ERROR)
        high_count = sum(1 for issue in issues if issue.severity == SeverityLevel.WARNING)
        medium_count = sum(1 for issue in issues if issue.severity == SeverityLevel.INFO)
        
        # Calculate security score (0-100)
        security_score = max(0, 100 - (critical_count * 25 + high_count * 10 + medium_count * 5))
        
        # Risk level assessment
        if critical_count > 0:
            risk_level = "CRITICAL"
        elif high_count > 3:
            risk_level = "HIGH"
        elif high_count > 0 or medium_count > 5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Issue categories
        categories = {}
        for issue in issues:
            category = issue.rule_name
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'total_issues': total_issues,
            'critical_count': critical_count,
            'high_count': high_count,
            'medium_count': medium_count,
            'security_score': security_score,
            'risk_level': risk_level,
            'issue_categories': categories,
            'compliance_percentage': security_score
        }
    
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