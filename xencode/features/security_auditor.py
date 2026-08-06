"""
Security Auditor Feature

Provides proactive vulnerability scanning, dependency analysis, and security reporting.
Integrates with security tools like Bandit and Snyk to identify OWASP Top 10 vulnerabilities.
"""

import asyncio
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from xencode.features.base import FeatureBase, FeatureConfig


class RiskLevel(Enum):
    """Risk level classification"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityType(Enum):
    """OWASP Top 10 and common vulnerability types"""
    INJECTION = "injection"  # SQL, Command, etc.
    BROKEN_AUTH = "broken_authentication"
    SENSITIVE_DATA = "sensitive_data_exposure"
    XXE = "xml_external_entities"
    BROKEN_ACCESS = "broken_access_control"
    SECURITY_MISCONFIG = "security_misconfiguration"
    XSS = "cross_site_scripting"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    VULNERABLE_COMPONENTS = "vulnerable_components"
    INSUFFICIENT_LOGGING = "insufficient_logging"
    HARDCODED_SECRETS = "hardcoded_secrets"
    WEAK_CRYPTO = "weak_cryptography"
    PATH_TRAVERSAL = "path_traversal"
    SSRF = "server_side_request_forgery"


@dataclass
class SecurityAuditorConfig:
    """Configuration for security auditor"""
    enabled: bool = True
    checks: List[str] = field(default_factory=lambda: [
        "owasp_top_10",
        "dependency_vulnerabilities",
        "code_patterns"
    ])
    tools: List[str] = field(default_factory=lambda: ["bandit", "snyk"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*/tests/*",
        "*/test_*",
        "*/.venv/*",
        "*/node_modules/*"
    ])
    max_severity: str = "info"  # Report all severities by default

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityAuditorConfig':
        """Create config from dictionary"""
        return cls(
            enabled=data.get('enabled', True),
            checks=data.get('checks', cls().checks),
            tools=data.get('tools', cls().tools),
            exclude_patterns=data.get('exclude_patterns', cls().exclude_patterns),
            max_severity=data.get('max_severity', 'info')
        )


@dataclass
class Vulnerability:
    """Represents a security vulnerability"""
    id: str
    type: VulnerabilityType
    risk_level: RiskLevel
    title: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    fix_suggestion: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'type': self.type.value,
            'risk_level': self.risk_level.value,
            'title': self.title,
            'description': self.description,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'code_snippet': self.code_snippet,
            'fix_suggestion': self.fix_suggestion,
            'cwe_id': self.cwe_id,
            'cvss_score': self.cvss_score,
            'references': self.references
        }


@dataclass
class DependencyVulnerability:
    """Represents a dependency vulnerability"""
    package_name: str
    current_version: str
    vulnerable_versions: str
    fixed_version: Optional[str]
    cve_id: str
    risk_level: RiskLevel
    description: str
    cvss_score: float
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'package_name': self.package_name,
            'current_version': self.current_version,
            'vulnerable_versions': self.vulnerable_versions,
            'fixed_version': self.fixed_version,
            'cve_id': self.cve_id,
            'risk_level': self.risk_level.value,
            'description': self.description,
            'cvss_score': self.cvss_score,
            'references': self.references
        }


@dataclass
class SecurityReport:
    """Security audit report"""
    timestamp: str
    scan_path: str
    vulnerabilities: List[Vulnerability]
    dependency_vulnerabilities: List[DependencyVulnerability]
    summary: Dict[str, int]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'scan_path': self.scan_path,
            'vulnerabilities': [v.to_dict() for v in self.vulnerabilities],
            'dependency_vulnerabilities': [d.to_dict() for d in self.dependency_vulnerabilities],
            'summary': self.summary,
            'recommendations': self.recommendations
        }


class VulnerabilityScanner:
    """Scans code for OWASP Top 10 vulnerabilities"""

    def __init__(self, exclude_patterns: List[str] = None):
        self.exclude_patterns = exclude_patterns or []
        self._init_patterns()

    def _init_patterns(self):
        """Initialize vulnerability detection patterns"""
        self.patterns = {
            VulnerabilityType.INJECTION: [
                (r'execute\s*\([^)]*%s', 'SQL injection via string formatting'),
                (r'execute\s*\([^)]*\+', 'SQL injection via string concatenation'),
                (r'eval\s*\(', 'Code injection via eval()'),
                (r'exec\s*\(', 'Code injection via exec()'),
                (r'os\.system\s*\([^)]*\+', 'Command injection via string concatenation'),
                (r'subprocess\.(call|run|Popen)\s*\([^)]*\+', 'Command injection in subprocess'),
            ],
            VulnerabilityType.XSS: [
                (r'innerHTML\s*=', 'Potential XSS via innerHTML'),
                (r'document\.write\s*\(', 'Potential XSS via document.write'),
                (r'\.html\s*\([^)]*\+', 'Potential XSS in HTML injection'),
            ],
            VulnerabilityType.HARDCODED_SECRETS: [
                (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
                (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
                (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
                (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token'),
                (r'aws[_-]?access[_-]?key', 'AWS access key'),
            ],
            VulnerabilityType.WEAK_CRYPTO: [
                (r'md5\s*\(', 'Weak hash algorithm: MD5'),
                (r'sha1\s*\(', 'Weak hash algorithm: SHA1'),
                (r'DES\s*\(', 'Weak encryption: DES'),
                (r'random\.random\s*\(', 'Weak random number generator'),
            ],
            VulnerabilityType.PATH_TRAVERSAL: [
                (r'open\s*\([^)]*\+', 'Path traversal via string concatenation'),
                (r'\.\./', 'Potential path traversal pattern'),
            ],
            VulnerabilityType.INSECURE_DESERIALIZATION: [
                (r'pickle\.loads?\s*\(', 'Insecure deserialization with pickle'),
                (r'yaml\.load\s*\([^,)]*\)', 'Insecure YAML deserialization'),
            ],
            VulnerabilityType.BROKEN_ACCESS: [
                (r'@app\.route\s*\([^)]*\)\s*\n\s*def\s+\w+\s*\([^)]*\):\s*\n(?!\s*@)',
                 'Missing authentication decorator'),
            ],
        }

    async def scan(self, path: str) -> List[Vulnerability]:
        """Scan path for vulnerabilities"""
        vulnerabilities = []
        scan_path = Path(path)

        if scan_path.is_file():
            vulnerabilities.extend(await self._scan_file(scan_path))
        elif scan_path.is_dir():
            for file_path in scan_path.rglob('*.py'):
                if self._should_exclude(file_path):
                    continue
                vulnerabilities.extend(await self._scan_file(file_path))

        return vulnerabilities

    def _should_exclude(self, file_path: Path) -> bool:
        """Check if file should be excluded"""
        file_str = str(file_path).replace('\\', '/')  # Normalize path separators
        for pattern in self.exclude_patterns:
            pattern_regex = pattern.replace('*', '.*')
            if re.search(pattern_regex, file_str):
                return True
        return False

    async def _scan_file(self, file_path: Path) -> List[Vulnerability]:
        """Scan a single file for vulnerabilities"""
        vulnerabilities = []

        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            for vuln_type, patterns in self.patterns.items():
                for pattern, description in patterns:
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            vuln = self._create_vulnerability(
                                vuln_type=vuln_type,
                                file_path=str(file_path),
                                line_number=line_num,
                                code_snippet=line.strip(),
                                description=description
                            )
                            vulnerabilities.append(vuln)
        except Exception:
            # Skip files that can't be read
            pass

        return vulnerabilities

    def _create_vulnerability(self, vuln_type: VulnerabilityType,
                            file_path: str, line_number: int,
                            code_snippet: str, description: str) -> Vulnerability:
        """Create a vulnerability object"""
        # Determine risk level based on vulnerability type
        risk_map = {
            VulnerabilityType.INJECTION: RiskLevel.CRITICAL,
            VulnerabilityType.HARDCODED_SECRETS: RiskLevel.CRITICAL,
            VulnerabilityType.XSS: RiskLevel.HIGH,
            VulnerabilityType.INSECURE_DESERIALIZATION: RiskLevel.HIGH,
            VulnerabilityType.WEAK_CRYPTO: RiskLevel.MEDIUM,
            VulnerabilityType.PATH_TRAVERSAL: RiskLevel.HIGH,
            VulnerabilityType.BROKEN_ACCESS: RiskLevel.HIGH,
        }

        risk_level = risk_map.get(vuln_type, RiskLevel.MEDIUM)

        # Generate fix suggestion
        fix_suggestion = self._generate_fix_suggestion(vuln_type, code_snippet)

        vuln_id = f"{vuln_type.value}_{hash(f'{file_path}:{line_number}')}"

        return Vulnerability(
            id=vuln_id,
            type=vuln_type,
            risk_level=risk_level,
            title=description,
            description=f"{description} found in {file_path}",
            file_path=file_path,
            line_number=line_number,
            code_snippet=code_snippet,
            fix_suggestion=fix_suggestion
        )

    def _generate_fix_suggestion(self, vuln_type: VulnerabilityType,
                                 code_snippet: str) -> str:
        """Generate fix suggestion for vulnerability"""
        suggestions = {
            VulnerabilityType.INJECTION: (
                "Use parameterized queries or prepared statements. "
                "Example: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
            ),
            VulnerabilityType.HARDCODED_SECRETS: (
                "Store secrets in environment variables or a secure vault. "
                "Example: password = os.environ.get('DB_PASSWORD')"
            ),
            VulnerabilityType.XSS: (
                "Sanitize user input and use safe DOM manipulation methods. "
                "Example: element.textContent = userInput (instead of innerHTML)"
            ),
            VulnerabilityType.WEAK_CRYPTO: (
                "Use strong cryptographic algorithms like SHA-256 or bcrypt. "
                "Example: hashlib.sha256(data.encode()).hexdigest()"
            ),
            VulnerabilityType.PATH_TRAVERSAL: (
                "Validate and sanitize file paths. Use os.path.abspath() and check against allowed directories."
            ),
            VulnerabilityType.INSECURE_DESERIALIZATION: (
                "Use safe deserialization methods. For YAML, use yaml.safe_load(). "
                "Avoid pickle for untrusted data."
            ),
            VulnerabilityType.BROKEN_ACCESS: (
                "Add authentication and authorization checks. "
                "Example: @login_required decorator"
            ),
        }

        return suggestions.get(vuln_type, "Review and fix this security issue.")



class DependencyAnalyzer:
    """Analyzes dependencies for known CVEs and security issues"""

    def __init__(self):
        self.known_vulnerabilities = self._load_vulnerability_database()

    def _load_vulnerability_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load known vulnerability database (simplified for demo)"""
        # In production, this would load from a real CVE database
        return {
            'requests': [
                {
                    'cve_id': 'CVE-2023-32681',
                    'vulnerable_versions': '<2.31.0',
                    'fixed_version': '2.31.0',
                    'risk_level': 'medium',
                    'cvss_score': 6.1,
                    'description': 'Unintended leak of Proxy-Authorization header',
                }
            ],
            'django': [
                {
                    'cve_id': 'CVE-2023-43665',
                    'vulnerable_versions': '<4.2.6',
                    'fixed_version': '4.2.6',
                    'risk_level': 'high',
                    'cvss_score': 7.5,
                    'description': 'Denial-of-service in django.utils.text.Truncator',
                }
            ],
            'flask': [
                {
                    'cve_id': 'CVE-2023-30861',
                    'vulnerable_versions': '<2.3.2',
                    'fixed_version': '2.3.2',
                    'risk_level': 'high',
                    'cvss_score': 7.5,
                    'description': 'Cookie parsing vulnerability',
                }
            ],
            'pyyaml': [
                {
                    'cve_id': 'CVE-2020-14343',
                    'vulnerable_versions': '<5.4',
                    'fixed_version': '5.4',
                    'risk_level': 'critical',
                    'cvss_score': 9.8,
                    'description': 'Arbitrary code execution via unsafe yaml.load()',
                }
            ],
        }

    async def analyze(self, path: str) -> List[DependencyVulnerability]:
        """Analyze dependencies for vulnerabilities"""
        vulnerabilities = []
        scan_path = Path(path)

        # Find dependency files
        dep_files = []
        if scan_path.is_file():
            if scan_path.name in ['requirements.txt', 'Pipfile', 'package.json', 'pom.xml']:
                dep_files.append(scan_path)
        else:
            dep_files.extend(scan_path.glob('**/requirements.txt'))
            dep_files.extend(scan_path.glob('**/Pipfile'))
            dep_files.extend(scan_path.glob('**/package.json'))

        for dep_file in dep_files:
            vulnerabilities.extend(await self._analyze_file(dep_file))

        return vulnerabilities

    async def _analyze_file(self, file_path: Path) -> List[DependencyVulnerability]:
        """Analyze a dependency file"""
        vulnerabilities = []

        try:
            if file_path.name == 'requirements.txt':
                vulnerabilities.extend(await self._analyze_requirements(file_path))
            elif file_path.name == 'package.json':
                vulnerabilities.extend(await self._analyze_package_json(file_path))
        except Exception:
            pass

        return vulnerabilities

    async def _analyze_requirements(self, file_path: Path) -> List[DependencyVulnerability]:
        """Analyze Python requirements.txt"""
        vulnerabilities = []
        content = file_path.read_text(encoding='utf-8')

        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse package name and version
            match = re.match(r'([a-zA-Z0-9_-]+)([=<>!]+)?([\d.]+)?', line)
            if match:
                package_name = match.group(1).lower()
                version = match.group(3) if match.group(3) else 'unknown'

                # Check against vulnerability database
                if package_name in self.known_vulnerabilities:
                    for vuln_data in self.known_vulnerabilities[package_name]:
                        if self._is_vulnerable(version, vuln_data['vulnerable_versions']):
                            vuln = DependencyVulnerability(
                                package_name=package_name,
                                current_version=version,
                                vulnerable_versions=vuln_data['vulnerable_versions'],
                                fixed_version=vuln_data['fixed_version'],
                                cve_id=vuln_data['cve_id'],
                                risk_level=RiskLevel(vuln_data['risk_level']),
                                description=vuln_data['description'],
                                cvss_score=vuln_data['cvss_score'],
                                references=[f"https://nvd.nist.gov/vuln/detail/{vuln_data['cve_id']}"]
                            )
                            vulnerabilities.append(vuln)

        return vulnerabilities

    async def _analyze_package_json(self, file_path: Path) -> List[DependencyVulnerability]:
        """Analyze Node.js package.json"""
        vulnerabilities = []

        try:
            content = json.loads(file_path.read_text(encoding='utf-8'))
            dependencies = content.get('dependencies', {})

            for package_name, version in dependencies.items():
                # Clean version string
                version = version.lstrip('^~')

                # Check against vulnerability database
                if package_name in self.known_vulnerabilities:
                    for vuln_data in self.known_vulnerabilities[package_name]:
                        if self._is_vulnerable(version, vuln_data['vulnerable_versions']):
                            vuln = DependencyVulnerability(
                                package_name=package_name,
                                current_version=version,
                                vulnerable_versions=vuln_data['vulnerable_versions'],
                                fixed_version=vuln_data['fixed_version'],
                                cve_id=vuln_data['cve_id'],
                                risk_level=RiskLevel(vuln_data['risk_level']),
                                description=vuln_data['description'],
                                cvss_score=vuln_data['cvss_score'],
                                references=[f"https://nvd.nist.gov/vuln/detail/{vuln_data['cve_id']}"]
                            )
                            vulnerabilities.append(vuln)
        except Exception:
            pass

        return vulnerabilities

    def _is_vulnerable(self, current_version: str, vulnerable_range: str) -> bool:
        """Check if current version is in vulnerable range"""
        if current_version == 'unknown':
            return True  # Assume vulnerable if version unknown

        # Simple version comparison (simplified for demo)
        try:
            if vulnerable_range.startswith('<'):
                threshold = vulnerable_range.lstrip('<')
                return self._compare_versions(current_version, threshold) < 0
            elif vulnerable_range.startswith('<='):
                threshold = vulnerable_range.lstrip('<=')
                return self._compare_versions(current_version, threshold) <= 0
        except Exception:
            return False

        return False

    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings"""
        parts1 = [int(x) for x in v1.split('.')]
        parts2 = [int(x) for x in v2.split('.')]

        # Pad to same length
        max_len = max(len(parts1), len(parts2))
        parts1.extend([0] * (max_len - len(parts1)))
        parts2.extend([0] * (max_len - len(parts2)))

        for p1, p2 in zip(parts1, parts2):
            if p1 < p2:
                return -1
            elif p1 > p2:
                return 1

        return 0



class SecurityReportGenerator:
    """Generates security audit reports"""

    def __init__(self):
        pass

    async def generate(self, scan_path: str,
                      vulnerabilities: List[Vulnerability],
                      dependency_vulnerabilities: List[DependencyVulnerability]) -> SecurityReport:
        """Generate comprehensive security report"""
        timestamp = datetime.now().isoformat()

        # Calculate summary statistics
        summary = self._calculate_summary(vulnerabilities, dependency_vulnerabilities)

        # Generate recommendations
        recommendations = self._generate_recommendations(vulnerabilities, dependency_vulnerabilities)

        return SecurityReport(
            timestamp=timestamp,
            scan_path=scan_path,
            vulnerabilities=vulnerabilities,
            dependency_vulnerabilities=dependency_vulnerabilities,
            summary=summary,
            recommendations=recommendations
        )

    def _calculate_summary(self, vulnerabilities: List[Vulnerability],
                          dependency_vulnerabilities: List[DependencyVulnerability]) -> Dict[str, int]:
        """Calculate summary statistics"""
        summary = {
            'total_vulnerabilities': len(vulnerabilities) + len(dependency_vulnerabilities),
            'code_vulnerabilities': len(vulnerabilities),
            'dependency_vulnerabilities': len(dependency_vulnerabilities),
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0
        }

        # Count by risk level
        for vuln in vulnerabilities:
            summary[vuln.risk_level.value] += 1

        for dep_vuln in dependency_vulnerabilities:
            summary[dep_vuln.risk_level.value] += 1

        return summary

    def _generate_recommendations(self, vulnerabilities: List[Vulnerability],
                                 dependency_vulnerabilities: List[DependencyVulnerability]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []

        # Check for critical issues
        critical_count = sum(1 for v in vulnerabilities if v.risk_level == RiskLevel.CRITICAL)
        critical_count += sum(1 for d in dependency_vulnerabilities if d.risk_level == RiskLevel.CRITICAL)

        if critical_count > 0:
            recommendations.append(
                f"⚠️  URGENT: {critical_count} critical vulnerabilities found. "
                "Address these immediately before deploying to production."
            )

        # Check for injection vulnerabilities
        injection_vulns = [v for v in vulnerabilities if v.type == VulnerabilityType.INJECTION]
        if injection_vulns:
            recommendations.append(
                f"🔒 Found {len(injection_vulns)} injection vulnerabilities. "
                "Always use parameterized queries and input validation."
            )

        # Check for hardcoded secrets
        secret_vulns = [v for v in vulnerabilities if v.type == VulnerabilityType.HARDCODED_SECRETS]
        if secret_vulns:
            recommendations.append(
                f"🔑 Found {len(secret_vulns)} hardcoded secrets. "
                "Move all secrets to environment variables or a secure vault."
            )

        # Check for outdated dependencies
        if dependency_vulnerabilities:
            recommendations.append(
                f"📦 {len(dependency_vulnerabilities)} vulnerable dependencies found. "
                "Update to the latest secure versions."
            )

        # General recommendations
        if not recommendations:
            recommendations.append("✅ No critical security issues found. Continue following security best practices.")
        else:
            recommendations.append(
                "📚 Review OWASP Top 10 guidelines: https://owasp.org/www-project-top-ten/"
            )
            recommendations.append(
                "🔍 Consider implementing automated security testing in your CI/CD pipeline."
            )

        return recommendations

    async def generate_markdown(self, report: SecurityReport) -> str:
        """Generate markdown format report"""
        md = []
        md.append("# Security Audit Report\n")
        md.append(f"**Generated:** {report.timestamp}\n")
        md.append(f"**Scan Path:** {report.scan_path}\n")
        md.append("\n## Summary\n")
        md.append(f"- **Total Vulnerabilities:** {report.summary['total_vulnerabilities']}")
        md.append(f"- **Code Vulnerabilities:** {report.summary['code_vulnerabilities']}")
        md.append(f"- **Dependency Vulnerabilities:** {report.summary['dependency_vulnerabilities']}")
        md.append("\n### By Risk Level\n")
        md.append(f"- 🔴 **Critical:** {report.summary['critical']}")
        md.append(f"- 🟠 **High:** {report.summary['high']}")
        md.append(f"- 🟡 **Medium:** {report.summary['medium']}")
        md.append(f"- 🟢 **Low:** {report.summary['low']}")
        md.append(f"- ℹ️  **Info:** {report.summary['info']}")

        # Code vulnerabilities
        if report.vulnerabilities:
            md.append("\n## Code Vulnerabilities\n")
            for vuln in sorted(report.vulnerabilities, key=lambda v: v.risk_level.value):
                md.append(f"\n### {vuln.title}")
                md.append(f"- **Risk Level:** {vuln.risk_level.value.upper()}")
                md.append(f"- **Type:** {vuln.type.value}")
                md.append(f"- **Location:** {vuln.file_path}:{vuln.line_number}")
                md.append(f"- **Code:** `{vuln.code_snippet}`")
                md.append(f"- **Fix:** {vuln.fix_suggestion}")

        # Dependency vulnerabilities
        if report.dependency_vulnerabilities:
            md.append("\n## Dependency Vulnerabilities\n")
            for dep_vuln in sorted(report.dependency_vulnerabilities,
                                  key=lambda d: d.risk_level.value):
                md.append(f"\n### {dep_vuln.package_name} - {dep_vuln.cve_id}")
                md.append(f"- **Risk Level:** {dep_vuln.risk_level.value.upper()}")
                md.append(f"- **Current Version:** {dep_vuln.current_version}")
                md.append(f"- **Vulnerable Versions:** {dep_vuln.vulnerable_versions}")
                md.append(f"- **Fixed Version:** {dep_vuln.fixed_version}")
                md.append(f"- **CVSS Score:** {dep_vuln.cvss_score}")
                md.append(f"- **Description:** {dep_vuln.description}")
                if dep_vuln.references:
                    md.append(f"- **References:** {', '.join(dep_vuln.references)}")

        # Recommendations
        md.append("\n## Recommendations\n")
        for rec in report.recommendations:
            md.append(f"- {rec}")

        return '\n'.join(md)

    async def generate_html(self, report: SecurityReport) -> str:
        """Generate HTML format report"""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html><head>")
        html.append("<title>Security Audit Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append(".critical { color: #d32f2f; }")
        html.append(".high { color: #f57c00; }")
        html.append(".medium { color: #fbc02d; }")
        html.append(".low { color: #388e3c; }")
        html.append(".vuln { border: 1px solid #ddd; padding: 10px; margin: 10px 0; }")
        html.append("</style>")
        html.append("</head><body>")
        html.append("<h1>Security Audit Report</h1>")
        html.append(f"<p><strong>Generated:</strong> {report.timestamp}</p>")
        html.append(f"<p><strong>Scan Path:</strong> {report.scan_path}</p>")

        html.append("<h2>Summary</h2>")
        html.append(f"<p>Total Vulnerabilities: {report.summary['total_vulnerabilities']}</p>")
        html.append(f"<p class='critical'>Critical: {report.summary['critical']}</p>")
        html.append(f"<p class='high'>High: {report.summary['high']}</p>")
        html.append(f"<p class='medium'>Medium: {report.summary['medium']}</p>")
        html.append(f"<p class='low'>Low: {report.summary['low']}</p>")

        if report.vulnerabilities:
            html.append("<h2>Code Vulnerabilities</h2>")
            for vuln in report.vulnerabilities:
                html.append(f"<div class='vuln {vuln.risk_level.value}'>")
                html.append(f"<h3>{vuln.title}</h3>")
                html.append(f"<p><strong>Location:</strong> {vuln.file_path}:{vuln.line_number}</p>")
                html.append(f"<p><strong>Code:</strong> <code>{vuln.code_snippet}</code></p>")
                html.append(f"<p><strong>Fix:</strong> {vuln.fix_suggestion}</p>")
                html.append("</div>")

        html.append("</body></html>")
        return '\n'.join(html)



class SecurityToolIntegration:
    """Integrates with external security tools like Bandit and Snyk"""

    def __init__(self):
        self.tools_available = self._check_tools()

    def _check_tools(self) -> Dict[str, bool]:
        """Check which security tools are available"""
        tools = {}

        # Check for Bandit
        try:
            result = subprocess.run(['bandit', '--version'],
                                  capture_output=True, text=True, timeout=5)
            tools['bandit'] = result.returncode == 0
        except Exception:
            tools['bandit'] = False

        # Check for Snyk
        try:
            result = subprocess.run(['snyk', '--version'],
                                  capture_output=True, text=True, timeout=5)
            tools['snyk'] = result.returncode == 0
        except Exception:
            tools['snyk'] = False

        return tools

    async def run_bandit(self, path: str) -> List[Vulnerability]:
        """Run Bandit security scanner for Python"""
        if not self.tools_available.get('bandit', False):
            return []

        vulnerabilities = []

        try:
            # Run Bandit with JSON output
            result = subprocess.run(
                ['bandit', '-r', path, '-f', 'json'],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.stdout:
                data = json.loads(result.stdout)

                for issue in data.get('results', []):
                    # Map Bandit severity to our risk levels
                    severity_map = {
                        'HIGH': RiskLevel.HIGH,
                        'MEDIUM': RiskLevel.MEDIUM,
                        'LOW': RiskLevel.LOW
                    }

                    risk_level = severity_map.get(issue.get('issue_severity', 'MEDIUM'),
                                                  RiskLevel.MEDIUM)

                    # Map Bandit issue types to our vulnerability types
                    vuln_type = self._map_bandit_issue_type(issue.get('test_id', ''))

                    vuln = Vulnerability(
                        id=f"bandit_{issue.get('test_id', '')}_{hash(issue.get('filename', ''))}",
                        type=vuln_type,
                        risk_level=risk_level,
                        title=issue.get('issue_text', 'Security issue'),
                        description=issue.get('issue_text', ''),
                        file_path=issue.get('filename', ''),
                        line_number=issue.get('line_number', 0),
                        code_snippet=issue.get('code', '').strip(),
                        fix_suggestion=issue.get('more_info', 'Review Bandit documentation'),
                        cwe_id=issue.get('test_id', None),
                        references=[issue.get('more_info', '')]
                    )
                    vulnerabilities.append(vuln)

        except Exception:
            # Tool execution failed, return empty list
            pass

        return vulnerabilities

    def _map_bandit_issue_type(self, test_id: str) -> VulnerabilityType:
        """Map Bandit test ID to vulnerability type"""
        mapping = {
            'B201': VulnerabilityType.INJECTION,  # flask_debug_true
            'B301': VulnerabilityType.INSECURE_DESERIALIZATION,  # pickle
            'B302': VulnerabilityType.WEAK_CRYPTO,  # marshal
            'B303': VulnerabilityType.WEAK_CRYPTO,  # md5
            'B304': VulnerabilityType.WEAK_CRYPTO,  # insecure_cipher
            'B305': VulnerabilityType.WEAK_CRYPTO,  # insecure_cipher_mode
            'B306': VulnerabilityType.WEAK_CRYPTO,  # mktemp_q
            'B307': VulnerabilityType.INJECTION,  # eval
            'B308': VulnerabilityType.BROKEN_AUTH,  # mark_safe
            'B310': VulnerabilityType.PATH_TRAVERSAL,  # urllib_urlopen
            'B311': VulnerabilityType.WEAK_CRYPTO,  # random
            'B312': VulnerabilityType.WEAK_CRYPTO,  # telnetlib
            'B313': VulnerabilityType.XSS,  # xml_bad_cElementTree
            'B314': VulnerabilityType.XXE,  # xml_bad_ElementTree
            'B315': VulnerabilityType.XXE,  # xml_bad_expatreader
            'B316': VulnerabilityType.XXE,  # xml_bad_expatbuilder
            'B317': VulnerabilityType.XXE,  # xml_bad_sax
            'B318': VulnerabilityType.XXE,  # xml_bad_minidom
            'B319': VulnerabilityType.XXE,  # xml_bad_pulldom
            'B320': VulnerabilityType.XXE,  # xml_bad_etree
            'B321': VulnerabilityType.INSECURE_DESERIALIZATION,  # ftplib
            'B322': VulnerabilityType.INJECTION,  # input
            'B323': VulnerabilityType.INSECURE_DESERIALIZATION,  # unverified_context
            'B324': VulnerabilityType.WEAK_CRYPTO,  # hashlib
            'B501': VulnerabilityType.BROKEN_AUTH,  # request_with_no_cert_validation
            'B502': VulnerabilityType.BROKEN_AUTH,  # ssl_with_bad_version
            'B503': VulnerabilityType.BROKEN_AUTH,  # ssl_with_bad_defaults
            'B504': VulnerabilityType.BROKEN_AUTH,  # ssl_with_no_version
            'B505': VulnerabilityType.WEAK_CRYPTO,  # weak_cryptographic_key
            'B506': VulnerabilityType.INSECURE_DESERIALIZATION,  # yaml_load
            'B507': VulnerabilityType.BROKEN_AUTH,  # ssh_no_host_key_verification
            'B601': VulnerabilityType.INJECTION,  # paramiko_calls
            'B602': VulnerabilityType.INJECTION,  # subprocess_popen_with_shell_equals_true
            'B603': VulnerabilityType.INJECTION,  # subprocess_without_shell_equals_true
            'B604': VulnerabilityType.INJECTION,  # any_other_function_with_shell_equals_true
            'B605': VulnerabilityType.INJECTION,  # start_process_with_a_shell
            'B606': VulnerabilityType.INJECTION,  # start_process_with_no_shell
            'B607': VulnerabilityType.INJECTION,  # start_process_with_partial_path
            'B608': VulnerabilityType.INJECTION,  # hardcoded_sql_expressions
            'B609': VulnerabilityType.INJECTION,  # linux_commands_wildcard_injection
        }

        return mapping.get(test_id, VulnerabilityType.SECURITY_MISCONFIG)

    async def run_snyk(self, path: str) -> List[DependencyVulnerability]:
        """Run Snyk dependency scanner"""
        if not self.tools_available.get('snyk', False):
            return []

        vulnerabilities = []

        try:
            # Run Snyk with JSON output
            result = subprocess.run(
                ['snyk', 'test', '--json', path],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.stdout:
                data = json.loads(result.stdout)

                for vuln in data.get('vulnerabilities', []):
                    # Map Snyk severity to our risk levels
                    severity_map = {
                        'critical': RiskLevel.CRITICAL,
                        'high': RiskLevel.HIGH,
                        'medium': RiskLevel.MEDIUM,
                        'low': RiskLevel.LOW
                    }

                    risk_level = severity_map.get(vuln.get('severity', 'medium').lower(),
                                                  RiskLevel.MEDIUM)

                    dep_vuln = DependencyVulnerability(
                        package_name=vuln.get('packageName', ''),
                        current_version=vuln.get('version', ''),
                        vulnerable_versions=vuln.get('semver', {}).get('vulnerable', ''),
                        fixed_version=vuln.get('fixedIn', [None])[0] if vuln.get('fixedIn') else None,
                        cve_id=vuln.get('identifiers', {}).get('CVE', [''])[0] or vuln.get('id', ''),
                        risk_level=risk_level,
                        description=vuln.get('title', ''),
                        cvss_score=vuln.get('cvssScore', 0.0),
                        references=[vuln.get('url', '')]
                    )
                    vulnerabilities.append(dep_vuln)

        except Exception:
            # Tool execution failed, return empty list
            pass

        return vulnerabilities



class SecurityAuditor(FeatureBase):
    """Main security auditor feature class"""

    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.auditor_config = SecurityAuditorConfig.from_dict(config.config)
        self.scanner = VulnerabilityScanner(self.auditor_config.exclude_patterns)
        self.dependency_analyzer = DependencyAnalyzer()
        self.report_generator = SecurityReportGenerator()
        self.tool_integration = SecurityToolIntegration()

    @property
    def name(self) -> str:
        """Feature name"""
        return "security_auditor"

    @property
    def description(self) -> str:
        """Feature description"""
        return "Proactive vulnerability scanning and security auditing"

    async def _initialize(self) -> None:
        """Initialize the security auditor"""
        pass

    async def _shutdown(self) -> None:
        """Shutdown the security auditor"""
        pass

    async def scan(self, path: str, use_external_tools: bool = True) -> Dict[str, Any]:
        """
        Scan code for security vulnerabilities

        Args:
            path: Path to scan (file or directory)
            use_external_tools: Whether to use external tools like Bandit/Snyk

        Returns:
            Dictionary with scan results
        """
        # Run internal vulnerability scanner
        vulnerabilities = await self.scanner.scan(path)

        # Run external tools if enabled
        if use_external_tools and 'bandit' in self.auditor_config.tools:
            bandit_vulns = await self.tool_integration.run_bandit(path)
            vulnerabilities.extend(bandit_vulns)

        # Analyze dependencies
        dependency_vulnerabilities = await self.dependency_analyzer.analyze(path)

        # Run Snyk if enabled
        if use_external_tools and 'snyk' in self.auditor_config.tools:
            snyk_vulns = await self.tool_integration.run_snyk(path)
            dependency_vulnerabilities.extend(snyk_vulns)

        # Generate report
        report = await self.report_generator.generate(
            scan_path=path,
            vulnerabilities=vulnerabilities,
            dependency_vulnerabilities=dependency_vulnerabilities
        )

        return {
            'success': True,
            'report': report.to_dict(),
            'summary': report.summary,
            'recommendations': report.recommendations
        }

    async def analyze(self, path: str) -> Dict[str, Any]:
        """
        Analyze dependencies for security issues

        Args:
            path: Path to analyze

        Returns:
            Dictionary with analysis results
        """
        dependency_vulnerabilities = await self.dependency_analyzer.analyze(path)

        return {
            'success': True,
            'vulnerabilities': [v.to_dict() for v in dependency_vulnerabilities],
            'count': len(dependency_vulnerabilities)
        }

    async def report(self, path: str, format: str = 'markdown') -> Dict[str, Any]:
        """
        Generate security report

        Args:
            path: Path to scan
            format: Report format ('markdown' or 'html')

        Returns:
            Dictionary with report content
        """
        # Run full scan
        scan_result = await self.scan(path)
        report_data = scan_result['report']

        # Reconstruct report object
        report = SecurityReport(
            timestamp=report_data['timestamp'],
            scan_path=report_data['scan_path'],
            vulnerabilities=[
                Vulnerability(**{**v, 'type': VulnerabilityType(v['type']),
                               'risk_level': RiskLevel(v['risk_level'])})
                for v in report_data['vulnerabilities']
            ],
            dependency_vulnerabilities=[
                DependencyVulnerability(**{**d, 'risk_level': RiskLevel(d['risk_level'])})
                for d in report_data['dependency_vulnerabilities']
            ],
            summary=report_data['summary'],
            recommendations=report_data['recommendations']
        )

        # Generate formatted report
        if format == 'markdown':
            content = await self.report_generator.generate_markdown(report)
        elif format == 'html':
            content = await self.report_generator.generate_html(report)
        else:
            return {
                'success': False,
                'error': f"Unsupported format: {format}"
            }

        return {
            'success': True,
            'format': format,
            'content': content
        }

    async def fix_suggestions(self, vulnerability_id: str) -> Dict[str, Any]:
        """
        Get detailed fix suggestions for a vulnerability

        Args:
            vulnerability_id: ID of the vulnerability

        Returns:
            Dictionary with fix suggestions
        """
        # This would look up the vulnerability and provide detailed fixes
        # For now, return a placeholder
        return {
            'success': True,
            'vulnerability_id': vulnerability_id,
            'suggestions': [
                'Review the code and apply the recommended fix',
                'Test the fix thoroughly',
                'Consider adding automated tests to prevent regression'
            ]
        }


    def get_cli_commands(self) -> List[Any]:
        """Get CLI commands for security auditor"""
        import click

        @click.group(name='security')
        def security_group():
            """Security auditing and vulnerability scanning"""
            pass

        @security_group.command(name='scan')
        @click.argument('path', type=click.Path(exists=True))
        @click.option('--no-external-tools', is_flag=True,
                     help='Disable external tools (Bandit, Snyk)')
        @click.option('--format', type=click.Choice(['json', 'text']),
                     default='text', help='Output format')
        def scan_cmd(path: str, no_external_tools: bool, format: str):
            """Scan code for security vulnerabilities"""

            async def run_scan():
                result = await self.scan(path, use_external_tools=not no_external_tools)

                if format == 'json':
                    click.echo(json.dumps(result, indent=2))
                else:
                    report = result['report']
                    click.echo("\n🔒 Security Scan Results")
                    click.echo(f"{'=' * 50}")
                    click.echo(f"Path: {report['scan_path']}")
                    click.echo(f"Timestamp: {report['timestamp']}")
                    click.echo("\n📊 Summary:")
                    click.echo(f"  Total Vulnerabilities: {report['summary']['total_vulnerabilities']}")
                    click.echo(f"  🔴 Critical: {report['summary']['critical']}")
                    click.echo(f"  🟠 High: {report['summary']['high']}")
                    click.echo(f"  🟡 Medium: {report['summary']['medium']}")
                    click.echo(f"  🟢 Low: {report['summary']['low']}")

                    if report['vulnerabilities']:
                        click.echo("\n🐛 Code Vulnerabilities:")
                        for vuln in report['vulnerabilities'][:5]:  # Show first 5
                            click.echo(f"  - [{vuln['risk_level'].upper()}] {vuln['title']}")
                            click.echo(f"    Location: {vuln['file_path']}:{vuln['line_number']}")

                    if report['dependency_vulnerabilities']:
                        click.echo("\n📦 Dependency Vulnerabilities:")
                        for dep in report['dependency_vulnerabilities'][:5]:  # Show first 5
                            click.echo(f"  - [{dep['risk_level'].upper()}] {dep['package_name']} {dep['cve_id']}")
                            click.echo(f"    Current: {dep['current_version']}, Fixed: {dep['fixed_version']}")

                    click.echo("\n💡 Recommendations:")
                    for rec in report['recommendations']:
                        click.echo(f"  {rec}")

            asyncio.run(run_scan())

        @security_group.command(name='dependencies')
        @click.argument('path', type=click.Path(exists=True))
        @click.option('--format', type=click.Choice(['json', 'text']),
                     default='text', help='Output format')
        def dependencies_cmd(path: str, format: str):
            """Analyze dependencies for vulnerabilities"""

            async def run_analysis():
                result = await self.analyze(path)

                if format == 'json':
                    click.echo(json.dumps(result, indent=2))
                else:
                    click.echo("\n📦 Dependency Analysis")
                    click.echo(f"{'=' * 50}")
                    click.echo(f"Found {result['count']} vulnerable dependencies\n")

                    for vuln in result['vulnerabilities']:
                        click.echo(f"Package: {vuln['package_name']}")
                        click.echo(f"  Risk: {vuln['risk_level'].upper()}")
                        click.echo(f"  Current: {vuln['current_version']}")
                        click.echo(f"  Fixed: {vuln['fixed_version']}")
                        click.echo(f"  CVE: {vuln['cve_id']}")
                        click.echo(f"  Description: {vuln['description']}\n")

            asyncio.run(run_analysis())

        @security_group.command(name='report')
        @click.argument('path', type=click.Path(exists=True))
        @click.option('--format', type=click.Choice(['markdown', 'html']),
                     default='markdown', help='Report format')
        @click.option('--output', type=click.Path(), help='Output file path')
        def report_cmd(path: str, format: str, output: str):
            """Generate security audit report"""

            async def run_report():
                result = await self.report(path, format=format)

                if result['success']:
                    if output:
                        Path(output).write_text(result['content'])
                        click.echo(f"✅ Report saved to {output}")
                    else:
                        click.echo(result['content'])
                else:
                    click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")

            asyncio.run(run_report())

        @security_group.command(name='audit')
        @click.argument('path', type=click.Path(exists=True))
        def audit_cmd(path: str):
            """Run full security audit"""

            async def run_audit():
                click.echo("🔍 Running full security audit...")
                click.echo("This may take a few minutes...\n")

                result = await self.scan(path, use_external_tools=True)
                report = result['report']

                click.echo("✅ Audit complete!")
                click.echo("\n📊 Results:")
                click.echo(f"  Total Issues: {report['summary']['total_vulnerabilities']}")
                click.echo(f"  Critical: {report['summary']['critical']}")
                click.echo(f"  High: {report['summary']['high']}")
                click.echo(f"  Medium: {report['summary']['medium']}")
                click.echo(f"  Low: {report['summary']['low']}")

                if report['summary']['critical'] > 0:
                    click.echo(f"\n⚠️  WARNING: {report['summary']['critical']} critical vulnerabilities found!")
                    click.echo("Address these immediately before deploying to production.")

            asyncio.run(run_audit())

        return [security_group]

    def get_tui_components(self) -> List[Any]:
        """Get TUI components for security auditor"""
        # TUI components would be implemented here
        return []

    def get_api_endpoints(self) -> List[Any]:
        """Get API endpoints for security auditor"""
        return [
            {
                'path': '/api/security/scan',
                'method': 'POST',
                'handler': self.scan,
            },
            {
                'path': '/api/security/analyze',
                'method': 'POST',
                'handler': self.analyze,
            },
            {
                'path': '/api/security/report',
                'method': 'POST',
                'handler': self.report,
            },
            {
                'path': '/api/security/fix-suggestions',
                'method': 'POST',
                'handler': self.fix_suggestions,
            },
        ]
