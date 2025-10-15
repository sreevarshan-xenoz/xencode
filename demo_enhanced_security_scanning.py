#!/usr/bin/env python3
"""
Demo: Enhanced Security Scanning with Bandit Integration

This demo showcases the enhanced security analyzer with:
- Bandit integration for Python security scanning
- Vulnerability database checks
- Comprehensive security reporting
- Multiple vulnerability detection methods
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from xencode.analyzers.security_analyzer import SecurityAnalyzer
from xencode.models.code_analysis import Language


async def demo_enhanced_security_scanning():
    """Demonstrate enhanced security scanning capabilities"""
    
    console = Console()
    console.print("🛡️ [bold cyan]Enhanced Security Scanning Demo[/bold cyan]\n")
    
    # Initialize the enhanced security analyzer
    analyzer = SecurityAnalyzer()
    
    # Check if Bandit is available
    bandit_status = "✅ Available" if analyzer.bandit_integration.bandit_available else "❌ Not Available"
    console.print(f"Bandit Integration: {bandit_status}")
    console.print(f"Vulnerability Database: ✅ Loaded")
    console.print(f"Security Report Generator: ✅ Ready\n")
    
    # Sample vulnerable code for demonstration
    vulnerable_python_code = '''
import os
import pickle
import hashlib
import subprocess
from flask import Flask, request
import yaml

app = Flask(__name__)
app.config['DEBUG'] = True  # Security issue: Debug mode in production

# Security issue: Hardcoded credentials
DATABASE_PASSWORD = "admin123"
API_KEY = "sk-1234567890abcdef"
SECRET_KEY = "super_secret_key"

@app.route('/execute')
def execute_command():
    cmd = request.args.get('cmd')
    # Security issue: Command injection
    os.system(cmd)
    subprocess.call(cmd, shell=True)
    return "Command executed"

@app.route('/load_data')
def load_data():
    data = request.args.get('data')
    # Security issue: Unsafe deserialization
    result = pickle.loads(data.encode())
    return str(result)

@app.route('/hash')
def hash_password():
    password = request.args.get('password')
    # Security issue: Weak cryptography
    hash_value = hashlib.md5(password.encode()).hexdigest()
    return hash_value

@app.route('/eval')
def eval_code():
    code = request.args.get('code')
    # Security issue: Code injection
    result = eval(code)
    return str(result)

@app.route('/config')
def load_config():
    config_file = request.args.get('config')
    # Security issue: Unsafe YAML loading
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    return str(config)

@app.route('/file')
def read_file():
    filename = request.args.get('file')
    # Security issue: Path traversal
    with open('/app/data/' + filename, 'r') as f:
        content = f.read()
    return content

# Security issue: Binding to all interfaces
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
'''
    
    vulnerable_javascript_code = '''
// Security issues in JavaScript code
const express = require('express');
const app = express();

// Hardcoded credentials
const API_KEY = "sk-1234567890abcdef";
const DATABASE_URL = "mongodb://admin:password123@localhost:27017/mydb";

app.get('/search', (req, res) => {
    const query = req.query.q;
    
    // XSS vulnerability
    res.send('<h1>Results for: ' + query + '</h1>');
    
    // Code injection
    eval('var result = ' + query);
    
    // Prototype pollution risk
    const userObj = JSON.parse(req.body.data);
    Object.assign({}, userObj);
});

app.get('/redirect', (req, res) => {
    const url = req.query.url;
    // Open redirect vulnerability
    res.redirect(url);
});

// Weak randomness
function generateToken() {
    return Math.random().toString(36).substring(2);
}

// Insecure direct object reference
app.get('/user/:id', (req, res) => {
    const userId = req.params.id;
    // No authorization check
    const user = database.getUser(userId);
    res.json(user);
});

app.listen(3000, '0.0.0.0');
'''
    
    # Demo 1: Python Security Analysis
    console.print("🐍 [bold yellow]Python Security Analysis[/bold yellow]")
    console.print("Analyzing vulnerable Python Flask application...\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running comprehensive security scan...", total=None)
        
        python_result = await analyzer.run_comprehensive_scan(
            vulnerable_python_code, 
            Language.PYTHON, 
            "vulnerable_app.py",
            generate_report=True
        )
        
        progress.update(task, completed=True)
    
    # Display Python results
    python_metrics = python_result['metrics']
    
    # Create metrics table
    metrics_table = Table(title="Python Security Scan Results")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="magenta")
    
    metrics_table.add_row("Total Issues", str(python_metrics['total_issues']))
    metrics_table.add_row("Critical Issues", str(python_metrics['critical_count']))
    metrics_table.add_row("High Severity", str(python_metrics['high_count']))
    metrics_table.add_row("Medium Severity", str(python_metrics['medium_count']))
    metrics_table.add_row("Security Score", f"{python_metrics['security_score']}/100")
    metrics_table.add_row("Risk Level", python_metrics['risk_level'])
    metrics_table.add_row("Compliance %", f"{python_metrics['compliance_percentage']}%")
    
    console.print(metrics_table)
    console.print()
    
    # Display top issues
    if python_result['analysis_issues']:
        console.print("🚨 [bold red]Top Security Issues Found:[/bold red]")
        
        issues_table = Table()
        issues_table.add_column("Severity", style="red")
        issues_table.add_column("Issue", style="yellow")
        issues_table.add_column("Line", style="cyan")
        issues_table.add_column("Rule", style="green")
        
        # Show top 10 issues
        for issue in python_result['analysis_issues'][:10]:
            severity_color = {
                'ERROR': '[red]CRITICAL[/red]',
                'WARNING': '[yellow]HIGH[/yellow]',
                'INFO': '[blue]MEDIUM[/blue]'
            }.get(issue.severity.value, issue.severity.value)
            
            issues_table.add_row(
                severity_color,
                issue.message[:50] + "..." if len(issue.message) > 50 else issue.message,
                str(issue.location.line),
                issue.rule_name
            )
        
        console.print(issues_table)
        console.print()
    
    # Demo 2: JavaScript Security Analysis
    console.print("🟨 [bold yellow]JavaScript Security Analysis[/bold yellow]")
    console.print("Analyzing vulnerable JavaScript Express application...\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running JavaScript security scan...", total=None)
        
        js_result = await analyzer.run_comprehensive_scan(
            vulnerable_javascript_code, 
            Language.JAVASCRIPT, 
            "vulnerable_app.js",
            generate_report=False
        )
        
        progress.update(task, completed=True)
    
    # Display JavaScript results
    js_metrics = js_result['metrics']
    
    js_metrics_table = Table(title="JavaScript Security Scan Results")
    js_metrics_table.add_column("Metric", style="cyan")
    js_metrics_table.add_column("Value", style="magenta")
    
    js_metrics_table.add_row("Total Issues", str(js_metrics['total_issues']))
    js_metrics_table.add_row("Critical Issues", str(js_metrics['critical_count']))
    js_metrics_table.add_row("High Severity", str(js_metrics['high_count']))
    js_metrics_table.add_row("Security Score", f"{js_metrics['security_score']}/100")
    js_metrics_table.add_row("Risk Level", js_metrics['risk_level'])
    
    console.print(js_metrics_table)
    console.print()
    
    # Demo 3: Security Report Generation
    console.print("📊 [bold green]Security Report Generation[/bold green]")
    
    if python_result['report']:
        # Show a portion of the detailed report
        report_lines = python_result['report'].split('\n')
        report_preview = '\n'.join(report_lines[:30])  # First 30 lines
        
        console.print(Panel(
            report_preview + "\n\n[dim]... (report truncated for demo)[/dim]",
            title="Security Analysis Report (Preview)",
            border_style="green"
        ))
        console.print()
    
    # Demo 4: Vulnerability Categories Analysis
    console.print("🔍 [bold blue]Vulnerability Categories Analysis[/bold blue]")
    
    # Combine issues from both scans
    all_issues = python_result['analysis_issues'] + js_result['analysis_issues']
    
    # Count issues by category
    category_counts = {}
    for issue in all_issues:
        category = issue.rule_name
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Create category table
    category_table = Table(title="Vulnerability Categories")
    category_table.add_column("Category", style="cyan")
    category_table.add_column("Count", style="magenta")
    category_table.add_column("Percentage", style="green")
    
    total_issues = len(all_issues)
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_issues) * 100 if total_issues > 0 else 0
        category_table.add_row(category, str(count), f"{percentage:.1f}%")
    
    console.print(category_table)
    console.print()
    
    # Demo 5: Security Recommendations
    console.print("💡 [bold green]Security Recommendations[/bold green]")
    
    recommendations = [
        "1. 🚨 Address all CRITICAL severity issues immediately",
        "2. 🔐 Remove hardcoded credentials and use environment variables",
        "3. 🛡️ Implement input validation and sanitization",
        "4. 🔒 Use parameterized queries to prevent SQL injection",
        "5. 🚫 Avoid eval() and similar code execution functions",
        "6. 🔑 Use strong cryptographic algorithms (SHA-256+)",
        "7. 🏗️ Implement proper error handling and logging",
        "8. 🔍 Set up automated security scanning in CI/CD pipeline",
        "9. 📚 Provide security training for development team",
        "10. 🔄 Regular security code reviews and penetration testing"
    ]
    
    for recommendation in recommendations:
        console.print(f"   {recommendation}")
    
    console.print()
    
    # Demo 6: Integration Status
    console.print("⚙️ [bold cyan]Integration Status[/bold cyan]")
    
    integration_table = Table()
    integration_table.add_column("Component", style="cyan")
    integration_table.add_column("Status", style="green")
    integration_table.add_column("Details", style="yellow")
    
    integration_table.add_row(
        "Bandit Integration",
        "✅ Active" if analyzer.bandit_integration.bandit_available else "❌ Inactive",
        "Python security scanning" if analyzer.bandit_integration.bandit_available else "Install bandit: pip install bandit"
    )
    integration_table.add_row(
        "Vulnerability Database",
        "✅ Active",
        f"CVE patterns loaded for {len(analyzer.vulnerability_db.cve_patterns)} languages"
    )
    integration_table.add_row(
        "OWASP Top 10",
        "✅ Active",
        f"{len(analyzer.vulnerability_db.owasp_top10)} categories covered"
    )
    integration_table.add_row(
        "Security Reporting",
        "✅ Active",
        "Summary, detailed, and executive reports available"
    )
    
    console.print(integration_table)
    console.print()
    
    # Summary
    total_python_issues = len(python_result['analysis_issues'])
    total_js_issues = len(js_result['analysis_issues'])
    
    console.print("📈 [bold magenta]Demo Summary[/bold magenta]")
    console.print(f"   • Python scan found {total_python_issues} security issues")
    console.print(f"   • JavaScript scan found {total_js_issues} security issues")
    console.print(f"   • Total vulnerabilities detected: {total_python_issues + total_js_issues}")
    console.print(f"   • Bandit integration: {'Available' if analyzer.bandit_integration.bandit_available else 'Not available'}")
    console.print(f"   • Security report generated: {'Yes' if python_result['report'] else 'No'}")
    
    console.print("\n🎉 [bold green]Enhanced Security Scanning Demo Complete![/bold green]")
    console.print("The enhanced security analyzer provides comprehensive vulnerability detection")
    console.print("with multiple scanning methods and detailed reporting capabilities.")


if __name__ == "__main__":
    asyncio.run(demo_enhanced_security_scanning())