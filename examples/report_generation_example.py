"""
Example demonstrating the review report generation functionality
"""

import asyncio
from xencode.features.code_review import ReportGenerator


async def main():
    """Demonstrate report generation"""
    
    # Sample review data
    review = {
        'summary': {
            'title': 'Add user authentication',
            'description': 'Implements JWT-based authentication',
            'files_analyzed': 3,
            'ai_summary': 'Good implementation with some security concerns that need attention'
        },
        'issues': [
            {
                'type': 'sqli',
                'severity': 'critical',
                'message': 'SQL injection vulnerability detected in user query',
                'file': 'auth/database.py',
                'line': 42,
                'column': 10
            },
            {
                'type': 'hardcoded_secret',
                'severity': 'high',
                'message': 'Hardcoded JWT secret key found',
                'file': 'config/settings.py',
                'line': 15,
                'column': 5
            },
            {
                'type': 'complexity',
                'severity': 'medium',
                'message': 'Function has high cyclomatic complexity (15)',
                'file': 'auth/validators.py',
                'line': 100,
                'column': 0
            }
        ],
        'suggestions': [
            {
                'title': 'Use Parameterized Queries',
                'description': 'Replace string concatenation with parameterized queries to prevent SQL injection',
                'severity': 'critical',
                'file': 'auth/database.py',
                'line': 42,
                'example': '''# Bad
query = f"SELECT * FROM users WHERE username = '{username}'"

# Good
query = "SELECT * FROM users WHERE username = ?"
cursor.execute(query, (username,))'''
            },
            {
                'title': 'Use Environment Variables for Secrets',
                'description': 'Store sensitive configuration in environment variables',
                'severity': 'high',
                'file': 'config/settings.py',
                'line': 15,
                'example': '''# Bad
JWT_SECRET = "hardcoded-secret-key"

# Good
import os
JWT_SECRET = os.environ.get("JWT_SECRET")'''
            }
        ],
        'patterns_detected': [
            {
                'type': 'complexity',
                'pattern': 'nested_structure',
                'file': 'auth/validators.py',
                'message': 'Deeply nested control structures detected (4 levels)'
            }
        ],
        'semantic_analysis': {
            'analysis': 'The authentication implementation follows modern best practices with JWT tokens. However, critical security issues need immediate attention before deployment.',
            'confidence': 0.88,
            'consensus_score': 0.94
        },
        'positive_feedback': [
            {
                'title': 'Excellent Test Coverage',
                'message': 'Comprehensive unit tests cover all authentication flows',
                'score': 95
            },
            {
                'title': 'Good Documentation',
                'message': 'Functions are well-documented with clear docstrings',
                'score': 90
            }
        ]
    }
    
    # Sample PR data
    pr_data = {
        'title': 'Add user authentication',
        'url': 'https://github.com/myorg/myapp/pull/123',
        'author': 'developer',
        'head_branch': 'feature/auth',
        'base_branch': 'main'
    }
    
    # Create report generator
    generator = ReportGenerator()
    
    print("=" * 80)
    print("REPORT GENERATION EXAMPLES")
    print("=" * 80)
    print()
    
    # Generate text report
    print("1. TEXT REPORT")
    print("-" * 80)
    text_report = generator.generate_text_report(review, pr_data)
    print(text_report)
    print()
    
    # Generate markdown report
    print("\n" + "=" * 80)
    print("2. MARKDOWN REPORT")
    print("-" * 80)
    markdown_report = generator.generate_markdown_report(review, pr_data)
    print(markdown_report)
    print()
    
    # Generate JSON report
    print("\n" + "=" * 80)
    print("3. JSON REPORT")
    print("-" * 80)
    import json
    json_report = generator.generate_json_report(review, pr_data)
    print(json.dumps(json_report, indent=2))
    print()
    
    # Save HTML report to file
    print("\n" + "=" * 80)
    print("4. HTML REPORT (saved to file)")
    print("-" * 80)
    html_report = generator.generate_html_report(review, pr_data)
    
    with open('review_report.html', 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print("HTML report saved to: review_report.html")
    print("Open this file in a web browser to view the formatted report.")
    print()


if __name__ == '__main__':
    asyncio.run(main())
