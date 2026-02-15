"""
Integration tests for AI Review Engine

Tests the complete flow of the AI review engine with real code examples.
"""

import pytest
from xencode.features.code_review import CodeReviewFeature, AIReviewEngine, CodeLinter
from xencode.features.base import FeatureConfig


class TestAIReviewEngineIntegration:
    """Integration tests for AI Review Engine"""
    
    @pytest.fixture
    async def feature(self):
        """Create and initialize code review feature"""
        config = FeatureConfig(
            name="code_review",
            enabled=True,
            config={}
        )
        
        feature = CodeReviewFeature(config)
        await feature._initialize()
        return feature
    
    @pytest.mark.asyncio
    async def test_complete_review_flow(self, feature):
        """Test complete review flow from code analysis to AI suggestions"""
        # Sample code with multiple issues
        files = [
            {
                'path': 'auth.py',
                'content': '''
def authenticate(username, password):
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    
    # Hardcoded secret
    api_key = "sk_live_1234567890abcdef"
    
    # Weak crypto
    import hashlib
    hash = hashlib.md5(password.encode()).hexdigest()
    
    return True
''',
                'language': 'python'
            },
            {
                'path': 'frontend.js',
                'content': '''
function displayMessage(msg) {
    // XSS vulnerability
    document.getElementById('output').innerHTML = msg;
    
    // Weak comparison
    if (value == 5) {
        console.log('equal');
    }
    
    // Weak random
    const token = Math.random().toString(36);
}
''',
                'language': 'javascript'
            }
        ]
        
        # Run code linter
        linter = CodeLinter()
        code_analysis = await linter.analyze(files)
        
        # Verify linter found issues
        assert code_analysis['summary']['total_issues'] > 0
        assert code_analysis['summary']['by_severity']['critical'] > 0
        
        # Run AI review engine
        engine = AIReviewEngine()
        await engine.initialize()
        
        review = await engine.generate_review(
            'Security fixes',
            'Fixing multiple security vulnerabilities',
            files,
            code_analysis
        )
        
        # Verify review structure
        assert 'summary' in review
        assert 'issues' in review
        assert 'suggestions' in review
        assert 'patterns_detected' in review
        assert 'semantic_analysis' in review
        
        # Verify issues were captured
        assert len(review['issues']) > 0
        
        # Verify suggestions were generated
        assert len(review['suggestions']) > 0
        
        # Verify critical issues have suggestions
        critical_issues = [i for i in review['issues'] if i['severity'] == 'critical']
        assert len(critical_issues) > 0
        
        # Verify SQL injection suggestion
        sqli_suggestions = [s for s in review['suggestions'] if 'SQL' in s.get('title', '')]
        assert len(sqli_suggestions) > 0
        assert 'parameterized' in sqli_suggestions[0]['description'].lower()
        
        # Verify XSS suggestion
        xss_suggestions = [s for s in review['suggestions'] if 'XSS' in s.get('title', '')]
        assert len(xss_suggestions) > 0
        
        # Verify hardcoded secrets suggestion
        secret_suggestions = [s for s in review['suggestions'] if 'Secret' in s.get('title', '')]
        assert len(secret_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_review_clean_code(self, feature):
        """Test review of clean code with no issues"""
        files = [
            {
                'path': 'utils.py',
                'content': '''
def calculate_total(items: list) -> float:
    """Calculate the total price of items.
    
    Args:
        items: List of items with price attribute
        
    Returns:
        float: Total price of all items
    """
    return sum(item.price for item in items)
''',
                'language': 'python'
            }
        ]
        
        # Run code linter
        linter = CodeLinter()
        code_analysis = await linter.analyze(files)
        
        # Run AI review engine
        engine = AIReviewEngine()
        await engine.initialize()
        
        review = await engine.generate_review(
            'Add utility function',
            'Adding helper function',
            files,
            code_analysis
        )
        
        # Verify positive feedback for clean code
        assert len(review['positive_feedback']) > 0
    
    @pytest.mark.asyncio
    async def test_review_with_tests(self, feature):
        """Test review recognizes test files"""
        files = [
            {
                'path': 'src/calculator.py',
                'content': '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
''',
                'language': 'python'
            },
            {
                'path': 'tests/test_calculator.py',
                'content': '''
def test_add():
    """Test addition function."""
    assert add(2, 3) == 5
''',
                'language': 'python'
            }
        ]
        
        # Run code linter
        linter = CodeLinter()
        code_analysis = await linter.analyze(files)
        
        # Run AI review engine
        engine = AIReviewEngine()
        await engine.initialize()
        
        review = await engine.generate_review(
            'Add calculator with tests',
            'Implementing calculator functionality',
            files,
            code_analysis
        )
        
        # Verify positive feedback for including tests
        test_feedback = [f for f in review['positive_feedback'] if 'Test' in f.get('title', '')]
        assert len(test_feedback) > 0
    
    @pytest.mark.asyncio
    async def test_pattern_detection(self, feature):
        """Test pattern detection in complex code"""
        files = [
            {
                'path': 'complex.py',
                'content': '''
def process_data(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] > 0:
                for k in range(10):
                    x = data[i][j] * k
                    print(x)
''',
                'language': 'python'
            }
        ]
        
        # Run code linter
        linter = CodeLinter()
        code_analysis = await linter.analyze(files)
        
        # Run AI review engine
        engine = AIReviewEngine()
        await engine.initialize()
        
        review = await engine.generate_review(
            'Add data processing',
            'Processing nested data structures',
            files,
            code_analysis
        )
        
        # Verify complexity patterns were detected
        complexity_patterns = [p for p in review['patterns_detected'] if p['type'] == 'complexity']
        assert len(complexity_patterns) > 0
        
        # Verify suggestions for complexity
        complexity_suggestions = [s for s in review['suggestions'] if 'Complexity' in s.get('title', '')]
        assert len(complexity_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_severity_levels(self, feature):
        """Test handling of multiple severity levels"""
        files = [
            {
                'path': 'mixed.py',
                'content': '''
def process(user_input):
    # Critical: SQL injection
    cursor.execute("SELECT * FROM users WHERE id = " + user_input)
    
    # Medium: Bare except
    try:
        risky_operation()
    except:
        pass
    
    # Low: Single letter variable
    x = 10
''',
                'language': 'python'
            }
        ]
        
        # Run code linter
        linter = CodeLinter()
        code_analysis = await linter.analyze(files)
        
        # Run AI review engine
        engine = AIReviewEngine()
        await engine.initialize()
        
        review = await engine.generate_review(
            'Mixed severity issues',
            'Code with various issue severities',
            files,
            code_analysis
        )
        
        # Verify multiple severity levels
        severities = set(i['severity'] for i in review['issues'])
        assert 'critical' in severities
        assert len(severities) > 1
        
        # Verify suggestions prioritize critical issues
        critical_suggestions = [s for s in review['suggestions'] if s.get('severity') == 'critical']
        assert len(critical_suggestions) > 0
