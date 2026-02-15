"""
Integration test for Bitbucket PR Analyzer
"""

import pytest
from xencode.features.code_review import CodeReviewFeature, CodeReviewConfig
from xencode.features.base import FeatureConfig


class TestBitbucketIntegration:
    """Integration tests for Bitbucket PR analyzer"""

    @pytest.mark.asyncio
    async def test_bitbucket_analyzer_registered(self):
        """Test that Bitbucket analyzer is registered in CodeReviewFeature"""
        config = FeatureConfig(
            name="code_review",
            enabled=True,
            config=CodeReviewConfig()
        )
        
        feature = CodeReviewFeature(config)
        await feature._initialize()
        
        # Verify Bitbucket is in the analyzers
        assert 'bitbucket' in feature._pr_analyzers
        assert feature._pr_analyzers['bitbucket'] is not None
        
        await feature._shutdown()

    @pytest.mark.asyncio
    async def test_bitbucket_in_default_integrations(self):
        """Test that Bitbucket is in default integrations"""
        config = CodeReviewConfig()
        
        assert 'bitbucket' in config.integrations
        assert 'github' in config.integrations
        assert 'gitlab' in config.integrations

    @pytest.mark.asyncio
    async def test_analyze_pr_with_bitbucket_platform(self):
        """Test analyze_pr method accepts bitbucket platform"""
        config = FeatureConfig(
            name="code_review",
            enabled=True,
            config=CodeReviewConfig()
        )
        
        feature = CodeReviewFeature(config)
        await feature._initialize()
        
        # Mock the analyzer's fetch_pr method
        async def mock_fetch_pr(pr_url):
            return {
                'url': pr_url,
                'title': 'Test PR',
                'description': 'Test description',
                'platform': 'bitbucket',
                'timestamp': '2024-01-01T00:00:00Z',
                'state': 'open',
                'author': 'test-author',
                'base_branch': 'main',
                'head_branch': 'feature',
                'files': [],
                'commits': [],
                'comments': [],
                'review_count': 0,
                'additions': 0,
                'deletions': 0,
                'changed_files': 0
            }
        
        feature._pr_analyzers['bitbucket'].fetch_pr = mock_fetch_pr
        
        # This should not raise an error
        result = await feature.analyze_pr(
            'https://bitbucket.org/workspace/repo/pull-requests/123',
            platform='bitbucket'
        )
        
        assert result is not None
        assert 'summary' in result
        
        await feature._shutdown()
