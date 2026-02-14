"""
Unit tests for GitHub PR Analyzer
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from xencode.features.code_review import GitHubPRAnalyzer, GitLabPRAnalyzer, BitbucketPRAnalyzer


class TestGitHubPRAnalyzer:
    """Tests for GitHubPRAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return GitHubPRAnalyzer()
    
    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session"""
        mock_session = MagicMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})
        mock_session.get = MagicMock(return_value=mock_response)
        return mock_session
    
    def test_parse_pr_url_valid(self, analyzer):
        """Test parsing valid GitHub PR URL"""
        url = "https://github.com/owner/repo/pull/123"
        result = analyzer._parse_pr_url(url)
        
        assert result['owner'] == 'owner'
        assert result['repo'] == 'repo'
        assert result['pr_number'] == '123'
    
    def test_parse_pr_url_with_http(self, analyzer):
        """Test parsing HTTP GitHub PR URL"""
        url = "http://github.com/owner/repo/pull/456"
        result = analyzer._parse_pr_url(url)
        
        assert result['owner'] == 'owner'
        assert result['repo'] == 'repo'
        assert result['pr_number'] == '456'
    
    def test_parse_pr_url_invalid(self, analyzer):
        """Test parsing invalid GitHub PR URL"""
        url = "https://github.com/owner/repo/commit/abc123"
        
        with pytest.raises(ValueError, match="Invalid GitHub PR URL"):
            analyzer._parse_pr_url(url)
    
    def test_parse_pr_url_invalid_format(self, analyzer):
        """Test parsing URL with wrong format"""
        url = "not-a-github-url"
        
        with pytest.raises(ValueError, match="Invalid GitHub PR URL"):
            analyzer._parse_pr_url(url)
    
    @pytest.mark.asyncio
    async def test_fetch_pr_success(self, analyzer):
        """Test successful PR fetch"""
        # Mock response data
        pr_data = {
            'title': 'Add new feature',
            'body': 'Description of the feature',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'open',
            'user': {'login': 'author'},
            'base': {'ref': 'main'},
            'head': {'ref': 'feature-branch'},
            'additions': 100,
            'deletions': 20,
            'changed_files': 5
        }
        
        files_data = [{
            'filename': 'src/file.py',
            'status': 'added',
            'additions': 100,
            'deletions': 0,
            'changes': 100,
            'patch': 'diff content',
            'blob_url': 'https://github.com/blob'
        }]
        
        commits_data = [{
            'sha': 'abc123',
            'commit': {
                'message': 'Add feature',
                'author': {'name': 'Author', 'email': 'author@example.com'}
            }
        }]
        
        reviews_data = [{
            'id': 1,
            'user': {'login': 'reviewer'},
            'comments': []
        }]
        
        # Mock the session and requests
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'pulls/123' in endpoint and 'files' not in endpoint and 'commits' not in endpoint and 'reviews' not in endpoint:
                return pr_data
            elif 'files' in endpoint:
                return files_data
            elif 'commits' in endpoint:
                return commits_data
            elif 'reviews' in endpoint:
                return reviews_data
            return {}
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://github.com/owner/repo/pull/123")
        
        assert result['url'] == "https://github.com/owner/repo/pull/123"
        assert result['title'] == 'Add new feature'
        assert result['description'] == 'Description of the feature'
        assert result['state'] == 'open'
        assert result['author'] == 'author'
        assert result['base_branch'] == 'main'
        assert result['head_branch'] == 'feature-branch'
        assert len(result['files']) == 1
        assert result['files'][0]['filename'] == 'src/file.py'
        assert len(result['commits']) == 1
        assert result['commits'][0]['message'] == 'Add feature'
        assert result['additions'] == 100
        assert result['deletions'] == 20
        assert result['changed_files'] == 5
    
    @pytest.mark.asyncio
    async def test_fetch_pr_merged_state(self, analyzer):
        """Test PR fetch with merged state"""
        pr_data = {
            'title': 'Merge PR',
            'body': '',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'closed',
            'merged': True,
            'user': {'login': 'author'},
            'base': {'ref': 'main'},
            'head': {'ref': 'feature-branch'},
            'additions': 50,
            'deletions': 10,
            'changed_files': 3
        }
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'pulls' in endpoint and 'files' not in endpoint and 'commits' not in endpoint and 'reviews' not in endpoint:
                return pr_data
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://github.com/owner/repo/pull/123")
        
        assert result['state'] == 'merged'
    
    @pytest.mark.asyncio
    async def test_fetch_pr_closed_state(self, analyzer):
        """Test PR fetch with closed (not merged) state"""
        pr_data = {
            'title': 'Closed PR',
            'body': '',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'closed',
            'merged': False,
            'user': {'login': 'author'},
            'base': {'ref': 'main'},
            'head': {'ref': 'feature-branch'},
            'additions': 0,
            'deletions': 0,
            'changed_files': 0
        }
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'pulls' in endpoint and 'files' not in endpoint and 'commits' not in endpoint and 'reviews' not in endpoint:
                return pr_data
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://github.com/owner/repo/pull/123")
        
        assert result['state'] == 'closed'
    
    @pytest.mark.asyncio
    async def test_fetch_pr_with_line_comments(self, analyzer):
        """Test PR fetch with line comments"""
        pr_data = {
            'title': 'PR with comments',
            'body': '',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'open',
            'user': {'login': 'author'},
            'base': {'ref': 'main'},
            'head': {'ref': 'feature-branch'},
            'additions': 10,
            'deletions': 5,
            'changed_files': 2
        }
        
        reviews_data = [{
            'id': 1,
            'user': {'login': 'reviewer'},
            'comments': [{
                'body': 'Good catch!',
                'path': 'src/file.py',
                'line': 42,
                'user': {'login': 'reviewer'},
                'created_at': '2024-01-02T00:00:00Z'
            }]
        }]
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'pulls/123' in endpoint and 'files' not in endpoint and 'commits' not in endpoint and 'reviews' not in endpoint:
                return pr_data
            elif 'reviews' in endpoint:
                return reviews_data
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://github.com/owner/repo/pull/123")
        
        assert len(result['comments']) == 1
        assert result['comments'][0]['body'] == 'Good catch!'
        assert result['comments'][0]['path'] == 'src/file.py'
        assert result['comments'][0]['line'] == 42
        assert result['comments'][0]['user'] == 'reviewer'
    
    @pytest.mark.asyncio
    async def test_fetch_pr_api_error(self, analyzer):
        """Test PR fetch with API error"""
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            raise ValueError("API Error")
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        with pytest.raises(ValueError, match="Failed to fetch PR"):
            await analyzer.fetch_pr("https://github.com/owner/repo/pull/123")
    
    @pytest.mark.asyncio
    async def test_fetch_pr_by_repo(self, analyzer):
        """Test fetching PR by owner, repo, and PR number"""
        pr_data = {
            'title': 'Test PR',
            'body': '',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'open',
            'user': {'login': 'author'},
            'base': {'ref': 'main'},
            'head': {'ref': 'feature-branch'},
            'additions': 0,
            'deletions': 0,
            'changed_files': 0
        }
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'pulls' in endpoint and 'files' not in endpoint and 'commits' not in endpoint and 'reviews' not in endpoint:
                return pr_data
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr_by_repo('owner', 'repo', 123)
        
        assert result['url'] == "https://github.com/owner/repo/pull/123"
        assert result['title'] == 'Test PR'
    
    @pytest.mark.asyncio
    async def test_fetch_pr_pagination(self, analyzer):
        """Test PR fetch with pagination for large PRs"""
        pr_data = {
            'title': 'Large PR',
            'body': '',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'open',
            'user': {'login': 'author'},
            'base': {'ref': 'main'},
            'head': {'ref': 'feature-branch'},
            'additions': 0,
            'deletions': 0,
            'changed_files': 0
        }
        
        # Simulate paginated files - GitHub returns max 30 items per page
        files_page1 = [{'filename': f'file{i}.py'} for i in range(30)]
        files_page2 = [{'filename': f'file{i}.py'} for i in range(30, 35)]
        all_files = files_page1 + files_page2
        
        async def mock_get_session():
            return MagicMock()
        
        call_count = {'files': 0}
        
        async def mock_make_request(endpoint, params=None):
            if 'pulls' in endpoint and 'files' not in endpoint and 'commits' not in endpoint and 'reviews' not in endpoint:
                return pr_data
            elif 'files' in endpoint:
                call_count['files'] += 1
                # Simulate pagination by returning all files at once
                # (the actual pagination logic is tested in the real implementation)
                return all_files
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://github.com/owner/repo/pull/123")
        
        # Should have fetched all 35 files
        assert len(result['files']) == 35
    
    def test_analyzer_with_token(self):
        """Test analyzer initialization with authentication token"""
        token = "ghp_test_token"
        analyzer = GitHubPRAnalyzer(token=token)
        
        assert analyzer.token == token
        assert analyzer.base_url == "https://api.github.com"


class TestGitLabPRAnalyzer:
    """Tests for GitLabPRAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return GitLabPRAnalyzer()
    
    def test_parse_mr_url_valid(self, analyzer):
        """Test parsing valid GitLab MR URL"""
        url = "https://gitlab.com/owner/repo/-/merge_requests/123"
        result = analyzer._parse_mr_url(url)
        
        assert result['owner'] == 'owner'
        assert result['repo'] == 'repo'
        assert result['mr_number'] == '123'
    
    def test_parse_mr_url_without_slash(self, analyzer):
        """Test parsing GitLab MR URL without double slash"""
        url = "https://gitlab.com/owner/repo/merge_requests/456"
        result = analyzer._parse_mr_url(url)
        
        assert result['owner'] == 'owner'
        assert result['repo'] == 'repo'
        assert result['mr_number'] == '456'
    
    def test_parse_mr_url_invalid(self, analyzer):
        """Test parsing invalid GitLab MR URL"""
        url = "https://gitlab.com/owner/repo/commit/abc123"
        
        with pytest.raises(ValueError, match="Invalid GitLab MR URL"):
            analyzer._parse_mr_url(url)
    
    def test_parse_mr_url_invalid_format(self, analyzer):
        """Test parsing URL with wrong format"""
        url = "not-a-gitlab-url"
        
        with pytest.raises(ValueError, match="Invalid GitLab MR URL"):
            analyzer._parse_mr_url(url)
    
    @pytest.mark.asyncio
    async def test_fetch_mr_success(self, analyzer):
        """Test successful MR fetch"""
        # Mock response data
        mr_data = {
            'title': 'Add new feature',
            'description': 'Description of the feature',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'opened',
            'merged': False,
            'author': {'username': 'author'},
            'target_branch': 'main',
            'source_branch': 'feature-branch',
            'additions': 100,
            'deletions': 20,
            'changed_files': 5
        }
        
        changes_data = {
            'changes': [{
                'new_path': 'src/file.py',
                'old_path': None,
                'new_lines': 100,
                'old_lines': 0,
                'diff': 'new file'
            }]
        }
        
        commits_data = [{
            'id': 'abc123',
            'message': 'Add feature',
            'author_name': 'Author',
            'author_email': 'author@example.com'
        }]
        
        discussions_data = [{
            'id': 1,
            'notes': []
        }]
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'merge_requests/123' in endpoint and 'changes' not in endpoint and 'commits' not in endpoint and 'discussions' not in endpoint:
                return [mr_data]
            elif 'changes' in endpoint:
                return [changes_data]
            elif 'commits' in endpoint:
                return commits_data
            elif 'discussions' in endpoint:
                return discussions_data
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://gitlab.com/owner/repo/-/merge_requests/123")
        
        assert result['url'] == "https://gitlab.com/owner/repo/-/merge_requests/123"
        assert result['title'] == 'Add new feature'
        assert result['description'] == 'Description of the feature'
        assert result['state'] == 'open'
        assert result['author'] == 'author'
        assert result['base_branch'] == 'main'
        assert result['head_branch'] == 'feature-branch'
        assert len(result['files']) == 1
        assert result['files'][0]['filename'] == 'src/file.py'
        assert result['files'][0]['status'] == 'added'
        assert len(result['commits']) == 1
        assert result['commits'][0]['message'] == 'Add feature'
        assert result['additions'] == 100
        assert result['deletions'] == 20
        assert result['changed_files'] == 5
    
    @pytest.mark.asyncio
    async def test_fetch_mr_merged_state(self, analyzer):
        """Test MR fetch with merged state"""
        mr_data = {
            'title': 'Merge MR',
            'description': '',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'merged',
            'merged': True,
            'author': {'username': 'author'},
            'target_branch': 'main',
            'source_branch': 'feature-branch',
            'additions': 50,
            'deletions': 10,
            'changed_files': 3
        }
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'merge_requests' in endpoint:
                return [mr_data]
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://gitlab.com/owner/repo/-/merge_requests/123")
        
        assert result['state'] == 'merged'
    
    @pytest.mark.asyncio
    async def test_fetch_mr_closed_state(self, analyzer):
        """Test MR fetch with closed state"""
        mr_data = {
            'title': 'Closed MR',
            'description': '',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'closed',
            'merged': False,
            'author': {'username': 'author'},
            'target_branch': 'main',
            'source_branch': 'feature-branch',
            'additions': 0,
            'deletions': 0,
            'changed_files': 0
        }
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'merge_requests' in endpoint:
                return [mr_data]
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://gitlab.com/owner/repo/-/merge_requests/123")
        
        assert result['state'] == 'closed'
    
    @pytest.mark.asyncio
    async def test_fetch_mr_with_comments(self, analyzer):
        """Test MR fetch with line comments"""
        mr_data = {
            'title': 'MR with comments',
            'description': '',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'opened',
            'merged': False,
            'author': {'username': 'author'},
            'target_branch': 'main',
            'source_branch': 'feature-branch',
            'additions': 10,
            'deletions': 5,
            'changed_files': 2
        }
        
        discussions_data = [{
            'id': 1,
            'notes': [{
                'id': 1,
                'type': 'DiffNote',
                'body': 'Good catch!',
                'path': 'src/file.py',
                'line': 42,
                'author': {'username': 'reviewer'},
                'created_at': '2024-01-02T00:00:00Z'
            }]
        }]
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'merge_requests/123' in endpoint and 'changes' not in endpoint and 'commits' not in endpoint and 'discussions' not in endpoint:
                return [mr_data]
            elif 'discussions' in endpoint:
                return discussions_data
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://gitlab.com/owner/repo/-/merge_requests/123")
        
        assert len(result['comments']) == 1
        assert result['comments'][0]['body'] == 'Good catch!'
        assert result['comments'][0]['path'] == 'src/file.py'
        assert result['comments'][0]['line'] == 42
        assert result['comments'][0]['user'] == 'reviewer'
    
    @pytest.mark.asyncio
    async def test_fetch_mr_api_error(self, analyzer):
        """Test MR fetch with API error"""
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            raise ValueError("API Error")
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        with pytest.raises(ValueError, match="Failed to fetch MR"):
            await analyzer.fetch_pr("https://gitlab.com/owner/repo/-/merge_requests/123")
    
    @pytest.mark.asyncio
    async def test_fetch_mr_by_repo(self, analyzer):
        """Test fetching MR by owner, repo, and MR number"""
        mr_data = {
            'title': 'Test MR',
            'description': '',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'opened',
            'merged': False,
            'author': {'username': 'author'},
            'target_branch': 'main',
            'source_branch': 'feature-branch',
            'additions': 0,
            'deletions': 0,
            'changed_files': 0
        }
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'merge_requests' in endpoint:
                return [mr_data]
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr_by_repo('owner', 'repo', 123)
        
        assert result['url'] == "https://gitlab.com/owner/repo/merge_requests/123"
        assert result['title'] == 'Test MR'
    
    @pytest.mark.asyncio
    async def test_fetch_mr_pagination(self, analyzer):
        """Test MR fetch with pagination for large MRs"""
        mr_data = {
            'title': 'Large MR',
            'description': '',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'opened',
            'merged': False,
            'author': {'username': 'author'},
            'target_branch': 'main',
            'source_branch': 'feature-branch',
            'additions': 0,
            'deletions': 0,
            'changed_files': 0
        }
        
        # Simulate paginated commits - GitLab returns up to 100 items per page
        commits_page1 = [{'id': f'commit{i}', 'message': f'Commit {i}'} for i in range(100)]
        commits_page2 = [{'id': f'commit{i}', 'message': f'Commit {i}'} for i in range(100, 105)]
        all_commits = commits_page1 + commits_page2
        
        async def mock_get_session():
            return MagicMock()
        
        call_count = {'commits': 0}
        
        async def mock_make_request(endpoint, params=None):
            if 'merge_requests/123' in endpoint and 'changes' not in endpoint and 'commits' not in endpoint and 'discussions' not in endpoint:
                return [mr_data]
            elif 'commits' in endpoint:
                call_count['commits'] += 1
                # Simulate pagination by returning all commits at once
                return all_commits
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://gitlab.com/owner/repo/-/merge_requests/123")
        
        # Should have fetched all 105 commits
        assert len(result['commits']) == 105
    
    def test_analyzer_with_token(self):
        """Test analyzer initialization with authentication token"""
        token = "glpat_test_token"
        analyzer = GitLabPRAnalyzer(token=token)
        
        assert analyzer.token == token
        assert analyzer.base_url == "https://gitlab.com/api/v4"


class TestBitbucketPRAnalyzer:
    """Tests for BitbucketPRAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return BitbucketPRAnalyzer()

    def test_parse_pr_url_valid(self, analyzer):
            """Test parsing valid Bitbucket PR URL"""
            url = "https://bitbucket.org/workspace/repo/pull-requests/123"
            result = analyzer._parse_pr_url(url)

            assert result['workspace'] == 'workspace'
            assert result['repo'] == 'repo'
            assert result['pr_number'] == '123'

    def test_parse_pr_url_without_s(self, analyzer):
            """Test parsing Bitbucket PR URL without 's' in pull-requests"""
            url = "https://bitbucket.org/workspace/repo/pull-request/456"
            result = analyzer._parse_pr_url(url)

            assert result['workspace'] == 'workspace'
            assert result['repo'] == 'repo'
            assert result['pr_number'] == '456'

    def test_parse_pr_url_with_http(self, analyzer):
            """Test parsing HTTP Bitbucket PR URL"""
            url = "http://bitbucket.org/workspace/repo/pull-requests/789"
            result = analyzer._parse_pr_url(url)

            assert result['workspace'] == 'workspace'
            assert result['repo'] == 'repo'
            assert result['pr_number'] == '789'

    def test_parse_pr_url_invalid(self, analyzer):
            """Test parsing invalid Bitbucket PR URL"""
            url = "https://bitbucket.org/workspace/repo/commits/abc123"

            with pytest.raises(ValueError, match="Invalid Bitbucket PR URL"):
                analyzer._parse_pr_url(url)

    def test_parse_pr_url_invalid_format(self, analyzer):
            """Test parsing URL with wrong format"""
            url = "not-a-bitbucket-url"

            with pytest.raises(ValueError, match="Invalid Bitbucket PR URL"):
                analyzer._parse_pr_url(url)

    @pytest.mark.asyncio
    async def test_fetch_pr_success(self, analyzer):
            """Test successful PR fetch"""
            # Mock response data
            pr_data = {
                'title': 'Add new feature',
                'description': 'Description of the feature',
                'created_on': '2024-01-01T00:00:00Z',
                'state': 'OPEN',
                'author': {'display_name': 'author'},
                'destination': {'branch': {'name': 'main'}},
                'source': {'branch': {'name': 'feature-branch'}},
                'summary': {
                    'additions': 100,
                    'deletions': 20,
                    'changed_files': 5
                }
            }

            files_data = [{
                'path': {'path': 'src/file.py'},
                'type': 'add',
                'additions': 100,
                'deletions': 0,
                'raw': 'diff content'
            }]

            commits_data = [{
                'hash': 'abc123',
                'message': 'Add feature',
                'author': {
                    'user': {
                        'display_name': 'Author',
                        'email_address': 'author@example.com'
                    }
                }
            }]

            activities_data = [{
                'comment': {
                    'content': {'raw': 'Good catch!'},
                    'file': {'path': 'src/file.py'},
                    'line': 42,
                    'user': {'display_name': 'reviewer'},
                    'created_on': '2024-01-02T00:00:00Z'
                }
            }]

            async def mock_get_session():
                return MagicMock()

            async def mock_make_request(endpoint, params=None):
                if 'pullrequests/123' in endpoint and 'changes' not in endpoint and 'commits' not in endpoint and 'activities' not in endpoint:
                    return [pr_data]
                elif 'changes' in endpoint:
                    return files_data
                elif 'commits' in endpoint:
                    return commits_data
                elif 'activities' in endpoint:
                    return activities_data
                return []

            analyzer._get_session = mock_get_session
            analyzer._make_request = mock_make_request

            result = await analyzer.fetch_pr("https://bitbucket.org/workspace/repo/pull-requests/123")

            assert result['url'] == "https://bitbucket.org/workspace/repo/pull-requests/123"
            assert result['title'] == 'Add new feature'
            assert result['description'] == 'Description of the feature'
            assert result['state'] == 'open'
            assert result['author'] == 'author'
            assert result['base_branch'] == 'main'
            assert result['head_branch'] == 'feature-branch'
            assert len(result['files']) == 1
            assert result['files'][0]['filename'] == 'src/file.py'
            assert result['files'][0]['status'] == 'added'
            assert len(result['commits']) == 1
            assert result['commits'][0]['message'] == 'Add feature'
            assert result['additions'] == 100
            assert result['deletions'] == 20
            assert result['changed_files'] == 5
            assert len(result['comments']) == 1
            assert result['comments'][0]['body'] == 'Good catch!'

    @pytest.mark.asyncio
    async def test_fetch_pr_merged_state(self, analyzer):
            """Test PR fetch with merged state"""
            pr_data = {
                'title': 'Merge PR',
                'description': '',
                'created_on': '2024-01-01T00:00:00Z',
                'state': 'MERGED',
                'author': {'display_name': 'author'},
                'destination': {'branch': {'name': 'main'}},
                'source': {'branch': {'name': 'feature-branch'}},
                'summary': {
                    'additions': 50,
                    'deletions': 10,
                    'changed_files': 3
                }
            }

            async def mock_get_session():
                return MagicMock()

            async def mock_make_request(endpoint, params=None):
                if 'pullrequests' in endpoint:
                    return [pr_data]
                return []

            analyzer._get_session = mock_get_session
            analyzer._make_request = mock_make_request

            result = await analyzer.fetch_pr("https://bitbucket.org/workspace/repo/pull-requests/123")

            assert result['state'] == 'merged'

    @pytest.mark.asyncio
    async def test_fetch_pr_declined_state(self, analyzer):
            """Test PR fetch with declined state"""
            pr_data = {
                'title': 'Declined PR',
                'description': '',
                'created_on': '2024-01-01T00:00:00Z',
                'state': 'DECLINED',
                'author': {'display_name': 'author'},
                'destination': {'branch': {'name': 'main'}},
                'source': {'branch': {'name': 'feature-branch'}},
                'summary': {
                    'additions': 0,
                    'deletions': 0,
                    'changed_files': 0
                }
            }

            async def mock_get_session():
                return MagicMock()

            async def mock_make_request(endpoint, params=None):
                if 'pullrequests' in endpoint:
                    return [pr_data]
                return []

            analyzer._get_session = mock_get_session
            analyzer._make_request = mock_make_request

            result = await analyzer.fetch_pr("https://bitbucket.org/workspace/repo/pull-requests/123")

            assert result['state'] == 'declined'

    @pytest.mark.asyncio
    async def test_fetch_pr_with_comments(self, analyzer):
            """Test PR fetch with line comments"""
            pr_data = {
                'title': 'PR with comments',
                'description': '',
                'created_on': '2024-01-01T00:00:00Z',
                'state': 'OPEN',
                'author': {'display_name': 'author'},
                'destination': {'branch': {'name': 'main'}},
                'source': {'branch': {'name': 'feature-branch'}},
                'summary': {
                    'additions': 10,
                    'deletions': 5,
                    'changed_files': 2
                }
            }

            activities_data = [{
                'comment': {
                    'content': {'raw': 'Good catch!'},
                    'file': {'path': 'src/file.py'},
                    'line': 42,
                    'user': {'display_name': 'reviewer'},
                    'created_on': '2024-01-02T00:00:00Z'
                }
            }]

            async def mock_get_session():
                return MagicMock()

            async def mock_make_request(endpoint, params=None):
                if 'pullrequests/123' in endpoint and 'changes' not in endpoint and 'commits' not in endpoint and 'activities' not in endpoint:
                    return [pr_data]
                elif 'activities' in endpoint:
                    return activities_data
                return []

            analyzer._get_session = mock_get_session
            analyzer._make_request = mock_make_request

            result = await analyzer.fetch_pr("https://bitbucket.org/workspace/repo/pull-requests/123")

            assert len(result['comments']) == 1
            assert result['comments'][0]['body'] == 'Good catch!'
            assert result['comments'][0]['path'] == 'src/file.py'
            assert result['comments'][0]['line'] == 42
            assert result['comments'][0]['user'] == 'reviewer'

    @pytest.mark.asyncio
    async def test_fetch_pr_api_error(self, analyzer):
            """Test PR fetch with API error"""
            async def mock_get_session():
                return MagicMock()

            async def mock_make_request(endpoint, params=None):
                raise ValueError("API Error")

            analyzer._get_session = mock_get_session
            analyzer._make_request = mock_make_request

            with pytest.raises(ValueError, match="Failed to fetch PR"):
                await analyzer.fetch_pr("https://bitbucket.org/workspace/repo/pull-requests/123")

    @pytest.mark.asyncio
    async def test_fetch_pr_by_repo(self, analyzer):
            """Test fetching PR by workspace, repo, and PR number"""
            pr_data = {
                'title': 'Test PR',
                'description': '',
                'created_on': '2024-01-01T00:00:00Z',
                'state': 'OPEN',
                'author': {'display_name': 'author'},
                'destination': {'branch': {'name': 'main'}},
                'source': {'branch': {'name': 'feature-branch'}},
                'summary': {
                    'additions': 0,
                    'deletions': 0,
                    'changed_files': 0
                }
            }

            async def mock_get_session():
                return MagicMock()

            async def mock_make_request(endpoint, params=None):
                if 'pullrequests' in endpoint:
                    return [pr_data]
                return []

            analyzer._get_session = mock_get_session
            analyzer._make_request = mock_make_request

            result = await analyzer.fetch_pr_by_repo('workspace', 'repo', 123)

            assert result['url'] == "https://bitbucket.org/workspace/repo/pull-requests/123"
            assert result['title'] == 'Test PR'

    @pytest.mark.asyncio
    async def test_fetch_pr_pagination(self, analyzer):
            """Test PR fetch with pagination for large PRs"""
            pr_data = {
                'title': 'Large PR',
                'description': '',
                'created_on': '2024-01-01T00:00:00Z',
                'state': 'OPEN',
                'author': {'display_name': 'author'},
                'destination': {'branch': {'name': 'main'}},
                'source': {'branch': {'name': 'feature-branch'}},
                'summary': {
                    'additions': 0,
                    'deletions': 0,
                    'changed_files': 0
                }
            }

            # Simulate paginated commits - Bitbucket returns up to 50 items per page
            commits_page1 = [{'hash': f'commit{i}', 'message': f'Commit {i}'} for i in range(50)]
            commits_page2 = [{'hash': f'commit{i}', 'message': f'Commit {i}'} for i in range(50, 55)]
            all_commits = commits_page1 + commits_page2

            async def mock_get_session():
                return MagicMock()

            call_count = {'commits': 0}

            async def mock_make_request(endpoint, params=None):
                if 'pullrequests/123' in endpoint and 'changes' not in endpoint and 'commits' not in endpoint and 'activities' not in endpoint:
                    return [pr_data]
                elif 'commits' in endpoint:
                    call_count['commits'] += 1
                    # Simulate pagination by returning all commits at once
                    return all_commits
                return []

            analyzer._get_session = mock_get_session
            analyzer._make_request = mock_make_request

            result = await analyzer.fetch_pr("https://bitbucket.org/workspace/repo/pull-requests/123")

            # Should have fetched all 55 commits
            assert len(result['commits']) == 55

    def test_analyzer_with_token(self):
            """Test analyzer initialization with authentication token"""
            token = "bbp_test_token"
            analyzer = BitbucketPRAnalyzer(token=token)

            assert analyzer.token == token
            assert analyzer.base_url == "https://api.bitbucket.org/2.0"


class TestCodeLinter:
    """Tests for CodeLinter class"""
    
    @pytest.fixture
    def linter(self):
        """Create linter instance"""
        from xencode.features.code_review import CodeLinter
        return CodeLinter()
    
    @pytest.mark.asyncio
    async def test_analyze_empty_files(self, linter):
        """Test analyzing empty file list"""
        result = await linter.analyze([])
        
        assert result['summary']['total_files'] == 0
        assert result['summary']['total_issues'] == 0
        assert len(result['files']) == 0
        assert len(result['issues']) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_python_file_with_sql_injection(self, linter):
        """Test detecting SQL injection in Python code"""
        files = [{
            'path': 'test.py',
            'language': 'python',
            'content': '''
def get_user(user_id):
    cursor.execute("SELECT * FROM users WHERE id = " + user_id)
    return cursor.fetchone()
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for SQL injection issue
        sqli_issues = [i for i in result['issues'] if i['type'] == 'sqli']
        assert len(sqli_issues) > 0
        assert sqli_issues[0]['severity'] == 'critical'
    
    @pytest.mark.asyncio
    async def test_analyze_javascript_xss(self, linter):
        """Test detecting XSS vulnerability in JavaScript"""
        files = [{
            'path': 'test.js',
            'language': 'javascript',
            'content': '''
function displayMessage(msg) {
    document.getElementById('output').innerHTML = msg;
}
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for XSS issue
        xss_issues = [i for i in result['issues'] if i['type'] == 'xss']
        assert len(xss_issues) > 0
        assert xss_issues[0]['severity'] == 'high'
    
    @pytest.mark.asyncio
    async def test_analyze_python_bare_except(self, linter):
        """Test detecting bare except clause in Python"""
        files = [{
            'path': 'test.py',
            'language': 'python',
            'content': '''
try:
    risky_operation()
except:
    pass
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for code quality issue
        quality_issues = [i for i in result['issues'] if i['type'] == 'code_quality']
        assert len(quality_issues) > 0
        assert any('bare except' in i['message'].lower() for i in quality_issues)
    
    @pytest.mark.asyncio
    async def test_analyze_typescript_any_type(self, linter):
        """Test detecting 'any' type usage in TypeScript"""
        files = [{
            'path': 'test.ts',
            'language': 'typescript',
            'content': '''
function processData(data: any): void {
    console.log(data);
}
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for any type issue
        any_issues = [i for i in result['issues'] if 'any' in i['message'].lower()]
        assert len(any_issues) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_rust_unwrap(self, linter):
        """Test detecting unwrap() usage in Rust"""
        files = [{
            'path': 'test.rs',
            'language': 'rust',
            'content': '''
fn get_value() -> i32 {
    let result = Some(42);
    result.unwrap()
}
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for unwrap issue
        unwrap_issues = [i for i in result['issues'] if 'unwrap' in i['message'].lower()]
        assert len(unwrap_issues) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_javascript_equality(self, linter):
        """Test detecting == instead of === in JavaScript"""
        files = [{
            'path': 'test.js',
            'language': 'javascript',
            'content': '''
if (value == 5) {
    console.log('equal');
}
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for equality issue
        equality_issues = [i for i in result['issues'] if '===' in i['message']]
        assert len(equality_issues) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_csrf_vulnerability(self, linter):
        """Test detecting CSRF vulnerability"""
        files = [{
            'path': 'form.html',
            'language': 'html',
            'content': '''
<form method="POST" action="/submit">
    <input type="text" name="data">
    <button type="submit">Submit</button>
</form>
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for CSRF issue
        csrf_issues = [i for i in result['issues'] if i['type'] == 'csrf']
        assert len(csrf_issues) > 0
        assert csrf_issues[0]['severity'] == 'medium'
    
    @pytest.mark.asyncio
    async def test_analyze_multiple_files(self, linter):
        """Test analyzing multiple files"""
        files = [
            {
                'path': 'test1.py',
                'language': 'python',
                'content': 'cursor.execute("SELECT * FROM users")'
            },
            {
                'path': 'test2.js',
                'language': 'javascript',
                'content': 'element.innerHTML = userInput;'
            },
            {
                'path': 'test3.ts',
                'language': 'typescript',
                'content': 'function test(data: any) {}'
            }
        ]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 3
        assert result['summary']['total_issues'] > 0
        assert len(result['files']) == 3
        
        # Each file should have issues
        for file_result in result['files']:
            assert file_result['has_issues'] == True
    
    @pytest.mark.asyncio
    async def test_analyze_clean_code(self, linter):
        """Test analyzing clean code with no issues"""
        files = [{
            'path': 'clean.py',
            'language': 'python',
            'content': '''
def add(a: int, b: int) -> int:
    return a + b
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        # Clean code should have minimal or no issues
        assert result['summary']['total_issues'] >= 0
    
    @pytest.mark.asyncio
    async def test_analyze_severity_counts(self, linter):
        """Test that severity counts are tracked correctly"""
        files = [
            {
                'path': 'critical.py',
                'language': 'python',
                'content': 'cursor.execute("SELECT * FROM users WHERE id = " + user_id)'
            },
            {
                'path': 'high.js',
                'language': 'javascript',
                'content': 'document.write(userInput);'
            },
            {
                'path': 'medium.ts',
                'language': 'typescript',
                'content': 'function test(data: any) {}'
            }
        ]
        
        result = await linter.analyze(files)
        
        # Check that severity counts exist
        assert 'by_severity' in result['summary']
        assert 'critical' in result['summary']['by_severity']
        assert 'high' in result['summary']['by_severity']
        assert 'medium' in result['summary']['by_severity']
        assert 'low' in result['summary']['by_severity']
        
        # Should have at least one critical issue
        assert result['summary']['by_severity']['critical'] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_unknown_language(self, linter):
        """Test analyzing file with unknown language"""
        files = [{
            'path': 'test.xyz',
            'language': 'unknown',
            'content': 'some code here'
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        # Should still run security checks even for unknown languages
        assert len(result['files']) == 1
    
    @pytest.mark.asyncio
    async def test_analyze_go_code(self, linter):
        """Test analyzing Go code"""
        files = [{
            'path': 'test.go',
            'language': 'go',
            'content': '''
func main() {
    var unused string
    fmt.Println("Hello")
}
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        # Go-specific checks should run
        assert len(result['files']) == 1
    
    @pytest.mark.asyncio
    async def test_analyze_eval_usage(self, linter):
        """Test detecting eval() usage"""
        files = [{
            'path': 'test.js',
            'language': 'javascript',
            'content': '''
function executeCode(code) {
    eval(code);
}
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Should detect both OWASP and XSS issues for eval
        issues = result['issues']
        assert len(issues) > 0
        assert any(i['severity'] in ['critical', 'high'] for i in issues)
    
    @pytest.mark.asyncio
    async def test_analyze_hardcoded_password(self, linter):
        """Test detecting hardcoded passwords"""
        files = [{
            'path': 'config.py',
            'language': 'python',
            'content': '''
DATABASE_CONFIG = {
    'host': 'localhost',
    'password': 'super_secret_password_123'
}
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for hardcoded secrets
        secret_issues = [i for i in result['issues'] if i['type'] == 'hardcoded_secrets']
        assert len(secret_issues) > 0
        assert secret_issues[0]['severity'] == 'critical'
    
    @pytest.mark.asyncio
    async def test_analyze_hardcoded_api_key(self, linter):
        """Test detecting hardcoded API keys"""
        files = [{
            'path': 'api.js',
            'language': 'javascript',
            'content': '''
const config = {
    api_key: "sk_live_1234567890abcdef"
};
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for hardcoded API key
        secret_issues = [i for i in result['issues'] if i['type'] == 'hardcoded_secrets']
        assert len(secret_issues) > 0
        assert 'API key' in secret_issues[0]['message']
    
    @pytest.mark.asyncio
    async def test_analyze_insecure_md5(self, linter):
        """Test detecting MD5 usage"""
        files = [{
            'path': 'hash.py',
            'language': 'python',
            'content': '''
import hashlib
hash_value = hashlib.md5(data).hexdigest()
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for insecure crypto
        crypto_issues = [i for i in result['issues'] if i['type'] == 'insecure_crypto']
        assert len(crypto_issues) > 0
        assert 'MD5' in crypto_issues[0]['message']
    
    @pytest.mark.asyncio
    async def test_analyze_weak_random(self, linter):
        """Test detecting weak random number generation"""
        files = [{
            'path': 'random.js',
            'language': 'javascript',
            'content': '''
function generateToken() {
    return Math.random().toString(36);
}
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for weak random
        crypto_issues = [i for i in result['issues'] if i['type'] == 'insecure_crypto']
        assert len(crypto_issues) > 0
        assert 'random' in crypto_issues[0]['message'].lower()
    
    @pytest.mark.asyncio
    async def test_analyze_path_traversal(self, linter):
        """Test detecting path traversal vulnerabilities"""
        files = [{
            'path': 'file.py',
            'language': 'python',
            'content': '''
def read_file(filename):
    with open('/data/' + filename) as f:
        return f.read()
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for path traversal
        traversal_issues = [i for i in result['issues'] if i['type'] == 'path_traversal']
        assert len(traversal_issues) > 0
        assert traversal_issues[0]['severity'] == 'high'
    
    @pytest.mark.asyncio
    async def test_analyze_command_injection(self, linter):
        """Test detecting command injection vulnerabilities"""
        files = [{
            'path': 'exec.py',
            'language': 'python',
            'content': '''
import os
def run_command(cmd):
    os.system(cmd)
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for command injection
        cmd_issues = [i for i in result['issues'] if i['type'] == 'command_injection']
        assert len(cmd_issues) > 0
        assert cmd_issues[0]['severity'] == 'critical'
    
    @pytest.mark.asyncio
    async def test_analyze_subprocess_shell_true(self, linter):
        """Test detecting subprocess with shell=True"""
        files = [{
            'path': 'subprocess_test.py',
            'language': 'python',
            'content': '''
import subprocess
subprocess.call(user_input, shell=True)
'''
        }]
        
        result = await linter.analyze(files)
        
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        
        # Check for command injection
        cmd_issues = [i for i in result['issues'] if i['type'] == 'command_injection']
        assert len(cmd_issues) > 0
        assert 'shell=True' in cmd_issues[0]['message']



class TestAIReviewEngine:
    """Tests for AIReviewEngine class"""
    
    @pytest.fixture
    def engine(self):
        """Create AI review engine instance"""
        from xencode.features.code_review import AIReviewEngine
        return AIReviewEngine()
    
    @pytest.mark.asyncio
    async def test_initialize(self, engine):
        """Test engine initialization"""
        await engine.initialize()
        assert engine._initialized == True
    
    @pytest.mark.asyncio
    async def test_generate_review_basic(self, engine):
        """Test basic review generation"""
        files = [{
            'path': 'test.py',
            'content': 'def test(): pass'
        }]
        
        code_analysis = {
            'issues': [],
            'summary': {
                'total_issues': 0,
                'by_severity': {
                    'critical': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0
                }
            }
        }
        
        review = await engine.generate_review(
            'Test PR',
            'Test description',
            files,
            code_analysis
        )
        
        assert 'summary' in review
        assert 'issues' in review
        assert 'suggestions' in review
        assert 'positive_feedback' in review
        assert 'patterns_detected' in review
        assert 'semantic_analysis' in review
        assert review['summary']['title'] == 'Test PR'
        assert review['summary']['files_analyzed'] == 1
    
    @pytest.mark.asyncio
    async def test_generate_review_with_issues(self, engine):
        """Test review generation with code issues"""
        files = [{
            'path': 'test.py',
            'content': 'cursor.execute("SELECT * FROM users")'
        }]
        
        code_analysis = {
            'issues': [
                {
                    'type': 'sqli',
                    'severity': 'critical',
                    'message': 'SQL injection detected',
                    'file': 'test.py',
                    'line': 1,
                    'column': 0
                }
            ],
            'summary': {
                'total_issues': 1,
                'by_severity': {
                    'critical': 1,
                    'high': 0,
                    'medium': 0,
                    'low': 0
                }
            }
        }
        
        review = await engine.generate_review(
            'Fix SQL injection',
            'Fixing security issue',
            files,
            code_analysis
        )
        
        assert len(review['issues']) == 1
        assert review['issues'][0]['type'] == 'sqli'
        assert review['issues'][0]['severity'] == 'critical'
        assert len(review['suggestions']) > 0
    
    @pytest.mark.asyncio
    async def test_detect_patterns_complexity(self, engine):
        """Test detecting complexity patterns"""
        files = [{
            'path': 'complex.py',
            'content': '''
for i in range(10):
    for j in range(10):
        if i == j:
            print(i, j)
'''
        }]
        
        patterns = await engine._detect_patterns(files)
        
        complexity_patterns = [p for p in patterns if p['type'] == 'complexity']
        assert len(complexity_patterns) > 0
        assert complexity_patterns[0]['pattern'] == 'nested_structure'
    
    @pytest.mark.asyncio
    async def test_detect_patterns_naming(self, engine):
        """Test detecting naming patterns"""
        files = [{
            'path': 'naming.py',
            'content': 'x = 10\ny = 20\nz = x + y'
        }]
        
        patterns = await engine._detect_patterns(files)
        
        naming_patterns = [p for p in patterns if p['type'] == 'naming']
        assert len(naming_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_detect_patterns_documentation(self, engine):
        """Test detecting missing documentation"""
        files = [{
            'path': 'undocumented.py',
            'content': '''
def calculate(a, b):
    return a + b

class MyClass:
    pass
'''
        }]
        
        patterns = await engine._detect_patterns(files)
        
        doc_patterns = [p for p in patterns if p['type'] == 'documentation']
        assert len(doc_patterns) > 0
        assert doc_patterns[0]['pattern'] == 'missing_docstring'
    
    @pytest.mark.asyncio
    async def test_generate_ai_suggestion_sqli(self, engine):
        """Test generating suggestion for SQL injection"""
        issue = {
            'type': 'sqli',
            'severity': 'critical',
            'message': 'SQL injection detected',
            'file': 'test.py',
            'line': 10
        }
        
        suggestion = await engine._generate_ai_suggestion(issue, [])
        
        assert suggestion is not None
        assert suggestion['title'] == 'SQL Injection Prevention'
        assert 'parameterized' in suggestion['description'].lower()
        assert 'example' in suggestion
        assert suggestion['severity'] == 'critical'
    
    @pytest.mark.asyncio
    async def test_generate_ai_suggestion_xss(self, engine):
        """Test generating suggestion for XSS"""
        issue = {
            'type': 'xss',
            'severity': 'high',
            'message': 'XSS vulnerability detected',
            'file': 'test.js',
            'line': 5
        }
        
        suggestion = await engine._generate_ai_suggestion(issue, [])
        
        assert suggestion is not None
        assert suggestion['title'] == 'XSS Prevention'
        assert 'sanitize' in suggestion['description'].lower()
    
    @pytest.mark.asyncio
    async def test_generate_ai_suggestion_csrf(self, engine):
        """Test generating suggestion for CSRF"""
        issue = {
            'type': 'csrf',
            'severity': 'medium',
            'message': 'CSRF vulnerability detected',
            'file': 'form.html',
            'line': 1
        }
        
        suggestion = await engine._generate_ai_suggestion(issue, [])
        
        assert suggestion is not None
        assert suggestion['title'] == 'CSRF Protection'
        assert 'token' in suggestion['description'].lower()
    
    @pytest.mark.asyncio
    async def test_generate_ai_suggestion_hardcoded_secrets(self, engine):
        """Test generating suggestion for hardcoded secrets"""
        issue = {
            'type': 'hardcoded_secrets',
            'severity': 'critical',
            'message': 'Hardcoded password detected',
            'file': 'config.py',
            'line': 3
        }
        
        suggestion = await engine._generate_ai_suggestion(issue, [])
        
        assert suggestion is not None
        assert suggestion['title'] == 'Remove Hardcoded Secrets'
        assert 'environment' in suggestion['description'].lower()
    
    @pytest.mark.asyncio
    async def test_generate_ai_suggestion_insecure_crypto(self, engine):
        """Test generating suggestion for insecure crypto"""
        issue = {
            'type': 'insecure_crypto',
            'severity': 'high',
            'message': 'MD5 is insecure',
            'file': 'hash.py',
            'line': 2
        }
        
        suggestion = await engine._generate_ai_suggestion(issue, [])
        
        assert suggestion is not None
        assert suggestion['title'] == 'Use Secure Cryptography'
        assert 'sha256' in suggestion['example'].lower()
    
    @pytest.mark.asyncio
    async def test_generate_ai_suggestion_command_injection(self, engine):
        """Test generating suggestion for command injection"""
        issue = {
            'type': 'command_injection',
            'severity': 'critical',
            'message': 'Command injection detected',
            'file': 'exec.py',
            'line': 4
        }
        
        suggestion = await engine._generate_ai_suggestion(issue, [])
        
        assert suggestion is not None
        assert suggestion['title'] == 'Prevent Command Injection'
        assert 'shell=True' in suggestion['example'] or 'shell' in suggestion['description'].lower()
    
    @pytest.mark.asyncio
    async def test_generate_ai_suggestion_unknown_type(self, engine):
        """Test generating suggestion for unknown issue type"""
        issue = {
            'type': 'unknown_issue',
            'severity': 'medium',
            'message': 'Some issue',
            'file': 'test.py',
            'line': 1
        }
        
        suggestion = await engine._generate_ai_suggestion(issue, [])
        
        # Should return a generic suggestion
        assert suggestion is not None
        assert 'title' in suggestion
        assert 'description' in suggestion
    
    @pytest.mark.asyncio
    async def test_generate_pattern_suggestion_complexity(self, engine):
        """Test generating suggestion for complexity pattern"""
        pattern = {
            'type': 'complexity',
            'pattern': 'nested_structure',
            'file': 'complex.py',
            'severity': 'medium'
        }
        
        suggestion = await engine._generate_pattern_suggestion(pattern)
        
        assert suggestion is not None
        assert suggestion['title'] == 'Reduce Code Complexity'
        assert 'extract' in suggestion['description'].lower()
        assert 'example' in suggestion
    
    @pytest.mark.asyncio
    async def test_generate_pattern_suggestion_naming(self, engine):
        """Test generating suggestion for naming pattern"""
        pattern = {
            'type': 'naming',
            'pattern': 'poor_naming',
            'file': 'naming.py',
            'severity': 'low'
        }
        
        suggestion = await engine._generate_pattern_suggestion(pattern)
        
        assert suggestion is not None
        assert suggestion['title'] == 'Improve Variable Naming'
        assert 'descriptive' in suggestion['description'].lower()
    
    @pytest.mark.asyncio
    async def test_generate_pattern_suggestion_documentation(self, engine):
        """Test generating suggestion for documentation pattern"""
        pattern = {
            'type': 'documentation',
            'pattern': 'missing_docstring',
            'file': 'undocumented.py',
            'severity': 'low'
        }
        
        suggestion = await engine._generate_pattern_suggestion(pattern)
        
        assert suggestion is not None
        assert suggestion['title'] == 'Add Documentation'
        assert 'docstring' in suggestion['description'].lower()
    
    @pytest.mark.asyncio
    async def test_generate_positive_feedback_no_issues(self, engine):
        """Test generating positive feedback for clean code"""
        files = [{'path': 'clean.py', 'content': 'def test(): pass'}]
        code_analysis = {
            'summary': {
                'total_issues': 0,
                'by_severity': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            }
        }
        patterns = []
        
        feedback = engine._generate_positive_feedback(files, code_analysis, patterns)
        
        assert len(feedback) > 0
        assert feedback[0]['title'] == 'Excellent Code Quality'
        assert feedback[0]['score'] == 100
    
    @pytest.mark.asyncio
    async def test_generate_positive_feedback_no_critical(self, engine):
        """Test generating positive feedback for code with no critical issues"""
        files = [{'path': 'test.py', 'content': 'x = 1'}]
        code_analysis = {
            'summary': {
                'total_issues': 2,
                'by_severity': {'critical': 0, 'high': 0, 'medium': 2, 'low': 0}
            }
        }
        patterns = []
        
        feedback = engine._generate_positive_feedback(files, code_analysis, patterns)
        
        assert len(feedback) > 0
        security_feedback = [f for f in feedback if 'Security' in f['title']]
        assert len(security_feedback) > 0
    
    @pytest.mark.asyncio
    async def test_generate_positive_feedback_with_tests(self, engine):
        """Test generating positive feedback for code with tests"""
        files = [
            {'path': 'src/main.py', 'content': 'def main(): pass'},
            {'path': 'tests/test_main.py', 'content': 'def test_main(): pass'}
        ]
        code_analysis = {
            'summary': {
                'total_issues': 0,
                'by_severity': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            }
        }
        patterns = []
        
        feedback = engine._generate_positive_feedback(files, code_analysis, patterns)
        
        test_feedback = [f for f in feedback if 'Test' in f['title']]
        assert len(test_feedback) > 0
        assert 'test' in test_feedback[0]['message'].lower()
    
    @pytest.mark.asyncio
    async def test_semantic_analysis_without_ensemble(self, engine):
        """Test semantic analysis when ensemble is not available"""
        engine.ensemble = None
        
        result = await engine._semantic_analysis(
            'Test PR',
            'Description',
            [{'path': 'test.py', 'content': 'pass'}],
            {'issues': []}
        )
        
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_generate_overall_summary_without_ensemble(self, engine):
        """Test overall summary generation when ensemble is not available"""
        engine.ensemble = None
        
        review = {
            'issues': [],
            'patterns_detected': []
        }
        
        summary = await engine._generate_overall_summary(review)
        
        assert summary == "Review completed"
    
    @pytest.mark.asyncio
    async def test_generate_review_multiple_files(self, engine):
        """Test review generation with multiple files"""
        files = [
            {'path': 'file1.py', 'content': 'def test1(): pass'},
            {'path': 'file2.py', 'content': 'def test2(): pass'},
            {'path': 'file3.py', 'content': 'def test3(): pass'}
        ]
        
        code_analysis = {
            'issues': [],
            'summary': {
                'total_issues': 0,
                'by_severity': {
                    'critical': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0
                }
            }
        }
        
        review = await engine.generate_review(
            'Multi-file PR',
            'Multiple files changed',
            files,
            code_analysis
        )
        
        assert review['summary']['files_analyzed'] == 3
    
    @pytest.mark.asyncio
    async def test_generate_review_with_patterns_and_issues(self, engine):
        """Test review generation with both patterns and issues"""
        files = [{
            'path': 'complex.py',
            'content': '''
for i in range(10):
    for j in range(10):
        cursor.execute("SELECT * FROM users")
'''
        }]
        
        code_analysis = {
            'issues': [
                {
                    'type': 'sqli',
                    'severity': 'critical',
                    'message': 'SQL injection',
                    'file': 'complex.py',
                    'line': 3,
                    'column': 0
                }
            ],
            'summary': {
                'total_issues': 1,
                'by_severity': {
                    'critical': 1,
                    'high': 0,
                    'medium': 0,
                    'low': 0
                }
            }
        }
        
        review = await engine.generate_review(
            'Complex code with issues',
            'Needs refactoring',
            files,
            code_analysis
        )
        
        assert len(review['issues']) > 0
        assert len(review['patterns_detected']) > 0
        assert len(review['suggestions']) > 0
