"""
AI Code Reviewer Feature

Intelligent code review system that analyzes pull requests and provides
actionable suggestions for improving code quality and security.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from .base import FeatureBase, FeatureConfig, FeatureStatus


@dataclass
class CodeReviewConfig:
    """Configuration for code review"""
    enabled: bool = True
    supported_languages: List[str] = field(default_factory=lambda: ['python', 'javascript', 'typescript', 'go', 'rust'])
    severity_levels: List[str] = field(default_factory=lambda: ['critical', 'high', 'medium', 'low'])
    integrations: List[str] = field(default_factory=lambda: ['github', 'gitlab', 'bitbucket'])
    security_checks: List[str] = field(default_factory=lambda: ['owasp', 'sqli', 'xss', 'csrf'])
    output_format: str = 'json'


class CodeReviewFeature(FeatureBase):
    """AI Code Reviewer feature implementation"""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.config = config
        self.status = FeatureStatus.DISABLED
        self._initialized = False
        self._pr_analyzers: Dict[str, Any] = {}
        self._code_linter: Optional[Any] = None
        self._review_engine: Optional[Any] = None
        self._report_generator: Optional[ReportGenerator] = None
    
    @property
    def name(self) -> str:
        """Feature name"""
        return 'code_review'
    
    @property
    def description(self) -> str:
        """Feature description"""
        return 'AI-powered code review system for pull requests and code analysis'
    
    async def _initialize(self) -> None:
        """Initialize the code review feature"""
        # Initialize PR analyzers for each supported platform
        self._pr_analyzers = {
            'github': GitHubPRAnalyzer(),
            'gitlab': GitLabPRAnalyzer(),
            'bitbucket': BitbucketPRAnalyzer()
        }
        
        # Initialize code linter
        self._code_linter = CodeLinter()
        
        # Initialize AI review engine
        self._review_engine = AIReviewEngine()
        
        # Initialize report generator
        self._report_generator = ReportGenerator()
        
        self._initialized = True
    
    async def _shutdown(self) -> None:
        """Shutdown the code review feature"""
        self._pr_analyzers.clear()
        self._code_linter = None
        self._review_engine = None
        self._report_generator = None
        self._initialized = False
    
    async def analyze_pr(self, pr_url: str, platform: str = 'github') -> Dict[str, Any]:
        """Analyze a pull request"""
        if platform not in self._pr_analyzers:
            raise ValueError(f"Unsupported platform: {platform}")
        
        analyzer = self._pr_analyzers[platform]
        pr_data = await analyzer.fetch_pr(pr_url)
        
        if not pr_data:
            raise ValueError(f"Failed to fetch PR: {pr_url}")
        
        # Analyze code changes
        code_analysis = await self._code_linter.analyze(pr_data['files'])
        
        # Generate AI review
        review = await self._review_engine.generate_review(
            pr_data['title'],
            pr_data['description'],
            pr_data['files'],
            code_analysis
        )
        
        # Generate report
        report = self._generate_report(pr_data, code_analysis, review)
        
        return report
    
    async def analyze_file(self, file_path: str, language: str = None) -> Dict[str, Any]:
        """Analyze a single file"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r') as f:
            content = f.read()
        
        # Analyze the file
        analysis = await self._code_linter.analyze([{
            'path': str(path),
            'content': content,
            'language': language or self._detect_language(path)
        }])
        
        # Generate AI review
        review = await self._review_engine.generate_review(
            path.name,
            '',
            [{'path': str(path), 'content': content}],
            analysis
        )
        
        return {
            'file': str(path),
            'analysis': analysis,
            'review': review
        }
    
    async def analyze_directory(self, directory: str, patterns: List[str] = None) -> Dict[str, Any]:
        """Analyze an entire directory"""
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find files to analyze
        files = []
        for pattern in patterns or ['**/*.py', '**/*.js', '**/*.ts', '**/*.go', '**/*.rs']:
            files.extend(dir_path.glob(pattern))
        
        # Analyze each file
        file_analyses = {}
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                analysis = await self._code_linter.analyze([{
                    'path': str(file_path),
                    'content': content,
                    'language': self._detect_language(file_path)
                }])
                
                file_analyses[str(file_path)] = analysis
            except Exception as e:
                file_analyses[str(file_path)] = {'error': str(e)}
        
        return {
            'directory': str(directory),
            'files_analyzed': len(file_analyses),
            'analyses': file_analyses
        }
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.sh': 'bash',
            '.md': 'markdown'
        }
        
        return extensions.get(file_path.suffix.lower(), 'unknown')
    
    def _generate_report(self, pr_data: Dict[str, Any], 
                        code_analysis: Dict[str, Any],
                        review: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive review report"""
        # Count issues by severity
        severity_counts = {}
        for issue in review.get('issues', []):
            severity = issue.get('severity', 'medium')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Calculate overall quality score
        total_issues = len(review.get('issues', []))
        critical_issues = severity_counts.get('critical', 0)
        high_issues = severity_counts.get('high', 0)
        
        quality_score = 100
        quality_score -= critical_issues * 20
        quality_score -= high_issues * 10
        quality_score -= severity_counts.get('medium', 0) * 5
        quality_score -= severity_counts.get('low', 0) * 2
        quality_score = max(0, quality_score)
        
        return {
            'pr': {
                'url': pr_data.get('url', ''),
                'title': pr_data.get('title', ''),
                'description': pr_data.get('description', ''),
                'platform': pr_data.get('platform', 'github')
            },
            'files_analyzed': len(pr_data.get('files', [])),
            'code_analysis': code_analysis,
            'review': review,
            'summary': {
                'total_issues': total_issues,
                'severity_counts': severity_counts,
                'quality_score': quality_score,
                'generated_at': pr_data.get('timestamp', '')
            }
        }
    
    def generate_formatted_report(self, review: Dict[str, Any], 
                                  pr_data: Dict[str, Any] = None,
                                  format: str = 'text') -> str:
        """Generate a formatted review report
        
        Args:
            review: Review data from AIReviewEngine
            pr_data: Optional PR metadata
            format: Report format ('text', 'markdown', 'html', 'json')
            
        Returns:
            Formatted report string
        """
        if not self._report_generator:
            self._report_generator = ReportGenerator()
        
        if format == 'text':
            return self._report_generator.generate_text_report(review, pr_data)
        elif format == 'markdown':
            return self._report_generator.generate_markdown_report(review, pr_data)
        elif format == 'html':
            return self._report_generator.generate_html_report(review, pr_data)
        elif format == 'json':
            import json
            return json.dumps(self._report_generator.generate_json_report(review, pr_data), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_cli_commands(self) -> List[Any]:
        """Get CLI commands for this feature"""
        from .core.cli import FeatureCommandGroup
        
        # This will be implemented in task 2.1.8
        return []
    
    def get_tui_components(self) -> List[Any]:
        """Get TUI components for this feature"""
        from xencode.tui.widgets.code_review_panel import (
            CodeReviewPanel,
            ReviewHistoryPanel,
            ReviewSummaryPanel
        )
        
        return [
            {
                'name': 'code_review_panel',
                'class': CodeReviewPanel,
                'description': 'Main code review interface with PR analysis'
            },
            {
                'name': 'review_history',
                'class': ReviewHistoryPanel,
                'description': 'Review history panel'
            },
            {
                'name': 'review_summary',
                'class': ReviewSummaryPanel,
                'description': 'Review summary statistics'
            }
        ]
    
    def get_api_endpoints(self) -> List[Any]:
        """Get API endpoints for this feature"""
        # This will be implemented in task 1.4.1
        return []


class GitHubPRAnalyzer:
    """Analyzer for GitHub pull requests"""
    
    def __init__(self, token: str = None):
        self.base_url = "https://api.github.com"
        self.token = token
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            import aiohttp
            headers = {'Accept': 'application/vnd.github.v3+json'}
            if self.token:
                headers['Authorization'] = f'token {self.token}'
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def _close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()
    
    def _parse_pr_url(self, pr_url: str) -> Dict[str, str]:
        """Parse GitHub PR URL and extract repository info"""
        # Pattern: https://github.com/owner/repo/pull/number
        pattern = r'https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)'
        match = re.match(pattern, pr_url)
        
        if not match:
            raise ValueError(f"Invalid GitHub PR URL: {pr_url}")
        
        return {
            'owner': match.group(1),
            'repo': match.group(2),
            'pr_number': match.group(3)
        }
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make GitHub API request with pagination support"""
        session = await self._get_session()
        
        url = f"{self.base_url}{endpoint}"
        all_results = []
        page = 1
        max_pages = 10  # Prevent infinite loops
        
        while page <= max_pages:
            request_params = params.copy() if params else {}
            request_params['page'] = page
            
            try:
                async with session.get(url, params=request_params) as response:
                    if response.status == 404:
                        raise ValueError(f"Resource not found: {endpoint}")
                    if response.status == 401:
                        raise ValueError("GitHub API authentication failed. Please provide a valid token.")
                    if response.status == 403:
                        raise ValueError(f"GitHub API rate limit exceeded. Endpoint: {endpoint}")
                    if response.status != 200:
                        raise ValueError(f"GitHub API request failed with status {response.status}")
                    
                    data = await response.json()
                    
                    # Handle both single response and list response
                    if isinstance(data, list):
                        all_results.extend(data)
                        # GitHub API returns max 30 items per page when there are more pages
                        # We need to check if we got exactly 30 items and continue to next page
                        if len(data) < 30:
                            break
                    else:
                        return data
                    
                    page += 1
                    
            except aiohttp.ClientError as e:
                raise ConnectionError(f"Failed to connect to GitHub API: {e}")
        
        return all_results
    
    async def fetch_pr(self, pr_url: str) -> Optional[Dict[str, Any]]:
        """Fetch PR data from GitHub"""
        repo_info = self._parse_pr_url(pr_url)
        
        try:
            # Fetch PR details
            pr_endpoint = f"/repos/{repo_info['owner']}/{repo_info['repo']}/pulls/{repo_info['pr_number']}"
            pr_data = await self._make_request(pr_endpoint)
            
            # Fetch PR files (changes)
            files_endpoint = f"/repos/{repo_info['owner']}/{repo_info['repo']}/pulls/{repo_info['pr_number']}/files"
            files = await self._make_request(files_endpoint, {'per_page': 100})
            
            # Fetch PR commits
            commits_endpoint = f"/repos/{repo_info['owner']}/{repo_info['repo']}/pulls/{repo_info['pr_number']}/commits"
            commits = await self._make_request(commits_endpoint, {'per_page': 100})
            
            # Fetch PR reviews/comments
            reviews_endpoint = f"/repos/{repo_info['owner']}/{repo_info['repo']}/pulls/{repo_info['pr_number']}/reviews"
            reviews = await self._make_request(reviews_endpoint, {'per_page': 100})
            
            # Extract commit messages
            commit_messages = []
            for commit in commits:
                if isinstance(commit, dict):
                    commit_messages.append({
                        'sha': commit.get('sha', ''),
                        'message': commit.get('commit', {}).get('message', ''),
                        'author': commit.get('commit', {}).get('author', {})
                    })
            
            # Extract file changes
            file_changes = []
            for file_data in files:
                file_changes.append({
                    'filename': file_data.get('filename', ''),
                    'status': file_data.get('status', ''),
                    'additions': file_data.get('additions', 0),
                    'deletions': file_data.get('deletions', 0),
                    'changes': file_data.get('changes', 0),
                    'patch': file_data.get('patch', ''),
                    'blob_url': file_data.get('blob_url', '')
                })
            
            # Extract line comments
            line_comments = []
            for review in reviews:
                if isinstance(review, dict):
                    review_comments = review.get('comments', [])
                    for comment in review_comments:
                        if isinstance(comment, dict):
                            line_comments.append({
                                'body': comment.get('body', ''),
                                'path': comment.get('path', ''),
                                'line': comment.get('line', 0),
                                'user': comment.get('user', {}).get('login', ''),
                                'created_at': comment.get('created_at', '')
                            })
            
            # Determine PR state
            state = pr_data.get('state', 'open')
            merged = pr_data.get('merged', False)
            
            if state == 'closed' and merged:
                pr_state = 'merged'
            elif state == 'closed':
                pr_state = 'closed'
            else:
                pr_state = 'open'
            
            return {
                'url': pr_url,
                'title': pr_data.get('title', ''),
                'description': pr_data.get('body', ''),
                'platform': 'github',
                'timestamp': pr_data.get('created_at', ''),
                'state': pr_state,
                'author': pr_data.get('user', {}).get('login', ''),
                'base_branch': pr_data.get('base', {}).get('ref', ''),
                'head_branch': pr_data.get('head', {}).get('ref', ''),
                'files': file_changes,
                'commits': commit_messages,
                'comments': line_comments,
                'review_count': len(reviews),
                'additions': pr_data.get('additions', 0),
                'deletions': pr_data.get('deletions', 0),
                'changed_files': pr_data.get('changed_files', 0)
            }
            
        except Exception as e:
            raise ValueError(f"Failed to fetch PR {pr_url}: {str(e)}")
        finally:
            await self._close_session()
    
    async def fetch_pr_by_repo(self, owner: str, repo: str, pr_number: int) -> Optional[Dict[str, Any]]:
        """Fetch PR data using owner, repo, and PR number directly"""
        pr_url = f"https://github.com/{owner}/{repo}/pull/{pr_number}"
        return await self.fetch_pr(pr_url)


class GitLabPRAnalyzer:
    """Analyzer for GitLab merge requests"""
    
    def __init__(self, token: str = None):
        self.base_url = "https://gitlab.com/api/v4"
        self.token = token
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            import aiohttp
            headers = {'Accept': 'application/json'}
            if self.token:
                headers['PRIVATE-TOKEN'] = self.token
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def _close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()
    
    def _parse_mr_url(self, mr_url: str) -> Dict[str, Any]:
        """Parse GitLab MR URL and extract project info"""
        # Pattern: https://gitlab.com/owner/repo/-/merge_requests/number
        # or: https://gitlab.com/owner/repo/merge_requests/number
        pattern = r'https?://gitlab\.com/([^/]+)/([^/-]+)(?:/-)?/merge_requests/(\d+)'
        match = re.match(pattern, mr_url)
        
        if not match:
            raise ValueError(f"Invalid GitLab MR URL: {mr_url}")
        
        return {
            'owner': match.group(1),
            'repo': match.group(2),
            'mr_number': match.group(3)
        }
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Make GitLab API request with pagination support"""
        session = await self._get_session()
        
        url = f"{self.base_url}{endpoint}"
        all_results = []
        page = 1
        per_page = 100  # GitLab supports up to 100 items per page
        max_pages = 10  # Prevent infinite loops
        
        while page <= max_pages:
            request_params = params.copy() if params else {}
            request_params['page'] = page
            request_params['per_page'] = per_page
            
            try:
                async with session.get(url, params=request_params) as response:
                    if response.status == 404:
                        raise ValueError(f"Resource not found: {endpoint}")
                    if response.status == 401:
                        raise ValueError("GitLab API authentication failed. Please provide a valid token.")
                    if response.status == 403:
                        raise ValueError(f"GitLab API rate limit exceeded. Endpoint: {endpoint}")
                    if response.status != 200:
                        raise ValueError(f"GitLab API request failed with status {response.status}")
                    
                    data = await response.json()
                    
                    # GitLab API returns a list for list endpoints
                    if isinstance(data, list):
                        all_results.extend(data)
                        # If we got fewer items than per_page, we've reached the last page
                        if len(data) < per_page:
                            break
                    else:
                        return [data]
                    
                    page += 1
                    
            except aiohttp.ClientError as e:
                raise ConnectionError(f"Failed to connect to GitLab API: {e}")
        
        return all_results
    
    async def fetch_pr(self, mr_url: str) -> Optional[Dict[str, Any]]:
        """Fetch MR data from GitLab"""
        repo_info = self._parse_mr_url(mr_url)
        
        try:
            # URL encode the project path (owner/repo)
            project_path = f"{repo_info['owner']}/{repo_info['repo']}"
            
            # Fetch MR details
            mr_endpoint = f"/projects/{project_path}/merge_requests/{repo_info['mr_number']}"
            mr_data = await self._make_request(mr_endpoint)
            
            if not mr_data:
                raise ValueError(f"MR not found: {mr_url}")
            
            mr = mr_data[0]
            
            # Fetch MR changes (files)
            changes_endpoint = f"/projects/{project_path}/merge_requests/{repo_info['mr_number']}/changes"
            changes_data = await self._make_request(changes_endpoint)
            changes = changes_data[0] if changes_data else {}
            
            # Fetch MR commits
            commits_endpoint = f"/projects/{project_path}/merge_requests/{repo_info['mr_number']}/commits"
            commits = await self._make_request(commits_endpoint)
            
            # Fetch MR discussions (comments)
            discussions_endpoint = f"/projects/{project_path}/merge_requests/{repo_info['mr_number']}/discussions"
            discussions = await self._make_request(discussions_endpoint)
            
            # Extract commit messages
            commit_messages = []
            for commit in commits:
                if isinstance(commit, dict):
                    commit_messages.append({
                        'sha': commit.get('id', ''),
                        'message': commit.get('message', ''),
                        'author': {
                            'name': commit.get('author_name', ''),
                            'email': commit.get('author_email', '')
                        }
                    })
            
            # Extract file changes from changes data
            file_changes = []
            if changes:
                changes_list = changes.get('changes', [])
                for change in changes_list:
                    file_changes.append({
                        'filename': change.get('new_path', change.get('old_path', '')),
                        'status': self._map_change_status(change),
                        'additions': change.get('new_lines', 0),
                        'deletions': change.get('old_lines', 0),
                        'changes': change.get('new_lines', 0) + change.get('old_lines', 0),
                        'patch': change.get('diff', ''),
                        'blob_url': ''
                    })
            
            # Extract comments from discussions
            line_comments = []
            for discussion in discussions:
                if isinstance(discussion, dict):
                    notes = discussion.get('notes', [])
                    for note in notes:
                        if isinstance(note, dict) and note.get('type') == 'DiffNote':
                            line_comments.append({
                                'body': note.get('body', ''),
                                'path': note.get('path', ''),
                                'line': note.get('line', 0),
                                'user': note.get('author', {}).get('username', ''),
                                'created_at': note.get('created_at', '')
                            })
            
            # Determine MR state
            state = mr.get('state', 'opened')
            merged = mr.get('merged', False)
            
            if merged:
                pr_state = 'merged'
            elif state == 'closed':
                pr_state = 'closed'
            else:
                pr_state = 'open'
            
            return {
                'url': mr_url,
                'title': mr.get('title', ''),
                'description': mr.get('description', ''),
                'platform': 'gitlab',
                'timestamp': mr.get('created_at', ''),
                'state': pr_state,
                'author': mr.get('author', {}).get('username', ''),
                'base_branch': mr.get('target_branch', ''),
                'head_branch': mr.get('source_branch', ''),
                'files': file_changes,
                'commits': commit_messages,
                'comments': line_comments,
                'review_count': len(discussions),
                'additions': mr.get('additions', 0),
                'deletions': mr.get('deletions', 0),
                'changed_files': mr.get('changed_files', 0)
            }
            
        except Exception as e:
            raise ValueError(f"Failed to fetch MR {mr_url}: {str(e)}")
        finally:
            await self._close_session()
    
    def _map_change_status(self, change: Dict[str, Any]) -> str:
        """Map GitLab change status to standard format"""
        diff = change.get('diff', '')
        if diff.startswith('new file'):
            return 'added'
        elif diff.startswith('deleted file'):
            return 'deleted'
        elif diff.startswith('rename'):
            return 'renamed'
        else:
            return 'modified'
    
    async def fetch_pr_by_repo(self, owner: str, repo: str, mr_number: int) -> Optional[Dict[str, Any]]:
        """Fetch MR data using owner, repo, and MR number directly"""
        mr_url = f"https://gitlab.com/{owner}/{repo}/merge_requests/{mr_number}"
        return await self.fetch_pr(mr_url)


class BitbucketPRAnalyzer:
    """Analyzer for Bitbucket pull requests"""

    def __init__(self, token: str = None):
        self.base_url = "https://api.bitbucket.org/2.0"
        self.token = token
        self.session = None

    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            import aiohttp
            headers = {'Accept': 'application/json'}
            if self.token:
                headers['Authorization'] = f'Bearer {self.token}'
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def _close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()

    def _parse_pr_url(self, pr_url: str) -> Dict[str, str]:
        """Parse Bitbucket PR URL and extract repository info"""
        # Pattern: https://bitbucket.org/workspace/repo/pull-requests/number
        # or: https://bitbucket.org/workspace/repo/pull-request/number
        pattern = r'https?://bitbucket\.org/([^/]+)/([^/]+)/pull-requests?/(\d+)'
        match = re.match(pattern, pr_url)

        if not match:
            raise ValueError(f"Invalid Bitbucket PR URL: {pr_url}")

        return {
            'workspace': match.group(1),
            'repo': match.group(2),
            'pr_number': match.group(3)
        }

    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Make Bitbucket API request with pagination support"""
        session = await self._get_session()

        url = f"{self.base_url}{endpoint}"
        all_results = []
        page = 1
        per_page = 50  # Bitbucket supports up to 50 items per page
        max_pages = 10  # Prevent infinite loops

        while page <= max_pages:
            request_params = params.copy() if params else {}
            request_params['page'] = page
            request_params['pagelen'] = per_page

            try:
                async with session.get(url, params=request_params) as response:
                    if response.status == 404:
                        raise ValueError(f"Resource not found: {endpoint}")
                    if response.status == 401:
                        raise ValueError("Bitbucket API authentication failed. Please provide a valid token.")
                    if response.status == 403:
                        raise ValueError(f"Bitbucket API rate limit exceeded. Endpoint: {endpoint}")
                    if response.status != 200:
                        raise ValueError(f"Bitbucket API request failed with status {response.status}")

                    data = await response.json()

                    # Bitbucket API returns a dict with 'values' list for list endpoints
                    if isinstance(data, dict) and 'values' in data:
                        values = data.get('values', [])
                        all_results.extend(values)

                        # Check if there are more pages
                        next_page = data.get('next')
                        if not next_page or len(values) < per_page:
                            break

                        page += 1
                    else:
                        return [data]

            except aiohttp.ClientError as e:
                raise ConnectionError(f"Failed to connect to Bitbucket API: {e}")

        return all_results

    async def fetch_pr(self, pr_url: str) -> Optional[Dict[str, Any]]:
        """Fetch PR data from Bitbucket"""
        repo_info = self._parse_pr_url(pr_url)

        try:
            workspace = repo_info['workspace']
            repo = repo_info['repo']
            pr_number = repo_info['pr_number']

            # Fetch PR details
            pr_endpoint = f"/repositories/{workspace}/{repo}/pullrequests/{pr_number}"
            pr_data = await self._make_request(pr_endpoint)

            if not pr_data:
                raise ValueError(f"PR not found: {pr_url}")

            pr = pr_data[0]

            # Fetch PR files (changes)
            files_endpoint = f"/repositories/{workspace}/{repo}/pullrequests/{pr_number}/changes"
            files = await self._make_request(files_endpoint)

            # Fetch PR commits
            commits_endpoint = f"/repositories/{workspace}/{repo}/pullrequests/{pr_number}/commits"
            commits = await self._make_request(commits_endpoint)

            # Fetch PR activities (comments)
            activities_endpoint = f"/repositories/{workspace}/{repo}/pullrequests/{pr_number}/activities"
            activities = await self._make_request(activities_endpoint)

            # Extract commit messages
            commit_messages = []
            for commit in commits:
                if isinstance(commit, dict):
                    commit_messages.append({
                        'sha': commit.get('hash', ''),
                        'message': commit.get('message', ''),
                        'author': {
                            'name': commit.get('author', {}).get('user', {}).get('display_name', ''),
                            'email': commit.get('author', {}).get('user', {}).get('email_address', '')
                        }
                    })

            # Extract file changes
            file_changes = []
            for file_data in files:
                if isinstance(file_data, dict):
                    file_changes.append({
                        'filename': file_data.get('path', {}).get('path', file_data.get('new', {}).get('path', '')),
                        'status': self._map_change_status(file_data),
                        'additions': file_data.get('additions', 0),
                        'deletions': file_data.get('deletions', 0),
                        'changes': file_data.get('additions', 0) + file_data.get('deletions', 0),
                        'patch': file_data.get('raw', ''),
                        'blob_url': ''
                    })

            # Extract comments from activities
            line_comments = []
            for activity in activities:
                if isinstance(activity, dict):
                    # Bitbucket stores comments in activities
                    if activity.get('comment') and isinstance(activity.get('comment'), dict):
                        comment = activity['comment']
                        line_comments.append({
                            'body': comment.get('content', {}).get('raw', ''),
                            'path': comment.get('file', {}).get('path', ''),
                            'line': comment.get('line', 0),
                            'user': comment.get('user', {}).get('display_name', ''),
                            'created_at': comment.get('created_on', '')
                        })

            # Determine PR state
            state = pr.get('state', 'OPEN')

            if state == 'MERGED':
                pr_state = 'merged'
            elif state == 'DECLINED':
                pr_state = 'declined'
            else:
                pr_state = 'open'

            return {
                'url': pr_url,
                'title': pr.get('title', ''),
                'description': pr.get('description', ''),
                'platform': 'bitbucket',
                'timestamp': pr.get('created_on', ''),
                'state': pr_state,
                'author': pr.get('author', {}).get('display_name', ''),
                'base_branch': pr.get('destination', {}).get('branch', {}).get('name', ''),
                'head_branch': pr.get('source', {}).get('branch', {}).get('name', ''),
                'files': file_changes,
                'commits': commit_messages,
                'comments': line_comments,
                'review_count': len([a for a in activities if a.get('comment')]),
                'additions': pr.get('summary', {}).get('additions', 0),
                'deletions': pr.get('summary', {}).get('deletions', 0),
                'changed_files': pr.get('summary', {}).get('changed_files', 0)
            }

        except Exception as e:
            raise ValueError(f"Failed to fetch PR {pr_url}: {str(e)}")
        finally:
            await self._close_session()

    def _map_change_status(self, change: Dict[str, Any]) -> str:
        """Map Bitbucket change status to standard format"""
        status = change.get('type', '')

        if status == 'add':
            return 'added'
        elif status == 'delete':
            return 'deleted'
        elif status == 'move':
            return 'renamed'
        else:
            return 'modified'

    async def fetch_pr_by_repo(self, workspace: str, repo: str, pr_number: int) -> Optional[Dict[str, Any]]:
        """Fetch PR data using workspace, repo, and PR number directly"""
        pr_url = f"https://bitbucket.org/{workspace}/{repo}/pull-requests/{pr_number}"
        return await self.fetch_pr(pr_url)



class CodeLinter:
    """Code linter with security checks"""
    
    def __init__(self):
        self.security_checks = {
            'owasp': self._check_owasp,
            'sqli': self._check_sqli,
            'xss': self._check_xss,
            'csrf': self._check_csrf,
            'hardcoded_secrets': self._check_hardcoded_secrets,
            'insecure_crypto': self._check_insecure_crypto,
            'path_traversal': self._check_path_traversal,
            'command_injection': self._check_command_injection
        }
    
    async def analyze(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze files for code quality and security issues"""
        results = {
            'files': [],
            'issues': [],
            'summary': {
                'total_files': len(files),
                'total_issues': 0,
                'by_severity': {
                    'critical': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0
                }
            }
        }
        
        for file_data in files:
            file_result = self._analyze_file(file_data)
            results['files'].append(file_result)
            results['issues'].extend(file_result['issues'])
            results['summary']['total_issues'] += len(file_result['issues'])
            
            for issue in file_result['issues']:
                severity = issue.get('severity', 'medium')
                results['summary']['by_severity'][severity] = \
                    results['summary']['by_severity'].get(severity, 0) + 1
        
        return results
    
    def _analyze_file(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single file"""
        content = file_data.get('content', '')
        path = file_data.get('path', '')
        language = file_data.get('language', 'unknown')
        
        issues = []
        
        # Run security checks
        for check_name, check_func in self.security_checks.items():
            check_issues = check_func(content, path)
            issues.extend(check_issues)
        
        # Run language-specific checks
        language_checks = {
            'python': self._check_python,
            'javascript': self._check_javascript,
            'typescript': self._check_typescript,
            'go': self._check_go,
            'rust': self._check_rust
        }
        
        if language in language_checks:
            language_issues = language_checks[language](content, path)
            issues.extend(language_issues)
        
        return {
            'path': path,
            'language': language,
            'issues': issues,
            'has_issues': len(issues) > 0
        }
    
    def _check_owasp(self, content: str, path: str) -> List[Dict[str, Any]]:
        """Check for OWASP Top 10 vulnerabilities"""
        issues = []
        
        # Check for SQL injection patterns
        sqli_patterns = [
            r"execute\s*\(\s*['\"]",
            r"eval\s*\(",
            r"exec\s*\("
        ]
        
        for pattern in sqli_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    'type': 'sqli',
                    'severity': 'critical',
                    'message': 'Potential SQL injection detected',
                    'file': path,
                    'line': 0,
                    'column': 0
                })
        
        return issues
    
    def _check_sqli(self, content: str, path: str) -> List[Dict[str, Any]]:
        """Check for SQL injection vulnerabilities"""
        issues = []
        
        sqli_patterns = [
            r"execute\s*\(\s*['\"]",
            r"cursor\.execute\s*\(",
            r"db\.query\s*\("
        ]
        
        for pattern in sqli_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    'type': 'sqli',
                    'severity': 'critical',
                    'message': 'Potential SQL injection detected',
                    'file': path,
                    'line': 0,
                    'column': 0
                })
        
        return issues
    
    def _check_xss(self, content: str, path: str) -> List[Dict[str, Any]]:
        """Check for XSS vulnerabilities"""
        issues = []
        
        xss_patterns = [
            r"innerHTML\s*=",
            r"document\.write\s*\(",
            r"eval\s*\("
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    'type': 'xss',
                    'severity': 'high',
                    'message': 'Potential XSS vulnerability detected',
                    'file': path,
                    'line': 0,
                    'column': 0
                })
        
        return issues
    
    def _check_csrf(self, content: str, path: str) -> List[Dict[str, Any]]:
        """Check for CSRF vulnerabilities"""
        issues = []
        
        # Check for missing CSRF tokens in web applications
        if 'form' in content.lower() and 'csrf' not in content.lower():
            issues.append({
                'type': 'csrf',
                'severity': 'medium',
                'message': 'Potential CSRF vulnerability - missing CSRF token',
                'file': path,
                'line': 0,
                'column': 0
            })
        
        return issues
    
    def _check_python(self, content: str, path: str) -> List[Dict[str, Any]]:
        """Python-specific code quality checks"""
        issues = []
        
        # Check for bare except clauses
        if re.search(r"except\s*:", content):
            issues.append({
                'type': 'code_quality',
                'severity': 'medium',
                'message': 'Bare except clause detected - should catch specific exceptions',
                'file': path,
                'line': 0,
                'column': 0
            })
        
        # Check for unused imports
        if re.search(r"^import\s+\w+", content, re.MULTILINE):
            issues.append({
                'type': 'code_quality',
                'severity': 'low',
                'message': 'Import statement detected - verify it is used',
                'file': path,
                'line': 0,
                'column': 0
            })
        
        return issues
    
    def _check_javascript(self, content: str, path: str) -> List[Dict[str, Any]]:
        """JavaScript-specific code quality checks"""
        issues = []
        
        # Check for == instead of ===
        if re.search(r"(?<![=!])==(?!=[^=])", content):
            issues.append({
                'type': 'code_quality',
                'severity': 'low',
                'message': 'Use === instead of == for strict equality',
                'file': path,
                'line': 0,
                'column': 0
            })
        
        return issues
    
    def _check_typescript(self, content: str, path: str) -> List[Dict[str, Any]]:
        """TypeScript-specific code quality checks"""
        issues = []
        
        # Check for any type usage
        if re.search(r":\s*any\b", content):
            issues.append({
                'type': 'code_quality',
                'severity': 'medium',
                'message': 'Avoid using "any" type - use specific types instead',
                'file': path,
                'line': 0,
                'column': 0
            })
        
        return issues
    
    def _check_go(self, content: str, path: str) -> List[Dict[str, Any]]:
        """Go-specific code quality checks"""
        issues = []
        
        # Check for unused variables
        if re.search(r"var\s+\w+\s+", content):
            issues.append({
                'type': 'code_quality',
                'severity': 'low',
                'message': 'Variable declared but may not be used',
                'file': path,
                'line': 0,
                'column': 0
            })
        
        return issues
    
    def _check_rust(self, content: str, path: str) -> List[Dict[str, Any]]:
        """Rust-specific code quality checks"""
        issues = []
        
        # Check for unwrap without proper error handling
        if re.search(r"\.unwrap\(\)", content):
            issues.append({
                'type': 'code_quality',
                'severity': 'medium',
                'message': 'unwrap() may panic - consider proper error handling',
                'file': path,
                'line': 0,
                'column': 0
            })
        
        return issues
    
    def _check_hardcoded_secrets(self, content: str, path: str) -> List[Dict[str, Any]]:
        """Check for hardcoded secrets and credentials"""
        issues = []
        
        # Patterns for common secret types (handle both = and : assignment, with or without quotes around key)
        secret_patterns = [
            (r'["\']?password["\']?\s*[:=]\s*["\'][^"\']{3,}["\']', 'Hardcoded password detected'),
            (r'["\']?api[_-]?key["\']?\s*[:=]\s*["\'][^"\']{10,}["\']', 'Hardcoded API key detected'),
            (r'["\']?secret[_-]?key["\']?\s*[:=]\s*["\'][^"\']{10,}["\']', 'Hardcoded secret key detected'),
            (r'["\']?token["\']?\s*[:=]\s*["\'][^"\']{10,}["\']', 'Hardcoded token detected'),
            (r'["\']?aws[_-]?access[_-]?key[_-]?id["\']?\s*[:=]\s*["\'][^"\']{10,}["\']', 'Hardcoded AWS access key detected'),
            (r'["\']?private[_-]?key["\']?\s*[:=]\s*["\'][^"\']{10,}["\']', 'Hardcoded private key detected'),
        ]
        
        for pattern, message in secret_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    'type': 'hardcoded_secrets',
                    'severity': 'critical',
                    'message': message,
                    'file': path,
                    'line': 0,
                    'column': 0
                })
        
        return issues
    
    def _check_insecure_crypto(self, content: str, path: str) -> List[Dict[str, Any]]:
        """Check for insecure cryptographic practices"""
        issues = []
        
        # Insecure hash algorithms
        insecure_hashes = [
            (r'\bmd5\b', 'MD5 is cryptographically broken - use SHA-256 or better'),
            (r'\bsha1\b', 'SHA-1 is deprecated - use SHA-256 or better'),
            (r'\bDES\b', 'DES encryption is insecure - use AES'),
        ]
        
        for pattern, message in insecure_hashes:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    'type': 'insecure_crypto',
                    'severity': 'high',
                    'message': message,
                    'file': path,
                    'line': 0,
                    'column': 0
                })
        
        # Check for weak random number generation
        weak_random_patterns = [
            r'Math\.random\(',
            r'random\.random\(',
            r'rand\(\)',
        ]
        
        for pattern in weak_random_patterns:
            if re.search(pattern, content):
                issues.append({
                    'type': 'insecure_crypto',
                    'severity': 'medium',
                    'message': 'Weak random number generator - use cryptographically secure random',
                    'file': path,
                    'line': 0,
                    'column': 0
                })
                break
        
        return issues
    
    def _check_path_traversal(self, content: str, path: str) -> List[Dict[str, Any]]:
        """Check for path traversal vulnerabilities"""
        issues = []
        
        # Path traversal patterns
        traversal_patterns = [
            r'open\s*\([^)]*\+',
            r'readFile\s*\([^)]*\+',
            r'File\s*\([^)]*\+',
            r'\.\./',
        ]
        
        for pattern in traversal_patterns:
            if re.search(pattern, content):
                issues.append({
                    'type': 'path_traversal',
                    'severity': 'high',
                    'message': 'Potential path traversal vulnerability - validate file paths',
                    'file': path,
                    'line': 0,
                    'column': 0
                })
                break
        
        return issues
    
    def _check_command_injection(self, content: str, path: str) -> List[Dict[str, Any]]:
        """Check for command injection vulnerabilities"""
        issues = []
        
        # Command injection patterns
        command_patterns = [
            (r'os\.system\s*\(', 'os.system() with user input can lead to command injection'),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', 'subprocess with shell=True is dangerous'),
            (r'exec\s*\(', 'exec() can execute arbitrary code'),
            (r'eval\s*\(', 'eval() can execute arbitrary code'),
            (r'child_process\.exec\s*\(', 'child_process.exec() can lead to command injection'),
        ]
        
        for pattern, message in command_patterns:
            if re.search(pattern, content):
                issues.append({
                    'type': 'command_injection',
                    'severity': 'critical',
                    'message': message,
                    'file': path,
                    'line': 0,
                    'column': 0
                })
        
        return issues


class AIReviewEngine:
    """AI-powered review engine using ensemble reasoning"""
    
    def __init__(self):
        self.ensemble = None  # Will be initialized with ensemble AI
        self._initialized = False
        
        # Pattern matching rules for common code issues
        self.patterns = {
            'complexity': [
                r'for\s+\w+\s+in.*:\s*for\s+\w+\s+in',  # Nested loops
                r'if\s+.*:\s*if\s+.*:\s*if',  # Deep nesting
            ],
            'naming': [
                r'\b[a-z]\b',  # Single letter variables
                r'\b[A-Z]{2,}\b',  # All caps (except constants)
            ],
            'documentation': [
                r'def\s+\w+\([^)]*\):\s*\n\s*[^"\']',  # Function without docstring
                r'class\s+\w+.*:\s*\n\s*[^"\']',  # Class without docstring
            ]
        }
    
    async def initialize(self):
        """Initialize the ensemble reasoning system"""
        if not self._initialized:
            try:
                from xencode.ai_ensembles import create_ensemble_reasoner
                self.ensemble = await create_ensemble_reasoner()
                self._initialized = True
            except Exception as e:
                # Fallback to basic review if ensemble not available
                self.ensemble = None
                self._initialized = True
    
    async def generate_review(self, title: str, description: str,
                             files: List[Dict[str, Any]],
                             code_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an AI-powered code review using ensemble reasoning"""
        # Ensure ensemble is initialized
        if not self._initialized:
            await self.initialize()
        
        review = {
            'summary': {
                'title': title,
                'description': description,
                'files_analyzed': len(files)
            },
            'issues': [],
            'suggestions': [],
            'positive_feedback': [],
            'patterns_detected': [],
            'semantic_analysis': {}
        }
        
        # Analyze code analysis results
        for issue in code_analysis.get('issues', []):
            review['issues'].append({
                'type': issue.get('type', 'unknown'),
                'severity': issue.get('severity', 'medium'),
                'message': issue.get('message', ''),
                'file': issue.get('file', ''),
                'line': issue.get('line', 0),
                'column': issue.get('column', 0)
            })
        
        # Perform pattern matching on files
        patterns_detected = await self._detect_patterns(files)
        review['patterns_detected'] = patterns_detected
        
        # Perform semantic analysis using ensemble AI
        if self.ensemble and len(files) > 0:
            semantic_analysis = await self._semantic_analysis(title, description, files, code_analysis)
            review['semantic_analysis'] = semantic_analysis
        
        # Generate AI-powered suggestions for each issue
        for issue in review['issues']:
            suggestion = await self._generate_ai_suggestion(issue, files)
            if suggestion:
                review['suggestions'].append(suggestion)
        
        # Generate suggestions for detected patterns
        for pattern in patterns_detected:
            pattern_suggestion = await self._generate_pattern_suggestion(pattern)
            if pattern_suggestion:
                review['suggestions'].append(pattern_suggestion)
        
        # Generate positive feedback
        positive = self._generate_positive_feedback(files, code_analysis, patterns_detected)
        if positive:
            review['positive_feedback'].extend(positive)
        
        # Generate overall review summary using AI
        if self.ensemble:
            overall_summary = await self._generate_overall_summary(review)
            review['summary']['ai_summary'] = overall_summary
        
        return review
    
    async def _detect_patterns(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect code patterns using regex matching"""
        detected = []
        
        for file_data in files:
            content = file_data.get('content', '')
            file_path = file_data.get('path', file_data.get('filename', ''))
            
            # Check complexity patterns
            for pattern in self.patterns['complexity']:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    detected.append({
                        'type': 'complexity',
                        'pattern': 'nested_structure',
                        'file': file_path,
                        'message': 'High complexity detected - consider refactoring',
                        'severity': 'medium'
                    })
            
            # Check naming patterns
            for pattern in self.patterns['naming']:
                matches = re.finditer(pattern, content)
                for match in matches:
                    detected.append({
                        'type': 'naming',
                        'pattern': 'poor_naming',
                        'file': file_path,
                        'message': 'Consider using more descriptive variable names',
                        'severity': 'low'
                    })
            
            # Check documentation patterns
            for pattern in self.patterns['documentation']:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    detected.append({
                        'type': 'documentation',
                        'pattern': 'missing_docstring',
                        'file': file_path,
                        'message': 'Missing documentation - add docstrings',
                        'severity': 'low'
                    })
        
        return detected
    
    async def _semantic_analysis(self, title: str, description: str,
                                 files: List[Dict[str, Any]],
                                 code_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic analysis using ensemble AI"""
        if not self.ensemble:
            return {}
        
        try:
            # Prepare context for AI analysis
            file_summaries = []
            for file_data in files[:5]:  # Limit to first 5 files to avoid token limits
                file_path = file_data.get('path', file_data.get('filename', ''))
                content = file_data.get('content', '')[:1000]  # First 1000 chars
                file_summaries.append(f"File: {file_path}\n{content}...")
            
            context = f"""
Title: {title}
Description: {description}

Files Changed:
{chr(10).join(file_summaries)}

Issues Found: {len(code_analysis.get('issues', []))}
"""
            
            # Query ensemble for semantic analysis
            from xencode.ai_ensembles import QueryRequest, EnsembleMethod
            
            query = QueryRequest(
                prompt=f"""Analyze this code change and provide semantic insights:

{context}

Provide:
1. Overall code quality assessment
2. Potential architectural concerns
3. Maintainability considerations
4. Performance implications

Keep response concise and actionable.""",
                models=["llama3.1:8b", "mistral:7b"],
                method=EnsembleMethod.VOTE,
                temperature=0.3
            )
            
            response = await self.ensemble.reason(query)
            
            return {
                'analysis': response.fused_response,
                'confidence': response.confidence,
                'consensus_score': response.consensus_score
            }
        
        except Exception as e:
            return {'error': f'Semantic analysis failed: {str(e)}'}
    
    async def _generate_ai_suggestion(self, issue: Dict[str, Any],
                                     files: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate AI-powered fix suggestion for an issue"""
        issue_type = issue.get('type', '')
        severity = issue.get('severity', 'medium')
        file_path = issue.get('file', '')
        
        # Static suggestions for known issue types
        static_suggestions = {
            'sqli': {
                'title': 'SQL Injection Prevention',
                'description': 'Use parameterized queries instead of string concatenation',
                'example': '''# Bad
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# Good
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))'''
            },
            'xss': {
                'title': 'XSS Prevention',
                'description': 'Sanitize user input and use proper escaping',
                'example': '''# Bad
element.innerHTML = user_input

# Good
element.textContent = user_input
# Or use a sanitization library'''
            },
            'csrf': {
                'title': 'CSRF Protection',
                'description': 'Add CSRF tokens to forms and validate on server',
                'example': '''<!-- Add CSRF token to form -->
<input type="hidden" name="csrf_token" value="{{ csrf_token() }}">

# Validate on server
if not validate_csrf_token(request.form['csrf_token']):
    abort(403)'''
            },
            'hardcoded_secrets': {
                'title': 'Remove Hardcoded Secrets',
                'description': 'Use environment variables or secret management',
                'example': '''# Bad
API_KEY = "sk-1234567890abcdef"

# Good
import os
API_KEY = os.environ.get('API_KEY')'''
            },
            'insecure_crypto': {
                'title': 'Use Secure Cryptography',
                'description': 'Replace weak algorithms with secure alternatives',
                'example': '''# Bad
import md5
hash = md5.new(data).hexdigest()

# Good
import hashlib
hash = hashlib.sha256(data.encode()).hexdigest()'''
            },
            'command_injection': {
                'title': 'Prevent Command Injection',
                'description': 'Avoid shell=True and validate inputs',
                'example': '''# Bad
subprocess.call(f"ls {user_input}", shell=True)

# Good
subprocess.call(["ls", user_input])'''
            }
        }
        
        if issue_type in static_suggestions:
            suggestion = static_suggestions[issue_type].copy()
            suggestion['severity'] = severity
            suggestion['file'] = file_path
            suggestion['line'] = issue.get('line', 0)
            return suggestion
        
        # For other issues, try to generate AI suggestion if ensemble available
        if self.ensemble:
            try:
                from xencode.ai_ensembles import QueryRequest, EnsembleMethod
                
                query = QueryRequest(
                    prompt=f"""Provide a fix suggestion for this code issue:

Issue Type: {issue_type}
Severity: {severity}
Message: {issue.get('message', '')}
File: {file_path}

Provide:
1. A clear title
2. Brief description of the fix
3. Code example showing before/after

Keep response concise.""",
                    models=["llama3.1:8b"],
                    method=EnsembleMethod.VOTE,
                    temperature=0.3
                )
                
                response = await self.ensemble.reason(query)
                
                return {
                    'title': f'Fix {issue_type}',
                    'description': response.fused_response,
                    'severity': severity,
                    'file': file_path,
                    'line': issue.get('line', 0),
                    'ai_generated': True
                }
            except Exception:
                pass
        
        # Fallback to generic suggestion
        return {
            'title': f'Address {issue_type}',
            'description': f'Review and fix the {issue_type} issue',
            'severity': severity,
            'file': file_path,
            'line': issue.get('line', 0)
        }
    
    async def _generate_pattern_suggestion(self, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate suggestion for detected pattern"""
        pattern_type = pattern.get('type', '')
        pattern_name = pattern.get('pattern', '')
        
        suggestions = {
            'complexity': {
                'title': 'Reduce Code Complexity',
                'description': 'Consider extracting nested logic into separate functions',
                'example': '''# Before
for item in items:
    for sub in item.subs:
        if sub.valid:
            process(sub)

# After
def process_valid_subs(item):
    for sub in item.subs:
        if sub.valid:
            process(sub)

for item in items:
    process_valid_subs(item)'''
            },
            'naming': {
                'title': 'Improve Variable Naming',
                'description': 'Use descriptive names that convey purpose',
                'example': '''# Bad
x = 10
a = calculate(x)

# Good
max_retries = 10
result = calculate(max_retries)'''
            },
            'documentation': {
                'title': 'Add Documentation',
                'description': 'Add docstrings to functions and classes',
                'example': '''def calculate_total(items):
    """Calculate the total price of items.
    
    Args:
        items: List of items with price attribute
        
    Returns:
        float: Total price of all items
    """
    return sum(item.price for item in items)'''
            }
        }
        
        if pattern_type in suggestions:
            suggestion = suggestions[pattern_type].copy()
            suggestion['file'] = pattern.get('file', '')
            suggestion['severity'] = pattern.get('severity', 'low')
            return suggestion
        
        return None
    
    async def _generate_overall_summary(self, review: Dict[str, Any]) -> str:
        """Generate overall review summary using AI"""
        if not self.ensemble:
            return "Review completed"
        
        try:
            from xencode.ai_ensembles import QueryRequest, EnsembleMethod
            
            total_issues = len(review['issues'])
            critical_issues = sum(1 for i in review['issues'] if i.get('severity') == 'critical')
            high_issues = sum(1 for i in review['issues'] if i.get('severity') == 'high')
            
            query = QueryRequest(
                prompt=f"""Summarize this code review in 2-3 sentences:

Total Issues: {total_issues}
Critical: {critical_issues}
High: {high_issues}
Patterns Detected: {len(review['patterns_detected'])}

Provide a concise, actionable summary for the developer.""",
                models=["llama3.1:8b"],
                method=EnsembleMethod.VOTE,
                temperature=0.3
            )
            
            response = await self.ensemble.reason(query)
            return response.fused_response
        
        except Exception:
            return "Review completed with issues detected"
    
    def _generate_positive_feedback(self, files: List[Dict[str, Any]],
                                   code_analysis: Dict[str, Any],
                                   patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate positive feedback based on code analysis"""
        feedback = []
        
        total_issues = code_analysis.get('summary', {}).get('total_issues', 0)
        critical_issues = code_analysis.get('summary', {}).get('by_severity', {}).get('critical', 0)
        
        if total_issues == 0 and len(patterns) == 0:
            feedback.append({
                'title': 'Excellent Code Quality',
                'message': 'No issues detected! Your code follows best practices.',
                'score': 100
            })
        elif critical_issues == 0:
            feedback.append({
                'title': 'Good Security Posture',
                'message': 'No critical security issues found.',
                'score': 85
            })
        
        # Check for good practices
        has_tests = any('test' in f.get('path', '').lower() for f in files)
        if has_tests:
            feedback.append({
                'title': 'Test Coverage',
                'message': 'Great job including tests with your changes!',
                'score': 90
            })
        
        return feedback



class ReportGenerator:
    """Generate formatted review reports with severity categorization"""
    
    def __init__(self):
        """Initialize the report generator"""
        self.severity_order = ['critical', 'high', 'medium', 'low']
        self.severity_colors = {
            'critical': '\033[91m',  # Red
            'high': '\033[93m',      # Yellow
            'medium': '\033[94m',    # Blue
            'low': '\033[92m',       # Green
            'reset': '\033[0m'       # Reset
        }
    
    def generate_text_report(self, review: Dict[str, Any], 
                            pr_data: Dict[str, Any] = None) -> str:
        """Generate a text-based review report
        
        Args:
            review: Review data from AIReviewEngine
            pr_data: Optional PR metadata
            
        Returns:
            Formatted text report
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("CODE REVIEW REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # PR Information
        if pr_data:
            lines.append("Pull Request Information:")
            lines.append(f"  Title: {pr_data.get('title', 'N/A')}")
            lines.append(f"  URL: {pr_data.get('url', 'N/A')}")
            lines.append(f"  Author: {pr_data.get('author', 'N/A')}")
            lines.append(f"  Branch: {pr_data.get('head_branch', 'N/A')} → {pr_data.get('base_branch', 'N/A')}")
            lines.append("")
        
        # Summary
        summary = review.get('summary', {})
        lines.append("Summary:")
        lines.append(f"  Files Analyzed: {summary.get('files_analyzed', 0)}")
        
        if 'ai_summary' in summary:
            lines.append(f"  AI Assessment: {summary['ai_summary']}")
        
        lines.append("")
        
        # Issues by Severity
        issues = review.get('issues', [])
        if issues:
            lines.append("Issues Found:")
            lines.append("")
            
            # Group issues by severity
            issues_by_severity = self._group_by_severity(issues)
            
            for severity in self.severity_order:
                if severity in issues_by_severity:
                    severity_issues = issues_by_severity[severity]
                    lines.append(f"  {severity.upper()} ({len(severity_issues)} issue(s)):")
                    lines.append("  " + "-" * 76)
                    
                    for issue in severity_issues:
                        lines.append(f"    [{issue.get('type', 'unknown').upper()}] {issue.get('message', '')}")
                        lines.append(f"    Location: {issue.get('file', 'N/A')}:{issue.get('line', 0)}")
                        lines.append("")
            
            lines.append("")
        else:
            lines.append("✓ No issues found!")
            lines.append("")
        
        # Suggestions
        suggestions = review.get('suggestions', [])
        if suggestions:
            lines.append("Suggestions:")
            lines.append("")
            
            # Group suggestions by severity
            suggestions_by_severity = self._group_by_severity(suggestions)
            
            for severity in self.severity_order:
                if severity in suggestions_by_severity:
                    severity_suggestions = suggestions_by_severity[severity]
                    
                    for suggestion in severity_suggestions:
                        lines.append(f"  [{severity.upper()}] {suggestion.get('title', 'Suggestion')}")
                        lines.append(f"  {suggestion.get('description', '')}")
                        
                        if 'file' in suggestion:
                            lines.append(f"  Location: {suggestion.get('file')}:{suggestion.get('line', 0)}")
                        
                        if 'example' in suggestion:
                            lines.append("  Example:")
                            for line in suggestion['example'].split('\n'):
                                lines.append(f"    {line}")
                        
                        lines.append("")
            
            lines.append("")
        
        # Patterns Detected
        patterns = review.get('patterns_detected', [])
        if patterns:
            lines.append("Patterns Detected:")
            lines.append("")
            
            for pattern in patterns:
                lines.append(f"  • {pattern.get('message', 'Pattern detected')}")
                lines.append(f"    Type: {pattern.get('type', 'unknown')}")
                lines.append(f"    File: {pattern.get('file', 'N/A')}")
                lines.append("")
            
            lines.append("")
        
        # Semantic Analysis
        semantic = review.get('semantic_analysis', {})
        if semantic and 'analysis' in semantic:
            lines.append("Semantic Analysis:")
            lines.append(f"  {semantic['analysis']}")
            
            if 'confidence' in semantic:
                lines.append(f"  Confidence: {semantic['confidence']:.2%}")
            
            if 'consensus_score' in semantic:
                lines.append(f"  Consensus Score: {semantic['consensus_score']:.2%}")
            
            lines.append("")
        
        # Positive Feedback
        positive = review.get('positive_feedback', [])
        if positive:
            lines.append("Positive Feedback:")
            lines.append("")
            
            for feedback in positive:
                lines.append(f"  ✓ {feedback.get('title', 'Good work!')}")
                lines.append(f"    {feedback.get('message', '')}")
                
                if 'score' in feedback:
                    lines.append(f"    Score: {feedback['score']}/100")
                
                lines.append("")
        
        # Footer
        lines.append("=" * 80)
        
        return '\n'.join(lines)
    
    def generate_markdown_report(self, review: Dict[str, Any],
                                 pr_data: Dict[str, Any] = None) -> str:
        """Generate a Markdown-formatted review report
        
        Args:
            review: Review data from AIReviewEngine
            pr_data: Optional PR metadata
            
        Returns:
            Formatted Markdown report
        """
        lines = []
        
        # Header
        lines.append("# Code Review Report")
        lines.append("")
        
        # PR Information
        if pr_data:
            lines.append("## Pull Request Information")
            lines.append("")
            lines.append(f"- **Title:** {pr_data.get('title', 'N/A')}")
            lines.append(f"- **URL:** {pr_data.get('url', 'N/A')}")
            lines.append(f"- **Author:** {pr_data.get('author', 'N/A')}")
            lines.append(f"- **Branch:** `{pr_data.get('head_branch', 'N/A')}` → `{pr_data.get('base_branch', 'N/A')}`")
            lines.append("")
        
        # Summary
        summary = review.get('summary', {})
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Files Analyzed:** {summary.get('files_analyzed', 0)}")
        
        if 'ai_summary' in summary:
            lines.append(f"- **AI Assessment:** {summary['ai_summary']}")
        
        lines.append("")
        
        # Issues by Severity
        issues = review.get('issues', [])
        if issues:
            lines.append("## Issues Found")
            lines.append("")
            
            # Group issues by severity
            issues_by_severity = self._group_by_severity(issues)
            
            for severity in self.severity_order:
                if severity in issues_by_severity:
                    severity_issues = issues_by_severity[severity]
                    emoji = self._get_severity_emoji(severity)
                    lines.append(f"### {emoji} {severity.upper()} ({len(severity_issues)} issue(s))")
                    lines.append("")
                    
                    for issue in severity_issues:
                        lines.append(f"#### {issue.get('type', 'unknown').upper()}")
                        lines.append("")
                        lines.append(f"**Message:** {issue.get('message', '')}")
                        lines.append("")
                        lines.append(f"**Location:** `{issue.get('file', 'N/A')}:{issue.get('line', 0)}`")
                        lines.append("")
        else:
            lines.append("## ✅ No Issues Found!")
            lines.append("")
        
        # Suggestions
        suggestions = review.get('suggestions', [])
        if suggestions:
            lines.append("## Suggestions")
            lines.append("")
            
            # Group suggestions by severity
            suggestions_by_severity = self._group_by_severity(suggestions)
            
            for severity in self.severity_order:
                if severity in suggestions_by_severity:
                    severity_suggestions = suggestions_by_severity[severity]
                    emoji = self._get_severity_emoji(severity)
                    lines.append(f"### {emoji} {severity.upper()}")
                    lines.append("")
                    
                    for suggestion in severity_suggestions:
                        lines.append(f"#### {suggestion.get('title', 'Suggestion')}")
                        lines.append("")
                        lines.append(suggestion.get('description', ''))
                        lines.append("")
                        
                        if 'file' in suggestion:
                            lines.append(f"**Location:** `{suggestion.get('file')}:{suggestion.get('line', 0)}`")
                            lines.append("")
                        
                        if 'example' in suggestion:
                            lines.append("**Example:**")
                            lines.append("")
                            lines.append("```")
                            lines.append(suggestion['example'])
                            lines.append("```")
                            lines.append("")
        
        # Patterns Detected
        patterns = review.get('patterns_detected', [])
        if patterns:
            lines.append("## Patterns Detected")
            lines.append("")
            
            for pattern in patterns:
                lines.append(f"- **{pattern.get('message', 'Pattern detected')}**")
                lines.append(f"  - Type: `{pattern.get('type', 'unknown')}`")
                lines.append(f"  - File: `{pattern.get('file', 'N/A')}`")
                lines.append("")
        
        # Semantic Analysis
        semantic = review.get('semantic_analysis', {})
        if semantic and 'analysis' in semantic:
            lines.append("## Semantic Analysis")
            lines.append("")
            lines.append(semantic['analysis'])
            lines.append("")
            
            if 'confidence' in semantic or 'consensus_score' in semantic:
                lines.append("**Metrics:**")
                lines.append("")
                
                if 'confidence' in semantic:
                    lines.append(f"- Confidence: {semantic['confidence']:.2%}")
                
                if 'consensus_score' in semantic:
                    lines.append(f"- Consensus Score: {semantic['consensus_score']:.2%}")
                
                lines.append("")
        
        # Positive Feedback
        positive = review.get('positive_feedback', [])
        if positive:
            lines.append("## ✨ Positive Feedback")
            lines.append("")
            
            for feedback in positive:
                lines.append(f"### ✅ {feedback.get('title', 'Good work!')}")
                lines.append("")
                lines.append(feedback.get('message', ''))
                lines.append("")
                
                if 'score' in feedback:
                    lines.append(f"**Score:** {feedback['score']}/100")
                    lines.append("")
        
        return '\n'.join(lines)
    
    def generate_json_report(self, review: Dict[str, Any],
                            pr_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a JSON-formatted review report
        
        Args:
            review: Review data from AIReviewEngine
            pr_data: Optional PR metadata
            
        Returns:
            Structured JSON report
        """
        import json
        from datetime import datetime, timezone
        
        # Group issues and suggestions by severity
        issues_by_severity = self._group_by_severity(review.get('issues', []))
        suggestions_by_severity = self._group_by_severity(review.get('suggestions', []))
        
        # Calculate statistics
        total_issues = len(review.get('issues', []))
        severity_counts = {
            severity: len(issues_by_severity.get(severity, []))
            for severity in self.severity_order
        }
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(severity_counts)
        
        report = {
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'report_version': '1.0'
            },
            'pr_info': pr_data or {},
            'summary': {
                **review.get('summary', {}),
                'total_issues': total_issues,
                'severity_counts': severity_counts,
                'quality_score': quality_score
            },
            'issues_by_severity': issues_by_severity,
            'suggestions_by_severity': suggestions_by_severity,
            'patterns_detected': review.get('patterns_detected', []),
            'semantic_analysis': review.get('semantic_analysis', {}),
            'positive_feedback': review.get('positive_feedback', [])
        }
        
        return report
    
    def generate_html_report(self, review: Dict[str, Any],
                            pr_data: Dict[str, Any] = None) -> str:
        """Generate an HTML-formatted review report
        
        Args:
            review: Review data from AIReviewEngine
            pr_data: Optional PR metadata
            
        Returns:
            Formatted HTML report
        """
        lines = []
        
        # HTML Header
        lines.append("<!DOCTYPE html>")
        lines.append("<html>")
        lines.append("<head>")
        lines.append("  <meta charset='UTF-8'>")
        lines.append("  <title>Code Review Report</title>")
        lines.append("  <style>")
        lines.append("    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }")
        lines.append("    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
        lines.append("    h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }")
        lines.append("    h2 { color: #555; margin-top: 30px; }")
        lines.append("    h3 { color: #666; }")
        lines.append("    .severity-critical { background: #ffebee; border-left: 4px solid #f44336; padding: 15px; margin: 10px 0; }")
        lines.append("    .severity-high { background: #fff3e0; border-left: 4px solid #ff9800; padding: 15px; margin: 10px 0; }")
        lines.append("    .severity-medium { background: #e3f2fd; border-left: 4px solid #2196F3; padding: 15px; margin: 10px 0; }")
        lines.append("    .severity-low { background: #e8f5e9; border-left: 4px solid #4CAF50; padding: 15px; margin: 10px 0; }")
        lines.append("    .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }")
        lines.append("    .badge-critical { background: #f44336; color: white; }")
        lines.append("    .badge-high { background: #ff9800; color: white; }")
        lines.append("    .badge-medium { background: #2196F3; color: white; }")
        lines.append("    .badge-low { background: #4CAF50; color: white; }")
        lines.append("    .info-box { background: #f9f9f9; padding: 15px; border-radius: 4px; margin: 15px 0; }")
        lines.append("    .positive { background: #e8f5e9; border-left: 4px solid #4CAF50; padding: 15px; margin: 10px 0; }")
        lines.append("    code { background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }")
        lines.append("    pre { background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; }")
        lines.append("  </style>")
        lines.append("</head>")
        lines.append("<body>")
        lines.append("  <div class='container'>")
        
        # Title
        lines.append("    <h1>📋 Code Review Report</h1>")
        
        # PR Information
        if pr_data:
            lines.append("    <div class='info-box'>")
            lines.append("      <h2>Pull Request Information</h2>")
            lines.append(f"      <p><strong>Title:</strong> {self._html_escape(pr_data.get('title', 'N/A'))}</p>")
            lines.append(f"      <p><strong>URL:</strong> <a href='{pr_data.get('url', '#')}'>{pr_data.get('url', 'N/A')}</a></p>")
            lines.append(f"      <p><strong>Author:</strong> {self._html_escape(pr_data.get('author', 'N/A'))}</p>")
            lines.append(f"      <p><strong>Branch:</strong> <code>{self._html_escape(pr_data.get('head_branch', 'N/A'))}</code> → <code>{self._html_escape(pr_data.get('base_branch', 'N/A'))}</code></p>")
            lines.append("    </div>")
        
        # Summary
        summary = review.get('summary', {})
        lines.append("    <div class='info-box'>")
        lines.append("      <h2>Summary</h2>")
        lines.append(f"      <p><strong>Files Analyzed:</strong> {summary.get('files_analyzed', 0)}</p>")
        
        if 'ai_summary' in summary:
            lines.append(f"      <p><strong>AI Assessment:</strong> {self._html_escape(summary['ai_summary'])}</p>")
        
        lines.append("    </div>")
        
        # Issues by Severity
        issues = review.get('issues', [])
        if issues:
            lines.append("    <h2>🔍 Issues Found</h2>")
            
            # Group issues by severity
            issues_by_severity = self._group_by_severity(issues)
            
            for severity in self.severity_order:
                if severity in issues_by_severity:
                    severity_issues = issues_by_severity[severity]
                    lines.append(f"    <h3><span class='badge badge-{severity}'>{severity.upper()}</span> ({len(severity_issues)} issue(s))</h3>")
                    
                    for issue in severity_issues:
                        lines.append(f"    <div class='severity-{severity}'>")
                        lines.append(f"      <h4>{self._html_escape(issue.get('type', 'unknown').upper())}</h4>")
                        lines.append(f"      <p>{self._html_escape(issue.get('message', ''))}</p>")
                        lines.append(f"      <p><strong>Location:</strong> <code>{self._html_escape(issue.get('file', 'N/A'))}:{issue.get('line', 0)}</code></p>")
                        lines.append("    </div>")
        else:
            lines.append("    <div class='positive'>")
            lines.append("      <h2>✅ No Issues Found!</h2>")
            lines.append("    </div>")
        
        # Suggestions
        suggestions = review.get('suggestions', [])
        if suggestions:
            lines.append("    <h2>💡 Suggestions</h2>")
            
            # Group suggestions by severity
            suggestions_by_severity = self._group_by_severity(suggestions)
            
            for severity in self.severity_order:
                if severity in suggestions_by_severity:
                    severity_suggestions = suggestions_by_severity[severity]
                    
                    for suggestion in severity_suggestions:
                        lines.append(f"    <div class='severity-{severity}'>")
                        lines.append(f"      <h4><span class='badge badge-{severity}'>{severity.upper()}</span> {self._html_escape(suggestion.get('title', 'Suggestion'))}</h4>")
                        lines.append(f"      <p>{self._html_escape(suggestion.get('description', ''))}</p>")
                        
                        if 'file' in suggestion:
                            lines.append(f"      <p><strong>Location:</strong> <code>{self._html_escape(suggestion.get('file'))}:{suggestion.get('line', 0)}</code></p>")
                        
                        if 'example' in suggestion:
                            lines.append("      <p><strong>Example:</strong></p>")
                            lines.append(f"      <pre><code>{self._html_escape(suggestion['example'])}</code></pre>")
                        
                        lines.append("    </div>")
        
        # Patterns Detected
        patterns = review.get('patterns_detected', [])
        if patterns:
            lines.append("    <h2>🔎 Patterns Detected</h2>")
            
            for pattern in patterns:
                lines.append("    <div class='info-box'>")
                lines.append(f"      <p><strong>{self._html_escape(pattern.get('message', 'Pattern detected'))}</strong></p>")
                lines.append(f"      <p>Type: <code>{self._html_escape(pattern.get('type', 'unknown'))}</code></p>")
                lines.append(f"      <p>File: <code>{self._html_escape(pattern.get('file', 'N/A'))}</code></p>")
                lines.append("    </div>")
        
        # Semantic Analysis
        semantic = review.get('semantic_analysis', {})
        if semantic and 'analysis' in semantic:
            lines.append("    <h2>🧠 Semantic Analysis</h2>")
            lines.append("    <div class='info-box'>")
            lines.append(f"      <p>{self._html_escape(semantic['analysis'])}</p>")
            
            if 'confidence' in semantic:
                lines.append(f"      <p><strong>Confidence:</strong> {semantic['confidence']:.2%}</p>")
            
            if 'consensus_score' in semantic:
                lines.append(f"      <p><strong>Consensus Score:</strong> {semantic['consensus_score']:.2%}</p>")
            
            lines.append("    </div>")
        
        # Positive Feedback
        positive = review.get('positive_feedback', [])
        if positive:
            lines.append("    <h2>✨ Positive Feedback</h2>")
            
            for feedback in positive:
                lines.append("    <div class='positive'>")
                lines.append(f"      <h4>✅ {self._html_escape(feedback.get('title', 'Good work!'))}</h4>")
                lines.append(f"      <p>{self._html_escape(feedback.get('message', ''))}</p>")
                
                if 'score' in feedback:
                    lines.append(f"      <p><strong>Score:</strong> {feedback['score']}/100</p>")
                
                lines.append("    </div>")
        
        # HTML Footer
        lines.append("  </div>")
        lines.append("</body>")
        lines.append("</html>")
        
        return '\n'.join(lines)
    
    def _group_by_severity(self, items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group items by severity level"""
        grouped = {}
        
        for item in items:
            severity = item.get('severity', 'medium')
            if severity not in grouped:
                grouped[severity] = []
            grouped[severity].append(item)
        
        return grouped
    
    def _get_severity_emoji(self, severity: str) -> str:
        """Get emoji for severity level"""
        emojis = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        return emojis.get(severity, '⚪')
    
    def _calculate_quality_score(self, severity_counts: Dict[str, int]) -> int:
        """Calculate overall quality score based on severity counts"""
        score = 100
        score -= severity_counts.get('critical', 0) * 20
        score -= severity_counts.get('high', 0) * 10
        score -= severity_counts.get('medium', 0) * 5
        score -= severity_counts.get('low', 0) * 2
        return max(0, score)
    
    def _html_escape(self, text: str) -> str:
        """Escape HTML special characters"""
        if not isinstance(text, str):
            text = str(text)
        
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))
