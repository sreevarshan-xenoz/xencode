"""Web content extractor using requests and BeautifulSoup."""

from pathlib import Path
from typing import Dict, Any, Optional
import requests

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class WebExtractor:
    """Extract content from web pages."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def extract(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a URL.
        
        Args:
            url: The URL to extract content from.
            
        Returns:
            Dictionary containing extracted content and metadata.
        """
        if not BS4_AVAILABLE:
            return {"error": "beautifulsoup4 is not installed. Run: pip install beautifulsoup4"}

        result = {
            "url": url,
            "title": "",
            "text": "",
            "metadata": {},
            "error": None
        }

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            if soup.title:
                result["title"] = soup.title.string.strip()
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                result["metadata"]["description"] = meta_desc['content']
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text from main content
            main_content = soup.find('main') or soup.find('article') or soup.body
            if main_content:
                result["text"] = main_content.get_text(separator='\n', strip=True)
            
            result["metadata"]["status_code"] = response.status_code
            result["metadata"]["content_type"] = response.headers.get('content-type', '')
            
        except requests.RequestException as e:
            result["error"] = f"Request failed: {str(e)}"
        except Exception as e:
            result["error"] = str(e)

        return result
