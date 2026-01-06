import unittest
from unittest.mock import MagicMock, patch
from xencode.devops.generator import DevOpsGenerator

class TestDevOpsGenerator(unittest.TestCase):
    @patch('xencode.devops.generator.ChatOllama')
    def test_analyze_project_and_generate(self, MockChatOllama):
        # Setup mock
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = 'FROM python:3.9\nRUN pip install -r requirements.txt'
        mock_llm.invoke.return_value = mock_response
        MockChatOllama.return_value = mock_llm
        
        generator = DevOpsGenerator()
        
        # Test analysis (mocking file reads is harder here without fs mock, 
        # so we'll test generate directly with passing context)
        context = {"requirements.txt": "flask"}
        
        dockerfile = generator.generate_dockerfile(context)
        self.assertIn("FROM python:3.9", dockerfile)
        
        # Test Docker Compose generation
        mock_response.content = 'services:\n  web:\n    build: .'
        compose = generator.generate_docker_compose(context)
        self.assertIn("services:", compose)

if __name__ == '__main__':
    unittest.main()
