import unittest
from unittest.mock import MagicMock, patch
from xencode.shell_genie.genie import ShellGenie

class TestShellGenie(unittest.TestCase):
    @patch('xencode.shell_genie.genie.ChatOllama')
    def test_generate_command_success(self, MockChatOllama):
        # Setup mock
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"command": "echo test", "explanation": "Test command"}'
        mock_llm.invoke.return_value = mock_response
        MockChatOllama.return_value = mock_llm
        
        genie = ShellGenie()
        cmd, exp = genie.generate_command("test instruction")
        
        self.assertEqual(cmd, "echo test")
        self.assertEqual(exp, "Test command")
        
    @patch('xencode.shell_genie.genie.ChatOllama')
    def test_generate_command_parse_error(self, MockChatOllama):
        # Setup mock with bad JSON
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = 'This is not JSON'
        mock_llm.invoke.return_value = mock_response
        MockChatOllama.return_value = mock_llm
        
        genie = ShellGenie()
        cmd, exp = genie.generate_command("test")
        
        self.assertEqual(cmd, "")
        self.assertIn("Failed to parse", exp)

if __name__ == '__main__':
    unittest.main()
