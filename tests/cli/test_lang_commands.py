"""
Tests for multi-language CLI commands.
"""

import pytest
from click.testing import CliRunner
from xencode.cli import cli


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


def test_lang_list(runner):
    """Test lang list command."""
    result = runner.invoke(cli, ['lang', 'list'])
    assert result.exit_code == 0
    assert 'Supported languages' in result.output or 'Supported Languages' in result.output
    assert 'English' in result.output


def test_lang_list_enabled_only(runner):
    """Test lang list with --enabled-only flag."""
    result = runner.invoke(cli, ['lang', 'list', '--enabled-only'])
    assert result.exit_code == 0


def test_lang_list_rtl_only(runner):
    """Test lang list with --rtl-only flag."""
    result = runner.invoke(cli, ['lang', 'list', '--rtl-only'])
    assert result.exit_code == 0
    # Should show Arabic and Hebrew
    assert 'Arabic' in result.output or 'Hebrew' in result.output or 'العربية' in result.output


def test_lang_set_valid(runner):
    """Test setting a valid language."""
    result = runner.invoke(cli, ['lang', 'set', 'es'])
    assert result.exit_code == 0
    assert 'Language set to' in result.output or 'Español' in result.output


def test_lang_set_invalid(runner):
    """Test setting an invalid language."""
    result = runner.invoke(cli, ['lang', 'set', 'invalid'])
    assert result.exit_code == 0  # Command runs but shows error
    assert 'not supported' in result.output


def test_lang_detect_system(runner):
    """Test language detection from system."""
    result = runner.invoke(cli, ['lang', 'detect'])
    assert result.exit_code == 0
    assert 'language' in result.output.lower()


def test_lang_detect_text(runner):
    """Test language detection from text."""
    result = runner.invoke(cli, ['lang', 'detect', 'Bonjour le monde'])
    assert result.exit_code == 0
    assert 'Detected' in result.output or 'language' in result.output.lower()


def test_lang_translate(runner):
    """Test text translation."""
    result = runner.invoke(cli, ['lang', 'translate', 'Hello world', '--to', 'es'])
    assert result.exit_code == 0
    assert 'Translation' in result.output or 'Translating' in result.output


def test_lang_translate_with_source(runner):
    """Test text translation with source language."""
    result = runner.invoke(cli, ['lang', 'translate', 'Bonjour', '--to', 'en', '--from', 'fr'])
    assert result.exit_code == 0


def test_lang_glossary(runner):
    """Test glossary display."""
    result = runner.invoke(cli, ['lang', 'glossary'])
    assert result.exit_code == 0
    assert 'Technical' in result.output or 'Glossary' in result.output


def test_lang_glossary_search(runner):
    """Test glossary search."""
    result = runner.invoke(cli, ['lang', 'glossary', '--search', 'function'])
    assert result.exit_code == 0


def test_lang_glossary_language(runner):
    """Test glossary for specific language."""
    result = runner.invoke(cli, ['lang', 'glossary', '--language', 'es'])
    assert result.exit_code == 0


def test_lang_help(runner):
    """Test lang command help."""
    result = runner.invoke(cli, ['lang', '--help'])
    assert result.exit_code == 0
    assert 'Multi-language' in result.output
    assert 'set' in result.output
    assert 'list' in result.output
    assert 'detect' in result.output
    assert 'translate' in result.output
    assert 'glossary' in result.output
