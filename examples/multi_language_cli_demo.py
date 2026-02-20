#!/usr/bin/env python3
"""
Demo script for multi-language CLI commands.

This script demonstrates the multi-language support features:
- Language listing
- Language detection
- Language switching
- Text translation
- Technical term glossary

Run this script to see the multi-language features in action.
"""

import subprocess
import sys


def run_command(cmd):
    """Run a CLI command and display output."""
    print(f"\n{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr}")
    
    return result.returncode == 0


def main():
    """Run multi-language CLI demos."""
    print("🌍 Xencode Multi-Language Support Demo")
    print("=" * 60)
    
    # 1. List all supported languages
    print("\n1. Listing all supported languages...")
    run_command(['python', '-m', 'xencode.cli', 'lang', 'list'])
    
    # 2. List only RTL languages
    print("\n2. Listing RTL (Right-to-Left) languages...")
    run_command(['python', '-m', 'xencode.cli', 'lang', 'list', '--rtl-only'])
    
    # 3. Detect system language
    print("\n3. Detecting system language...")
    run_command(['python', '-m', 'xencode.cli', 'lang', 'detect'])
    
    # 4. Detect language from text
    print("\n4. Detecting language from text...")
    run_command(['python', '-m', 'xencode.cli', 'lang', 'detect', 'Bonjour le monde'])
    
    # 5. Set language to Spanish
    print("\n5. Setting language to Spanish...")
    run_command(['python', '-m', 'xencode.cli', 'lang', 'set', 'es'])
    
    # 6. Translate text
    print("\n6. Translating text from English to Spanish...")
    run_command(['python', '-m', 'xencode.cli', 'lang', 'translate', 
                 'Hello world', '--to', 'es'])
    
    # 7. Show technical term glossary
    print("\n7. Showing technical term glossary...")
    run_command(['python', '-m', 'xencode.cli', 'lang', 'glossary'])
    
    # 8. Search glossary
    print("\n8. Searching glossary for 'function'...")
    run_command(['python', '-m', 'xencode.cli', 'lang', 'glossary', 
                 '--search', 'function'])
    
    # 9. Show glossary for specific language
    print("\n9. Showing glossary for Spanish...")
    run_command(['python', '-m', 'xencode.cli', 'lang', 'glossary', 
                 '--language', 'es'])
    
    # 10. Reset to English
    print("\n10. Resetting language to English...")
    run_command(['python', '-m', 'xencode.cli', 'lang', 'set', 'en'])
    
    print("\n" + "="*60)
    print("✅ Demo complete!")
    print("="*60)
    print("\nTry these commands yourself:")
    print("  xencode lang list")
    print("  xencode lang set es")
    print("  xencode lang translate 'Hello' --to fr")
    print("  xencode lang glossary")


if __name__ == '__main__':
    main()
