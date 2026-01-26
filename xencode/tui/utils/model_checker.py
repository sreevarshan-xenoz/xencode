#!/usr/bin/env python3
"""
Model Checker Utility

Checks for available Ollama models and manages model availability.
"""

import subprocess
import shutil
from typing import List, Dict, Optional
import logging

try:
    import ollama
    OLLAMA_LIB_AVAILABLE = True
except ImportError:
    OLLAMA_LIB_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelChecker:
    """Checks for available models in the system"""
    
    @staticmethod
    def is_ollama_installed() -> bool:
        """Check if Ollama is installed and accessible"""
        return shutil.which("ollama") is not None
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available Ollama models"""
        models = []
        
        # Try using library first
        if OLLAMA_LIB_AVAILABLE:
            try:
                # ollama.list() returns a dict with 'models' key
                response = ollama.list()
                if isinstance(response, dict) and 'models' in response:
                    # Each model is a dict, extract the name field
                    models_list = []
                    for m in response['models']:
                        if isinstance(m, dict):
                            # Try different possible name fields
                            name = m.get('name') or m.get('model') or m.get('id')
                            if name:
                                models_list.append(name)
                    if models_list:
                        return models_list
            except Exception as e:
                logger.warning(f"Failed to list models via library: {e}")
        
        # Fallback to CLI
        if ModelChecker.is_ollama_installed():
            try:
                result = subprocess.run(
                    ["ollama", "list"], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                lines = result.stdout.strip().split('\n')
                # Skip header line (NAME ID SIZE MODIFIED)
                if len(lines) > 1:
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                if models:
                    return models
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to list models via CLI: {e}")

        # Fallback to REST API (localhost)
        try:
            import urllib.request
            import json
            with urllib.request.urlopen("http://localhost:11434/api/tags") as url:
                data = json.loads(url.read().decode())
                if 'models' in data:
                    return [m['name'] for m in data['models']]
        except Exception as e:
            logger.warning(f"Failed to list models via REST API: {e}")
                
        return models
    
    @staticmethod
    def check_model_availability(model_name: str) -> bool:
        """Check if a specific model is available"""
        # Handle tags (e.g., 'llama3:latest' matches 'llama3')
        available = ModelChecker.get_available_models()
        
        # Exact match
        if model_name in available:
            return True
            
        # Check without tag if input has no tag
        if ":" not in model_name:
            for m in available:
                if m.split(":")[0] == model_name:
                    return True
                    
        return False
    
    @staticmethod
    def pull_model(model_name: str) -> bool:
        """Pull a model (blocking) - use with caution or in thread"""
        if not ModelChecker.is_ollama_installed():
            return False
            
        try:
            subprocess.run(
                ["ollama", "pull", model_name],
                check=True,
                capture_output=False  # Let it show progress if running in terminal
            )
            return True
        except subprocess.CalledProcessError:
            return False
