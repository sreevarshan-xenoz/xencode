"""
Integration with Xencode Model Management System

This module provides integration between ByteBot components and Xencode's
existing model management infrastructure.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# Import from Xencode core
from ..core import ModelManager
from ..intelligent_model_selector import FirstRunSetup
from ..model_providers import get_model_provider_manager


class ModelIntegration:
    """
    Handles integration between ByteBot and Xencode's model management system
    """
    
    def __init__(self, model_manager: ModelManager = None):
        self.model_manager = model_manager or ModelManager()
        self.provider_manager = None
        self.current_model = self.model_manager.current_model
        self.model_capabilities = {}
        
    def initialize_model_system(self):
        """Initialize the model management system"""
        try:
            # Refresh available models
            self.model_manager.refresh_models()
            
            # Initialize provider manager if available
            try:
                self.provider_manager = get_model_provider_manager()
            except ImportError:
                self.provider_manager = None
                
            # Set current model if not already set
            if not self.current_model:
                self.current_model = self._get_best_available_model()
                
            # Load model capabilities
            self._load_model_capabilities()
            
        except Exception as e:
            print(f"Warning: Error initializing model system: {e}")
            # Fallback to basic functionality
            pass
    
    def _get_best_available_model(self) -> str:
        """Get the best available model based on system capabilities"""
        try:
            # Try to use the intelligent model selector
            from ..intelligent_model_selector import get_smart_default_model
            return get_smart_default_model() or "llama3.1:8b"  # fallback
        except ImportError:
            # Fallback to basic model selection
            available = self.model_manager.available_models
            if available:
                return available[0]
            else:
                return "llama3.1:8b"  # default fallback
    
    def _load_model_capabilities(self):
        """Load capabilities for each available model"""
        # For now, define basic capabilities based on model names
        for model in self.model_manager.available_models:
            if any(name in model.lower() for name in ['codellama', 'deepseek-coder', 'starcoder']):
                self.model_capabilities[model] = {
                    'strengths': ['coding', 'programming'],
                    'specializations': ['code_generation', 'code_explanation'],
                    'recommended_for': ['development_tasks', 'code_review']
                }
            elif any(name in model.lower() for name in ['llama', 'mistral', 'mixtral']):
                self.model_capabilities[model] = {
                    'strengths': ['general_purpose', 'reasoning'],
                    'specializations': ['conversation', 'analysis'],
                    'recommended_for': ['general_tasks', 'analysis']
                }
            else:
                self.model_capabilities[model] = {
                    'strengths': ['general_purpose'],
                    'specializations': ['basic_tasks'],
                    'recommended_for': ['simple_queries']
                }
    
    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """Get capabilities for a specific model"""
        return self.model_capabilities.get(model_name, {
            'strengths': ['general_purpose'],
            'specializations': ['basic_tasks'],
            'recommended_for': ['simple_queries']
        })
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        try:
            success, message = self.model_manager.switch_model(model_name)
            if success:
                self.current_model = model_name
                return True
            else:
                print(f"Failed to switch model: {message}")
                return False
        except Exception as e:
            print(f"Error switching model: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.model_manager.available_models
    
    def check_model_health(self, model_name: str) -> bool:
        """Check if a model is healthy and responsive"""
        try:
            return self.model_manager.check_model_health(model_name)
        except Exception:
            return False
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        health_info = self.model_manager.model_health.get(self.current_model, {})
        
        return {
            'name': self.current_model,
            'health_status': health_info.get('status', 'unknown'),
            'response_time': health_info.get('response_time', 0),
            'capabilities': self.get_model_capabilities(self.current_model),
            'last_check': health_info.get('last_check', 0)
        }
    
    def select_model_for_task(self, task_description: str, context: Dict[str, Any] = None) -> str:
        """
        Select the most appropriate model for a given task
        
        Args:
            task_description: Description of the task to be performed
            context: Context information that might influence model selection
            
        Returns:
            Name of the selected model
        """
        task_lower = task_description.lower()
        
        # Check if this is a coding task
        coding_indicators = [
            'code', 'program', 'function', 'script', 'develop', 'implement',
            'write', 'create', 'build', 'debug', 'fix', 'algorithm', 'logic'
        ]
        
        is_coding_task = any(indicator in task_lower for indicator in coding_indicators)
        
        # Check if this is an analytical task
        analysis_indicators = [
            'analyze', 'analysis', 'compare', 'evaluate', 'assess', 'review',
            'examine', 'study', 'research', 'investigate', 'summarize'
        ]
        
        is_analysis_task = any(indicator in task_lower for indicator in analysis_indicators)
        
        # Select model based on task type
        for model in self.model_manager.available_models:
            caps = self.get_model_capabilities(model)
            
            if is_coding_task and 'coding' in caps.get('strengths', []):
                return model
            elif is_analysis_task and 'reasoning' in caps.get('strengths', []):
                return model
        
        # If no specialized model found, return current model or first available
        if self.current_model and self.current_model in self.model_manager.available_models:
            return self.current_model
        elif self.model_manager.available_models:
            return self.model_manager.available_models[0]
        else:
            return "llama3.1:8b"  # fallback


class ByteBotModelManager:
    """
    Enhanced model manager specifically for ByteBot operations
    """
    
    def __init__(self):
        self.integration = ModelIntegration()
        self.task_specific_models = {}
        self.performance_tracking = {}
        
    def initialize(self):
        """Initialize the ByteBot model management system"""
        self.integration.initialize_model_system()
        
    def get_model_for_bytebot_task(self, intent: str, task_type: str = "general") -> str:
        """
        Get the most appropriate model for a ByteBot task
        
        Args:
            intent: User's intent or command
            task_type: Type of task (e.g., "command_generation", "analysis", "planning")
            
        Returns:
            Name of the most appropriate model
        """
        # First, try to select based on task description
        model = self.integration.select_model_for_task(intent)
        
        # For specific ByteBot tasks, we might want to use different models
        if task_type == "command_generation":
            # For command generation, we might prefer faster models
            fast_models = [m for m in self.integration.get_available_models() 
                          if any(fast_indicator in m.lower() for fast_indicator in 
                                ['phi', 'gemma', 'qwen', 'llama3.2'])]
            if fast_models:
                return fast_models[0]
        elif task_type == "complex_reasoning":
            # For complex reasoning, prefer larger models
            large_models = [m for m in self.integration.get_available_models() 
                           if any(large_indicator in m.lower() for large_indicator in 
                                 ['70b', '72b', '8x', 'large'])]
            if large_models:
                return large_models[0]
        
        return model
    
    def track_model_performance(self, model_name: str, task_type: str, success: bool, response_time: float):
        """Track performance of models for different task types"""
        if model_name not in self.performance_tracking:
            self.performance_tracking[model_name] = {}
        
        if task_type not in self.performance_tracking[model_name]:
            self.performance_tracking[model_name][task_type] = {
                'attempts': 0,
                'successes': 0,
                'total_response_time': 0.0,
                'avg_response_time': 0.0
            }
        
        perf = self.performance_tracking[model_name][task_type]
        perf['attempts'] += 1
        if success:
            perf['successes'] += 1
        perf['total_response_time'] += response_time
        perf['avg_response_time'] = perf['total_response_time'] / perf['attempts']
    
    def get_model_recommendation(self, intent: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get model recommendation with reasoning
        
        Args:
            intent: User's intent or command
            context: Context information
            
        Returns:
            Dictionary with model recommendation and reasoning
        """
        recommended_model = self.get_model_for_bytebot_task(intent)
        current_model_info = self.integration.get_current_model_info()
        
        # Determine if we should switch models
        should_switch = recommended_model != current_model_info['name']
        
        # Get reasoning for recommendation
        task_lower = intent.lower()
        if 'code' in task_lower or 'program' in task_lower:
            reasoning = f"'{recommended_model}' is recommended for coding tasks due to its strong programming capabilities"
        elif any(word in task_lower for word in ['analyze', 'compare', 'evaluate']):
            reasoning = f"'{recommended_model}' is recommended for analytical tasks"
        else:
            reasoning = f"'{recommended_model}' is the best general-purpose model for this task"
        
        return {
            'recommended_model': recommended_model,
            'current_model': current_model_info['name'],
            'should_switch': should_switch,
            'reasoning': reasoning,
            'model_capabilities': self.integration.get_model_capabilities(recommended_model)
        }
    
    def validate_model_availability(self, model_name: str) -> Dict[str, Any]:
        """
        Validate that a model is available and suitable for use
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Dictionary with validation results
        """
        available_models = self.integration.get_available_models()
        
        result = {
            'model_name': model_name,
            'is_available': False,
            'is_healthy': False,
            'needs_download': False,
            'validation_errors': []
        }
        
        if model_name not in available_models:
            result['is_available'] = False
            result['needs_download'] = True
            result['validation_errors'].append(f"Model '{model_name}' is not available locally")
        else:
            result['is_available'] = True
            
            # Check health
            is_healthy = self.integration.check_model_health(model_name)
            result['is_healthy'] = is_healthy
            
            if not is_healthy:
                result['validation_errors'].append(f"Model '{model_name}' is not responding properly")
        
        return result


# Example usage
if __name__ == "__main__":
    # Example of how to use the model integration
    model_mgr = ByteBotModelManager()
    model_mgr.initialize()
    
    # Example tasks
    tasks = [
        "Write a Python script to list all files in a directory",
        "Analyze the performance of my application",
        "Explain how neural networks work",
        "Create a bash script to backup my files"
    ]
    
    print("Model Recommendations for Sample Tasks:")
    print("=" * 50)
    
    for task in tasks:
        recommendation = model_mgr.get_model_recommendation(task)
        print(f"\nTask: {task}")
        print(f"Recommended Model: {recommendation['recommended_model']}")
        print(f"Current Model: {recommendation['current_model']}")
        print(f"Should Switch: {recommendation['should_switch']}")
        print(f"Reasoning: {recommendation['reasoning']}")
        print("-" * 30)