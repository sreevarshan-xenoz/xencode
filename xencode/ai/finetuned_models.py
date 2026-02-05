"""
Domain-Specific Fine-Tuned Model Manager
Implements FineTunedModelManager for specialized models, automatic domain detection,
model performance monitoring, and model versioning/update mechanisms.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import json
import os
import hashlib
from datetime import datetime
import aiohttp
import requests
from pathlib import Path


logger = logging.getLogger(__name__)


class ModelDomain(Enum):
    """Supported domains for fine-tuned models."""
    TECHNICAL_CODING = "technical_coding"
    SCIENTIFIC_RESEARCH = "scientific_research"
    FINANCIAL_ANALYSIS = "financial_analysis"
    LEGAL_DOCUMENTS = "legal_documents"
    MEDICAL_HEALTHCARE = "medical_healthcare"
    CREATIVE_WRITING = "creative_writing"
    BUSINESS_STRATEGY = "business_strategy"
    EDUCATIONAL_TRAINING = "educational_training"


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class ModelSpecification:
    """Specification for a fine-tuned model."""
    model_id: str
    domain: ModelDomain
    provider: ModelProvider
    version: str
    capabilities: List[str]
    performance_metrics: Dict[str, float]
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    created_at: datetime = datetime.now()
    is_active: bool = True


@dataclass
class ModelPerformanceReport:
    """Performance report for a model."""
    model_id: str
    timestamp: datetime
    accuracy: float
    response_time: float
    throughput: float
    error_rate: float
    resource_utilization: Dict[str, float]


class DomainDetector:
    """Detects the domain of input text to select appropriate models."""
    
    def __init__(self):
        self.domain_keywords = {
            ModelDomain.TECHNICAL_CODING: [
                'function', 'class', 'variable', 'algorithm', 'code', 'programming', 
                'debug', 'refactor', 'library', 'framework', 'api', 'database',
                'javascript', 'python', 'java', 'c++', 'html', 'css', 'sql'
            ],
            ModelDomain.SCIENTIFIC_RESEARCH: [
                'hypothesis', 'experiment', 'data', 'research', 'study', 'analysis',
                'statistical', 'methodology', 'peer-reviewed', 'literature', 'journal',
                'quantitative', 'qualitative', 'sample', 'control', 'variable'
            ],
            ModelDomain.FINANCIAL_ANALYSIS: [
                'revenue', 'profit', 'loss', 'investment', 'portfolio', 'market',
                'stock', 'bond', 'equity', 'debt', 'cashflow', 'valuation', 'risk',
                'return', 'yield', 'dividend', 'earnings', 'balance sheet'
            ],
            ModelDomain.LEGAL_DOCUMENTS: [
                'contract', 'agreement', 'clause', 'section', 'court', 'law',
                'regulation', 'compliance', 'liability', 'jurisdiction', 'plaintiff',
                'defendant', 'attorney', 'brief', 'motion', 'precedent', 'statute'
            ],
            ModelDomain.MEDICAL_HEALTHCARE: [
                'patient', 'diagnosis', 'treatment', 'symptom', 'medication', 'therapy',
                'clinical', 'trial', 'pharmaceutical', 'dosage', 'prescription',
                'hospital', 'doctor', 'nurse', 'procedure', 'condition', 'disease'
            ],
            ModelDomain.CREATIVE_WRITING: [
                'story', 'narrative', 'character', 'plot', 'theme', 'metaphor',
                'imagery', 'poetry', 'verse', 'stanza', 'rhyme', 'alliteration',
                'personification', 'symbolism', 'foreshadowing', 'climax', 'setting'
            ],
            ModelDomain.BUSINESS_STRATEGY: [
                'strategy', 'market', 'competition', 'competitive', 'advantage',
                'revenue', 'growth', 'expansion', 'acquisition', 'merger', 'partnership',
                'stakeholder', 'shareholder', 'board', 'executive', 'management'
            ],
            ModelDomain.EDUCATIONAL_TRAINING: [
                'course', 'lesson', 'curriculum', 'teaching', 'learning', 'student',
                'education', 'training', 'assessment', 'evaluation', 'syllabus',
                'assignment', 'homework', 'exam', 'grade', 'feedback', 'pedagogy'
            ]
        }
        
    def detect_domain(self, text: str) -> ModelDomain:
        """Detect the domain of the input text."""
        text_lower = text.lower()
        scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[domain] = score
            
        # Return the domain with the highest score, or default to technical coding
        return max(scores, key=scores.get) if scores else ModelDomain.TECHNICAL_CODING


class ModelVersionManager:
    """Manages versioning and updates for fine-tuned models."""
    
    def __init__(self, storage_path: str = "./model_versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.version_registry: Dict[str, List[ModelSpecification]] = {}
        
    def register_model_version(self, model_spec: ModelSpecification):
        """Register a new version of a model."""
        if model_spec.model_id not in self.version_registry:
            self.version_registry[model_spec.model_id] = []
            
        # Add the new version to the registry
        self.version_registry[model_spec.model_id].append(model_spec)
        
        # Sort versions by creation date (newest first)
        self.version_registry[model_spec.model_id].sort(
            key=lambda x: x.created_at, reverse=True
        )
        
    def get_latest_version(self, model_id: str) -> Optional[ModelSpecification]:
        """Get the latest version of a model."""
        if model_id not in self.version_registry or not self.version_registry[model_id]:
            return None
        return self.version_registry[model_id][0]
        
    def get_all_versions(self, model_id: str) -> List[ModelSpecification]:
        """Get all versions of a model."""
        return self.version_registry.get(model_id, [])
        
    def update_model_endpoint(self, model_id: str, new_endpoint: str, version: str):
        """Update the endpoint for a specific version of a model."""
        if model_id not in self.version_registry:
            return False
            
        for model_spec in self.version_registry[model_id]:
            if model_spec.version == version:
                model_spec.endpoint_url = new_endpoint
                return True
                
        return False


class ModelPerformanceMonitor:
    """Monitors and tracks performance of fine-tuned models."""
    
    def __init__(self):
        self.performance_reports: Dict[str, List[ModelPerformanceReport]] = {}
        self.current_sessions: Dict[str, Dict[str, Any]] = {}  # model_id -> session_data
        
    def start_session(self, model_id: str, session_id: str):
        """Start a monitoring session for a model."""
        self.current_sessions[f"{model_id}_{session_id}"] = {
            'start_time': datetime.now(),
            'request_count': 0,
            'response_times': [],
            'error_count': 0
        }
        
    def record_request(self, model_id: str, session_id: str, response_time: float, success: bool = True):
        """Record a request to the model."""
        session_key = f"{model_id}_{session_id}"
        if session_key not in self.current_sessions:
            self.start_session(model_id, session_id)
            
        session = self.current_sessions[session_key]
        session['request_count'] += 1
        session['response_times'].append(response_time)
        
        if not success:
            session['error_count'] += 1
            
    def end_session(self, model_id: str, session_id: str) -> ModelPerformanceReport:
        """End a monitoring session and generate a performance report."""
        session_key = f"{model_id}_{session_id}"
        if session_key not in self.current_sessions:
            raise ValueError(f"No active session for {model_id} with id {session_id}")
            
        session = self.current_sessions[session_key]
        duration = (datetime.now() - session['start_time']).total_seconds()
        
        # Calculate metrics
        avg_response_time = sum(session['response_times']) / len(session['response_times']) if session['response_times'] else 0
        throughput = session['request_count'] / duration if duration > 0 else 0
        error_rate = session['error_count'] / session['request_count'] if session['request_count'] > 0 else 0
        
        report = ModelPerformanceReport(
            model_id=model_id,
            timestamp=datetime.now(),
            accuracy=0.92,  # Placeholder - would come from actual evaluation
            response_time=avg_response_time,
            throughput=throughput,
            error_rate=error_rate,
            resource_utilization={
                'cpu_percent': 45.0,  # Placeholder
                'memory_mb': 1024.0,  # Placeholder
                'gpu_util_percent': 60.0  # Placeholder
            }
        )
        
        # Store the report
        if model_id not in self.performance_reports:
            self.performance_reports[model_id] = []
        self.performance_reports[model_id].append(report)
        
        # Clean up session
        del self.current_sessions[session_key]
        
        return report
        
    def get_model_performance_history(self, model_id: str) -> List[ModelPerformanceReport]:
        """Get performance history for a model."""
        return self.performance_reports.get(model_id, [])
        
    def get_average_performance(self, model_id: str) -> Optional[ModelPerformanceReport]:
        """Get average performance metrics for a model."""
        reports = self.performance_reports.get(model_id, [])
        if not reports:
            return None
            
        # Calculate averages
        avg_accuracy = sum(r.accuracy for r in reports) / len(reports)
        avg_response_time = sum(r.response_time for r in reports) / len(reports)
        avg_throughput = sum(r.throughput for r in reports) / len(reports)
        avg_error_rate = sum(r.error_rate for r in reports) / len(reports)
        
        # Average resource utilization
        total_resources = {}
        for report in reports:
            for key, value in report.resource_utilization.items():
                if key not in total_resources:
                    total_resources[key] = []
                total_resources[key].append(value)
                
        avg_resource_utilization = {
            key: sum(values) / len(values) for key, values in total_resources.items()
        }
        
        return ModelPerformanceReport(
            model_id=model_id,
            timestamp=datetime.now(),
            accuracy=avg_accuracy,
            response_time=avg_response_time,
            throughput=avg_throughput,
            error_rate=avg_error_rate,
            resource_utilization=avg_resource_utilization
        )


class FineTunedModelManager:
    """
    Manages domain-specific fine-tuned models, including automatic domain detection,
    model selection, performance monitoring, and versioning/update mechanisms.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.domain_detector = DomainDetector()
        self.version_manager = ModelVersionManager()
        self.performance_monitor = ModelPerformanceMonitor()
        self.available_models: Dict[str, ModelSpecification] = {}
        self.model_loaders: Dict[ModelProvider, Callable] = {
            ModelProvider.OPENAI: self._load_openai_model,
            ModelProvider.ANTHROPIC: self._load_anthropic_model,
            ModelProvider.HUGGINGFACE: self._load_huggingface_model,
            ModelProvider.CUSTOM: self._load_custom_model
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_configuration(config_path)
            
    def register_model(self, model_spec: ModelSpecification):
        """Register a new fine-tuned model."""
        self.available_models[model_spec.model_id] = model_spec
        self.version_manager.register_model_version(model_spec)
        
    def load_configuration(self, config_path: str):
        """Load model configurations from a JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        for model_data in config_data.get('models', []):
            model_spec = ModelSpecification(
                model_id=model_data['model_id'],
                domain=ModelDomain(model_data['domain']),
                provider=ModelProvider(model_data['provider']),
                version=model_data['version'],
                capabilities=model_data['capabilities'],
                performance_metrics=model_data['performance_metrics'],
                endpoint_url=model_data.get('endpoint_url'),
                api_key=model_data.get('api_key'),
                model_path=model_data.get('model_path'),
                is_active=model_data.get('is_active', True)
            )
            self.register_model(model_spec)
            
    def detect_appropriate_model(self, input_text: str) -> Optional[ModelSpecification]:
        """Detect the appropriate model for the input text based on domain."""
        domain = self.domain_detector.detect_domain(input_text)
        
        # Find models that match the detected domain
        matching_models = [
            spec for spec in self.available_models.values()
            if spec.domain == domain and spec.is_active
        ]
        
        if not matching_models:
            logger.warning(f"No active models found for domain: {domain}")
            return None
            
        # Select the best model based on performance metrics
        # For simplicity, we'll select the one with the highest accuracy
        best_model = max(
            matching_models,
            key=lambda m: m.performance_metrics.get('accuracy', 0)
        )
        
        return best_model
        
    async def load_model(self, model_spec: ModelSpecification):
        """Load a model based on its specification."""
        loader = self.model_loaders.get(model_spec.provider)
        if not loader:
            raise ValueError(f"Unsupported model provider: {model_spec.provider}")
            
        return await loader(model_spec)
        
    async def _load_openai_model(self, model_spec: ModelSpecification):
        """Load an OpenAI model."""
        # In a real implementation, this would connect to OpenAI API
        # For now, we'll simulate the connection
        logger.info(f"Loading OpenAI model: {model_spec.model_id}")
        return {"provider": "openai", "model_id": model_spec.model_id, "loaded": True}
        
    async def _load_anthropic_model(self, model_spec: ModelSpecification):
        """Load an Anthropic model."""
        # In a real implementation, this would connect to Anthropic API
        # For now, we'll simulate the connection
        logger.info(f"Loading Anthropic model: {model_spec.model_id}")
        return {"provider": "anthropic", "model_id": model_spec.model_id, "loaded": True}
        
    async def _load_huggingface_model(self, model_spec: ModelSpecification):
        """Load a Hugging Face model."""
        # In a real implementation, this would load from Hugging Face Hub
        # For now, we'll simulate the loading
        logger.info(f"Loading Hugging Face model: {model_spec.model_id}")
        return {"provider": "huggingface", "model_id": model_spec.model_id, "loaded": True}
        
    async def _load_custom_model(self, model_spec: ModelSpecification):
        """Load a custom model."""
        # In a real implementation, this would load a locally stored model
        # For now, we'll simulate the loading
        logger.info(f"Loading custom model: {model_spec.model_id}")
        return {"provider": "custom", "model_id": model_spec.model_id, "loaded": True}
        
    async def process_with_optimal_model(
        self, 
        input_text: str, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process input text with the optimal model for its domain.
        
        Args:
            input_text: The input text to process
            session_id: Optional session ID for monitoring
            
        Returns:
            Dictionary containing the result and metadata
        """
        # Detect the appropriate model
        model_spec = self.detect_appropriate_model(input_text)
        if not model_spec:
            raise ValueError("No suitable model found for the input text")
            
        # Start monitoring if session ID provided
        if session_id:
            self.performance_monitor.start_session(model_spec.model_id, session_id)
            
        start_time = datetime.now()
        
        try:
            # Load and use the model
            loaded_model = await self.load_model(model_spec)
            
            # Simulate model processing
            # In a real implementation, this would call the actual model
            result = {
                "response": f"Processed by {model_spec.model_id} ({model_spec.domain.value})",
                "model_used": model_spec.model_id,
                "domain_detected": model_spec.domain.value,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
            success = True
        except Exception as e:
            logger.error(f"Error processing with model {model_spec.model_id}: {str(e)}")
            result = {
                "error": str(e),
                "model_used": model_spec.model_id,
                "domain_detected": model_spec.domain.value
            }
            success = False
            
        # Record performance metrics
        if session_id:
            response_time = (datetime.now() - start_time).total_seconds()
            self.performance_monitor.record_request(
                model_spec.model_id, session_id, response_time, success
            )
            
        return result
        
    def get_model_performance(self, model_id: str) -> Optional[ModelPerformanceReport]:
        """Get performance metrics for a specific model."""
        return self.performance_monitor.get_average_performance(model_id)
        
    def update_model_version(self, model_id: str, new_version: str, new_endpoint: str):
        """Update a model to a new version."""
        return self.version_manager.update_model_endpoint(model_id, new_endpoint, new_version)
        
    def get_available_models_by_domain(self, domain: ModelDomain) -> List[ModelSpecification]:
        """Get all available models for a specific domain."""
        return [
            spec for spec in self.available_models.values()
            if spec.domain == domain and spec.is_active
        ]


# Convenience function for easy use
async def process_with_domain_specific_model(
    input_text: str,
    manager: Optional[FineTunedModelManager] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to process text with the optimal domain-specific model.
    
    Args:
        input_text: The input text to process
        manager: Optional model manager instance (will create one if not provided)
        session_id: Optional session ID for monitoring
        
    Returns:
        Dictionary containing the result and metadata
    """
    if manager is None:
        manager = FineTunedModelManager()
        
    return await manager.process_with_optimal_model(input_text, session_id)