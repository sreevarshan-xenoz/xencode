#!/usr/bin/env python3
"""
Multi-Model Conversation System for Xencode
Allows switching between models mid-conversation and smart model selection
"""

import json
import time
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    """Types of queries for smart model selection"""
    CODE = "code"
    EXPLANATION = "explanation"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    GENERAL = "general"

@dataclass
class ModelCapabilities:
    """Model capabilities and performance characteristics"""
    name: str
    size: str
    strengths: List[str]
    speed_score: int  # 1-10, higher is faster
    quality_score: int  # 1-10, higher is better quality
    specialties: List[QueryType]
    context_window: int

class MultiModelManager:
    """Advanced multi-model conversation system"""
    
    def __init__(self):
        self.available_models = {}
        self.current_model = "qwen3:4b"
        self.conversation_history = []
        self.model_performance = {}
        self.load_model_capabilities()
        self.refresh_available_models()
    
    def load_model_capabilities(self):
        """Load known model capabilities"""
        self.model_capabilities = {
            "qwen3:4b": ModelCapabilities(
                name="qwen3:4b",
                size="2.5GB",
                strengths=["fast", "efficient", "general"],
                speed_score=9,
                quality_score=7,
                specialties=[QueryType.GENERAL, QueryType.EXPLANATION],
                context_window=8192
            ),
            "llama2:7b": ModelCapabilities(
                name="llama2:7b",
                size="3.8GB", 
                strengths=["balanced", "reasoning", "conversation"],
                speed_score=6,
                quality_score=8,
                specialties=[QueryType.ANALYSIS, QueryType.GENERAL],
                context_window=4096
            ),
            "codellama:7b": ModelCapabilities(
                name="codellama:7b",
                size="3.8GB",
                strengths=["programming", "code-review", "debugging"],
                speed_score=6,
                quality_score=9,
                specialties=[QueryType.CODE],
                context_window=16384
            ),
            "mistral:7b": ModelCapabilities(
                name="mistral:7b",
                size="4.1GB",
                strengths=["creative", "writing", "analysis"],
                speed_score=7,
                quality_score=8,
                specialties=[QueryType.CREATIVE, QueryType.ANALYSIS],
                context_window=8192
            )
        }
    
    def refresh_available_models(self):
        """Check which models are actually installed"""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            self.available_models = {}
            
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    if model_name in self.model_capabilities:
                        self.available_models[model_name] = self.model_capabilities[model_name]
                        
        except Exception:
            # Fallback to default model
            if "qwen3:4b" in self.model_capabilities:
                self.available_models["qwen3:4b"] = self.model_capabilities["qwen3:4b"]
    
    def detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query to suggest best model"""
        query_lower = query.lower()
        
        # Code-related keywords
        code_keywords = [
            "code", "function", "class", "debug", "error", "bug", "programming",
            "python", "javascript", "java", "c++", "rust", "go", "sql",
            "algorithm", "data structure", "api", "framework", "library",
            "compile", "syntax", "variable", "loop", "condition"
        ]
        
        # Creative keywords
        creative_keywords = [
            "story", "poem", "creative", "write", "imagine", "fiction",
            "character", "plot", "narrative", "essay", "article", "blog"
        ]
        
        # Analysis keywords
        analysis_keywords = [
            "analyze", "compare", "evaluate", "assess", "review", "critique",
            "pros and cons", "advantages", "disadvantages", "trade-offs"
        ]
        
        # Explanation keywords
        explanation_keywords = [
            "explain", "how", "why", "what is", "define", "describe",
            "tutorial", "guide", "learn", "understand", "concept"
        ]
        
        # Count keyword matches
        code_score = sum(1 for keyword in code_keywords if keyword in query_lower)
        creative_score = sum(1 for keyword in creative_keywords if keyword in query_lower)
        analysis_score = sum(1 for keyword in analysis_keywords if keyword in query_lower)
        explanation_score = sum(1 for keyword in explanation_keywords if keyword in query_lower)
        
        # Determine query type based on highest score
        scores = {
            QueryType.CODE: code_score,
            QueryType.CREATIVE: creative_score,
            QueryType.ANALYSIS: analysis_score,
            QueryType.EXPLANATION: explanation_score
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        
        return QueryType.GENERAL
    
    def suggest_best_model(self, query: str) -> Tuple[str, str]:
        """Suggest the best model for a given query"""
        query_type = self.detect_query_type(query)
        
        # Find models that specialize in this query type
        specialized_models = [
            model for model in self.available_models.values()
            if query_type in model.specialties
        ]
        
        if specialized_models:
            # Choose the best specialized model (balance quality and speed)
            best_model = max(
                specialized_models,
                key=lambda m: m.quality_score * 0.7 + m.speed_score * 0.3
            )
            reason = f"Specialized for {query_type.value} tasks"
        else:
            # Fall back to best general model
            general_models = [
                model for model in self.available_models.values()
                if QueryType.GENERAL in model.specialties
            ]
            
            if general_models:
                best_model = max(
                    general_models,
                    key=lambda m: m.quality_score * 0.6 + m.speed_score * 0.4
                )
                reason = "Best general-purpose model available"
            else:
                # Ultimate fallback
                best_model = list(self.available_models.values())[0]
                reason = "Default model"
        
        return best_model.name, reason
    
    def switch_model(self, new_model: str) -> bool:
        """Switch to a different model"""
        if new_model in self.available_models:
            old_model = self.current_model
            self.current_model = new_model
            
            # Add model switch to conversation history
            self.conversation_history.append({
                "type": "model_switch",
                "from": old_model,
                "to": new_model,
                "timestamp": time.time()
            })
            
            return True
        return False
    
    def get_model_comparison(self) -> Dict:
        """Get comparison of available models"""
        comparison = {}
        
        for model_name, model in self.available_models.items():
            comparison[model_name] = {
                "size": model.size,
                "speed": "⚡" * model.speed_score,
                "quality": "⭐" * model.quality_score,
                "strengths": ", ".join(model.strengths),
                "specialties": [s.value for s in model.specialties],
                "context_window": f"{model.context_window:,} tokens"
            }
        
        return comparison
    
    def get_conversation_context(self, max_messages: int = 10) -> str:
        """Get recent conversation context for model switching"""
        recent_messages = self.conversation_history[-max_messages:]
        
        context_parts = []
        for msg in recent_messages:
            if msg["type"] == "user":
                context_parts.append(f"User: {msg['content'][:100]}...")
            elif msg["type"] == "assistant":
                context_parts.append(f"Assistant ({msg.get('model', 'unknown')}): {msg['content'][:100]}...")
            elif msg["type"] == "model_switch":
                context_parts.append(f"[Switched from {msg['from']} to {msg['to']}]")
        
        return "\n".join(context_parts)

# Example usage functions
def demo_multi_model_system():
    """Demonstrate the multi-model system"""
    manager = MultiModelManager()
    
    print("🚀 Multi-Model System Demo")
    print("=" * 40)
    
    # Show available models
    print("\n📋 Available Models:")
    comparison = manager.get_model_comparison()
    for model, info in comparison.items():
        print(f"  • {model}: {info['strengths']} ({info['size']})")
    
    # Test query type detection and model suggestion
    test_queries = [
        "Write a Python function to sort a list",
        "Explain quantum computing in simple terms", 
        "Write a creative story about space travel",
        "Compare the pros and cons of React vs Vue"
    ]
    
    print("\n🎯 Smart Model Suggestions:")
    for query in test_queries:
        suggested_model, reason = manager.suggest_best_model(query)
        query_type = manager.detect_query_type(query)
        print(f"  Query: '{query[:50]}...'")
        print(f"  Type: {query_type.value}")
        print(f"  Suggested: {suggested_model} ({reason})")
        print()

if __name__ == "__main__":
    demo_multi_model_system()