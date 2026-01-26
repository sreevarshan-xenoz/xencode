"""
Configuration Manager for Hybrid Model Architecture

Handles configuration for the hybrid model system including provider settings,
model preferences, and routing rules.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import yaml

from ..smart_config_manager import ConfigurationManager


class ModelPreferenceType(Enum):
    """Types of model preferences"""
    PERFORMANCE = "performance"
    PRIVACY = "privacy"
    COST = "cost"
    LATENCY = "latency"
    ACCURACY = "accuracy"


@dataclass
class ModelPreference:
    """Preference for model selection"""
    preference_type: ModelPreferenceType
    weight: float  # 0.0 to 1.0
    enabled: bool = True


@dataclass
class ProviderConfig:
    """Configuration for a model provider"""
    name: str
    enabled: bool
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    default_model: Optional[str] = None
    max_concurrent_requests: int = 5
    timeout: int = 30  # seconds
    retry_attempts: int = 3


@dataclass
class RoutingRule:
    """Rule for routing tasks to specific models"""
    name: str
    condition: str  # e.g., "task_type == 'coding'", "sensitivity > 3"
    preferred_provider: Optional[str] = None
    preferred_model: Optional[str] = None
    fallback_providers: List[str] = None
    
    def __post_init__(self):
        if self.fallback_providers is None:
            self.fallback_providers = []


@dataclass
class HybridModelConfig:
    """Configuration for the hybrid model architecture"""
    providers: List[ProviderConfig]
    preferences: List[ModelPreference]
    routing_rules: List[RoutingRule]
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    fallback_enabled: bool = True
    max_model_switches: int = 3  # Maximum number of model switches per request
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        
        # Convert enums to strings
        for pref in result['preferences']:
            pref['preference_type'] = pref['preference_type'].value
        
        for prov in result['providers']:
            prov['name'] = prov['name']
        
        for rule in result['routing_rules']:
            rule['fallback_providers'] = rule['fallback_providers'] or []
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HybridModelConfig':
        """Create from dictionary"""
        # Convert string values back to enums
        preferences = []
        for pref_data in data.get('preferences', []):
            pref = ModelPreference(
                preference_type=ModelPreferenceType(pref_data['preference_type']),
                weight=pref_data['weight'],
                enabled=pref_data.get('enabled', True)
            )
            preferences.append(pref)
        
        providers = []
        for prov_data in data.get('providers', []):
            prov = ProviderConfig(
                name=prov_data['name'],
                enabled=prov_data['enabled'],
                api_key=prov_data.get('api_key'),
                endpoint=prov_data.get('endpoint'),
                default_model=prov_data.get('default_model'),
                max_concurrent_requests=prov_data.get('max_concurrent_requests', 5),
                timeout=prov_data.get('timeout', 30),
                retry_attempts=prov_data.get('retry_attempts', 3)
            )
            providers.append(prov)
        
        routing_rules = []
        for rule_data in data.get('routing_rules', []):
            rule = RoutingRule(
                name=rule_data['name'],
                condition=rule_data['condition'],
                preferred_provider=rule_data.get('preferred_provider'),
                preferred_model=rule_data.get('preferred_model'),
                fallback_providers=rule_data.get('fallback_providers', [])
            )
            routing_rules.append(rule)
        
        return cls(
            providers=providers,
            preferences=preferences,
            routing_rules=routing_rules,
            cache_enabled=data.get('cache_enabled', True),
            cache_ttl_seconds=data.get('cache_ttl_seconds', 3600),
            fallback_enabled=data.get('fallback_enabled', True),
            max_model_switches=data.get('max_model_switches', 3)
        )


class HybridModelConfigManager:
    """Configuration manager for hybrid model architecture"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".xencode" / "hybrid_model_config.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize with default configuration
        self.config = self._get_default_config()
        
        # Load existing config if it exists
        if self.config_path.exists():
            self.load_config()
        else:
            self.save_config()
    
    def _get_default_config(self) -> HybridModelConfig:
        """Get default configuration"""
        return HybridModelConfig(
            providers=[
                ProviderConfig(
                    name="local_ollama",
                    enabled=True,
                    max_concurrent_requests=3,
                    timeout=60
                ),
                ProviderConfig(
                    name="openai",
                    enabled=False,  # Disabled by default due to API key requirement
                    max_concurrent_requests=5,
                    timeout=30
                ),
                ProviderConfig(
                    name="anthropic",
                    enabled=False,  # Disabled by default due to API key requirement
                    max_concurrent_requests=3,
                    timeout=45
                )
            ],
            preferences=[
                ModelPreference(
                    preference_type=ModelPreferenceType.PRIVACY,
                    weight=0.8,
                    enabled=True
                ),
                ModelPreference(
                    preference_type=ModelPreferenceType.PERFORMANCE,
                    weight=0.7,
                    enabled=True
                ),
                ModelPreference(
                    preference_type=ModelPreferenceType.COST,
                    weight=0.5,
                    enabled=True
                )
            ],
            routing_rules=[
                RoutingRule(
                    name="high_sensitivity_tasks",
                    condition="sensitivity_level >= 4",
                    preferred_provider="local_ollama",
                    fallback_providers=["openai", "anthropic"]
                ),
                RoutingRule(
                    name="coding_tasks",
                    condition="task_type == 'coding'",
                    preferred_provider="local_ollama",
                    preferred_model="codellama",
                    fallback_providers=["openai"]
                ),
                RoutingRule(
                    name="creative_tasks",
                    condition="task_type == 'creative'",
                    preferred_provider="openai",
                    preferred_model="gpt-4",
                    fallback_providers=["anthropic", "local_ollama"]
                )
            ],
            cache_enabled=True,
            cache_ttl_seconds=3600,
            fallback_enabled=True,
            max_model_switches=3
        )
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self.config = HybridModelConfig.from_dict(data)
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            data = self.config.to_dict()
            
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(data, f, default_flow_style=False)
                else:
                    json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider"""
        for provider in self.config.providers:
            if provider.name == provider_name:
                return provider
        return None
    
    def set_provider_api_key(self, provider_name: str, api_key: str) -> bool:
        """Set API key for a provider"""
        for provider in self.config.providers:
            if provider.name == provider_name:
                provider.api_key = api_key
                provider.enabled = True  # Enable provider when API key is set
                self.save_config()
                return True
        return False
    
    def add_routing_rule(self, rule: RoutingRule) -> bool:
        """Add a new routing rule"""
        # Check if rule with same name already exists
        for existing_rule in self.config.routing_rules:
            if existing_rule.name == rule.name:
                return False  # Rule already exists
        
        self.config.routing_rules.append(rule)
        self.save_config()
        return True
    
    def remove_routing_rule(self, rule_name: str) -> bool:
        """Remove a routing rule"""
        for i, rule in enumerate(self.config.routing_rules):
            if rule.name == rule_name:
                del self.config.routing_rules[i]
                self.save_config()
                return True
        return False
    
    def get_matching_rules(self, task_context: Dict[str, Any]) -> List[RoutingRule]:
        """Get routing rules that match the given task context"""
        matching_rules = []
        
        for rule in self.config.routing_rules:
            # Simple condition evaluation - in a real implementation, 
            # this would use a more sophisticated rule engine
            try:
                # Create a safe evaluation context
                eval_context = task_context.copy()
                
                # Evaluate the condition
                if eval(rule.condition, {"__builtins__": {}}, eval_context):
                    matching_rules.append(rule)
            except:
                # If condition evaluation fails, skip this rule
                continue
        
        return matching_rules
    
    def update_preference(self, preference_type: ModelPreferenceType, weight: float) -> bool:
        """Update a preference weight"""
        for pref in self.config.preferences:
            if pref.preference_type == preference_type:
                pref.weight = weight
                self.save_config()
                return True
        return False
    
    def get_provider_priority(self) -> List[str]:
        """Get list of providers in priority order based on preferences"""
        # This is a simplified implementation
        # In a real system, this would use the preferences to determine priority
        enabled_providers = [p.name for p in self.config.providers if p.enabled]
        
        # Prioritize local providers first (for privacy)
        local_providers = [p for p in enabled_providers if p == "local_ollama"]
        cloud_providers = [p for p in enabled_providers if p != "local_ollama"]
        
        return local_providers + cloud_providers


# Global configuration manager instance
_config_manager: Optional[HybridModelConfigManager] = None


def get_hybrid_config_manager() -> HybridModelConfigManager:
    """Get the global hybrid model configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = HybridModelConfigManager()
    return _config_manager


# Example usage
if __name__ == "__main__":
    import tempfile
    
    # Create a temporary config file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_config_path = Path(f.name)
    
    config_manager = HybridModelConfigManager(config_path=temp_config_path)
    
    print("Testing Hybrid Model Configuration Manager...")
    
    # Test getting provider config
    ollama_config = config_manager.get_provider_config("local_ollama")
    print(f"Ollama config enabled: {ollama_config.enabled if ollama_config else 'Not found'}")
    
    # Test setting API key
    config_manager.set_provider_api_key("openai", "test-key-123")
    openai_config = config_manager.get_provider_config("openai")
    print(f"OpenAI config after setting key - enabled: {openai_config.enabled if openai_config else 'Not found'}, has_key: {bool(openai_config.api_key) if openai_config else 'Not found'}")
    
    # Test adding a routing rule
    new_rule = RoutingRule(
        name="test_rule",
        condition="task_type == 'testing'",
        preferred_provider="local_ollama"
    )
    config_manager.add_routing_rule(new_rule)
    print(f"Added new rule: {new_rule.name}")
    
    # Test getting provider priority
    priority = config_manager.get_provider_priority()
    print(f"Provider priority: {priority}")
    
    # Test updating preference
    config_manager.update_preference(ModelPreferenceType.LATENCY, 0.9)
    print("Updated latency preference")
    
    # Clean up
    temp_config_path.unlink()
    
    print("✅ Configuration manager tests completed!")