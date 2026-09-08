#!/usr/bin/env python3
"""
Prompt Routing Layer

Routes prompts to the best provider/model based on task classification and policy.

Features:
- Task classification (code, chat, reasoning, creative, etc.)
- Provider/model selection based on task type
- Policy-based routing with cost/latency constraints
- Fallback chain support
- Context-aware routing decisions
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

console = Console()


class TaskType(Enum):
    """Task type enumeration"""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_EXPLANATION = "code_explanation"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TEST_GENERATION = "test_generation"
    DOCUMENTATION = "documentation"
    CHAT = "chat"
    REASONING = "reasoning"
    CREATIVE = "creative"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    GENERAL = "general"


class ProviderType(Enum):
    """Provider type enumeration"""
    LOCAL_OLLAMA = "local_ollama"
    CLOUD_QWEN = "cloud_qwen"
    CLOUD_OPENROUTER = "cloud_openrouter"
    CLOUD_ANTHROPIC = "cloud_anthropic"
    CLOUD_OPENAI = "cloud_openai"


@dataclass
class TaskClassification:
    """Classification result for a task"""
    task_type: TaskType
    confidence: float
    complexity: str  # low, medium, high
    requires_context: bool
    requires_reasoning: bool
    requires_coding: bool
    keywords: List[str] = field(default_factory=list)
    language: Optional[str] = None
    estimated_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_type': self.task_type.value,
            'confidence': self.confidence,
            'complexity': self.complexity,
            'requires_context': self.requires_context,
            'requires_reasoning': self.requires_reasoning,
            'requires_coding': self.requires_coding,
            'keywords': self.keywords,
            'language': self.language,
            'estimated_tokens': self.estimated_tokens,
        }


@dataclass
class RoutingDecision:
    """Routing decision for a prompt"""
    provider: ProviderType
    model: str
    reason: str
    fallback_chain: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_latency_ms: int = 0
    policy_applied: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'provider': self.provider.value,
            'model': self.model,
            'reason': self.reason,
            'fallback_chain': self.fallback_chain,
            'estimated_cost': self.estimated_cost,
            'estimated_latency_ms': self.estimated_latency_ms,
            'policy_applied': self.policy_applied,
        }


@dataclass
class RoutingPolicy:
    """
    Routing policy configuration

    Attributes:
        name: Policy name
        priority: Priority order (lower = higher priority)
        task_types: Task types this policy applies to
        preferred_provider: Preferred provider type
        preferred_model: Preferred model ID
        fallback_chain: Fallback model chain
        max_cost_per_request: Maximum cost per request in USD
        max_latency_ms: Maximum acceptable latency in milliseconds
        require_local_first: If True, try local models first
        context_window_min: Minimum required context window
        capabilities_required: Required capabilities (code, chat, etc.)
    """
    name: str
    priority: int = 0
    task_types: List[TaskType] = field(default_factory=list)
    preferred_provider: Optional[ProviderType] = None
    preferred_model: Optional[str] = None
    fallback_chain: List[str] = field(default_factory=list)
    max_cost_per_request: float = 0.01
    max_latency_ms: int = 5000
    require_local_first: bool = False
    context_window_min: int = 4096
    capabilities_required: List[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'priority': self.priority,
            'task_types': [t.value for t in self.task_types],
            'preferred_provider': self.preferred_provider.value if self.preferred_provider else None,
            'preferred_model': self.preferred_model,
            'fallback_chain': self.fallback_chain,
            'max_cost_per_request': self.max_cost_per_request,
            'max_latency_ms': self.max_latency_ms,
            'require_local_first': self.require_local_first,
            'context_window_min': self.context_window_min,
            'capabilities_required': self.capabilities_required,
            'enabled': self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoutingPolicy':
        return cls(
            name=data['name'],
            priority=data.get('priority', 0),
            task_types=[TaskType(t) for t in data.get('task_types', [])],
            preferred_provider=ProviderType(data['preferred_provider']) if data.get('preferred_provider') else None,
            preferred_model=data.get('preferred_model'),
            fallback_chain=data.get('fallback_chain', []),
            max_cost_per_request=data.get('max_cost_per_request', 0.01),
            max_latency_ms=data.get('max_latency_ms', 5000),
            require_local_first=data.get('require_local_first', False),
            context_window_min=data.get('context_window_min', 4096),
            capabilities_required=data.get('capabilities_required', []),
            enabled=data.get('enabled', True),
        )


class TaskClassifier:
    """
    Classifies prompts into task types using keyword matching and heuristics
    """

    # Keyword patterns for task classification
    TASK_PATTERNS: Dict[TaskType, List[str]] = {
        TaskType.CODE_GENERATION: [
            r'\b(write|create|generate|implement)\s+(code|function|class|method|script)',
            r'\b(write|create|generate|implement)\s+(a|an|the)?\s*\w+\s+(function|class|method|script|program)',
            r'\b(write|create|generate|implement)\s+(this\s+)?(code|a\s+code)\s+(for|to)\b',
            r'\bimplement\s+(a|an|the)?\s*(feature|function|class)',
            r'\bbuild\s+(a|an|the)?\s*(app|application|component|module)',
        ],
        TaskType.CODE_REVIEW: [
            r'\b(review|analyze|check|evaluate)\s+(this\s+)?code',
            r'\bcode\s+review\b',
            r'\bfind\s+(bugs|issues|problems|vulnerabilities)',
            r'\bimprove\s+(this\s+)?code',
        ],
        TaskType.CODE_EXPLANATION: [
            r'\b(explain|describe|what does|how does)\s+(this\s+)?code',
            r'\bwhat\s+is\s+(this|that)\s+(doing|for)\b',
            r'\bhow\s+does\s+(this|that)\s+work\b',
        ],
        TaskType.DEBUGGING: [
            r'\b(debug|fix|solve|resolve)\s+(this\s+)?(error|bug|issue|problem)',
            r'\b(not working|doesn\'t work|broken|failed)\b',
            r'\berror\b.*\b(line|at|in)\b',
            r'\bexception\b.*\b(thrown|raised|occurred)\b',
            r'\btraceback\b',
            r'\b\w*(Error|Exception)\b',
        ],
        TaskType.REFACTORING: [
            r'\b(refactor|optimize|improve|clean up)\s+(this\s+)?code',
            r'\b(make it|more)\s+(efficient|readable|maintainable|clean)',
            r'\brestructure\b',
            r'\breorganize\b',
        ],
        TaskType.TEST_GENERATION: [
            r'\b(write|create|generate)\s+tests?(s)?',
            r'\btest\s+(case|suite|coverage)',
            r'\bunit\s+test\b',
            r'\bintegration\s+test\b',
        ],
        TaskType.DOCUMENTATION: [
            r'\b(document|write\s+docs|comment)\s+(this\s+)?code',
            r'\b(add|write)\s+(documentation|comments|docstrings)',
            r'\bexplain\s+in\s+detail\b',
        ],
        TaskType.REASONING: [
            r'\b(think|reason|analyze|evaluate|compare)\b',
            r'\bwhat\s+is\s+the\s+(best|optimal|correct)\s+way',
            r'\bpros\s+and\s+cons\b',
            r'\btrade[- ]offs?\b',
        ],
        TaskType.CREATIVE: [
            r'\b(brainstorm|ideate|come up with|imagine)\b',
            r'\bcreative\s+(idea|solution|approach)\b',
            r'\bdesign\s+(a|an|the)?\s*(system|architecture|pattern)\b',
        ],
        TaskType.RESEARCH: [
            r'\b(search|find|look up|investigate)\s+information',
            r'\bwhat\s+are\s+the\s+(latest|best|top)\b',
            r'\bresearch\s+(paper|article|resource)\b',
        ],
        TaskType.CHAT: [
            r'\b(hello|hi|hey|good morning|good afternoon)\b',
            r'\bhow\s+are\s+you\b',
            r'\b(thanks|thank you|please)\b',
            r'\b(what\s+is|tell me about|define)\b',
        ],
    }

    # Complexity indicators
    COMPLEXITY_HIGH_PATTERNS = [
        r'\b(complex|advanced|sophisticated|intricate)\b',
        r'\b(multi[- ]step|multi[- ]stage|multi[- ]layer)\b',
        r'\b(architecture|design pattern|system design)\b',
        r'\b(optimization|performance|scalability)\b',
    ]

    COMPLEXITY_LOW_PATTERNS = [
        r'\b(simple|easy|basic|quick|small)\b',
        r'\b(one[- ]liner|snippet|example)\b',
        r'\b(hello world|beginner|intro)\b',
    ]

    # Programming language patterns
    LANGUAGE_PATTERNS: Dict[str, List[str]] = {
        'python': [r'\bpython\b', r'\.py\b', r'\bdef\s+\w+\s*\(', r'\bimport\s+\w+'],
        'javascript': [r'\bjavascript\b', r'\bjs\b', r'\.js\b', r'=>', r'\bconst\s+\w+\s*='],
        'typescript': [r'\btypescript\b', r'\bts\b', r'\.ts\b', r':\s*(string|number|boolean|any)\b'],
        'java': [r'\bjava\b', r'\.java\b', r'\bpublic\s+class\b', r'\bSystem\.out\.println\b'],
        'rust': [r'\brust\b', r'\.rs\b', r'\bfn\s+\w+\s*\(', r'\blet\s+mut\b'],
        'go': [r'\bgo\b', r'\.go\b', r'\bfunc\s+\w+\s*\(', r'\bvar\s+\w+\s+'],
        'cpp': [r'\bc\+\+\b', r'\bcpp\b', r'\.cpp\b', r'\bstd::\w+', r'\b#include\b'],
        'sql': [r'\bsql\b', r'\.sql\b', r'\bSELECT\b', r'\bFROM\b', r'\bWHERE\b'],
    }

    def __init__(self):
        # Pre-compile patterns for performance
        self.compiled_task_patterns = {
            task_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for task_type, patterns in self.TASK_PATTERNS.items()
        }

        self.compiled_complexity_high = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.COMPLEXITY_HIGH_PATTERNS
        ]
        self.compiled_complexity_low = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.COMPLEXITY_LOW_PATTERNS
        ]

        self.compiled_languages = {
            lang: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for lang, patterns in self.LANGUAGE_PATTERNS.items()
        }

    def classify(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> TaskClassification:
        """
        Classify a prompt into a task type

        Args:
            prompt: User prompt text
            context: Optional context (file type, conversation history, etc.)

        Returns:
            TaskClassification result
        """
        # Score each task type
        task_scores: Dict[TaskType, float] = dict.fromkeys(TaskType, 0.0)
        matched_keywords: Dict[TaskType, List[str]] = {t: [] for t in TaskType}

        for task_type, patterns in self.compiled_task_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(prompt)
                if matches:
                    task_scores[task_type] += len(matches)
                    matched_keywords[task_type].extend([m for m in matches if isinstance(m, str)])

        # Boost scores based on context
        if context:
            if context.get('file_type') == '.py':
                task_scores[TaskType.CODE_GENERATION] *= 1.2
                task_scores[TaskType.CODE_REVIEW] *= 1.2
            if context.get('has_error', False):
                task_scores[TaskType.DEBUGGING] *= 1.5

        # Find best matching task type
        best_task = max(task_scores, key=task_scores.get)
        best_score = task_scores[best_task]

        # If no strong match, default to general
        if best_score < 1.0:
            best_task = TaskType.GENERAL
            best_score = 0.5

        # Calculate confidence (normalize to 0-1)
        confidence = min(1.0, best_score / 5.0)

        # Determine complexity
        complexity = self._detect_complexity(prompt)

        # Detect programming language
        language = self._detect_language(prompt, context)

        # Estimate tokens
        estimated_tokens = len(prompt) // 4

        # Determine if requires context/reasoning/coding
        requires_reasoning = best_task in [
            TaskType.REASONING, TaskType.ANALYSIS, TaskType.CODE_REVIEW, TaskType.DEBUGGING
        ]
        requires_coding = best_task in [
            TaskType.CODE_GENERATION, TaskType.CODE_REVIEW, TaskType.REFACTORING,
            TaskType.TEST_GENERATION, TaskType.DEBUGGING
        ]
        requires_context = best_task in [
            TaskType.CODE_GENERATION, TaskType.CODE_REVIEW, TaskType.REFACTORING,
            TaskType.DEBUGGING, TaskType.CODE_EXPLANATION
        ]

        return TaskClassification(
            task_type=best_task,
            confidence=confidence,
            complexity=complexity,
            requires_context=requires_context,
            requires_reasoning=requires_reasoning,
            requires_coding=requires_coding,
            keywords=matched_keywords[best_task][:5],  # Top 5 keywords
            language=language,
            estimated_tokens=estimated_tokens,
        )

    def _detect_complexity(self, prompt: str) -> str:
        """Detect task complexity"""
        high_score = sum(
            len(pattern.findall(prompt))
            for pattern in self.compiled_complexity_high
        )
        low_score = sum(
            len(pattern.findall(prompt))
            for pattern in self.compiled_complexity_low
        )

        if high_score > low_score:
            return 'high'
        elif low_score > high_score:
            return 'low'
        else:
            return 'medium'

    def _detect_language(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Detect programming language"""
        # Check context first
        if context:
            if 'file_type' in context:
                ext = context['file_type']
                for lang, patterns in self.LANGUAGE_PATTERNS.items():
                    if any(re.search(p, ext, re.IGNORECASE) for p in patterns):
                        return lang

        # Check prompt content
        lang_scores: Dict[str, int] = dict.fromkeys(self.LANGUAGE_PATTERNS, 0)

        for lang, patterns in self.compiled_languages.items():
            for pattern in patterns:
                matches = pattern.findall(prompt)
                if matches:
                    lang_scores[lang] += len(matches)

        best_lang = max(lang_scores, key=lang_scores.get)
        if lang_scores[best_lang] > 0:
            return best_lang

        return None


class PromptRouter:
    """
    Routes prompts to appropriate provider/model based on classification and policy
    """

    # Default model mappings by task type
    DEFAULT_MODEL_MAP: Dict[TaskType, Dict[ProviderType, str]] = {
        TaskType.CODE_GENERATION: {
            ProviderType.CLOUD_QWEN: "qwen3-coder-next-instruct",
            ProviderType.LOCAL_OLLAMA: "qwen2.5-coder:7b",
            ProviderType.CLOUD_OPENROUTER: "qwen/qwen-2.5-coder-32b-instruct",
        },
        TaskType.CODE_REVIEW: {
            ProviderType.CLOUD_QWEN: "qwen3-coder-next-instruct",
            ProviderType.LOCAL_OLLAMA: "qwen2.5-coder:7b",
            ProviderType.CLOUD_OPENROUTER: "qwen/qwen-2.5-coder-32b-instruct",
        },
        TaskType.DEBUGGING: {
            ProviderType.CLOUD_QWEN: "qwen3-coder-next-instruct",
            ProviderType.LOCAL_OLLAMA: "qwen2.5-coder:7b",
        },
        TaskType.REASONING: {
            ProviderType.CLOUD_QWEN: "qwen-max",
            ProviderType.LOCAL_OLLAMA: "qwen2.5:72b",
            ProviderType.CLOUD_ANTHROPIC: "claude-3-5-sonnet-20241022",
        },
        TaskType.CHAT: {
            ProviderType.LOCAL_OLLAMA: "llama3.2:3b",
            ProviderType.CLOUD_QWEN: "qwen-turbo",
        },
        TaskType.DOCUMENTATION: {
            ProviderType.LOCAL_OLLAMA: "qwen2.5:7b",
            ProviderType.CLOUD_QWEN: "qwen-turbo",
        },
        TaskType.GENERAL: {
            ProviderType.LOCAL_OLLAMA: "qwen2.5:7b",
            ProviderType.CLOUD_QWEN: "qwen-turbo",
        },
    }

    # Provider capabilities
    PROVIDER_CAPABILITIES: Dict[ProviderType, Dict[str, Any]] = {
        ProviderType.LOCAL_OLLAMA: {
            'cost_per_1k_tokens': 0.0,
            'avg_latency_ms': 500,
            'privacy': 'high',
            'reliability': 'medium',
        },
        ProviderType.CLOUD_QWEN: {
            'cost_per_1k_tokens': 0.002,
            'avg_latency_ms': 1500,
            'privacy': 'medium',
            'reliability': 'high',
        },
        ProviderType.CLOUD_OPENROUTER: {
            'cost_per_1k_tokens': 0.003,
            'avg_latency_ms': 2000,
            'privacy': 'medium',
            'reliability': 'high',
        },
        ProviderType.CLOUD_ANTHROPIC: {
            'cost_per_1k_tokens': 0.01,
            'avg_latency_ms': 2500,
            'privacy': 'medium',
            'reliability': 'very_high',
        },
    }

    def __init__(
        self,
        policies: Optional[List[RoutingPolicy]] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize prompt router

        Args:
            policies: Optional list of routing policies
            config_path: Optional path to configuration file
        """
        self.classifier = TaskClassifier()
        self.policies: List[RoutingPolicy] = policies or []
        self.config_path = config_path or Path.home() / ".xencode" / "routing_config.json"

        # Load default policies if none provided
        if not self.policies:
            self.policies = self._create_default_policies()

        # Load configuration from file if exists
        if self.config_path.exists():
            self._load_config()

        # Provider health tracking
        self.provider_health: Dict[ProviderType, Dict[str, Any]] = {
            p: {'status': 'healthy', 'last_error': None, 'error_count': 0}
            for p in ProviderType
        }

    def _create_default_policies(self) -> List[RoutingPolicy]:
        """Create default routing policies"""
        return [
            RoutingPolicy(
                name="local_first",
                priority=1,
                task_types=[TaskType.CHAT, TaskType.DOCUMENTATION, TaskType.GENERAL],
                preferred_provider=ProviderType.LOCAL_OLLAMA,
                require_local_first=True,
            ),
            RoutingPolicy(
                name="coding_tasks",
                priority=2,
                task_types=[
                    TaskType.CODE_GENERATION, TaskType.CODE_REVIEW,
                    TaskType.DEBUGGING, TaskType.REFACTORING,
                ],
                preferred_provider=ProviderType.CLOUD_QWEN,
                preferred_model="qwen3-coder-next-instruct",
                fallback_chain=[
                    "qwen3-coder-next-instruct",
                    "qwen-coder-plus",
                    "qwen2.5-coder:7b",
                ],
                capabilities_required=["code"],
            ),
            RoutingPolicy(
                name="reasoning_tasks",
                priority=3,
                task_types=[TaskType.REASONING, TaskType.ANALYSIS, TaskType.RESEARCH],
                preferred_provider=ProviderType.CLOUD_QWEN,
                preferred_model="qwen-max",
                fallback_chain=[
                    "qwen-max",
                    "qwen-plus",
                    "qwen2.5:72b",
                ],
                capabilities_required=["reasoning"],
            ),
            RoutingPolicy(
                name="cost_optimized",
                priority=10,
                max_cost_per_request=0.005,
                max_latency_ms=3000,
            ),
        ]

    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)

            # Load policies
            if 'policies' in data:
                self.policies = [
                    RoutingPolicy.from_dict(p) for p in data['policies']
                ]

            # Load provider health
            if 'provider_health' in data:
                for provider_str, health in data['provider_health'].items():
                    provider = ProviderType(provider_str)
                    self.provider_health[provider] = health

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load routing config: {e}[/yellow]")

    def _save_config(self):
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'policies': [p.to_dict() for p in self.policies],
                'provider_health': {
                    p.value: h for p, h in self.provider_health.items()
                },
                'last_updated': datetime.now().isoformat(),
            }

            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save routing config: {e}[/yellow]")

    def route(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Route a prompt to appropriate provider/model

        Args:
            prompt: User prompt text
            context: Optional context information
            user_preferences: Optional user preferences (model override, cost limits, etc.)

        Returns:
            RoutingDecision with provider, model, and rationale
        """
        # Classify the task
        classification = self.classifier.classify(prompt, context)

        # Find matching policy
        matching_policy = self._find_matching_policy(
            classification.task_type,
            user_preferences,
        )

        # Make routing decision
        decision = self._make_routing_decision(
            classification,
            matching_policy,
            user_preferences,
        )

        # Log decision
        console.print(
            f"[blue]Routing:[/blue] {classification.task_type.value} → "
            f"{decision.provider.value}/{decision.model}"
        )

        return decision

    def _find_matching_policy(
        self,
        task_type: TaskType,
        user_preferences: Optional[Dict[str, Any]] = None,
    ) -> Optional[RoutingPolicy]:
        """Find the best matching policy for a task type"""
        # Sort policies by priority
        sorted_policies = sorted(self.policies, key=lambda p: p.priority)

        for policy in sorted_policies:
            if not policy.enabled:
                continue

            # Check if policy applies to this task type
            if policy.task_types and task_type not in policy.task_types:
                continue

            # Check user preferences override
            if user_preferences:
                if user_preferences.get('force_model'):
                    # User forced a specific model, use cost-optimized policy
                    if policy.name == "cost_optimized":
                        return policy
                    continue

            return policy

        # Default to first policy if no match
        return sorted_policies[0] if sorted_policies else None

    def _make_routing_decision(
        self,
        classification: TaskClassification,
        policy: Optional[RoutingPolicy],
        user_preferences: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Make routing decision based on classification and policy"""
        # Check for user override
        if user_preferences and user_preferences.get('force_model'):
            model = user_preferences['force_model']
            return RoutingDecision(
                provider=ProviderType.LOCAL_OLLAMA,  # Default to local for overrides
                model=model,
                reason=f"User override: {model}",
                policy_applied="user_override",
            )

        # Get default model for task type
        default_models = self.DEFAULT_MODEL_MAP.get(classification.task_type, {})

        # Determine provider and model
        if policy:
            # Use policy preferences
            provider = policy.preferred_provider or ProviderType.LOCAL_OLLAMA
            model = policy.preferred_model or default_models.get(
                provider,
                self._get_default_model(provider, classification.task_type),
            )
            fallback_chain = policy.fallback_chain or [model]
            policy_name = policy.name
        else:
            # Use defaults
            if classification.requires_coding:
                provider = ProviderType.CLOUD_QWEN
                model = default_models.get(provider, "qwen3-coder-next-instruct")
            else:
                provider = ProviderType.LOCAL_OLLAMA
                model = default_models.get(provider, "qwen2.5:7b")
            fallback_chain = [model]
            policy_name = "default"

        # Check provider health
        if self.provider_health[provider]['status'] != 'healthy':
            # Try to use fallback
            if fallback_chain:
                console.print(
                    f"[yellow]Warning: Provider {provider.value} unhealthy, using fallback[/yellow]"
                )
                model = fallback_chain[0]

        # Estimate cost and latency
        provider_caps = self.PROVIDER_CAPABILITIES.get(provider, {})
        estimated_cost = provider_caps.get('cost_per_1k_tokens', 0) * (classification.estimated_tokens / 1000)
        estimated_latency = provider_caps.get('avg_latency_ms', 1000)

        # Adjust for complexity
        if classification.complexity == 'high':
            estimated_latency = int(estimated_latency * 1.5)
        elif classification.complexity == 'low':
            estimated_latency = int(estimated_latency * 0.7)

        return RoutingDecision(
            provider=provider,
            model=model,
            reason=f"Task: {classification.task_type.value}, Policy: {policy_name}, Confidence: {classification.confidence:.2f}",
            fallback_chain=fallback_chain,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            policy_applied=policy_name,
        )

    def _get_default_model(self, provider: ProviderType, task_type: TaskType) -> str:
        """Get default model for provider and task type"""
        default_models = self.DEFAULT_MODEL_MAP.get(task_type, {})
        return default_models.get(provider, "qwen2.5:7b")

    def update_provider_health(
        self,
        provider: ProviderType,
        status: str,
        error: Optional[str] = None,
    ):
        """
        Update provider health status

        Args:
            provider: Provider type
            status: Health status (healthy, degraded, unhealthy)
            error: Optional error message
        """
        self.provider_health[provider] = {
            'status': status,
            'last_error': error,
            'error_count': self.provider_health[provider].get('error_count', 0) + (1 if error else 0),
        }

        # Auto-recover after timeout
        if status == 'healthy':
            self.provider_health[provider]['error_count'] = 0

        self._save_config()

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models by provider"""
        models = []

        for _task_type, provider_models in self.DEFAULT_MODEL_MAP.items():
            for provider, model in provider_models.items():
                if model not in [m['model'] for m in models]:
                    models.append({
                        'model': model,
                        'provider': provider.value,
                        'task_types': [
                            t.value for t, pm in self.DEFAULT_MODEL_MAP.items()
                            if pm.get(provider) == model
                        ],
                    })

        return models

    def add_policy(self, policy: RoutingPolicy):
        """Add a new routing policy"""
        self.policies.append(policy)
        self._save_config()

    def remove_policy(self, policy_name: str) -> bool:
        """Remove a policy by name"""
        for i, policy in enumerate(self.policies):
            if policy.name == policy_name:
                del self.policies[i]
                self._save_config()
                return True
        return False

    def get_policy(self, policy_name: str) -> Optional[RoutingPolicy]:
        """Get a policy by name"""
        for policy in self.policies:
            if policy.name == policy_name:
                return policy
        return None


# Global router instance
_router: Optional[PromptRouter] = None


def get_router(config_path: Optional[Path] = None) -> PromptRouter:
    """Get or create global prompt router"""
    global _router
    if _router is None:
        _router = PromptRouter(config_path=config_path)
    return _router


def route_prompt(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    user_preferences: Optional[Dict[str, Any]] = None,
) -> RoutingDecision:
    """
    Convenience function to route a prompt

    Args:
        prompt: User prompt text
        context: Optional context information
        user_preferences: Optional user preferences

    Returns:
        RoutingDecision with provider, model, and rationale
    """
    router = get_router()
    return router.route(prompt, context, user_preferences)


if __name__ == "__main__":
    # Demo/test the router
    import sys

    def demo_routing():
        console.print("[bold blue]Prompt Routing Layer Demo[/bold blue]\n")

        router = PromptRouter()

        # Test prompts
        test_prompts = [
            "Write a Python function to sort a list",
            "Review this code for security issues",
            "Why is my code throwing a TypeError?",
            "Explain how async/await works",
            "What's the best way to structure a microservice?",
            "Hello, how are you?",
        ]

        for prompt in test_prompts:
            console.print(f"\n[bold]Prompt:[/bold] {prompt}")
            decision = router.route(prompt)
            console.print(f"  [green]Provider:[/green] {decision.provider.value}")
            console.print(f"  [green]Model:[/green] {decision.model}")
            console.print(f"  [green]Reason:[/green] {decision.reason}")
            console.print(f"  [green]Est. Cost:[/green] ${decision.estimated_cost:.4f}")
            console.print(f"  [green]Est. Latency:[/green] {decision.estimated_latency_ms}ms")

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_routing()
    else:
        console.print("Prompt Routing Layer module")
        console.print("Usage: python -m xencode.routing.prompt_router --demo")
