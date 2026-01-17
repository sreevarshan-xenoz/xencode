"""
Cross-domain expertise combination system for multi-agent systems in Xencode
"""
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import sqlite3
import threading
from pathlib import Path
from collections import defaultdict
import re


class DomainType(Enum):
    """Types of domains in the system."""
    DATA_SCIENCE = "data_science"
    WEB_DEVELOPMENT = "web_development"
    SECURITY_ANALYSIS = "security_analysis"
    DEVOPS = "devops"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    BUSINESS_LOGIC = "business_logic"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    GENERAL = "general"


class TranslationType(Enum):
    """Types of knowledge translation."""
    TERMINOLOGY = "terminology"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    CONTEXTUAL = "contextual"


@dataclass
class DomainKnowledge:
    """Represents knowledge within a specific domain."""
    knowledge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    domain: DomainType = DomainType.GENERAL
    concept: str = ""
    definition: str = ""
    terminology_variants: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    usage_examples: List[str] = field(default_factory=list)
    confidence_score: float = 0.5  # 0.0 to 1.0
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationRule:
    """Rule for translating knowledge between domains."""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_domain: DomainType = DomainType.GENERAL
    target_domain: DomainType = DomainType.GENERAL
    source_concept: str = ""
    target_concept: str = ""
    translation_type: TranslationType = TranslationType.TERMINOLOGY
    translation_mapping: Dict[str, str] = field(default_factory=dict)
    confidence_score: float = 0.5  # 0.0 to 1.0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CrossDomainRequest:
    """Request for cross-domain expertise combination."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requesting_agent: str = ""
    source_domain: DomainType = DomainType.GENERAL
    target_domain: DomainType = DomainType.GENERAL
    request_description: str = ""
    required_expertise: List[DomainType] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DomainBridgeAgent:
    """Agent that facilitates cross-domain collaboration."""
    
    def __init__(self, agent_id: str, source_domain: DomainType, target_domain: DomainType):
        self.agent_id = agent_id
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.translation_rules: List[TranslationRule] = []
        self.knowledge_cache: Dict[str, DomainKnowledge] = {}
        self.access_lock = threading.RLock()
    
    def translate_concept(self, concept: str, translation_type: TranslationType = TranslationType.TERMINOLOGY) -> str:
        """Translate a concept from source domain to target domain."""
        with self.access_lock:
            # Look for direct translation rules
            for rule in self.translation_rules:
                if (rule.source_concept.lower() == concept.lower() and 
                    rule.translation_type == translation_type):
                    rule.usage_count += 1
                    rule.last_used = datetime.now()
                    return rule.target_concept
            
            # If no direct rule, try to find similar concepts
            # This is a simplified approach - in a real system, this would use more sophisticated matching
            for rule in self.translation_rules:
                if concept.lower() in rule.source_concept.lower() or rule.source_concept.lower() in concept.lower():
                    rule.usage_count += 1
                    rule.last_used = datetime.now()
                    return rule.target_concept
            
            # If no translation found, return the original concept with a note
            return f"[UNTRANSLATED:{concept}]"
    
    def add_translation_rule(self, rule: TranslationRule):
        """Add a translation rule to the bridge."""
        with self.access_lock:
            self.translation_rules.append(rule)
    
    def get_domain_knowledge(self, concept: str) -> Optional[DomainKnowledge]:
        """Get knowledge about a concept in the target domain."""
        with self.access_lock:
            # Check cache first
            cache_key = f"{self.target_domain.value}_{concept.lower()}"
            if cache_key in self.knowledge_cache:
                return self.knowledge_cache[cache_key]
            
            # In a real system, this would query a knowledge base
            # For now, return None
            return None


class KnowledgeTranslationSystem:
    """System for translating knowledge between domains."""
    
    def __init__(self, db_path: str = "domain_knowledge.db"):
        self.db_path = db_path
        self.translation_rules: Dict[str, TranslationRule] = {}
        self.domain_knowledge: Dict[str, DomainKnowledge] = {}
        self.bridge_agents: Dict[Tuple[DomainType, DomainType], DomainBridgeAgent] = {}
        self.access_lock = threading.RLock()
        
        # Initialize database
        self._init_db()
        
        # Initialize default translations
        self._init_default_translations()
    
    def _init_db(self):
        """Initialize the domain knowledge database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create domain_knowledge table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS domain_knowledge (
                knowledge_id TEXT PRIMARY KEY,
                domain TEXT,
                concept TEXT,
                definition TEXT,
                terminology_variants TEXT,
                related_concepts TEXT,
                usage_examples TEXT,
                confidence_score REAL,
                last_updated TEXT,
                metadata TEXT
            )
        ''')
        
        # Create translation_rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translation_rules (
                rule_id TEXT PRIMARY KEY,
                source_domain TEXT,
                target_domain TEXT,
                source_concept TEXT,
                target_concept TEXT,
                translation_type TEXT,
                translation_mapping TEXT,
                confidence_score REAL,
                usage_count INTEGER,
                last_used TEXT,
                created_at TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON domain_knowledge(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_concept ON domain_knowledge(concept)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rule_source_target ON translation_rules(source_domain, target_domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rule_concept ON translation_rules(source_concept)')
        
        conn.commit()
        conn.close()
    
    def _init_default_translations(self):
        """Initialize default translation rules."""
        default_rules = [
            # Data Science to Web Development
            TranslationRule(
                source_domain=DomainType.DATA_SCIENCE,
                target_domain=DomainType.WEB_DEVELOPMENT,
                source_concept="data visualization",
                target_concept="chart implementation",
                translation_type=TranslationType.CONCEPTUAL,
                translation_mapping={"matplotlib": "Chart.js", "seaborn": "D3.js", "plotly": "Plotly.js"},
                confidence_score=0.8
            ),
            TranslationRule(
                source_domain=DomainType.DATA_SCIENCE,
                target_domain=DomainType.WEB_DEVELOPMENT,
                source_concept="data processing",
                target_concept="data handling",
                translation_type=TranslationType.PROCEDURAL,
                translation_mapping={"pandas": "data processing libraries", "numpy": "mathematical computations"},
                confidence_score=0.7
            ),
            
            # Web Development to Data Science
            TranslationRule(
                source_domain=DomainType.WEB_DEVELOPMENT,
                target_domain=DomainType.DATA_SCIENCE,
                source_concept="API integration",
                target_concept="data ingestion",
                translation_type=TranslationType.CONCEPTUAL,
                translation_mapping={"REST API": "data source", "GraphQL": "structured data query"},
                confidence_score=0.8
            ),
            
            # Security to DevOps
            TranslationRule(
                source_domain=DomainType.SECURITY_ANALYSIS,
                target_domain=DomainType.DEVOPS,
                source_concept="vulnerability scan",
                target_concept="security pipeline",
                translation_type=TranslationType.PROCEDURAL,
                translation_mapping={"OWASP ZAP": "security scanning tool", "Nessus": "vulnerability scanner"},
                confidence_score=0.9
            ),
            
            # DevOps to Security
            TranslationRule(
                source_domain=DomainType.DEVOPS,
                target_domain=DomainType.SECURITY_ANALYSIS,
                source_concept="CI/CD pipeline",
                target_concept="secure deployment process",
                translation_type=TranslationType.CONCEPTUAL,
                translation_mapping={"Jenkins": "automation server", "Docker": "containerization platform"},
                confidence_score=0.85
            )
        ]
        
        for rule in default_rules:
            self.add_translation_rule(rule)
    
    def add_domain_knowledge(self, knowledge: DomainKnowledge):
        """Add knowledge to a specific domain."""
        with self.access_lock:
            # Store in memory
            self.domain_knowledge[knowledge.knowledge_id] = knowledge
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO domain_knowledge
                (knowledge_id, domain, concept, definition, terminology_variants, related_concepts, usage_examples, confidence_score, last_updated, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                knowledge.knowledge_id,
                knowledge.domain.value,
                knowledge.concept,
                knowledge.definition,
                json.dumps(knowledge.terminology_variants),
                json.dumps(knowledge.related_concepts),
                json.dumps(knowledge.usage_examples),
                knowledge.confidence_score,
                knowledge.last_updated.isoformat(),
                json.dumps(knowledge.metadata)
            ))
            
            conn.commit()
            conn.close()
    
    def add_translation_rule(self, rule: TranslationRule):
        """Add a translation rule between domains."""
        with self.access_lock:
            # Store in memory
            self.translation_rules[rule.rule_id] = rule
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO translation_rules
                (rule_id, source_domain, target_domain, source_concept, target_concept, translation_type, translation_mapping, confidence_score, usage_count, last_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id,
                rule.source_domain.value,
                rule.target_domain.value,
                rule.source_concept,
                rule.target_concept,
                rule.translation_type.value,
                json.dumps(rule.translation_mapping),
                rule.confidence_score,
                rule.usage_count,
                rule.last_used.isoformat() if rule.last_used else None,
                rule.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
    
    def get_bridge_agent(self, source_domain: DomainType, target_domain: DomainType) -> DomainBridgeAgent:
        """Get or create a bridge agent for domain translation."""
        with self.access_lock:
            key = (source_domain, target_domain)
            if key not in self.bridge_agents:
                agent_id = f"bridge_{source_domain.value}_to_{target_domain.value}"
                bridge_agent = DomainBridgeAgent(agent_id, source_domain, target_domain)
                
                # Load relevant translation rules
                for rule in self.translation_rules.values():
                    if rule.source_domain == source_domain and rule.target_domain == target_domain:
                        bridge_agent.add_translation_rule(rule)
                
                self.bridge_agents[key] = bridge_agent
            
            return self.bridge_agents[key]
    
    def translate_request(self, request: CrossDomainRequest) -> str:
        """Translate a cross-domain request and coordinate expertise."""
        # Get bridge agents for required domains
        bridge_agents = []
        for domain in request.required_expertise:
            if domain != request.source_domain:
                bridge_agent = self.get_bridge_agent(request.source_domain, domain)
                bridge_agents.append(bridge_agent)
        
        # Translate the request description using bridge agents
        translated_descriptions = []
        for bridge_agent in bridge_agents:
            # This is a simplified translation - in reality, this would be more sophisticated
            translated_desc = self._translate_text(request.request_description, bridge_agent)
            translated_descriptions.append({
                'domain': bridge_agent.target_domain.value,
                'translated_request': translated_desc
            })
        
        # Combine all translations into a comprehensive request
        result = f"Original request: {request.request_description}\n"
        result += f"Source domain: {request.source_domain.value}\n"
        result += f"Target domains: {[d.value for d in request.required_expertise]}\n\n"
        
        for trans_desc in translated_descriptions:
            result += f"Translated for {trans_desc['domain']}: {trans_desc['translated_request']}\n"
        
        return result
    
    def _translate_text(self, text: str, bridge_agent: DomainBridgeAgent) -> str:
        """Translate text using a bridge agent."""
        # Simple keyword replacement approach
        translated_text = text
        for rule in bridge_agent.translation_rules:
            if rule.source_concept.lower() in text.lower():
                translated_text = re.sub(
                    r'\b' + re.escape(rule.source_concept) + r'\b',
                    rule.target_concept,
                    translated_text,
                    flags=re.IGNORECASE
                )
        
        return translated_text
    
    def get_cross_domain_knowledge(self, concept: str, source_domain: DomainType, 
                                 target_domain: DomainType) -> List[DomainKnowledge]:
        """Get knowledge about a concept across domains."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find knowledge in source domain
        cursor.execute('SELECT * FROM domain_knowledge WHERE domain = ? AND concept LIKE ?', 
                      (source_domain.value, f'%{concept}%'))
        source_results = cursor.fetchall()
        
        # Find knowledge in target domain
        cursor.execute('SELECT * FROM domain_knowledge WHERE domain = ? AND concept LIKE ?', 
                      (target_domain.value, f'%{concept}%'))
        target_results = cursor.fetchall()
        
        conn.close()
        
        knowledge_items = []
        
        # Convert source results
        for row in source_results:
            knowledge = DomainKnowledge(
                knowledge_id=row[0],
                domain=DomainType(row[1]),
                concept=row[2],
                definition=row[3],
                terminology_variants=json.loads(row[4]) if row[4] else [],
                related_concepts=json.loads(row[5]) if row[5] else [],
                usage_examples=json.loads(row[6]) if row[6] else [],
                confidence_score=row[7],
                last_updated=datetime.fromisoformat(row[8]),
                metadata=json.loads(row[9]) if row[9] else {}
            )
            knowledge_items.append(knowledge)
        
        # Convert target results
        for row in target_results:
            knowledge = DomainKnowledge(
                knowledge_id=row[0],
                domain=DomainType(row[1]),
                concept=row[2],
                definition=row[3],
                terminology_variants=json.loads(row[4]) if row[4] else [],
                related_concepts=json.loads(row[5]) if row[5] else [],
                usage_examples=json.loads(row[6]) if row[6] else [],
                confidence_score=row[7],
                last_updated=datetime.fromisoformat(row[8]),
                metadata=json.loads(row[9]) if row[9] else {}
            )
            knowledge_items.append(knowledge)
        
        return knowledge_items


class CrossDomainCoordinator:
    """Coordinates cross-domain expertise combination."""
    
    def __init__(self, knowledge_translation_system: KnowledgeTranslationSystem):
        self.knowledge_translation_system = knowledge_translation_system
        self.active_requests: Dict[str, CrossDomainRequest] = {}
        self.access_lock = threading.RLock()
    
    def create_cross_domain_request(self, requesting_agent: str, source_domain: DomainType,
                                 target_domain: DomainType, request_description: str,
                                 required_expertise: List[DomainType],
                                 context: Dict[str, Any] = None,
                                 priority: int = 1) -> str:
        """Create a request for cross-domain expertise combination."""
        context = context or {}
        
        request = CrossDomainRequest(
            requesting_agent=requesting_agent,
            source_domain=source_domain,
            target_domain=target_domain,
            request_description=request_description,
            required_expertise=required_expertise,
            context=context,
            priority=priority
        )
        
        with self.access_lock:
            self.active_requests[request.request_id] = request
        
        return request.request_id
    
    def process_cross_domain_request(self, request_id: str) -> str:
        """Process a cross-domain request and return the result."""
        with self.access_lock:
            if request_id not in self.active_requests:
                return "Request not found"
            
            request = self.active_requests[request_id]
            request.status = "in_progress"
        
        # Use the knowledge translation system to process the request
        result = self.knowledge_translation_system.translate_request(request)
        
        # Update request status
        with self.access_lock:
            request = self.active_requests[request_id]
            request.status = "completed"
            request.result = result
        
        return result
    
    def get_domain_expertise_map(self, domains: List[DomainType]) -> Dict[str, List[str]]:
        """Get a map of expertise across domains."""
        expertise_map = {}
        
        for domain in domains:
            # In a real system, this would query the knowledge base
            # For now, we'll return some sample expertise
            expertise_samples = {
                DomainType.DATA_SCIENCE: ["data analysis", "machine learning", "statistical modeling", "visualization"],
                DomainType.WEB_DEVELOPMENT: ["frontend", "backend", "API development", "database design"],
                DomainType.SECURITY_ANALYSIS: ["vulnerability assessment", "penetration testing", "secure coding", "compliance"],
                DomainType.DEVOPS: ["CI/CD", "containerization", "infrastructure", "monitoring"],
                DomainType.TESTING: ["unit testing", "integration testing", "automation", "quality assurance"],
                DomainType.DOCUMENTATION: ["technical writing", "knowledge management", "tutorials", "guides"]
            }
            
            expertise_map[domain.value] = expertise_samples.get(domain, ["general expertise"])
        
        return expertise_map


class HybridReasoningEngine:
    """Engine for combining reasoning across different domains."""
    
    def __init__(self, cross_domain_coordinator: CrossDomainCoordinator):
        self.cross_domain_coordinator = cross_domain_coordinator
        self.reasoning_strategies: Dict[str, callable] = {}
        self.access_lock = threading.RLock()
    
    def register_reasoning_strategy(self, strategy_name: str, strategy_func: callable):
        """Register a reasoning strategy."""
        with self.access_lock:
            self.reasoning_strategies[strategy_name] = strategy_func
    
    def combine_reasoning(self, problem_description: str, domains: List[DomainType]) -> str:
        """Combine reasoning from multiple domains to solve a problem."""
        # Get expertise map
        expertise_map = self.cross_domain_coordinator.get_domain_expertise_map(domains)
        
        # Create a cross-domain request to gather relevant knowledge
        request_id = self.cross_domain_coordinator.create_cross_domain_request(
            requesting_agent="hybrid_reasoning_engine",
            source_domain=domains[0] if domains else DomainType.GENERAL,
            target_domain=domains[-1] if domains else DomainType.GENERAL,
            request_description=problem_description,
            required_expertise=domains
        )
        
        # Process the request
        result = self.cross_domain_coordinator.process_cross_domain_request(request_id)
        
        # Combine the results with domain-specific reasoning
        combined_reasoning = f"Problem: {problem_description}\n\n"
        combined_reasoning += "Cross-domain analysis:\n"
        combined_reasoning += result
        combined_reasoning += "\n\nDomain expertise considered:\n"
        
        for domain in domains:
            expertise = expertise_map.get(domain.value, [])
            combined_reasoning += f"- {domain.value}: {', '.join(expertise[:3])}\n"
        
        # Add a conclusion based on combined reasoning
        combined_reasoning += "\nConclusion: This problem requires integrated expertise from multiple domains. "
        combined_reasoning += "The solution should consider both technical implementation aspects "
        combined_reasoning += "and domain-specific constraints."
        
        return combined_reasoning


class CrossDomainExpertiseSystem:
    """Main system for cross-domain expertise combination."""
    
    def __init__(self, db_path: str = "domain_knowledge.db"):
        self.knowledge_translation_system = KnowledgeTranslationSystem(db_path)
        self.cross_domain_coordinator = CrossDomainCoordinator(self.knowledge_translation_system)
        self.hybrid_reasoning_engine = HybridReasoningEngine(self.cross_domain_coordinator)
        self.access_lock = threading.RLock()
    
    def request_cross_domain_expertise(self, requesting_agent: str, source_domain: DomainType,
                                    target_domains: List[DomainType], problem_description: str,
                                    context: Dict[str, Any] = None) -> str:
        """Request expertise combination across multiple domains."""
        context = context or {}
        
        # Create and process the request
        request_id = self.cross_domain_coordinator.create_cross_domain_request(
            requesting_agent=requesting_agent,
            source_domain=source_domain,
            target_domain=target_domains[-1] if target_domains else source_domain,
            request_description=problem_description,
            required_expertise=target_domains,
            context=context
        )
        
        result = self.cross_domain_coordinator.process_cross_domain_request(request_id)
        return result
    
    def get_domain_bridge(self, source_domain: DomainType, target_domain: DomainType) -> DomainBridgeAgent:
        """Get a bridge agent for translating between domains."""
        return self.knowledge_translation_system.get_bridge_agent(source_domain, target_domain)
    
    def add_domain_knowledge(self, knowledge: DomainKnowledge):
        """Add knowledge to a specific domain."""
        self.knowledge_translation_system.add_domain_knowledge(knowledge)
    
    def add_translation_rule(self, rule: TranslationRule):
        """Add a translation rule between domains."""
        self.knowledge_translation_system.add_translation_rule(rule)
    
    def perform_hybrid_reasoning(self, problem_description: str, domains: List[DomainType]) -> str:
        """Perform reasoning that combines multiple domains."""
        return self.hybrid_reasoning_engine.combine_reasoning(problem_description, domains)
    
    def get_cross_domain_insights(self, concept: str, domains: List[DomainType]) -> Dict[str, Any]:
        """Get insights about a concept across multiple domains."""
        insights = {
            'concept': concept,
            'domains': [d.value for d in domains],
            'translations': {},
            'domain_knowledge': {}
        }
        
        for domain in domains:
            # Get knowledge in this domain
            knowledge_items = self.knowledge_translation_system.get_cross_domain_knowledge(
                concept, domain, domain
            )
            insights['domain_knowledge'][domain.value] = [
                {
                    'concept': ki.concept,
                    'definition': ki.definition,
                    'confidence': ki.confidence_score
                } for ki in knowledge_items
            ]
        
        # Get translations between domains
        if len(domains) >= 2:
            for i in range(len(domains)-1):
                source_domain = domains[i]
                target_domain = domains[i+1]
                
                bridge_agent = self.get_domain_bridge(source_domain, target_domain)
                # For now, just note that a bridge exists
                insights['translations'][f"{source_domain.value}_to_{target_domain.value}"] = "Bridge available"
        
        return insights


# Helper functions for common operations
def create_domain_knowledge(domain: DomainType, concept: str, definition: str) -> DomainKnowledge:
    """Create domain knowledge."""
    return DomainKnowledge(
        domain=domain,
        concept=concept,
        definition=definition,
        confidence_score=0.8
    )


def create_translation_rule(source_domain: DomainType, target_domain: DomainType, 
                          source_concept: str, target_concept: str) -> TranslationRule:
    """Create a translation rule."""
    return TranslationRule(
        source_domain=source_domain,
        target_domain=target_domain,
        source_concept=source_concept,
        target_concept=target_concept,
        confidence_score=0.7
    )


def get_cross_domain_solution(problem: str, domains: List[DomainType]) -> str:
    """Get a solution that combines multiple domains."""
    system = CrossDomainExpertiseSystem()
    return system.perform_hybrid_reasoning(problem, domains)