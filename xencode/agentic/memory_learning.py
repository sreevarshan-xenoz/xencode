"""
Memory and learning system for multi-agent systems in Xencode
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
import pickle
import hashlib
from collections import defaultdict


class MemoryType(Enum):
    """Types of memory in the agent system."""
    EPISODIC = "episodic"  # Specific events and experiences
    SEMANTIC = "semantic"  # Factual knowledge and concepts
    PROCEDURAL = "procedural"  # Skills and procedures
    WORKING = "working"     # Short-term memory for current tasks
    LONG_TERM = "long_term" # Long-term knowledge storage


class KnowledgeSourceType(Enum):
    """Sources of knowledge in the system."""
    PERSONAL_EXPERIENCE = "personal_experience"
    SHARED_KNOWLEDGE = "shared_knowledge"
    TRAINING_DATA = "training_data"
    EXTERNAL_SOURCE = "external_source"


@dataclass
class MemoryEntry:
    """Represents a single memory entry."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: KnowledgeSourceType = KnowledgeSourceType.PERSONAL_EXPERIENCE
    tags: Set[str] = field(default_factory=set)
    confidence: float = 1.0  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge in the shared knowledge base."""
    knowledge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    source_agents: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    category: str = ""
    creation_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    quality_score: float = 0.5  # 0.0 to 1.0
    usage_count: int = 0
    verified: bool = False


@dataclass
class LearningPattern:
    """Represents a learned pattern from historical tasks."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    trigger_conditions: List[Dict[str, Any]] = field(default_factory=list)
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)
    success_rate: float = 0.0  # 0.0 to 1.0
    last_used: Optional[datetime] = None
    created_date: datetime = field(default_factory=datetime.now)


class AgentMemory:
    """Individual agent memory system."""
    
    def __init__(self, agent_id: str, db_path: Optional[str] = None):
        self.agent_id = agent_id
        self.db_path = db_path or f"agent_{agent_id}_memory.db"
        self.local_memory: Dict[str, MemoryEntry] = {}
        self.access_lock = threading.RLock()
        
        # Initialize database
        self._init_db()
        
    def _init_db(self):
        """Initialize the memory database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                agent_id TEXT,
                memory_type TEXT,
                content TEXT,
                metadata TEXT,
                timestamp TEXT,
                source TEXT,
                tags TEXT,
                confidence REAL,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_memories ON memories(agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)')
        
        conn.commit()
        conn.close()
    
    def store_memory(self, memory_entry: MemoryEntry):
        """Store a memory entry."""
        with self.access_lock:
            # Update the in-memory cache
            self.local_memory[memory_entry.memory_id] = memory_entry
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO memories 
                (memory_id, agent_id, memory_type, content, metadata, timestamp, source, tags, confidence, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory_entry.memory_id,
                memory_entry.agent_id,
                memory_entry.memory_type.value,
                memory_entry.content,
                json.dumps(memory_entry.metadata),
                memory_entry.timestamp.isoformat(),
                memory_entry.source.value,
                json.dumps(list(memory_entry.tags)),
                memory_entry.confidence,
                memory_entry.access_count,
                memory_entry.last_accessed.isoformat() if memory_entry.last_accessed else None
            ))
            
            conn.commit()
            conn.close()
    
    def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory."""
        with self.access_lock:
            # Check local cache first
            if memory_id in self.local_memory:
                memory = self.local_memory[memory_id]
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                self.store_memory(memory)  # Update access count
                return memory
            
            # Check database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM memories WHERE memory_id = ?', (memory_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                memory = MemoryEntry(
                    memory_id=row[0],
                    agent_id=row[1],
                    memory_type=MemoryType(row[2]),
                    content=row[3],
                    metadata=json.loads(row[4]) if row[4] else {},
                    timestamp=datetime.fromisoformat(row[5]),
                    source=KnowledgeSourceType(row[6]),
                    tags=set(json.loads(row[7])) if row[7] else set(),
                    confidence=row[8],
                    access_count=row[9],
                    last_accessed=datetime.fromisoformat(row[10]) if row[10] else None
                )
                
                # Update access count
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                self.store_memory(memory)
                
                # Cache in local memory
                self.local_memory[memory_id] = memory
                return memory
            
            return None
    
    def search_memories(self, query: str = "", memory_type: Optional[MemoryType] = None, 
                       tags: Optional[Set[str]] = None, limit: int = 10) -> List[MemoryEntry]:
        """Search for memories based on criteria."""
        with self.access_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            conditions = ["agent_id = ?"]
            params = [self.agent_id]
            
            if query:
                conditions.append("content LIKE ?")
                params.append(f"%{query}%")
                
            if memory_type:
                conditions.append("memory_type = ?")
                params.append(memory_type.value)
                
            if tags:
                # This is a simplified tag search - in practice, you'd want more sophisticated tag matching
                conditions.append("tags LIKE ?")
                params.append(f"%{list(tags)[0]}%" if tags else "%")
            
            where_clause = " AND ".join(conditions)
            query_sql = f"SELECT * FROM memories WHERE {where_clause} ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query_sql, params)
            rows = cursor.fetchall()
            conn.close()
            
            memories = []
            for row in rows:
                memory = MemoryEntry(
                    memory_id=row[0],
                    agent_id=row[1],
                    memory_type=MemoryType(row[2]),
                    content=row[3],
                    metadata=json.loads(row[4]) if row[4] else {},
                    timestamp=datetime.fromisoformat(row[5]),
                    source=KnowledgeSourceType(row[6]),
                    tags=set(json.loads(row[7])) if row[7] else set(),
                    confidence=row[8],
                    access_count=row[9],
                    last_accessed=datetime.fromisoformat(row[10]) if row[10] else None
                )
                memories.append(memory)
                
                # Cache in local memory
                self.local_memory[memory.memory_id] = memory
            
            return memories
    
    def forget_memory(self, memory_id: str) -> bool:
        """Forget a specific memory."""
        with self.access_lock:
            # Remove from local cache
            if memory_id in self.local_memory:
                del self.local_memory[memory_id]
            
            # Remove from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            deleted = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            return deleted


class SharedKnowledgeBase:
    """Shared knowledge base accessible by all agents."""
    
    def __init__(self, db_path: str = "shared_knowledge.db"):
        self.db_path = db_path
        self.access_lock = threading.RLock()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize the shared knowledge database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create knowledge table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_items (
                knowledge_id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                source_agents TEXT,
                tags TEXT,
                category TEXT,
                creation_date TEXT,
                last_updated TEXT,
                quality_score REAL,
                usage_count INTEGER DEFAULT 0,
                verified BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON knowledge_items(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags ON knowledge_items(tags)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality ON knowledge_items(quality_score)')
        
        conn.commit()
        conn.close()
    
    def add_knowledge(self, knowledge_item: KnowledgeItem):
        """Add a knowledge item to the shared base."""
        with self.access_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO knowledge_items
                (knowledge_id, title, content, source_agents, tags, category, creation_date, last_updated, quality_score, usage_count, verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                knowledge_item.knowledge_id,
                knowledge_item.title,
                knowledge_item.content,
                json.dumps(list(knowledge_item.source_agents)),
                json.dumps(list(knowledge_item.tags)),
                knowledge_item.category,
                knowledge_item.creation_date.isoformat(),
                knowledge_item.last_updated.isoformat(),
                knowledge_item.quality_score,
                knowledge_item.usage_count,
                knowledge_item.verified
            ))
            
            conn.commit()
            conn.close()
    
    def get_knowledge(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """Retrieve a specific knowledge item."""
        with self.access_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM knowledge_items WHERE knowledge_id = ?', (knowledge_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return KnowledgeItem(
                    knowledge_id=row[0],
                    title=row[1],
                    content=row[2],
                    source_agents=set(json.loads(row[3])) if row[3] else set(),
                    tags=set(json.loads(row[4])) if row[4] else set(),
                    category=row[5],
                    creation_date=datetime.fromisoformat(row[6]),
                    last_updated=datetime.fromisoformat(row[7]),
                    quality_score=row[8],
                    usage_count=row[9],
                    verified=bool(row[10])
                )
            
            return None
    
    def search_knowledge(self, query: str = "", category: str = "", tags: Optional[Set[str]] = None, 
                        min_quality: float = 0.0, limit: int = 10) -> List[KnowledgeItem]:
        """Search for knowledge items based on criteria."""
        with self.access_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if query:
                conditions.append("content LIKE ? OR title LIKE ?")
                params.extend([f"%{query}%", f"%{query}%"])
                
            if category:
                conditions.append("category = ?")
                params.append(category)
                
            if tags:
                # Simplified tag search
                for tag in tags:
                    conditions.append("tags LIKE ?")
                    params.append(f"%{tag}%")
            
            conditions.append("quality_score >= ?")
            params.append(min_quality)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query_sql = f"SELECT * FROM knowledge_items WHERE {where_clause} ORDER BY quality_score DESC, usage_count DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query_sql, params)
            rows = cursor.fetchall()
            conn.close()
            
            knowledge_items = []
            for row in rows:
                knowledge_items.append(KnowledgeItem(
                    knowledge_id=row[0],
                    title=row[1],
                    content=row[2],
                    source_agents=set(json.loads(row[3])) if row[3] else set(),
                    tags=set(json.loads(row[4])) if row[4] else set(),
                    category=row[5],
                    creation_date=datetime.fromisoformat(row[6]),
                    last_updated=datetime.fromisoformat(row[7]),
                    quality_score=row[8],
                    usage_count=row[9],
                    verified=bool(row[10])
                ))
            
            return knowledge_items
    
    def update_knowledge_usage(self, knowledge_id: str, increment: int = 1):
        """Increment the usage count of a knowledge item."""
        with self.access_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('UPDATE knowledge_items SET usage_count = usage_count + ? WHERE knowledge_id = ?',
                          (increment, knowledge_id))
            
            conn.commit()
            conn.close()
    
    def update_quality_score(self, knowledge_id: str, new_score: float):
        """Update the quality score of a knowledge item."""
        with self.access_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('UPDATE knowledge_items SET quality_score = ?, last_updated = ? WHERE knowledge_id = ?',
                          (new_score, datetime.now().isoformat(), knowledge_id))
            
            conn.commit()
            conn.close()


class ExperienceSharingSystem:
    """System for sharing experiences between agents."""
    
    def __init__(self, shared_knowledge_base: SharedKnowledgeBase):
        self.shared_knowledge_base = shared_knowledge_base
        self.experience_queue: List[Tuple[str, KnowledgeItem]] = []  # (agent_id, knowledge_item)
        self.access_lock = threading.RLock()
    
    def share_experience(self, agent_id: str, experience: str, tags: Set[str], 
                        category: str = "experience") -> str:
        """Share an experience from an agent to the shared knowledge base."""
        with self.access_lock:
            knowledge_item = KnowledgeItem(
                title=f"Experience from {agent_id}",
                content=experience,
                source_agents={agent_id},
                tags=tags,
                category=category,
                quality_score=0.7  # Default quality for shared experiences
            )
            
            self.shared_knowledge_base.add_knowledge(knowledge_item)
            return knowledge_item.knowledge_id
    
    def get_shared_experiences(self, agent_id: str, tags: Optional[Set[str]] = None, 
                              category: str = "experience", limit: int = 10) -> List[KnowledgeItem]:
        """Get shared experiences relevant to an agent."""
        return self.shared_knowledge_base.search_knowledge(
            query="", 
            category=category, 
            tags=tags, 
            min_quality=0.3, 
            limit=limit
        )
    
    def get_agent_reputation(self, agent_id: str) -> float:
        """Calculate an agent's reputation based on quality of shared knowledge."""
        all_knowledge = self.shared_knowledge_base.search_knowledge()
        agent_knowledge = [k for k in all_knowledge if agent_id in k.source_agents]
        
        if not agent_knowledge:
            return 0.5  # Default reputation
        
        avg_quality = sum(k.quality_score for k in agent_knowledge) / len(agent_knowledge)
        return avg_quality


class HistoricalTaskPatterns:
    """System for recognizing and learning from historical task patterns."""
    
    def __init__(self):
        self.patterns: Dict[str, LearningPattern] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.access_lock = threading.RLock()
    
    def record_task_completion(self, task_description: str, agent_id: str, 
                             result: str, success: bool, execution_time: float):
        """Record a completed task for pattern analysis."""
        with self.access_lock:
            task_record = {
                'task_id': str(uuid.uuid4()),
                'task_description': task_description,
                'agent_id': agent_id,
                'result': result,
                'success': success,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            self.task_history.append(task_record)
            
            # Analyze for potential patterns
            self._analyze_patterns(task_record)
    
    def _analyze_patterns(self, task_record: Dict[str, Any]):
        """Analyze task records for potential patterns."""
        # This is a simplified pattern analysis
        # In a real implementation, this would use more sophisticated ML techniques
        
        # Look for similar task descriptions
        similar_tasks = [
            tr for tr in self.task_history 
            if tr != task_record and 
            self._calculate_similarity(tr['task_description'], task_record['task_description']) > 0.7
        ]
        
        if len(similar_tasks) >= 3:  # At least 3 similar tasks to form a pattern
            # Create or update a pattern
            pattern_key = self._generate_pattern_key(task_record['task_description'])
            
            if pattern_key not in self.patterns:
                # Create new pattern
                pattern = LearningPattern(
                    name=f"Pattern for: {task_record['task_description'][:50]}...",
                    description=f"Tasks similar to: {task_record['task_description']}",
                    trigger_conditions=[{
                        'type': 'similarity',
                        'threshold': 0.7,
                        'reference_text': task_record['task_description']
                    }],
                    recommended_actions=[{
                        'action': 'delegate_to_agent',
                        'agent_type': task_record['agent_id'],
                        'confidence': 0.8
                    }]
                )
                self.patterns[pattern_key] = pattern
            
            # Update pattern statistics
            pattern = self.patterns[pattern_key]
            successful_tasks = [tr for tr in similar_tasks if tr['success']]
            pattern.success_rate = len(successful_tasks) / len(similar_tasks)
            pattern.last_used = datetime.now()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simplified)."""
        # Simplified similarity calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 1.0 if not text1 and not text2 else 0.0
        
        return len(intersection) / len(union)
    
    def _generate_pattern_key(self, text: str) -> str:
        """Generate a key for a pattern based on the text."""
        # Use hash of the first 100 characters as a key
        return hashlib.md5(text[:100].lower().encode()).hexdigest()
    
    def get_applicable_patterns(self, current_task: str) -> List[Tuple[LearningPattern, float]]:
        """Get patterns applicable to the current task."""
        applicable = []
        
        for pattern in self.patterns.values():
            for condition in pattern.trigger_conditions:
                if condition['type'] == 'similarity':
                    similarity = self._calculate_similarity(condition['reference_text'], current_task)
                    if similarity >= condition['threshold']:
                        applicable.append((pattern, similarity))
        
        # Sort by similarity score
        applicable.sort(key=lambda x: x[1], reverse=True)
        return applicable
    
    def get_task_recommendation(self, current_task: str) -> Optional[Dict[str, Any]]:
        """Get recommendations for the current task based on historical patterns."""
        applicable_patterns = self.get_applicable_patterns(current_task)
        
        if applicable_patterns:
            best_pattern, similarity = applicable_patterns[0]
            
            if best_pattern.success_rate > 0.6:  # Only recommend if success rate is high enough
                return {
                    'pattern_id': best_pattern.pattern_id,
                    'recommendation': best_pattern.recommended_actions[0]['action'],
                    'confidence': best_pattern.success_rate,
                    'similarity': similarity
                }
        
        return None


class AgentLearningSystem:
    """Main system for agent memory and learning."""
    
    def __init__(self, shared_knowledge_db: str = "shared_knowledge.db"):
        self.shared_knowledge_base = SharedKnowledgeBase(shared_knowledge_db)
        self.experience_sharing = ExperienceSharingSystem(self.shared_knowledge_base)
        self.historical_patterns = HistoricalTaskPatterns()
        self.agent_memories: Dict[str, AgentMemory] = {}
        self.access_lock = threading.RLock()
    
    def get_agent_memory(self, agent_id: str) -> AgentMemory:
        """Get or create memory for an agent."""
        with self.access_lock:
            if agent_id not in self.agent_memories:
                self.agent_memories[agent_id] = AgentMemory(agent_id)
            return self.agent_memories[agent_id]
    
    def share_knowledge_between_agents(self, source_agent: str, target_agent: str, 
                                     knowledge_content: str, tags: Set[str]) -> str:
        """Share knowledge from one agent to another via the shared base."""
        knowledge_id = self.experience_sharing.share_experience(
            source_agent, knowledge_content, tags, "cross_agent_learning"
        )
        
        # Update target agent's local memory with the shared knowledge
        target_memory = self.get_agent_memory(target_agent)
        memory_entry = MemoryEntry(
            agent_id=target_agent,
            memory_type=MemoryType.SEMANTIC,
            content=knowledge_content,
            metadata={'source_agent': source_agent, 'shared_via': 'cross_agent_learning'},
            tags=tags,
            source=KnowledgeSourceType.SHARED_KNOWLEDGE
        )
        target_memory.store_memory(memory_entry)
        
        return knowledge_id
    
    def learn_from_task(self, agent_id: str, task_description: str, result: str, 
                       success: bool, execution_time: float):
        """Learn from a completed task."""
        # Record in historical patterns
        self.historical_patterns.record_task_completion(
            task_description, agent_id, result, success, execution_time
        )
        
        # Optionally share successful experiences
        if success and len(result) > 50:  # Only share substantial results
            self.experience_sharing.share_experience(
                agent_id, 
                f"Successfully completed task: {task_description}. Result: {result}", 
                {"task_result", "success", agent_id}
            )
    
    def get_learning_recommendation(self, agent_id: str, current_task: str) -> Optional[Dict[str, Any]]:
        """Get learning-based recommendations for a task."""
        # Get pattern-based recommendations
        pattern_rec = self.historical_patterns.get_task_recommendation(current_task)
        
        # Get knowledge-based recommendations
        relevant_knowledge = self.shared_knowledge_base.search_knowledge(
            query=current_task.split()[0] if current_task.split() else "",  # Use first word as query
            min_quality=0.5,
            limit=5
        )
        
        return {
            'pattern_recommendation': pattern_rec,
            'relevant_knowledge': relevant_knowledge,
            'agent_reputation': self.experience_sharing.get_agent_reputation(agent_id)
        }


# Helper functions for common operations
def create_memory_from_task_result(agent_id: str, task: str, result: str, 
                                 success: bool = True) -> MemoryEntry:
    """Create a memory entry from a task result."""
    return MemoryEntry(
        agent_id=agent_id,
        memory_type=MemoryType.EPISODIC,
        content=f"Task: {task}\nResult: {result}",
        metadata={'task_success': success, 'result_length': len(result)},
        tags={'task_result', 'experience'},
        source=KnowledgeSourceType.PERSONAL_EXPERIENCE
    )


def create_knowledge_from_solution(title: str, solution: str, tags: Set[str]) -> KnowledgeItem:
    """Create a knowledge item from a solution."""
    return KnowledgeItem(
        title=title,
        content=solution,
        tags=tags,
        category="solution",
        quality_score=0.8
    )