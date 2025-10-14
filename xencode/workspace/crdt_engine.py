#!/usr/bin/env python3
"""
CRDT Collaboration Engine

Implements Conflict-free Replicated Data Types (CRDT) for real-time
collaboration with automatic conflict resolution and consistency guarantees.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from xencode.models.workspace import Change, ChangeType, Conflict, WorkspaceFile


class VectorClock:
    """Vector clock for tracking causality in distributed systems"""
    
    def __init__(self, clock: Optional[Dict[str, int]] = None):
        self.clock = clock or {}
    
    def increment(self, node_id: str) -> None:
        """Increment clock for node"""
        self.clock[node_id] = self.clock.get(node_id, 0) + 1
    
    def update(self, other_clock: Dict[str, int]) -> None:
        """Update clock with another clock (take maximum)"""
        for node_id, timestamp in other_clock.items():
            self.clock[node_id] = max(self.clock.get(node_id, 0), timestamp)
    
    def compare(self, other_clock: Dict[str, int]) -> str:
        """Compare with another clock"""
        # Returns: 'before', 'after', 'concurrent', or 'equal'
        
        self_keys = set(self.clock.keys())
        other_keys = set(other_clock.keys())
        all_keys = self_keys | other_keys
        
        self_greater = False
        other_greater = False
        
        for key in all_keys:
            self_val = self.clock.get(key, 0)
            other_val = other_clock.get(key, 0)
            
            if self_val > other_val:
                self_greater = True
            elif other_val > self_val:
                other_greater = True
        
        if self_greater and not other_greater:
            return 'after'
        elif other_greater and not self_greater:
            return 'before'
        elif not self_greater and not other_greater:
            return 'equal'
        else:
            return 'concurrent'
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary"""
        return self.clock.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'VectorClock':
        """Create from dictionary"""
        return cls(data.copy())


class TextOperation:
    """Represents a text operation for CRDT"""
    
    def __init__(self, 
                 op_type: str,  # 'insert', 'delete', 'retain'
                 position: int,
                 content: str = "",
                 length: int = 0):
        self.op_type = op_type
        self.position = position
        self.content = content
        self.length = length if length > 0 else len(content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'op_type': self.op_type,
            'position': self.position,
            'content': self.content,
            'length': self.length
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextOperation':
        """Create from dictionary"""
        return cls(
            op_type=data['op_type'],
            position=data['position'],
            content=data.get('content', ''),
            length=data.get('length', 0)
        )


class CRDTDocument:
    """CRDT document for collaborative text editing"""
    
    def __init__(self, initial_content: str = "", node_id: str = ""):
        self.content = initial_content
        self.node_id = node_id
        self.vector_clock = VectorClock()
        self.operations: List[Tuple[TextOperation, VectorClock, str]] = []  # (op, clock, author)
        self.applied_operations: Set[str] = set()  # Operation IDs
    
    def apply_operation(self, operation: TextOperation, clock: VectorClock, author_id: str, op_id: str) -> bool:
        """Apply operation to document"""
        
        # Check if operation already applied
        if op_id in self.applied_operations:
            return False
        
        # Update vector clock
        self.vector_clock.update(clock.to_dict())
        self.vector_clock.increment(self.node_id)
        
        # Apply operation
        if operation.op_type == 'insert':
            self._apply_insert(operation)
        elif operation.op_type == 'delete':
            self._apply_delete(operation)
        
        # Record operation
        self.operations.append((operation, clock, author_id))
        self.applied_operations.add(op_id)
        
        return True
    
    def _apply_insert(self, operation: TextOperation) -> None:
        """Apply insert operation"""
        pos = max(0, min(operation.position, len(self.content)))
        self.content = self.content[:pos] + operation.content + self.content[pos:]
    
    def _apply_delete(self, operation: TextOperation) -> None:
        """Apply delete operation"""
        start = max(0, min(operation.position, len(self.content)))
        end = max(start, min(start + operation.length, len(self.content)))
        self.content = self.content[:start] + self.content[end:]
    
    def create_insert_operation(self, position: int, content: str) -> Tuple[TextOperation, VectorClock, str]:
        """Create insert operation"""
        self.vector_clock.increment(self.node_id)
        operation = TextOperation('insert', position, content)
        return operation, VectorClock(self.vector_clock.to_dict()), f"{self.node_id}_{int(time.time() * 1000)}"
    
    def create_delete_operation(self, position: int, length: int) -> Tuple[TextOperation, VectorClock, str]:
        """Create delete operation"""
        self.vector_clock.increment(self.node_id)
        operation = TextOperation('delete', position, length=length)
        return operation, VectorClock(self.vector_clock.to_dict()), f"{self.node_id}_{int(time.time() * 1000)}"
    
    def get_content(self) -> str:
        """Get current document content"""
        return self.content
    
    def get_vector_clock(self) -> Dict[str, int]:
        """Get current vector clock"""
        return self.vector_clock.to_dict()


class ConflictResolver:
    """Resolves conflicts between concurrent changes"""
    
    def __init__(self, strategy: str = "last_writer_wins"):
        self.strategy = strategy
    
    async def resolve_conflict(self, change_a: Change, change_b: Change) -> Optional[Change]:
        """Resolve conflict between two changes"""
        
        if self.strategy == "last_writer_wins":
            return await self._last_writer_wins(change_a, change_b)
        elif self.strategy == "merge":
            return await self._merge_changes(change_a, change_b)
        elif self.strategy == "manual":
            return None  # Requires manual resolution
        else:
            return await self._last_writer_wins(change_a, change_b)
    
    async def _last_writer_wins(self, change_a: Change, change_b: Change) -> Change:
        """Last writer wins strategy"""
        if change_a.timestamp > change_b.timestamp:
            return change_a
        elif change_b.timestamp > change_a.timestamp:
            return change_b
        else:
            # Same timestamp, use author ID as tiebreaker
            return change_a if change_a.author_id < change_b.author_id else change_b
    
    async def _merge_changes(self, change_a: Change, change_b: Change) -> Optional[Change]:
        """Attempt to merge changes"""
        
        # Simple merge strategy for text operations
        if (change_a.change_type == ChangeType.INSERT and 
            change_b.change_type == ChangeType.INSERT):
            
            # If insertions are at different positions, both can be applied
            if abs(change_a.position - change_b.position) > max(len(change_a.content), len(change_b.content)):
                # Create merged change
                merged_change = Change(
                    workspace_id=change_a.workspace_id,
                    file_id=change_a.file_id,
                    change_type=ChangeType.UPDATE,
                    content=f"MERGED: {change_a.content} | {change_b.content}",
                    author_id="system",
                    timestamp=datetime.now(),
                    parent_changes=[change_a.id, change_b.id]
                )
                return merged_change
        
        # If merge not possible, fall back to last writer wins
        return await self._last_writer_wins(change_a, change_b)


class CRDTEngine:
    """Main CRDT engine for conflict-free collaboration"""
    
    def __init__(self, node_id: str = "default"):
        self.node_id = node_id
        self.documents: Dict[str, CRDTDocument] = {}  # file_id -> CRDTDocument
        self.conflict_resolver = ConflictResolver()
        
        # Change tracking
        self.pending_changes: Dict[str, List[Change]] = {}  # file_id -> changes
        self.applied_changes: Dict[str, Set[str]] = {}  # file_id -> change_ids
        
        # Synchronization
        self.sync_callbacks: List[callable] = []
    
    def register_document(self, file_id: str, initial_content: str = "") -> None:
        """Register document for CRDT tracking"""
        if file_id not in self.documents:
            self.documents[file_id] = CRDTDocument(initial_content, self.node_id)
            self.pending_changes[file_id] = []
            self.applied_changes[file_id] = set()
    
    def unregister_document(self, file_id: str) -> None:
        """Unregister document"""
        if file_id in self.documents:
            del self.documents[file_id]
            del self.pending_changes[file_id]
            del self.applied_changes[file_id]
    
    async def apply_change(self, change: Change) -> Tuple[bool, Optional[Conflict]]:
        """Apply change to document"""
        
        file_id = change.file_id
        
        # Ensure document is registered
        if file_id not in self.documents:
            self.register_document(file_id)
        
        document = self.documents[file_id]
        
        # Check if change already applied
        if change.id in self.applied_changes[file_id]:
            return True, None
        
        # Check for conflicts
        conflict = await self._detect_conflict(change)
        if conflict:
            # Attempt to resolve conflict
            resolved_change = await self.conflict_resolver.resolve_conflict(
                conflict.change_a, conflict.change_b
            )
            
            if resolved_change:
                # Apply resolved change
                success = await self._apply_change_to_document(resolved_change, document)
                if success:
                    self.applied_changes[file_id].add(resolved_change.id)
                return success, None
            else:
                # Manual resolution required
                return False, conflict
        
        # Apply change directly
        success = await self._apply_change_to_document(change, document)
        if success:
            self.applied_changes[file_id].add(change.id)
        
        return success, None
    
    async def _apply_change_to_document(self, change: Change, document: CRDTDocument) -> bool:
        """Apply change to CRDT document"""
        
        try:
            if change.change_type == ChangeType.INSERT:
                operation = TextOperation('insert', change.position, change.content)
            elif change.change_type == ChangeType.DELETE:
                operation = TextOperation('delete', change.position, length=change.length)
            elif change.change_type == ChangeType.UPDATE:
                # For updates, we treat as delete + insert
                if change.length > 0:
                    delete_op = TextOperation('delete', change.position, length=change.length)
                    document.apply_operation(
                        delete_op,
                        VectorClock.from_dict(change.vector_clock),
                        change.author_id,
                        f"{change.id}_delete"
                    )
                
                if change.content:
                    insert_op = TextOperation('insert', change.position, change.content)
                    document.apply_operation(
                        insert_op,
                        VectorClock.from_dict(change.vector_clock),
                        change.author_id,
                        f"{change.id}_insert"
                    )
                
                return True
            else:
                return False
            
            # Apply operation
            clock = VectorClock.from_dict(change.vector_clock)
            return document.apply_operation(operation, clock, change.author_id, change.id)
            
        except Exception as e:
            print(f"Error applying change to document: {e}")
            return False
    
    async def _detect_conflict(self, change: Change) -> Optional[Conflict]:
        """Detect conflicts with pending changes"""
        
        file_id = change.file_id
        pending = self.pending_changes.get(file_id, [])
        
        for pending_change in pending:
            if await self._changes_conflict(change, pending_change):
                conflict = Conflict(
                    workspace_id=change.workspace_id,
                    file_id=file_id,
                    change_a=change,
                    change_b=pending_change,
                    detected_at=datetime.now(),
                    resolution_strategy=self.conflict_resolver.strategy
                )
                return conflict
        
        return None
    
    async def _changes_conflict(self, change_a: Change, change_b: Change) -> bool:
        """Check if two changes conflict"""
        
        # Changes from same author don't conflict
        if change_a.author_id == change_b.author_id:
            return False
        
        # Check vector clock relationship
        clock_a = VectorClock.from_dict(change_a.vector_clock)
        clock_b = change_b.vector_clock
        
        relationship = clock_a.compare(clock_b)
        
        # Concurrent changes might conflict
        if relationship == 'concurrent':
            # Check if changes affect overlapping regions
            return self._changes_overlap(change_a, change_b)
        
        return False
    
    def _changes_overlap(self, change_a: Change, change_b: Change) -> bool:
        """Check if changes affect overlapping text regions"""
        
        # Calculate affected ranges
        range_a = self._get_change_range(change_a)
        range_b = self._get_change_range(change_b)
        
        # Check for overlap
        return not (range_a[1] <= range_b[0] or range_b[1] <= range_a[0])
    
    def _get_change_range(self, change: Change) -> Tuple[int, int]:
        """Get the text range affected by a change"""
        
        start = change.position
        
        if change.change_type == ChangeType.INSERT:
            end = start + len(change.content)
        elif change.change_type == ChangeType.DELETE:
            end = start + change.length
        elif change.change_type == ChangeType.UPDATE:
            end = start + max(change.length, len(change.content))
        else:
            end = start
        
        return (start, end)
    
    def create_change(self, 
                     workspace_id: str,
                     file_id: str,
                     change_type: ChangeType,
                     position: int,
                     content: str = "",
                     length: int = 0,
                     author_id: str = "") -> Change:
        """Create a new change"""
        
        # Ensure document exists
        if file_id not in self.documents:
            self.register_document(file_id)
        
        document = self.documents[file_id]
        
        # Create change with vector clock
        change = Change(
            workspace_id=workspace_id,
            file_id=file_id,
            change_type=change_type,
            position=position,
            length=length,
            content=content,
            author_id=author_id or self.node_id,
            timestamp=datetime.now(),
            vector_clock=document.get_vector_clock()
        )
        
        # Update document vector clock
        document.vector_clock.increment(self.node_id)
        change.vector_clock = document.get_vector_clock()
        
        return change
    
    def get_document_content(self, file_id: str) -> str:
        """Get current document content"""
        if file_id in self.documents:
            return self.documents[file_id].get_content()
        return ""
    
    def get_document_vector_clock(self, file_id: str) -> Dict[str, int]:
        """Get document vector clock"""
        if file_id in self.documents:
            return self.documents[file_id].get_vector_clock()
        return {}
    
    async def sync_changes(self, changes: List[Change]) -> List[Conflict]:
        """Sync changes from remote nodes"""
        
        conflicts = []
        
        for change in changes:
            success, conflict = await self.apply_change(change)
            if conflict:
                conflicts.append(conflict)
        
        # Notify sync callbacks
        for callback in self.sync_callbacks:
            try:
                await callback(changes, conflicts)
            except Exception as e:
                print(f"Error in sync callback: {e}")
        
        return conflicts
    
    def add_sync_callback(self, callback: callable) -> None:
        """Add callback for sync events"""
        self.sync_callbacks.append(callback)
    
    def remove_sync_callback(self, callback: callable) -> None:
        """Remove sync callback"""
        if callback in self.sync_callbacks:
            self.sync_callbacks.remove(callback)
    
    def get_pending_changes(self, file_id: str) -> List[Change]:
        """Get pending changes for file"""
        return self.pending_changes.get(file_id, []).copy()
    
    def clear_pending_changes(self, file_id: str) -> None:
        """Clear pending changes for file"""
        if file_id in self.pending_changes:
            self.pending_changes[file_id].clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CRDT engine statistics"""
        
        total_operations = sum(len(doc.operations) for doc in self.documents.values())
        total_pending = sum(len(changes) for changes in self.pending_changes.values())
        
        return {
            'node_id': self.node_id,
            'registered_documents': len(self.documents),
            'total_operations': total_operations,
            'total_pending_changes': total_pending,
            'sync_callbacks': len(self.sync_callbacks),
            'conflict_resolution_strategy': self.conflict_resolver.strategy
        }


# Global CRDT engine instance
crdt_engine = CRDTEngine()