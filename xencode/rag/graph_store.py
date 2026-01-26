import os
import json
import networkx as nx
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

class GraphStore:
    """
    Stores and manages code relationships in a graph structure.
    Uses NetworkX for graph operations and persists as JSON.
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize the GraphStore.
        
        Args:
            persist_path: Path to the JSON file for persistence. 
                         Defaults to .xencode/graph_store.json
        """
        if persist_path is None:
            persist_path = os.path.join(os.getcwd(), ".xencode", "graph_store.json")
            
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.graph = nx.MultiDiGraph()
        self._load()
        
    def add_node(self, node_id: str, node_type: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a node to the graph."""
        self.graph.add_node(node_id, type=node_type, **(metadata or {}))
        
    def add_relationship(self, source_id: str, target_id: str, rel_type: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a directed edge between two nodes."""
        # Ensure nodes exist
        if source_id not in self.graph:
            self.add_node(source_id, "unknown")
        if target_id not in self.graph:
            self.add_node(target_id, "unknown")
            
        self.graph.add_edge(source_id, target_id, type=rel_type, **(metadata or {}))
        
    def get_related_nodes(self, node_id: str, depth: int = 1, rel_types: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Traverse the graph to find related nodes up to a certain depth.
        """
        if node_id not in self.graph:
            return []
            
        related_nodes = []
        # Use BFS to find nodes within depth
        visited = {node_id}
        queue = [(node_id, 0)]
        
        while queue:
            current_id, current_depth = queue.pop(0)
            
            if current_depth > 0:
                node_data = self.graph.nodes[current_id].copy()
                node_data['id'] = current_id
                related_nodes.append(node_data)
                
            if current_depth < depth:
                # Out-edges (depends on, calls, contains)
                for _, neighbor, edge_data in self.graph.out_edges(current_id, data=True):
                    if neighbor not in visited:
                        if rel_types is None or edge_data.get('type') in rel_types:
                            visited.add(neighbor)
                            queue.append((neighbor, current_depth + 1))
                            
                # In-edges (referenced by, called by, contained by)
                for neighbor, _, edge_data in self.graph.in_edges(current_id, data=True):
                    if neighbor not in visited:
                        if rel_types is None or edge_data.get('type') in rel_types:
                            visited.add(neighbor)
                            queue.append((neighbor, current_depth + 1))
                            
        return related_nodes

    def persist(self):
        """Save the graph to disk."""
        data = nx.node_link_data(self.graph)
        with open(self.persist_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    def _load(self):
        """Load the graph from disk if it exists."""
        if self.persist_path.exists():
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
            except Exception as e:
                print(f"Warning: Failed to load graph store: {e}")
                self.graph = nx.MultiDiGraph()

    def clear(self):
        """Clear the graph and delete the persistent file."""
        self.graph.clear()
        if self.persist_path.exists():
            self.persist_path.unlink()
