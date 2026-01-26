#!/usr/bin/env python3
"""
Visual Workflow Builder for Xencode

Interactive visual interface for creating and modifying AI workflows with drag-and-drop
pipeline creation for complex tasks and template library for common workflow patterns.
"""

import asyncio
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text

console = Console()


class NodeType(Enum):
    """Types of nodes in the workflow"""
    INPUT = "input"
    PROCESSING = "processing"
    AI_MODEL = "ai_model"
    CONDITIONAL = "conditional"
    OUTPUT = "output"
    DATA_SOURCE = "data_source"
    TRANSFORMATION = "transformation"


class ConnectionType(Enum):
    """Types of connections between nodes"""
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow"
    TRIGGER = "trigger"


@dataclass
class WorkflowNode:
    """Represents a node in the workflow"""
    id: str
    node_type: NodeType
    name: str
    position: tuple[int, int] = (0, 0)  # x, y coordinates
    properties: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)  # IDs of input nodes
    outputs: List[str] = field(default_factory=list)  # IDs of output nodes
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"node_{uuid.uuid4()}"


@dataclass
class WorkflowConnection:
    """Represents a connection between two nodes"""
    id: str
    source_node_id: str
    target_node_id: str
    connection_type: ConnectionType
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"conn_{uuid.uuid4()}"


@dataclass
class WorkflowTemplate:
    """Template for common workflow patterns"""
    id: str
    name: str
    description: str
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    author: str = "system"


class WorkflowBuilder:
    """Visual workflow builder with drag-and-drop interface"""
    
    def __init__(self):
        self.nodes: Dict[str, WorkflowNode] = {}
        self.connections: Dict[str, WorkflowConnection] = {}
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.selected_node: Optional[str] = None
        self.workflow_name: str = "Untitled Workflow"
        self.workflow_description: str = ""
        
    def add_node(self, node_type: NodeType, name: str, x: int = 0, y: int = 0, 
                 properties: Optional[Dict[str, Any]] = None) -> str:
        """Add a new node to the workflow"""
        node_id = f"node_{uuid.uuid4()}"
        node = WorkflowNode(
            id=node_id,
            node_type=node_type,
            name=name,
            position=(x, y),
            properties=properties or {}
        )
        self.nodes[node_id] = node
        return node_id
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its connections"""
        if node_id not in self.nodes:
            return False
            
        # Remove all connections involving this node
        connections_to_remove = []
        for conn_id, connection in self.connections.items():
            if connection.source_node_id == node_id or connection.target_node_id == node_id:
                connections_to_remove.append(conn_id)
        
        for conn_id in connections_to_remove:
            del self.connections[conn_id]
        
        # Remove the node
        del self.nodes[node_id]
        return True
    
    def connect_nodes(self, source_id: str, target_id: str, 
                     connection_type: ConnectionType = ConnectionType.DATA_FLOW,
                     properties: Optional[Dict[str, Any]] = None) -> str:
        """Connect two nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source or target node does not exist")
            
        connection_id = f"conn_{uuid.uuid4()}"
        connection = WorkflowConnection(
            id=connection_id,
            source_node_id=source_id,
            target_node_id=target_id,
            connection_type=connection_type,
            properties=properties or {}
        )
        self.connections[connection_id] = connection
        return connection_id
    
    def disconnect_nodes(self, connection_id: str) -> bool:
        """Remove a connection between nodes"""
        if connection_id in self.connections:
            del self.connections[connection_id]
            return True
        return False
    
    def move_node(self, node_id: str, x: int, y: int) -> bool:
        """Move a node to a new position"""
        if node_id in self.nodes:
            self.nodes[node_id].position = (x, y)
            return True
        return False
    
    def update_node_properties(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update properties of a node"""
        if node_id in self.nodes:
            self.nodes[node_id].properties.update(properties)
            return True
        return False
    
    def get_neighbors(self, node_id: str) -> Dict[str, List[str]]:
        """Get neighboring nodes (inputs and outputs)"""
        if node_id not in self.nodes:
            return {"inputs": [], "outputs": []}
            
        inputs = []
        outputs = []
        
        for conn in self.connections.values():
            if conn.target_node_id == node_id:
                inputs.append(conn.source_node_id)
            elif conn.source_node_id == node_id:
                outputs.append(conn.target_node_id)
                
        return {"inputs": inputs, "outputs": outputs}
    
    def validate_workflow(self) -> List[str]:
        """Validate the workflow for common issues"""
        errors = []
        
        # Check for orphaned nodes (nodes with no connections except input/output)
        input_output_types = {NodeType.INPUT, NodeType.OUTPUT}
        for node_id, node in self.nodes.items():
            neighbors = self.get_neighbors(node_id)
            has_inputs = len(neighbors["inputs"]) > 0
            has_outputs = len(neighbors["outputs"]) > 0
            
            if node.node_type not in input_output_types and not has_inputs and not has_outputs:
                errors.append(f"Node '{node.name}' ({node_id}) is not connected to any other nodes")
        
        # Check for cycles in the graph (simplified check)
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(node_id):
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False
                
            visited.add(node_id)
            rec_stack.add(node_id)
            
            neighbors = self.get_neighbors(node_id)
            for output_node_id in neighbors["outputs"]:
                if has_cycle_util(output_node_id):
                    return True
                    
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle_util(node_id):
                    errors.append("Workflow contains cycles which may cause infinite loops")
                    break
        
        return errors
    
    def execute_workflow(self) -> Dict[str, Any]:
        """Execute the workflow (stub implementation)"""
        # This would typically connect to the AI ensemble system
        # For now, return a mock execution result
        validation_errors = self.validate_workflow()
        if validation_errors:
            return {
                "status": "error",
                "errors": validation_errors,
                "result": None
            }
        
        # Mock execution - in reality, this would execute each node in dependency order
        execution_log = []
        for node_id, node in self.nodes.items():
            execution_log.append({
                "node_id": node_id,
                "node_name": node.name,
                "node_type": node.node_type.value,
                "status": "executed",
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "status": "success",
            "execution_log": execution_log,
            "result": "Workflow executed successfully",
            "node_count": len(self.nodes),
            "connection_count": len(self.connections)
        }
    
    def save_workflow(self, filename: str) -> bool:
        """Save workflow to a file"""
        try:
            workflow_data = {
                "name": self.workflow_name,
                "description": self.workflow_description,
                "nodes": [
                    {
                        "id": node.id,
                        "type": node.node_type.value,
                        "name": node.name,
                        "position": node.position,
                        "properties": node.properties,
                        "inputs": node.inputs,
                        "outputs": node.outputs,
                        "metadata": node.metadata
                    }
                    for node in self.nodes.values()
                ],
                "connections": [
                    {
                        "id": conn.id,
                        "source": conn.source_node_id,
                        "target": conn.target_node_id,
                        "type": conn.connection_type.value,
                        "properties": conn.properties
                    }
                    for conn in self.connections.values()
                ],
                "saved_at": datetime.now().isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(workflow_data, f, indent=2)
                
            return True
        except Exception as e:
            console.print(f"[red]Error saving workflow: {e}[/red]")
            return False
    
    def load_workflow(self, filename: str) -> bool:
        """Load workflow from a file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            
            # Clear current workflow
            self.nodes.clear()
            self.connections.clear()
            
            # Set workflow properties
            self.workflow_name = workflow_data.get("name", "Untitled Workflow")
            self.workflow_description = workflow_data.get("description", "")
            
            # Load nodes
            for node_data in workflow_data["nodes"]:
                node = WorkflowNode(
                    id=node_data["id"],
                    node_type=NodeType(node_data["type"]),
                    name=node_data["name"],
                    position=tuple(node_data["position"]),
                    properties=node_data["properties"],
                    inputs=node_data["inputs"],
                    outputs=node_data["outputs"],
                    metadata=node_data.get("metadata", {})
                )
                self.nodes[node_data["id"]] = node
            
            # Load connections
            for conn_data in workflow_data["connections"]:
                connection = WorkflowConnection(
                    id=conn_data["id"],
                    source_node_id=conn_data["source"],
                    target_node_id=conn_data["target"],
                    connection_type=ConnectionType(conn_data["type"]),
                    properties=conn_data["properties"]
                )
                self.connections[conn_data["id"]] = connection
            
            return True
        except Exception as e:
            console.print(f"[red]Error loading workflow: {e}[/red]")
            return False
    
    def create_template(self, name: str, description: str, tags: List[str] = None) -> str:
        """Create a template from the current workflow"""
        template_id = f"template_{uuid.uuid4()}"
        template = WorkflowTemplate(
            id=template_id,
            name=name,
            description=description,
            nodes=list(self.nodes.values()),
            connections=list(self.connections.values()),
            tags=tags or []
        )
        self.templates[template_id] = template
        return template_id
    
    def apply_template(self, template_id: str) -> bool:
        """Apply a template to the current workflow"""
        if template_id not in self.templates:
            return False
            
        template = self.templates[template_id]
        
        # Clear current workflow
        self.nodes.clear()
        self.connections.clear()
        
        # Apply template
        for node in template.nodes:
            # Create new IDs for the nodes to avoid conflicts
            new_node = WorkflowNode(
                id=f"node_{uuid.uuid4()}",
                node_type=node.node_type,
                name=node.name,
                position=node.position,
                properties=node.properties,
                inputs=node.inputs,
                outputs=node.outputs,
                metadata=node.metadata
            )
            self.nodes[new_node.id] = new_node
        
        # Map old IDs to new IDs for connections
        id_mapping = {}
        for old_node in template.nodes:
            for new_node in self.nodes.values():
                if (new_node.name == old_node.name and 
                    new_node.node_type == old_node.node_type):
                    id_mapping[old_node.id] = new_node.id
                    break
        
        # Apply connections with new IDs
        for conn in template.connections:
            new_source_id = id_mapping.get(conn.source_node_id)
            new_target_id = id_mapping.get(conn.target_node_id)
            
            if new_source_id and new_target_id:
                new_conn = WorkflowConnection(
                    id=f"conn_{uuid.uuid4()}",
                    source_node_id=new_source_id,
                    target_node_id=new_target_id,
                    connection_type=conn.connection_type,
                    properties=conn.properties
                )
                self.connections[new_conn.id] = new_conn
        
        return True
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get statistics about the current workflow"""
        node_types = {}
        for node in self.nodes.values():
            node_type = node.node_type.value
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        return {
            "node_count": len(self.nodes),
            "connection_count": len(self.connections),
            "node_types": node_types,
            "template_count": len(self.templates),
            "is_valid": len(self.validate_workflow()) == 0
        }
    
    def display_workflow(self):
        """Display the workflow in the console"""
        stats = self.get_workflow_stats()
        
        console.print(Panel(
            f"[bold blue]Workflow: {self.workflow_name}[/bold blue]\n"
            f"Description: {self.workflow_description}\n"
            f"Nodes: {stats['node_count']} | Connections: {stats['connection_count']}\n"
            f"Valid: {'Yes' if stats['is_valid'] else 'No'}",
            title="Workflow Overview",
            border_style="blue"
        ))
        
        # Display nodes table
        nodes_table = Table(title="Nodes")
        nodes_table.add_column("ID", style="cyan")
        nodes_table.add_column("Name", style="magenta")
        nodes_table.add_column("Type", style="green")
        nodes_table.add_column("Position", style="yellow")
        
        for node in self.nodes.values():
            nodes_table.add_row(
                node.id[:8] + "...",  # Truncate ID
                node.name,
                node.node_type.value,
                f"({node.position[0]}, {node.position[1]})"
            )
        
        console.print(nodes_table)
        
        # Display connections table
        connections_table = Table(title="Connections")
        connections_table.add_column("ID", style="cyan")
        connections_table.add_column("Source", style="magenta")
        connections_table.add_column("Target", style="magenta")
        connections_table.add_column("Type", style="green")
        
        for conn in self.connections.values():
            connections_table.add_row(
                conn.id[:8] + "...",  # Truncate ID
                self.nodes[conn.source_node_id].name if conn.source_node_id in self.nodes else "Unknown",
                self.nodes[conn.target_node_id].name if conn.target_node_id in self.nodes else "Unknown",
                conn.connection_type.value
            )
        
        console.print(connections_table)


async def demo_visual_workflow_builder():
    """Demonstrate the visual workflow builder"""
    console.print("[bold green]Initializing Visual Workflow Builder[/bold green]\n")
    
    builder = WorkflowBuilder()
    builder.workflow_name = "Code Review Workflow"
    builder.workflow_description = "AI-powered code review and analysis workflow"
    
    # Add nodes to the workflow
    input_node = builder.add_node(NodeType.INPUT, "Code Input", 0, 0, {
        "source": "git_repository",
        "format": "diff"
    })
    
    analysis_node = builder.add_node(NodeType.PROCESSING, "Code Analysis", 20, 0, {
        "tool": "static_analysis",
        "checks": ["security", "style", "complexity"]
    })
    
    ai_node = builder.add_node(NodeType.AI_MODEL, "AI Review", 40, 0, {
        "model": "llama3.1:8b",
        "task": "code_review"
    })
    
    output_node = builder.add_node(NodeType.OUTPUT, "Review Output", 60, 0, {
        "format": "structured_report",
        "destination": "pull_request"
    })
    
    # Connect the nodes
    builder.connect_nodes(input_node, analysis_node)
    builder.connect_nodes(analysis_node, ai_node)
    builder.connect_nodes(ai_node, output_node)
    
    # Display the workflow
    builder.display_workflow()
    
    # Validate the workflow
    errors = builder.validate_workflow()
    if errors:
        console.print("\n[red]Validation Errors:[/red]")
        for error in errors:
            console.print(f"  • {error}")
    else:
        console.print("\n[green]Workflow is valid![/green]")
    
    # Execute the workflow (mock)
    console.print("\n[blue]Executing workflow...[/blue]")
    result = builder.execute_workflow()
    
    if result["status"] == "success":
        console.print(f"[green]{result['result']}[/green]")
        console.print(f"  Nodes processed: {result['node_count']}")
        console.print(f"  Connections: {result['connection_count']}")
    else:
        console.print(f"[red]Execution failed: {result['errors']}[/red]")
    
    # Create a template
    template_id = builder.create_template(
        "Code Review Template",
        "Standard template for code review workflows",
        ["ai", "development", "review"]
    )
    console.print(f"\n[blue]Created template: {template_id}[/blue]")
    
    # Show workflow stats
    stats = builder.get_workflow_stats()
    console.print(f"\n[bold]Workflow Stats:[/bold]")
    console.print(f"  Nodes: {stats['node_count']}")
    console.print(f"  Connections: {stats['connection_count']}")
    console.print(f"  Templates: {stats['template_count']}")
    console.print(f"  Valid: {stats['is_valid']}")
    
    console.print("\n[green]Visual Workflow Builder Demo Completed[/green]")


if __name__ == "__main__":
    asyncio.run(demo_visual_workflow_builder())