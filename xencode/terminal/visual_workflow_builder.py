"""
Visual Workflow Builder for Xencode

Implements a visual interface for creating and modifying AI workflows with
drag-and-drop pipeline creation for complex tasks and template library
for common workflow patterns.
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
import logging

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.prompt import Prompt

from ..ai.hybrid_model_architecture import HybridModelManager, TaskContext
from ..cache.advanced_memory_manager import get_memory_manager, CachePriority


logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the workflow"""
    START = "start"
    INPUT = "input"
    PROCESS = "process"
    DECISION = "decision"
    OUTPUT = "output"
    MODEL_CALL = "model_call"
    DATA_TRANSFORM = "data_transform"
    CONDITIONAL_BRANCH = "conditional_branch"
    LOOP = "loop"
    END = "end"


class ConnectionType(Enum):
    """Types of connections between nodes"""
    SEQUENCE = "sequence"
    CONDITIONAL = "conditional"
    LOOP_BACK = "loop_back"
    PARALLEL = "parallel"


@dataclass
class NodePosition:
    """Position of a node in the visual canvas"""
    x: float
    y: float
    width: float = 150.0
    height: float = 80.0


@dataclass
class WorkflowNode:
    """A node in the workflow"""
    id: str
    node_type: NodeType
    title: str
    description: str
    config: Dict[str, Any]  # Configuration specific to the node type
    position: NodePosition
    inputs: List[str]  # IDs of input nodes
    outputs: List[str]  # IDs of output nodes
    metadata: Dict[str, Any]  # Additional metadata
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


@dataclass
class WorkflowConnection:
    """A connection between two nodes"""
    id: str
    source_node_id: str
    target_node_id: str
    connection_type: ConnectionType
    label: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkflowTemplate:
    """A template for common workflow patterns"""
    id: str
    name: str
    description: str
    category: str  # e.g., "development", "analysis", "automation"
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]
    created_at: datetime
    author: str
    tags: List[str]
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()


class NodeProcessor(ABC):
    """Abstract base class for processing different node types"""
    
    @abstractmethod
    async def process(self, node: WorkflowNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the node and return output data"""
        pass


class InputNodeProcessor(NodeProcessor):
    """Processor for input nodes"""
    
    async def process(self, node: WorkflowNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input node - typically gets data from user or external source"""
        input_type = node.config.get("input_type", "text")
        prompt = node.config.get("prompt", "Enter input:")
        
        # For demo purposes, we'll return static data
        # In a real implementation, this would get input from user or external source
        if input_type == "text":
            return {"data": f"Sample input for {node.title}"}
        elif input_type == "file":
            return {"data": f"File content for {node.title}"}
        else:
            return {"data": f"Input data for {node.title}"}


class ProcessNodeProcessor(NodeProcessor):
    """Processor for process nodes"""
    
    async def process(self, node: WorkflowNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a generic process node"""
        operation = node.config.get("operation", "identity")
        
        # Get input data from context
        input_data = context.get("input_data", {})
        
        # Perform operation based on configuration
        if operation == "uppercase":
            processed_data = {k: str(v).upper() for k, v in input_data.items()}
        elif operation == "lowercase":
            processed_data = {k: str(v).lower() for k, v in input_data.items()}
        elif operation == "length":
            processed_data = {k: len(str(v)) for k, v in input_data.items()}
        else:
            # Default: pass through
            processed_data = input_data
        
        return {"processed_data": processed_data}


class ModelCallNodeProcessor(NodeProcessor):
    """Processor for model call nodes"""
    
    def __init__(self, hybrid_manager: HybridModelManager):
        self.hybrid_manager = hybrid_manager
    
    async def process(self, node: WorkflowNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a model call node"""
        model_config = node.config.get("model_config", {})
        prompt_template = node.config.get("prompt_template", "{input}")
        
        # Get input data
        input_data = context.get("input_data", {})
        
        # Format prompt with input data
        if isinstance(input_data, dict):
            formatted_prompt = prompt_template.format(**input_data)
        else:
            formatted_prompt = prompt_template.format(input=input_data)
        
        # Determine task context
        task_type = model_config.get("task_type", "general")
        sensitivity = model_config.get("sensitivity", 2)
        complexity = model_config.get("complexity", 3)
        
        task_context = TaskContext(
            task_type=task_type,
            sensitivity_level=sensitivity,
            complexity_level=complexity,
            urgency_level=2,
            context_size=len(formatted_prompt),
            required_capabilities=["reasoning"],
            user_preferences={}
        )
        
        # Call the hybrid model
        response = await self.hybrid_manager.generate(formatted_prompt, task_context)
        
        return {"model_response": response, "input_used": formatted_prompt}


class DecisionNodeProcessor(NodeProcessor):
    """Processor for decision nodes"""
    
    async def process(self, node: WorkflowNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a decision node"""
        condition = node.config.get("condition", "true")
        input_data = context.get("input_data", {})
        
        # Evaluate condition
        # This is a simplified implementation
        # In a real system, this would use a more sophisticated rule engine
        try:
            # Create a safe evaluation context
            eval_context = input_data.copy()
            eval_context.update({"len": len, "str": str, "int": int, "float": float})
            
            # Evaluate the condition
            result = eval(condition, {"__builtins__": {}}, eval_context)
            decision_result = bool(result)
        except:
            # If condition evaluation fails, default to true
            decision_result = True
        
        return {"decision": decision_result, "condition_evaluated": condition}


class WorkflowEngine:
    """Executes workflows defined by nodes and connections"""
    
    def __init__(self):
        self.hybrid_manager = None
        self.memory_manager = get_memory_manager()
        self.processors = {
            NodeType.INPUT: InputNodeProcessor(),
            NodeType.PROCESS: ProcessNodeProcessor(),
            NodeType.MODEL_CALL: None,  # Will be set when hybrid manager is available
            NodeType.DECISION: DecisionNodeProcessor(),
            # Add other processors as needed
        }
        self.console = Console()
    
    def set_hybrid_manager(self, hybrid_manager: HybridModelManager):
        """Set the hybrid model manager for model call nodes"""
        self.hybrid_manager = hybrid_manager
        self.processors[NodeType.MODEL_CALL] = ModelCallNodeProcessor(hybrid_manager)
    
    async def execute_workflow(self, 
                             nodes: List[WorkflowNode], 
                             connections: List[WorkflowConnection],
                             initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow"""
        if initial_data is None:
            initial_data = {}
        
        # Find the start node
        start_nodes = [node for node in nodes if node.node_type == NodeType.START]
        if not start_nodes:
            # Fallback: check for INPUT nodes as start nodes
            start_nodes = [node for node in nodes if node.node_type == NodeType.INPUT]
            
        if not start_nodes:
            # Fallback: check for any node with no inputs (roots)
            node_inputs = set()
            for conn in connections:
                node_inputs.add(conn.target_node_id)
            
            start_nodes = [node for node in nodes if node.id not in node_inputs]

        if not start_nodes:
            # If still no start node, just pick the first one
            if nodes:
                start_nodes = [nodes[0]]
            else:
                raise ValueError("No nodes in workflow")
        
        start_node = start_nodes[0]
        
        # Execute the workflow graph
        execution_context = {
            "workflow_data": initial_data.copy(),
            "node_outputs": {},
            "visited_nodes": set()
        }
        
        result = await self._execute_node_recursive(start_node, nodes, connections, execution_context)
        
        return result
    
    async def _execute_node_recursive(self, 
                                   current_node: WorkflowNode, 
                                   all_nodes: List[WorkflowNode],
                                   connections: List[WorkflowConnection],
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively execute nodes in the workflow"""
        if current_node.id in context["visited_nodes"]:
            # Prevent infinite loops
            return context["workflow_data"]
        
        context["visited_nodes"].add(current_node.id)
        
        # Execute the current node
        processor = self.processors.get(current_node.node_type)
        if processor is None:
            logger.warning(f"No processor found for node type: {current_node.node_type}")
            node_output = {"data": f"Skipped node {current_node.title}"}
        else:
            # Prepare input data for the node
            node_input_context = {
                "input_data": context["workflow_data"],
                "previous_outputs": context["node_outputs"]
            }
            
            node_output = await processor.process(current_node, node_input_context)
        
        # Store the output
        context["node_outputs"][current_node.id] = node_output
        context["workflow_data"].update(node_output)
        
        # Get outgoing connections
        outgoing_connections = [
            conn for conn in connections 
            if conn.source_node_id == current_node.id
        ]
        
        # Execute connected nodes
        for connection in outgoing_connections:
            target_node = next(
                (node for node in all_nodes if node.id == connection.target_node_id), 
                None
            )
            
            if target_node:
                # For conditional connections, check if condition is met
                if connection.connection_type == ConnectionType.CONDITIONAL:
                    # This would check the decision result
                    decision_result = context["node_outputs"].get(current_node.id, {}).get("decision", True)
                    if not decision_result:
                        continue  # Skip this branch
                
                await self._execute_node_recursive(target_node, all_nodes, connections, context)
        
        return context["workflow_data"]


class WorkflowGenerator:
    """Generates workflows from natural language descriptions"""
    
    def __init__(self, hybrid_manager: HybridModelManager):
        self.hybrid_manager = hybrid_manager
    
    async def generate_workflow(self, description: str) -> Optional[Dict[str, Any]]:
        """Generate a workflow structure from a description"""
        prompt = f"""
        You are an expert AI workflow architect. Your goal is to convert a natural language description of a workflow into a JSON structure compatible with the Xencode Workflow Builder.

        Available Node Types:
        - start: Entry point
        - input: Get user input (config: input_type, prompt)
        - process: transform data (config: operation like "uppercase", "lowercase", "length")
        - decision: branching logic (config: condition)
        - output: display result
        - model_call: AI model invocation (config: model_config, prompt_template)
        - data_transform: general transformation
        - conditional_branch: branch based on condition
        - loop: repeat execution
        - end: workflow termination

        Available Connection Types:
        - sequence: normal flow
        - conditional: branch flow
        - loop_back: return to previous node
        - parallel: concurrent flow

        Output Format (JSON):
        {{
          "nodes": [
            {{
              "id": "unique_id",
              "node_type": "node_type_value",
              "title": "Readable Title",
              "description": "What this node does",
              "config": {{ ... }},
              "position": {{ "x": 100, "y": 100 }}
            }}
          ],
          "connections": [
            {{
              "source_node_id": "source_id",
              "target_node_id": "target_id",
              "connection_type": "sequence"
            }}
          ]
        }}

        User Description: {description}

        Generate the JSON workflow structure. Ensure node IDs are unique strings. Position nodes logically (e.g., left to right). Return ONLY the JSON.
        """
        
        try:
            response = await self.hybrid_manager.generate(prompt)
            
            # Clean up response if it contains markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error generating workflow: {e}")
            return None


class WorkflowBuilder:
    """Visual workflow builder with drag-and-drop interface"""
    
    def __init__(self):
        self.nodes: List[WorkflowNode] = []
        self.connections: List[WorkflowConnection] = []
        self.templates: List[WorkflowTemplate] = []
        self.engine = WorkflowEngine()
        self.console = Console()
        self.hybrid_manager = None
        
        # Load default templates
        self._load_default_templates()
        
    def set_hybrid_manager(self, hybrid_manager: HybridModelManager):
        """Set the hybrid model manager"""
        self.hybrid_manager = hybrid_manager
        self.engine.set_hybrid_manager(hybrid_manager)
        
    async def generate_from_description(self, description: str) -> bool:
        """Generate a workflow from natural language description"""
        if not self.hybrid_manager:
            return False
            
        generator = WorkflowGenerator(self.hybrid_manager)
        workflow_data = await generator.generate_workflow(description)
        
        if not workflow_data:
            return False
            
        # Clear current workflow
        self.nodes = []
        self.connections = []
        
        try:
            # Load nodes
            for node_data in workflow_data.get("nodes", []):
                # Handle potential enum conversion issues
                node_type_str = node_data["node_type"]
                try:
                    node_type = NodeType(node_type_str)
                except ValueError:
                    # Fallback or mapping if needed
                    logger.warning(f"Unknown node type: {node_type_str}, defaulting to PROCESS")
                    node_type = NodeType.PROCESS

                node = WorkflowNode(
                    id=node_data["id"],
                    node_type=node_type,
                    title=node_data["title"],
                    description=node_data.get("description", ""),
                    config=node_data.get("config", {}),
                    position=NodePosition(**node_data["position"]),
                    inputs=[], # Will be populated by connections
                    outputs=[], # Will be populated by connections
                    metadata=node_data.get("metadata", {})
                )
                self.nodes.append(node)
            
            # Load connections
            for conn_data in workflow_data.get("connections", []):
                conn_type_str = conn_data["connection_type"]
                try:
                    conn_type = ConnectionType(conn_type_str)
                except ValueError:
                    conn_type = ConnectionType.SEQUENCE

                connection = WorkflowConnection(
                    id=str(uuid.uuid4()), # Generate new ID just in case
                    source_node_id=conn_data["source_node_id"],
                    target_node_id=conn_data["target_node_id"],
                    connection_type=conn_type,
                    label=conn_data.get("label", ""),
                    metadata=conn_data.get("metadata", {})
                )
                
                # Update node inputs/outputs
                source_node = self.get_node_by_id(connection.source_node_id)
                target_node = self.get_node_by_id(connection.target_node_id)
                
                if source_node and target_node:
                    source_node.outputs.append(target_node.id)
                    target_node.inputs.append(source_node.id)
                    self.connections.append(connection)
            
            return True
        except Exception as e:
            logger.error(f"Error applying generated workflow: {e}")
            return False
    
    def _load_default_templates(self):
        """Load default workflow templates"""
        # Simple text processing template
        text_processing_nodes = [
            WorkflowNode(
                id="input_1",
                node_type=NodeType.INPUT,
                title="Input Text",
                description="Get text input from user",
                config={"input_type": "text", "prompt": "Enter text to process:"},
                position=NodePosition(x=100, y=100),
                inputs=[],
                outputs=["process_1"],
                metadata={}
            ),
            WorkflowNode(
                id="process_1",
                node_type=NodeType.PROCESS,
                title="Process Text",
                description="Transform the input text",
                config={"operation": "uppercase"},
                position=NodePosition(x=300, y=100),
                inputs=["input_1"],
                outputs=["output_1"],
                metadata={}
            ),
            WorkflowNode(
                id="output_1",
                node_type=NodeType.OUTPUT,
                title="Output Result",
                description="Display the result",
                config={},
                position=NodePosition(x=500, y=100),
                inputs=["process_1"],
                outputs=[],
                metadata={}
            )
        ]
        
        text_processing_connections = [
            WorkflowConnection(
                id="conn_1",
                source_node_id="input_1",
                target_node_id="process_1",
                connection_type=ConnectionType.SEQUENCE
            ),
            WorkflowConnection(
                id="conn_2",
                source_node_id="process_1",
                target_node_id="output_1",
                connection_type=ConnectionType.SEQUENCE
            )
        ]
        
        text_processing_template = WorkflowTemplate(
            id="text_processing",
            name="Text Processing Workflow",
            description="A simple workflow that takes text input, processes it, and outputs the result",
            category="development",
            nodes=text_processing_nodes,
            connections=text_processing_connections,
            created_at=datetime.now(),
            author="System",
            tags=["text", "processing", "simple"]
        )
        
        self.templates.append(text_processing_template)
    
    def add_node(self, 
                 node_type: NodeType, 
                 title: str, 
                 description: str, 
                 config: Dict[str, Any],
                 x: float = 100, 
                 y: float = 100) -> WorkflowNode:
        """Add a node to the workflow"""
        node = WorkflowNode(
            id=str(uuid.uuid4()),
            node_type=node_type,
            title=title,
            description=description,
            config=config,
            position=NodePosition(x=x, y=y),
            inputs=[],
            outputs=[],
            metadata={}
        )
        
        self.nodes.append(node)
        return node
    
    def connect_nodes(self, 
                     source_node_id: str, 
                     target_node_id: str, 
                     connection_type: ConnectionType = ConnectionType.SEQUENCE,
                     label: str = "") -> WorkflowConnection:
        """Connect two nodes"""
        connection = WorkflowConnection(
            id=str(uuid.uuid4()),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            connection_type=connection_type,
            label=label
        )
        
        # Update node connections
        for node in self.nodes:
            if node.id == source_node_id:
                node.outputs.append(target_node_id)
            elif node.id == target_node_id:
                node.inputs.append(source_node_id)
        
        self.connections.append(connection)
        return connection
    
    def get_node_by_id(self, node_id: str) -> Optional[WorkflowNode]:
        """Get a node by its ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def visualize_workflow(self) -> str:
        """Create a text-based visualization of the workflow"""
        if not self.nodes:
            return "No nodes in workflow"
        
        # Create a tree view of the workflow
        tree = Tree("Workflow Visualization")
        
        # Find start nodes
        start_nodes = [node for node in self.nodes if node.node_type == NodeType.START]
        if not start_nodes:
            # If no start node, use first node
            start_nodes = [self.nodes[0]]
        
        visited = set()
        
        def add_node_to_tree(node: WorkflowNode, parent_tree: Tree):
            if node.id in visited:
                return
            visited.add(node.id)
            
            node_text = f"[bold]{node.title}[/bold] ({node.node_type.value})"
            if node.description:
                node_text += f"\n[dim]{node.description}[/dim]"
            
            node_tree = parent_tree.add(node_text)
            
            # Add output connections
            for output_id in node.outputs:
                output_node = self.get_node_by_id(output_id)
                if output_node and output_node.id not in visited:
                    add_node_to_tree(output_node, node_tree)
        
        for start_node in start_nodes:
            add_node_to_tree(start_node, tree)
        
        return tree
    
    def save_workflow(self, file_path: Path) -> bool:
        """Save the current workflow to a file"""
        try:
            workflow_data = {
                "nodes": [
                    {
                        "id": node.id,
                        "node_type": node.node_type.value,
                        "title": node.title,
                        "description": node.description,
                        "config": node.config,
                        "position": {
                            "x": node.position.x,
                            "y": node.position.y,
                            "width": node.position.width,
                            "height": node.position.height
                        },
                        "inputs": node.inputs,
                        "outputs": node.outputs,
                        "metadata": node.metadata
                    }
                    for node in self.nodes
                ],
                "connections": [
                    {
                        "id": conn.id,
                        "source_node_id": conn.source_node_id,
                        "target_node_id": conn.target_node_id,
                        "connection_type": conn.connection_type.value,
                        "label": conn.label,
                        "metadata": conn.metadata
                    }
                    for conn in self.connections
                ],
                "saved_at": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(workflow_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving workflow: {e}")
            return False
    
    def load_workflow(self, file_path: Path) -> bool:
        """Load a workflow from a file"""
        try:
            with open(file_path, 'r') as f:
                workflow_data = json.load(f)
            
            # Clear current workflow
            self.nodes = []
            self.connections = []
            
            # Load nodes
            for node_data in workflow_data["nodes"]:
                node = WorkflowNode(
                    id=node_data["id"],
                    node_type=NodeType(node_data["node_type"]),
                    title=node_data["title"],
                    description=node_data["description"],
                    config=node_data["config"],
                    position=NodePosition(**node_data["position"]),
                    inputs=node_data["inputs"],
                    outputs=node_data["outputs"],
                    metadata=node_data["metadata"]
                )
                self.nodes.append(node)
            
            # Load connections
            for conn_data in workflow_data["connections"]:
                connection = WorkflowConnection(
                    id=conn_data["id"],
                    source_node_id=conn_data["source_node_id"],
                    target_node_id=conn_data["target_node_id"],
                    connection_type=ConnectionType(conn_data["connection_type"]),
                    label=conn_data.get("label", ""),
                    metadata=conn_data.get("metadata", {})
                )
                self.connections.append(connection)
            
            return True
        except Exception as e:
            logger.error(f"Error loading workflow: {e}")
            return False
    
    def get_templates_by_category(self, category: str) -> List[WorkflowTemplate]:
        """Get templates by category"""
        return [template for template in self.templates if template.category == category]
    
    def instantiate_template(self, template_id: str) -> bool:
        """Instantiate a template as the current workflow"""
        template = next((t for t in self.templates if t.id == template_id), None)
        if not template:
            return False
        
        # Clear current workflow
        self.nodes = []
        self.connections = []
        
        # Add template nodes and connections
        for node in template.nodes:
            # Create new IDs for the instantiated nodes
            new_node = WorkflowNode(
                id=str(uuid.uuid4()),
                node_type=node.node_type,
                title=node.title,
                description=node.description,
                config=node.config.copy(),
                position=NodePosition(
                    x=node.position.x,
                    y=node.position.y,
                    width=node.position.width,
                    height=node.position.height
                ),
                inputs=node.inputs.copy(),
                outputs=node.outputs.copy(),
                metadata=node.metadata.copy()
            )
            self.nodes.append(new_node)
        
        for conn in template.connections:
            new_conn = WorkflowConnection(
                id=str(uuid.uuid4()),
                source_node_id="",  # Will be updated after all nodes are added
                target_node_id="",   # Will be updated after all nodes are added
                connection_type=conn.connection_type,
                label=conn.label,
                metadata=conn.metadata.copy()
            )
            # Map old IDs to new IDs
            old_source_id = conn.source_node_id
            old_target_id = conn.target_node_id
            
            # Find corresponding new IDs
            for orig_node, new_node in zip(template.nodes, self.nodes):
                if orig_node.id == old_source_id:
                    new_conn.source_node_id = new_node.id
                if orig_node.id == old_target_id:
                    new_conn.target_node_id = new_node.id
            
            self.connections.append(new_conn)
        
        return True
    
    async def execute_current_workflow(self, initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the current workflow"""
        return await self.engine.execute_workflow(self.nodes, self.connections, initial_data)


class VisualWorkflowInterface:
    """Interactive visual interface for the workflow builder"""
    
    def __init__(self):
        self.builder = WorkflowBuilder()
        self.console = Console()
    
    async def run_interactive_mode(self):
        """Run the interactive workflow builder"""
        self.console.print(Panel("[bold blue]Xencode Visual Workflow Builder[/bold blue]"))
        
        # Try to initialize hybrid manager if not already set
        if not self.builder.hybrid_manager:
            try:
                from ..ai.hybrid_model_architecture import get_hybrid_model_manager
                self.builder.set_hybrid_manager(get_hybrid_model_manager())
            except ImportError:
                pass
        
        while True:
            self.console.print("\n[bold]Workflow Builder Menu:[/bold]")
            self.console.print("1. View current workflow")
            self.console.print("2. Add node")
            self.console.print("3. Connect nodes")
            self.console.print("4. View templates")
            self.console.print("5. Load template")
            self.console.print("6. Execute workflow")
            self.console.print("7. Save workflow")
            self.console.print("8. Load workflow")
            self.console.print("9. Generate workflow from description")
            self.console.print("10. Exit")
            
            choice = Prompt.ask("Select an option", choices=[str(i) for i in range(1, 11)])
            
            if choice == "1":
                self._view_workflow()
            elif choice == "2":
                await self._add_node_interactive()
            elif choice == "3":
                await self._connect_nodes_interactive()
            elif choice == "4":
                self._view_templates()
            elif choice == "5":
                await self._load_template_interactive()
            elif choice == "6":
                await self._execute_workflow_interactive()
            elif choice == "7":
                await self._save_workflow_interactive()
            elif choice == "8":
                await self._load_workflow_interactive()
            elif choice == "9":
                await self._generate_workflow_interactive()
            elif choice == "10":
                self.console.print("[green]Goodbye![/green]")
                break
                
    async def _generate_workflow_interactive(self):
        """Interactively generate a workflow from description"""
        self.console.print("\n[bold]Generate Workflow:[/bold]")
        
        if not self.builder.hybrid_manager:
            self.console.print("[red]Hybrid Model Manager not available. Cannot generate workflow.[/red]")
            return
            
        description = Prompt.ask("Describe the workflow you want to create")
        
        with self.console.status("[bold green]Generating workflow...[/bold green]"):
            success = await self.builder.generate_from_description(description)
            
        if success:
            self.console.print("[green]Workflow generated successfully![/green]")
            self._view_workflow()
        else:
            self.console.print("[red]Failed to generate workflow[/red]")
    
    def _view_workflow(self):
        """View the current workflow"""
        self.console.print("\n[bold]Current Workflow:[/bold]")
        tree = self.builder.visualize_workflow()
        self.console.print(tree)
    
    async def _add_node_interactive(self):
        """Interactively add a node"""
        self.console.print("\n[bold]Add Node:[/bold]")
        
        # Show available node types
        node_types = [nt.value for nt in NodeType]
        self.console.print("Available node types:")
        for i, node_type in enumerate(node_types, 1):
            self.console.print(f"{i}. {node_type}")
        
        type_choice = int(Prompt.ask("Select node type")) - 1
        node_type = NodeType(node_types[type_choice])
        
        title = Prompt.ask("Enter node title")
        description = Prompt.ask("Enter node description (optional)", default="")
        
        # Get configuration based on node type
        config = {}
        if node_type == NodeType.INPUT:
            input_type = Prompt.ask("Input type (text/file/other)", default="text")
            prompt_text = Prompt.ask("Prompt for user", default="Enter input:")
            config = {"input_type": input_type, "prompt": prompt_text}
        elif node_type == NodeType.PROCESS:
            operation = Prompt.ask("Operation (uppercase/lowercase/length/other)", default="identity")
            config = {"operation": operation}
        elif node_type == NodeType.MODEL_CALL:
            task_type = Prompt.ask("Task type", default="general")
            model_name = Prompt.ask("Model name (optional)", default="")
            prompt_template = Prompt.ask("Prompt template", default="{input}")
            config = {
                "model_config": {"task_type": task_type, "model_name": model_name},
                "prompt_template": prompt_template
            }
        elif node_type == NodeType.DECISION:
            condition = Prompt.ask("Condition (Python expression)", default="True")
            config = {"condition": condition}
        
        x = float(Prompt.ask("X position", default="100"))
        y = float(Prompt.ask("Y position", default="100"))
        
        node = self.builder.add_node(node_type, title, description, config, x, y)
        self.console.print(f"[green]Added node: {node.title}[/green]")
    
    async def _connect_nodes_interactive(self):
        """Interactively connect nodes"""
        if len(self.builder.nodes) < 2:
            self.console.print("[red]Need at least 2 nodes to connect[/red]")
            return
        
        self.console.print("\n[bold]Available Nodes:[/bold]")
        for i, node in enumerate(self.builder.nodes):
            self.console.print(f"{i+1}. {node.title} ({node.id})")
        
        source_idx = int(Prompt.ask("Select source node")) - 1
        target_idx = int(Prompt.ask("Select target node")) - 1
        
        source_node = self.builder.nodes[source_idx]
        target_node = self.builder.nodes[target_idx]
        
        # Show connection types
        conn_types = [ct.value for ct in ConnectionType]
        self.console.print("Connection types:")
        for i, conn_type in enumerate(conn_types, 1):
            self.console.print(f"{i}. {conn_type}")
        
        type_choice = int(Prompt.ask("Select connection type")) - 1
        conn_type = ConnectionType(conn_types[type_choice])
        
        label = Prompt.ask("Connection label (optional)", default="")
        
        connection = self.builder.connect_nodes(source_node.id, target_node.id, conn_type, label)
        self.console.print(f"[green]Connected {source_node.title} to {target_node.title}[/green]")
    
    def _view_templates(self):
        """View available templates"""
        self.console.print("\n[bold]Available Templates:[/bold]")
        
        categories = set(template.category for template in self.builder.templates)
        
        for category in categories:
            self.console.print(f"\n[underline]{category}[/underline]:")
            templates = self.builder.get_templates_by_category(category)
            for template in templates:
                self.console.print(f"  • {template.name}: {template.description}")
    
    async def _load_template_interactive(self):
        """Interactively load a template"""
        self.console.print("\n[bold]Load Template:[/bold]")
        
        # Show all templates
        for i, template in enumerate(self.builder.templates):
            self.console.print(f"{i+1}. {template.name} ({template.category}): {template.description}")
        
        if not self.builder.templates:
            self.console.print("[yellow]No templates available[/yellow]")
            return
        
        template_idx = int(Prompt.ask("Select template to load")) - 1
        template = self.builder.templates[template_idx]
        
        success = self.builder.instantiate_template(template.id)
        if success:
            self.console.print(f"[green]Loaded template: {template.name}[/green]")
        else:
            self.console.print("[red]Failed to load template[/red]")
    
    async def _execute_workflow_interactive(self):
        """Interactively execute the workflow"""
        if not self.builder.nodes:
            self.console.print("[red]No workflow to execute[/red]")
            return
        
        self.console.print("\n[bold]Executing Workflow...[/bold]")
        
        # Get initial data if needed
        initial_data_str = Prompt.ask("Initial data (JSON format, optional)", default="{}")
        try:
            initial_data = json.loads(initial_data_str) if initial_data_str.strip() else {}
        except json.JSONDecodeError:
            self.console.print("[red]Invalid JSON format[/red]")
            return
        
        try:
            result = await self.builder.execute_current_workflow(initial_data)
            self.console.print("[bold]Execution Result:[/bold]")
            self.console.print_json(data=result)
        except Exception as e:
            self.console.print(f"[red]Execution failed: {e}[/red]")
    
    async def _save_workflow_interactive(self):
        """Interactively save the workflow"""
        filename = Prompt.ask("Enter filename to save workflow")
        file_path = Path(filename)
        
        if not file_path.suffix:
            file_path = file_path.with_suffix(".json")
        
        success = self.builder.save_workflow(file_path)
        if success:
            self.console.print(f"[green]Workflow saved to {file_path}[/green]")
        else:
            self.console.print("[red]Failed to save workflow[/red]")
    
    async def _load_workflow_interactive(self):
        """Interactively load a workflow"""
        filename = Prompt.ask("Enter filename to load workflow")
        file_path = Path(filename)
        
        if not file_path.suffix:
            file_path = file_path.with_suffix(".json")
        
        if not file_path.exists():
            self.console.print(f"[red]File {file_path} does not exist[/red]")
            return
        
        success = self.builder.load_workflow(file_path)
        if success:
            self.console.print(f"[green]Workflow loaded from {file_path}[/green]")
        else:
            self.console.print("[red]Failed to load workflow[/red]")


# Global workflow builder instance
_workflow_builder: Optional[VisualWorkflowInterface] = None


def get_visual_workflow_builder() -> VisualWorkflowInterface:
    """Get the global visual workflow builder instance"""
    global _workflow_builder
    if _workflow_builder is None:
        _workflow_builder = VisualWorkflowInterface()
    return _workflow_builder


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Check if we should run in interactive mode
    import sys
    
    # Mock Hybrid Manager for testing
    class MockHybridManager:
        async def generate(self, prompt, context=None):
            return f"Mock response for: {prompt[:30]}..."

    if "--interactive" in sys.argv:
        async def run_interactive():
            vwb = get_visual_workflow_builder()
            # Set mock manager if real one fails
            try:
                from ..ai.hybrid_model_architecture import get_hybrid_model_manager
                vwb.builder.set_hybrid_manager(get_hybrid_model_manager())
            except (ImportError, ValueError):
                print("Using mock manager for interactive mode")
                vwb.builder.set_hybrid_manager(MockHybridManager())
                
            await vwb.run_interactive_mode()
        
        try:
            asyncio.run(run_interactive())
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        async def test_visual_workflow_builder():
            """Test the visual workflow builder"""
            print("Testing Visual Workflow Builder...")
            
            # Create workflow builder
            vwb = get_visual_workflow_builder()
            
            # Set mock manager
            vwb.builder.set_hybrid_manager(MockHybridManager())
            
            print("\n1. Testing basic workflow creation:")
            # Add nodes to the workflow
            input_node = vwb.builder.add_node(
                NodeType.INPUT,
                "User Input",
                "Get input from user",
                {"input_type": "text", "prompt": "Enter your query:"},
                x=100, y=100
            )
            
            model_node = vwb.builder.add_node(
                NodeType.MODEL_CALL,
                "AI Model",
                "Process with AI model",
                {
                    "model_config": {"task_type": "general", "sensitivity": 2},
                    "prompt_template": "Analyze this query and provide a response: {input}"
                },
                x=300, y=100
            )
            
            output_node = vwb.builder.add_node(
                NodeType.OUTPUT,
                "Display Output",
                "Show the result",
                {},
                x=500, y=100
            )
            
            # Connect the nodes
            vwb.builder.connect_nodes(input_node.id, model_node.id)
            vwb.builder.connect_nodes(model_node.id, output_node.id)
            
            print(f"Created workflow with {len(vwb.builder.nodes)} nodes and {len(vwb.builder.connections)} connections")
            
            print("\n2. Testing workflow visualization:")
            tree = vwb.builder.visualize_workflow()
            print("Workflow visualization created successfully")
            
            print("\n3. Testing template functionality:")
            templates = vwb.builder.get_templates_by_category("development")
            print(f"Found {len(templates)} development templates")
            
            # Test instantiating a template
            if vwb.builder.templates:
                first_template = vwb.builder.templates[0]
                success = vwb.builder.instantiate_template(first_template.id)
                print(f"Template instantiation: {'Success' if success else 'Failed'}")
            
            print("\n4. Testing workflow execution:")
            try:
                # This should now succeed with the mock manager
                result = await vwb.builder.execute_current_workflow({"input": "test query"})
                print(f"Execution result: {result}")
            except Exception as e:
                print(f"Execution error: {type(e).__name__}: {e}")
            
            print("\n✅ Visual Workflow Builder tests completed!")
        
        # Run the test
        asyncio.run(test_visual_workflow_builder())