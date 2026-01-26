"""
Plan Graph Visualization and Storage System

This module handles the storage, retrieval, and visualization of plan graphs
for ByteBot operations.
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from pathlib import Path

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class PlanGraphStorage:
    """
    Handles storage and retrieval of plan graphs
    """
    
    def __init__(self, storage_dir: str = None):
        self.storage_dir = Path(storage_dir or "./bytebot_plans")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def save_plan(self, plan: Dict[str, Any], plan_id: str = None) -> str:
        """
        Save a plan to storage
        
        Args:
            plan: The plan to save
            plan_id: Optional ID for the plan (generates new ID if not provided)
            
        Returns:
            The ID of the saved plan
        """
        if plan_id is None:
            plan_id = plan.get("id", str(uuid.uuid4()))
        
        # Ensure plan has an ID
        plan["id"] = plan_id
        
        # Add timestamp if not present
        if "timestamp" not in plan:
            plan["timestamp"] = datetime.now().isoformat()
        
        # Create file path
        file_path = self.storage_dir / f"{plan_id}.json"
        
        # Write plan to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        
        return plan_id
    
    def load_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a plan from storage
        
        Args:
            plan_id: The ID of the plan to load
            
        Returns:
            The loaded plan or None if not found
        """
        file_path = self.storage_dir / f"{plan_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def list_plans(self) -> List[Dict[str, Any]]:
        """
        List all stored plans with basic information
        
        Returns:
            List of dictionaries with plan IDs and metadata
        """
        plans = []
        
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    plan = json.load(f)
                    
                    # Extract basic info
                    plan_info = {
                        "id": plan.get("id", file_path.stem),
                        "intent": plan.get("intent", "Unknown intent"),
                        "timestamp": plan.get("timestamp", "Unknown"),
                        "step_count": len(plan.get("steps", [])),
                        "file_path": str(file_path)
                    }
                    plans.append(plan_info)
            except Exception:
                # Skip corrupted files
                continue
        
        # Sort by timestamp (most recent first)
        plans.sort(key=lambda x: x["timestamp"], reverse=True)
        return plans
    
    def delete_plan(self, plan_id: str) -> bool:
        """
        Delete a plan from storage
        
        Args:
            plan_id: The ID of the plan to delete
            
        Returns:
            True if deleted, False if not found
        """
        file_path = self.storage_dir / f"{plan_id}.json"
        
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def get_plan_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored plans
        
        Returns:
            Dictionary with plan statistics
        """
        plans = self.list_plans()
        
        if not plans:
            return {
                "total_plans": 0,
                "total_steps": 0,
                "avg_steps_per_plan": 0,
                "date_range": None
            }
        
        total_steps = sum(plan.get("step_count", 0) for plan in plans)
        avg_steps = total_steps / len(plans) if plans else 0
        
        # Get date range
        timestamps = [plan["timestamp"] for plan in plans if plan["timestamp"] != "Unknown"]
        if timestamps:
            timestamps.sort()
            date_range = {"start": timestamps[0], "end": timestamps[-1]}
        else:
            date_range = None
        
        return {
            "total_plans": len(plans),
            "total_steps": total_steps,
            "avg_steps_per_plan": avg_steps,
            "date_range": date_range
        }


class PlanGraphVisualizer:
    """
    Handles visualization of plan graphs
    """
    
    def __init__(self):
        self.networkx_available = NETWORKX_AVAILABLE
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
    
    def create_graph_from_plan(self, plan: Dict[str, Any]):
        """
        Create a NetworkX graph from a plan
        
        Args:
            plan: The plan to visualize
            
        Returns:
            NetworkX DiGraph representing the plan
        """
        if not self.networkx_available:
            raise ImportError("NetworkX is required for graph visualization")
        
        G = nx.DiGraph()  # Directed graph since plans have dependencies
        
        # Add nodes (steps)
        for step in plan.get("steps", []):
            G.add_node(step["id"], 
                      label=step.get("description", "")[:30],  # Truncate long descriptions
                      command=step.get("command", "")[:50],    # Truncate long commands
                      type=step.get("type", "command"),
                      risk=step.get("estimated_risk", 0.0))
        
        # Add edges (dependencies)
        for from_id, to_id in plan.get("dependencies", []):
            if from_id in G.nodes and to_id in G.nodes:
                G.add_edge(from_id, to_id)
        
        return G
    
    def visualize_plan_matplotlib(self, plan: Dict[str, Any], save_path: str = None):
        """
        Visualize a plan using matplotlib
        
        Args:
            plan: The plan to visualize
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure object
        """
        if not self.networkx_available or not self.matplotlib_available:
            raise ImportError("Both NetworkX and matplotlib are required for visualization")
        
        G = self.create_graph_from_plan(plan)
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            risk = G.nodes[node].get('risk', 0.0)
            # Color based on risk (red for high risk, green for low risk)
            if risk > 0.7:
                node_colors.append('red')
            elif risk > 0.3:
                node_colors.append('orange')
            else:
                node_colors.append('lightgreen')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray', arrows=True)
        
        # Draw labels
        labels = {node: f"{G.nodes[node]['label'][:20]}..." if len(G.nodes[node]['label']) > 20 else G.nodes[node]['label'] 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Add title
        plt.title(f"Plan: {plan.get('intent', 'Unknown Intent')[:50]}", size=14)
        plt.axis('off')
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return plt.gcf()
    
    def visualize_plan_text(self, plan: Dict[str, Any]) -> str:
        """
        Create a text-based visualization of the plan
        
        Args:
            plan: The plan to visualize
            
        Returns:
            String representation of the plan
        """
        lines = []
        lines.append(f"Plan ID: {plan.get('id', 'Unknown')}")
        lines.append(f"Intent: {plan.get('intent', 'Unknown')}")
        lines.append(f"Timestamp: {plan.get('timestamp', 'Unknown')}")
        lines.append("")
        
        # Create dependency map
        dependency_map = {}
        for from_id, to_id in plan.get("dependencies", []):
            if to_id not in dependency_map:
                dependency_map[to_id] = []
            dependency_map[to_id].append(from_id)
        
        # List steps with dependencies
        for i, step in enumerate(plan.get("steps", []), 1):
            risk_str = f" (Risk: {step.get('estimated_risk', 0.0):.2f})" if step.get('estimated_risk') is not None else ""
            lines.append(f"{i}. [{step.get('type', 'command')}] {step.get('description', 'Unknown')}{risk_str}")
            lines.append(f"   Command: {step.get('command', 'Unknown')}")
            
            # Show dependencies
            if step["id"] in dependency_map:
                deps = dependency_map[step["id"]]
                lines.append(f"   Depends on: {', '.join(deps)}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_plan_report(self, plan: Dict[str, Any]) -> str:
        """
        Generate a detailed report for a plan
        
        Args:
            plan: The plan to report on
            
        Returns:
            Detailed report as string
        """
        report = []
        report.append("=" * 60)
        report.append("BYTEBOT PLAN REPORT")
        report.append("=" * 60)
        report.append(f"Plan ID: {plan.get('id', 'Unknown')}")
        report.append(f"Intent: {plan.get('intent', 'Unknown')}")
        report.append(f"Timestamp: {plan.get('timestamp', 'Unknown')}")
        report.append(f"Generated by: {plan.get('metadata', {}).get('generated_by', 'Unknown')}")
        report.append("")
        
        # Step summary
        steps = plan.get("steps", [])
        report.append(f"STEPS ({len(steps)} total):")
        report.append("-" * 30)
        
        for i, step in enumerate(steps, 1):
            risk_level = self._get_risk_level(step.get('estimated_risk', 0.0))
            report.append(f"{i}. {step.get('description', 'Unknown')}")
            report.append(f"   Type: {step.get('type', 'command')}")
            report.append(f"   Command: {step.get('command', 'Unknown')}")
            report.append(f"   Risk: {step.get('estimated_risk', 0.0):.2f} ({risk_level})")
            report.append(f"   Est. Duration: {step.get('estimated_duration', 0.0):.1f}s")
            report.append("")
        
        # Dependencies
        dependencies = plan.get("dependencies", [])
        if dependencies:
            report.append(f"DEPENDENCIES ({len(dependencies)} total):")
            report.append("-" * 30)
            for from_id, to_id in dependencies:
                report.append(f"   {from_id} -> {to_id}")
            report.append("")
        
        # Metadata
        metadata = plan.get("metadata", {})
        if metadata:
            report.append("METADATA:")
            report.append("-" * 30)
            for key, value in metadata.items():
                report.append(f"   {key}: {value}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to level string"""
        if risk_score >= 0.7:
            return "HIGH"
        elif risk_score >= 0.3:
            return "MEDIUM"
        elif risk_score > 0:
            return "LOW"
        else:
            return "MINIMAL"


class PlanGraphManager:
    """
    Main manager for plan graph storage and visualization
    """
    
    def __init__(self, storage_dir: str = None):
        self.storage = PlanGraphStorage(storage_dir)
        self.visualizer = PlanGraphVisualizer()
    
    def save_and_visualize_plan(self, plan: Dict[str, Any], 
                               include_visualization: bool = True,
                               include_report: bool = True) -> Dict[str, str]:
        """
        Save a plan and optionally create visualization and report
        
        Args:
            plan: The plan to save and visualize
            include_visualization: Whether to create visualization
            include_report: Whether to create detailed report
            
        Returns:
            Dictionary with paths to saved files
        """
        results = {}
        
        # Save the plan
        plan_id = self.storage.save_plan(plan)
        results["plan_file"] = str(self.storage.storage_dir / f"{plan_id}.json")
        
        # Create visualization if requested and available
        if include_visualization and self.visualizer.matplotlib_available:
            viz_path = self.storage.storage_dir / f"{plan_id}_viz.png"
            try:
                self.visualizer.visualize_plan_matplotlib(plan, str(viz_path))
                results["visualization"] = str(viz_path)
            except Exception as e:
                print(f"Could not create visualization: {e}")
        
        # Create report if requested
        if include_report:
            report_path = self.storage.storage_dir / f"{plan_id}_report.txt"
            report = self.visualizer.generate_plan_report(plan)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            results["report"] = str(report_path)
        
        return results
    
    def get_plan_summary(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a stored plan
        
        Args:
            plan_id: ID of the plan to summarize
            
        Returns:
            Summary dictionary or None if plan not found
        """
        plan = self.storage.load_plan(plan_id)
        if not plan:
            return None
        
        return {
            "id": plan.get("id"),
            "intent": plan.get("intent"),
            "timestamp": plan.get("timestamp"),
            "step_count": len(plan.get("steps", [])),
            "dependency_count": len(plan.get("dependencies", [])),
            "total_estimated_risk": sum(s.get("estimated_risk", 0) for s in plan.get("steps", [])),
            "max_risk": max((s.get("estimated_risk", 0) for s in plan.get("steps", [])), default=0),
            "plan_type": plan.get("metadata", {}).get("plan_type", "unknown")
        }
    
    def search_plans_by_intent(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for plans containing a search term in the intent
        
        Args:
            search_term: Term to search for in plan intents
            
        Returns:
            List of matching plan summaries
        """
        all_plans = self.storage.list_plans()
        matching = []
        
        search_lower = search_term.lower()
        for plan_info in all_plans:
            if search_lower in plan_info["intent"].lower():
                matching.append(plan_info)
        
        return matching


# Example usage
if __name__ == "__main__":
    # Example of how to use the plan graph storage and visualization
    print("Plan Graph Storage and Visualization System")
    print("=" * 50)
    
    # Create manager
    manager = PlanGraphManager("./test_plans")
    
    # Example plan structure (this would normally come from the Planner component)
    example_plan = {
        "id": "test-plan-123",
        "intent": "Create a new directory and initialize git repository",
        "steps": [
            {
                "id": "step-1",
                "type": "command",
                "description": "Create new directory",
                "command": "mkdir my_project",
                "dependencies": [],
                "estimated_risk": 0.1,
                "estimated_duration": 1.0,
                "metadata": {"purpose": "directory_creation"}
            },
            {
                "id": "step-2", 
                "type": "command",
                "description": "Change to new directory",
                "command": "cd my_project",
                "dependencies": ["step-1"],
                "estimated_risk": 0.05,
                "estimated_duration": 0.5,
                "metadata": {"purpose": "directory_navigation"}
            },
            {
                "id": "step-3",
                "type": "command", 
                "description": "Initialize git repository",
                "command": "git init",
                "dependencies": ["step-2"],
                "estimated_risk": 0.05,
                "estimated_duration": 2.0,
                "metadata": {"purpose": "git_initialization"}
            }
        ],
        "dependencies": [
            ["step-1", "step-2"],
            ["step-2", "step-3"]
        ],
        "context": {"current_directory": "/home/user"},
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "plan_type": "file_operation",
            "generated_by": "ByteBot Planner"
        }
    }
    
    # Save and visualize the plan
    print("Saving and visualizing example plan...")
    results = manager.save_and_visualize_plan(example_plan)
    
    print(f"Plan saved to: {results.get('plan_file', 'Failed')}")
    print(f"Report saved to: {results.get('report', 'Not created')}")
    if 'visualization' in results:
        print(f"Visualization saved to: {results['visualization']}")
    else:
        print("Visualization not created (matplotlib/networkx not available)")
    
    # List stored plans
    print(f"\nStored plans: {len(manager.storage.list_plans())}")
    
    # Show plan statistics
    stats = manager.storage.get_plan_statistics()
    print(f"Plan statistics: {stats}")