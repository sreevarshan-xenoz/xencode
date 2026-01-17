"""
Integration layer connecting communication protocol with agent system
"""
from typing import Dict, Optional, List, Any
from ..coordinator import AgentCoordinator, Agent
from ..specialized.coordinator import SpecializedAgentCoordinator
from .protocol import MessageBroker, InMemoryProtocol
from .message import Message, MessageType, MessageTemplates
from .channels import ChannelManager
import asyncio


class AgentCommunicationLayer:
    """
    Integration layer that adds communication capabilities to the existing agent system.
    """
    
    def __init__(self):
        self.message_broker = MessageBroker()
        self.channel_manager = ChannelManager()
        self.protocols: Dict[str, InMemoryProtocol] = {}
        self.agent_coordinators: List = []
        
        # Start the message broker processing
        self.message_broker.start_processing()
        
    def integrate_with_coordinator(self, coordinator: AgentCoordinator):
        """
        Integrate communication layer with an agent coordinator.
        """
        for agent_type, agent in coordinator.agents.items():
            agent_id = f"basic_{agent_type.value}_{id(agent)}"
            protocol = InMemoryProtocol(agent_id, self.message_broker)
            self.protocols[agent_id] = protocol
            
            # Add communication methods to the agent
            agent.communication_protocol = protocol
            agent.send_message = self._create_send_method(protocol)
            agent.receive_message = self._create_receive_method(protocol)
            
        # Add communication layer to coordinator
        coordinator.communication_layer = self
        self.agent_coordinators.append(coordinator)
        
    def integrate_with_specialized_coordinator(self, coordinator: SpecializedAgentCoordinator):
        """
        Integrate communication layer with a specialized agent coordinator.
        """
        for agent_type, agent in coordinator.agents.items():
            agent_id = f"specialized_{agent_type.value}_{id(agent)}"
            protocol = InMemoryProtocol(agent_id, self.message_broker)
            self.protocols[agent_id] = protocol
            
            # Add communication methods to the agent
            agent.communication_protocol = protocol
            agent.send_message = self._create_send_method(protocol)
            agent.receive_message = self._create_receive_method(protocol)
            
        # Add communication layer to coordinator
        coordinator.communication_layer = self
        self.agent_coordinators.append(coordinator)
        
    def _create_send_method(self, protocol: InMemoryProtocol):
        """Create a send_message method bound to a specific protocol."""
        async def send_message(receiver_id: str, content: str, message_type: MessageType = MessageType.REQUEST):
            message = Message(
                message_type=message_type,
                sender_id=protocol.agent_id,
                receiver_id=receiver_id,
                content=content
            )
            return await protocol.send_message(message)
        return send_message
        
    def _create_receive_method(self, protocol: InMemoryProtocol):
        """Create a receive_message method bound to a specific protocol."""
        async def receive_message(timeout: float = 30.0):
            return await protocol.receive_message(timeout)
        return receive_message
        
    async def send_task_between_agents(self, sender_id: str, receiver_id: str, 
                                     task: str, timeout: float = 30.0) -> Optional[Message]:
        """
        Send a task from one agent to another and wait for response.
        """
        return await self.message_broker.send_task_request(sender_id, receiver_id, task, timeout)
        
    def create_secure_channel(self, agent1_id: str, agent2_id: str) -> bool:
        """
        Create a secure communication channel between two agents.
        """
        try:
            self.channel_manager.create_channel(agent1_id, agent2_id)
            return True
        except Exception:
            return False
            
    def get_communication_stats(self) -> Dict[str, Any]:
        """
        Get communication statistics.
        """
        stats = self.message_broker.get_statistics()
        stats['active_channels'] = len(self.channel_manager.channels)
        stats['registered_protocols'] = len(self.protocols)
        return stats
        
    def shutdown(self):
        """
        Shutdown the communication layer.
        """
        self.message_broker.stop_processing()


# Example usage function
async def example_usage():
    """
    Example showing how to use the communication layer with agents.
    """
    from .coordinator import AgentCoordinator, AgentType
    
    # Create coordinator and communication layer
    coordinator = AgentCoordinator()
    comm_layer = AgentCommunicationLayer()
    
    # Integrate them
    comm_layer.integrate_with_coordinator(coordinator)
    
    # Get agent IDs
    agent_ids = list(comm_layer.protocols.keys())
    if len(agent_ids) >= 2:
        agent1_id, agent2_id = agent_ids[0], agent_ids[1]
        
        # Send a message between agents
        result = await comm_layer.send_task_between_agents(
            agent1_id, agent2_id, "Hello from agent 1!", 10.0
        )
        print(f"Message exchange result: {result}")
        
        # Create secure channel
        success = comm_layer.create_secure_channel(agent1_id, agent2_id)
        print(f"Secure channel created: {success}")
        
        # Print stats
        stats = comm_layer.get_communication_stats()
        print(f"Communication stats: {stats}")
    
    # Cleanup
    comm_layer.shutdown()


if __name__ == "__main__":
    asyncio.run(example_usage())