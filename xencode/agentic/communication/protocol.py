"""
Communication protocol for inter-agent communication in Xencode
"""
from typing import Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod
import asyncio
import json
from datetime import datetime
import threading
from queue import Queue, Empty
import time

from .message import Message, MessageType, MessageStatus, MessageTemplates


class CommunicationProtocol(ABC):
    """Abstract base class for communication protocols between agents."""
    
    @abstractmethod
    async def send_message(self, message: Message) -> bool:
        """Send a message to another agent."""
        pass
    
    @abstractmethod
    async def receive_message(self, timeout: float = 30.0) -> Optional[Message]:
        """Receive a message from another agent."""
        pass
    
    @abstractmethod
    def register_listener(self, callback: Callable[[Message], None]):
        """Register a callback to handle incoming messages."""
        pass


class MessageBroker:
    """Central message broker for coordinating communication between agents."""
    
    def __init__(self):
        self.agents: Dict[str, CommunicationProtocol] = {}
        self.message_queue: Queue = Queue()
        self.listeners: List[Callable[[Message], None]] = []
        self.running = False
        self.broker_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.failed_deliveries = 0
        
    def register_agent(self, agent_id: str, protocol: CommunicationProtocol):
        """Register an agent with the message broker."""
        self.agents[agent_id] = protocol
        
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the message broker."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            
    def send_message(self, message: Message) -> bool:
        """Route a message to the appropriate agent."""
        if message.receiver_id in self.agents:
            try:
                # Add to internal queue for processing
                self.message_queue.put(message)
                self.messages_sent += 1
                return True
            except Exception as e:
                print(f"Error queuing message: {e}")
                self.failed_deliveries += 1
                return False
        else:
            print(f"Receiver {message.receiver_id} not found")
            self.failed_deliveries += 1
            return False
            
    def broadcast_message(self, message: Message) -> int:
        """Broadcast a message to all registered agents."""
        delivered_count = 0
        for agent_id, protocol in self.agents.items():
            if agent_id != message.sender_id:  # Don't send to self
                broadcast_msg = message.clone()
                broadcast_msg.receiver_id = agent_id
                if self.send_message(broadcast_msg):
                    delivered_count += 1
        return delivered_count
    
    def register_listener(self, callback: Callable[[Message], None]):
        """Register a listener for all messages."""
        self.listeners.append(callback)
        
    def start_processing(self):
        """Start the message processing loop."""
        if not self.running:
            self.running = True
            self.broker_thread = threading.Thread(target=self._process_messages, daemon=True)
            self.broker_thread.start()
            
    def stop_processing(self):
        """Stop the message processing loop."""
        self.running = False
        if self.broker_thread:
            self.broker_thread.join(timeout=5.0)  # Wait up to 5 seconds
            
    def _process_messages(self):
        """Internal method to process messages from the queue."""
        while self.running:
            try:
                # Get message from queue with timeout
                try:
                    message = self.message_queue.get(timeout=1.0)
                    self.messages_received += 1
                    
                    # Update message status
                    message.status = MessageStatus.DELIVERED
                    
                    # Notify listeners
                    for listener in self.listeners:
                        try:
                            listener(message)
                        except Exception as e:
                            print(f"Error in message listener: {e}")
                    
                    # Deliver to recipient
                    if message.receiver_id in self.agents:
                        try:
                            # For simplicity, we'll just call the receive method directly
                            # In a real implementation, this would use the protocol
                            pass
                        except Exception as e:
                            print(f"Error delivering message to {message.receiver_id}: {e}")
                            message.status = MessageStatus.FAILED
                            self.failed_deliveries += 1
                            
                    self.message_queue.task_done()
                    
                except Empty:
                    # Queue is empty, continue loop
                    continue
                    
            except Exception as e:
                print(f"Error in message processing: {e}")
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'failed_deliveries': self.failed_deliveries,
            'registered_agents': len(self.agents),
            'queued_messages': self.message_queue.qsize(),
            'running': self.running
        }
    
    async def send_task_request(self, sender_id: str, receiver_id: str, 
                               task: str, timeout: float = 30.0) -> Optional[Message]:
        """Send a task request and wait for a response."""
        # Create task request message
        task_msg = MessageTemplates.create_task_request(sender_id, receiver_id, task)
        
        # Send the message
        if self.send_message(task_msg):
            # In a real implementation, we would wait for a response
            # For now, we'll simulate a response
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Create a simulated response
            response_msg = Message(
                message_type=MessageType.TASK_RESULT,
                sender_id=receiver_id,
                receiver_id=sender_id,
                content=f"Processed task: {task}",
                payload={'task_id': task_msg.payload['task_id'], 'result': 'Simulated result'},
                correlation_id=task_msg.message_id,
                reply_to=task_msg.message_id
            )
            
            # Notify listeners about the response
            for listener in self.listeners:
                try:
                    listener(response_msg)
                except Exception as e:
                    print(f"Error in response listener: {e}")
                    
            return response_msg
            
        return None


# Simple in-memory communication protocol implementation
class InMemoryProtocol(CommunicationProtocol):
    """Simple in-memory communication protocol for local agent communication."""
    
    def __init__(self, agent_id: str, broker: MessageBroker):
        self.agent_id = agent_id
        self.broker = broker
        self.message_buffer: List[Message] = []
        self.buffer_lock = threading.Lock()
        
        # Register with broker
        self.broker.register_agent(agent_id, self)
        
        # Register to receive messages destined for this agent
        def message_handler(msg: Message):
            if msg.receiver_id == agent_id:
                with self.buffer_lock:
                    self.message_buffer.append(msg)
                    
        self.broker.register_listener(message_handler)
        
    async def send_message(self, message: Message) -> bool:
        """Send a message through the broker."""
        message.sender_id = self.agent_id
        return self.broker.send_message(message)
        
    async def receive_message(self, timeout: float = 30.0) -> Optional[Message]:
        """Receive a message from the buffer."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.buffer_lock:
                if self.message_buffer:
                    return self.message_buffer.pop(0)
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
        return None
        
    def register_listener(self, callback: Callable[[Message], None]):
        """Register a callback to handle incoming messages."""
        def wrapper(msg: Message):
            if msg.receiver_id == self.agent_id:
                callback(msg)
        self.broker.register_listener(wrapper)
        
    def get_buffer_size(self) -> int:
        """Get the current size of the message buffer."""
        with self.buffer_lock:
            return len(self.message_buffer)